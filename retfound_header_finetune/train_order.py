"""
Train an ISNT-ORDER predictor on a frozen RETFound backbone.

Pipeline:  image -> RETFound(frozen) -> [B,197,1024]
                  -> pooling (--pool) -> [B,1024]
                  -> ORDER head       -> [B,4] raw scores  (NO sigmoid)
           predicted ISNT order = argsort(-scores)  over (I,S,N,T)

The model NEVER outputs rim magnitudes — only relative scores. The 4
ground-truth rim values (from isnt_manifest.csv, via dataset_isnt) are used
ONLY to build the supervision: the pairwise order label and the gap weight.

Loss — PURE gap-weighted pairwise ranking (RankNet logistic):
    for every quadrant pair (a,b):
        gap  = |rim_a - rim_b|
        if gap < --tie-eps:  SKIP  (near-tie = noise-level, don't-care)
        sign = +1 if rim_a > rim_b else -1
        L  += gap * softplus( -sign * (s_a - s_b) )
    loss = sum(L) / sum(kept gaps)          # scale-stable
Big-gap (clinically meaningful) pairs dominate; near-ties are gated out.

Optional: --aux-value-weight LAMBDA (default 0 = PURE ranking). If >0, an
auxiliary value head + SmoothL1 is added during TRAINING ONLY (discarded at
inference) — a one-flag probe of "does value supervision help the ranking?".

Model selection: best checkpoint by min VAL mean-Kendall-distance
(0 = perfect order, 6 = fully reversed). NOT exact-order (noise-dominated).

Reuses the SAME 5 poolings and the SAME data split as the CDR/ISNT ablations.

Usage:
    python train_order.py --pool tier2a
    python train_order.py --pool tier3 --tie-eps 0.02
    python train_order.py --pool tier2a --aux-value-weight 0.3   # probe
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from retfound_backbone import load_retfound
from dataset_isnt import build_dataloaders, RIM_NAMES   # reuse ISNT data

from baseline import BaselinePool
from tier1 import Tier1Pool
from tier2a import Tier2aPool
from tier2b import Tier2bPool
from tier3 import Tier3Pool

POOLS = {
    "baseline": lambda: BaselinePool(1024, mode="mean"),
    "tier1":    lambda: Tier1Pool(1024),
    "tier2a":   lambda: Tier2aPool(1024, hidden=256, gated=False),
    "tier2b":   lambda: Tier2bPool(1024),
    "tier3":    lambda: Tier3Pool(1024, num_heads=8),
}

N_Q = 4                                   # I, S, N, T
PAIRS = list(itertools.combinations(range(N_Q), 2))   # 6 pairs


class OrderHead(nn.Module):
    """LayerNorm -> (Linear | MLP) -> 4 RAW scores. No sigmoid: ranking
    needs unbounded logits (sigmoid would compress pairwise gradients and
    re-introduce the saturation we already fixed for the regression heads)."""

    def __init__(self, in_dim=1024, kind="linear", hidden=512, dropout=0.2):
        super().__init__()
        if kind == "linear":
            self.net = nn.Sequential(nn.LayerNorm(in_dim),
                                     nn.Linear(in_dim, N_Q))
        elif kind == "mlp":
            self.net = nn.Sequential(nn.LayerNorm(in_dim),
                                     nn.Linear(in_dim, hidden), nn.GELU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden, N_Q))
        else:
            raise ValueError(kind)

    def forward(self, x):
        return self.net(x)                # [B,4] raw scores


class AuxValueHead(nn.Module):
    """Training-only auxiliary: regress the 4 rim values (sigmoid->[0,1]).
    Created ONLY when --aux-value-weight > 0; never used at inference."""

    def __init__(self, in_dim=1024):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, N_Q))

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class OrderModel(nn.Module):
    def __init__(self, backbone, pool, head, aux=None, lora=False):
        super().__init__()
        self.backbone = backbone          # frozen (or LoRA-adapted)
        self.pool = pool
        self.head = head
        self.aux = aux                    # optional value head
        self.lora = lora

    def forward(self, x):
        if self.lora:
            tokens = self.backbone.forward_tokens(x)      # grad -> LoRA
        else:
            with torch.no_grad():
                tokens = self.backbone.forward_tokens(x)  # [B,197,1024]
        pooled, _ = self.pool(tokens)                     # [B,1024]
        scores = self.head(pooled)                        # [B,4]
        aux = self.aux(pooled) if self.aux is not None else None
        return scores, aux


def gap_pairwise_loss(scores, rim, tie_eps):
    """scores,rim: [B,4] (order I,S,N,T). Returns scalar.
    Pure gap-weighted RankNet over the 6 pairs, near-ties (< tie_eps) gated."""
    B = scores.size(0)
    total = scores.new_zeros(())
    wsum = scores.new_zeros(())
    for a, b in PAIRS:
        gap = (rim[:, a] - rim[:, b]).abs()               # [B]
        keep = gap >= tie_eps
        if keep.sum() == 0:
            continue
        sign = torch.sign(rim[:, a] - rim[:, b])          # +1 if a>b
        diff = scores[:, a] - scores[:, b]                 # want sign*diff >0
        # RankNet logistic: softplus(-sign*diff)
        term = F.softplus(-sign * diff) * gap             # gap-weighted
        total = total + (term * keep).sum()
        wsum = wsum + (gap * keep).sum()
    return total / wsum.clamp_min(1e-8)


@torch.no_grad()
def evaluate(model, loader, device, tie_eps):
    model.eval()
    S, R, D = [], [], []
    for x, y, _ids, ds in loader:
        x = x.to(device)
        s, _ = model(x)
        S.append(s.cpu()); R.append(y); D.extend(ds)
    S = torch.cat(S).numpy()              # [N,4] scores
    R = torch.cat(R).numpy()              # [N,4] gt rim values
    return _order_metrics(S, R, D, tie_eps)


def _order_metrics(S, R, dsets, tie_eps):
    def block(s, r):
        n = len(s)
        if n == 0:
            return {}
        exact = 0
        kendall = 0.0
        wcorrect = 0.0
        wtot = 0.0
        for i in range(n):
            op = list(np.argsort(-s[i]))           # predicted order
            og = list(np.argsort(-r[i]))           # gt order
            if op == og:
                exact += 1
            posg = {q: k for k, q in enumerate(og)}
            kendall += sum(1 for a, b in itertools.combinations(op, 2)
                           if posg[a] > posg[b])
            for a, b in PAIRS:
                gap = abs(r[i, a] - r[i, b])
                if gap < tie_eps:
                    continue
                gt_a_hi = r[i, a] > r[i, b]
                pr_a_hi = s[i, a] > s[i, b]
                wtot += gap
                if gt_a_hi == pr_a_hi:
                    wcorrect += gap
        return {
            "n": int(n),
            "kendall": float(kendall / n),           # 0..6, lower better
            "exact_order": float(exact / n),
            "gap_pair_acc": float(wcorrect / wtot) if wtot > 0 else float("nan"),
        }

    res = {"overall": block(S, R)}
    dsets = np.array(dsets)
    for d in ("LAG", "Papila"):
        m = dsets == d
        if m.any():
            res[d] = block(S[m], R[m])
    return res, S, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, choices=list(POOLS))
    ap.add_argument("--head", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--tie-eps", type=float, default=0.02,
                    help="pairs with |rim_a-rim_b| < this are dropped "
                         "(near-tie = label-noise level, ~0.02)")
    ap.add_argument("--aux-value-weight", type=float, default=0.0,
                    help="0 = PURE ranking. >0 adds a training-only "
                         "value-regression aux loss (probe).")
    ap.add_argument("--aux-beta", type=float, default=0.04,
                    help="SmoothL1 beta for the aux value loss")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=10,
                    help="early-stop patience on val Kendall")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--lora", action="store_true",
                    help="LoRA-adapt the ViT (base frozen), trained jointly "
                         "with pool+head. Default off = frozen backbone.")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    ap.add_argument("--warmstart", default=None,
                    help="frozen-backbone best.pt to init pool+head from.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ts = time.strftime("%Y%m%d_%H%M%S")
    pure = args.aux_value_weight <= 0
    tag = (f"order_{args.pool}_{args.head}"
           f"{'' if pure else f'_aux{args.aux_value_weight}'}_{ts}")
    out_dir = Path(args.out or f"outputs/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out={out_dir}")

    tr, va, te, info = build_dataloaders(
        batch_size=args.batch_size, val_frac=args.val_frac,
        seed=args.seed, num_workers=args.num_workers)
    print(f"data: {info}")

    backbone = load_retfound(freeze=not args.lora, lora=args.lora,
                             lora_rank=args.lora_rank)
    pool = POOLS[args.pool]()
    head = OrderHead(1024, kind=args.head)
    aux = None if pure else AuxValueHead(1024)
    model = OrderModel(backbone, pool, head, aux, lora=args.lora).to(device)

    if args.warmstart:
        wck = torch.load(args.warmstart, map_location=device,
                         weights_only=False)
        model.pool.load_state_dict(wck["pool_state"])
        model.head.load_state_dict(wck["head_state"])
        print(f"warm-started pool+head from {args.warmstart}")

    head_pool = list(model.pool.parameters()) + list(model.head.parameters())
    if model.aux is not None:
        head_pool += list(model.aux.parameters())
    lora_params = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = [{"params": [p for p in head_pool if p.requires_grad],
               "init_lr": args.lr}]
    if args.lora and lora_params:
        groups.append({"params": lora_params, "init_lr": args.lora_lr})
    opt = torch.optim.AdamW(groups, lr=args.lr,
                            weight_decay=args.weight_decay)
    n_hp = sum(p.numel() for p in head_pool if p.requires_grad)
    n_lr = sum(p.numel() for p in lora_params)
    print(f"pool={args.pool} head={args.head} lora={args.lora}"
          f"{f'(r{args.lora_rank})' if args.lora else ''} "
          f"{'PURE-rank' if pure else f'aux={args.aux_value_weight}'} "
          f"tie_eps={args.tie_eps}  pool+head={n_hp/1e6:.3f}M"
          f"{f' + LoRA={n_lr/1e6:.3f}M' if args.lora else ''}")

    def lr_at(ep):
        if ep < args.warmup_epochs:
            return (ep + 1) / max(1, args.warmup_epochs)
        prog = (ep - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * prog))

    aux_crit = nn.SmoothL1Loss(beta=args.aux_beta)

    best_val = float("inf")               # min Kendall
    best_ep = -1
    bad = 0
    hist = []

    def lora_sd():
        if not args.lora:
            return {}
        return {n: p.detach().cpu() for n, p in
                model.backbone.named_parameters() if p.requires_grad}

    for ep in range(args.epochs):
        sched = lr_at(ep)
        for g in opt.param_groups:
            g["lr"] = g["init_lr"] * sched

        model.train()
        if not args.lora:
            model.backbone.eval()
        tr_loss = 0.0
        for x, y, _i, _d in tr:
            x, y = x.to(device), y.to(device)         # y = rim values [B,4]
            scores, auxp = model(x)
            loss = gap_pairwise_loss(scores, y, args.tie_eps)
            if not pure:
                loss = loss + args.aux_value_weight * aux_crit(auxp, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(x)
        tr_loss /= max(1, info["n_train"])

        vres, _, _ = evaluate(model, va, device, args.tie_eps)
        vk = vres["overall"]["kendall"]
        hist.append({"epoch": ep, "lr": opt.param_groups[0]["lr"],
                     "train_loss": tr_loss, "val_kendall": vk,
                     "val_exact": vres["overall"]["exact_order"],
                     "val_gap_pair_acc": vres["overall"]["gap_pair_acc"]})
        print(f"ep {ep:>3} lr={opt.param_groups[0]['lr']:.2e} "
              f"train_loss={tr_loss:.5f} val_kendall={vk:.4f} "
              f"(exact={vres['overall']['exact_order']:.2f} "
              f"gpa={vres['overall']['gap_pair_acc']:.3f})"
              f"{'  *best' if vk < best_val else ''}")

        if vk < best_val:
            best_val, best_ep, bad = vk, ep, 0
            torch.save({"epoch": ep, "pool": args.pool, "head": args.head,
                        "pool_state": model.pool.state_dict(),
                        "head_state": model.head.state_dict(),
                        "lora_state": lora_sd(),
                        "val_kendall": vk},
                       out_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop @ ep {ep} (no val improve {args.patience})")
                break

    ck = torch.load(out_dir / "best.pt", map_location=device,
                    weights_only=False)
    model.pool.load_state_dict(ck["pool_state"])
    model.head.load_state_dict(ck["head_state"])
    if ck.get("lora_state"):
        model.backbone.load_state_dict(ck["lora_state"], strict=False)
    test_res, S, R = evaluate(model, te, device, args.tie_eps)

    summary = {
        "task": "isnt_order", "pool": args.pool, "head": args.head,
        "pure_ranking": pure, "aux_value_weight": args.aux_value_weight,
        "tie_eps": args.tie_eps, "target_order": list(RIM_NAMES),
        "best_epoch": best_ep, "best_val_kendall": best_val,
        "data_info": info, "args": vars(args),
        "test_metrics": test_res, "history": hist,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    np.savez(out_dir / "test_preds.npz", scores=S, gt_rim=R)

    print("\n==== TEST (best ckpt, epoch %d) ====" % best_ep)
    for k, v in test_res.items():
        if v:
            print(f"  [{k}] n={v['n']}  Kendall={v['kendall']:.4f}  "
                  f"exact_order={v['exact_order']:.3f}  "
                  f"gap_pair_acc={v['gap_pair_acc']:.4f}")
    print("  (Kendall 0=perfect .. 6=reversed; exact_order=all-4-right "
          "frac; gap_pair_acc=gap-weighted pairwise correct)")
    print(f"\nDone. Artifacts in {out_dir}/")


if __name__ == "__main__":
    main()

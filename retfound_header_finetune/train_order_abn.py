"""
Plan A — JOINT head: ISNT-ORDER (ranking) + per-quadrant ABNORMALITY (3-class).

Frozen RETFound -> SHARED pooling (--pool) -> shared pooled[B,1024], which feeds
TWO heads off the SAME representation:
    OrderHead : [B,1024] -> 4 raw scores   (pairwise gap-weighted RankNet, as in
                train_order.py — predicts the I/S/N/T thickness ORDER)
    AbnHead   : [B,1024] -> [B,4,3] logits  (per-quadrant 0=normal/1=mild/2=severe,
                LLM-judged from the expert Step3 text; class-weighted CE)

total_loss = rank_loss + abn_weight * abn_loss
Both gradients flow back into the SHARED pooling (and LoRA, if enabled) — this is
the multi-task coupling the user asked for (Plan A). The two OUTPUTS never fight;
they only share the rim representation.

Model selection: by a COMBINED val score that rewards good ranking AND good
abnormality F1:  sel = (kendall / 6)  +  (1 - abn_macroF1)   [lower is better].
(kendall 0..6 lower better; abn_macroF1 0..1 higher better -> 1-F1 lower better.)
Both sub-metrics are also logged so you can see each task on its own.

Reuses the SAME 5 poolings + SAME split as every other ablation.

Usage:
    python train_order_abn.py --pool tier2a
    python train_order_abn.py --pool tier3 --abn-weight 1.0 --head mlp
    python train_order_abn.py --pool tier2a --warmstart outputs/order_tier2a_*/best.pt
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
from dataset_order_abn import (build_dataloaders, abn_class_weights,
                               RIM_NAMES, N_ABN_CLASSES)

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

N_Q = 4
PAIRS = list(itertools.combinations(range(N_Q), 2))   # 6 pairs


def _maybe_ln(in_dim, pre_normed):
    # when pre_normed (LR-inject path already LN'd the pooled part), skip the
    # head's own LayerNorm so we don't re-normalize away the LR embedding.
    return [] if pre_normed else [nn.LayerNorm(in_dim)]


class OrderHead(nn.Module):
    """[LayerNorm] -> (Linear|MLP) -> 4 raw scores (no sigmoid; ranking)."""
    def __init__(self, in_dim=1024, kind="linear", hidden=512, dropout=0.2,
                 pre_normed=False):
        super().__init__()
        ln = _maybe_ln(in_dim, pre_normed)
        if kind == "linear":
            self.net = nn.Sequential(*ln, nn.Linear(in_dim, N_Q))
        elif kind == "mlp":
            self.net = nn.Sequential(*ln, nn.Linear(in_dim, hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, N_Q))
        else:
            raise ValueError(kind)

    def forward(self, x):
        return self.net(x)                       # [B,4]


class AbnHead(nn.Module):
    """[LayerNorm] -> (Linear|MLP) -> 4*3 logits -> [B,4,3]."""
    def __init__(self, in_dim=1024, kind="linear", hidden=512, dropout=0.2,
                 pre_normed=False):
        super().__init__()
        out = N_Q * N_ABN_CLASSES
        ln = _maybe_ln(in_dim, pre_normed)
        if kind == "linear":
            self.net = nn.Sequential(*ln, nn.Linear(in_dim, out))
        elif kind == "mlp":
            self.net = nn.Sequential(*ln, nn.Linear(in_dim, hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, out))
        else:
            raise ValueError(kind)

    def forward(self, x):
        return self.net(x).view(-1, N_Q, N_ABN_CLASSES)   # [B,4,3]


class JointModel(nn.Module):
    def __init__(self, backbone, pool, order_head, abn_head, lora=False,
                 lr_inject=False, lr_dim=32):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.order_head = order_head
        self.abn_head = abn_head
        self.lora = lora
        self.lr_inject = lr_inject
        if lr_inject:
            # LayerNorm the pooled BEFORE concat so the 1024-dim feature is
            # normalized; the LR embedding (lr_dim, NOT normalized) keeps its
            # own scale so the laterality signal isn't washed out.
            self.pre_ln = nn.LayerNorm(1024)
            self.lr_emb = nn.Embedding(2, lr_dim)   # R=0 / L=1 -> lr_dim vector

    def forward(self, x, lr=None):
        if self.lora:
            tokens = self.backbone.forward_tokens(x)
        else:
            with torch.no_grad():
                tokens = self.backbone.forward_tokens(x)
        pooled, _ = self.pool(tokens)                # SHARED [B,1024]
        if self.lr_inject:
            feat = torch.cat([self.pre_ln(pooled), self.lr_emb(lr)], dim=1)  # [B,1024+lr_dim]
            return self.order_head(feat), self.abn_head(feat)
        return self.order_head(pooled), self.abn_head(pooled)


def gap_pairwise_loss(scores, rim, tie_eps):
    """Pure gap-weighted RankNet over 6 pairs (identical to train_order.py)."""
    total = scores.new_zeros(())
    wsum = scores.new_zeros(())
    for a, b in PAIRS:
        gap = (rim[:, a] - rim[:, b]).abs()
        keep = gap >= tie_eps
        if keep.sum() == 0:
            continue
        sign = torch.sign(rim[:, a] - rim[:, b])
        diff = scores[:, a] - scores[:, b]
        term = F.softplus(-sign * diff) * gap
        total = total + (term * keep).sum()
        wsum = wsum + (gap * keep).sum()
    return total / wsum.clamp_min(1e-8)


def abn_loss_fn(logits, target, class_w):
    """logits [B,4,3], target [B,4] long. Sum per-quadrant class-weighted CE,
    mean over the 4 quadrants. class_w [4,3]."""
    loss = logits.new_zeros(())
    for j in range(N_Q):
        loss = loss + F.cross_entropy(logits[:, j, :], target[:, j],
                                      weight=class_w[j])
    return loss / N_Q


@torch.no_grad()
def evaluate(model, loader, device, tie_eps):
    model.eval()
    S, R, AL, AP, D = [], [], [], [], []
    for x, rim, abn, lr, _ids, ds in loader:
        x = x.to(device); lr = lr.to(device)
        s, al = model(x, lr)
        S.append(s.cpu()); R.append(rim); AL.append(al.cpu()); AP.append(abn)
        D.extend(ds)
    S = torch.cat(S).numpy()
    R = torch.cat(R).numpy()
    abn_logits = torch.cat(AL)                 # [N,4,3]
    abn_pred = abn_logits.argmax(-1).numpy()   # [N,4]
    abn_true = torch.cat(AP).numpy()           # [N,4]
    return _metrics(S, R, abn_pred, abn_true, D, tie_eps)


def _macro_f1(true, pred, n_cls=N_ABN_CLASSES):
    f1s = []
    for c in range(n_cls):
        tp = np.sum((pred == c) & (true == c))
        fp = np.sum((pred == c) & (true != c))
        fn = np.sum((pred != c) & (true == c))
        if tp + fp == 0 or tp + fn == 0:
            if (true == c).sum() == 0:        # class absent in truth -> skip
                continue
            f1s.append(0.0); continue
        prec = tp / (tp + fp); rec = tp / (tp + fn)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else float("nan")


def _metrics(S, R, abn_pred, abn_true, dsets, tie_eps):
    def block(s, r, ap, at):
        n = len(s)
        if n == 0:
            return {}
        exact = 0; kendall = 0.0; wcorrect = 0.0; wtot = 0.0
        for i in range(n):
            op = list(np.argsort(-s[i])); og = list(np.argsort(-r[i]))
            if op == og:
                exact += 1
            posg = {q: k for k, q in enumerate(og)}
            kendall += sum(1 for a, b in itertools.combinations(op, 2)
                           if posg[a] > posg[b])
            for a, b in PAIRS:
                gap = abs(r[i, a] - r[i, b])
                if gap < tie_eps:
                    continue
                wtot += gap
                if (r[i, a] > r[i, b]) == (s[i, a] > s[i, b]):
                    wcorrect += gap
        # abnormality metrics
        per_q_f1 = [_macro_f1(at[:, j], ap[:, j]) for j in range(N_Q)]
        abn_acc = float((ap == at).mean())
        # binary "any-abnormal" (>=1) per quadrant, then n_abn count corr
        nb_pred = (ap >= 1).sum(1); nb_true = (at >= 1).sum(1)
        nb_mae = float(np.abs(nb_pred - nb_true).mean())
        return {
            "n": int(n),
            "kendall": float(kendall / n),
            "exact_order": float(exact / n),
            "gap_pair_acc": float(wcorrect / wtot) if wtot > 0 else float("nan"),
            "abn_macro_f1": float(np.nanmean(per_q_f1)),
            "abn_per_quad_f1": {RIM_NAMES[j]: per_q_f1[j] for j in range(N_Q)},
            "abn_acc": abn_acc,
            "n_abn_mae": nb_mae,
        }
    res = {"overall": block(S, R, abn_pred, abn_true)}
    dsets = np.array(dsets)
    for d in ("LAG", "Papila"):
        m = dsets == d
        if m.any():
            res[d] = block(S[m], R[m], abn_pred[m], abn_true[m])
    return res, S, R, abn_pred, abn_true


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, choices=list(POOLS))
    ap.add_argument("--head", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--abn-weight", type=float, default=1.0,
                    help="lambda on the abnormality loss (rank loss weight=1).")
    ap.add_argument("--rank-weight", type=float, default=1.0,
                    help="weight on the ranking loss. Set 0 to train the "
                         "ABNORMALITY head ONLY (no ordering) — ablation to test "
                         "whether joint ranking was hurting abn quality. When 0, "
                         "model selection switches to abn-macroF1 only.")
    ap.add_argument("--tie-eps", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-class-weight", action="store_true",
                    help="disable inverse-freq class weights on abn CE")
    ap.add_argument("--augment", action="store_true",
                    help="on-the-fly TRAIN augmentation: rotate ±5° + zoom-in "
                         "(RandomResizedCrop scale 0.85-1.0), NO flip. val/test "
                         "unchanged. rim labels are scale/rotation invariant.")
    ap.add_argument("--square", action="store_true",
                    help="center-crop to square BEFORE resize (unify aspect "
                         "ratio: Papila 2576x1934 no longer squeezed). Applied "
                         "to train/val/test consistently.")
    ap.add_argument("--lr-inject", action="store_true",
                    help="tier3_LR: inject laterality (R/L) AFTER pooling — "
                         "LayerNorm(pooled) then concat Embedding(2,lr_dim); "
                         "heads take 1024+lr_dim. Needs --lr-json.")
    ap.add_argument("--lr-json", default=None,
                    help="laterality_pred_733.json (id->{lr:0/1}) for injection.")
    ap.add_argument("--lr-dim", type=int, default=32,
                    help="laterality embedding dim (default 32).")
    ap.add_argument("--all-left", action="store_true",
                    help="unify all eyes to LEFT orientation: flip R(lr=0) images "
                         "horizontally + swap Nasal<->Temporal labels (rim & abn). "
                         "Forces square. Applied to train/val/test. Needs --lr-json.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    ap.add_argument("--warmstart", default=None,
                    help="frozen-backbone order best.pt to init pool+order_head")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = f"orderabn_{args.pool}_{args.head}_w{args.abn_weight}_{ts}"
    out_dir = Path(args.out or f"outputs/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out={out_dir}")

    tr, va, te, info = build_dataloaders(
        batch_size=args.batch_size, val_frac=args.val_frac,
        seed=args.seed, num_workers=args.num_workers, augment=args.augment,
        square=args.square, lr_json=args.lr_json, all_left=args.all_left)
    print(f"data: {info}")

    backbone = load_retfound(freeze=not args.lora, lora=args.lora,
                             lora_rank=args.lora_rank)
    pool = POOLS[args.pool]()
    head_in = 1024 + args.lr_dim if args.lr_inject else 1024
    order_head = OrderHead(head_in, kind=args.head, pre_normed=args.lr_inject)
    abn_head = AbnHead(head_in, kind=args.head, pre_normed=args.lr_inject)
    model = JointModel(backbone, pool, order_head, abn_head,
                       lora=args.lora, lr_inject=args.lr_inject,
                       lr_dim=args.lr_dim).to(device)

    if args.warmstart:
        wck = torch.load(args.warmstart, map_location=device, weights_only=False)
        model.pool.load_state_dict(wck["pool_state"])
        try:
            model.order_head.load_state_dict(wck["head_state"])
            print(f"warm-started pool+order_head from {args.warmstart}")
        except Exception as e:
            print(f"(order_head warmstart skipped: {e}); pool loaded")

    class_w = (torch.ones(N_Q, N_ABN_CLASSES) if args.no_class_weight
               else abn_class_weights(info)).to(device)
    print("abn class weights [4,3]:\n", class_w.cpu().numpy())

    params = (list(model.pool.parameters())
              + list(model.order_head.parameters())
              + list(model.abn_head.parameters()))
    if args.lr_inject:   # the laterality embedding + pre-concat LayerNorm
        params += list(model.lr_emb.parameters()) + list(model.pre_ln.parameters())
    lora_params = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = [{"params": [p for p in params if p.requires_grad],
               "init_lr": args.lr}]
    if args.lora and lora_params:
        groups.append({"params": lora_params, "init_lr": args.lora_lr})
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)

    def lr_at(ep):
        if ep < args.warmup_epochs:
            return (ep + 1) / max(1, args.warmup_epochs)
        prog = (ep - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * prog))

    def lora_sd():
        if not args.lora:
            return {}
        return {n: p.detach().cpu() for n, p in
                model.backbone.named_parameters() if p.requires_grad}

    best_sel = float("inf"); best_ep = -1; bad = 0; hist = []

    for ep in range(args.epochs):
        sched = lr_at(ep)
        for g in opt.param_groups:
            g["lr"] = g["init_lr"] * sched
        model.train()
        if not args.lora:
            model.backbone.eval()
        tr_loss = tr_rank = tr_abn = 0.0
        for x, rim, abn, lr, _i, _d in tr:
            x, rim, abn, lr = x.to(device), rim.to(device), abn.to(device), lr.to(device)
            scores, abn_logits = model(x, lr)
            lr_rank = gap_pairwise_loss(scores, rim, args.tie_eps)
            la = abn_loss_fn(abn_logits, abn, class_w)
            loss = args.rank_weight * lr_rank + args.abn_weight * la
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * len(x)
            tr_rank += lr_rank.item() * len(x)
            tr_abn += la.item() * len(x)
        nrm = max(1, info["n_train"])
        tr_loss /= nrm; tr_rank /= nrm; tr_abn /= nrm

        vres, *_ = evaluate(model, va, device, args.tie_eps)
        vk = vres["overall"]["kendall"]
        vf1 = vres["overall"]["abn_macro_f1"]
        # selection: abn-only when ranking is off, else combined (both lower=better)
        sel = (1.0 - vf1) if args.rank_weight == 0 else (vk / 6.0) + (1.0 - vf1)
        hist.append({"epoch": ep, "lr": opt.param_groups[0]["lr"],
                     "train_loss": tr_loss, "train_rank": tr_rank,
                     "train_abn": tr_abn, "val_kendall": vk,
                     "val_abn_macro_f1": vf1, "val_sel": sel,
                     "val_exact": vres["overall"]["exact_order"],
                     "val_abn_acc": vres["overall"]["abn_acc"]})
        print(f"ep {ep:>3} lr={opt.param_groups[0]['lr']:.2e} "
              f"loss={tr_loss:.4f}(rank={tr_rank:.4f} abn={tr_abn:.4f}) "
              f"val_kendall={vk:.4f} val_abnF1={vf1:.4f} "
              f"val_abnAcc={vres['overall']['abn_acc']:.3f} sel={sel:.4f}"
              f"{'  *best' if sel < best_sel else ''}")

        if sel < best_sel:
            best_sel, best_ep, bad = sel, ep, 0
            ckpt = {"epoch": ep, "pool": args.pool, "head": args.head,
                    "abn_weight": args.abn_weight,
                    "lr_inject": args.lr_inject, "lr_dim": args.lr_dim,
                    "square": args.square,
                    "pool_state": model.pool.state_dict(),
                    "order_head_state": model.order_head.state_dict(),
                    "abn_head_state": model.abn_head.state_dict(),
                    "lora_state": lora_sd(),
                    "val_sel": sel, "val_kendall": vk,
                    "val_abn_macro_f1": vf1}
            if args.lr_inject:   # save laterality embedding + pre-concat LN for full reproducibility
                ckpt["lr_emb_state"] = model.lr_emb.state_dict()
                ckpt["pre_ln_state"] = model.pre_ln.state_dict()
            torch.save(ckpt, out_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop @ ep {ep} (no val improve {args.patience})")
                break

    ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.pool.load_state_dict(ck["pool_state"])
    model.order_head.load_state_dict(ck["order_head_state"])
    model.abn_head.load_state_dict(ck["abn_head_state"])
    if ck.get("lr_emb_state"):
        model.lr_emb.load_state_dict(ck["lr_emb_state"])
        model.pre_ln.load_state_dict(ck["pre_ln_state"])
    if ck.get("lora_state"):
        model.backbone.load_state_dict(ck["lora_state"], strict=False)
    test_res, S, R, AP, AT = evaluate(model, te, device, args.tie_eps)

    summary = {
        "task": "isnt_order_plus_abn_jointA",
        "pool": args.pool, "head": args.head, "abn_weight": args.abn_weight,
        "tie_eps": args.tie_eps, "target_order": list(RIM_NAMES),
        "best_epoch": best_ep, "best_val_sel": best_sel,
        "data_info": info, "args": vars(args),
        "test_metrics": test_res, "history": hist,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    np.savez(out_dir / "test_preds.npz", scores=S, gt_rim=R,
             abn_pred=AP, abn_true=AT)

    o = test_res["overall"]
    print(f"\n==== TEST (best ckpt ep {best_ep}) ====")
    print(f"  ORDER : Kendall={o['kendall']:.4f} exact={o['exact_order']:.3f} "
          f"gap_pair_acc={o['gap_pair_acc']:.4f}")
    print(f"  ABN   : macroF1={o['abn_macro_f1']:.4f} acc={o['abn_acc']:.3f} "
          f"n_abn_MAE={o['n_abn_mae']:.3f}")
    print(f"          per-quad F1: {o['abn_per_quad_f1']}")
    print(f"\nDone. Artifacts in {out_dir}/")


if __name__ == "__main__":
    main()

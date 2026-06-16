"""
TRI-HEAD joint model: ISNT-order (rank) + per-quadrant rim ABN + CDR verdict,
all off ONE shared pooling on the frozen RETFound backbone.

    Frozen RETFound -> SHARED pooling (--pool) -> pooled[B,1024]
        ├── OrderHead : [B,1024] -> 4 raw scores   (gap-weighted RankNet)        [step3 order]
        ├── AbnHead   : [B,1024] -> [B,4,3] logits  (per-quad 0/1/2, class-wt CE) [step3 rim]
        └── CdrHead   : [B,1024] -> [B,2,3] logits  (vert/horiz CDR level, CE)    [step2 CDR]

Loss = UNCERTAINTY-WEIGHTED sum (Kendall et al. 2018): three learnable log-vars
s_rank, s_abn, s_cdr automatically balance the tasks (no hand-tuned weights):
    total = Σ_i [ exp(-s_i) * L_i + s_i ]
The three gradients all flow into the SHARED pooling (multi-task coupling).

Model selection: combined val score (lower=better)
    sel = kendall/6 + (1 - abn_macroF1) + (1 - cdr_macroF1)

Sweep the SAME 5 poolings × 2 head types as the single-task ablations:
    python train_tri_head.py --pool tier3 --head linear
    python train_tri_head.py --pool tier2a --head mlp
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
from dataset_tri import (build_dataloaders, abn_class_weights, cdr_class_weights,
                         RIM_NAMES, CDR_NAMES, N_ABN_CLASSES, N_CDR_CLASSES,
                         N_Q, N_CDR)

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

PAIRS = list(itertools.combinations(range(N_Q), 2))   # 6 pairs


# ----------------------------- heads -----------------------------
def _head(in_dim, out, kind, hidden=512, dropout=0.2):
    if kind == "linear":
        return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out))
    if kind == "mlp":
        return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden),
                             nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out))
    raise ValueError(kind)


class OrderHead(nn.Module):
    def __init__(self, in_dim=1024, kind="linear"):
        super().__init__(); self.net = _head(in_dim, N_Q, kind)
    def forward(self, x):
        return self.net(x)                                   # [B,4]


class AbnHead(nn.Module):
    def __init__(self, in_dim=1024, kind="linear"):
        super().__init__(); self.net = _head(in_dim, N_Q * N_ABN_CLASSES, kind)
    def forward(self, x):
        return self.net(x).view(-1, N_Q, N_ABN_CLASSES)      # [B,4,3]


class CdrHead(nn.Module):
    def __init__(self, in_dim=1024, kind="linear"):
        super().__init__(); self.net = _head(in_dim, N_CDR * N_CDR_CLASSES, kind)
    def forward(self, x):
        return self.net(x).view(-1, N_CDR, N_CDR_CLASSES)    # [B,2,3]


class TriModel(nn.Module):
    """Shared pooling -> 3 heads + learnable per-task log-vars (uncertainty wt)."""
    def __init__(self, backbone, pool, order_head, abn_head, cdr_head):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.order_head = order_head
        self.abn_head = abn_head
        self.cdr_head = cdr_head
        # log variances s_i for [rank, abn, cdr]; init 0 -> precision 1
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        with torch.no_grad():
            tokens = self.backbone.forward_tokens(x)
        pooled, _ = self.pool(tokens)                        # SHARED [B,1024]
        return (self.order_head(pooled), self.abn_head(pooled),
                self.cdr_head(pooled))

    def combine(self, l_rank, l_abn, l_cdr):
        losses = torch.stack([l_rank, l_abn, l_cdr])         # [3]
        # total = Σ exp(-s_i) L_i + s_i   (Kendall multi-task uncertainty)
        return (torch.exp(-self.log_vars) * losses + self.log_vars).sum()


# ----------------------------- losses -----------------------------
def gap_pairwise_loss(scores, rim, tie_eps):
    total = scores.new_zeros(()); wsum = scores.new_zeros(())
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


def _ce_multi(logits, target, class_w, n_ch):
    """logits [B,ch,C], target [B,ch] -> mean class-weighted CE over channels."""
    loss = logits.new_zeros(())
    for j in range(n_ch):
        loss = loss + F.cross_entropy(logits[:, j, :], target[:, j],
                                      weight=class_w[j])
    return loss / n_ch


# ----------------------------- metrics -----------------------------
def _macro_f1(true, pred, n_cls):
    f1s = []
    for c in range(n_cls):
        tp = np.sum((pred == c) & (true == c))
        fp = np.sum((pred == c) & (true != c))
        fn = np.sum((pred != c) & (true == c))
        if (true == c).sum() == 0:
            continue
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0); continue
        prec = tp / (tp + fp); rec = tp / (tp + fn)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else float("nan")


@torch.no_grad()
def evaluate(model, loader, device, tie_eps):
    model.eval()
    S, R, AL, AP, CL, CP, D = [], [], [], [], [], [], []
    for x, rim, abn, cdr, _ids, ds in loader:
        x = x.to(device)
        s, al, cl = model(x)
        S.append(s.cpu()); R.append(rim)
        AL.append(al.cpu()); AP.append(abn)
        CL.append(cl.cpu()); CP.append(cdr)
        D.extend(ds)
    S = torch.cat(S).numpy(); R = torch.cat(R).numpy()
    abn_pred = torch.cat(AL).argmax(-1).numpy(); abn_true = torch.cat(AP).numpy()
    cdr_pred = torch.cat(CL).argmax(-1).numpy(); cdr_true = torch.cat(CP).numpy()
    return _metrics(S, R, abn_pred, abn_true, cdr_pred, cdr_true, D, tie_eps)


def _metrics(S, R, abn_pred, abn_true, cdr_pred, cdr_true, dsets, tie_eps):
    def block(s, r, ap, at, cp, ct):
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
        abn_q_f1 = [_macro_f1(at[:, j], ap[:, j], N_ABN_CLASSES) for j in range(N_Q)]
        cdr_q_f1 = [_macro_f1(ct[:, j], cp[:, j], N_CDR_CLASSES) for j in range(N_CDR)]
        return {
            "n": int(n),
            "kendall": float(kendall / n),
            "exact_order": float(exact / n),
            "gap_pair_acc": float(wcorrect / wtot) if wtot > 0 else float("nan"),
            "abn_macro_f1": float(np.nanmean(abn_q_f1)),
            "abn_per_quad_f1": {RIM_NAMES[j]: abn_q_f1[j] for j in range(N_Q)},
            "abn_acc": float((ap == at).mean()),
            "cdr_macro_f1": float(np.nanmean(cdr_q_f1)),
            "cdr_per_ch_f1": {CDR_NAMES[j]: cdr_q_f1[j] for j in range(N_CDR)},
            "cdr_acc": float((cp == ct).mean()),
        }
    res = {"overall": block(S, R, abn_pred, abn_true, cdr_pred, cdr_true)}
    dsets = np.array(dsets)
    for d in ("LAG", "Papila"):
        m = dsets == d
        if m.any():
            res[d] = block(S[m], R[m], abn_pred[m], abn_true[m],
                           cdr_pred[m], cdr_true[m])
    return res, S, R, abn_pred, abn_true, cdr_pred, cdr_true


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, choices=list(POOLS))
    ap.add_argument("--head", default="linear", choices=["linear", "mlp"])
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
    ap.add_argument("--no-class-weight", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = f"tri_{args.pool}_{args.head}_{ts}"
    out_dir = Path(args.out or f"outputs/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out={out_dir}")

    tr, va, te, info = build_dataloaders(
        batch_size=args.batch_size, val_frac=args.val_frac,
        seed=args.seed, num_workers=args.num_workers)
    print(f"data: n_train={info['n_train']} n_val={info['n_val']} n_test={info['n_test']}")

    backbone = load_retfound(freeze=True, lora=False)
    pool = POOLS[args.pool]()
    order_head = OrderHead(1024, kind=args.head)
    abn_head = AbnHead(1024, kind=args.head)
    cdr_head = CdrHead(1024, kind=args.head)
    model = TriModel(backbone, pool, order_head, abn_head, cdr_head).to(device)

    if args.no_class_weight:
        abn_w = torch.ones(N_Q, N_ABN_CLASSES).to(device)
        cdr_w = torch.ones(N_CDR, N_CDR_CLASSES).to(device)
    else:
        abn_w = abn_class_weights(info).to(device)
        cdr_w = cdr_class_weights(info).to(device)
    print("abn class weights [4,3]:\n", abn_w.cpu().numpy())
    print("cdr class weights [2,3]:\n", cdr_w.cpu().numpy())

    params = (list(model.pool.parameters())
              + list(model.order_head.parameters())
              + list(model.abn_head.parameters())
              + list(model.cdr_head.parameters())
              + [model.log_vars])
    opt = torch.optim.AdamW([{"params": [p for p in params if p.requires_grad],
                              "init_lr": args.lr}],
                            lr=args.lr, weight_decay=args.weight_decay)

    def lr_at(ep):
        if ep < args.warmup_epochs:
            return (ep + 1) / max(1, args.warmup_epochs)
        prog = (ep - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * prog))

    best_sel = float("inf"); best_ep = -1; bad = 0; hist = []

    for ep in range(args.epochs):
        sched = lr_at(ep)
        for g in opt.param_groups:
            g["lr"] = g["init_lr"] * sched
        model.train(); model.backbone.eval()
        tl = tr_rank = tr_abn = tr_cdr = 0.0
        for x, rim, abn, cdr, _i, _d in tr:
            x, rim = x.to(device), rim.to(device)
            abn, cdr = abn.to(device), cdr.to(device)
            scores, abn_logits, cdr_logits = model(x)
            l_rank = gap_pairwise_loss(scores, rim, args.tie_eps)
            l_abn = _ce_multi(abn_logits, abn, abn_w, N_Q)
            l_cdr = _ce_multi(cdr_logits, cdr, cdr_w, N_CDR)
            loss = model.combine(l_rank, l_abn, l_cdr)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(x); tr_rank += l_rank.item() * len(x)
            tr_abn += l_abn.item() * len(x); tr_cdr += l_cdr.item() * len(x)
        nrm = max(1, info["n_train"])
        tl /= nrm; tr_rank /= nrm; tr_abn /= nrm; tr_cdr /= nrm

        vres, *_ = evaluate(model, va, device, args.tie_eps)
        o = vres["overall"]
        vk, vabn, vcdr = o["kendall"], o["abn_macro_f1"], o["cdr_macro_f1"]
        sel = vk / 6.0 + (1.0 - vabn) + (1.0 - vcdr)
        lv = model.log_vars.detach().cpu().numpy()
        hist.append({"epoch": ep, "lr": opt.param_groups[0]["lr"],
                     "train_loss": tl, "train_rank": tr_rank,
                     "train_abn": tr_abn, "train_cdr": tr_cdr,
                     "val_kendall": vk, "val_abn_macro_f1": vabn,
                     "val_cdr_macro_f1": vcdr, "val_sel": sel,
                     "log_vars": lv.tolist()})
        print(f"ep {ep:>3} lr={opt.param_groups[0]['lr']:.2e} "
              f"loss={tl:.3f}(rk={tr_rank:.3f} abn={tr_abn:.3f} cdr={tr_cdr:.3f}) "
              f"vK={vk:.3f} vAbnF1={vabn:.3f} vCdrF1={vcdr:.3f} "
              f"sel={sel:.4f} s={np.round(lv,2).tolist()}"
              f"{'  *best' if sel < best_sel else ''}")

        if sel < best_sel:
            best_sel, best_ep, bad = sel, ep, 0
            torch.save({"epoch": ep, "pool": args.pool, "head": args.head,
                        "pool_state": model.pool.state_dict(),
                        "order_head_state": model.order_head.state_dict(),
                        "abn_head_state": model.abn_head.state_dict(),
                        "cdr_head_state": model.cdr_head.state_dict(),
                        "log_vars": model.log_vars.detach().cpu(),
                        "val_sel": sel, "val_kendall": vk,
                        "val_abn_macro_f1": vabn, "val_cdr_macro_f1": vcdr},
                       out_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop @ ep {ep} (patience {args.patience})")
                break

    ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.pool.load_state_dict(ck["pool_state"])
    model.order_head.load_state_dict(ck["order_head_state"])
    model.abn_head.load_state_dict(ck["abn_head_state"])
    model.cdr_head.load_state_dict(ck["cdr_head_state"])
    test_res, S, R, AP, AT, CP, CT = evaluate(model, te, device, args.tie_eps)

    summary = {
        "task": "tri_head_order_abn_cdr",
        "pool": args.pool, "head": args.head, "tie_eps": args.tie_eps,
        "loss": "uncertainty_weighted (learnable log_vars rank/abn/cdr)",
        "best_epoch": best_ep, "best_val_sel": best_sel,
        "final_log_vars": ck["log_vars"].tolist(),
        "rim_order": list(RIM_NAMES), "cdr_order": list(CDR_NAMES),
        "data_info": info, "args": vars(args),
        "test_metrics": test_res, "history": hist,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    np.savez(out_dir / "test_preds.npz", scores=S, gt_rim=R,
             abn_pred=AP, abn_true=AT, cdr_pred=CP, cdr_true=CT)

    o = test_res["overall"]
    print(f"\n==== TEST (best ep {best_ep}) pool={args.pool} head={args.head} ====")
    print(f"  ORDER: Kendall={o['kendall']:.4f} exact={o['exact_order']:.3f}")
    print(f"  ABN  : macroF1={o['abn_macro_f1']:.4f} acc={o['abn_acc']:.3f} {o['abn_per_quad_f1']}")
    print(f"  CDR  : macroF1={o['cdr_macro_f1']:.4f} acc={o['cdr_acc']:.3f} {o['cdr_per_ch_f1']}")
    print(f"Done -> {out_dir}/")


if __name__ == "__main__":
    main()

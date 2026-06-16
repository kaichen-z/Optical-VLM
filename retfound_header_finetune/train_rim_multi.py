"""
MULTI-OBJECTIVE rim head — ONE head bank off ONE shared pooled[B,1024], three
outputs predicted JOINTLY, supervised from FIVE angles (the user's hypothesis:
multi-angle supervision of the same rim representation helps every sub-task).

Frozen RETFound -> shared pooling (--pool) -> shared pooled[B,1024] feeds three
linear/mlp heads built off the SAME representation:
    value : [B,1024] -> 4 values, sigmoid -> [0,1]   (rim absolute regression)
    order : [B,1024] -> 4 raw scores (no sigmoid)     (ranking; gap RankNet)
    abn   : [B,1024] -> [B,4,3] logits                (per-quad 3-class)

LOSS = w_value   * L_value                 SmoothL1(value, GT rim)
     + w_order   * L_order                 gap-weighted RankNet(order score, GT order)
     + w_abn     * L_abn                    per-quad class-weighted CE(abn, GT abn)
     + w_vo_cons * L_value_order_consistency
                                            gap RankNet using PREDICTED VALUES as the
                                            scores vs GT-value order (the order implied
                                            by the value head must match GT order)
     + w_va_cons * L_value_abn_consistency
                                            hinge tying predicted VALUES to GT abn class:
                                              normal  (0): value >= thr        (penalize value<thr)
                                              abnorm (>=1): value <  thr        (penalize value>=thr)
                                              severe  (2): value <  thr*0.6     (extra push)
                                            thr = per-quad normal-range lower bound
                                            (I/S 0.30, N 0.22, T 0.18 — from dataset).

All gradients flow into the SHARED pooling. The three heads share the rim feature.

Model selection (lower=better, documented):
    sel = (1 - abn_macroF1)  +  (kendall / 6)  +  value_MAE
  - abn macro-F1   0..1 higher better -> (1 - F1) lower better
  - order kendall  0..6 lower better  -> /6 normalizes to 0..1
  - value MAE      ~0..0.3 lower better (rim units; left unscaled, small by nature)
  All three sub-metrics also logged so each task is visible on its own.

Reuses the SAME 5 poolings + SAME split (seed=42) as train_order_abn.py so the
abn-F1 is directly comparable to the joint order+abn baseline (no-square tier3
abn macroF1 = 0.572).

Usage:
    python train_rim_multi.py --pool tier3
    python train_rim_multi.py --pool tier3 --head mlp --w-va-consistency 0.5
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
from dataset_rim_multi import (build_dataloaders, abn_class_weights,
                               RIM_NAMES, N_ABN_CLASSES, QUAD_LOWER)

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


# --------------------------------------------------------------------------- #
# Heads (LayerNorm BEFORE Linear — RETFound tokens are large; without LN the
# sigmoid saturates / gradients die. This is critical, replicated from tier2/3.)
# --------------------------------------------------------------------------- #
class ValueHead(nn.Module):
    """[LayerNorm] -> (Linear|MLP) -> 4 logits -> sigmoid -> [0,1] rim values."""
    def __init__(self, in_dim=1024, kind="linear", hidden=512, dropout=0.2):
        super().__init__()
        if kind == "linear":
            self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, N_Q))
        elif kind == "mlp":
            self.net = nn.Sequential(nn.LayerNorm(in_dim),
                                     nn.Linear(in_dim, hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, N_Q))
        else:
            raise ValueError(kind)

    def forward(self, x):
        return torch.sigmoid(self.net(x))        # [B,4] in (0,1)


class OrderHead(nn.Module):
    """[LayerNorm] -> (Linear|MLP) -> 4 raw scores (no sigmoid; ranking)."""
    def __init__(self, in_dim=1024, kind="linear", hidden=512, dropout=0.2):
        super().__init__()
        if kind == "linear":
            self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, N_Q))
        elif kind == "mlp":
            self.net = nn.Sequential(nn.LayerNorm(in_dim),
                                     nn.Linear(in_dim, hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, N_Q))
        else:
            raise ValueError(kind)

    def forward(self, x):
        return self.net(x)                       # [B,4]


class AbnHead(nn.Module):
    """[LayerNorm] -> (Linear|MLP) -> 4*3 logits -> [B,4,3]."""
    def __init__(self, in_dim=1024, kind="linear", hidden=512, dropout=0.2):
        super().__init__()
        out = N_Q * N_ABN_CLASSES
        if kind == "linear":
            self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out))
        elif kind == "mlp":
            self.net = nn.Sequential(nn.LayerNorm(in_dim),
                                     nn.Linear(in_dim, hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, out))
        else:
            raise ValueError(kind)

    def forward(self, x):
        return self.net(x).view(-1, N_Q, N_ABN_CLASSES)   # [B,4,3]


class MultiModel(nn.Module):
    def __init__(self, backbone, pool, value_head, order_head, abn_head,
                 lora=False):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.value_head = value_head
        self.order_head = order_head
        self.abn_head = abn_head
        self.lora = lora

    def forward(self, x):
        if self.lora:
            tokens = self.backbone.forward_tokens(x)
        else:
            with torch.no_grad():
                tokens = self.backbone.forward_tokens(x)
        pooled, _ = self.pool(tokens)            # SHARED [B,1024]
        return (self.value_head(pooled),
                self.order_head(pooled),
                self.abn_head(pooled))


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def gap_pairwise_loss(scores, rim, tie_eps):
    """Gap-weighted RankNet over 6 pairs. IDENTICAL to train_order_abn.py.
    `scores` are the things being ranked (order-head scores OR predicted values);
    `rim` provides the GT order (sign) and gap weight."""
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
    """logits [B,4,3], target [B,4] long. Per-quad class-weighted CE, mean /4."""
    loss = logits.new_zeros(())
    for j in range(N_Q):
        loss = loss + F.cross_entropy(logits[:, j, :], target[:, j],
                                      weight=class_w[j])
    return loss / N_Q


def value_abn_consistency_loss(values, abn, lower, severe_frac=0.6):
    """Differentiable hinge tying predicted VALUES to the GT abnormality class.
      normal  (abn==0): value should be >= thr     -> penalty relu(thr - value)
      abnorm  (abn>=1): value should be <  thr      -> penalty relu(value - thr)
      severe  (abn==2): value should be <  thr*frac -> extra relu(value - thr*frac)
    `values` [B,4] in (0,1); `abn` [B,4] long; `lower` [4] per-quad thresholds.
    Mean over all B*4 quadrant entries (severe extra averaged the same way)."""
    thr = lower.view(1, -1)                       # [1,4]
    normal = (abn == 0).float()
    abnormal = (abn >= 1).float()
    severe = (abn == 2).float()
    pen_normal = F.relu(thr - values) * normal             # want value>=thr
    pen_abn = F.relu(values - thr) * abnormal              # want value<thr
    pen_severe = F.relu(values - thr * severe_frac) * severe  # extra push
    return (pen_normal + pen_abn + pen_severe).mean()


# --------------------------------------------------------------------------- #
# Eval / metrics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device, tie_eps):
    model.eval()
    V, S, R, AL, AP, D = [], [], [], [], [], []
    for x, value, abn, _ids, ds in loader:
        x = x.to(device)
        vpred, s, al = model(x)
        V.append(vpred.cpu()); S.append(s.cpu()); R.append(value)
        AL.append(al.cpu()); AP.append(abn); D.extend(ds)
    V = torch.cat(V).numpy()                    # [N,4] predicted values
    S = torch.cat(S).numpy()                    # [N,4] order scores
    R = torch.cat(R).numpy()                    # [N,4] GT rim values
    abn_logits = torch.cat(AL)                  # [N,4,3]
    abn_pred = abn_logits.argmax(-1).numpy()    # [N,4]
    abn_true = torch.cat(AP).numpy()            # [N,4]
    return _metrics(V, S, R, abn_pred, abn_true, D, tie_eps)


def _macro_f1(true, pred, n_cls=N_ABN_CLASSES):
    f1s = []
    for c in range(n_cls):
        tp = np.sum((pred == c) & (true == c))
        fp = np.sum((pred == c) & (true != c))
        fn = np.sum((pred != c) & (true == c))
        if tp + fp == 0 or tp + fn == 0:
            if (true == c).sum() == 0:
                continue
            f1s.append(0.0); continue
        prec = tp / (tp + fp); rec = tp / (tp + fn)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else float("nan")


def _metrics(V, S, R, abn_pred, abn_true, dsets, tie_eps):
    def block(v, s, r, ap, at):
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
        # value regression metrics
        ae = np.abs(v - r)                                  # [n,4]
        value_mae = float(ae.mean())
        value_mae_q = {RIM_NAMES[j]: float(ae[:, j].mean()) for j in range(N_Q)}
        # order implied by predicted VALUES (consistency view at eval)
        vexact = 0
        for i in range(n):
            if list(np.argsort(-v[i])) == list(np.argsort(-r[i])):
                vexact += 1
        # abnormality metrics
        per_q_f1 = [_macro_f1(at[:, j], ap[:, j]) for j in range(N_Q)]
        abn_acc = float((ap == at).mean())
        nb_pred = (ap >= 1).sum(1); nb_true = (at >= 1).sum(1)
        nb_mae = float(np.abs(nb_pred - nb_true).mean())
        return {
            "n": int(n),
            "value_mae": value_mae,
            "value_mae_per_quad": value_mae_q,
            "value_order_exact": float(vexact / n),
            "kendall": float(kendall / n),
            "exact_order": float(exact / n),
            "gap_pair_acc": float(wcorrect / wtot) if wtot > 0 else float("nan"),
            "abn_macro_f1": float(np.nanmean(per_q_f1)),
            "abn_per_quad_f1": {RIM_NAMES[j]: per_q_f1[j] for j in range(N_Q)},
            "abn_acc": abn_acc,
            "n_abn_mae": nb_mae,
        }
    res = {"overall": block(V, S, R, abn_pred, abn_true)}
    dsets = np.array(dsets)
    for d in ("LAG", "Papila"):
        m = dsets == d
        if m.any():
            res[d] = block(V[m], S[m], R[m], abn_pred[m], abn_true[m])
    return res, V, S, R, abn_pred, abn_true


def sel_from(metrics_overall):
    """Combined selection score, lower=better (documented at top of file)."""
    return ((1.0 - metrics_overall["abn_macro_f1"])
            + (metrics_overall["kendall"] / 6.0)
            + metrics_overall["value_mae"])


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="tier3", choices=list(POOLS))
    ap.add_argument("--head", default="linear", choices=["linear", "mlp"])
    # five loss weights
    ap.add_argument("--w-value", type=float, default=1.0)
    ap.add_argument("--w-order", type=float, default=1.0)
    ap.add_argument("--w-abn", type=float, default=1.0)
    ap.add_argument("--w-vo-consistency", type=float, default=0.5,
                    help="value->order consistency (gap RankNet on predicted "
                         "values vs GT order)")
    ap.add_argument("--w-va-consistency", type=float, default=0.5,
                    help="value<->abn consistency hinge")
    ap.add_argument("--smoothl1-beta", type=float, default=0.04,
                    help="SmoothL1 beta on value regression (rim scale)")
    ap.add_argument("--severe-frac", type=float, default=0.6,
                    help="severe threshold = quad_lower * severe_frac")
    ap.add_argument("--tie-eps", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-class-weight", action="store_true")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--square", action="store_true",
                    help="center-crop to square before resize (HURT earlier; off)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = f"rimmulti_{args.pool}_{args.head}_{ts}"
    out_dir = Path(args.out or f"outputs/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out={out_dir}")

    tr, va, te, info = build_dataloaders(
        batch_size=args.batch_size, val_frac=args.val_frac,
        seed=args.seed, num_workers=args.num_workers, augment=args.augment,
        square=args.square)
    print(f"data: {info}")

    backbone = load_retfound(freeze=not args.lora, lora=args.lora,
                             lora_rank=args.lora_rank)
    pool = POOLS[args.pool]()
    value_head = ValueHead(1024, kind=args.head)
    order_head = OrderHead(1024, kind=args.head)
    abn_head = AbnHead(1024, kind=args.head)
    model = MultiModel(backbone, pool, value_head, order_head, abn_head,
                       lora=args.lora).to(device)

    class_w = (torch.ones(N_Q, N_ABN_CLASSES) if args.no_class_weight
               else abn_class_weights(info)).to(device)
    print("abn class weights [4,3]:\n", class_w.cpu().numpy())
    lower = torch.tensor(QUAD_LOWER, dtype=torch.float32, device=device)  # [4]

    params = (list(model.pool.parameters())
              + list(model.value_head.parameters())
              + list(model.order_head.parameters())
              + list(model.abn_head.parameters()))
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
        agg = {"loss": 0., "value": 0., "order": 0., "abn": 0.,
               "vo": 0., "va": 0.}
        for x, value, abn, _i, _d in tr:
            x, value, abn = x.to(device), value.to(device), abn.to(device)
            vpred, scores, abn_logits = model(x)
            l_value = F.smooth_l1_loss(vpred, value, beta=args.smoothl1_beta)
            l_order = gap_pairwise_loss(scores, value, args.tie_eps)
            l_abn = abn_loss_fn(abn_logits, abn, class_w)
            # consistency: predicted-values induce an order -> match GT order
            l_vo = gap_pairwise_loss(vpred, value, args.tie_eps)
            # consistency: predicted values vs GT abn class hinge
            l_va = value_abn_consistency_loss(vpred, abn, lower,
                                              severe_frac=args.severe_frac)
            loss = (args.w_value * l_value + args.w_order * l_order
                    + args.w_abn * l_abn + args.w_vo_consistency * l_vo
                    + args.w_va_consistency * l_va)
            opt.zero_grad(); loss.backward(); opt.step()
            bs = len(x)
            agg["loss"] += loss.item() * bs
            agg["value"] += l_value.item() * bs
            agg["order"] += l_order.item() * bs
            agg["abn"] += l_abn.item() * bs
            agg["vo"] += l_vo.item() * bs
            agg["va"] += l_va.item() * bs
        nrm = max(1, info["n_train"])
        for k in agg:
            agg[k] /= nrm

        vres, *_ = evaluate(model, va, device, args.tie_eps)
        vo = vres["overall"]
        sel = sel_from(vo)
        hist.append({"epoch": ep, "lr": opt.param_groups[0]["lr"],
                     "train_loss": agg["loss"], "train_value": agg["value"],
                     "train_order": agg["order"], "train_abn": agg["abn"],
                     "train_vo": agg["vo"], "train_va": agg["va"],
                     "val_value_mae": vo["value_mae"],
                     "val_kendall": vo["kendall"],
                     "val_abn_macro_f1": vo["abn_macro_f1"],
                     "val_exact": vo["exact_order"],
                     "val_abn_acc": vo["abn_acc"], "val_sel": sel})
        print(f"ep {ep:>3} lr={opt.param_groups[0]['lr']:.2e} "
              f"loss={agg['loss']:.4f}(v={agg['value']:.4f} o={agg['order']:.4f} "
              f"a={agg['abn']:.4f} vo={agg['vo']:.4f} va={agg['va']:.4f}) "
              f"val_MAE={vo['value_mae']:.4f} val_kendall={vo['kendall']:.4f} "
              f"val_abnF1={vo['abn_macro_f1']:.4f} sel={sel:.4f}"
              f"{'  *best' if sel < best_sel else ''}")

        if sel < best_sel:
            best_sel, best_ep, bad = sel, ep, 0
            torch.save({"epoch": ep, "pool": args.pool, "head": args.head,
                        "weights": {"value": args.w_value, "order": args.w_order,
                                    "abn": args.w_abn,
                                    "vo": args.w_vo_consistency,
                                    "va": args.w_va_consistency},
                        "pool_state": model.pool.state_dict(),
                        "value_head_state": model.value_head.state_dict(),
                        "order_head_state": model.order_head.state_dict(),
                        "abn_head_state": model.abn_head.state_dict(),
                        "lora_state": lora_sd(),
                        "val_sel": sel, "val_value_mae": vo["value_mae"],
                        "val_kendall": vo["kendall"],
                        "val_abn_macro_f1": vo["abn_macro_f1"]},
                       out_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop @ ep {ep} (no val improve {args.patience})")
                break

    ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.pool.load_state_dict(ck["pool_state"])
    model.value_head.load_state_dict(ck["value_head_state"])
    model.order_head.load_state_dict(ck["order_head_state"])
    model.abn_head.load_state_dict(ck["abn_head_state"])
    if ck.get("lora_state"):
        model.backbone.load_state_dict(ck["lora_state"], strict=False)
    test_res, V, S, R, AP, AT = evaluate(model, te, device, args.tie_eps)

    summary = {
        "task": "rim_multi_objective_value_order_abn",
        "pool": args.pool, "head": args.head,
        "loss_weights": {"value": args.w_value, "order": args.w_order,
                         "abn": args.w_abn, "vo_consistency": args.w_vo_consistency,
                         "va_consistency": args.w_va_consistency},
        "smoothl1_beta": args.smoothl1_beta, "severe_frac": args.severe_frac,
        "tie_eps": args.tie_eps, "target_order": list(RIM_NAMES),
        "quad_lower": list(QUAD_LOWER),
        "sel_def": "(1 - abn_macroF1) + (kendall/6) + value_MAE  [lower=better]",
        "best_epoch": best_ep, "best_val_sel": best_sel,
        "data_info": info, "args": vars(args),
        "test_metrics": test_res, "history": hist,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    np.savez(out_dir / "test_preds.npz", value_pred=V, order_scores=S,
             gt_rim=R, abn_pred=AP, abn_true=AT)

    o = test_res["overall"]
    print(f"\n==== TEST (best ckpt ep {best_ep}) ====")
    print(f"  VALUE : MAE={o['value_mae']:.4f}  per-quad={o['value_mae_per_quad']}")
    print(f"          value-order exact={o['value_order_exact']:.3f}")
    print(f"  ORDER : Kendall={o['kendall']:.4f} exact={o['exact_order']:.3f} "
          f"gap_pair_acc={o['gap_pair_acc']:.4f}")
    print(f"  ABN   : macroF1={o['abn_macro_f1']:.4f} acc={o['abn_acc']:.3f} "
          f"n_abn_MAE={o['n_abn_mae']:.3f}")
    print(f"          per-quad F1: {o['abn_per_quad_f1']}")
    print(f"  sel={best_sel:.4f}")
    print(f"\nDone. Artifacts in {out_dir}/")


if __name__ == "__main__":
    main()

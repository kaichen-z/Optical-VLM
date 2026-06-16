"""
QUAD-HEAD joint model: rim ORDER + rim ABN + CDR verdict + 5 SIGNS, all off ONE
shared pooling on the frozen RETFound backbone.

    Frozen RETFound -> SHARED pooling (--pool) -> pooled[B,1024]
        ├── OrderHead : -> 4 raw scores    (gap-weighted RankNet)         [step3 order]
        ├── AbnHead   : -> [B,4,3]          (per-quad 0/1/2, class-wt CE)  [step3 rim]
        ├── CdrHead   : -> [B,2,3]          (vert/horiz CDR level, CE)     [step2 CDR]
        └── SignsHead : -> [B,5,3]          (5 signs absent/present/unc,   [step4 signs]
                                             class-wt CE, ignore NA=-1)

Loss = uncertainty-weighted sum (Kendall): 4 learnable log-vars auto-balance.
    total = Σ_i [ exp(-s_i) * L_i + s_i ]       i in {rank, abn, cdr, sign}
Selection (lower=better): kendall/6 + (1-abnF1) + (1-cdrF1) + (1-signF1)

Sweep 5 poolings × 2 head types:
    python train_quad_head.py --pool tier2a --head linear
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
from dataset_quad import (build_dataloaders, abn_class_weights, cdr_class_weights,
                          sign_class_weights, RIM_NAMES, CDR_NAMES, SIGN_NAMES,
                          N_ABN_CLASSES, N_CDR_CLASSES, N_SIGN_CLASSES,
                          N_Q, N_CDR, N_SIGN)

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

PAIRS = list(itertools.combinations(range(N_Q), 2))
SIGN_IGNORE = -1


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
    def forward(self, x): return self.net(x)


class MultiClsHead(nn.Module):
    """-> [B, n_ch, n_cls]."""
    def __init__(self, in_dim, n_ch, n_cls, kind="linear"):
        super().__init__(); self.n_ch = n_ch; self.n_cls = n_cls
        self.net = _head(in_dim, n_ch * n_cls, kind)
    def forward(self, x):
        return self.net(x).view(-1, self.n_ch, self.n_cls)


class QuadModel(nn.Module):
    def __init__(self, backbone, pool, order_head, abn_head, cdr_head, signs_head):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.order_head = order_head
        self.abn_head = abn_head
        self.cdr_head = cdr_head
        self.signs_head = signs_head
        self.log_vars = nn.Parameter(torch.zeros(4))   # rank, abn, cdr, sign

    def forward(self, x):
        with torch.no_grad():
            tokens = self.backbone.forward_tokens(x)
        pooled, _ = self.pool(tokens)
        return (self.order_head(pooled), self.abn_head(pooled),
                self.cdr_head(pooled), self.signs_head(pooled))

    def combine(self, l_rank, l_abn, l_cdr, l_sign):
        losses = torch.stack([l_rank, l_abn, l_cdr, l_sign])
        return (torch.exp(-self.log_vars) * losses + self.log_vars).sum()


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


def _ce_multi(logits, target, class_w, n_ch, ignore=None):
    """mean class-weighted CE over channels; ignore-index channels with no
    valid labels in the batch are skipped (avoids nan)."""
    loss = logits.new_zeros(()); cnt = 0
    for j in range(n_ch):
        t = target[:, j]
        if ignore is not None:
            if (t != ignore).sum() == 0:
                continue
            loss = loss + F.cross_entropy(logits[:, j, :], t, weight=class_w[j],
                                          ignore_index=ignore)
        else:
            loss = loss + F.cross_entropy(logits[:, j, :], t, weight=class_w[j])
        cnt += 1
    return loss / max(cnt, 1)


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
    S, R, AL, AP, CL, CP, GL, GP, D = [], [], [], [], [], [], [], [], []
    for x, rim, abn, cdr, sign, _ids, ds in loader:
        x = x.to(device)
        s, al, cl, gl = model(x)
        S.append(s.cpu()); R.append(rim)
        AL.append(al.cpu()); AP.append(abn)
        CL.append(cl.cpu()); CP.append(cdr)
        GL.append(gl.cpu()); GP.append(sign)
        D.extend(ds)
    S = torch.cat(S).numpy(); R = torch.cat(R).numpy()
    abn_pred = torch.cat(AL).argmax(-1).numpy(); abn_true = torch.cat(AP).numpy()
    cdr_pred = torch.cat(CL).argmax(-1).numpy(); cdr_true = torch.cat(CP).numpy()
    sign_pred = torch.cat(GL).argmax(-1).numpy(); sign_true = torch.cat(GP).numpy()
    return _metrics(S, R, abn_pred, abn_true, cdr_pred, cdr_true,
                    sign_pred, sign_true, D, tie_eps)


def _metrics(S, R, abn_pred, abn_true, cdr_pred, cdr_true,
             sign_pred, sign_true, dsets, tie_eps):
    def block(s, r, ap, at, cp, ct, gp, gt):
        n = len(s)
        if n == 0:
            return {}
        exact = 0; kendall = 0.0
        for i in range(n):
            op = list(np.argsort(-s[i])); og = list(np.argsort(-r[i]))
            if op == og:
                exact += 1
            posg = {q: k for k, q in enumerate(og)}
            kendall += sum(1 for a, b in itertools.combinations(op, 2)
                           if posg[a] > posg[b])
        abn_f1 = [_macro_f1(at[:, j], ap[:, j], N_ABN_CLASSES) for j in range(N_Q)]
        cdr_f1 = [_macro_f1(ct[:, j], cp[:, j], N_CDR_CLASSES) for j in range(N_CDR)]
        # signs: per-disease, mask NA (-1)
        sign_f1 = []; sign_acc = []
        for j in range(N_SIGN):
            m = gt[:, j] != SIGN_IGNORE
            if m.sum() == 0:
                sign_f1.append(float("nan")); sign_acc.append(float("nan")); continue
            sign_f1.append(_macro_f1(gt[m, j], gp[m, j], N_SIGN_CLASSES))
            sign_acc.append(float((gp[m, j] == gt[m, j]).mean()))
        return {
            "n": int(n),
            "kendall": float(kendall / n),
            "exact_order": float(exact / n),
            "abn_macro_f1": float(np.nanmean(abn_f1)),
            "abn_per_quad_f1": {RIM_NAMES[j]: abn_f1[j] for j in range(N_Q)},
            "abn_acc": float((ap == at).mean()),
            "cdr_macro_f1": float(np.nanmean(cdr_f1)),
            "cdr_acc": float((cp == ct).mean()),
            "sign_macro_f1": float(np.nanmean(sign_f1)),
            "sign_per_disease_f1": {SIGN_NAMES[j]: sign_f1[j] for j in range(N_SIGN)},
            "sign_acc": float(np.nanmean(sign_acc)),
        }
    res = {"overall": block(S, R, abn_pred, abn_true, cdr_pred, cdr_true,
                            sign_pred, sign_true)}
    dsets = np.array(dsets)
    for d in ("LAG", "Papila"):
        m = dsets == d
        if m.any():
            res[d] = block(S[m], R[m], abn_pred[m], abn_true[m], cdr_pred[m],
                           cdr_true[m], sign_pred[m], sign_true[m])
    return res, S, R, abn_pred, abn_true, cdr_pred, cdr_true, sign_pred, sign_true


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
    out_dir = Path(args.out or f"outputs/quad_{args.pool}_{args.head}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out={out_dir}")

    tr, va, te, info = build_dataloaders(
        batch_size=args.batch_size, val_frac=args.val_frac,
        seed=args.seed, num_workers=args.num_workers)
    print(f"data: n_train={info['n_train']} n_val={info['n_val']} n_test={info['n_test']}")

    backbone = load_retfound(freeze=True, lora=False)
    pool = POOLS[args.pool]()
    order_head = OrderHead(1024, kind=args.head)
    abn_head = MultiClsHead(1024, N_Q, N_ABN_CLASSES, kind=args.head)
    cdr_head = MultiClsHead(1024, N_CDR, N_CDR_CLASSES, kind=args.head)
    signs_head = MultiClsHead(1024, N_SIGN, N_SIGN_CLASSES, kind=args.head)
    model = QuadModel(backbone, pool, order_head, abn_head, cdr_head, signs_head).to(device)

    if args.no_class_weight:
        abn_w = torch.ones(N_Q, N_ABN_CLASSES).to(device)
        cdr_w = torch.ones(N_CDR, N_CDR_CLASSES).to(device)
        sign_w = torch.ones(N_SIGN, N_SIGN_CLASSES).to(device)
    else:
        abn_w = abn_class_weights(info).to(device)
        cdr_w = cdr_class_weights(info).to(device)
        sign_w = sign_class_weights(info).to(device)

    params = (list(model.pool.parameters())
              + list(model.order_head.parameters())
              + list(model.abn_head.parameters())
              + list(model.cdr_head.parameters())
              + list(model.signs_head.parameters())
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
        tl = 0.0
        for x, rim, abn, cdr, sign, _i, _d in tr:
            x, rim = x.to(device), rim.to(device)
            abn, cdr, sign = abn.to(device), cdr.to(device), sign.to(device)
            scores, abn_logits, cdr_logits, sign_logits = model(x)
            l_rank = gap_pairwise_loss(scores, rim, args.tie_eps)
            l_abn = _ce_multi(abn_logits, abn, abn_w, N_Q)
            l_cdr = _ce_multi(cdr_logits, cdr, cdr_w, N_CDR)
            l_sign = _ce_multi(sign_logits, sign, sign_w, N_SIGN, ignore=SIGN_IGNORE)
            loss = model.combine(l_rank, l_abn, l_cdr, l_sign)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(x)
        tl /= max(1, info["n_train"])

        vres, *_ = evaluate(model, va, device, args.tie_eps)
        o = vres["overall"]
        vk, va_, vc, vg = (o["kendall"], o["abn_macro_f1"],
                           o["cdr_macro_f1"], o["sign_macro_f1"])
        sel = vk / 6.0 + (1 - va_) + (1 - vc) + (1 - vg)
        lv = model.log_vars.detach().cpu().numpy()
        hist.append({"epoch": ep, "train_loss": tl, "val_kendall": vk,
                     "val_abn_f1": va_, "val_cdr_f1": vc, "val_sign_f1": vg,
                     "val_sel": sel, "log_vars": lv.tolist()})
        print(f"ep {ep:>3} loss={tl:.3f} vK={vk:.3f} vAbn={va_:.3f} "
              f"vCdr={vc:.3f} vSign={vg:.3f} sel={sel:.4f} "
              f"s={np.round(lv,2).tolist()}{'  *best' if sel < best_sel else ''}")

        if sel < best_sel:
            best_sel, best_ep, bad = sel, ep, 0
            torch.save({"epoch": ep, "pool": args.pool, "head": args.head,
                        "pool_state": model.pool.state_dict(),
                        "order_head_state": model.order_head.state_dict(),
                        "abn_head_state": model.abn_head.state_dict(),
                        "cdr_head_state": model.cdr_head.state_dict(),
                        "signs_head_state": model.signs_head.state_dict(),
                        "log_vars": model.log_vars.detach().cpu(),
                        "val_sel": sel}, out_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop @ ep {ep}")
                break

    ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.pool.load_state_dict(ck["pool_state"])
    model.order_head.load_state_dict(ck["order_head_state"])
    model.abn_head.load_state_dict(ck["abn_head_state"])
    model.cdr_head.load_state_dict(ck["cdr_head_state"])
    model.signs_head.load_state_dict(ck["signs_head_state"])
    test_res, S, R, AP, AT, CP, CT, GP, GT = evaluate(model, te, device, args.tie_eps)

    summary = {
        "task": "quad_head_order_abn_cdr_signs",
        "pool": args.pool, "head": args.head, "tie_eps": args.tie_eps,
        "loss": "uncertainty_weighted (4 learnable log_vars)",
        "best_epoch": best_ep, "best_val_sel": best_sel,
        "final_log_vars": ck["log_vars"].tolist(),
        "rim_order": list(RIM_NAMES), "cdr_order": list(CDR_NAMES),
        "sign_order": list(SIGN_NAMES),
        "data_info": info, "args": vars(args),
        "test_metrics": test_res, "history": hist,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    test_ids = [r["id"] for r in te.dataset.rows]   # shuffle=False -> aligned with preds
    np.savez(out_dir / "test_preds.npz", scores=S, gt_rim=R,
             abn_pred=AP, abn_true=AT, cdr_pred=CP, cdr_true=CT,
             sign_pred=GP, sign_true=GT, ids=np.array(test_ids))

    o = test_res["overall"]
    print(f"\n==== TEST (best ep {best_ep}) pool={args.pool} head={args.head} ====")
    print(f"  ORDER: Kendall={o['kendall']:.4f} exact={o['exact_order']:.3f}")
    print(f"  ABN  : macroF1={o['abn_macro_f1']:.4f} acc={o['abn_acc']:.3f}")
    print(f"  CDR  : macroF1={o['cdr_macro_f1']:.4f} acc={o['cdr_acc']:.3f}")
    print(f"  SIGNS: macroF1={o['sign_macro_f1']:.4f} acc={o['sign_acc']:.3f} {o['sign_per_disease_f1']}")
    print(f"Done -> {out_dir}/")


if __name__ == "__main__":
    main()

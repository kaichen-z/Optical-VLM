"""Train the shared-pooling multi-head on a frozen RETFound backbone.

    pooled[B,1024]
        |-- OrderHead   -> 4 scores        rim order, gap-weighted RankNet
        |-- MultiClsHead -> [B,4,3]         per-quadrant rim status, weighted CE
        |-- CdrRegHead  -> [B,2] in [0,1]   vertical/horizontal CDR, SmoothL1
        |-- MultiClsHead -> [B,5,3]         glaucomatous signs, weighted CE
        |-- DxHead      -> [B,2]            binary glaucoma, weighted CE

The five losses are combined with learnable uncertainty weights. The checkpoint
is selected on the four intermediate indicators; the diagnosis head is trained
and reported but does not drive selection.

Pick one of the five poolings and one of the two head kinds:
    python train.py --pool tier2a --head linear
"""
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
from poolings import make_pool, POOL_NAMES
from heads import OrderHead, MultiClsHead, CdrRegHead, DxHead
from dataset import (build_dataloaders, abn_class_weights, sign_class_weights,
                     dx_class_weights, RIM_NAMES, SIGN_NAMES,
                     N_ABN_CLASSES, N_SIGN_CLASSES, N_Q, N_SIGN, N_DX)

PAIRS = list(itertools.combinations(range(N_Q), 2))
SIGN_IGNORE = -1
CDR_BETA = 0.075


class MultiHeadModel(nn.Module):
    def __init__(self, backbone, pool, order_head, abn_head, cdr_head, signs_head, dx_head):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.order_head = order_head
        self.abn_head = abn_head
        self.cdr_head = cdr_head
        self.signs_head = signs_head
        self.dx_head = dx_head
        self.log_vars = nn.Parameter(torch.zeros(5))   # rank, abn, cdr, sign, dx

    def forward(self, x):
        with torch.no_grad():
            tokens = self.backbone.forward_tokens(x)
        pooled, _ = self.pool(tokens)
        return (self.order_head(pooled), self.abn_head(pooled), self.cdr_head(pooled),
                self.signs_head(pooled), self.dx_head(pooled))

    def combine(self, *losses):
        stacked = torch.stack(losses)
        return (torch.exp(-self.log_vars) * stacked + self.log_vars).sum()


def gap_pairwise_loss(scores, rim, tie_eps):
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


def ce_multi(logits, target, class_w, n_ch, ignore=None):
    loss = logits.new_zeros(())
    cnt = 0
    for j in range(n_ch):
        t = target[:, j]
        if ignore is not None and (t != ignore).sum() == 0:
            continue
        loss = loss + F.cross_entropy(logits[:, j, :], t, weight=class_w[j],
                                      ignore_index=ignore if ignore is not None else -100)
        cnt += 1
    return loss / max(cnt, 1)


def macro_f1(true, pred, n_cls):
    f1s = []
    for c in range(n_cls):
        if (true == c).sum() == 0:
            continue
        tp = np.sum((pred == c) & (true == c))
        fp = np.sum((pred == c) & (true != c))
        fn = np.sum((pred != c) & (true == c))
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else float("nan")


def pearson(a, b):
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def roc_auc(y_true, y_score):
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    pos, neg = y_score[y_true == 1], y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score))
    ranks[order] = np.arange(1, len(y_score) + 1)
    _, inv, cnt = np.unique(y_score, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def dx_metrics(dx_true, dx_pred, dx_prob):
    dx_true, dx_pred = np.asarray(dx_true), np.asarray(dx_pred)
    tp = int(((dx_pred == 1) & (dx_true == 1)).sum())
    tn = int(((dx_pred == 0) & (dx_true == 0)).sum())
    fp = int(((dx_pred == 1) & (dx_true == 0)).sum())
    fn = int(((dx_pred == 0) & (dx_true == 1)).sum())
    sn = tp / (tp + fn) if tp + fn else 0.0
    sp = tn / (tn + fp) if tn + fp else 0.0
    return {"balanced_acc": (sn + sp) / 2, "accuracy": (tp + tn) / max(len(dx_true), 1),
            "sensitivity": sn, "specificity": sp, "auc": roc_auc(dx_true, dx_prob),
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}}


def indicator_metrics(scores, rim, abn_pred, abn_true, cdr_pred, cdr_true,
                      sign_pred, sign_true, tie_eps):
    n = len(scores)
    exact = 0
    kendall = 0.0
    for i in range(n):
        op = list(np.argsort(-scores[i]))
        og = list(np.argsort(-rim[i]))
        if op == og:
            exact += 1
        pos = {q: k for k, q in enumerate(og)}
        kendall += sum(1 for a, b in itertools.combinations(op, 2) if pos[a] > pos[b])
    abn_f1 = [macro_f1(abn_true[:, j], abn_pred[:, j], N_ABN_CLASSES) for j in range(N_Q)]
    mae_v = float(np.abs(cdr_pred[:, 0] - cdr_true[:, 0]).mean())
    mae_h = float(np.abs(cdr_pred[:, 1] - cdr_true[:, 1]).mean())
    sign_f1 = []
    for j in range(N_SIGN):
        m = sign_true[:, j] != SIGN_IGNORE
        sign_f1.append(macro_f1(sign_true[m, j], sign_pred[m, j], N_SIGN_CLASSES)
                       if m.sum() else float("nan"))
    return {
        "n": n, "kendall": float(kendall / n), "exact_order": float(exact / n),
        "abn_macro_f1": float(np.nanmean(abn_f1)),
        "abn_per_quad_f1": {RIM_NAMES[j]: abn_f1[j] for j in range(N_Q)},
        "cdr_mae": float((mae_v + mae_h) / 2), "cdr_mae_v": mae_v, "cdr_mae_h": mae_h,
        "cdr_pearson_v": pearson(cdr_pred[:, 0], cdr_true[:, 0]),
        "cdr_pearson_h": pearson(cdr_pred[:, 1], cdr_true[:, 1]),
        "sign_macro_f1": float(np.nanmean(sign_f1)),
        "sign_per_disease_f1": {SIGN_NAMES[j]: sign_f1[j] for j in range(N_SIGN)},
    }


@torch.no_grad()
def evaluate(model, loader, device, tie_eps):
    model.eval()
    buf = {k: [] for k in ("s", "rim", "abn_l", "abn_t", "cdr_p", "cdr_t",
                           "sign_l", "sign_t", "dx_l", "dx_t")}
    ids = []
    for x, rim, abn, cdr, sign, dx, id_, _ds in loader:
        s, al, cp, sl, dl = model(x.to(device))
        buf["s"].append(s.cpu()); buf["rim"].append(rim)
        buf["abn_l"].append(al.cpu()); buf["abn_t"].append(abn)
        buf["cdr_p"].append(cp.cpu()); buf["cdr_t"].append(cdr)
        buf["sign_l"].append(sl.cpu()); buf["sign_t"].append(sign)
        buf["dx_l"].append(dl.cpu()); buf["dx_t"].append(dx)
        ids.extend(id_)
    s = torch.cat(buf["s"]).numpy()
    rim = torch.cat(buf["rim"]).numpy()
    abn_pred = torch.cat(buf["abn_l"]).argmax(-1).numpy()
    abn_true = torch.cat(buf["abn_t"]).numpy()
    cdr_pred = torch.cat(buf["cdr_p"]).numpy()
    cdr_true = torch.cat(buf["cdr_t"]).numpy()
    sign_pred = torch.cat(buf["sign_l"]).argmax(-1).numpy()
    sign_true = torch.cat(buf["sign_t"]).numpy()
    dx_logits = torch.cat(buf["dx_l"])
    dx_prob = torch.softmax(dx_logits, -1)[:, 1].numpy()
    dx_pred = dx_logits.argmax(-1).numpy()
    dx_true = torch.cat(buf["dx_t"]).numpy()
    ind = indicator_metrics(s, rim, abn_pred, abn_true, cdr_pred, cdr_true,
                            sign_pred, sign_true, tie_eps)
    return ind, dx_metrics(dx_true, dx_pred, dx_prob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, choices=POOL_NAMES)
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
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out or f"outputs/{args.pool}_{args.head}_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tr, va, te, info = build_dataloaders(batch_size=args.batch_size, val_frac=args.val_frac,
                                         seed=args.seed, num_workers=args.num_workers)
    print(f"data: train={info['n_train']} val={info['n_val']} test={info['n_test']}  "
          f"dx_train={info['dx_train_counts']}")

    backbone = load_retfound(freeze=True)
    dim = 1024
    model = MultiHeadModel(
        backbone, make_pool(args.pool, dim),
        OrderHead(dim, N_Q, args.head),
        MultiClsHead(dim, N_Q, N_ABN_CLASSES, args.head),
        CdrRegHead(dim, kind=args.head),
        MultiClsHead(dim, N_SIGN, N_SIGN_CLASSES, args.head),
        DxHead(dim, args.head),
    ).to(device)

    abn_w = abn_class_weights(info).to(device)
    sign_w = sign_class_weights(info).to(device)
    dx_w = dx_class_weights(info).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    def lr_scale(ep):
        if ep < args.warmup_epochs:
            return (ep + 1) / max(1, args.warmup_epochs)
        prog = (ep - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * prog))

    best_sel = float("inf")
    best_ep = -1
    bad = 0
    hist = []

    for ep in range(args.epochs):
        for g in opt.param_groups:
            g["lr"] = args.lr * lr_scale(ep)
        model.train()
        model.backbone.eval()
        running = 0.0
        for x, rim, abn, cdr, sign, dx, _i, _d in tr:
            x, rim = x.to(device), rim.to(device)
            abn, cdr, sign, dx = abn.to(device), cdr.to(device), sign.to(device), dx.to(device)
            scores, abn_logits, cdr_pred, sign_logits, dx_logits = model(x)
            l_rank = gap_pairwise_loss(scores, rim, args.tie_eps)
            l_abn = ce_multi(abn_logits, abn, abn_w, N_Q)
            l_cdr = F.smooth_l1_loss(cdr_pred, cdr, beta=CDR_BETA)
            l_sign = ce_multi(sign_logits, sign, sign_w, N_SIGN, ignore=SIGN_IGNORE)
            l_dx = F.cross_entropy(dx_logits, dx, weight=dx_w)
            loss = model.combine(l_rank, l_abn, l_cdr, l_sign, l_dx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(x)
        train_loss = running / max(1, info["n_train"])

        vind, vdx = evaluate(model, va, device, args.tie_eps)
        sel = vind["kendall"] / 6.0 + (1 - vind["abn_macro_f1"]) + \
            vind["cdr_mae"] * 5.0 + (1 - vind["sign_macro_f1"])
        hist.append({"epoch": ep, "train_loss": train_loss, "val_sel": sel,
                     "val_dx_bacc": vdx["balanced_acc"], "val_dx_auc": vdx["auc"]})
        print(f"ep {ep:>3} loss={train_loss:.3f} sel={sel:.4f} "
              f"vDXbacc={vdx['balanced_acc']:.3f} vDXauc={vdx['auc']:.3f}"
              f"{'  *best' if sel < best_sel else ''}")

        if sel < best_sel:
            best_sel, best_ep, bad = sel, ep, 0
            torch.save({"epoch": ep, "pool": args.pool, "head": args.head,
                        "pool_state": model.pool.state_dict(),
                        "order_head_state": model.order_head.state_dict(),
                        "abn_head_state": model.abn_head.state_dict(),
                        "cdr_head_state": model.cdr_head.state_dict(),
                        "signs_head_state": model.signs_head.state_dict(),
                        "dx_head_state": model.dx_head.state_dict()},
                       out_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop @ ep {ep}")
                break

    ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    for name in ("pool", "order_head", "abn_head", "cdr_head", "signs_head", "dx_head"):
        getattr(model, name).load_state_dict(ck[f"{name}_state"])
    test_ind, test_dx = evaluate(model, te, device, args.tie_eps)

    summary = {"pool": args.pool, "head": args.head, "best_epoch": best_ep,
               "args": vars(args), "test_indicators": test_ind, "test_dx": test_dx,
               "history": hist}
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    print(f"\n== TEST (best epoch {best_ep}) pool={args.pool} head={args.head} ==")
    print(f"  Kendall={test_ind['kendall']:.4f}  rim-F1={test_ind['abn_macro_f1']:.4f}  "
          f"CDR-MAE={test_ind['cdr_mae']:.4f}  signs-F1={test_ind['sign_macro_f1']:.4f}")
    print(f"  dx: bacc={test_dx['balanced_acc']:.4f} AUC={test_dx['auc']:.4f} "
          f"Sn={test_dx['sensitivity']:.3f} Sp={test_dx['specificity']:.3f}")
    print(f"saved -> {out_dir}/")


if __name__ == "__main__":
    main()

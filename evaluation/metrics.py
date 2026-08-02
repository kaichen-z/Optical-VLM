"""Metric functions for the two evaluation axes: the Stage-1 clinical indicators
and the Stage-2 diagnosis. Pure numpy, no sklearn."""
import itertools

import numpy as np


# ---------------------------------------------------------------- classification
def classification(y_true, y_pred):
    """y_true / y_pred: 0/1 arrays (1 = glaucoma). Returns balanced accuracy,
    sensitivity, specificity, accuracy, MCC and the confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sn = tp / (tp + fn) if tp + fn else 0.0
    sp = tn / (tn + fp) if tn + fp else 0.0
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {
        "balanced_acc": (sn + sp) / 2,
        "sensitivity": sn,
        "specificity": sp,
        "accuracy": (tp + tn) / max(len(y_true), 1),
        "mcc": float(mcc),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "n": int(len(y_true)),
    }


# ---------------------------------------------------------------- CDR (numeric)
def _pearson(a, b):
    if len(a) < 2 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def cdr(pred, gt):
    """pred / gt: 1-D arrays of one CDR axis. MAE, RMSE, signed bias, Pearson r."""
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    err = pred - gt
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
        "pearson": _pearson(pred, gt),
        "n": int(len(gt)),
    }


# ---------------------------------------------------------------- ISNT rim order
def isnt_order(pred_orders, gt_orders):
    """pred_orders / gt_orders: lists of 4-element sequences (the four rim sectors
    ranked thickest -> thinnest). Kendall discordant-pair count (0-6, lower better),
    chance-scaled tau (1 - disc/3), and exact-order match rate."""
    disc_total = 0.0
    exact = 0
    n = 0
    for po, go in zip(pred_orders, gt_orders):
        if po is None or go is None:
            continue
        n += 1
        if list(po) == list(go):
            exact += 1
        gpos = {q: i for i, q in enumerate(go)}
        ppos = {q: i for i, q in enumerate(po)}
        disc = sum(1 for a, b in itertools.combinations(go, 2)
                   if (gpos[a] < gpos[b]) != (ppos[a] < ppos[b]))
        disc_total += disc
    if n == 0:
        return {"kendall_discordant": float("nan"), "kendall_tau": float("nan"),
                "exact_match": float("nan"), "n": 0}
    disc_mean = disc_total / n
    return {
        "kendall_discordant": disc_mean,
        "kendall_tau": 1 - disc_mean / 3.0,
        "exact_match": exact / n,
        "n": n,
    }


# ---------------------------------------------------------------- ordinal grades
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


def quadratic_weighted_kappa(true, pred, n_cls):
    """QWK for ordinal labels: penalizes errors by squared rank distance."""
    true, pred = np.asarray(true), np.asarray(pred)
    o = np.zeros((n_cls, n_cls))
    for t, p in zip(true, pred):
        o[t, p] += 1
    w = np.array([[(i - j) ** 2 / (n_cls - 1) ** 2 for j in range(n_cls)]
                  for i in range(n_cls)])
    act = o.sum(1)
    prd = o.sum(0)
    e = np.outer(act, prd) / o.sum()
    denom = (w * e).sum()
    return float(1 - (w * o).sum() / denom) if denom else float("nan")


def rim_status(pred, gt, n_cls=3):
    """pred / gt: [N, 4] integer grades (0 normal / 1 mild / 2 severe) over the four
    rim quadrants. macro-F1, accuracy, QWK and ordinal (rank-distance) MAE, pooled
    over all quadrants."""
    pred, gt = np.asarray(pred).ravel(), np.asarray(gt).ravel()
    return {
        "macro_f1": macro_f1(gt, pred, n_cls),
        "accuracy": float((pred == gt).mean()),
        "qwk": quadratic_weighted_kappa(gt, pred, n_cls),
        "ordinal_mae": float(np.abs(pred - gt).mean()),
        "n": int(len(gt)),
    }


# ---------------------------------------------------------------- signs
def signs(pred, gt, n_cls=3, ignore=-1):
    """pred / gt: [N, K] labels (0 absent / 1 present / 2 uncertain); `ignore`
    marks items with no ground truth. Primary metrics are computed only over
    items where the ground truth is not `absent`, so a model cannot score by
    predicting the majority 'absent' class. FULL metrics (incl. absent) are
    reported alongside."""
    pred, gt = np.asarray(pred).ravel(), np.asarray(gt).ravel()
    keep = gt != ignore
    pred, gt = pred[keep], gt[keep]

    nz = gt != 0                                  # ground-truth present or uncertain
    present_true = gt == 1
    present_pred = pred == 1
    pr_rec = float((present_pred & present_true).sum() / max(present_true.sum(), 1))
    pr_prec = float((present_pred & present_true).sum() / max(present_pred.sum(), 1))
    detect = float(((pred != 0) & nz).sum() / max(nz.sum(), 1))   # noticed anything
    return {
        "nz_accuracy": float((pred[nz] == gt[nz]).mean()) if nz.sum() else float("nan"),
        "nz_macro_f1": macro_f1(gt[nz], pred[nz], n_cls) if nz.sum() else float("nan"),
        "present_recall": pr_rec,
        "present_precision": pr_prec,
        "detection_recall": detect,
        "full_accuracy": float((pred == gt).mean()),
        "full_macro_f1": macro_f1(gt, pred, n_cls),
        "n_nonzero": int(nz.sum()),
        "n_total": int(len(gt)),
    }

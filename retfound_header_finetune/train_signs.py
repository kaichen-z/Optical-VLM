"""
Train a 5-disease x 3-class glaucomatous-signs classifier on a frozen
RETFound backbone.

Pipeline:  image -> RETFound(frozen) -> [B,197,1024]
                  -> pooling (--pool) -> [B,1024]
                  -> SignsHead        -> 15 logits -> view [B,5,3]
           per disease: 3-way softmax over (absent, present, uncertain)

Loss (per the agreed design):
  * 5 INDEPENDENT 3-way cross-entropies (diseases are independent;
    the 3 classes within a disease are mutually exclusive)
  * NA (-1) is ignored (ignore_index) — not supervised, not in the denom
  * inverse-frequency CLASS WEIGHTS per disease (train split) so the rare
    present/uncertain classes are not drowned by ~75% absent
    (--no-class-weight to disable;  --loss focal for focal variant)
  * total = mean over the 5 diseases (equal weight)

Select : best ckpt by max VAL mean macro-F1 over the 5 diseases
         (accuracy is a base-rate trap here and is reported but NOT used).
Report : per-disease per-class recall + macro-F1, overall, LAG vs Papila.

Reuses the SAME 5 poolings + frozen backbone + data split as CDR/ISNT.
Independent of train.py / train_isnt.py / train_order.py.

Usage:
    python train_signs.py --pool tier2a
    python train_signs.py --pool tier3 --loss focal
    python train_signs.py --pool tier2a --no-class-weight
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from retfound_backbone import load_retfound
from dataset_signs import (build_dataloaders, SIGN_NAMES,
                           NUM_DISEASE, NUM_CLASS)

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
CLASS_NAMES = ("absent", "present", "uncertain")


class SignsHead(nn.Module):
    """LayerNorm -> (Linear|MLP) -> 5*3 raw logits -> [B,5,3]."""

    def __init__(self, in_dim=1024, kind="linear", hidden=512, dropout=0.2):
        super().__init__()
        out = NUM_DISEASE * NUM_CLASS
        if kind == "linear":
            self.net = nn.Sequential(nn.LayerNorm(in_dim),
                                     nn.Linear(in_dim, out))
        elif kind == "mlp":
            self.net = nn.Sequential(nn.LayerNorm(in_dim),
                                     nn.Linear(in_dim, hidden), nn.GELU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden, out))
        else:
            raise ValueError(kind)

    def forward(self, x):
        return self.net(x).view(-1, NUM_DISEASE, NUM_CLASS)


class SignsModel(nn.Module):
    def __init__(self, backbone, pool, head, lora=False):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.head = head
        self.lora = lora

    def forward(self, x):
        if self.lora:
            tokens = self.backbone.forward_tokens(x)
        else:
            with torch.no_grad():
                tokens = self.backbone.forward_tokens(x)
        pooled, _ = self.pool(tokens)
        return self.head(pooled)                  # [B,5,3]


def class_weights(train_cc):
    """train_cc: [5][3] counts. w[d,c] = Nvalid_d / (K * n_dc); n_dc==0->0."""
    W = torch.zeros(NUM_DISEASE, NUM_CLASS)
    for d in range(NUM_DISEASE):
        n = train_cc[d]
        tot = sum(n)
        K = sum(1 for x in n if x > 0)
        for c in range(NUM_CLASS):
            W[d, c] = (tot / (K * n[c])) if n[c] > 0 else 0.0
    return W


def signs_loss(logits, y, w, use_focal, gamma=2.0):
    """logits [B,5,3], y [B,5] in {-1,0,1,2}. Mean over diseases of the
    masked, class-weighted CE (or focal). Diseases with no valid label in
    the batch are skipped."""
    total = logits.new_zeros(())
    nd = 0
    for d in range(NUM_DISEASE):
        ld, yd = logits[:, d, :], y[:, d]
        m = yd >= 0
        if m.sum() == 0:
            continue
        ld, yd = ld[m], yd[m]
        wd = w[d].to(ld.device)
        if use_focal:
            logp = F.log_softmax(ld, dim=1)
            p = logp.exp()
            pt = p.gather(1, yd[:, None]).squeeze(1)
            wt = wd[yd]
            loss_d = (-wt * (1 - pt) ** gamma
                      * logp.gather(1, yd[:, None]).squeeze(1)).mean()
        else:
            loss_d = F.cross_entropy(ld, yd, weight=wd)
        total = total + loss_d
        nd += 1
    return total / max(nd, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    P, Y, D = [], [], []
    for x, y, _ids, ds in loader:
        x = x.to(device)
        pred = model(x).argmax(-1).cpu()          # [B,5]
        P.append(pred); Y.append(y); D.extend(ds)
    P = torch.cat(P).numpy()
    Y = torch.cat(Y).numpy()
    return _metrics(P, Y, D), P, Y


def _disease_block(pred, gt):
    """pred,gt: [N] in {0,1,2}, gt may be -1 (skip). Return per-class
    recall + macro-F1 over classes present in gt."""
    valid = gt >= 0
    pred, gt = pred[valid], gt[valid]
    out = {"n": int(len(gt))}
    if len(gt) == 0:
        return out
    f1s, recs = [], {}
    for c in range(NUM_CLASS):
        tp = int(((pred == c) & (gt == c)).sum())
        fp = int(((pred == c) & (gt != c)).sum())
        fn = int(((pred != c) & (gt == c)).sum())
        sup = int((gt == c).sum())
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (2 * prec * rec / (prec + rec)
              if (prec + rec) and not np.isnan(rec) else 0.0)
        recs[CLASS_NAMES[c]] = rec
        if sup > 0:                               # macro over present classes
            f1s.append(f1)
    out["recall"] = recs
    out["macro_f1"] = float(np.mean(f1s)) if f1s else 0.0
    out["acc"] = float((pred == gt).mean())       # base-rate-inflated; FYI
    return out


def _metrics(P, Y, dsets):
    dsets = np.array(dsets)
    res = {}
    for scope, mask in [("overall", np.ones(len(P), bool)),
                        ("LAG", dsets == "LAG"),
                        ("Papila", dsets == "Papila")]:
        if mask.sum() == 0:
            continue
        per = {}
        mf1 = []
        for d, name in enumerate(SIGN_NAMES):
            b = _disease_block(P[mask, d], Y[mask, d])
            per[name] = b
            if "macro_f1" in b:
                mf1.append(b["macro_f1"])
        res[scope] = {"mean_macro_f1": float(np.mean(mf1)) if mf1 else 0.0,
                      "per_disease": per}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, choices=list(POOLS))
    ap.add_argument("--head", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--loss", default="wce", choices=["wce", "focal"],
                    help="wce=weighted CE (default), focal=focal loss")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--no-class-weight", action="store_true",
                    help="disable inverse-freq class weights (ablation)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=10)
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
    tag = f"signs_{args.pool}_{args.head}_{args.loss}_{ts}"
    out_dir = Path(args.out or f"outputs/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out={out_dir}")

    tr, va, te, info = build_dataloaders(
        batch_size=args.batch_size, val_frac=args.val_frac,
        seed=args.seed, num_workers=args.num_workers)
    print(f"data: {info}")

    if args.no_class_weight:
        W = torch.ones(NUM_DISEASE, NUM_CLASS)
    else:
        W = class_weights(info["train_class_counts"])
    print("class weights [disease x (abs,pres,unc)]:\n", W.numpy().round(2))

    backbone = load_retfound(freeze=not args.lora, lora=args.lora,
                             lora_rank=args.lora_rank)
    pool = POOLS[args.pool]()
    head = SignsHead(1024, kind=args.head)
    model = SignsModel(backbone, pool, head, lora=args.lora).to(device)

    if args.warmstart:
        wck = torch.load(args.warmstart, map_location=device,
                         weights_only=False)
        model.pool.load_state_dict(wck["pool_state"])
        model.head.load_state_dict(wck["head_state"])
        print(f"warm-started pool+head from {args.warmstart}")

    head_pool = list(model.pool.parameters()) + list(model.head.parameters())
    lora_params = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = [{"params": [p for p in head_pool if p.requires_grad],
               "init_lr": args.lr}]
    if args.lora and lora_params:
        groups.append({"params": lora_params, "init_lr": args.lora_lr})
    opt = torch.optim.AdamW(groups, lr=args.lr,
                            weight_decay=args.weight_decay)
    n_hp = sum(p.numel() for p in head_pool if p.requires_grad)
    n_lr = sum(p.numel() for p in lora_params)
    print(f"pool={args.pool} head={args.head} loss={args.loss} "
          f"lora={args.lora}{f'(r{args.lora_rank})' if args.lora else ''} "
          f"cw={not args.no_class_weight}  pool+head={n_hp/1e6:.3f}M"
          f"{f' + LoRA={n_lr/1e6:.3f}M' if args.lora else ''}")

    def lr_at(ep):
        if ep < args.warmup_epochs:
            return (ep + 1) / max(1, args.warmup_epochs)
        prog = (ep - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * prog))

    best_val = -1.0
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
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = signs_loss(logits, y, W, args.loss == "focal",
                              args.focal_gamma)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(x)
        tr_loss /= max(1, info["n_train"])

        vres, _, _ = evaluate(model, va, device)
        vmf1 = vres["overall"]["mean_macro_f1"]
        hist.append({"epoch": ep, "lr": opt.param_groups[0]["lr"],
                     "train_loss": tr_loss, "val_mean_macro_f1": vmf1})
        print(f"ep {ep:>3} lr={opt.param_groups[0]['lr']:.2e} "
              f"train_loss={tr_loss:.5f} val_mF1={vmf1:.4f}"
              f"{'  *best' if vmf1 > best_val else ''}")

        if vmf1 > best_val:
            best_val, best_ep, bad = vmf1, ep, 0
            torch.save({"epoch": ep, "pool": args.pool, "head": args.head,
                        "pool_state": model.pool.state_dict(),
                        "head_state": model.head.state_dict(),
                        "lora_state": lora_sd(),
                        "val_mean_macro_f1": vmf1}, out_dir / "best.pt")
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
    test_res, P, Y = evaluate(model, te, device)

    summary = {
        "task": "signs", "pool": args.pool, "head": args.head,
        "loss": args.loss, "class_weight": not args.no_class_weight,
        "disease_order": list(SIGN_NAMES),
        "best_epoch": best_ep, "best_val_mean_macro_f1": best_val,
        "data_info": info, "args": vars(args),
        "test_metrics": test_res, "history": hist,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    np.savez(out_dir / "test_preds.npz", pred=P, gt=Y)

    print("\n==== TEST (best ckpt, epoch %d) ====" % best_ep)
    for scope in ("overall", "LAG", "Papila"):
        if scope not in test_res:
            continue
        r = test_res[scope]
        print(f"\n[{scope}] mean_macro_F1 = {r['mean_macro_f1']:.4f}")
        for name, b in r["per_disease"].items():
            if "recall" not in b:
                continue
            rc = b["recall"]
            print(f"  {name:<22s} mF1={b['macro_f1']:.3f} acc={b['acc']:.2f}"
                  f"  recall[abs={rc['absent']:.2f} "
                  f"pres={rc['present']:.2f} unc={rc['uncertain']:.2f}]")
    print("\n  (selection = val mean_macro_F1; acc shown is base-rate-"
          "inflated, do NOT use it to judge)")
    print(f"\nDone. Artifacts in {out_dir}/")


if __name__ == "__main__":
    main()

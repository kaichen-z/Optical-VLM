"""
Train a left/right-eye (laterality) classifier on a frozen RETFound
backbone, via horizontal-flip synthesis on REFUGE 1200.

  image -> RETFound(frozen) -> [B,197,1024]
        -> pooling (--pool)  -> [B,1024]
        -> LateralityHead    -> 2 logits   (0='R' orig, 1='L' flipped)

REFUGE is laterality-normalized (all one orientation, verified), so a
horizontal mirror == the opposite eye. dataset_laterality random-flips
TRAIN (p=0.5) and yields VAL/TEST each image x2 (orig+flipped) -> exactly
balanced -> plain CrossEntropy, no class weights needed.

Purpose: a reusable OD/OS detector to orient ISNT N/T on images that
lack fovea coords (e.g. the 733). NOTE: REFUGE itself doesn't need it
(constant orientation). Whether the 733 need it (mixed vs normalized)
is still open & separate.

Select: best ckpt by max VAL macro-F1 (== accuracy here, balanced).
Mirrors train_cdr_verdict.py; reuses the 5 poolings + frozen backbone.

Usage: python train_laterality.py --pool tier2a
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
from dataset_laterality import build_dataloaders, CLASS_NAMES, NUM_CLASS

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


class LateralityHead(nn.Module):
    """LayerNorm -> (Linear|MLP) -> 2 logits."""

    def __init__(self, in_dim=1024, kind="linear", hidden=512, dropout=0.2):
        super().__init__()
        if kind == "linear":
            self.net = nn.Sequential(nn.LayerNorm(in_dim),
                                     nn.Linear(in_dim, NUM_CLASS))
        elif kind == "mlp":
            self.net = nn.Sequential(nn.LayerNorm(in_dim),
                                     nn.Linear(in_dim, hidden), nn.GELU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden, NUM_CLASS))
        else:
            raise ValueError(kind)

    def forward(self, x):
        return self.net(x)                       # [B,2]


class LateralityModel(nn.Module):
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
        return self.head(pooled)                  # [B,2]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    P, Y = [], []
    for x, y, _i, _d in loader:
        P.append(model(x.to(device)).argmax(-1).cpu())
        Y.append(y)
    P = torch.cat(P).numpy()
    Y = torch.cat(Y).numpy()
    acc = float((P == Y).mean()) if len(Y) else 0.0
    rec, f1s = {}, []
    for c in range(NUM_CLASS):
        tp = int(((P == c) & (Y == c)).sum())
        fp = int(((P == c) & (Y != c)).sum())
        fn = int(((P != c) & (Y == c)).sum())
        r = tp / (tp + fn) if (tp + fn) else float("nan")
        p = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) and not np.isnan(r) else 0.0
        rec[CLASS_NAMES[c]] = r
        f1s.append(f1)
    return {"acc": acc, "macro_f1": float(np.mean(f1s)),
            "recall": rec,
            "confusion": {CLASS_NAMES[t]:
                          {CLASS_NAMES[pp]: int(((Y == t) & (P == pp)).sum())
                           for pp in range(NUM_CLASS)}
                          for t in range(NUM_CLASS)}}, P, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, choices=list(POOLS))
    ap.add_argument("--head", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    ap.add_argument("--warmstart", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = f"lat_{args.pool}_{args.head}_{ts}"
    out_dir = Path(args.out or f"outputs/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out={out_dir}")

    tr, va, te, info = build_dataloaders(
        batch_size=args.batch_size, val_frac=args.val_frac,
        test_frac=args.test_frac, seed=args.seed,
        num_workers=args.num_workers)
    print(f"data: {info}")

    backbone = load_retfound(freeze=not args.lora, lora=args.lora,
                             lora_rank=args.lora_rank)
    pool = POOLS[args.pool]()
    head = LateralityHead(1024, kind=args.head)
    model = LateralityModel(backbone, pool, head, lora=args.lora).to(device)

    if args.warmstart:
        wck = torch.load(args.warmstart, map_location=device,
                         weights_only=False)
        model.pool.load_state_dict(wck["pool_state"])
        model.head.load_state_dict(wck["head_state"])

    head_pool = list(model.pool.parameters()) + list(model.head.parameters())
    lora_params = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = [{"params": [p for p in head_pool if p.requires_grad],
               "init_lr": args.lr}]
    if args.lora and lora_params:
        groups.append({"params": lora_params, "init_lr": args.lora_lr})
    opt = torch.optim.AdamW(groups, lr=args.lr,
                            weight_decay=args.weight_decay)
    n_hp = sum(p.numel() for p in head_pool if p.requires_grad)
    print(f"pool={args.pool} head={args.head} lora={args.lora}  "
          f"pool+head={n_hp/1e6:.3f}M")

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
        nseen = 0
        for x, y, _i, _d in tr:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(x)
            nseen += len(x)
        tr_loss /= max(1, nseen)

        vres, _, _ = evaluate(model, va, device)
        vmf1 = vres["macro_f1"]
        hist.append({"epoch": ep, "lr": opt.param_groups[0]["lr"],
                     "train_loss": tr_loss, "val_macro_f1": vmf1,
                     "val_acc": vres["acc"]})
        print(f"ep {ep:>3} lr={opt.param_groups[0]['lr']:.2e} "
              f"loss={tr_loss:.5f} val_acc={vres['acc']:.4f} "
              f"val_mF1={vmf1:.4f}{'  *best' if vmf1 > best_val else ''}")

        if vmf1 > best_val:
            best_val, best_ep, bad = vmf1, ep, 0
            torch.save({"epoch": ep, "pool": args.pool, "head": args.head,
                        "pool_state": model.pool.state_dict(),
                        "head_state": model.head.state_dict(),
                        "lora_state": lora_sd(),
                        "val_macro_f1": vmf1}, out_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop @ ep {ep}")
                break

    ck = torch.load(out_dir / "best.pt", map_location=device,
                    weights_only=False)
    model.pool.load_state_dict(ck["pool_state"])
    model.head.load_state_dict(ck["head_state"])
    if ck.get("lora_state"):
        model.backbone.load_state_dict(ck["lora_state"], strict=False)
    test_res, P, Y = evaluate(model, te, device)

    summary = {
        "task": "laterality", "pool": args.pool, "head": args.head,
        "best_epoch": best_ep, "best_val_macro_f1": best_val,
        "data_info": info, "args": vars(args),
        "test_metrics": test_res, "history": hist,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    np.savez(out_dir / "test_preds.npz", pred=P, gt=Y)

    print("\n==== TEST (best ckpt ep %d) ====" % best_ep)
    print(f"  acc={test_res['acc']:.4f}  macro_F1={test_res['macro_f1']:.4f}")
    print(f"  recall={ {k: round(v,3) for k,v in test_res['recall'].items()} }")
    print(f"  confusion={test_res['confusion']}")
    print(f"\nDone. Artifacts in {out_dir}/")


if __name__ == "__main__":
    main()

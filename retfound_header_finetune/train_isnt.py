"""
Train an ISNT rim-thickness regression head on a frozen RETFound backbone.

Pipeline:  image -> RETFound(frozen) -> [B,197,1024]
                  -> pooling (--pool) -> [B,1024]
                  -> head (--head)    -> sigmoid -> [B,4]
                     (Inferior, Superior, Nasal, Temporal  — FIXED order)

Loss   : --loss smoothl1 (Huber, --beta default 0.04, tuned to the ISNT
         target scale: rim std ~0.09, ~half of CDR) OR --loss l1 (pure MAE).
         Equal weight on the 4 quadrants.
Select : best checkpoint by VAL overall MAE (val carved from the 613 train,
         stratified by mean rim thickness).
Report : final metrics on the 120 balanced TEST set ONLY at the end —
         per-quadrant + overall MAE/RMSE/Pearson, broken down LAG vs Papila,
         plus a 4-panel pred-vs-GT scatter PNG.

Reuses the SAME 5 poolings as the CDR ablation (baseline/tier1/2a/2b/3) so
`for p in ...; do python train_isnt.py --pool $p; done` is the whole sweep.
Fully separate from train.py (CDR) — neither touches the other.

Usage (on the GPU server, after push.sh):
    python train_isnt.py --pool tier2a
    python train_isnt.py --pool tier3 --loss l1
    python train_isnt.py --pool tier3 --loss smoothl1 --beta 0.04
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from retfound_backbone import load_retfound
from heads import build_head
from dataset_isnt import build_dataloaders, RIM_NAMES

# pooling registry (identical to the CDR ablation)
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

N_OUT = 4   # I, S, N, T


class ISNTModel(nn.Module):
    def __init__(self, backbone, pool, head):
        super().__init__()
        self.backbone = backbone        # frozen
        self.pool = pool
        self.head = head

    def forward(self, x):
        with torch.no_grad():
            tokens = self.backbone.forward_tokens(x)      # [B,197,1024]
        pooled, attn = self.pool(tokens)                  # [B,1024], [B,197]
        return self.head(pooled), attn                    # [B,4]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gts, dsets = [], [], []
    for x, y, _ids, ds in loader:
        x = x.to(device)
        p, _ = model(x)
        preds.append(p.cpu()); gts.append(y); dsets.extend(ds)
    P = torch.cat(preds).numpy()        # [N,4]
    G = torch.cat(gts).numpy()          # [N,4]
    return _metrics(P, G, dsets)


def _metrics(P, G, dsets):
    def block(p, g):
        if len(p) == 0:
            return {}
        err = np.abs(p - g)
        out = {
            "n": int(len(p)),
            "mae": float(err.mean()),
            "rmse": float(np.sqrt(((p - g) ** 2).mean())),
        }
        for j, name in enumerate(RIM_NAMES):           # I,S,N,T
            out[f"mae_{name[0]}"] = float(err[:, j].mean())
            if p[:, j].std() > 1e-6 and g[:, j].std() > 1e-6:
                out[f"r_{name[0]}"] = float(np.corrcoef(p[:, j], g[:, j])[0, 1])
            else:
                out[f"r_{name[0]}"] = float("nan")
        return out

    res = {"overall": block(P, G)}
    dsets = np.array(dsets)
    for d in ("LAG", "Papila"):
        m = dsets == d
        if m.any():
            res[d] = block(P[m], G[m])
    return res, P, G


def _fmt(v):
    return (f"  [{{k}}] n={v['n']} MAE={v['mae']:.4f} RMSE={v['rmse']:.4f}  "
            f"I({v['mae_I']:.4f},r{v['r_I']:.2f}) "
            f"S({v['mae_S']:.4f},r{v['r_S']:.2f}) "
            f"N({v['mae_N']:.4f},r{v['r_N']:.2f}) "
            f"T({v['mae_T']:.4f},r{v['r_T']:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, choices=list(POOLS))
    ap.add_argument("--head", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--loss", default="smoothl1", choices=["smoothl1", "l1"],
                    help="smoothl1 (Huber, see --beta) or l1 (pure MAE)")
    ap.add_argument("--beta", type=float, default=0.04,
                    help="SmoothL1 beta — tuned to ISNT scale (rim std ~0.09, "
                         "~half of CDR's 0.075). Ignored when --loss l1.")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=10,
                    help="early-stop patience on val MAE")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default=None,
                    help="output dir; default outputs/isnt_<pool>_<head>_<loss>_<ts>")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = f"isnt_{args.pool}_{args.head}_{args.loss}_{ts}"
    out_dir = Path(args.out or f"outputs/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out={out_dir}")

    tr, va, te, info = build_dataloaders(
        batch_size=args.batch_size, val_frac=args.val_frac,
        seed=args.seed, num_workers=args.num_workers)
    print(f"data: {info}")

    backbone = load_retfound(freeze=True)
    pool = POOLS[args.pool]()
    head = build_head(args.head, out_dim=N_OUT)            # 1024 -> 4
    model = ISNTModel(backbone, pool, head).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"pool={args.pool} head={args.head} loss={args.loss}"
          f"{'' if args.loss=='l1' else f' beta={args.beta}'}  "
          f"trainable params={n_params/1e6:.3f}M")

    opt = torch.optim.AdamW(trainable, lr=args.lr,
                            weight_decay=args.weight_decay)

    def lr_at(ep):
        if ep < args.warmup_epochs:
            return (ep + 1) / max(1, args.warmup_epochs)
        prog = (ep - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * prog))

    if args.loss == "l1":
        crit = nn.L1Loss()
    else:
        crit = nn.SmoothL1Loss(beta=args.beta)

    best_val = float("inf")
    best_ep = -1
    bad = 0
    hist = []

    for ep in range(args.epochs):
        for g in opt.param_groups:
            g["lr"] = args.lr * lr_at(ep)

        model.train()
        model.backbone.eval()
        tr_loss = 0.0
        for x, y, _i, _d in tr:
            x, y = x.to(device), y.to(device)
            pred, _ = model(x)
            loss = crit(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(x)
        tr_loss /= max(1, info["n_train"])

        vres, _, _ = evaluate(model, va, device)
        val_mae = vres["overall"]["mae"]
        hist.append({"epoch": ep, "lr": opt.param_groups[0]["lr"],
                     "train_loss": tr_loss, "val_mae": val_mae})
        print(f"ep {ep:>3} lr={opt.param_groups[0]['lr']:.2e} "
              f"train_loss={tr_loss:.5f} val_mae={val_mae:.5f}"
              f"{'  *best' if val_mae < best_val else ''}")

        if val_mae < best_val:
            best_val, best_ep, bad = val_mae, ep, 0
            torch.save({"epoch": ep, "pool": args.pool, "head": args.head,
                        "loss": args.loss, "beta": args.beta,
                        "pool_state": model.pool.state_dict(),
                        "head_state": model.head.state_dict(),
                        "val_mae": val_mae},
                       out_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop @ ep {ep} (no val improve {args.patience} eps)")
                break

    ck = torch.load(out_dir / "best.pt", map_location=device,
                    weights_only=False)
    model.pool.load_state_dict(ck["pool_state"])
    model.head.load_state_dict(ck["head_state"])
    test_res, P, G = evaluate(model, te, device)

    summary = {
        "task": "isnt", "pool": args.pool, "head": args.head,
        "loss": args.loss, "beta": (None if args.loss == "l1" else args.beta),
        "target_order": list(RIM_NAMES),
        "best_epoch": best_ep, "best_val_mae": best_val,
        "data_info": info, "args": vars(args),
        "test_metrics": test_res, "history": hist,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    np.savez(out_dir / "test_preds.npz", pred=P, gt=G)

    print("\n==== TEST (best ckpt, epoch %d) ====" % best_ep)
    for k, v in test_res.items():
        if v:
            print(_fmt(v).format(k=k))

    # 4-panel scatter (pred vs gt), best-effort
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        for j, name in enumerate(RIM_NAMES):
            ax[j].scatter(G[:, j], P[:, j], s=14, alpha=0.6)
            ax[j].plot([0, 1], [0, 1], "r--", lw=1)
            ax[j].set_xlabel("GT"); ax[j].set_ylabel("pred")
            ax[j].set_title(f"{name}  "
                            f"(MAE={np.abs(P[:, j]-G[:, j]).mean():.4f})")
            ax[j].set_xlim(0, 1); ax[j].set_ylim(0, 1)
        fig.suptitle(f"ISNT  {args.pool}+{args.head}+{args.loss}  "
                     f"(test n={len(P)})")
        fig.tight_layout()
        fig.savefig(out_dir / "scatter.png", dpi=120)
        print(f"  scatter -> {out_dir/'scatter.png'}")
    except Exception as e:
        print(f"  (scatter skipped: {e})")

    print(f"\nDone. Artifacts in {out_dir}/")


if __name__ == "__main__":
    main()

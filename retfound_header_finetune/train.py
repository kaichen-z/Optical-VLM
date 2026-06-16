"""
Train a CDR regression head on a frozen RETFound backbone.

Pipeline:  image -> RETFound(frozen) -> [B,197,1024]
                  -> pooling (--pool) -> [B,1024]
                  -> head (--head)    -> sigmoid -> [B,2]  (cdr_v, cdr_h)

Loss   : SmoothL1 (Huber), beta tuned to CDR scale (default 0.075),
         equal weight on cdr_v and cdr_h.
Select : best checkpoint by VAL MAE (carved from the 613 train, stratified).
Report : final metrics on the 120 balanced TEST set ONLY at the end —
         MAE / RMSE / Pearson per-CDR and overall, broken down by
         LAG vs Papila, plus a pred-vs-GT scatter PNG.

The 5 poolings (baseline/tier1/tier2a/tier2b/tier3) are a CLI flag so the
whole ablation is `for p in ...; do python train.py --pool $p; done`.

Usage (on the GPU server, after push.sh):
    python train.py --pool baseline
    python train.py --pool tier2b --head mlp --epochs 80
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
from dataset import build_dataloaders

# pooling registry
from baseline import BaselinePool
from tier1 import Tier1Pool
from tier2a import Tier2aPool
from tier2b import Tier2bPool
from tier3 import Tier3Pool

POOLS = {
    "baseline": lambda: BaselinePool(1024, mode="mean"),   # == Grace --global_pool
    "tier1":    lambda: Tier1Pool(1024),
    "tier2a":   lambda: Tier2aPool(1024, hidden=256, gated=False),
    "tier2b":   lambda: Tier2bPool(1024),
    "tier3":    lambda: Tier3Pool(1024, num_heads=8),
}


class CDRModel(nn.Module):
    def __init__(self, backbone, pool, head, lora=False):
        super().__init__()
        self.backbone = backbone        # frozen (or LoRA-adapted)
        self.pool = pool
        self.head = head
        self.lora = lora                # if True, grads must flow to LoRA

    def forward(self, x):
        if self.lora:
            tokens = self.backbone.forward_tokens(x)      # grad -> LoRA adapters
        else:
            with torch.no_grad():       # backbone fully frozen — no grad
                tokens = self.backbone.forward_tokens(x)
        pooled, attn = self.pool(tokens)                  # [B,1024], [B,197]
        return self.head(pooled), attn                    # [B,2]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gts, dsets = [], [], []
    for x, y, _ids, ds in loader:
        x = x.to(device)
        p, _ = model(x)
        preds.append(p.cpu()); gts.append(y); dsets.extend(ds)
    P = torch.cat(preds).numpy()        # [N,2]
    G = torch.cat(gts).numpy()          # [N,2]
    return _metrics(P, G, dsets)


def _metrics(P, G, dsets):
    def block(p, g):
        if len(p) == 0:
            return {}
        err = np.abs(p - g)
        out = {
            "n": int(len(p)),
            "mae_v": float(err[:, 0].mean()),
            "mae_h": float(err[:, 1].mean()),
            "mae": float(err.mean()),
            "rmse": float(np.sqrt(((p - g) ** 2).mean())),
        }
        for j, name in ((0, "pearson_v"), (1, "pearson_h")):
            if p[:, j].std() > 1e-6 and g[:, j].std() > 1e-6:
                out[name] = float(np.corrcoef(p[:, j], g[:, j])[0, 1])
            else:
                out[name] = float("nan")
        return out

    res = {"overall": block(P, G)}
    dsets = np.array(dsets)
    for d in ("LAG", "Papila"):
        m = dsets == d
        if m.any():
            res[d] = block(P[m], G[m])
    return res, P, G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, choices=list(POOLS))
    ap.add_argument("--head", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--beta", type=float, default=0.075,
                    help="SmoothL1 beta — MUST be ~CDR error scale (~0.05-0.1), "
                         "NOT torch default 1.0 (which degenerates to MSE here).")
    ap.add_argument("--patience", type=int, default=10,
                    help="early-stop patience on val MAE")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="run ALL --epochs (disable val-MAE early stop). "
                         "Probe for 'does a lower train-loss ckpt actually "
                         "beat the val-picked ckpt on the 120 test?' — also "
                         "saves best_trainloss.pt and tests BOTH at the end.")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default=None,
                    help="output dir; default outputs/<pool>_<head>_<ts>")
    # --- LoRA fine-tuning of the RETFound backbone ---
    ap.add_argument("--lora", action="store_true",
                    help="inject LoRA into the ViT (base frozen) and train it "
                         "JOINTLY with pool+head. Default off = frozen backbone.")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-lr", type=float, default=1e-4,
                    help="LR for the LoRA adapters (discriminative; pool+head "
                         "use --lr).")
    ap.add_argument("--warmstart", default=None,
                    help="path to a frozen-backbone best.pt to init pool+head "
                         "from (recommended with --lora: start from the known-"
                         "good frozen solution, LoRA init≈identity).")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out or f"outputs/{args.pool}_{args.head}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out={out_dir}")

    # --- data ---
    tr, va, te, info = build_dataloaders(
        batch_size=args.batch_size, val_frac=args.val_frac,
        seed=args.seed, num_workers=args.num_workers)
    print(f"data: {info}")

    # --- model ---
    backbone = load_retfound(freeze=not args.lora, lora=args.lora,
                             lora_rank=args.lora_rank)
    pool = POOLS[args.pool]()
    head = build_head(args.head)
    model = CDRModel(backbone, pool, head, lora=args.lora).to(device)

    # warm-start pool+head from a frozen-backbone best.pt (recommended w/ LoRA)
    if args.warmstart:
        ck = torch.load(args.warmstart, map_location=device,
                        weights_only=False)
        model.pool.load_state_dict(ck["pool_state"])
        model.head.load_state_dict(ck["head_state"])
        print(f"warm-started pool+head from {args.warmstart} "
              f"(ep{ck.get('epoch')}, val_mae={ck.get('val_mae')})")

    # --- optim: discriminative LR (LoRA adapters vs pool+head) ---
    head_pool = list(model.pool.parameters()) + list(model.head.parameters())
    hp_ids = {id(p) for p in head_pool}
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
          f"{f'(r{args.lora_rank})' if args.lora else ''}  "
          f"trainable: pool+head={n_hp/1e6:.3f}M"
          f"{f' + LoRA={n_lr/1e6:.3f}M' if args.lora else ''}")

    def lr_at(ep):
        if ep < args.warmup_epochs:
            return (ep + 1) / max(1, args.warmup_epochs)
        prog = (ep - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * prog))            # cosine to 0

    def lora_sd():
        """LoRA adapter weights (trainable backbone params) for ckpt; {} if
        not --lora so frozen-run checkpoints stay byte-identical."""
        if not args.lora:
            return {}
        return {n: p.detach().cpu() for n, p in
                model.backbone.named_parameters() if p.requires_grad}

    crit = nn.SmoothL1Loss(beta=args.beta)

    best_val = float("inf")
    best_ep = -1
    best_tr_loss = float("inf")          # for the min-train-loss checkpoint
    best_tr_ep = -1
    bad = 0
    hist = []

    for ep in range(args.epochs):
        sched = lr_at(ep)
        for g in opt.param_groups:
            g["lr"] = g["init_lr"] * sched     # per-group discriminative LR

        model.train()
        if not args.lora:
            model.backbone.eval()         # fully-frozen backbone stays eval
        # with --lora: keep backbone in train() so LoRA dropout is active
        # (base weights are frozen regardless via requires_grad=False)
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

        # checkpoint A: lowest VAL MAE (the principled selection)
        if val_mae < best_val:
            best_val, best_ep, bad = val_mae, ep, 0
            torch.save({"epoch": ep, "pool": args.pool, "head": args.head,
                        "pool_state": model.pool.state_dict(),
                        "head_state": model.head.state_dict(),
                        "lora_state": lora_sd(),
                        "val_mae": val_mae},
                       out_dir / "best.pt")
        else:
            bad += 1

        # checkpoint B: lowest TRAIN loss — ONLY in the --no-early-stop probe
        # (tests whether a more-trained / lower-train-loss model generalises
        # better, i.e. whether the tiny n_val=62 early-stop is just noise).
        # Gated so normal ablation runs are byte-identical to before.
        if args.no_early_stop and tr_loss < best_tr_loss:
            best_tr_loss, best_tr_ep = tr_loss, ep
            torch.save({"epoch": ep, "pool": args.pool, "head": args.head,
                        "pool_state": model.pool.state_dict(),
                        "head_state": model.head.state_dict(),
                        "lora_state": lora_sd(),
                        "train_loss": tr_loss},
                       out_dir / "best_trainloss.pt")

        if not args.no_early_stop and bad >= args.patience:
            print(f"early stop @ ep {ep} (no val improve {args.patience} eps)")
            break

    # --- final test ---
    # Always: the val-picked checkpoint (the principled selection).
    # Additionally (only with --no-early-stop): the min-train-loss checkpoint,
    # so we can DIRECTLY compare on the 120 test whether pushing train-loss
    # lower actually helps or is just overfitting the 551 train samples.
    def _eval_ckpt(path, tag, ep_of):
        ck = torch.load(path, map_location=device, weights_only=False)
        model.pool.load_state_dict(ck["pool_state"])
        model.head.load_state_dict(ck["head_state"])
        if ck.get("lora_state"):          # restore best-epoch LoRA adapters
            model.backbone.load_state_dict(ck["lora_state"], strict=False)
        res, P, G = evaluate(model, te, device)
        print(f"\n==== TEST [{tag}] (ckpt epoch {ep_of}) ====")
        for k, v in res.items():
            if v:
                print(f"  [{k}] n={v['n']} MAE={v['mae']:.4f} "
                      f"(v={v['mae_v']:.4f} h={v['mae_h']:.4f}) "
                      f"RMSE={v['rmse']:.4f} "
                      f"r_v={v.get('pearson_v', float('nan')):.3f} "
                      f"r_h={v.get('pearson_h', float('nan')):.3f}")
        np.savez(out_dir / f"test_preds_{tag}.npz", pred=P, gt=G)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(11, 5))
            for j, name in ((0, "vertical CDR"), (1, "horizontal CDR")):
                ax[j].scatter(G[:, j], P[:, j], s=14, alpha=0.6)
                ax[j].plot([0, 1], [0, 1], "r--", lw=1)
                ax[j].set_xlabel("GT"); ax[j].set_ylabel("pred")
                ax[j].set_title(f"{name}  (MAE={np.abs(P[:, j]-G[:, j]).mean():.4f})")
                ax[j].set_xlim(0, 1); ax[j].set_ylim(0, 1)
            fig.suptitle(f"{args.pool}+{args.head} [{tag}] (test n={len(P)})")
            fig.tight_layout()
            fig.savefig(out_dir / f"scatter_{tag}.png", dpi=120)
            print(f"  scatter -> {out_dir/f'scatter_{tag}.png'}")
        except Exception as e:
            print(f"  (scatter skipped: {e})")
        return res

    test_val = _eval_ckpt(out_dir / "best.pt", "val_best", best_ep)

    summary = {
        "pool": args.pool, "head": args.head,
        "best_epoch": best_ep, "best_val_mae": best_val,
        "no_early_stop": args.no_early_stop,
        "data_info": info,
        "args": vars(args),
        "test_metrics": test_val,           # the principled (val-picked) result
        "history": hist,
    }

    if args.no_early_stop:
        test_tr = _eval_ckpt(out_dir / "best_trainloss.pt",
                             "min_trainloss", best_tr_ep)
        summary["best_trainloss_epoch"] = best_tr_ep
        summary["best_train_loss"] = best_tr_loss
        summary["test_metrics_min_trainloss"] = test_tr
        dv = test_val["overall"]["mae"]
        dt = test_tr["overall"]["mae"]
        print(f"\n---- probe verdict ----")
        print(f"  val-picked ckpt (ep {best_ep}):       test MAE = {dv:.4f}")
        print(f"  min-train-loss ckpt (ep {best_tr_ep}): test MAE = {dt:.4f}")
        if dt < dv - 0.003:
            print("  => lower train-loss ckpt WINS: early-stop on n=62 val "
                  "was leaving performance on the table (val too noisy).")
        elif dt > dv + 0.003:
            print("  => lower train-loss ckpt LOSES: it's overfitting the 551 "
                  "train samples; val early-stop was right.")
        else:
            print("  => within noise (|Δ|≤0.003 on n=120): can't distinguish; "
                  "selection is in the noise floor — argues for k-fold CV.")

    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Artifacts in {out_dir}/")


if __name__ == "__main__":
    main()

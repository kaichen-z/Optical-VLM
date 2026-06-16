#!/usr/bin/env python3
"""Load a full-613 quad-head best.pt and dump TEST(120) predictions (all 4 heads)
WITH ids. Used to build the 120-test indicators for the quad-mixed LoRA pipeline.

  python predict_quad.py --ckpt outputs/quad_tier2a_linear/best.pt --out outputs/quad_tier2a_linear/test_preds_ids.npz
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from retfound_backbone import load_retfound
from dataset_quad import (build_dataloaders, N_Q, N_ABN_CLASSES, N_CDR,
                          N_CDR_CLASSES, N_SIGN, N_SIGN_CLASSES)
from train_quad_head import OrderHead, MultiClsHead, QuadModel, POOLS


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    bb = load_retfound(freeze=True, lora=False).to(dev).eval()
    pool = POOLS[ck["pool"]]()
    oh = OrderHead(1024, kind=ck["head"])
    ah = MultiClsHead(1024, N_Q, N_ABN_CLASSES, kind=ck["head"])
    chh = MultiClsHead(1024, N_CDR, N_CDR_CLASSES, kind=ck["head"])
    sh = MultiClsHead(1024, N_SIGN, N_SIGN_CLASSES, kind=ck["head"])
    m = QuadModel(bb, pool, oh, ah, chh, sh).to(dev).eval()
    m.pool.load_state_dict(ck["pool_state"])
    m.order_head.load_state_dict(ck["order_head_state"])
    m.abn_head.load_state_dict(ck["abn_head_state"])
    m.cdr_head.load_state_dict(ck["cdr_head_state"])
    m.signs_head.load_state_dict(ck["signs_head_state"])

    _, _, te, _ = build_dataloaders(batch_size=32, num_workers=4)
    S, AP, CP, GP, IDS = [], [], [], [], []
    for x, _rim, _abn, _cdr, _sign, ids, _ds in te:
        o_s, a_l, c_l, g_l = m(x.to(dev))
        S.append(o_s.cpu().numpy())
        AP.append(a_l.argmax(-1).cpu().numpy())
        CP.append(c_l.argmax(-1).cpu().numpy())
        GP.append(g_l.argmax(-1).cpu().numpy())
        IDS += list(ids)
    np.savez(args.out, scores=np.concatenate(S), abn_pred=np.concatenate(AP),
             cdr_pred=np.concatenate(CP), sign_pred=np.concatenate(GP),
             ids=np.array(IDS))
    print(f"wrote {args.out}  ({len(IDS)} test ids, pool={ck['pool']} head={ck['head']})")


if __name__ == "__main__":
    main()

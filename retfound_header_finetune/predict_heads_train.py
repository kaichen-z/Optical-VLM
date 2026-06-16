#!/usr/bin/env python3
"""
Plan B-train (naive / in-sample): produce RETFound-PREDICTED indicators for
the 613 TRAIN ids, using the SAME 3 best heads' best.pt that produced the
120-test predicted_indicators.json.

WARNING (methodological, stated honestly): the 613 train ids were USED to
train these heads, so these predictions are IN-SAMPLE -> optimistically
clean (less noisy than the held-out 120-test predictions). This is the
"B-naive" quick look the user asked for; the rigorous fix is K-fold OOF.

Output keys / value format are byte-identical to build_predicted_indicators
(reuses its RIM_LETTERS / SIGN_KEYS / SIGN_OUT / IDX2LBL), so the rebuilt
SFT train prompts match the predicted 120-test prompts exactly.

Writes: qwen35_icl_glaucoma/DATA/predicted_indicators_train613.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import build_predicted_indicators as B          # reuse exact conversion maps
from config import PHOTOS_DIR, CDR_MANIFEST, ISNT_MANIFEST, SIGNS_MANIFEST
from retfound_backbone import load_retfound

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "qwen35_icl_glaucoma" / "DATA" / "predicted_indicators_train613.json"

CDR_CK = HERE / "outputs/tier2a_linear_20260516_012723/best.pt"
ORD_CK = HERE / "outputs/order_tier2b_linear_20260516_044017/best.pt"
SIG_CK = HERE / "outputs/signs_tier1_linear_wce_20260516_054817/best.pt"


def _train_loader(Dataset, rows, tf, bs=32):
    tr = [r for r in rows if r["split"] == "train"]
    ds = Dataset(tr, PHOTOS_DIR, tf)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4,
                    pin_memory=True, drop_last=False)
    return tr, dl


@torch.no_grad()
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = load_retfound(freeze=True).to(dev).eval()   # shared frozen ViT
    pred: dict[str, dict] = {}

    # ---------- CDR (tier2a) ----------
    import dataset as DS
    from train import CDRModel, POOLS as CPOOLS
    from heads import build_head
    ck = torch.load(CDR_CK, map_location=dev, weights_only=False)
    pool = CPOOLS[ck["pool"]](); head = build_head(ck["head"])
    pool.load_state_dict(ck["pool_state"]); head.load_state_dict(ck["head_state"])
    m = CDRModel(backbone, pool, head).to(dev).eval()
    rows = DS._load_manifest(Path(CDR_MANIFEST))
    tr, dl = _train_loader(DS.CDRDataset, rows, DS._eval_transform())
    n = 0
    for x, _, ids, _ in dl:
        p, _ = m(x.to(dev))                                # [B,2] sigmoid
        p = p.cpu().numpy()
        for k, sid in enumerate(ids):
            pred.setdefault(sid, {})
            pred[sid]["vertical_CDR"] = round(float(p[k, 0]), 3)
            pred[sid]["horizontal_CDR"] = round(float(p[k, 1]), 3)
            n += 1
    print(f"CDR   train: {n} ids (ckpt ep{ck.get('epoch')} "
          f"val_mae={ck.get('val_mae')})")

    # ---------- ISNT order (order_tier2b) ----------
    import dataset_isnt as DI
    from train_order import OrderModel, OrderHead, POOLS as OPOOLS
    ck = torch.load(ORD_CK, map_location=dev, weights_only=False)
    pool = OPOOLS[ck["pool"]](); head = OrderHead(1024, kind=ck["head"])
    pool.load_state_dict(ck["pool_state"]); head.load_state_dict(ck["head_state"])
    m = OrderModel(backbone, pool, head, aux=None).to(dev).eval()
    rows = DI._load_manifest(Path(ISNT_MANIFEST))
    tr, dl = _train_loader(DI.ISNTDataset, rows, DI._eval_transform())
    n = 0
    for x, _, ids, _ in dl:
        s, _ = m(x.to(dev))                                # [B,4] scores I,S,N,T
        s = s.cpu().numpy()
        for k, sid in enumerate(ids):
            order = np.argsort(-s[k])                       # desc
            pred.setdefault(sid, {})
            pred[sid]["ISNT_observed_order"] = ">".join(
                B.RIM_LETTERS[j] for j in order)
            n += 1
    print(f"ORDER train: {n} ids (ckpt ep{ck.get('epoch')} "
          f"val_kendall={ck.get('val_kendall')})")

    # ---------- signs (signs_tier1) ----------
    import dataset_signs as DG
    from train_signs import SignsModel, SignsHead, POOLS as SPOOLS
    ck = torch.load(SIG_CK, map_location=dev, weights_only=False)
    pool = SPOOLS[ck["pool"]](); head = SignsHead(1024, kind=ck["head"])
    pool.load_state_dict(ck["pool_state"]); head.load_state_dict(ck["head_state"])
    m = SignsModel(backbone, pool, head).to(dev).eval()
    rows = DG._load_manifest(Path(SIGNS_MANIFEST))           # NO status filter
    tr, dl = _train_loader(DG.SignsDataset, rows, DG._eval_transform())
    n = 0
    for x, _, ids, _ in dl:
        c = m(x.to(dev)).argmax(-1).cpu().numpy()           # [B,5] classes
        for k, sid in enumerate(ids):
            pred.setdefault(sid, {})
            for j, key in enumerate(B.SIGN_KEYS):
                pred[sid][B.SIGN_OUT[key]] = B.IDX2LBL.get(int(c[k, j]),
                                                           "absent")
            pred[sid]["Disc_hemorrhage"] = "absent"          # decision C
            n += 1
    print(f"SIGNS train: {n} ids")

    OUT.write_text(json.dumps(pred, indent=1, ensure_ascii=False))
    have_cdr = sum("vertical_CDR" in v for v in pred.values())
    have_isnt = sum("ISNT_observed_order" in v for v in pred.values())
    have_sig = sum("Notching" in v for v in pred.values())
    print(f"\nWrote {OUT}  ({len(pred)} train ids)")
    print(f"  per-field coverage: CDR={have_cdr} ISNT={have_isnt} "
          f"SIGNS={have_sig}")
    sid0 = next(iter(pred))
    print(f"  sample {sid0}: {json.dumps(pred[sid0], ensure_ascii=False)}")


if __name__ == "__main__":
    main()

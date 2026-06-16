#!/usr/bin/env python3
"""
IN-SAMPLE (NON-OOF) predicted indicators for the 613 TRAIN ids, using the
SAME head configs as the OOF/v2_cw pipeline but each head trained on ALL 613
and then predicting those same 613 (-> optimistically clean / in-sample).

This is the control for "does OOF matter?": output format is BYTE-IDENTICAL
to predicted_indicators_train_oof.json (same keys/value formats), so the ONLY
difference downstream is OOF vs in-sample indicator noise.

Heads (full-613 best.pt):
  CDR   : outputs/cdr_tier2a_full613       (tier2a linear, numeric V/H, sigmoid)
  RIM   : outputs/orderabn_tier3_w1        (tier3 linear, order scores + per-quad abn)
  SIGNS : outputs/signs_tier1_full613      (tier1 linear, 5 signs 3-cls)

Writes: ../qwen35_icl_glaucoma/DATA/predicted_indicators_train_insample.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import build_predicted_indicators as B          # reuse SIGN_KEYS/SIGN_OUT/IDX2LBL
from config import PHOTOS_DIR, CDR_MANIFEST, ISNT_MANIFEST, SIGNS_MANIFEST
from retfound_backbone import load_retfound

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "qwen35_icl_glaucoma" / "DATA" / "predicted_indicators_train_insample.json"

CDR_CK = HERE / "outputs/cdr_tier2a_full613/best.pt"
RIM_CK = HERE / "outputs/orderabn_tier3_w1/best.pt"
SIG_CK = HERE / "outputs/signs_tier1_full613/best.pt"

# rim formatting — match assemble_oof.py add_rim EXACTLY
RIM_LETTERS = ["I", "S", "N", "T"]
RIM_NAMES = ["Inferior", "Superior", "Nasal", "Temporal"]
ABN_LBL = {0: "normal", 1: "mild thinning", 2: "severe thinning"}


def _train_loader(Dataset, rows, tf, bs=32, **kw):
    tr = [r for r in rows if r["split"] == "train"]
    ds = Dataset(tr, PHOTOS_DIR, tf, **kw)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4,
                    pin_memory=True, drop_last=False)
    return tr, dl


@torch.no_grad()
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = load_retfound(freeze=True).to(dev).eval()
    pred: dict[str, dict] = {}

    # ---------- CDR (tier2a, numeric) ----------
    import dataset as DS
    from train import CDRModel, POOLS as CPOOLS
    from heads import build_head
    ck = torch.load(CDR_CK, map_location=dev, weights_only=False)
    pool = CPOOLS[ck["pool"]](); head = build_head(ck["head"])
    pool.load_state_dict(ck["pool_state"]); head.load_state_dict(ck["head_state"])
    m = CDRModel(backbone, pool, head).to(dev).eval()
    rows = DS._load_manifest(Path(CDR_MANIFEST))
    _, dl = _train_loader(DS.CDRDataset, rows, DS._eval_transform())
    n = 0
    for x, _, ids, _ in dl:
        p, _ = m(x.to(dev))                                 # [B,2] sigmoid
        p = p.cpu().numpy()
        for k, sid in enumerate(ids):
            pred.setdefault(sid, {})
            pred[sid]["vertical_CDR"] = round(float(p[k, 0]), 3)
            pred[sid]["horizontal_CDR"] = round(float(p[k, 1]), 3)
            n += 1
    print(f"CDR   train: {n} ids (ckpt ep{ck.get('epoch')} val_mae={ck.get('val_mae')})")

    # ---------- RIM order + per-quadrant abn (orderabn tier3) ----------
    import dataset_order_abn as DOA
    from train_order_abn import OrderHead, AbnHead, JointModel, POOLS as RPOOLS
    ck = torch.load(RIM_CK, map_location=dev, weights_only=False)
    pool = RPOOLS[ck["pool"]]()
    oh = OrderHead(1024, kind=ck["head"]); ah = AbnHead(1024, kind=ck["head"])
    pool.load_state_dict(ck["pool_state"])
    oh.load_state_dict(ck["order_head_state"]); ah.load_state_dict(ck["abn_head_state"])
    m = JointModel(backbone, pool, oh, ah, lora=False, lr_inject=False).to(dev).eval()
    rows = DOA._load_manifest(Path(ISNT_MANIFEST))
    _, dl = _train_loader(DOA.OrderAbnDataset, rows, DOA._eval_transform())
    n = 0
    for x, _rim, _abn, _lr, ids, _ds in dl:
        s, al = m(x.to(dev))                                # s[B,4] scores, al[B,4,3]
        s = s.cpu().numpy(); ap = al.argmax(-1).cpu().numpy()
        for k, sid in enumerate(ids):
            order = np.argsort(-s[k])                       # desc by score
            st = {RIM_NAMES[j]: ABN_LBL[int(ap[k, j])] for j in range(4)}
            pred.setdefault(sid, {})
            pred[sid]["ISNT_observed_order"] = ">".join(RIM_LETTERS[j] for j in order)
            pred[sid]["ISNT_quadrant_status"] = st
            pred[sid]["ISNT_quadrant_status_str"] = "; ".join(
                f"{q}: {st[q]}" for q in RIM_NAMES)
            n += 1
    print(f"RIM   train: {n} ids (ckpt ep{ck.get('epoch')} "
          f"kendall={ck.get('val_kendall')} abnF1={ck.get('val_abn_macro_f1')})")

    # ---------- signs (tier1) ----------
    import dataset_signs as DG
    from train_signs import SignsModel, SignsHead, POOLS as SPOOLS
    ck = torch.load(SIG_CK, map_location=dev, weights_only=False)
    pool = SPOOLS[ck["pool"]](); head = SignsHead(1024, kind=ck["head"])
    pool.load_state_dict(ck["pool_state"]); head.load_state_dict(ck["head_state"])
    m = SignsModel(backbone, pool, head).to(dev).eval()
    rows = DG._load_manifest(Path(SIGNS_MANIFEST))
    _, dl = _train_loader(DG.SignsDataset, rows, DG._eval_transform())
    n = 0
    for x, _, ids, _ in dl:
        c = m(x.to(dev)).argmax(-1).cpu().numpy()           # [B,5] classes
        for k, sid in enumerate(ids):
            pred.setdefault(sid, {})
            for j, key in enumerate(B.SIGN_KEYS):
                pred[sid][B.SIGN_OUT[key]] = B.IDX2LBL.get(int(c[k, j]), "absent")
            pred[sid]["Disc_hemorrhage"] = "absent"
            n += 1
    print(f"SIGNS train: {n} ids")

    OUT.write_text(json.dumps(pred, indent=1, ensure_ascii=False))
    cov = (sum("vertical_CDR" in v for v in pred.values()),
           sum("ISNT_observed_order" in v for v in pred.values()),
           sum("Notching" in v for v in pred.values()))
    print(f"\nWrote {OUT}  ({len(pred)} ids)  coverage CDR/ISNT/SIGNS={cov}")
    sid0 = next(iter(pred))
    print(f"sample {sid0}: {json.dumps(pred[sid0], ensure_ascii=False)}")


if __name__ == "__main__":
    main()

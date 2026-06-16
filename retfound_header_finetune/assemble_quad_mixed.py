"""
Assemble MIXED-pooling quad-head predicted indicators (Step2 = CDR LEVEL/verdict).
Per-indicator best quad config (from 10-combo scan):
  order + CDR-verdict <- tier2a-linear   (Kendall 1.592 / cdr F1 0.692)
  rim per-quad abn    <- tier2b-linear   (abn F1 0.594)
  signs               <- tier2a-mlp      (sign F1 0.570)

TRAIN 613: 5-fold OOF (outputs/oof_quad_{t2aL,t2bL,t2aM}_f{0..4}/test_preds.npz, each w/ ids)
TEST  120: full-613 heads (outputs/quad_{...}/test_preds_ids.npz from predict_quad.py)

Output (qualitative, byte-compatible with predicted_indicators_*_cdrq.json):
  ../qwen35_icl_glaucoma/DATA/predicted_indicators_train_quadmix.json
  ../qwen35_icl_glaucoma/DATA/predicted_indicators_test_quadmix.json
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "qwen35_icl_glaucoma" / "DATA"
K = 5
RIM_LETTERS = ["I", "S", "N", "T"]
RIM_NAMES = ["Inferior", "Superior", "Nasal", "Temporal"]
ABN_LBL = {0: "normal", 1: "mild thinning", 2: "severe thinning"}
CDR_VERDICT = {0: "normal", 1: "borderline", 2: "abnormal"}
SIGN_KEYS = ["notch", "rnfl", "bayo", "beta", "ppa"]
SIGN_OUT = {"notch": "Notching", "rnfl": "RNFL_defect", "bayo": "Bayoneting_sign",
            "beta": "Beta_zone_atrophy", "ppa": "Peripapillary_atrophy"}
IDX2SIGN = {0: "absent", 1: "present", 2: "uncertain"}


def add_order_cdr(pred, npz):
    S, CP, ids = npz["scores"], npz["cdr_pred"], npz["ids"]
    for i, sid in enumerate(ids):
        sid = str(sid)
        order = np.argsort(-S[i])
        d = pred.setdefault(sid, {})
        d["vertical_CDR_verdict"] = CDR_VERDICT[int(CP[i, 0])]
        d["horizontal_CDR_verdict"] = CDR_VERDICT[int(CP[i, 1])]
        d["ISNT_observed_order"] = ">".join(RIM_LETTERS[j] for j in order)


def add_abn(pred, npz):
    AP, ids = npz["abn_pred"], npz["ids"]
    for i, sid in enumerate(ids):
        sid = str(sid)
        st = {RIM_NAMES[j]: ABN_LBL[int(AP[i, j])] for j in range(4)}
        d = pred.setdefault(sid, {})
        d["ISNT_quadrant_status"] = st
        d["ISNT_quadrant_status_str"] = "; ".join(f"{q}: {st[q]}" for q in RIM_NAMES)


def add_signs(pred, npz):
    GP, ids = npz["sign_pred"], npz["ids"]
    for i, sid in enumerate(ids):
        sid = str(sid)
        d = pred.setdefault(sid, {})
        for j, k in enumerate(SIGN_KEYS):
            d[SIGN_OUT[k]] = IDX2SIGN[int(GP[i, j])]
        d["Disc_hemorrhage"] = "absent"


def order_keys(d):
    """reorder each id's dict to the canonical key order used by indicator_block."""
    ORDER = ["vertical_CDR_verdict", "horizontal_CDR_verdict", "ISNT_observed_order",
             "ISNT_quadrant_status", "ISNT_quadrant_status_str", "Notching",
             "RNFL_defect", "Bayoneting_sign", "Beta_zone_atrophy",
             "Peripapillary_atrophy", "Disc_hemorrhage"]
    return {k: d[k] for k in ORDER if k in d}


def build_train():
    pred = {}
    for f in range(K):
        add_order_cdr(pred, np.load(HERE / f"outputs/oof_quad_t2aL_f{f}/test_preds.npz"))
        add_abn(pred, np.load(HERE / f"outputs/oof_quad_t2bL_f{f}/test_preds.npz"))
        add_signs(pred, np.load(HERE / f"outputs/oof_quad_t2aM_f{f}/test_preds.npz"))
    pred = {sid: order_keys(d) for sid, d in pred.items()}
    # every id must have all 11 fields
    bad = [sid for sid, d in pred.items() if len(d) != 11]
    assert not bad, f"incomplete ids: {bad[:5]} (n={len(bad)})"
    p = DATA / "predicted_indicators_train_quadmix.json"
    p.write_text(json.dumps(pred, indent=1, ensure_ascii=False))
    print(f"TRAIN quadmix: {len(pred)} ids -> {p}")


def build_test():
    pred = {}
    add_order_cdr(pred, np.load(HERE / "outputs/quad_tier2a_linear/test_preds_ids.npz"))
    add_abn(pred, np.load(HERE / "outputs/quad_tier2b_linear/test_preds_ids.npz"))
    add_signs(pred, np.load(HERE / "outputs/quad_tier2a_mlp/test_preds_ids.npz"))
    pred = {sid: order_keys(d) for sid, d in pred.items()}
    bad = [sid for sid, d in pred.items() if len(d) != 11]
    assert not bad, f"incomplete ids: {bad[:5]}"
    p = DATA / "predicted_indicators_test_quadmix.json"
    p.write_text(json.dumps(pred, indent=1, ensure_ascii=False))
    print(f"TEST quadmix: {len(pred)} ids -> {p}")


if __name__ == "__main__":
    build_train()
    build_test()

"""
Assemble MIXED-pooling quadreg (NUMERIC CDR) predicted indicators, per-indicator best:
  CDR (numeric)        <- tier2a-mlp    (MAE 0.070)
  order               <- tier2a-linear  (Kendall 1.642)
  signs               <- tier2a-linear  (sign F1 0.568)
  rim per-quad abn    <- tier2b-mlp     (abn F1 0.59)

TRAIN 613: 5-fold OOF (outputs/oof_quadreg_{t2aM,t2aL,t2bM}_f{0..4}/test_preds.npz w/ ids)
TEST  120: full-613 heads (outputs/quadreg_{...}/test_preds_ids.npz from predict_quadreg.py)

Output = NUMERIC format (byte-compatible with predicted_indicators_train_oof.json):
  ../qwen35_icl_glaucoma/DATA/predicted_indicators_train_quadreg.json
  ../qwen35_icl_glaucoma/DATA/predicted_indicators_test_quadreg.json
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
SIGN_KEYS = ["notch", "rnfl", "bayo", "beta", "ppa"]
SIGN_OUT = {"notch": "Notching", "rnfl": "RNFL_defect", "bayo": "Bayoneting_sign",
            "beta": "Beta_zone_atrophy", "ppa": "Peripapillary_atrophy"}
IDX2SIGN = {0: "absent", 1: "present", 2: "uncertain"}
ORDER = ["vertical_CDR", "horizontal_CDR", "ISNT_observed_order", "ISNT_quadrant_status",
         "ISNT_quadrant_status_str", "Notching", "RNFL_defect", "Bayoneting_sign",
         "Beta_zone_atrophy", "Peripapillary_atrophy", "Disc_hemorrhage"]


def add_cdr(pred, npz):
    CP, ids = npz["cdr_pred"], npz["ids"]
    for i, sid in enumerate(ids):
        d = pred.setdefault(str(sid), {})
        d["vertical_CDR"] = round(float(CP[i, 0]), 3)
        d["horizontal_CDR"] = round(float(CP[i, 1]), 3)


def add_order(pred, npz):
    S, ids = npz["scores"], npz["ids"]
    for i, sid in enumerate(ids):
        order = np.argsort(-S[i])
        pred.setdefault(str(sid), {})["ISNT_observed_order"] = ">".join(RIM_LETTERS[j] for j in order)


def add_abn(pred, npz):
    AP, ids = npz["abn_pred"], npz["ids"]
    for i, sid in enumerate(ids):
        st = {RIM_NAMES[j]: ABN_LBL[int(AP[i, j])] for j in range(4)}
        d = pred.setdefault(str(sid), {})
        d["ISNT_quadrant_status"] = st
        d["ISNT_quadrant_status_str"] = "; ".join(f"{q}: {st[q]}" for q in RIM_NAMES)


def add_signs(pred, npz):
    GP, ids = npz["sign_pred"], npz["ids"]
    for i, sid in enumerate(ids):
        d = pred.setdefault(str(sid), {})
        for j, k in enumerate(SIGN_KEYS):
            d[SIGN_OUT[k]] = IDX2SIGN[int(GP[i, j])]
        d["Disc_hemorrhage"] = "absent"


def finalize(pred):
    out = {sid: {k: d[k] for k in ORDER if k in d} for sid, d in pred.items()}
    bad = [sid for sid, d in out.items() if len(d) != 11]
    assert not bad, f"incomplete: {bad[:5]} (n={len(bad)})"
    return out


def build_train():
    pred = {}
    for f in range(K):
        add_cdr(pred, np.load(HERE / f"outputs/oof_quadreg_t2aM_f{f}/test_preds.npz"))
        add_order(pred, np.load(HERE / f"outputs/oof_quadreg_t2aL_f{f}/test_preds.npz"))
        add_signs(pred, np.load(HERE / f"outputs/oof_quadreg_t2aL_f{f}/test_preds.npz"))
        add_abn(pred, np.load(HERE / f"outputs/oof_quadreg_t2bM_f{f}/test_preds.npz"))
    out = finalize(pred)
    p = DATA / "predicted_indicators_train_quadreg.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"TRAIN quadreg: {len(out)} -> {p}")


def build_test():
    pred = {}
    add_cdr(pred, np.load(HERE / "outputs/quadreg_tier2a_mlp/test_preds_ids.npz"))
    add_order(pred, np.load(HERE / "outputs/quadreg_tier2a_linear/test_preds_ids.npz"))
    add_signs(pred, np.load(HERE / "outputs/quadreg_tier2a_linear/test_preds_ids.npz"))
    add_abn(pred, np.load(HERE / "outputs/quadreg_tier2b_mlp/test_preds_ids.npz"))
    out = finalize(pred)
    p = DATA / "predicted_indicators_test_quadreg.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"TEST quadreg: {len(out)} -> {p}")


if __name__ == "__main__":
    build_train()
    build_test()

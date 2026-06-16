"""
Assemble OOF predicted indicators (Method B) for the 613 train ids from the
15 fold runs, AND rebuild the 120-test indicators from the SAME 3 full-train
heads (so train & test use identical heads: CDR tier2a, rim orderabn tier3,
signs tier1). Verifies npz<->manifest gt alignment per fold (hard abort on
mismatch). Output keys byte-identical to predicted_indicators.json.

  train -> ../qwen35_icl_glaucoma/DATA/predicted_indicators_train_oof.json
  test  -> ../qwen35_icl_glaucoma/DATA/predicted_indicators_test_newheads.json
"""
import csv, json, glob
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
LBL2IDX = {"absent": 0, "present": 1, "uncertain": 2, "NA": -1, "": -1}
IDX2LBL = {0: "absent", 1: "present", 2: "uncertain"}


def _csv(p): return list(csv.DictReader(open(p, newline="", encoding="utf-8")))


def cdr_rows(rows, split):
    out = []
    for r in rows:
        if r["status"] != "ok": continue
        if r["cdr_v"] in ("", "None") or r["cdr_h"] in ("", "None"): continue
        if r["split"] == split: out.append((r["id"], float(r["cdr_v"]), float(r["cdr_h"])))
    return out


def isnt_rows(rows, split):
    RK = ("rim_i", "rim_s", "rim_n", "rim_t")
    out = []
    for r in rows:
        if r["status"] != "ok": continue
        if any(r[k] in ("", "None") for k in RK): continue
        if r["split"] == split: out.append((r["id"], [float(r[k]) for k in RK]))
    return out


def signs_rows(rows, split):
    out = []
    for r in rows:
        if r["split"] == split:
            out.append((r["id"], [LBL2IDX.get(r[f"lbl_{k}"], -1) for k in SIGN_KEYS]))
    return out


def add_cdr(pred, npz, rows):
    P, G = npz["pred"], npz["gt"]
    assert len(rows) == len(P), f"CDR len {len(rows)}/{len(P)}"
    for i, (sid, cv, ch) in enumerate(rows):
        assert abs(G[i, 0]-cv) < 1e-3 and abs(G[i, 1]-ch) < 1e-3, f"CDR gt mismatch {sid}"
        pred.setdefault(sid, {})
        pred[sid]["vertical_CDR"] = round(float(P[i, 0]), 3)
        pred[sid]["horizontal_CDR"] = round(float(P[i, 1]), 3)


def add_rim(pred, npz, rows):
    S, GR, AB = npz["scores"], npz["gt_rim"], npz["abn_pred"]
    assert len(rows) == len(S), f"RIM len {len(rows)}/{len(S)}"
    for i, (sid, rim) in enumerate(rows):
        assert all(abs(GR[i, j]-rim[j]) < 1e-3 for j in range(4)), f"RIM gt mismatch {sid}"
        order = np.argsort(-S[i])
        pred.setdefault(sid, {})
        pred[sid]["ISNT_observed_order"] = ">".join(RIM_LETTERS[j] for j in order)
        st = {RIM_NAMES[j]: ABN_LBL[int(AB[i, j])] for j in range(4)}
        pred[sid]["ISNT_quadrant_status"] = st
        pred[sid]["ISNT_quadrant_status_str"] = "; ".join(f"{q}: {st[q]}" for q in RIM_NAMES)


def add_signs(pred, npz, rows):
    P, G = npz["pred"], npz["gt"]
    assert len(rows) == len(P), f"SIGN len {len(rows)}/{len(P)}"
    for i, (sid, y) in enumerate(rows):
        assert list(G[i]) == y, f"SIGN gt mismatch {sid}"
        pred.setdefault(sid, {})
        for j, k in enumerate(SIGN_KEYS):
            pred[sid][SIGN_OUT[k]] = IDX2LBL.get(int(P[i, j]), "absent")
        pred[sid]["Disc_hemorrhage"] = "absent"


def cdr_npz(d):
    g = glob.glob(str(d / "test_preds_*.npz")) + glob.glob(str(d / "test_preds.npz"))
    assert g, f"no cdr npz in {d}"
    return np.load(g[0])


def build_train_oof():
    pred = {}
    cdrM = {f: _csv(HERE/f"oof/cdr_fold{f}.csv") for f in range(K)}
    isntM = {f: _csv(HERE/f"oof/isnt_fold{f}.csv") for f in range(K)}
    signM = {f: _csv(HERE/f"oof/signs_fold{f}.csv") for f in range(K)}
    for f in range(K):
        add_cdr(pred, cdr_npz(HERE/f"outputs/oof_cdr_f{f}"), cdr_rows(cdrM[f], "test"))
        add_rim(pred, np.load(HERE/f"outputs/oof_rim_f{f}/test_preds.npz"), isnt_rows(isntM[f], "test"))
        add_signs(pred, np.load(HERE/f"outputs/oof_signs_f{f}/test_preds.npz"), signs_rows(signM[f], "test"))
    out = DATA / "predicted_indicators_train_oof.json"
    out.write_text(json.dumps(pred, indent=1, ensure_ascii=False))
    print(f"TRAIN OOF: {len(pred)} ids -> {out}")
    return pred


def build_test_newheads():
    pred = {}
    cdrM = _csv(HERE/"cdr_manifest.csv"); isntM = _csv(HERE/"isnt_manifest.csv"); signM = _csv(HERE/"signs_manifest_clean.csv")
    add_cdr(pred, cdr_npz(HERE/"outputs/tier2a_linear_20260516_012723"), cdr_rows(cdrM, "test"))
    add_rim(pred, np.load(HERE/"outputs/orderabn_tier3_w1/test_preds.npz"), isnt_rows(isntM, "test"))
    add_signs(pred, np.load(HERE/"outputs/signs_tier1_linear_wce_20260516_054817/test_preds.npz"), signs_rows(signM, "test"))
    out = DATA / "predicted_indicators_test_newheads.json"
    out.write_text(json.dumps(pred, indent=1, ensure_ascii=False))
    print(f"TEST newheads: {len(pred)} ids -> {out}")
    return pred


if __name__ == "__main__":
    import sys
    if "--test-only" in sys.argv:
        build_test_newheads()
    else:
        build_train_oof(); build_test_newheads()

#!/usr/bin/env python3
"""
Build predicted_indicators.json from the BEST RETFound header per task,
to replace the GT oracle indicators in the VLM pipeline (deployable test).

Best heads (by val metric):
  CDR    : outputs/tier2a_linear_20260516_012723          (val MAE 0.059)
  ORDER  : outputs/order_tier2b_linear_20260516_044017     (val Kendall 1.726)
  SIGNS  : outputs/signs_tier1_linear_wce_20260516_054817  (val macroF1 0.667)

CRITICAL: each header's test_preds.npz rows are in that header's test
DataLoader order = its manifest CSV order, filtered the SAME way the
dataset code filters (test loader is shuffle=False). We reconstruct that
id order per task AND VERIFY it against the npz's stored `gt` (cdr values /
rim values / sign classes must match the manifest for the reconstructed
ids) before trusting the mapping. Any mismatch -> hard abort.

Output keys are IDENTICAL to oracle_indicators.json so the VLM harness
reads it unchanged (only the values differ: predicted vs GT).
Disc_hemorrhage = "absent" (decision C: signs head dropped it, GT ~99% absent).
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "qwen35_icl_glaucoma" / "DATA" / "predicted_indicators.json"

CDR_DIR = HERE / "outputs/tier2a_linear_20260516_012723"
ORD_DIR = HERE / "outputs/order_tier2b_linear_20260516_044017"
SIG_DIR = HERE / "outputs/signs_tier1_linear_wce_20260516_054817"

LBL2IDX = {"absent": 0, "present": 1, "uncertain": 2, "NA": -1, "": -1}
IDX2LBL = {0: "absent", 1: "present", 2: "uncertain"}
RIM_LETTERS = ["I", "S", "N", "T"]                 # train_order col order: Inf,Sup,Nas,Tmp
SIGN_KEYS = ["notch", "rnfl", "bayo", "beta", "ppa"]
SIGN_OUT = {"notch": "Notching", "rnfl": "RNFL_defect",
            "bayo": "Bayoneting_sign", "beta": "Beta_zone_atrophy",
            "ppa": "Peripapillary_atrophy"}


def _csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cdr_test_ids():
    """Replicate dataset.py _load_manifest: status==ok, cdr_v/h present."""
    ids = []
    for r in _csv(HERE / "cdr_manifest.csv"):
        if r["status"] != "ok":
            continue
        if r["cdr_v"] in ("", "None") or r["cdr_h"] in ("", "None"):
            continue
        if r["split"] == "test":
            ids.append((r["id"], float(r["cdr_v"]), float(r["cdr_h"])))
    return ids


def isnt_test_ids():
    """Replicate dataset_isnt.py: status==ok, all rim present."""
    RK = ("rim_i", "rim_s", "rim_n", "rim_t")
    ids = []
    for r in _csv(HERE / "isnt_manifest.csv"):
        if r["status"] != "ok":
            continue
        if any(r[k] in ("", "None") for k in RK):
            continue
        if r["split"] == "test":
            ids.append((r["id"], [float(r[k]) for k in RK]))
    return ids


def signs_test_ids():
    """Replicate dataset_signs.py: NO status filter, all rows."""
    ids = []
    for r in _csv(HERE / "signs_manifest_clean.csv"):
        if r["split"] == "test":
            y = [LBL2IDX.get(r[f"lbl_{k}"], -1) for k in SIGN_KEYS]
            ids.append((r["id"], y))
    return ids


def main():
    pred = {}                                  # id -> indicator dict

    # ---- CDR ----
    z = np.load(CDR_DIR / "test_preds.npz")
    P, G = z["pred"], z["gt"]
    rows = cdr_test_ids()
    assert len(rows) == len(P) == 120, f"CDR len {len(rows)}/{len(P)}"
    for i, (sid, cv, ch) in enumerate(rows):
        assert abs(G[i, 0] - cv) < 1e-3 and abs(G[i, 1] - ch) < 1e-3, \
            f"CDR gt mismatch @{i} {sid}: manifest({cv},{ch}) vs npz{tuple(G[i])}"
        pred.setdefault(sid, {})
        pred[sid]["vertical_CDR"] = round(float(P[i, 0]), 3)
        pred[sid]["horizontal_CDR"] = round(float(P[i, 1]), 3)
    print(f"CDR   : 120 rows, gt-aligned OK  (e.g. {rows[0][0]} -> "
          f"pred {pred[rows[0][0]]['vertical_CDR']}/"
          f"{pred[rows[0][0]]['horizontal_CDR']})")

    # ---- ISNT order ----
    z = np.load(ORD_DIR / "test_preds.npz")
    S, GR = z["scores"], z["gt_rim"]
    rows = isnt_test_ids()
    assert len(rows) == len(S) == 120, f"ORD len {len(rows)}/{len(S)}"
    for i, (sid, rim) in enumerate(rows):
        assert all(abs(GR[i, j] - rim[j]) < 1e-3 for j in range(4)), \
            f"ISNT gt mismatch @{i} {sid}"
        order = np.argsort(-S[i])               # desc -> ranking of I,S,N,T
        pred.setdefault(sid, {})
        pred[sid]["ISNT_observed_order"] = ">".join(RIM_LETTERS[j] for j in order)
    print(f"ORDER : 120 rows, gt-aligned OK  (e.g. {rows[0][0]} -> "
          f"{pred[rows[0][0]]['ISNT_observed_order']})")

    # ---- signs ----
    z = np.load(SIG_DIR / "test_preds.npz")
    P, G = z["pred"], z["gt"]
    rows = signs_test_ids()
    assert len(rows) == len(P) == 120, f"SIGN len {len(rows)}/{len(P)}"
    for i, (sid, y) in enumerate(rows):
        assert list(G[i]) == y, \
            f"SIGN gt mismatch @{i} {sid}: manifest{y} vs npz{list(G[i])}"
        pred.setdefault(sid, {})
        for j, k in enumerate(SIGN_KEYS):
            pred[sid][SIGN_OUT[k]] = IDX2LBL.get(int(P[i, j]), "absent")
        pred[sid]["Disc_hemorrhage"] = "absent"          # decision C
    print(f"SIGNS : 120 rows, gt-aligned OK  (e.g. {rows[0][0]} -> "
          f"{ {SIGN_OUT[k]: pred[rows[0][0]][SIGN_OUT[k]] for k in SIGN_KEYS} })")

    assert len(pred) == 120, f"final {len(pred)} ids (expect 120)"
    OUT.write_text(json.dumps(pred, indent=1, ensure_ascii=False))
    print(f"\nWrote {OUT}  ({len(pred)} ids)")

    # side-by-side sanity vs oracle for 3 ids
    oracle = json.loads((OUT.parent / "oracle_indicators.json").read_text())
    print("\n--- predicted vs ORACLE (3 ids) ---")
    for sid in list(pred)[:3]:
        o, p = oracle.get(sid, {}), pred[sid]
        print(f"  {sid}")
        print(f"    CDR  GT {o.get('vertical_CDR')}/{o.get('horizontal_CDR')}"
              f"  PRED {p['vertical_CDR']}/{p['horizontal_CDR']}")
        print(f"    ISNT GT {o.get('ISNT_observed_order')}"
              f"  PRED {p['ISNT_observed_order']}")
        print(f"    PPA  GT {o.get('Peripapillary_atrophy')}"
              f"  PRED {p['Peripapillary_atrophy']}")


if __name__ == "__main__":
    main()

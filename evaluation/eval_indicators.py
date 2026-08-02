"""Stage-1 evaluation: score the four predicted clinical indicators against the
ground-truth manifests.

    python eval_indicators.py \
        --pred predicted_indicators.json \
        --cdr-manifest cdr_manifest.csv \
        --isnt-manifest isnt_manifest.csv \
        --signs-manifest signs_manifest.csv \
        --split test --out stage1_metrics.json

`predicted_indicators.json` is the output of retfound_finetune/predict_indicators.py.
The manifests are the same CSVs used for training (see retfound_finetune/dataset.py).
"""
import argparse
import csv
import json

import numpy as np

import metrics

RIM_KEYS = ("rim_i", "rim_s", "rim_n", "rim_t")
ABN_KEYS = ("abn_i", "abn_s", "abn_n", "abn_t")
RIM_NAMES = ("Inferior", "Superior", "Nasal", "Temporal")
RIM_LETTER = {"Inferior": "I", "Superior": "S", "Nasal": "N", "Temporal": "T"}
SIGN_KEYS = ("notch", "rnfl", "bayo", "beta", "ppa")
SIGN_FIELD = {"notch": "Notching", "rnfl": "RNFL_defect", "bayo": "Bayoneting_sign",
              "beta": "Beta_zone_atrophy", "ppa": "Peripapillary_atrophy"}
LBL2IDX = {"absent": 0, "present": 1, "uncertain": 2, "NA": -1, "": -1}
STATUS2IDX = {"normal": 0, "mild thinning": 1, "severe thinning": 2}


def read(path, split):
    return [r for r in csv.DictReader(open(path))
            if split is None or r.get("split") == split]


def order_from_ratios(ratios):
    # rank the four sectors thickest -> thinnest, return their I/S/N/T letters
    return [RIM_LETTER[RIM_NAMES[q]] for q in np.argsort(-np.asarray(ratios))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--cdr-manifest", required=True)
    ap.add_argument("--isnt-manifest", required=True)
    ap.add_argument("--signs-manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pred = json.load(open(args.pred))
    cdr_gt = {r["id"]: r for r in read(args.cdr_manifest, args.split)}
    isnt_gt = {r["id"]: r for r in read(args.isnt_manifest, args.split)}
    signs_gt = {r["id"]: r for r in read(args.signs_manifest, args.split)}
    ids = [i for i in pred if i in cdr_gt and i in isnt_gt and i in signs_gt]
    print(f"scoring {len(ids)} cases (split={args.split})")

    # --- CDR ---
    pv, gv, ph, gh = [], [], [], []
    for i in ids:
        if cdr_gt[i].get("cdr_v") in ("", "None"):
            continue
        pv.append(pred[i]["vertical_CDR"]); gv.append(float(cdr_gt[i]["cdr_v"]))
        ph.append(pred[i]["horizontal_CDR"]); gh.append(float(cdr_gt[i]["cdr_h"]))
    cdr_res = {"vertical": metrics.cdr(pv, gv), "horizontal": metrics.cdr(ph, gh)}

    # --- ISNT rim order ---
    po, go = [], []
    for i in ids:
        r = isnt_gt[i]
        if any(r[k] in ("", "None") for k in RIM_KEYS):
            continue
        go.append(order_from_ratios([float(r[k]) for k in RIM_KEYS]))
        po.append(pred[i]["ISNT_observed_order"].replace(" ", "").split(">"))
    isnt_res = metrics.isnt_order(po, go)

    # --- per-quadrant rim status ---
    pr, gr = [], []
    for i in ids:
        r = isnt_gt[i]
        if any(r.get(k, "") in ("", "None") for k in ABN_KEYS):
            continue
        gr.append([int(r[k]) for k in ABN_KEYS])
        pr.append([STATUS2IDX[pred[i]["ISNT_quadrant_status"][n]] for n in RIM_NAMES])
    rim_res = metrics.rim_status(np.array(pr), np.array(gr))

    # --- glaucomatous signs ---
    ps, gs = [], []
    for i in ids:
        gs.append([LBL2IDX.get(signs_gt[i].get(f"lbl_{k}", ""), -1) for k in SIGN_KEYS])
        ps.append([LBL2IDX.get(pred[i].get(SIGN_FIELD[k], ""), -1) for k in SIGN_KEYS])
    signs_res = metrics.signs(np.array(ps), np.array(gs))

    out = {"n": len(ids), "split": args.split,
           "cdr": cdr_res, "isnt_order": isnt_res, "rim_status": rim_res, "signs": signs_res}
    print(json.dumps(out, indent=2))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print("saved ->", args.out)


if __name__ == "__main__":
    main()

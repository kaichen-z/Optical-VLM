"""
Assemble the QUALITATIVE-CDR (Step2 verdict) variant of the predicted-indicator
files, parallel to assemble_oof.py's numeric output.

CDR comes from the cdr-VERDICT head (3-class normal/borderline/abnormal x V/H):
  train 613 -> 5-fold OOF heads outputs/oof_cdrv_f{0..4}/test_preds.npz
               (each npz: pred[N,2] class idx, gt[N,2], ids[N]) -> held-out fold
  test  120 -> full-613 head outputs/cdrv_tier2a/test_preds.npz
               (pred[120,2], gt[120,2]; NO ids -> reconstruct from manifest
               filtered "test" order + assert gt match, exactly like assemble_oof)

Everything ELSE (rim order+per-quadrant status, signs) is copied verbatim from
the numeric predicted_indicators_*.json so the ONLY difference vs the numeric
variant is: vertical_CDR/horizontal_CDR (floats) -> vertical_CDR_verdict/
horizontal_CDR_verdict (normal|borderline|abnormal).

Outputs:
  ../qwen35_icl_glaucoma/DATA/predicted_indicators_train_oof_cdrq.json
  ../qwen35_icl_glaucoma/DATA/predicted_indicators_test_newheads_cdrq.json
"""
import csv, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "qwen35_icl_glaucoma" / "DATA"
K = 5
IDX2V = {0: "normal", 1: "borderline", 2: "abnormal"}
LBL2IDX = {"normal": 0, "borderline": 1, "abnormal": 2, "": -1, "None": -1}
NUM_TRAIN, NUM_TEST = 613, 120


def _csv(p):
    return list(csv.DictReader(open(p, newline="", encoding="utf-8")))


def verdict_rows(rows, split):
    """(id, [v_idx, h_idx]) in the SAME filtered CSV order dataset_cdr_verdict
    uses: status==ok and cdr_v/cdr_h present."""
    out = []
    for r in rows:
        if r["status"] != "ok":
            continue
        if r["cdr_v"] in ("", "None") or r["cdr_h"] in ("", "None"):
            continue
        if r["split"] != split:
            continue
        out.append((r["id"], [LBL2IDX[(r.get("cdr_v_verdict") or "").strip()],
                              LBL2IDX[(r.get("cdr_h_verdict") or "").strip()]]))
    return out


def swap_cdr(base, verdict):
    """Copy base indicator dict, drop numeric CDR, add verdict CDR."""
    out = {}
    for sid, ind in base.items():
        d = dict(ind)
        d.pop("vertical_CDR", None)
        d.pop("horizontal_CDR", None)
        if sid not in verdict:
            raise SystemExit(f"missing verdict for id {sid}")
        vv, hv = verdict[sid]
        # rebuild dict so verdict CDR leads (mirror numeric key order)
        ordered = {"vertical_CDR_verdict": vv, "horizontal_CDR_verdict": hv}
        ordered.update(d)
        out[sid] = ordered
    return out


def build_train():
    verdict = {}
    for f in range(K):
        d = np.load(HERE / f"outputs/oof_cdrv_f{f}/test_preds.npz")
        P, ids = d["pred"], d["ids"]
        # cross-check: ids match this fold's held-out (split==test) rows
        fold_rows = verdict_rows(_csv(HERE / f"oof/cdr_fold{f}.csv"), "test")
        assert len(fold_rows) == len(P) == len(ids), \
            f"f{f} len {len(fold_rows)}/{len(P)}/{len(ids)}"
        gt_by_id = {sid: y for sid, y in fold_rows}
        for i, sid in enumerate(ids):
            sid = str(sid)
            # gt sanity: npz gt must equal manifest verdict gt for this id
            assert list(d["gt"][i]) == gt_by_id[sid], f"gt mismatch {sid}"
            verdict[sid] = [IDX2V[int(P[i, 0])], IDX2V[int(P[i, 1])]]
    base = json.loads((DATA / "predicted_indicators_train_oof.json").read_text())
    assert len(base) == NUM_TRAIN, f"train base {len(base)}"
    assert len(verdict) == NUM_TRAIN, f"train verdict {len(verdict)}"
    out = swap_cdr(base, verdict)
    p = DATA / "predicted_indicators_train_oof_cdrq.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"TRAIN cdrq: {len(out)} ids -> {p}")


def build_test():
    d = np.load(HERE / "outputs/cdrv_tier2a/test_preds.npz")
    P, G = d["pred"], d["gt"]
    rows = verdict_rows(_csv(HERE / "cdr_manifest.csv"), "test")
    assert len(rows) == len(P) == NUM_TEST, f"test len {len(rows)}/{len(P)}"
    verdict = {}
    for i, (sid, y) in enumerate(rows):
        assert list(G[i]) == y, f"test gt mismatch {sid} {list(G[i])} vs {y}"
        verdict[sid] = [IDX2V[int(P[i, 0])], IDX2V[int(P[i, 1])]]
    base = json.loads((DATA / "predicted_indicators_test_newheads.json").read_text())
    assert len(base) == NUM_TEST, f"test base {len(base)}"
    out = swap_cdr(base, verdict)
    p = DATA / "predicted_indicators_test_newheads_cdrq.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"TEST cdrq: {len(out)} ids -> {p}")


if __name__ == "__main__":
    build_train()
    build_test()

"""
Assemble the final target-consistent multimodal SFT dataset for 27B VLM LoRA.
Each record: image + (query + OOF-predicted indicator block) -> target CoT
(Step1 orig + Step2-6 rewritten + Diagnosis=GT). Format byte-compatible with
run_medgemma_indicators (SYSTEM_PROMPT, TEST_QUERY_TEXT, indicator_block) so
train and deploy match. Stratified 10% val split (seed 42).

Out: DATA/sft_train_tc.jsonl  (fields: id, image, system, prompt, target, true_diagnosis, split)
"""
import json, ast, glob, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
D = HERE / "DATA"

SYSTEM_PROMPT = ("You are an expert ophthalmologist assessing a retinal fundus photograph "
    "for glaucoma. You are given the fundus image together with automated measurements "
    "(cup-to-disc ratios, the ISNT rim order, per-quadrant rim status, and glaucomatous "
    "signs). These measurements are produced by an automated model and may be imprecise; "
    "weigh both the image and the measurements to reach your assessment.")
TEST_QUERY_TEXT = ("Analyze the optic nerve head and generate a clinical reasoning chain to "
    "assess for possible glaucoma. Give your final diagnosis in one of the "
    "following categories: not likely / borderline / likely.")
SIGN_KEYS = ["Notching", "RNFL_defect", "Disc_hemorrhage",
             "Bayoneting_sign", "Beta_zone_atrophy", "Peripapillary_atrophy"]
STEP_ORDER = ["Step1 - Image Quality Assessment", "Step2 - CDR Evaluation",
              "Step3 - ISNT Rule Analysis", "Step4 - Glaucomatous Signs Check",
              "Step5 - Structural Summary", "Step6 - Final Classification",
              "Diagnosis Classification"]


def indicator_block(o):
    L = []
    if "vertical_CDR" in o: L.append(f"Vertical CDR: {o['vertical_CDR']}")
    if "horizontal_CDR" in o: L.append(f"Horizontal CDR: {o['horizontal_CDR']}")
    if "ISNT_observed_order" in o: L.append(f"ISNT observed order: {o['ISNT_observed_order']}")
    if "ISNT_quadrant_status_str" in o: L.append(f"Per-quadrant rim status: {o['ISNT_quadrant_status_str']}")
    for k in SIGN_KEYS:
        if k in o: L.append(f"{k.replace('_',' ')}: {o[k]}")
    return ("\n\nMeasured clinical indicators for THIS patient "
            "(verified; use them in your analysis):\n" + "\n".join(L))


def main():
    oof = json.loads((D / "predicted_indicators_train_oof.json").read_text())
    sft = {r["id"]: r for r in (json.loads(l) for l in open(D / "sft_trainA.jsonl"))}
    rw = {}
    for f in sorted(glob.glob(str(D / "rewrites/batch_*.json"))):
        rw.update(json.load(open(f)))

    ids = [r["id"] for r in (json.loads(l) for l in open(D / "rewrite_input.jsonl"))]
    # stratified 10% val by GT dx
    by = {}
    for sid in ids:
        by.setdefault(sft[sid]["true_diagnosis"], []).append(sid)
    rng = random.Random(42)
    val = set()
    for dx, lst in by.items():
        l = lst[:]; rng.shuffle(l)
        n = max(1, round(0.10 * len(l)))
        val.update(l[:n])

    recs = []
    for sid in ids:
        orig = ast.literal_eval(sft[sid]["target"])
        tgt = {"Step1 - Image Quality Assessment": orig["Step1 - Image Quality Assessment"]}
        for k in ["Step2 - CDR Evaluation", "Step3 - ISNT Rule Analysis",
                  "Step4 - Glaucomatous Signs Check", "Step5 - Structural Summary",
                  "Step6 - Final Classification"]:
            tgt[k] = rw[sid][k]
        tgt["Diagnosis Classification"] = orig["Diagnosis Classification"]  # = GT
        tgt = {k: tgt[k] for k in STEP_ORDER}                                # fixed order
        recs.append({
            "id": sid,
            "image": sid,                                                    # resolve to photos/<id>.jpg at train time
            "system": SYSTEM_PROMPT,
            "prompt": TEST_QUERY_TEXT + indicator_block(oof[sid]),
            "target": str(tgt),                                             # python-dict-style, matches deploy ICL
            "true_diagnosis": sft[sid]["true_diagnosis"],
            "split": "val" if sid in val else "train",
        })
    out = D / "sft_train_tc.jsonl"
    with open(out, "w") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections
    sp = collections.Counter(r["split"] for r in recs)
    dxsp = collections.Counter((r["split"], r["true_diagnosis"]) for r in recs)
    print(f"wrote {out}: {len(recs)} recs  split={dict(sp)}")
    print("by split×dx:", dict(dxsp))


if __name__ == "__main__":
    main()

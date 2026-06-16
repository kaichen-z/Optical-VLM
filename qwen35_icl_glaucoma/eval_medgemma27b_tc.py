#!/usr/bin/env python3
"""
Evaluate the fine-tuned MedGemma-27B (base + LoRA adapter) on the 120 test set,
ZERO-SHOT (NO ICL, NO crop) — exactly matching the SFT format:
  system(A) + user(image + query + predicted-indicator block) -> generate CoT.
Indicators come from predicted_indicators_test_newheads.json (SAME heads as the
OOF train indicators: CDR tier2a / rim tier3 / signs tier1).

Usage:
  CUDA_VISIBLE_DEVICES=1,2 python eval_medgemma27b_tc.py \
     --base /data/.../medgemma-27b-it/snapshots/<snap> \
     --adapter outputs/mg27_lora_tc \
     --indicators DATA/predicted_indicators_test_newheads.json
"""
from __future__ import annotations
import argparse, json, time, collections
from pathlib import Path
import torch
from PIL import Image
import run_qwen35_incontext as H   # parse_diagnosis, load_test_records

_HERE = Path(__file__).resolve().parent
TEST_PHOTOS = _HERE / "DATA/test_120/photos"
TEST_DESC = _HERE / "DATA/test_120/descriptions"

# MUST match assemble_sft_tc.py exactly
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


def indicator_block(o):
    L = []
    if "vertical_CDR_verdict" in o: L.append(f"Vertical CDR: {o['vertical_CDR_verdict']}")
    elif "vertical_CDR" in o: L.append(f"Vertical CDR: {o['vertical_CDR']}")
    if "horizontal_CDR_verdict" in o: L.append(f"Horizontal CDR: {o['horizontal_CDR_verdict']}")
    elif "horizontal_CDR" in o: L.append(f"Horizontal CDR: {o['horizontal_CDR']}")
    if "ISNT_observed_order" in o: L.append(f"ISNT observed order: {o['ISNT_observed_order']}")
    if "ISNT_quadrant_status_str" in o: L.append(f"Per-quadrant rim status: {o['ISNT_quadrant_status_str']}")
    for k in SIGN_KEYS:
        if k in o: L.append(f"{k.replace('_',' ')}: {o[k]}")
    return ("\n\nMeasured clinical indicators for THIS patient "
            "(verified; use them in your analysis):\n" + "\n".join(L))


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--indicators", default=str(_HERE / "DATA/predicted_indicators_test_newheads.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--only-ids", default=None, help="file of ids (one per line) to restrict to")
    args = ap.parse_args()

    IND = json.loads(Path(args.indicators).read_text())
    proc = AutoProcessor.from_pretrained(args.base, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print("loaded base+adapter")

    test = H.load_test_records(TEST_PHOTOS, TEST_DESC)
    if args.only_ids:
        keep = set(open(args.only_ids).read().split())
        test = [r for r in test if r["id"] in keep]
    if args.num_shards > 1:
        test = test[args.shard_id::args.num_shards]   # strided
    print(f"this run: {len(test)} cases (shard {args.shard_id}/{args.num_shards})")
    out_dir = Path(args.out or f"outputs/eval_mg27_tc_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, r in enumerate(test):
        sid = r["id"]
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": Image.open(r["image_path"]).convert("RGB")},
                {"type": "text", "text": TEST_QUERY_TEXT + indicator_block(IND.get(sid, {}))}]},
        ]
        try:
            inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1)
            txt = proc.decode(gen[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        except Exception as e:
            txt = f"<<GEN_ERROR {e}>>"
        label, status = H.parse_diagnosis(txt)
        rec = {"id": sid, "true": r["true_diagnosis"], "pred": label,
               "parse_status": status, "output": txt}
        results.append(rec)
        with (out_dir / "predictions.jsonl").open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (i + 1) % 10 == 0: print(f"{i+1}/{len(test)}")

    norm = lambda s: ("not likely" if "not" in s else "borderline" if "border" in s
                      else "likely" if "likely" in s else s)
    cls = ["not likely", "borderline", "likely"]
    valid = [r for r in results if r["pred"] in cls]
    acc = sum(norm(r["true"]) == r["pred"] for r in valid) / max(len(valid), 1)
    conf = {t: collections.Counter() for t in cls}
    for r in valid: conf[norm(r["true"])][r["pred"]] += 1
    summary = {"adapter": args.adapter, "indicators": Path(args.indicators).name,
               "n": len(results), "parsed": len(valid), "acc": round(acc, 4),
               "per_class": {t: {"n": sum(conf[t].values()),
                   "recall": round(conf[t][t]/sum(conf[t].values()), 3) if sum(conf[t].values()) else None,
                   "confusion": dict(conf[t])} for t in cls}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("->", out_dir)


if __name__ == "__main__":
    main()

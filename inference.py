#!/usr/bin/env python3
"""End-to-end inference for a single fundus image.

    image --> RETFound + trained heads (Stage 1) --> indicator block
          --> MedGemma-27B + LoRA (Stage 2)       --> six-step CoT + diagnosis

    python inference.py \
        --stage1-ckpt retfound_finetune/outputs/tier2a_linear/best.pt \
        --model-id google/medgemma-27b-it \
        --adapter medgemma27b_finetune/outputs/mg27_lora \
        --image path/to/fundus.jpg
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "retfound_finetune"))
from predict_indicators import load_stage1, predict_batch, TF   # noqa: E402

SYSTEM = ("You are an expert ophthalmologist assessing a retinal fundus photograph "
          "for glaucoma. You are given the fundus image together with automated "
          "measurements (cup-to-disc ratios, the ISNT rim order, per-quadrant rim "
          "status, glaucomatous signs, and an overall glaucoma probability). These "
          "measurements are produced by an automated model and may be imprecise; weigh "
          "both the image and the measurements to reach your assessment.")

QUERY = ("Analyze the optic nerve head and generate a clinical reasoning chain to "
         "assess for possible glaucoma. Give your final diagnosis as one of the "
         "following: not likely / likely.")

SIGN_FIELDS = ["Notching", "RNFL_defect", "Bayoneting_sign",
               "Beta_zone_atrophy", "Peripapillary_atrophy"]


def indicator_block(ind):
    lines = [f"Vertical CDR: {ind['vertical_CDR']}",
             f"Horizontal CDR: {ind['horizontal_CDR']}",
             f"ISNT observed order: {ind['ISNT_observed_order']}",
             f"Per-quadrant rim status: {ind['ISNT_quadrant_status_str']}"]
    for k in SIGN_FIELDS:
        if k in ind:
            lines.append(f"{k.replace('_', ' ')}: {ind[k]}")
    lines.append(
        f"Automated overall glaucoma probability: {ind['glaucoma_probability']} "
        f"({ind['glaucoma_confidence']} confidence) - automated image-level estimate, "
        f"may be imprecise; weigh against the image and the measurements above.")
    return ("\n\nMeasured clinical indicators for THIS patient "
            "(verified; use them in your analysis):\n" + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-ckpt", required=True, help="RETFound heads checkpoint (best.pt)")
    ap.add_argument("--model-id", required=True, help="MedGemma-27B snapshot dir or HF id")
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir")
    ap.add_argument("--image", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Stage 1: predicted indicators + diagnosis probability
    backbone, heads = load_stage1(args.stage1_ckpt, device)
    img = Image.open(args.image).convert("RGB")
    ind = predict_batch(backbone, heads, TF(img).unsqueeze(0).to(device))[0]

    # Stage 2: MedGemma + LoRA generates the reasoning chain
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, args.adapter).eval()

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "image"},
                                     {"type": "text", "text": QUERY + indicator_block(ind)}]},
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[[img]], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    report = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print("=== Stage-1 predicted indicators ===")
    for k, v in ind.items():
        print(f"  {k}: {v}")
    print("\n=== Generated reasoning chain ===")
    print(report)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MedGemma-27B VLM LoRA SFT on the target-consistent dataset (sft_train_tc.jsonl).

Input  per sample : fundus image + (query + OOF-predicted indicator block)
Target per sample : full 6-step CoT (Step1 orig + Step2-6 rewritten) + GT diagnosis
Goal              : zero-shot (NO ICL) glaucoma CoT that weighs image + noisy indicators.

Design (discussed & agreed):
  * bf16 + device_map="auto" (shard 27B across visible GPUs; no quantization).
  * LoRA on ATTENTION ONLY (q/k/v/o_proj) of the language model, r=16, alpha=32,
    dropout=0.05, NO lm_head, NO MLP  -> anti-overfit for 552 small data (Plan A finding).
  * Vision tower FROZEN.
  * Loss masked to assistant tokens ONLY, via the two-apply_chat_template method
    (prefix length per example) -> NO hardcoded special-token ids (works on 27B).
  * Oversample minority diagnosis classes x2 (borderline, likely); train split is
    62% not-likely while the 120-test is balanced.  NO image augmentation (this
    project's augment track record is negative).
  * Early stopping on eval_loss (patience).

Usage (grace, 6 GPUs):
  CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 python train_medgemma27b_tc.py \
     --model-id /data/home/gracechen/.cache/huggingface/hub/models--google--medgemma-27b-it/snapshots/2d3e00ea38b50018bf5dd3aa1009457cd2d5a48f \
     --out outputs/mg27_lora_tc
"""
from __future__ import annotations
import argparse, ast, json, collections
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset

_HERE = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True, help="local 27b snapshot dir or HF id")
    p.add_argument("--data", default=str(_HERE / "DATA/sft_train_tc.jsonl"))
    p.add_argument("--photos", default=None,
                   help="dir of <id>.jpg; default resolves COT_EYES_ROOT or repo path")
    p.add_argument("--out", default="outputs/mg27_lora_tc")
    p.add_argument("--epochs", type=float, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-seq-len", type=int, default=1536)
    p.add_argument("--oversample", type=int, default=2,
                   help="repeat factor for minority (borderline,likely) train rows")
    p.add_argument("--class-weighted", action="store_true",
                   help="GR-LoRA style: per-sample loss x inverse-freq class weight (clip+normalize)")
    p.add_argument("--w-max", type=float, default=10.0, help="class weight clip")
    p.add_argument("--bl-mult", type=float, default=1.0,
                   help="extra multiplier on the borderline class weight, applied AFTER "
                        "mean-normalization (nl/lk left unchanged). 2.0 = double borderline penalty.")
    p.add_argument("--bl-renorm", action="store_true",
                   help="after applying --bl-mult, re-normalize the 3 weights back to mean=1 "
                        "so overall loss scale (and effective LR) is unchanged.")
    p.add_argument("--exclude-ids", default=None,
                   help="path to a file of image ids (one per line) to DROP from the train split")
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--eval-steps", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=-1, help="cap total steps")
    p.add_argument("--smoke", action="store_true", help="no eval/save/early-stop; few steps")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def resolve_photos(arg):
    import os
    if arg:
        return Path(arg)
    root = os.environ.get("COT_EYES_ROOT")
    if root:
        p = Path(root) / "Dataset" / "all" / "photos"
        if p.exists():
            return p
    return _HERE.parent / "COT_Eyes" / "Dataset" / "all" / "photos"


class TCDataset(Dataset):
    """Builds chat messages [system, user(image+prompt), assistant(target)].
    Oversamples minority classes on the train split only."""
    def __init__(self, jsonl, photos, split, oversample=1, exclude_ids=None):
        rows = [json.loads(l) for l in open(jsonl)]
        rows = [r for r in rows if r["split"] == split]
        if split == "train" and exclude_ids:
            n0 = len(rows)
            rows = [r for r in rows if str(r["image"]) not in exclude_ids]
            print(f"TCDataset[train]: dropped {n0 - len(rows)} excluded ids (of {len(exclude_ids)} requested)")
        if split == "train" and oversample > 1:
            extra = []
            for r in rows:
                if r["true_diagnosis"] in ("borderline", "likely"):
                    extra += [r] * (oversample - 1)
            rows = rows + extra
        self.rows = rows
        self.photos = Path(photos)
        self.class_counts = collections.Counter(r["true_diagnosis"] for r in rows)
        print(f"TCDataset[{split}]: {len(rows)} samples  dx={dict(self.class_counts)}")

    DX2ID = {"not likely": 0, "borderline": 1, "likely": 2}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.photos / f"{r['image']}.jpg").convert("RGB")
        sys_user = [
            {"role": "system", "content": [{"type": "text", "text": r["system"]}]},
            {"role": "user", "content": [{"type": "image"},
                                          {"type": "text", "text": r["prompt"]}]},
        ]
        full = sys_user + [{"role": "assistant",
                            "content": [{"type": "text", "text": r["target"]}]}]
        return {"image": img, "sys_user": sys_user, "full": full,
                "class_id": self.DX2ID[r["true_diagnosis"]]}


class TCCollator:
    """Tokenize the full conversation; mask everything before the assistant answer
    by measuring the prefix (system+user+assistant-header) token length per example.
    No hardcoded special-token ids."""
    def __init__(self, processor, max_length):
        self.p = processor
        self.max_length = max_length

    def __call__(self, exs):
        images = [[e["image"]] for e in exs]
        full_txt = [self.p.apply_chat_template(e["full"], add_generation_prompt=False,
                                               tokenize=False) for e in exs]
        batch = self.p(text=full_txt, images=images, padding=True, truncation=True,
                       max_length=self.max_length, return_tensors="pt")
        labels = batch["input_ids"].clone()
        pad_id = self.p.tokenizer.pad_token_id
        for i, e in enumerate(exs):
            # prefix = system+user + assistant generation header (no answer yet)
            pre_txt = self.p.apply_chat_template(e["sys_user"], add_generation_prompt=True,
                                                 tokenize=False)
            pre = self.p(text=[pre_txt], images=[[e["image"]]], return_tensors="pt",
                         truncation=True, max_length=self.max_length)
            plen = int(pre["input_ids"].shape[1])
            labels[i, :plen] = -100
        labels[labels == pad_id] = -100
        batch["labels"] = labels
        batch["class_id"] = torch.tensor([e["class_id"] for e in exs], dtype=torch.long)
        return batch


def make_weighted_trainer(base_cls):
    """GR-LoRA style: per-sample loss weighted by inverse-freq class weight.
    Manual shifted token CE (reduction='none') -> per-sample mean over valid
    tokens -> x class weight -> batch mean.  base_cls = transformers.Trainer."""
    class WeightedTrainer(base_cls):
        def __init__(self, *a, class_weights=None, **kw):
            super().__init__(*a, **kw)
            # registered as buffer-like tensor; moved to device lazily in compute_loss
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            class_id = inputs.pop("class_id")
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits  # (B, T, V)
            # shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            B, Tm1, V = shift_logits.shape
            tok_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, V), shift_labels.view(-1),
                ignore_index=-100, reduction="none").view(B, Tm1)
            valid = (shift_labels != -100).float()
            # per-sample mean over its own assistant tokens
            per_sample = (tok_loss * valid).sum(1) / valid.sum(1).clamp_min(1.0)
            w = self.class_weights.to(per_sample.device)[class_id]
            loss = (per_sample * w).mean()
            return (loss, outputs) if return_outputs else loss
    return WeightedTrainer


def main():
    from transformers import (AutoModelForImageTextToText, AutoProcessor,
                              TrainingArguments, Trainer, EarlyStoppingCallback)
    from peft import LoraConfig, get_peft_model
    args = parse_args()
    torch.manual_seed(args.seed)
    photos = resolve_photos(args.photos)
    print("photos:", photos)

    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False
    # freeze vision tower
    for attr in ("vision_tower", "vision_model"):
        if hasattr(model, attr):
            for prm in getattr(model, attr).parameters():
                prm.requires_grad_(False)
            print(f"froze {attr}")

    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                      lora_dropout=args.lora_dropout, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    exclude_ids = None
    if args.exclude_ids:
        exclude_ids = {l.strip() for l in open(args.exclude_ids) if l.strip()}
        print(f"excluding {len(exclude_ids)} ids from train: {sorted(exclude_ids)}")
    train_ds = TCDataset(args.data, photos, "train", oversample=args.oversample,
                         exclude_ids=exclude_ids)
    val_ds = TCDataset(args.data, photos, "val", oversample=1)
    collator = TCCollator(processor, args.max_seq_len)

    common = dict(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_ratio=0.05, weight_decay=0.01, max_grad_norm=1.0,
        lr_scheduler_type="cosine", logging_steps=5, report_to="none",
        remove_unused_columns=False, label_names=["labels"],
        dataloader_num_workers=2, seed=args.seed, max_steps=args.max_steps,
    )
    cbs = []
    if args.smoke:
        targs = TrainingArguments(eval_strategy="no", save_strategy="no",
                                  load_best_model_at_end=False, **common)
    else:
        targs = TrainingArguments(
            eval_strategy="steps", eval_steps=args.eval_steps,
            save_strategy="steps", save_steps=args.eval_steps, save_total_limit=2,
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False, **common)
        cbs = [EarlyStoppingCallback(early_stopping_patience=args.patience)]
    if args.class_weighted:
        # inverse-freq, clip w_max, normalize so mean weight == 1
        counts = train_ds.class_counts
        N = sum(counts.values())
        raw = {}
        for dx, idx in TCDataset.DX2ID.items():
            nc = counts.get(dx, 0)
            raw[idx] = min(N / nc, args.w_max) if nc else args.w_max
        mean_w = sum(raw.values()) / len(raw)
        cw = [raw[i] / mean_w for i in range(len(raw))]
        # extra borderline penalty on top of normalization (index 1 = borderline)
        cw[TCDataset.DX2ID["borderline"]] *= args.bl_mult
        if args.bl_renorm:
            m = sum(cw) / len(cw)
            cw = [x / m for x in cw]   # back to mean=1 so overall loss scale (LR) unchanged
        class_weights = torch.tensor(cw, dtype=torch.float32)
        print(f"class weights (nl/bl/lk, mean-normalized, bl_mult={args.bl_mult}, "
              f"renorm={args.bl_renorm}): {[round(x,3) for x in cw]}")
        WeightedTrainer = make_weighted_trainer(Trainer)
        trainer = WeightedTrainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=(None if args.smoke else val_ds),
                      processing_class=processor, data_collator=collator,
                      callbacks=cbs, class_weights=class_weights)
    else:
        trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=(None if args.smoke else val_ds),
                      processing_class=processor, data_collator=collator, callbacks=cbs)
    trainer.train()
    trainer.save_model(args.out)
    processor.save_pretrained(args.out)
    print("saved LoRA adapter ->", args.out)


if __name__ == "__main__":
    main()

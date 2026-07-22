#!/usr/bin/env python3
"""LoRA fine-tune MedGemma-27B for glaucoma chain-of-thought diagnosis.

Each training sample is a fundus image plus a prompt carrying the Stage-1
indicator block (four measurements + a diagnosis probability); the target is a
six-step reasoning chain ending in a binary diagnosis. LoRA is placed on the
language-model attention projections only; the vision tower stays frozen and the
loss is computed on the assistant tokens alone.

    CUDA_VISIBLE_DEVICES=0,1 python train.py \
        --model-id google/medgemma-27b-it \
        --data data/sft_train.jsonl --photos data/images --out outputs/mg27_lora
"""
import argparse
import collections
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

DX2ID = {"not likely": 0, "likely": 1}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True, help="local snapshot dir or HF id")
    p.add_argument("--data", required=True, help="SFT jsonl (see README for schema)")
    p.add_argument("--photos", required=True, help="dir of <image>.jpg")
    p.add_argument("--out", default="outputs/mg27_lora")
    p.add_argument("--epochs", type=float, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-seq-len", type=int, default=1536)
    p.add_argument("--eval-steps", type=int, default=40)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class SFTDataset(Dataset):
    """Reads the jsonl and builds [system, user(image+prompt), assistant(target)]
    chat messages for one split."""
    def __init__(self, jsonl, photos, split):
        self.rows = [r for r in map(json.loads, open(jsonl)) if r["split"] == split]
        self.photos = Path(photos)
        self.class_counts = collections.Counter(r["true_diagnosis"] for r in self.rows)
        print(f"[{split}] {len(self.rows)} samples  dx={dict(self.class_counts)}")

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
                "class_id": DX2ID[r["true_diagnosis"]]}


class Collator:
    """Tokenize the full conversation and mask everything before the assistant
    answer by measuring the prompt prefix length per example (no hardcoded
    special-token ids, which keeps this robust across processor versions)."""
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
            pre_txt = self.p.apply_chat_template(e["sys_user"], add_generation_prompt=True,
                                                 tokenize=False)
            pre = self.p(text=[pre_txt], images=[[e["image"]]], return_tensors="pt",
                         truncation=True, max_length=self.max_length)
            labels[i, :int(pre["input_ids"].shape[1])] = -100
        labels[labels == pad_id] = -100
        batch["labels"] = labels
        batch["class_id"] = torch.tensor([e["class_id"] for e in exs], dtype=torch.long)
        return batch


def make_weighted_trainer(base_cls):
    """Per-sample loss scaled by an inverse-frequency class weight, so the
    minority diagnosis is not drowned out."""
    class WeightedTrainer(base_cls):
        def __init__(self, *a, class_weights=None, **kw):
            super().__init__(*a, **kw)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            class_id = inputs.pop("class_id")
            labels = inputs.pop("labels")
            logits = model(**inputs).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            b, tm1, v = shift_logits.shape
            tok_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, v), shift_labels.view(-1),
                ignore_index=-100, reduction="none").view(b, tm1)
            valid = (shift_labels != -100).float()
            per_sample = (tok_loss * valid).sum(1) / valid.sum(1).clamp_min(1.0)
            w = self.class_weights.to(per_sample.device)[class_id]
            loss = (per_sample * w).mean()
            return (loss, logits) if return_outputs else loss
    return WeightedTrainer


def class_weights(counts):
    n = sum(counts.values())
    raw = {DX2ID[k]: n / counts[k] for k in counts}
    mean_w = sum(raw.values()) / len(raw)
    w = [1.0] * len(DX2ID)
    for idx, val in raw.items():
        w[idx] = val / mean_w
    return torch.tensor(w, dtype=torch.float32)


def main():
    from transformers import (AutoModelForImageTextToText, AutoProcessor,
                              TrainingArguments, Trainer, EarlyStoppingCallback)
    from peft import LoraConfig, get_peft_model
    args = parse_args()
    torch.manual_seed(args.seed)

    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False
    for attr in ("vision_tower", "vision_model"):
        if hasattr(model, attr):
            for prm in getattr(model, attr).parameters():
                prm.requires_grad_(False)

    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                      lora_dropout=args.lora_dropout, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = SFTDataset(args.data, args.photos, "train")
    val_ds = SFTDataset(args.data, args.photos, "val")
    collator = Collator(processor, args.max_seq_len)
    cw = class_weights(train_ds.class_counts)
    print("class weights:", {k: round(cw[DX2ID[k]].item(), 3) for k in train_ds.class_counts})

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_ratio=args.warmup_ratio, weight_decay=0.01, max_grad_norm=1.0,
        lr_scheduler_type="cosine", logging_steps=5, report_to="none",
        remove_unused_columns=False, label_names=["labels"],
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, dataloader_num_workers=2, seed=args.seed,
    )
    WeightedTrainer = make_weighted_trainer(Trainer)
    trainer = WeightedTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=processor, data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        class_weights=cw)
    trainer.train()
    trainer.save_model(args.out)
    processor.save_pretrained(args.out)
    print("saved LoRA adapter ->", args.out)


if __name__ == "__main__":
    main()

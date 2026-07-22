# Glaucoma-CoT

Glaucoma chain-of-thought diagnosis from a single fundus photograph.

A fundus image is first passed through a frozen RETFound backbone with trained
pooling and prediction heads (Stage 1), which produce four intermediate clinical
measurements plus a diagnosis probability. These are written into a prompt and
handed to a LoRA-fine-tuned MedGemma-27B (Stage 2), which returns a six-step
clinical reasoning chain ending in a binary diagnosis.

```
fundus image
    -> RETFound (frozen) + pooling + heads   ->  CDR, ISNT rim order,
                                                  per-quadrant rim status,
                                                  glaucomatous signs, dx probability
    -> MedGemma-27B + LoRA                    ->  six-step reasoning chain + diagnosis
```

## Layout

```
retfound_finetune/       Stage 1: train the RETFound heads, then predict indicators
medgemma27b_finetune/    Stage 2: LoRA fine-tune MedGemma-27B on the reasoning data
inference.py             run the full image -> reasoning-chain pipeline
```

## Install

```
pip install -r requirements.txt
```

You also need the RETFound MAE ViT-L/16 (`natureCFP`) weights and a MedGemma-27B
snapshot. Point `RETFOUND_CKPT` at the former and pass the latter with `--model-id`.

## Stage 1 — RETFound heads

The backbone stays frozen; a pooling module and five prediction heads are trained
on top of it. Five poolings (`baseline`, `tier1`, `tier2a`, `tier2b`, `tier3`) and
two head kinds (`linear`, `mlp`) are available; `tier2a` with a `linear` head is
used in the paper.

```
cd retfound_finetune
python train.py --pool tier2a --head linear --out outputs/tier2a_linear
python predict_indicators.py \
    --ckpt outputs/tier2a_linear/best.pt \
    --images 'data/images/*.jpg' \
    --out predicted_indicators.json
```

Training expects four CSV manifests (`isnt`, `cdr`, `signs`, `dx`) and an image
folder; see `dataset.py` for the column names and set the paths with the
`DATA_ROOT` / `*_MANIFEST` / `PHOTOS_DIR` environment variables.

## Stage 2 — MedGemma-27B

LoRA is placed on the language-model attention projections; the vision tower is
frozen and the loss is computed on the assistant tokens only.

```
cd medgemma27b_finetune
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --model-id google/medgemma-27b-it \
    --data data/sft_train.jsonl \
    --photos data/images \
    --out outputs/mg27_lora
```

Each line of `sft_train.jsonl` is one sample:

```json
{
  "image": "<image id, resolves to <photos>/<id>.jpg>",
  "system": "<system prompt>",
  "prompt": "<query + Stage-1 indicator block>",
  "target": "<six-step reasoning chain + diagnosis>",
  "true_diagnosis": "likely | not likely",
  "split": "train | val"
}
```

## Inference

```
python inference.py \
    --stage1-ckpt retfound_finetune/outputs/tier2a_linear/best.pt \
    --model-id google/medgemma-27b-it \
    --adapter medgemma27b_finetune/outputs/mg27_lora \
    --image path/to/fundus.jpg
```

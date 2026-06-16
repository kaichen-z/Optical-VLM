# Qwen3.5-35B-A3B In-Context Learning on Glaucoma Fundus Images

**Task**: Use Qwen3.5-35B-A3B (MoE: 35B total / 3B active) to do **6-shot in-context learning** on the COT_Eyes glaucoma dataset, then compare against Grace's previously-tested MedGemma / Qwen2.5-VL results.

**Why a separate folder**: Keeps my own work (yuzhench) cleanly isolated from Grace's `COT_Eyes/` codebase.

**Inspired by**: `yuzhench@100.109.238.5:/home/yuzhench/Desktop/SHBI-Final-Project/` — that project's `src/{models,prompts,inference}.py` already established the right Qwen3.5-VL loading recipe (correct HF class, processor padding side, `enable_thinking=False`, `qwen_vl_utils.process_vision_info`).

---

## What this folder contains

```
qwen35_icl_glaucoma/
├── README.md                     # this file
├── requirements.txt              # Python deps (mirrors SHBI project)
├── run_qwen35_incontext.py       # main inference script
├── compute_metrics.py            # standalone metrics tool (rerun on saved pickles)
└── outputs/                      # results land here (one subfolder per run)
```

---

## Where to run

The model needs **~70 GB VRAM in bf16** (or ~18 GB at 4-bit with bnb). Two real options:

| Machine | GPUs | Recommended config |
|---|---|---|
| **`yuzhench@100.109.238.5`** | 2× RTX PRO 6000 Blackwell (97 GB × 2) | bf16, `device_map="auto"` for tensor parallel |
| **`<user>@<lab-server>`** (lab) | A6000 48 GB single | `--load-in-4bit` |
| Local laptop | (likely no GPU) | not feasible |

**Recommended: deploy to `100.109.238.5`** because the env (transformers ≥ 4.57, qwen-vl-utils, model cache) is already set up there from the SHBI-Final-Project.

---

## How to run

### Option A — Run on `100.109.238.5` (recommended)

```bash
# 1) Sync this whole folder (code + DATA/) to the GPU machine
rsync -avh --progress \
  /home/yuzhench/Desktop/Research/Harvard_AI/EYE/qwen35_icl_glaucoma/ \
  yuzhench@100.109.238.5:/home/yuzhench/Desktop/qwen35_icl_glaucoma/

# 2) SSH over
ssh yuzhench@100.109.238.5
cd ~/Desktop/qwen35_icl_glaucoma

# 3) Activate the conda env that has transformers >= 4.57 + qwen-vl-utils + bnb
#    (this is the env SHBI-Final-Project actually uses for its 35B-A3B runs)
conda activate llm     # OR: source /home/yuzhench/miniforge3/envs/llm/bin/activate

# 4) Point HF cache at the EXISTING 67 GB model checkpoint
#    (otherwise transformers will redownload Qwen3.5-35B-A3B from scratch!)
export HF_HOME=/home/yuzhench/Desktop/SHBI-Final-Project/models_weight

# 5) Smoke test (5 samples, ~1 min)
python run_qwen35_incontext.py --num-samples 5

# 6) Full 120-sample eval (~15-25 min on 2× RTX PRO 6000)
python run_qwen35_incontext.py

# 7) Back on laptop, pull results
rsync -avh --progress \
  yuzhench@100.109.238.5:/home/yuzhench/Desktop/qwen35_icl_glaucoma/outputs/ \
  /home/yuzhench/Desktop/Research/Harvard_AI/EYE/qwen35_icl_glaucoma/outputs/
```

**Why both `conda activate llm` AND `HF_HOME=...` are required:**
- `conda activate llm` — gives Python the right transformers/peft/qwen-vl-utils versions
- `HF_HOME=...` — tells HF library to use the EXISTING 67 GB model cache instead of redownloading

You can wrap them into a one-liner if you prefer:
```bash
HF_HOME=/home/yuzhench/Desktop/SHBI-Final-Project/models_weight \
  /home/yuzhench/miniforge3/envs/llm/bin/python run_qwen35_incontext.py --num-samples 5
```

### Option B — Run on lab server (`<lab-server>`)

Need to first push code + data, then run with `--load-in-4bit`. Lab server may not have transformers ≥ 4.57 — check first.

---

## Model & in-context setup

- **Model**: `Qwen/Qwen3.5-35B-A3B` (default; override with `--model-name`).
- **In-context examples**: 6 worked examples from `In Context Training Dataset/` — IDs `026, 085, 307, 499, 684, 685`. Each shows the model the full 6-step CoT format we want.
- **Test set**: 727 images from `In Context Test Dataset/` (727 = 733 total − 6 in-context).
- **Prompt format**: system prompt + 6 (user image / assistant CoT) turns + final user image. Uses Qwen chat template with `enable_thinking=False` to suppress `<think>` wrappers.
- **Decoding**: greedy (`do_sample=False`, `max_new_tokens=768`).

---

## Outputs

Each run creates `outputs/<run_tag>/` containing:
- `config.json` — full run parameters (model, ICL IDs, prompts)
- `predictions.jsonl` — one line per test sample (streaming-safe)
- `predictions.pkl` — same data as a pandas DataFrame (matches Grace's pickle format)
- `metrics.json` — accuracy, macro F1, per-class F1, distributions, timing, peak VRAM

To recompute metrics from a saved pickle:

```bash
python compute_metrics.py outputs/<run_tag>/predictions.pkl
```

---

## Comparison baselines (from Grace's Zero Shot / In-Context experiments)

These are the numbers I'm trying to beat with Qwen3.5-35B-A3B:

| Method | Accuracy | Macro F1 |
|---|---|---|
| Always-predict-`not_likely` | 62.8% | 0.257 |
| MedGemma-4b-it zero-shot + CoT | 64.1% | 0.350 |
| Qwen2.5-VL-7B zero-shot + CoT | 54.4% | 0.336 |
| MedGemma-4b-it + 6 in-context examples | **63.8%** | **0.516** |
| Qwen2.5-VL-7B + 6 in-context examples | 59.3% | 0.339 |
| **Qwen3.5-35B-A3B + 6 in-context examples** | **(this experiment)** | — |

The interesting comparison is whether scaling Qwen 7B → 35B-A3B lifts in-context performance up to/past MedGemma.

---

## Caveats & honest notes

- **`Qwen/Qwen3.5-35B-A3B` may be a text-primarily model**. The SHBI project ran it on TextVQA, but per the official Qwen blog (Feb 2026) Qwen3.5 is reported as text-focused with optional vision support via the unified head. If multimodal support fails at runtime, fall back to `--model-name Qwen/Qwen3-VL-30B-A3B-Instruct` (which is unambiguously multimodal and similar in scale).
- **Inference time**: a single sample with 6 image-bearing in-context examples + 1 test image involves processing 7 images through the vision encoder. Expect **~5-15 seconds per sample on 2× RTX PRO 6000**, so **727 samples ≈ 1-3 hours**.
- **Greedy decoding** chosen for reproducibility. You can swap to sampling for a different operating point.
- **Parsing diagnosis**: the script extracts the label from `Diagnosis Classification: <X>` line; falls back to keyword scanning if that line is missing. If the model frequently produces labels in unexpected formats, refine `parse_diagnosis()`.

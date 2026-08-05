<div align="center">
<h1>Learning Ophthalmologist Clinical Reasoning for Glaucoma Diagnosis from Fundus Images</h1>

A reasoning-driven vision–language framework that writes a <ins>**six-step clinical chain-of-thought**</ins><br>
before making the glaucoma decision — plus <ins>**BIOVLM**</ins>, the first expert-annotated<br>
glaucoma reasoning dataset (1,077 fundus photographs).

<a href="https://www.researchgate.net/publication/410997386_Learning_Ophthalmologist_Clinical_Reasoning_for_Glaucoma_Diagnosis_from_Fundus_Images"><img src="https://img.shields.io/badge/Paper-npj%20Digital%20Medicine-b31b1b" alt="Paper"></a>
<a href="https://glaucoma-cot.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&user=8S6_34oAAAAJ&sortby=pubdate&citation_for_view=8S6_34oAAAAJ:UxriW0iASnsC"><img src="https://img.shields.io/badge/Google_Scholar-4285F4" alt="Google Scholar"></a>
<a href="https://github.com/kaichen-z/Optical-VLM/stargazers"><img src="https://img.shields.io/github/stars/kaichen-z/Optical-VLM?style=flat&color=blue" alt="Stars"></a>

**[Harvard Ophthalmology AI Lab, Harvard Medical School](https://ophai.hms.harvard.edu/)**;
**[Media Lab, MIT](https://www.media.mit.edu/)**;
**[HKUST](https://cse.hkust.edu.hk/)**;
**[University of Louisiana at Lafayette](https://louisiana.edu/)**;
**[Istanbul Medipol University](https://www.medipol.edu.tr/en)**

[Kaichen Zhou](https://kaichen-z.github.io/)\*, Yuzhen Chen\*, Elif Yildiz\*, Min Shi, David Dai, [Grace Chen](https://gracee-chen.github.io/), Jiale Zheng, He Wang, [Fangneng Zhan](https://fnzhan.com/), Chhavi Saini, Lucy Q. Shen, Yike Guo, [Paul Pu Liang†](https://pliang279.github.io/), [Mengyu Wang†](https://wang.hms.harvard.edu/team/dr-wang/)

**(\*: Equal contribution&nbsp;&nbsp;†: Corresponding authors)**
</div>

```bibtex
@article{zhou2026glaucoma,
  title   = {Learning Ophthalmologist Clinical Reasoning for Glaucoma Diagnosis from Fundus Images},
  author  = {Zhou, Kaichen and Chen, Yuzhen and Yildiz, Elif and Shi, Min and Dai, David
             and Chen, Grace and Zheng, Jiale and Wang, He and Zhan, Fangneng and Saini, Chhavi
             and Shen, Lucy Q. and Guo, Yike and Liang, Paul Pu and Wang, Mengyu},
  journal = {npj Digital Medicine (under review)},
  year    = {2026}
}
```

> **Which branch do I want?** The clean camera-ready code for the paper lives on the
> [**`camera_ready_code`**](https://github.com/kaichen-z/Optical-VLM/tree/camera_ready_code) branch —
> every command in this README refers to it. This `main` branch keeps an earlier exploratory
> three-class codebase (different module names, three-level *not likely / borderline / likely*
> label set, older result); it is preserved for history in
> [`reports/legacy-3class-experiment.md`](reports/legacy-3class-experiment.md) and is **not** the
> code the paper reports.

## Overview

Ophthalmologists do not classify a fundus photograph in one step — they evaluate optic-nerve-head
characteristics in sequence and only then commit to a diagnosis. Existing AI screening systems skip
that process. This repository implements a framework that reproduces it explicitly:

1. **Stage 1 — RETFound indicator heads.** A **frozen** RETFound ViT-L/16 backbone with trained
   pooling + prediction heads turns the image into *structured clinical evidence*: vertical /
   transverse cup-to-disc ratio, ISNT rim ordering, per-quadrant rim status, six glaucomatous
   signs, and an initial diagnostic probability.
2. **Stage 2 — MedGemma-27B + LoRA.** The image **and** that evidence block are handed to a
   LoRA-fine-tuned MedGemma-27B, which writes a **six-step reasoning report** ending in a binary
   glaucoma / non-glaucoma decision.

The backbone is the same one used by conventional screening pipelines, so the gain comes from the
added clinical reasoning, not from a stronger visual encoder.

```
fundus image
    │
    ├─► RETFound (frozen) + pooling + heads      ──►  CDR (vertical / transverse)
    │                                                 ISNT rim order
    │                                                 per-quadrant rim status
    │                                                 6 glaucomatous signs
    │                                                 diagnostic probability
    ▼
image + structured clinical evidence
    │
    ▼
MedGemma-27B + LoRA   ──►  six-step reasoning report  ──►  glaucoma / non-glaucoma
```

### The six-step chain-of-thought

| Step | Content |
|---|---|
| 1 | **Image quality** — focus, contrast, artifacts, disc-margin clarity |
| 2 | **CDR evaluation** — vertical & transverse cup-to-disc ratio vs. the glaucomatous cutoff |
| 3 | **ISNT rule** — rank inferior / superior / nasal / temporal rim widths, flag violations |
| 4 | **Glaucomatous signs** — notching, RNFL defect, bayoneting, β-zone / peripapillary atrophy, disc hemorrhage |
| 5 | **Structural summary** — integrate the findings, weigh them against the automated probability |
| 6 | **Final classification** — binary decision with justification |

## Results

All numbers on the 160-image test split (80 glaucoma / 80 non-glaucoma).

**Report correctness & agreement with expert reports**

| Metric | **Ours** | Qwen3.5-VL | Claude | GPT-5.5 |
|---|:---:|:---:|:---:|:---:|
| CDR MAE ↓ | **0.070** | 0.102 | 0.135 | 0.127 |
| ISNT Kendall distance ↓ | **1.73** | 2.07 | 2.27 | n/a |
| Rim-level macro-F1 ↑ | **0.656** | 0.512 | 0.510 | 0.373 |
| Signs macro-F1 ↑ | **0.613** | 0.501 | 0.465 | 0.375 |
| BERTScore-F1 ↑ | **0.874** | 0.867 | 0.869 | 0.866 |
| ROUGE-Lsum ↑ | 0.454 | 0.416 | **0.461** | 0.439 |

**Diagnosis (balanced accuracy)**

| Model | Bal. Acc. | 95% CI |
|---|:---:|:---:|
| **Ours (VLM)** | **94.7%** | 90.6–98.1 |
| RetiZero | 93.6% | 88.1–98.0 |
| RETFound | 93.1% | 87.6–97.5 |
| DINOv2-L | 84.7% | 77.1–91.2 |
| Qwen3.5-VL | 84.3% | 78.8–89.5 |
| Claude | 74.4% | 67.5–81.0 |
| GPT-5.5 | 71.9% | 65.1–78.2 |

**Ablations — the reasoning stage is what carries the diagnosis**

| Configuration | Bal. Acc. | Sens. | Spec. |
|---|:---:|:---:|:---:|
| Full framework | **94.69%** | 92.88% | 96.50% |
| Abl. 1 — image only | 83.75% | 97.50% | 70.00% |
| Abl. 2 — image + diagnosis probability only | 70.63% | 98.75% | 42.50% |

| Leave-one-indicator-out | Bal. Acc. |
|---|:---:|
| Ours (all indicators) | **94.69%** |
| − per-quadrant rim status | 94.38% |
| − ISNT rim ordering | 94.37% |
| − glaucomatous signs | 93.75% |
| − cup-to-disc ratio | 92.50% |

CDR is the single most informative cue; every indicator still contributes complementary signal.

## BIOVLM dataset

1,077 color fundus photographs from the public **LAG** and **Papila** datasets, each paired with a
complete expert-authored six-step reasoning report (321.6 ± 46.3 words) and a binary label.

| Characteristic | Value (glaucoma / non-glaucoma) |
|---|---|
| Fundus images | 1,077 (430 / 647) |
| LAG database | 659 (343 / 316) |
| Papila dataset | 418 (87 / 331) |
| Training split | 825 (315 / 510) |
| Validation split | 92 (35 / 57) |
| Test split | 160 (80 / 80) |
| Avg. report length | 321.6 ± 46.3 words |

Six reasoning indicators are annotated per image (steps 2–4 of the CoT): vertical & transverse CDR
(continuous), ISNT rim ranking, per-quadrant rim abnormality (3 ordinal levels × 4 quadrants), and
six glaucomatous structural signs.

**Access.** The source photographs come from the two public datasets and must be obtained from
their original providers under their own licences:

| Source | Link |
|---|---|
| LAG (Large-scale Attention-based Glaucoma) | https://github.com/smilell/AG-CNN |
| Papila | https://figshare.com/articles/dataset/PAPILA/14798004 |

The expert six-step CoT annotations, the indicator labels and the train / val / test split ids are
**not yet released**; they will be posted on the [project page](https://glaucoma-cot.github.io/)
once the paper's review process finishes.

## Install

```bash
git clone -b camera_ready_code https://github.com/kaichen-z/Optical-VLM.git
cd Optical-VLM
pip install -r requirements.txt          # pinned to the environment the paper was run in
pip install -r evaluation/requirements-eval.txt   # only for report-quality metrics
```

`requirements.txt` pins the CUDA 12.8 torch build; install the matching wheels from
[pytorch.org](https://pytorch.org) if your setup differs. `transformers==5.1.0` is a hard floor —
the code uses the `dtype=` argument, which older releases do not accept.

## Base models

Both stages are fine-tuned **on top of public pretrained checkpoints**. Our own trained weights are
not released at this time; the checkpoints below are what you fine-tune *from*.

| Stage | Base checkpoint | Notes |
|---|---|---|
| Stage 1 (RETFound) | [`YukunZhou/RETFound_mae_natureCFP`](https://huggingface.co/YukunZhou/RETFound_mae_natureCFP) | MAE ViT-L/16 (`RETFound_mae_natureCFP.pth`). Non-commercial licence. |
| Stage 2 (MedGemma-27B) | [`google/medgemma-27b-it`](https://huggingface.co/google/medgemma-27b-it) | **Gated** — accept the Health AI Developer Foundations terms and authenticate first. |

```bash
huggingface-cli download YukunZhou/RETFound_mae_natureCFP
huggingface-cli download google/medgemma-27b-it        # requires accepting the gated terms
```

Point `RETFOUND_CKPT` at the RETFound `.pth`, and pass the MedGemma snapshot with `--model-id`.

## Quick Start

Once both stages are trained (below), one fundus image → one six-step report:

```bash
python inference.py \
    --stage1-ckpt retfound_finetune/outputs/tier2a_linear/best.pt \
    --model-id google/medgemma-27b-it \
    --adapter medgemma27b_finetune/outputs/mg27_lora \
    --image path/to/fundus.jpg
```

## Layout

```
retfound_finetune/       Stage 1: train the RETFound heads, then predict the indicators
medgemma27b_finetune/    Stage 2: LoRA fine-tune MedGemma-27B on the reasoning data
evaluation/              Stage-1 indicator metrics + Stage-2 diagnosis / report-quality metrics
inference.py             full image -> reasoning-chain pipeline
```

## Stage 1 — RETFound heads

The backbone stays frozen; only a pooling module and the prediction heads are trained. Five
poolings (`baseline`, `tier1`, `tier2a`, `tier2b`, `tier3`) and two head kinds (`linear`, `mlp`)
are available; **`tier2a` + `linear` is the configuration used in the paper**.

```bash
cd retfound_finetune
python train.py --pool tier2a --head linear --out outputs/tier2a_linear
python predict_indicators.py \
    --ckpt outputs/tier2a_linear/best.pt \
    --images 'data/images/*.jpg' \
    --out predicted_indicators.json
```

Training expects four CSV manifests (`isnt`, `cdr`, `signs`, `dx`) and an image folder; see
`dataset.py` for the column names, and set the paths with the `DATA_ROOT` / `*_MANIFEST` /
`PHOTOS_DIR` environment variables.

## Stage 2 — MedGemma-27B

LoRA sits on the language-model attention projections; the vision tower is frozen and the loss is
computed on the assistant tokens only.

```bash
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

Note that the Stage-1 indicators written into `prompt` are the **predicted** ones, so the model
learns to reason from noisy measurements rather than from clean ground truth.

## Evaluation

`evaluation/` scores the two axes reported in the paper — see
[`evaluation/README.md`](https://github.com/kaichen-z/Optical-VLM/blob/camera_ready_code/evaluation/README.md)
for the full metric list.

```bash
# Stage 1 — clinical indicators (CDR MAE/RMSE/r, ISNT Kendall, rim macro-F1 + QWK, signs macro-F1)
python evaluation/eval_indicators.py \
    --pred predicted_indicators.json \
    --cdr-manifest cdr_manifest.csv \
    --isnt-manifest isnt_manifest.csv \
    --signs-manifest signs_manifest.csv \
    --split test --out stage1_metrics.json

# Stage 2 (1) — parsed diagnosis: balanced accuracy / sensitivity / specificity / MCC
python evaluation/eval_diagnosis.py --pred generations.jsonl --out diagnosis_metrics.json

# Stage 2 (2) — report quality vs. the expert reference: BERTScore-F1 + ROUGE
python evaluation/eval_report.py --pred generations.jsonl --out report_metrics.json
```

`generations.jsonl` is one JSON object per line:
`{"id", "true_diagnosis": "likely"|"not likely", "generated": "<report>", "reference": "<expert report>"}`.

## Acknowledgements

Built on [RETFound](https://github.com/rmaphoh/RETFound_MAE) and
[MedGemma](https://huggingface.co/google/medgemma-27b-it); the fundus photographs come from the
[LAG](https://github.com/smilell/AG-CNN) and [Papila](https://figshare.com/articles/dataset/PAPILA/14798004)
datasets. Thanks to the authors of these works and to the ophthalmologists who authored the 1,077
expert reasoning reports.

## Licence & intended use

Research use only. This code is **not** a medical device and must not be used for clinical
decision-making. The RETFound weights carry a non-commercial licence and MedGemma is distributed
under the Health AI Developer Foundations terms; both apply to any derived checkpoint.

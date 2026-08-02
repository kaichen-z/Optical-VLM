# Evaluation

Two axes: the Stage-1 clinical indicators, and the Stage-2 report (scored two ways).

```
metrics.py            shared metric functions (numpy only)
eval_indicators.py    Stage 1: four predicted indicators vs ground truth
eval_diagnosis.py     Stage 2 (1): parsed diagnosis -> balanced acc / Sn / Sp / MCC
eval_report.py        Stage 2 (2): generated report vs expert reference -> BERTScore + ROUGE
```

`eval_report.py` needs two extra packages:

```
pip install -r requirements-eval.txt
```

## Stage 1 — clinical indicators

Scores the four indicators produced by `retfound_finetune/predict_indicators.py`
against the ground-truth manifests:

- **CDR** (vertical & horizontal): MAE, RMSE, signed bias, Pearson r.
- **ISNT rim order**: Kendall discordant-pair count (0-6), chance-scaled tau, exact-order match.
- **Per-quadrant rim status** (ordinal, normal/mild/severe): macro-F1, accuracy, quadratic
  weighted kappa, ordinal (rank-distance) MAE.
- **Glaucomatous signs**: metrics restricted to items whose ground truth is not `absent`
  (so predicting the majority class does not score), plus the FULL figures.

```
python eval_indicators.py \
    --pred predicted_indicators.json \
    --cdr-manifest cdr_manifest.csv \
    --isnt-manifest isnt_manifest.csv \
    --signs-manifest signs_manifest.csv \
    --split test --out stage1_metrics.json
```

## Stage 2 — diagnosis (method 1)

Parses the final diagnosis out of each generated report and scores it. Reports the
number of reports that could not be parsed (counted as incorrect).

```
python eval_diagnosis.py --pred generations.jsonl --out diagnosis_metrics.json
```

`generations.jsonl`: `{"id", "true_diagnosis": "likely"|"not likely", "generated": "<report>"}`.

## Stage 2 — report quality (method 2)

Compares each generated reasoning chain to the expert reference: BERTScore-F1
(semantic, raw) and ROUGE-1/2/L/Lsum (lexical).

```
python eval_report.py --pred generations.jsonl --out report_metrics.json
```

`generations.jsonl`: `{"id", "generated": "<report>", "reference": "<expert report>"}`.

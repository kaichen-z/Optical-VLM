"""Stage-2 evaluation, method 2: report quality.

Compares each generated reasoning chain to the expert reference report with a
semantic metric (BERTScore-F1) and a lexical metric (ROUGE). Requires
`bert-score` and `rouge-score` (see requirements-eval.txt).

    python eval_report.py --pred generations.jsonl --out report_metrics.json

Each line of `generations.jsonl` is {"id", "generated": "<model report>",
"reference": "<expert report>"}. BERTScore is reported raw (no baseline rescaling),
so values sit in the familiar 0.8-0.9 range; ROUGE-Lsum is the headline lexical score.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bertscore-model", default="roberta-large")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.pred)]
    cand = [str(r["generated"]) for r in rows]
    ref = [str(r["reference"]) for r in rows]
    print(f"scoring {len(rows)} report pairs")

    from bert_score import score as bert_score
    from rouge_score import rouge_scorer

    _, _, f1 = bert_score(cand, ref, model_type=args.bertscore_model,
                          lang="en", rescale_with_baseline=False, verbose=False)
    bert_f1 = float(f1.mean())

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
    acc = {k: [] for k in ("rouge1", "rouge2", "rougeL", "rougeLsum")}
    for c, r in zip(cand, ref):
        s = scorer.score(r, c)
        for k in acc:
            acc[k].append(s[k].fmeasure)
    rouge = {k: float(sum(v) / len(v)) for k, v in acc.items()}

    out = {"n": len(rows), "bertscore_f1_raw": bert_f1, **rouge}
    print(json.dumps(out, indent=2))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print("saved ->", args.out)


if __name__ == "__main__":
    main()

"""Stage-2 evaluation, method 1: diagnosis accuracy.

Parses the final diagnosis out of each generated report and scores it against the
ground truth (balanced accuracy, sensitivity, specificity, MCC, confusion matrix).

    python eval_diagnosis.py --pred generations.jsonl --out diagnosis_metrics.json

Each line of `generations.jsonl` is {"id", "true_diagnosis": "likely"|"not likely",
"generated": "<the model's report text>"}. The report ends in a
`'Diagnosis Classification': '<likely|not likely>'` field; a report that cannot be
parsed is counted as an incorrect prediction and reported under `unparsed`.
"""
import argparse
import json
import re

import metrics

DX2ID = {"not likely": 0, "likely": 1}


def parse_diagnosis(text):
    """Return 'likely' / 'not likely' / None. Reads the explicit Diagnosis
    Classification field first, then falls back to the last stated call."""
    text = str(text)
    m = re.search(r"Diagnosis Classification['\"]?\s*[:=]\s*['\"]?\s*(not likely|likely)",
                  text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # fallback: last occurrence of a clean call in the text
    hits = re.findall(r"(not likely|unlikely to have glaucoma|likely to have glaucoma|likely)",
                      text, re.IGNORECASE)
    if hits:
        last = hits[-1].lower()
        return "not likely" if "not likely" in last or "unlikely" in last else "likely"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="jsonl of generated reports")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.pred)]
    y_true, y_pred = [], []
    unparsed = []
    for r in rows:
        true = str(r.get("true_diagnosis", r.get("true"))).lower()
        gen = r.get("generated", r.get("raw", ""))
        call = parse_diagnosis(gen)
        y_true.append(DX2ID[true])
        if call is None:
            unparsed.append(r.get("id"))
            y_pred.append(1 - DX2ID[true])          # unparsed -> counted wrong
        else:
            y_pred.append(DX2ID[call])

    res = metrics.classification(y_true, y_pred)
    res["unparsed"] = len(unparsed)
    res["unparsed_ids"] = unparsed
    print(json.dumps({k: v for k, v in res.items() if k != "unparsed_ids"}, indent=2))
    if unparsed:
        print(f"unparsed: {len(unparsed)} -> {unparsed}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
        print("saved ->", args.out)


if __name__ == "__main__":
    main()

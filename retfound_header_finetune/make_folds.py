"""
K-fold OOF setup. Build a single id->fold map over the 613 TRAIN ids (seeded,
shared by all 3 heads), then for each head's manifest and each fold f write a
fold-manifest containing ONLY the 613 train rows, with fold-f rows marked
split='test' and the other 4 folds marked split='train' (original 120 test
rows are dropped). Each head trainer, run on a fold-manifest, then trains on
4 folds and dumps test_preds.npz for the held-out fold f.

Outputs: oof/folds.json (id->fold) + oof/{cdr,isnt,signs}_fold{f}.csv
"""
import csv, json, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
OOF = HERE / "oof"; OOF.mkdir(exist_ok=True)
K = 5
SEED = 42
MANIFESTS = {
    "cdr":   HERE / "cdr_manifest.csv",
    "isnt":  HERE / "isnt_manifest.csv",
    "signs": HERE / "signs_manifest_clean.csv",
}


def main():
    # train ids = intersection across the 3 manifests (should be the 613)
    rows = {k: list(csv.DictReader(open(p))) for k, p in MANIFESTS.items()}
    train_ids = {k: {r["id"] for r in rows[k] if r["split"] == "train"} for k in rows}
    common = sorted(set.intersection(*train_ids.values()))
    print("train ids per manifest:", {k: len(v) for k, v in train_ids.items()},
          "| common:", len(common))

    # seeded shuffle -> 5 contiguous folds (shared id->fold for all heads)
    rng = random.Random(SEED)
    ids = common[:]; rng.shuffle(ids)
    id2fold = {sid: (i % K) for i, sid in enumerate(ids)}   # round-robin over shuffled order
    (OOF / "folds.json").write_text(json.dumps(id2fold))
    from collections import Counter
    print("fold sizes:", dict(Counter(id2fold.values())))

    # write fold manifests per head (only the common train rows)
    for k, p in MANIFESTS.items():
        hdr = rows[k][0].keys()
        crows = [r for r in rows[k] if r["id"] in id2fold]
        for f in range(K):
            out = OOF / f"{k}_fold{f}.csv"
            with open(out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(hdr)); w.writeheader()
                for r in crows:
                    rr = dict(r)
                    rr["split"] = "test" if id2fold[r["id"]] == f else "train"
                    w.writerow(rr)
        print(f"{k}: wrote {K} fold manifests ({len(crows)} rows each)")
    print("done ->", OOF)


if __name__ == "__main__":
    main()

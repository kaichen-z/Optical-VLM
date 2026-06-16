"""
TRI-task dataset: ISNT-order + per-quadrant rim ABN + CDR verdict, on ONE image.

Joins the TWO manifests by id (verified: identical 733 ids, identical split,
train 613 / test 120, 0 split mismatches):
  - isnt_manifest.csv : rim_i/s/n/t (ranking) + abn_i/s/n/t (3-class per quadrant)
  - cdr_manifest.csv  : cdr_v_verdict / cdr_h_verdict (3-class qualitative CDR)

Returns (image, rim[4] float, abn[4] long, cdr[2] long, id, dataset).
  rim / abn order = (Inferior, Superior, Nasal, Temporal)  [step3]
  cdr order       = (vertical, horizontal)                  [step2]
  abn / cdr levels = 0 normal / 1 (mild|borderline) / 2 (severe|abnormal)

Split policy mirrors dataset_order_abn (test = the 120; val carved from train,
stratified by binned MEAN rim thickness, seed=42) so rim metrics stay comparable
to the rim-only baselines. Plain 224 resize (no square, no augment) = the config
that gave the best rim/cdr single-task numbers.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

try:
    from config import PHOTOS_DIR, ISNT_MANIFEST
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import PHOTOS_DIR, ISNT_MANIFEST

_HERE = Path(__file__).resolve().parent
CDR_MANIFEST = _HERE / "cdr_manifest.csv"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224

RIM_KEYS = ("rim_i", "rim_s", "rim_n", "rim_t")
ABN_KEYS = ("abn_i", "abn_s", "abn_n", "abn_t")
RIM_NAMES = ("Inferior", "Superior", "Nasal", "Temporal")
CDR_KEYS = ("cdr_v_verdict", "cdr_h_verdict")
CDR_NAMES = ("vertical", "horizontal")
VMAP = {"normal": 0, "borderline": 1, "abnormal": 2}
N_ABN_CLASSES = 3
N_CDR_CLASSES = 3
N_Q = 4
N_CDR = 2


def _eval_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _load_cdr_verdicts(path: Path) -> dict:
    """id -> [cdr_v_level, cdr_h_level] (ints), only rows with both valid."""
    out = {}
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        if r["status"] != "ok":
            continue
        if r.get(CDR_KEYS[0]) not in VMAP or r.get(CDR_KEYS[1]) not in VMAP:
            continue
        out[r["id"]] = [VMAP[r[CDR_KEYS[0]]], VMAP[r[CDR_KEYS[1]]]]
    return out


def _load_manifest(isnt_path: Path, cdr_map: dict) -> list[dict]:
    rows = []
    for r in csv.DictReader(open(isnt_path, newline="", encoding="utf-8")):
        if r["status"] != "ok":
            continue
        if any(r[k] in ("", "None") for k in RIM_KEYS):
            continue
        if any(r.get(k, "") in ("", "None") for k in ABN_KEYS):
            continue
        if r["id"] not in cdr_map:          # need CDR labels too (join)
            continue
        rows.append({
            "id": r["id"],
            "split": r["split"],
            "dataset": r["dataset"],
            "rim": [float(r[k]) for k in RIM_KEYS],   # [I,S,N,T]
            "abn": [int(r[k]) for k in ABN_KEYS],      # [I,S,N,T]
            "cdr": list(cdr_map[r["id"]]),             # [v,h]
        })
    return rows


def _stratified_val_split(train_rows, val_frac, seed):
    """Stratify val by binned MEAN rim thickness (same as dataset_order_abn)."""
    rng = random.Random(seed)
    bins = {}
    for r in train_rows:
        b = min(int(sum(r["rim"]) / 4.0 * 10), 9)
        bins.setdefault(b, []).append(r)
    val, tr = [], []
    for b, items in bins.items():
        items = items[:]
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_frac))
        val.extend(items[:n_val])
        tr.extend(items[n_val:])
    return tr, val


class TriDataset(Dataset):
    def __init__(self, rows, photos_dir, transform):
        self.rows = rows
        self.photos_dir = Path(photos_dir)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.photos_dir / f"{r['id']}.jpg").convert("RGB")
        x = self.transform(img)
        rim = torch.tensor(r["rim"], dtype=torch.float32)   # [4]
        abn = torch.tensor(r["abn"], dtype=torch.long)       # [4]
        cdr = torch.tensor(r["cdr"], dtype=torch.long)       # [2]
        return x, rim, abn, cdr, r["id"], r["dataset"]


def build_dataloaders(batch_size=32, val_frac=0.10, seed=42, num_workers=4):
    cdr_map = _load_cdr_verdicts(CDR_MANIFEST)
    rows = _load_manifest(Path(ISNT_MANIFEST), cdr_map)
    train_all = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    train_rows, val_rows = _stratified_val_split(train_all, val_frac, seed)

    tf = _eval_transform()
    ds_tr = TriDataset(train_rows, PHOTOS_DIR, tf)
    ds_va = TriDataset(val_rows, PHOTOS_DIR, tf)
    ds_te = TriDataset(test_rows, PHOTOS_DIR, tf)
    dl = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                   num_workers=num_workers, pin_memory=True,
                                   drop_last=False)

    from collections import Counter
    def _abn_counts(rows):
        c = [Counter() for _ in range(N_Q)]
        for r in rows:
            for j, v in enumerate(r["abn"]):
                c[j][v] += 1
        return [dict(x) for x in c]

    def _cdr_counts(rows):
        c = [Counter() for _ in range(N_CDR)]
        for r in rows:
            for j, v in enumerate(r["cdr"]):
                c[j][v] += 1
        return [dict(x) for x in c]

    def _count_ds(rows):
        return dict(Counter(r["dataset"] for r in rows))

    info = {
        "n_train": len(train_rows), "n_val": len(val_rows),
        "n_test": len(test_rows),
        "test_by_dataset": _count_ds(test_rows),
        "train_by_dataset": _count_ds(train_rows),
        "rim_order": list(RIM_NAMES), "cdr_order": list(CDR_NAMES),
        "abn_train_class_counts": _abn_counts(train_rows),
        "cdr_train_class_counts": _cdr_counts(train_rows),
    }
    return dl(ds_tr, True), dl(ds_va, False), dl(ds_te, False), info


def _inv_freq_weights(counts, n_ch, n_cls):
    import numpy as np
    W = np.ones((n_ch, n_cls), dtype="float32")
    for j in range(n_ch):
        c = np.maximum(np.array([counts[j].get(k, 0) for k in range(n_cls)],
                                dtype="float64"), 1.0)
        W[j] = (c.sum() / (n_cls * c)).astype("float32")   # inverse freq, mean~1
    return torch.tensor(W)


def abn_class_weights(info):
    return _inv_freq_weights(info["abn_train_class_counts"], N_Q, N_ABN_CLASSES)


def cdr_class_weights(info):
    return _inv_freq_weights(info["cdr_train_class_counts"], N_CDR, N_CDR_CLASSES)


if __name__ == "__main__":
    tr, va, te, info = build_dataloaders(batch_size=8, num_workers=0)
    print("info:", info)
    x, rim, abn, cdr, ids, ds = next(iter(tr))
    print(f"x={tuple(x.shape)} rim={tuple(rim.shape)} abn={tuple(abn.shape)} cdr={tuple(cdr.shape)}")
    print(f"rim0={rim[0].tolist()} abn0={abn[0].tolist()} cdr0={cdr[0].tolist()} id0={ids[0]}")
    print("abn weights:\n", abn_class_weights(info))
    print("cdr weights:\n", cdr_class_weights(info))

"""
Dataset for the MULTI-OBJECTIVE rim head (value + order + abnormality, ONE head
off ONE pooled[B,1024]). Parallel to dataset_order_abn.py; we do NOT touch that
module — this is a separate, self-contained copy so concurrent edits don't clash.

Reads isnt_manifest.csv, which carries BOTH:
  - rim_i/s/n/t : the 4 human rim ratios in [0,1]  (order I,S,N,T)
                  -> used as BOTH the regression target (value) AND the source
                     of the ranking order (order target is derived from values).
  - abn_i/s/n/t : per-quadrant 3-class verdict judged by an LLM from the
                  expert's Step3 text: 0=normal, 1=mild thinning, 2=severe.

Returns (image, value[4 float], abn[4 long], id, dataset).
`value` doubles as the order target since the order is derived from the values.
Column order is the fixed (Inferior, Superior, Nasal, Temporal) == I,S,N,T.

Split policy: IDENTICAL to dataset_order_abn.py (seed=42) so results are
comparable: test = the 120; val = stratified 10% of train by binned MEAN rim.
No augmentation by default (aug was tried and HURT). square=True available but
NOT recommended (square preprocessing HURT in earlier tests).
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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224

RIM_KEYS = ("rim_i", "rim_s", "rim_n", "rim_t")
ABN_KEYS = ("abn_i", "abn_s", "abn_n", "abn_t")
RIM_NAMES = ("Inferior", "Superior", "Nasal", "Temporal")
N_ABN_CLASSES = 3   # 0 normal / 1 mild / 2 severe

# Per-quadrant normal-range lower bounds in rim-ratio units, in the I,S,N,T
# column order. Standard ISNT thresholds used by the dataset: Superior/Inferior
# 0.30, Nasal 0.22, Temporal 0.18. (Used by the value<->abn consistency loss.)
QUAD_LOWER = (0.30, 0.30, 0.22, 0.18)   # I, S, N, T


class _SquareCenterCrop:
    """Crop the largest centered square (side=min(W,H)) BEFORE resize."""
    def __call__(self, img):
        w, h = img.size
        s = min(w, h)
        l = (w - s) // 2; t = (h - s) // 2
        return img.crop((l, t, l + s, t + s))


def _eval_transform(square=False):
    ops = []
    if square:
        ops.append(_SquareCenterCrop())
    ops += [
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return T.Compose(ops)


def _train_transform():
    """On-the-fly augmentation for TRAIN only (rotate ±5° + zoom-in, NO flip).
    rim labels are scale/rotation invariant. (aug HURT earlier; off by default.)"""
    return T.Compose([
        T.RandomRotation(degrees=5, fill=0),
        T.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.00), ratio=(1.0, 1.0)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _train_transform_square():
    return T.Compose([
        _SquareCenterCrop(),
        T.RandomRotation(degrees=5, fill=0),
        T.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.00), ratio=(1.0, 1.0)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _load_manifest(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok":
                continue
            if any(r[k] in ("", "None") for k in RIM_KEYS):
                continue
            if any(r.get(k, "") in ("", "None") for k in ABN_KEYS):
                continue
            rows.append({
                "id": r["id"],
                "split": r["split"],
                "dataset": r["dataset"],
                "rim": [float(r[k]) for k in RIM_KEYS],   # [I,S,N,T]
                "abn": [int(r[k]) for k in ABN_KEYS],      # [I,S,N,T] in {0,1,2}
            })
    return rows


def _stratified_val_split(train_rows, val_frac, seed):
    """Identical to dataset_order_abn: stratify val by binned MEAN rim."""
    rng = random.Random(seed)
    bins = {}
    for r in train_rows:
        mean_rim = sum(r["rim"]) / 4.0
        b = min(int(mean_rim * 10), 9)
        bins.setdefault(b, []).append(r)
    val, tr = [], []
    for b, items in bins.items():
        items = items[:]
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_frac))
        val.extend(items[:n_val])
        tr.extend(items[n_val:])
    return tr, val


class RimMultiDataset(Dataset):
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
        value = torch.tensor(r["rim"], dtype=torch.float32)   # [4] in [0,1]
        abn = torch.tensor(r["abn"], dtype=torch.long)         # [4] in {0,1,2}
        return x, value, abn, r["id"], r["dataset"]


def build_dataloaders(batch_size=32, val_frac=0.10, seed=42, num_workers=4,
                      augment=False, square=False):
    rows = _load_manifest(Path(ISNT_MANIFEST))
    train_all = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    train_rows, val_rows = _stratified_val_split(train_all, val_frac, seed)

    if augment:
        tf_train = _train_transform_square() if square else _train_transform()
    else:
        tf_train = _eval_transform(square=square)
    tf = _eval_transform(square=square)
    ds_tr = RimMultiDataset(train_rows, PHOTOS_DIR, tf_train)
    ds_va = RimMultiDataset(val_rows, PHOTOS_DIR, tf)
    ds_te = RimMultiDataset(test_rows, PHOTOS_DIR, tf)

    dl = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                   num_workers=num_workers, pin_memory=True,
                                   drop_last=False)
    from collections import Counter
    def _cls_counts(rows):
        c = [Counter() for _ in range(4)]
        for r in rows:
            for j, v in enumerate(r["abn"]):
                c[j][v] += 1
        return [dict(x) for x in c]
    info = {
        "n_train": len(train_rows), "n_val": len(val_rows),
        "n_test": len(test_rows),
        "test_by_dataset": _count_ds(test_rows),
        "train_by_dataset": _count_ds(train_rows),
        "target_order": list(RIM_NAMES),
        "quad_lower": list(QUAD_LOWER),
        "abn_train_class_counts": _cls_counts(train_rows),
        "augment": bool(augment),
        "square": bool(square),
    }
    return dl(ds_tr, True), dl(ds_va, False), dl(ds_te, False), info


def _count_ds(rows):
    from collections import Counter
    return dict(Counter(r["dataset"] for r in rows))


def abn_class_weights(train_rows_info):
    """Inverse-frequency class weights per quadrant from train counts. [4,3]."""
    import numpy as np
    counts = train_rows_info["abn_train_class_counts"]
    W = np.ones((4, N_ABN_CLASSES), dtype="float32")
    for j in range(4):
        c = np.array([counts[j].get(k, 0) for k in range(N_ABN_CLASSES)],
                     dtype="float64")
        c = np.maximum(c, 1.0)
        w = c.sum() / (N_ABN_CLASSES * c)     # inverse freq, mean ~1
        W[j] = w.astype("float32")
    return torch.tensor(W)


if __name__ == "__main__":
    tr, va, te, info = build_dataloaders(batch_size=8, num_workers=0)
    print("info:", info)
    xb, value, abn, ids, dss = next(iter(tr))
    print(f"x={tuple(xb.shape)} value={tuple(value.shape)} abn={tuple(abn.shape)}")
    print(f"value[0]={value[0].tolist()}  abn[0]={abn[0].tolist()}  id0={ids[0]}")
    print("abn class weights [4,3]:\n", abn_class_weights(info))

"""
Dataset for the JOINT (ISNT-order + per-quadrant abnormality) head — Plan A.

Reads isnt_manifest.csv, which now carries BOTH:
  - rim_i/s/n/t        : the 4 human rim ratios   (ranking supervision)
  - abn_i/s/n/t        : per-quadrant 3-class verdict judged by an LLM from the
                         expert's Step3 text:  0=normal, 1=mild thinning,
                         2=severe thinning  ("thin-but-benign" -> 0, per spec)

Returns (image, rim[4], abn[4 long], id, dataset). The ORDER target order is
the SAME fixed (Inferior, Superior, Nasal, Temporal) as dataset_isnt so the
ranking head is identical; abn[] is in the same I,S,N,T column order.

Split policy: identical to dataset_isnt (test = the 120; val carved from train,
stratified by binned MEAN rim thickness). No augmentation. We do NOT change
dataset_isnt.py — this is a separate module so the existing order/isnt runs
stay byte-identical.
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


class _SquareCenterCrop:
    """Crop the largest centered square (side=min(W,H)) BEFORE resize, so a
    wide image (e.g. Papila 2576x1934) is not horizontally squeezed when made
    224x224. Matches the user's 'center-crop to square then 500' idea (the 500
    intermediate is irrelevant since we resize to 224 anyway)."""
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
    """On-the-fly augmentation for TRAIN only (val/test stay _eval_transform):
      - small rotation ±5° (I/S/N/T quadrant identity preserved at this angle)
      - random zoom-in via RandomResizedCrop scale 0.85-1.00 (rim is a RATIO,
        scale-invariant, so zoom does NOT change the rim labels)
      - NO horizontal/vertical flip (would swap Nasal<->Temporal and break labels)
    Rotation is applied first (fill with black like the fundus border), then the
    zoom-crop, so we never crop in empty rotated corners aggressively.
    """
    return T.Compose([
        T.RandomRotation(degrees=5, fill=0),
        T.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.00), ratio=(1.0, 1.0)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _train_transform_square():
    """Same augmentation but square-center-crop first (unified aspect ratio).
    RandomResizedCrop already outputs a square IMG_SIZE; we just pre-square so
    the random crop samples from a non-squeezed square instead of a wide image."""
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
    """Same as dataset_isnt: stratify val by binned MEAN rim thickness."""
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


class OrderAbnDataset(Dataset):
    def __init__(self, rows, photos_dir, transform, lr_map=None, all_left=False):
        self.rows = rows
        self.photos_dir = Path(photos_dir)
        self.transform = transform
        self.lr_map = lr_map or {}   # id -> 0/1 laterality (R=0/L=1)
        self.all_left = all_left     # unify to LEFT: flip R(lr=0) + swap N/T labels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.photos_dir / f"{r['id']}.jpg").convert("RGB")
        rim = list(r["rim"])   # [I,S,N,T]
        abn = list(r["abn"])
        if self.all_left and int(self.lr_map.get(r["id"], 0)) == 0:
            # R eye -> mirror to L orientation, and swap Nasal<->Temporal (idx 2,3)
            from PIL import ImageOps
            img = ImageOps.mirror(img)
            rim[2], rim[3] = rim[3], rim[2]
            abn[2], abn[3] = abn[3], abn[2]
        x = self.transform(img)
        rim = torch.tensor(rim, dtype=torch.float32)   # [4]
        abn = torch.tensor(abn, dtype=torch.long)       # [4] in {0,1,2}
        lr = torch.tensor(int(self.lr_map.get(r["id"], 0)), dtype=torch.long)  # 0/1
        return x, rim, abn, lr, r["id"], r["dataset"]


def _load_lr_map(path):
    import json
    d = json.loads(Path(path).read_text())
    return {k: int(v["lr"]) for k, v in d.items()}


def build_dataloaders(batch_size=32, val_frac=0.10, seed=42, num_workers=4,
                      augment=False, square=False, lr_json=None, all_left=False):
    rows = _load_manifest(Path(ISNT_MANIFEST))
    train_all = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    train_rows, val_rows = _stratified_val_split(train_all, val_frac, seed)

    # all_left forces square (the experiment spec: unify to left + reshape 500).
    if all_left:
        square = True
    # square=True: center-crop to square before resize (unified aspect ratio,
    # applied to TRAIN/VAL/TEST consistently so eval matches train).
    if augment:
        tf_train = _train_transform_square() if square else _train_transform()
    else:
        tf_train = _eval_transform(square=square)
    tf = _eval_transform(square=square)
    lr_map = _load_lr_map(lr_json) if lr_json else None
    if all_left and lr_map is None:
        raise ValueError("--all-left requires --lr-json (needs R/L to know what to flip)")
    ds_tr = OrderAbnDataset(train_rows, PHOTOS_DIR, tf_train, lr_map, all_left=all_left)
    ds_va = OrderAbnDataset(val_rows, PHOTOS_DIR, tf, lr_map, all_left=all_left)
    ds_te = OrderAbnDataset(test_rows, PHOTOS_DIR, tf, lr_map, all_left=all_left)

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
        "abn_train_class_counts": _cls_counts(train_rows),
        "augment": bool(augment),
        "square": bool(square),
        "all_left": bool(all_left),
    }
    return dl(ds_tr, True), dl(ds_va, False), dl(ds_te, False), info


def _count_ds(rows):
    from collections import Counter
    return dict(Counter(r["dataset"] for r in rows))


def abn_class_weights(train_rows_info):
    """Inverse-frequency class weights per quadrant from train counts.
    Returns tensor [4,3]."""
    import numpy as np
    counts = train_rows_info["abn_train_class_counts"]   # list of dicts
    W = np.ones((4, N_ABN_CLASSES), dtype="float32")
    for j in range(4):
        c = np.array([counts[j].get(k, 0) for k in range(N_ABN_CLASSES)], dtype="float64")
        c = np.maximum(c, 1.0)
        w = c.sum() / (N_ABN_CLASSES * c)     # inverse freq, mean ~1
        W[j] = w.astype("float32")
    return torch.tensor(W)


if __name__ == "__main__":
    tr, va, te, info = build_dataloaders(batch_size=8, num_workers=0)
    print("info:", info)
    xb, rim, abn, ids, dss = next(iter(tr))
    print(f"x={tuple(xb.shape)} rim={tuple(rim.shape)} abn={tuple(abn.shape)}")
    print(f"rim[0]={rim[0].tolist()}  abn[0]={abn[0].tolist()}  id0={ids[0]}")
    print("abn class weights [4,3]:\n", abn_class_weights(info))

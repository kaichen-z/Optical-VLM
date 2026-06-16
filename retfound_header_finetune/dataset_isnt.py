"""
ISNT rim-thickness regression dataset / dataloaders.

Reads isnt_manifest.csv (produced by preprocess_isnt.py — already has the
4 human-annotated rim ratios and the train/test split baked in), loads the
fundus image, returns (image_tensor, [rim_I, rim_S, rim_N, rim_T]).

FIXED target order: Inferior, Superior, Nasal, Temporal.

Splits: identical policy to the CDR dataset —
    - test  : the 120 held-out balanced images (manifest split == "test")
    - train : remaining 613; carve a VAL set (default 10%, stratified by
              binned MEAN rim thickness so val spans the full range).
              The 120 test set is ONLY for the final report.

Preprocess: resize whole image to 224 (RETFound native), ImageNet norm,
NO augmentation — same as the CDR round so the pooling ablation is clean.
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

# Fixed quadrant order for the 4-d regression target.
RIM_KEYS = ("rim_i", "rim_s", "rim_n", "rim_t")
RIM_NAMES = ("Inferior", "Superior", "Nasal", "Temporal")


def _eval_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
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
            rows.append({
                "id": r["id"],
                "split": r["split"],
                "dataset": r["dataset"],
                "rim": [float(r[k]) for k in RIM_KEYS],   # [I,S,N,T]
            })
    return rows


def _stratified_val_split(train_rows: list[dict], val_frac: float, seed: int):
    """Carve val from train, stratified by binned MEAN rim thickness so val
    covers the full thickness range (not just mid-range cases)."""
    rng = random.Random(seed)
    bins = {}
    for r in train_rows:
        mean_rim = sum(r["rim"]) / 4.0
        b = min(int(mean_rim * 10), 9)               # 10 bins over [0,1)
        bins.setdefault(b, []).append(r)
    val, tr = [], []
    for b, items in bins.items():
        items = items[:]
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_frac))
        val.extend(items[:n_val])
        tr.extend(items[n_val:])
    return tr, val


class ISNTDataset(Dataset):
    def __init__(self, rows: list[dict], photos_dir: Path, transform):
        self.rows = rows
        self.photos_dir = Path(photos_dir)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.photos_dir / f"{r['id']}.jpg").convert("RGB")
        x = self.transform(img)
        y = torch.tensor(r["rim"], dtype=torch.float32)   # [I,S,N,T]
        return x, y, r["id"], r["dataset"]


def build_dataloaders(batch_size: int = 32, val_frac: float = 0.10,
                      seed: int = 42, num_workers: int = 4):
    rows = _load_manifest(Path(ISNT_MANIFEST))
    train_all = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    train_rows, val_rows = _stratified_val_split(train_all, val_frac, seed)

    tf = _eval_transform()
    ds_tr = ISNTDataset(train_rows, PHOTOS_DIR, tf)
    ds_va = ISNTDataset(val_rows, PHOTOS_DIR, tf)
    ds_te = ISNTDataset(test_rows, PHOTOS_DIR, tf)

    dl = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                   num_workers=num_workers, pin_memory=True,
                                   drop_last=False)
    info = {
        "n_train": len(train_rows), "n_val": len(val_rows),
        "n_test": len(test_rows),
        "test_by_dataset": _count_ds(test_rows),
        "train_by_dataset": _count_ds(train_rows),
        "target_order": list(RIM_NAMES),
    }
    return dl(ds_tr, True), dl(ds_va, False), dl(ds_te, False), info


def _count_ds(rows):
    from collections import Counter
    return dict(Counter(r["dataset"] for r in rows))


if __name__ == "__main__":
    tr, va, te, info = build_dataloaders(batch_size=8, num_workers=0)
    print("dataloader info:", info)
    xb, yb, ids, dss = next(iter(tr))
    print(f"batch  x={tuple(xb.shape)}  y={tuple(yb.shape)}  "
          f"y[0]={yb[0].tolist()}  id[0]={ids[0]}  ds[0]={dss[0]}")

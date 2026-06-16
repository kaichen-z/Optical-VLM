"""
CDR regression dataset / dataloaders.

Reads cdr_manifest.csv (produced by preprocess_cdr.py — already has the
human-annotated cdr_v / cdr_h and the train/test split baked in), loads the
corresponding fundus image, and returns (image_tensor, [cdr_v, cdr_h]).

Splits:
    - test  : the 120 held-out balanced images (manifest split == "test")
    - train : the remaining 613 (manifest split == "train"), from which we
              carve a VAL set (default 10%, stratified by binned cdr_v) for
              early-stopping / checkpoint selection. The 120 test set is
              ONLY touched for the final report — never for model selection.

Preprocess (decided): resize whole image to 224x224 (RETFound's native
pretrain size, no disc ROI crop), ImageNet normalization, NO augmentation.
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
    from config import PHOTOS_DIR, CDR_MANIFEST
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import PHOTOS_DIR, CDR_MANIFEST

import os as _os
# Optional override (e.g. cdr_manifest_refugeaug.csv) WITHOUT touching
# config.py or clobbering the canonical cdr_manifest.csv. Backward compatible:
# unset -> original manifest, exactly as before.
CDR_MANIFEST = Path(_os.environ.get("CDR_MANIFEST", str(CDR_MANIFEST)))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224


def _eval_transform():
    # No augmentation (per decision). Same transform for train/val/test in
    # this first round so the pooling ablation is clean.
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
            if r["cdr_v"] in ("", "None") or r["cdr_h"] in ("", "None"):
                continue
            rows.append({
                "id": r["id"],
                "split": r["split"],
                "dataset": r.get("dataset", "?"),
                "cdr_v": float(r["cdr_v"]),
                "cdr_h": float(r["cdr_h"]),
                # absolute path override for rows whose image is NOT in
                # PHOTOS_DIR (e.g. REFUGE aug); "" -> PHOTOS_DIR/<id>.jpg
                "img_path": (r.get("img_path") or "").strip(),
            })
    return rows


def _stratified_val_split(train_rows: list[dict], val_frac: float, seed: int):
    """Carve val from train, stratified by binned cdr_v so val covers the
    full CDR range (not just easy mid-range cases)."""
    rng = random.Random(seed)
    bins = {}
    for r in train_rows:
        b = min(int(r["cdr_v"] * 10), 9)         # 10 bins over [0,1)
        bins.setdefault(b, []).append(r)
    val, tr = [], []
    for b, items in bins.items():
        items = items[:]
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_frac))
        val.extend(items[:n_val])
        tr.extend(items[n_val:])
    return tr, val


class CDRDataset(Dataset):
    def __init__(self, rows: list[dict], photos_dir: Path, transform):
        self.rows = rows
        self.photos_dir = Path(photos_dir)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        _ip = r.get("img_path")
        _p = Path(_ip) if _ip else (self.photos_dir / f"{r['id']}.jpg")
        img = Image.open(_p).convert("RGB")
        x = self.transform(img)
        y = torch.tensor([r["cdr_v"], r["cdr_h"]], dtype=torch.float32)
        return x, y, r["id"], r["dataset"]


def build_dataloaders(batch_size: int = 32, val_frac: float = 0.10,
                      seed: int = 42, num_workers: int = 4):
    rows = _load_manifest(Path(CDR_MANIFEST))
    train_all = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    train_rows, val_rows = _stratified_val_split(train_all, val_frac, seed)

    tf = _eval_transform()
    ds_tr = CDRDataset(train_rows, PHOTOS_DIR, tf)
    ds_va = CDRDataset(val_rows, PHOTOS_DIR, tf)
    ds_te = CDRDataset(test_rows, PHOTOS_DIR, tf)

    dl = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                   num_workers=num_workers, pin_memory=True,
                                   drop_last=False)
    info = {
        "n_train": len(train_rows), "n_val": len(val_rows),
        "n_test": len(test_rows),
        "test_by_dataset": _count_ds(test_rows),
        "train_by_dataset": _count_ds(train_rows),
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

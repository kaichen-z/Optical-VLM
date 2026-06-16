"""
Glaucomatous-signs (5-disease, 3-class) dataset / dataloaders.

Reads signs_manifest_clean.csv (curated human-reviewed labels from
signs_label_map.json — NOT regex). Disc hemorrhage is DROPPED (degenerate,
0% present). Remaining 5 signs, each a 3-class target:

    absent = 0   present = 1   uncertain = 2   NA = -1 (masked in loss)

FIXED disease order: Notching, RNFL defect, Bayoneting, Beta-zone PPA-beta,
Peripapillary atrophy.  y = LongTensor[5] in {0,1,2,-1}.

Split policy identical to CDR/ISNT (613 train / 120 test); val carved from
train, stratified by the number of `present` signs per eye so val spans
healthy <-> multi-sign eyes.
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
    from config import PHOTOS_DIR, SIGNS_MANIFEST
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import PHOTOS_DIR, SIGNS_MANIFEST

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224

# dropped 'dh' (Disc hemorrhage) — degenerate
SIGN_KEYS = ("notch", "rnfl", "bayo", "beta", "ppa")
SIGN_NAMES = ("Notching", "RNFL defect", "Bayoneting",
              "Beta-zone atrophy", "Peripapillary atrophy")
LBL2IDX = {"absent": 0, "present": 1, "uncertain": 2, "NA": -1, "": -1}
NUM_DISEASE = len(SIGN_KEYS)         # 5
NUM_CLASS = 3                        # absent / present / uncertain


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
            y = [LBL2IDX.get(r[f"lbl_{k}"], -1) for k in SIGN_KEYS]
            rows.append({"id": r["id"], "split": r["split"],
                         "dataset": r["dataset"], "y": y})
    return rows


def _stratified_val_split(train_rows, val_frac, seed):
    """Stratify by #present signs per eye (0..5) so val covers the
    healthy <-> heavily-affected spectrum."""
    rng = random.Random(seed)
    bins = {}
    for r in train_rows:
        b = sum(1 for v in r["y"] if v == 1)
        bins.setdefault(b, []).append(r)
    val, tr = [], []
    for b, items in bins.items():
        items = items[:]
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_frac))
        val.extend(items[:n_val])
        tr.extend(items[n_val:])
    return tr, val


class SignsDataset(Dataset):
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
        y = torch.tensor(r["y"], dtype=torch.long)        # [5] in {-1,0,1,2}
        return x, y, r["id"], r["dataset"]


def _class_counts(rows):
    """Per-disease valid (non-NA) class counts on the given rows."""
    cc = [[0, 0, 0] for _ in range(NUM_DISEASE)]
    for r in rows:
        for d, v in enumerate(r["y"]):
            if v >= 0:
                cc[d][v] += 1
    return cc


def build_dataloaders(batch_size=32, val_frac=0.10, seed=42, num_workers=4):
    rows = _load_manifest(Path(SIGNS_MANIFEST))
    train_all = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    train_rows, val_rows = _stratified_val_split(train_all, val_frac, seed)

    tf = _eval_transform()
    dl = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                   num_workers=num_workers, pin_memory=True,
                                   drop_last=False)
    info = {
        "n_train": len(train_rows), "n_val": len(val_rows),
        "n_test": len(test_rows),
        "test_by_dataset": _count_ds(test_rows),
        "train_by_dataset": _count_ds(train_rows),
        "disease_order": list(SIGN_NAMES),
        "train_class_counts": _class_counts(train_rows),   # [5][3]
    }
    return (dl(SignsDataset(train_rows, PHOTOS_DIR, tf), True),
            dl(SignsDataset(val_rows, PHOTOS_DIR, tf), False),
            dl(SignsDataset(test_rows, PHOTOS_DIR, tf), False), info)


def _count_ds(rows):
    from collections import Counter
    return dict(Counter(r["dataset"] for r in rows))


if __name__ == "__main__":
    tr, va, te, info = build_dataloaders(batch_size=8, num_workers=0)
    print("info:", info)
    xb, yb, ids, ds = next(iter(tr))
    print(f"x={tuple(xb.shape)} y={tuple(yb.shape)} y[0]={yb[0].tolist()} "
          f"id0={ids[0]} ds0={ds[0]}")

"""
Laterality (left/right eye) dataset via horizontal-flip synthesis.

REFUGE 1200 are ALL one fixed orientation (laterality-normalized; verified).
A horizontal mirror of a fundus image == the opposite-eye orientation, so
we synthesize a balanced 2-class set:

    class 0 = "R"  : original REFUGE orientation (fovea right of disc)
    class 1 = "L"  : horizontally flipped (fovea left of disc)

TRAIN: each __getitem__ random-flips p=0.5 (aug -> infinite balanced variety).
VAL/TEST: each image yielded TWICE (orig + flipped) -> exactly balanced,
deterministic, seed-independent.

Split: pooled 1200 random by id (most into train, per request), fixed seed.
img_path in laterality_manifest.csv already points at the grace images.
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
    from config import CDR_MANIFEST  # only for HERE resolution fallback
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

HERE = Path(__file__).resolve().parent
LAT_MANIFEST = HERE / "laterality_manifest.csv"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224
CLASS_NAMES = ("R", "L")          # 0 = original orient, 1 = flipped
NUM_CLASS = 2


def _post_transform():
    # resize+norm applied AFTER any flip (flip done on the PIL image)
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _load_manifest(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [{"id": r["id"], "dataset": r.get("dataset", "REFUGE"),
                 "img_path": r["img_path"]}
                for r in csv.DictReader(f)]


def _split(rows, val_frac, test_frac, seed):
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    n = len(rows)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    test = rows[:n_test]
    val = rows[n_test:n_test + n_val]
    train = rows[n_test + n_val:]
    return train, val, test


class LateralityDataset(Dataset):
    """train: random flip p=0.5. eval: each row x2 (orig + flipped)."""

    def __init__(self, rows, transform, mode):
        self.rows = rows
        self.transform = transform
        self.mode = mode                       # 'train' | 'eval'

    def __len__(self):
        return len(self.rows) if self.mode == "train" else 2 * len(self.rows)

    def __getitem__(self, i):
        if self.mode == "train":
            r = self.rows[i]
            flip = random.random() < 0.5
        else:
            r = self.rows[i // 2]
            flip = (i % 2 == 1)
        img = Image.open(r["img_path"]).convert("RGB")
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        x = self.transform(img)
        y = 1 if flip else 0                   # 1='L'(flipped) 0='R'(orig)
        return x, torch.tensor(y, dtype=torch.long), r["id"], r["dataset"]


def build_dataloaders(batch_size=32, val_frac=0.10, test_frac=0.10,
                      seed=42, num_workers=4):
    rows = _load_manifest(LAT_MANIFEST)
    train_rows, val_rows, test_rows = _split(rows, val_frac, test_frac, seed)
    tf = _post_transform()
    dl = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                   num_workers=num_workers, pin_memory=True,
                                   drop_last=False)
    info = {
        "n_base": len(rows),
        "n_train_base": len(train_rows),
        "n_val_base": len(val_rows), "n_test_base": len(test_rows),
        "n_val_eval": 2 * len(val_rows), "n_test_eval": 2 * len(test_rows),
        "classes": list(CLASS_NAMES),
        "manifest": str(LAT_MANIFEST),
    }
    return (dl(LateralityDataset(train_rows, tf, "train"), True),
            dl(LateralityDataset(val_rows, tf, "eval"), False),
            dl(LateralityDataset(test_rows, tf, "eval"), False), info)


if __name__ == "__main__":
    tr, va, te, info = build_dataloaders(batch_size=8, num_workers=0)
    print("info:", info)
    xb, yb, ids, ds = next(iter(tr))
    print(f"x={tuple(xb.shape)} y={yb.tolist()} id0={ids[0]}")

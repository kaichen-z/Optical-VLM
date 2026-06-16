"""
CDR qualitative-verdict dataset (2 channels x 3 classes).

Target = the DOCTOR'S Step2 qualitative verdict on the CDR indicator,
NOT the numeric ratio.  Read from the two columns appended to
cdr_manifest.csv (Claude per-row LLM-judged, not regex; see
project-cdr-verdict-gt):

    cdr_v_verdict (vertical)   cdr_h_verdict (horizontal)
    normal = 0   borderline = 1   abnormal = 2   (missing -> -1, masked)

y = LongTensor[2] = [vertical_idx, horizontal_idx].

Row filter & split policy IDENTICAL to dataset.py (the numeric-CDR
dataset): status==ok and cdr_v/cdr_h present, canonical 613/120 split
baked in the manifest -> directly comparable to every prior CDR run.
Optional per-row img_path override supported (same as dataset.py).
Val carved from train, stratified by max(v,h) severity so val spans the
normal <-> abnormal spectrum.
"""
from __future__ import annotations

import csv
import os
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

# same optional override hook dataset.py uses (backward compatible)
CDR_MANIFEST = Path(os.environ.get("CDR_MANIFEST", str(CDR_MANIFEST)))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224

CHANNEL_KEYS = ("cdr_v_verdict", "cdr_h_verdict")
CHANNEL_NAMES = ("vertical", "horizontal")
LBL2IDX = {"normal": 0, "borderline": 1, "abnormal": 2,
           "no_verdict": -1, "NA": -1, "": -1}
NUM_CH = len(CHANNEL_KEYS)            # 2
NUM_CLASS = 3                         # normal / borderline / abnormal


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
            # SAME filter as dataset.py so the 733/(613,120) set matches
            if r.get("status") != "ok":
                continue
            if r["cdr_v"] in ("", "None") or r["cdr_h"] in ("", "None"):
                continue
            y = [LBL2IDX.get((r.get(k) or "").strip(), -1)
                 for k in CHANNEL_KEYS]
            rows.append({
                "id": r["id"], "split": r["split"],
                "dataset": r.get("dataset", "?"),
                "img_path": (r.get("img_path") or "").strip(),
                "y": y,
            })
    return rows


def _stratified_val_split(train_rows, val_frac, seed):
    """Stratify by max(v,h) severity (0/1/2) so val spans normal..abnormal."""
    rng = random.Random(seed)
    bins = {}
    for r in train_rows:
        sev = max((v for v in r["y"] if v >= 0), default=0)
        bins.setdefault(sev, []).append(r)
    val, tr = [], []
    for _, items in bins.items():
        items = items[:]
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_frac))
        val.extend(items[:n_val])
        tr.extend(items[n_val:])
    return tr, val


class CDRVerdictDataset(Dataset):
    def __init__(self, rows, photos_dir, transform):
        self.rows = rows
        self.photos_dir = Path(photos_dir)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ip = r.get("img_path")
        p = Path(ip) if ip else (self.photos_dir / f"{r['id']}.jpg")
        img = Image.open(p).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(r["y"], dtype=torch.long)        # [2] in {-1,0,1,2}
        return x, y, r["id"], r["dataset"]


def _class_counts(rows):
    """Per-channel valid class counts -> [2][3]."""
    cc = [[0, 0, 0] for _ in range(NUM_CH)]
    for r in rows:
        for ch, v in enumerate(r["y"]):
            if v >= 0:
                cc[ch][v] += 1
    return cc


def _count_ds(rows):
    from collections import Counter
    return dict(Counter(r["dataset"] for r in rows))


def build_dataloaders(batch_size=32, val_frac=0.10, seed=42, num_workers=4):
    rows = _load_manifest(CDR_MANIFEST)
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
        "channel_order": list(CHANNEL_NAMES),
        "train_class_counts": _class_counts(train_rows),   # [2][3]
        "manifest": str(CDR_MANIFEST),
    }
    return (dl(CDRVerdictDataset(train_rows, PHOTOS_DIR, tf), True),
            dl(CDRVerdictDataset(val_rows, PHOTOS_DIR, tf), False),
            dl(CDRVerdictDataset(test_rows, PHOTOS_DIR, tf), False), info)


if __name__ == "__main__":
    tr, va, te, info = build_dataloaders(batch_size=8, num_workers=0)
    print("info:", info)
    xb, yb, ids, ds = next(iter(tr))
    print(f"x={tuple(xb.shape)} y={tuple(yb.shape)} y[0]={yb[0].tolist()} "
          f"id0={ids[0]} ds0={ds[0]}")

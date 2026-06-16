"""
QUAD-REG dataset: identical to dataset_quad EXCEPT the CDR target is the
NUMERIC ratio (cdr_v, cdr_h floats) instead of the 3-level verdict.

Returns (image, rim[4]f, abn[4]long, cdr[2]FLOAT, sign[5]long(-1=NA), id, dataset).
Reason: CDR-as-number is the stronger downstream signal (r≈0.87 with diagnosis);
discretizing it to 3 levels hurt MedGemma diagnosis. So the joint head should
REGRESS the CDR value. rim(order+abn) + signs targets unchanged from dataset_quad.

Join 3 manifests by id (733 all aligned, train 613 / test 120). Plain 224 resize.
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
    from config import PHOTOS_DIR, ISNT_MANIFEST, SIGNS_MANIFEST
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import PHOTOS_DIR, ISNT_MANIFEST, SIGNS_MANIFEST

_HERE = Path(__file__).resolve().parent
CDR_MANIFEST = _HERE / "cdr_manifest.csv"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224

RIM_KEYS = ("rim_i", "rim_s", "rim_n", "rim_t")
ABN_KEYS = ("abn_i", "abn_s", "abn_n", "abn_t")
RIM_NAMES = ("Inferior", "Superior", "Nasal", "Temporal")
CDR_NUM_KEYS = ("cdr_v", "cdr_h")
CDR_NAMES = ("vertical", "horizontal")
SIGN_KEYS = ("notch", "rnfl", "bayo", "beta", "ppa")
SIGN_NAMES = ("Notching", "RNFL defect", "Bayoneting",
              "Beta-zone atrophy", "Peripapillary atrophy")
LBL2IDX = {"absent": 0, "present": 1, "uncertain": 2, "NA": -1, "": -1}

N_ABN_CLASSES = 3
N_SIGN_CLASSES = 3
N_Q = 4
N_CDR = 2
N_SIGN = 5


def _eval_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _load_cdr_num(path):
    out = {}
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        if r["status"] != "ok":
            continue
        if r.get("cdr_v") in ("", "None") or r.get("cdr_h") in ("", "None"):
            continue
        out[r["id"]] = [float(r["cdr_v"]), float(r["cdr_h"])]
    return out


def _load_signs(path):
    out = {}
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        out[r["id"]] = [LBL2IDX.get(r.get(f"lbl_{k}", ""), -1) for k in SIGN_KEYS]
    return out


def _load_manifest(isnt_path, cdr_map, sign_map):
    rows = []
    for r in csv.DictReader(open(isnt_path, newline="", encoding="utf-8")):
        if r["status"] != "ok":
            continue
        if any(r[k] in ("", "None") for k in RIM_KEYS):
            continue
        if any(r.get(k, "") in ("", "None") for k in ABN_KEYS):
            continue
        if r["id"] not in cdr_map or r["id"] not in sign_map:
            continue
        rows.append({
            "id": r["id"], "split": r["split"], "dataset": r["dataset"],
            "rim": [float(r[k]) for k in RIM_KEYS],
            "abn": [int(r[k]) for k in ABN_KEYS],
            "cdr": list(cdr_map[r["id"]]),     # [v,h] FLOAT
            "sign": list(sign_map[r["id"]]),
        })
    return rows


def _stratified_val_split(train_rows, val_frac, seed):
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
        val.extend(items[:n_val]); tr.extend(items[n_val:])
    return tr, val


class QuadRegDataset(Dataset):
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
        return (x,
                torch.tensor(r["rim"], dtype=torch.float32),    # [4]
                torch.tensor(r["abn"], dtype=torch.long),        # [4]
                torch.tensor(r["cdr"], dtype=torch.float32),     # [2] FLOAT
                torch.tensor(r["sign"], dtype=torch.long),       # [5] (-1=NA)
                r["id"], r["dataset"])


def build_dataloaders(batch_size=32, val_frac=0.10, seed=42, num_workers=4):
    cdr_map = _load_cdr_num(CDR_MANIFEST)
    sign_map = _load_signs(SIGNS_MANIFEST)
    rows = _load_manifest(Path(ISNT_MANIFEST), cdr_map, sign_map)
    train_all = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    train_rows, val_rows = _stratified_val_split(train_all, val_frac, seed)

    tf = _eval_transform()
    dl = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                   num_workers=num_workers, pin_memory=True,
                                   drop_last=False)
    from collections import Counter

    def _multi_counts(rows, key, n_ch):
        c = [Counter() for _ in range(n_ch)]
        for r in rows:
            for j, v in enumerate(r[key]):
                c[j][v] += 1
        return [dict(x) for x in c]

    info = {
        "n_train": len(train_rows), "n_val": len(val_rows),
        "n_test": len(test_rows),
        "test_by_dataset": dict(Counter(r["dataset"] for r in test_rows)),
        "rim_order": list(RIM_NAMES), "cdr_order": list(CDR_NAMES),
        "sign_order": list(SIGN_NAMES), "cdr_target": "numeric",
        "abn_train_class_counts": _multi_counts(train_rows, "abn", N_Q),
        "sign_train_class_counts": _multi_counts(train_rows, "sign", N_SIGN),
    }
    return (dl(QuadRegDataset(train_rows, PHOTOS_DIR, tf), True),
            dl(QuadRegDataset(val_rows, PHOTOS_DIR, tf), False),
            dl(QuadRegDataset(test_rows, PHOTOS_DIR, tf), False), info)


def _inv_freq_weights(counts, n_ch, n_cls):
    import numpy as np
    W = np.ones((n_ch, n_cls), dtype="float32")
    for j in range(n_ch):
        c = np.maximum(np.array([counts[j].get(k, 0) for k in range(n_cls)],
                                dtype="float64"), 1.0)
        W[j] = (c.sum() / (n_cls * c)).astype("float32")
    return torch.tensor(W)


def abn_class_weights(info):
    return _inv_freq_weights(info["abn_train_class_counts"], N_Q, N_ABN_CLASSES)


def sign_class_weights(info):
    return _inv_freq_weights(info["sign_train_class_counts"], N_SIGN, N_SIGN_CLASSES)


if __name__ == "__main__":
    tr, va, te, info = build_dataloaders(batch_size=8, num_workers=0)
    print("info n_train/val/test:", info["n_train"], info["n_val"], info["n_test"])
    x, rim, abn, cdr, sign, ids, ds = next(iter(tr))
    print(f"x={tuple(x.shape)} rim={tuple(rim.shape)} abn={tuple(abn.shape)} "
          f"cdr={tuple(cdr.shape)}(float) sign={tuple(sign.shape)}")
    print("cdr0(numeric):", cdr[0].tolist(), "id0:", ids[0])

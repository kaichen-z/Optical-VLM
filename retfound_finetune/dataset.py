"""Dataset for the shared multi-head RETFound training.

Reads four CSV manifests keyed by image id:
    isnt   : rim_i/s/n/t (rim-width ratios) + abn_i/s/n/t (0/1/2 per-quad status)
    cdr    : cdr_v, cdr_h (numeric cup-to-disc ratios)
    signs  : lbl_<notch/rnfl/bayo/beta/ppa> in {absent, present, uncertain}
    dx     : label in {0, 1}   (0 = not glaucoma, 1 = glaucoma)

Each manifest carries a `split` column (train/val/test); the val fraction is
carved from the train+val pool, stratified on the diagnosis label.
Images are expected at <images>/<dataset>_<id>.jpg.

__getitem__ returns (image, rim, abn, cdr, sign, dx, id, dataset).
"""
import csv
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

_HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", _HERE / "data"))
CDR_MANIFEST = Path(os.environ.get("CDR_MANIFEST", DATA_ROOT / "cdr_manifest.csv"))
ISNT_MANIFEST = Path(os.environ.get("ISNT_MANIFEST", DATA_ROOT / "isnt_manifest.csv"))
SIGNS_MANIFEST = Path(os.environ.get("SIGNS_MANIFEST", DATA_ROOT / "signs_manifest.csv"))
DX_MANIFEST = Path(os.environ.get("DX_MANIFEST", DATA_ROOT / "dx_manifest.csv"))
PHOTOS_DIR = Path(os.environ.get("PHOTOS_DIR", DATA_ROOT / "images"))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224

RIM_KEYS = ("rim_i", "rim_s", "rim_n", "rim_t")
ABN_KEYS = ("abn_i", "abn_s", "abn_n", "abn_t")
RIM_NAMES = ("Inferior", "Superior", "Nasal", "Temporal")
CDR_KEYS = ("cdr_v", "cdr_h")
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
N_DX = 2


def _transform():
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
                      T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def _read(path):
    return list(csv.DictReader(open(path, newline="", encoding="utf-8")))


def _load_cdr(path):
    out = {}
    for r in _read(path):
        if r.get("cdr_v") in ("", "None") or r.get("cdr_h") in ("", "None"):
            continue
        out[r["id"]] = [float(r["cdr_v"]), float(r["cdr_h"])]
    return out


def _load_signs(path):
    return {r["id"]: [LBL2IDX.get(r.get(f"lbl_{k}", ""), -1) for k in SIGN_KEYS]
            for r in _read(path)}


def _load_dx(path):
    out = {}
    for r in _read(path):
        if r.get("label") in ("", "None"):
            continue
        out[str(r["id"])] = int(r["label"])
    return out


def _load_rows(isnt_path, cdr_map, sign_map, dx_map):
    rows = []
    for r in _read(isnt_path):
        if any(r[k] in ("", "None") for k in RIM_KEYS):
            continue
        if any(r.get(k, "") in ("", "None") for k in ABN_KEYS):
            continue
        if r["id"] not in cdr_map or r["id"] not in sign_map or r["id"] not in dx_map:
            continue
        rows.append({
            "id": r["id"], "split": r["split"], "dataset": r["dataset"],
            "rim": [float(r[k]) for k in RIM_KEYS],
            "abn": [int(r[k]) for k in ABN_KEYS],
            "cdr": list(cdr_map[r["id"]]),
            "sign": list(sign_map[r["id"]]),
            "dx": int(dx_map[r["id"]]),
        })
    return rows


def _stratified_val(train_rows, val_frac, seed):
    rng = random.Random(seed)
    bins = {0: [], 1: []}
    for r in train_rows:
        bins[r["dx"]].append(r)
    tr, val = [], []
    for items in bins.values():
        items = items[:]
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_frac))
        val.extend(items[:n_val])
        tr.extend(items[n_val:])
    return tr, val


class MultiHeadDataset(Dataset):
    def __init__(self, rows, photos_dir, transform):
        self.rows = rows
        self.photos_dir = Path(photos_dir)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.photos_dir / f"{r['dataset']}_{r['id']}.jpg").convert("RGB")
        return (self.transform(img),
                torch.tensor(r["rim"], dtype=torch.float32),
                torch.tensor(r["abn"], dtype=torch.long),
                torch.tensor(r["cdr"], dtype=torch.float32),
                torch.tensor(r["sign"], dtype=torch.long),
                torch.tensor(r["dx"], dtype=torch.long),
                r["id"], r["dataset"])


def build_dataloaders(batch_size=32, val_frac=0.10, seed=42, num_workers=4):
    cdr_map = _load_cdr(CDR_MANIFEST)
    sign_map = _load_signs(SIGNS_MANIFEST)
    dx_map = _load_dx(DX_MANIFEST)
    rows = _load_rows(ISNT_MANIFEST, cdr_map, sign_map, dx_map)

    train_all = [r for r in rows if r["split"] in ("train", "val")]
    test_rows = [r for r in rows if r["split"] == "test"]
    train_rows, val_rows = _stratified_val(train_all, val_frac, seed)

    tf = _transform()
    def loader(rows_, shuffle):
        return DataLoader(MultiHeadDataset(rows_, PHOTOS_DIR, tf),
                          batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    def multi_counts(rows_, key, n_ch):
        c = [Counter() for _ in range(n_ch)]
        for r in rows_:
            for j, v in enumerate(r[key]):
                c[j][v] += 1
        return [dict(x) for x in c]

    info = {
        "n_train": len(train_rows), "n_val": len(val_rows), "n_test": len(test_rows),
        "dx_train_counts": dict(Counter(r["dx"] for r in train_rows)),
        "dx_test_counts": dict(Counter(r["dx"] for r in test_rows)),
        "abn_train_class_counts": multi_counts(train_rows, "abn", N_Q),
        "sign_train_class_counts": multi_counts(train_rows, "sign", N_SIGN),
    }
    return (loader(train_rows, True), loader(val_rows, False),
            loader(test_rows, False), info)


def _inv_freq(counts, n_ch, n_cls):
    w = np.ones((n_ch, n_cls), dtype="float32")
    for j in range(n_ch):
        c = np.maximum([counts[j].get(k, 0) for k in range(n_cls)], 1.0)
        w[j] = (c.sum() / (n_cls * c)).astype("float32")
    return torch.tensor(w)


def abn_class_weights(info):
    return _inv_freq(info["abn_train_class_counts"], N_Q, N_ABN_CLASSES)


def sign_class_weights(info):
    return _inv_freq(info["sign_train_class_counts"], N_SIGN, N_SIGN_CLASSES)


def dx_class_weights(info):
    c = info["dx_train_counts"]
    arr = np.maximum([c.get(0, 0), c.get(1, 0)], 1.0)
    return torch.tensor((arr.sum() / (N_DX * arr)).astype("float32"))

"""
ISOLATED dataset for FLIP-AUGMENTED rim training (order + per-quadrant abn).

Flip augmentation = DOUBLE the TRAIN set with horizontal mirrors:
  - original 551 train images (labels as-is)
  - + 551 horizontally-flipped copies: image mirrored, and Nasal<->Temporal
    labels SWAPPED (rim_n<->rim_t, abn_n<->abn_t); Inferior/Superior unchanged.
  => 1102 train samples, shuffled.
VAL (62) and TEST (120) are NOT flipped (no leakage; test comparable to baseline).

No square, no laterality — clean flip-only augmentation. Same seed=42 stratified
split as dataset_order_abn so train/val/test membership matches the baselines.

Self-contained; does NOT import or modify dataset_order_abn.
"""
from __future__ import annotations
import csv, random
from pathlib import Path
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

try:
    from config import PHOTOS_DIR, ISNT_MANIFEST
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import PHOTOS_DIR, ISNT_MANIFEST

IMAGENET_MEAN=(0.485,0.456,0.406); IMAGENET_STD=(0.229,0.224,0.225); IMG_SIZE=224
RIM_KEYS=("rim_i","rim_s","rim_n","rim_t"); ABN_KEYS=("abn_i","abn_s","abn_n","abn_t")
RIM_NAMES=("Inferior","Superior","Nasal","Temporal"); N_ABN_CLASSES=3
# N,T are indices 2,3 in the I,S,N,T ordering -> swapped on horizontal flip

def _tf():
    return T.Compose([T.Resize((IMG_SIZE,IMG_SIZE)),T.ToTensor(),
                      T.Normalize(IMAGENET_MEAN,IMAGENET_STD)])

def _load_manifest(path):
    rows=[]
    for r in csv.DictReader(open(path,newline="",encoding="utf-8")):
        if r["status"]!="ok": continue
        if any(r[k] in ("","None") for k in RIM_KEYS): continue
        if any(r.get(k,"") in ("","None") for k in ABN_KEYS): continue
        rows.append({"id":r["id"],"split":r["split"],"dataset":r["dataset"],
                     "rim":[float(r[k]) for k in RIM_KEYS],
                     "abn":[int(r[k]) for k in ABN_KEYS]})
    return rows

def _stratified_val_split(train_rows,val_frac,seed):
    rng=random.Random(seed); bins={}
    for r in train_rows:
        b=min(int(sum(r["rim"])/4.0*10),9); bins.setdefault(b,[]).append(r)
    val,tr=[],[]
    for b,items in bins.items():
        items=items[:]; rng.shuffle(items)
        nv=max(1,round(len(items)*val_frac)); val.extend(items[:nv]); tr.extend(items[nv:])
    return tr,val

class RimFlipDataset(Dataset):
    """Each row carries a 'flip' flag. flip=True -> mirror image + swap N/T."""
    def __init__(self,rows,photos_dir,transform):
        self.rows=rows; self.photos_dir=Path(photos_dir); self.transform=transform
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        img=Image.open(self.photos_dir/f"{r['id']}.jpg").convert("RGB")
        rim=list(r["rim"]); abn=list(r["abn"])
        if r.get("flip"):
            img=ImageOps.mirror(img)
            rim[2],rim[3]=rim[3],rim[2]   # N<->T
            abn[2],abn[3]=abn[3],abn[2]
        x=self.transform(img)
        return x, torch.tensor(rim,dtype=torch.float32), torch.tensor(abn,dtype=torch.long), r["id"], r["dataset"]

def build_dataloaders(batch_size=32,val_frac=0.10,seed=42,num_workers=4):
    rows=_load_manifest(Path(ISNT_MANIFEST))
    train_all=[r for r in rows if r["split"]=="train"]
    test_rows=[r for r in rows if r["split"]=="test"]
    train_rows,val_rows=_stratified_val_split(train_all,val_frac,seed)
    # DOUBLE train: original + flipped
    tr_orig=[{**r,"flip":False} for r in train_rows]
    tr_flip=[{**r,"flip":True} for r in train_rows]
    tr_aug=tr_orig+tr_flip
    random.Random(seed).shuffle(tr_aug)
    val_rows=[{**r,"flip":False} for r in val_rows]
    test_rows=[{**r,"flip":False} for r in test_rows]
    tf=_tf()
    dl=lambda ds,sh: DataLoader(ds,batch_size=batch_size,shuffle=sh,
                                num_workers=num_workers,pin_memory=True,drop_last=False)
    from collections import Counter
    def cc(rows):
        c=[Counter() for _ in range(4)]
        for r in rows:
            ab=list(r["abn"])
            if r.get("flip"): ab[2],ab[3]=ab[3],ab[2]
            for j,v in enumerate(ab): c[j][v]+=1
        return [dict(x) for x in c]
    def cd(rows):
        from collections import Counter as C; return dict(C(r["dataset"] for r in rows))
    info={"n_train":len(tr_aug),"n_train_unique":len(train_rows),"n_val":len(val_rows),
          "n_test":len(test_rows),"flip_aug":True,
          "test_by_dataset":cd(test_rows),"target_order":list(RIM_NAMES),
          "abn_train_class_counts":cc(tr_aug)}
    return (dl(RimFlipDataset(tr_aug,PHOTOS_DIR,tf),True),
            dl(RimFlipDataset(val_rows,PHOTOS_DIR,tf),False),
            dl(RimFlipDataset(test_rows,PHOTOS_DIR,tf),False), info)

def abn_class_weights(info):
    import numpy as np
    counts=info["abn_train_class_counts"]; W=np.ones((4,N_ABN_CLASSES),dtype="float32")
    for j in range(4):
        c=np.maximum(np.array([counts[j].get(k,0) for k in range(N_ABN_CLASSES)],dtype="float64"),1.0)
        W[j]=(c.sum()/(N_ABN_CLASSES*c)).astype("float32")
    return torch.tensor(W)

if __name__=="__main__":
    tr,va,te,info=build_dataloaders(batch_size=8,num_workers=0)
    print("info:",info)
    xb,rim,abn,ids,ds=next(iter(tr))
    print("batch x",tuple(xb.shape),"rim",tuple(rim.shape),"abn",tuple(abn.shape))

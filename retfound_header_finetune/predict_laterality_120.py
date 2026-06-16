"""
Run the trained laterality (flip/OD-OS) head on the 120 COT_Eyes test images.

For each test image: image -> frozen RETFound -> pool -> head -> 2 logits.
class 0 = "R" (REFUGE original orientation, fovea RIGHT of disc),
class 1 = "L" (horizontally-flipped == opposite eye).

Ensembles all 5 pool tiers (majority vote + mean prob) if present, and also
reports each tier, so we can see agreement. Outputs predict_laterality_120.csv.

This is a TRANSFER test: the head was trained on REFUGE full-fundus flip
synthesis; the 120 are disc-cropped real L/R eyes. Output is to be eyeballed.
"""
import os, sys, glob, csv, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retfound_backbone import load_retfound
from baseline import BaselinePool
from tier1 import Tier1Pool
from tier2a import Tier2aPool
from tier2b import Tier2bPool
from tier3 import Tier3Pool
import torch.nn as nn

POOLS = {
    "baseline": lambda: BaselinePool(1024, mode="mean"),
    "tier1":    lambda: Tier1Pool(1024),
    "tier2a":   lambda: Tier2aPool(1024, hidden=256, gated=False),
    "tier2b":   lambda: Tier2bPool(1024),
    "tier3":    lambda: Tier3Pool(1024, num_heads=8),
}
CLASS_NAMES = ("R", "L")

class LatHead(nn.Module):
    def __init__(self, in_dim=1024):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 2))
    def forward(self, x): return self.net(x)

IMAGENET_MEAN=(0.485,0.456,0.406); IMAGENET_STD=(0.229,0.224,0.225)
tf = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(IMAGENET_MEAN,IMAGENET_STD)])

def find_ckpts():
    out={}
    for tier in POOLS:
        cands=sorted(glob.glob(str(HERE/f"outputs/lat_{tier}_*/best.pt")))
        if cands: out[tier]=cands[-1]
    return out

def load_model(tier, ckpt, device):
    bb=load_retfound(freeze=True, verbose=False)
    pool=POOLS[tier]()
    head=LatHead(1024)
    ck=torch.load(ckpt, map_location=device, weights_only=False)
    pool.load_state_dict(ck["pool_state"]); head.load_state_dict(ck["head_state"])
    bb=bb.to(device).eval(); pool=pool.to(device).eval(); head=head.to(device).eval()
    return bb,pool,head

def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    # locate test photos
    test_dir=None
    for c in [HERE.parent/"COT_Eyes/Dataset/test/photos",
              HERE.parent/"qwen35_icl_glaucoma/DATA/test_120/photos",
              Path(os.environ.get("TEST_SPLIT_DIR",""))]:
        if c and c.exists(): test_dir=c; break
    assert test_dir, "no test photos dir found"
    imgs=sorted(glob.glob(str(test_dir/"*.jpg")))
    print(f"device={device}  test_dir={test_dir}  n={len(imgs)}")

    ckpts=find_ckpts()
    print("tiers:", list(ckpts.keys()))
    # cache tokens once via baseline path? backbone is shared (frozen identical),
    # but each tier has its own pool; just load all models.
    models={t:load_model(t,p,device) for t,p in ckpts.items()}

    rows=[]
    with torch.no_grad():
        for ip in imgs:
            sid=Path(ip).stem
            x=tf(Image.open(ip).convert("RGB")).unsqueeze(0).to(device)
            per={}; probsL=[]
            # backbone tokens identical across tiers (same frozen weights) -> compute once
            anybb=next(iter(models.values()))[0]
            tok=anybb.forward_tokens(x)
            for t,(bb,pool,head) in models.items():
                pooled,_=pool(tok)
                logit=head(pooled)
                p=torch.softmax(logit,-1)[0]
                pred=int(p.argmax()); per[t]=(CLASS_NAMES[pred], float(p[1]))  # prob of L
                probsL.append(float(p[1]))
            meanL=float(np.mean(probsL))
            votes=[per[t][0] for t in per]
            nL=votes.count("L"); nR=votes.count("R")
            ens="L" if nL>nR else "R"
            rows.append({"id":sid,"ensemble":ens,"meanProb_L":round(meanL,3),
                         "votes_L":nL,"votes_R":nR,
                         **{f"{t}":per[t][0] for t in per},
                         **{f"{t}_pL":round(per[t][1],3) for t in per}})
    # write
    outp=HERE/"predict_laterality_120.csv"
    cols=list(rows[0].keys())
    with open(outp,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)
    nL=sum(1 for r in rows if r["ensemble"]=="L"); nR=len(rows)-nL
    print(f"wrote {outp}  ensemble: L={nL} R={nR}")
    # agreement stats
    full=sum(1 for r in rows if r["votes_L"] in (0,5))
    print(f"unanimous (5/5) cases: {full}/{len(rows)}")

if __name__=="__main__":
    main()

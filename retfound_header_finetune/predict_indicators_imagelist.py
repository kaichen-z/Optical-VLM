#!/usr/bin/env python3
"""Run the v2_cw indicator heads (CDR tier2a numeric / rim orderabn-tier3 /
signs tier1) on an ARBITRARY image folder (no manifest/GT needed) and dump
numeric predicted indicators in the SAME format as
predicted_indicators_test_newheads.json. Used for external REFUGE-400 test.

  python predict_indicators_imagelist.py --images '/data/.../REFUGE2/test/images/*.jpg' \
      --out ../qwen35_icl_glaucoma/DATA/predicted_indicators_refuge_test.json
"""
import argparse, glob, json
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from retfound_backbone import load_retfound
import build_predicted_indicators as B   # SIGN_KEYS / SIGN_OUT / IDX2LBL

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224
TF = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
RIM_LETTERS = ["I", "S", "N", "T"]
RIM_NAMES = ["Inferior", "Superior", "Nasal", "Temporal"]
ABN_LBL = {0: "normal", 1: "mild thinning", 2: "severe thinning"}


class ImgDS(torch.utils.data.Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        return TF(Image.open(p).convert("RGB")), Path(p).stem


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="glob of image files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cdr-ck", default="outputs/cdr_tier2a_full613/best.pt")
    ap.add_argument("--rim-ck", default="outputs/orderabn_tier3_w1/best.pt")
    ap.add_argument("--sig-ck", default="outputs/signs_tier1_full613/best.pt")
    args = ap.parse_args()
    paths = sorted(glob.glob(args.images))
    assert paths, f"no images match {args.images}"
    print(f"{len(paths)} images")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bb = load_retfound(freeze=True).to(dev).eval()

    def loader():
        return torch.utils.data.DataLoader(ImgDS(paths), batch_size=32,
                                           shuffle=False, num_workers=4)
    pred = {}

    # CDR (tier2a numeric)
    from train import CDRModel, POOLS as CP
    from heads import build_head
    ck = torch.load(args.cdr_ck, map_location=dev, weights_only=False)
    pool = CP[ck["pool"]](); head = build_head(ck["head"])
    pool.load_state_dict(ck["pool_state"]); head.load_state_dict(ck["head_state"])
    m = CDRModel(bb, pool, head).to(dev).eval()
    for x, ids in loader():
        p, _ = m(x.to(dev)); p = p.cpu().numpy()
        for k, sid in enumerate(ids):
            d = pred.setdefault(sid, {})
            d["vertical_CDR"] = round(float(p[k, 0]), 3)
            d["horizontal_CDR"] = round(float(p[k, 1]), 3)

    # RIM (orderabn tier3)
    from train_order_abn import OrderHead, AbnHead, JointModel, POOLS as RP
    ck = torch.load(args.rim_ck, map_location=dev, weights_only=False)
    pool = RP[ck["pool"]](); oh = OrderHead(1024, kind=ck["head"]); ah = AbnHead(1024, kind=ck["head"])
    pool.load_state_dict(ck["pool_state"])
    oh.load_state_dict(ck["order_head_state"]); ah.load_state_dict(ck["abn_head_state"])
    m = JointModel(bb, pool, oh, ah, lora=False, lr_inject=False).to(dev).eval()
    for x, ids in loader():
        s, al = m(x.to(dev)); s = s.cpu().numpy(); ap2 = al.argmax(-1).cpu().numpy()
        for k, sid in enumerate(ids):
            order = np.argsort(-s[k])
            st = {RIM_NAMES[j]: ABN_LBL[int(ap2[k, j])] for j in range(4)}
            d = pred.setdefault(sid, {})
            d["ISNT_observed_order"] = ">".join(RIM_LETTERS[j] for j in order)
            d["ISNT_quadrant_status"] = st
            d["ISNT_quadrant_status_str"] = "; ".join(f"{q}: {st[q]}" for q in RIM_NAMES)

    # SIGNS (tier1)
    from train_signs import SignsModel, SignsHead, POOLS as SP
    ck = torch.load(args.sig_ck, map_location=dev, weights_only=False)
    pool = SP[ck["pool"]](); head = SignsHead(1024, kind=ck["head"])
    pool.load_state_dict(ck["pool_state"]); head.load_state_dict(ck["head_state"])
    m = SignsModel(bb, pool, head).to(dev).eval()
    for x, ids in loader():
        c = m(x.to(dev)).argmax(-1).cpu().numpy()
        for k, sid in enumerate(ids):
            d = pred.setdefault(sid, {})
            for j, key in enumerate(B.SIGN_KEYS):
                d[B.SIGN_OUT[key]] = B.IDX2LBL.get(int(c[k, j]), "absent")
            d["Disc_hemorrhage"] = "absent"

    Path(args.out).write_text(json.dumps(pred, indent=1, ensure_ascii=False))
    print(f"wrote {args.out}  ({len(pred)} ids)")


if __name__ == "__main__":
    main()

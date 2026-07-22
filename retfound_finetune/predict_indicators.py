"""Run a trained multi-head checkpoint over a folder of fundus images and dump
the predicted indicators (four intermediate measurements + a diagnosis
probability) as JSON, keyed by image id. This is the file the VLM stage consumes.

    python predict_indicators.py --ckpt outputs/tier2a_linear/best.pt \
        --images 'data/images/*.jpg' --out predicted_indicators.json

The load_stage1 / predict_batch helpers are reused by the top-level inference.py.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from retfound_backbone import load_retfound
from poolings import make_pool
from heads import OrderHead, MultiClsHead, CdrRegHead, DxHead
from dataset import (RIM_NAMES, SIGN_NAMES, N_Q, N_ABN_CLASSES,
                     N_SIGN, N_SIGN_CLASSES, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD)

RIM_LETTER = {"Inferior": "I", "Superior": "S", "Nasal": "N", "Temporal": "T"}
ABN_LABEL = {0: "normal", 1: "mild thinning", 2: "severe thinning"}
SIGN_LABEL = {0: "absent", 1: "present", 2: "uncertain"}
SIGN_FIELD = {"Notching": "Notching", "RNFL defect": "RNFL_defect",
              "Bayoneting": "Bayoneting_sign", "Beta-zone atrophy": "Beta_zone_atrophy",
              "Peripapillary atrophy": "Peripapillary_atrophy"}
TF = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def confidence(prob):
    # Distance from the decision boundary, bucketed into words the VLM can use.
    d = abs(prob - 0.5)
    return "high" if d >= 0.35 else "moderate" if d >= 0.15 else "low"


def load_stage1(ckpt_path, device):
    """Return (backbone, heads) ready for inference from a training checkpoint."""
    backbone = load_retfound(freeze=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    dim = 1024
    heads = torch.nn.ModuleDict({
        "pool": make_pool(ckpt["pool"], dim),
        "order_head": OrderHead(dim, N_Q, ckpt["head"]),
        "abn_head": MultiClsHead(dim, N_Q, N_ABN_CLASSES, ckpt["head"]),
        "cdr_head": CdrRegHead(dim, kind=ckpt["head"]),
        "signs_head": MultiClsHead(dim, N_SIGN, N_SIGN_CLASSES, ckpt["head"]),
        "dx_head": DxHead(dim, ckpt["head"]),
    })
    for name in heads:
        heads[name].load_state_dict(ckpt[f"{name}_state"])
    return backbone.to(device).eval(), heads.to(device).eval()


@torch.no_grad()
def predict_batch(backbone, heads, images):
    """images: [B,3,224,224] tensor -> list of indicator dicts."""
    tokens = backbone.forward_tokens(images)
    pooled, _ = heads["pool"](tokens)
    order = heads["order_head"](pooled).cpu().numpy()
    abn = heads["abn_head"](pooled).argmax(-1).cpu().numpy()
    cdr = heads["cdr_head"](pooled).cpu().numpy()
    sign = heads["signs_head"](pooled).argmax(-1).cpu().numpy()
    dx_prob = torch.softmax(heads["dx_head"](pooled), -1)[:, 1].cpu().numpy()

    recs = []
    for i in range(len(order)):
        rank = np.argsort(-order[i])
        quad = {RIM_NAMES[q]: ABN_LABEL[int(abn[i, q])] for q in range(N_Q)}
        rec = {
            "vertical_CDR": round(float(cdr[i, 0]), 3),
            "horizontal_CDR": round(float(cdr[i, 1]), 3),
            "ISNT_observed_order": " > ".join(RIM_LETTER[RIM_NAMES[q]] for q in rank),
            "ISNT_quadrant_status": quad,
            "ISNT_quadrant_status_str": ", ".join(f"{k}: {v}" for k, v in quad.items()),
            "glaucoma_probability": round(float(dx_prob[i]), 3),
            "glaucoma_confidence": confidence(float(dx_prob[i])),
        }
        for j, name in enumerate(SIGN_NAMES):
            rec[SIGN_FIELD[name]] = SIGN_LABEL[int(sign[i, j])]
        recs.append(rec)
    return recs


class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return TF(Image.open(self.paths[i]).convert("RGB")), Path(self.paths[i]).stem


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", required=True, help="glob of image files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone, heads = load_stage1(args.ckpt, device)
    loader = torch.utils.data.DataLoader(
        ImageFolder(sorted(glob.glob(args.images))), batch_size=args.batch_size, num_workers=4)

    out = {}
    for imgs, ids in loader:
        for sid, rec in zip(ids, predict_batch(backbone, heads, imgs.to(device))):
            out[sid] = rec

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {len(out)} predictions -> {args.out}")


if __name__ == "__main__":
    main()

"""Paths and backbone constants. Override paths with env vars or CLI args."""
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# RETFound MAE ViT-L/16 natureCFP weights (.pth). Point RETFOUND_CKPT at yours.
RETFOUND_CKPT = Path(os.environ.get("RETFOUND_CKPT", _HERE / "weights" / "RETFound_mae_natureCFP.pth"))

# ViT-Large/16
PATCH_SIZE = 16
EMBED_DIM = 1024
DEPTH = 24
NUM_HEADS = 16
IMG_SIZE = 224
NUM_TOKENS = (IMG_SIZE // PATCH_SIZE) ** 2 + 1   # 197 (196 patches + CLS)

"""
Central config for the RETFound header-finetune project.

All paths resolve relative to THIS file unless overridden by env vars, so the
whole `retfound_header_finetune/` folder is portable: push it to any server,
no path editing needed.

Env overrides (set these if your layout differs):
    RETFOUND_CKPT      path to RETFound_mae_natureCFP.pth
    COT_EYES_ROOT      path to the COT_Eyes/ data tree
"""
from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# --- RETFound backbone weights ---
# Default: weights/ subfolder inside this package (push.sh copies the .pth here).
# Verified SHA256: e1e4f66a1b792eeb6e2efaf158f33be35c8255f36b3d17ed67cd5129da246485
RETFOUND_CKPT = Path(
    os.environ.get("RETFOUND_CKPT", _HERE / "weights" / "RETFound_mae_natureCFP.pth")
)
RETFOUND_CONFIG = Path(
    os.environ.get("RETFOUND_CONFIG", _HERE / "weights" / "config.json")
)

# --- Data (full 733: 245 LAG + 488 Papila) ---
# Resolution order:
#   1. env COT_EYES_ROOT  (explicit override)
#   2. ./DATA/ self-contained subfolder  (what push.sh creates on the server)
#   3. ../COT_Eyes/Dataset/all  (laptop layout, sibling folder)
_DATA_LOCAL = _HERE / "DATA"
if os.environ.get("COT_EYES_ROOT"):
    COT_EYES_ROOT = Path(os.environ["COT_EYES_ROOT"])
    PHOTOS_DIR = COT_EYES_ROOT / "Dataset" / "all" / "photos"
    DESCRIPTIONS_DIR = COT_EYES_ROOT / "Dataset" / "all" / "descriptions"
elif (_DATA_LOCAL / "photos").exists():
    COT_EYES_ROOT = _DATA_LOCAL          # self-contained DATA/ on the server
    PHOTOS_DIR = _DATA_LOCAL / "photos"
    DESCRIPTIONS_DIR = _DATA_LOCAL / "descriptions"
else:
    COT_EYES_ROOT = _HERE.parent / "COT_Eyes"   # laptop layout
    PHOTOS_DIR = COT_EYES_ROOT / "Dataset" / "all" / "photos"
    DESCRIPTIONS_DIR = COT_EYES_ROOT / "Dataset" / "all" / "descriptions"

# Canonical 120-image balanced TEST split source. Used at preprocessing time
# to label each manifest row split=test/train. The split is then BAKED INTO
# cdr_manifest.csv, so downstream training never needs this dir again
# (manifest is fully portable to any server).
#   - laptop:  COT_Eyes/Dataset/test/photos  (Grace's 120, == retfound_imagefolder/test)
#   - else:    env TEST_SPLIT_DIR override
if os.environ.get("TEST_SPLIT_DIR"):
    TEST_SPLIT_PHOTOS = Path(os.environ["TEST_SPLIT_DIR"])
else:
    TEST_SPLIT_PHOTOS = (_HERE.parent / "COT_Eyes" / "Dataset" / "test" / "photos")

CDR_MANIFEST = _HERE / "cdr_manifest.csv"
# env overrides (used for K-fold OOF: point each head at a fold-specific manifest)
ISNT_MANIFEST = Path(os.environ.get("ISNT_MANIFEST", str(_HERE / "isnt_manifest.csv")))   # 4 rim-thickness ratios (I/S/N/T)
SIGNS_MANIFEST = Path(os.environ.get("SIGNS_MANIFEST", str(_HERE / "signs_manifest_clean.csv")))  # 5 glaucomatous signs (3-cls)

# --- RETFound architecture (ViT-Large/16, from RETFound_mae natureCFP) ---
RETFOUND_ARCH = dict(
    patch_size=16,
    embed_dim=1024,
    depth=24,
    num_heads=16,
    mlp_ratio=4,
    img_size=224,
)
NUM_TOKENS = (RETFOUND_ARCH["img_size"] // RETFOUND_ARCH["patch_size"]) ** 2 + 1  # 197
EMBED_DIM = RETFOUND_ARCH["embed_dim"]                                            # 1024


def sanity_check(verbose: bool = True) -> bool:
    """Quick existence check for the things training needs."""
    ok = True
    for name, p in [
        ("RETFOUND_CKPT", RETFOUND_CKPT),
        ("PHOTOS_DIR", PHOTOS_DIR),
        ("DESCRIPTIONS_DIR", DESCRIPTIONS_DIR),
    ]:
        exists = p.exists()
        ok &= exists
        if verbose:
            print(f"  [{'OK ' if exists else 'MISSING'}] {name}: {p}")
    return ok


if __name__ == "__main__":
    print("retfound_header_finetune config:")
    print(f"  NUM_TOKENS = {NUM_TOKENS}, EMBED_DIM = {EMBED_DIM}")
    sanity_check()

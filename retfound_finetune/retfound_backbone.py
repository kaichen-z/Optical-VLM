"""RETFound (MAE ViT-L/16) backbone, frozen. forward_tokens returns the full
[B, 197, 1024] token sequence after the transformer blocks (no final norm,
no pooling) so each pooling module works from identical raw tokens."""
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import timm.models.vision_transformer

from config import RETFOUND_CKPT


def interpolate_pos_embed(model, checkpoint_model):
    # No-op when the input size matches the 224 pretrain size; kept for safety.
    if "pos_embed" not in checkpoint_model:
        return
    pos_embed = checkpoint_model["pos_embed"]
    dim = pos_embed.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra = model.pos_embed.shape[-2] - num_patches
    orig = int((pos_embed.shape[-2] - num_extra) ** 0.5)
    new = int(num_patches ** 0.5)
    if orig == new:
        return
    extra = pos_embed[:, :num_extra]
    grid = pos_embed[:, num_extra:].reshape(-1, orig, orig, dim).permute(0, 3, 1, 2)
    grid = torch.nn.functional.interpolate(grid, size=(new, new), mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).flatten(1, 2)
    checkpoint_model["pos_embed"] = torch.cat((extra, grid), dim=1)


class RETFoundViT(timm.models.vision_transformer.VisionTransformer):
    def forward_tokens(self, x):
        # Return all tokens after the blocks: [B, 197, 1024]. Pooling owns the norm.
        b = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward(self, x):
        return self.forward_tokens(x)


def _build_vit():
    return RETFoundViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0,
    )


def load_retfound(ckpt_path=None, freeze=True, verbose=True):
    ckpt_path = Path(ckpt_path or RETFOUND_CKPT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"RETFound checkpoint not found: {ckpt_path}")

    model = _build_vit()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    for old, new in (("backbone.", ""), ("mlp.w12.", "mlp.fc1."), ("mlp.w3.", "mlp.fc2.")):
        sd = {k.replace(old, new): v for k, v in sd.items()}
    for k in ("head.weight", "head.bias"):
        sd.pop(k, None)

    interpolate_pos_embed(model, sd)
    res = model.load_state_dict(sd, strict=False)
    if verbose:
        miss = [k for k in res.missing_keys if not k.startswith("head.")]
        print(f"[RETFound] loaded {ckpt_path.name}  missing(non-head)={len(miss)}")

    if freeze:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
    return model


if __name__ == "__main__":
    m = load_retfound(freeze=True)
    with torch.no_grad():
        t = m.forward_tokens(torch.randn(2, 3, 224, 224))
    print("forward_tokens ->", tuple(t.shape))   # (2, 197, 1024)

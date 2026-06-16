"""
RETFound (MAE ViT-L/16) backbone loader.

Mirrors EXACTLY how Grace's external_RETFound loads the weights, so the
backbone is bit-for-bit the same starting point as her experiments:

  * arch: timm VisionTransformer, patch16 / embed1024 / depth24 / heads16 /
          mlp_ratio4 / qkv_bias / LayerNorm(eps=1e-6)         == RETFound_mae()
  * load: torch.load(...)["model"], strip "backbone." prefix,
          drop shape-mismatched head.*, interpolate_pos_embed,
          load_state_dict(strict=False)

Difference from RETFound's models_vit.py: our forward returns the FULL token
sequence [B, 197, 1024] right after the transformer blocks (NO final norm,
NO pooling). The pooling module (baseline/tier1/2a/2b/3) is responsible for
its own normalization + collapse. This keeps the 5-pooling ablation clean
(every pooling sees identical raw tokens; backbone frozen).

Requires `timm` (same dependency as Grace's external_RETFound).
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import timm.models.vision_transformer
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "timm is required for the RETFound backbone (same as Grace's "
        "external_RETFound). Install: pip install timm"
    ) from e

try:
    from config import RETFOUND_CKPT
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import RETFOUND_CKPT


# ---------------------------------------------------------------------------
# interpolate_pos_embed — copied verbatim from RETFound util/pos_embed.py
# (no-op when input size == pretrain size 224, but kept for fidelity/safety)
# ---------------------------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)
        if orig_size != new_size:
            print(f"Position interpolate {orig_size}x{orig_size} -> {new_size}x{new_size}")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size),
                mode="bicubic", align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            checkpoint_model["pos_embed"] = torch.cat(
                (extra_tokens, pos_tokens), dim=1)


# ---------------------------------------------------------------------------
# RETFound ViT — same as external_RETFound/models_vit.py but forward_tokens()
# returns the full [B, 197, 1024] sequence after the blocks.
# ---------------------------------------------------------------------------
class RETFoundViT(timm.models.vision_transformer.VisionTransformer):
    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return ALL tokens after the transformer blocks: [B, 197, 1024].
        (No final norm, no pooling — the pooling module owns those.)"""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return x                       # [B, 197, 1024]

    # keep forward == forward_tokens so nn.Sequential / hooks behave sanely
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(x)


def _build_vit() -> RETFoundViT:
    # identical hyperparams to RETFound_mae()
    return RETFoundViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0,        # no classifier head; we add our own pooling+head
    )


def load_retfound(ckpt_path: str | Path | None = None,
                   freeze: bool = True,
                   lora: bool = False,
                   lora_rank: int = 16,
                   verbose: bool = True) -> RETFoundViT:
    """Build ViT-L/16 and load RETFound_mae natureCFP weights the same way
    Grace's external_RETFound/main_finetune.py does."""
    ckpt_path = Path(ckpt_path or RETFOUND_CKPT)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"RETFound checkpoint not found: {ckpt_path}\n"
            f"Set RETFOUND_CKPT env or place the .pth at weights/."
        )

    model = _build_vit()

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    checkpoint_model = checkpoint["model"] if "model" in checkpoint else checkpoint
    # key hygiene (same as Grace)
    checkpoint_model = {k.replace("backbone.", ""): v
                        for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("mlp.w12.", "mlp.fc1."): v
                        for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("mlp.w3.", "mlp.fc2."): v
                        for k, v in checkpoint_model.items()}
    # drop shape-mismatched classifier keys (we have num_classes=0 anyway)
    sd = model.state_dict()
    for k in ("head.weight", "head.bias"):
        if k in checkpoint_model and (k not in sd
                                      or checkpoint_model[k].shape != sd[k].shape):
            checkpoint_model.pop(k, None)

    interpolate_pos_embed(model, checkpoint_model)
    missing_unexpected = model.load_state_dict(checkpoint_model, strict=False)

    if verbose:
        miss = [k for k in missing_unexpected.missing_keys
                if not k.startswith(("head.",))]
        unexp = [k for k in missing_unexpected.unexpected_keys
                 if not k.startswith(("decoder", "mask_token"))]
        print(f"[RETFound] loaded from {ckpt_path}")
        print(f"           missing(non-head)={len(miss)}  "
              f"unexpected(non-decoder)={len(unexp)}")
        if miss:
            print(f"           e.g. missing: {miss[:5]}")
        if unexp:
            print(f"           e.g. unexpected: {unexp[:5]}")

    if lora:
        # Freeze ALL base weights, then inject LoRA adapters (only the
        # low-rank A/B become trainable). peft replaces the target Linear
        # submodules IN-PLACE inside `model`, so forward_tokens() is
        # unchanged and grads flow through the adapters. We keep/return the
        # original `model` object (peft mutates it in place).
        from peft import LoraConfig, get_peft_model
        for p in model.parameters():
            p.requires_grad = False
        cfg = LoraConfig(
            r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.05,
            bias="none",
            # timm ViT block Linear suffixes ONLY. Use "attn.proj" (not bare
            # "proj") so we do NOT also wrap patch_embed.proj (the patch-embed
            # Conv2d) — keep this a conventional attention+MLP ViT-LoRA.
            target_modules=["qkv", "attn.proj", "fc1", "fc2"],
        )
        get_peft_model(model, cfg)            # in-place adapter injection
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        if verbose:
            print(f"[RETFound] LoRA r={lora_rank} injected "
                  f"(qkv/proj/fc1/fc2); base FROZEN. trainable "
                  f"{n_tr/1e6:.3f}M / {n_all/1e6:.1f}M "
                  f"({100*n_tr/max(n_all,1):.2f}%)")
        return model

    if freeze:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        if verbose:
            print("[RETFound] backbone FROZEN (eval mode, no grad)")

    return model


if __name__ == "__main__":
    m = load_retfound(freeze=True)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        t = m.forward_tokens(x)
    n_total = sum(p.numel() for p in m.parameters())
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"forward_tokens out = {tuple(t.shape)}  (expect (2,197,1024))")
    print(f"params total={n_total/1e6:.1f}M  trainable={n_train/1e6:.1f}M")

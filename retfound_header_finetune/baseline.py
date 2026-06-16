"""
Baseline pooling — replicates RETFound's own default pooling.

This is the reference point the learnable-attention poolings (tier1/2a/2b/3)
must beat. Grace's RETFound runs used `--global_pool`, which in
RETFound's models_vit.py means:

    x = x[:, 1:, :].mean(dim=1)   # mean over the 196 patch tokens (CLS excluded)
    x = fc_norm(x)                # a LayerNorm

So `mode="mean"` here == Grace's exact baseline. `mode="cls"` is RETFound's
other option (global_pool=False): take the CLS token then LayerNorm.

Shared interface (same for every file in this folder):
    forward(tokens) -> (pooled, attn)
        tokens : [B, 197, 1024]   raw RETFound transformer-block output
                                  (197 = 1 CLS + 196 patches)
        pooled : [B, 1024]        the single fused vector
        attn   : [B, 197]         per-token weight (for heat-map viz)

NO classification / FC head here on purpose — the head (1024->2 or
1024->512->2) is a separate swappable module so pooling x head can be
ablated combinatorially.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BaselinePool(nn.Module):
    def __init__(self, dim: int = 1024, mode: str = "mean"):
        super().__init__()
        assert mode in ("mean", "cls"), "mode must be 'mean' or 'cls'"
        self.mode = mode
        self.dim = dim
        # RETFound applies a LayerNorm in both paths (fc_norm for global_pool,
        # self.norm for the CLS path). One LayerNorm here covers both.
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor):
        # tokens: [B, N, D]   N = 197
        B, N, D = tokens.shape

        if self.mode == "mean":
            pooled = tokens[:, 1:, :].mean(dim=1)          # [B, D]  patches only
            pooled = self.norm(pooled)
            # uniform weight over the 196 patches, 0 on CLS (for viz consistency)
            attn = torch.zeros(B, N, device=tokens.device, dtype=tokens.dtype)
            attn[:, 1:] = 1.0 / (N - 1)
        else:  # "cls"
            pooled = self.norm(tokens[:, 0])               # [B, D]  CLS token
            attn = torch.zeros(B, N, device=tokens.device, dtype=tokens.dtype)
            attn[:, 0] = 1.0

        return pooled, attn


if __name__ == "__main__":
    x = torch.randn(2, 197, 1024)
    for m in ("mean", "cls"):
        pool = BaselinePool(1024, mode=m)
        p, a = pool(x)
        print(f"[baseline:{m}]  pooled={tuple(p.shape)}  attn={tuple(a.shape)}  "
              f"attn_sum={a.sum(1).tolist()}")

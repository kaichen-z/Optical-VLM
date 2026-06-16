"""
Tier 2b — Single-query attention pooling (Route B, medium).

This IS the scaled dot-product attention from Transformers/LLMs, used
once as a pooling layer with a single learnable query token.

    q                learnable query, [1024]
    K_i = W_k . x_i  key  projection,  W_k: [1024,1024]   "matching label"
    V_i = W_v . x_i  value projection, W_v: [1024,1024]   "content"
    score_i = (q . K_i) / sqrt(d)        [B, 197]
    a       = softmax(scores)            [B, 197]
    pooled  = Σ a_i * V_i                [B, 1024]   weighted sum of VALUES

Key difference vs Route A (tier1/2a): the token is split into K (used
only for relevance) and V (used only for the aggregated content). This
decoupling lets the model use one set of features to decide *where to
look* and a different set for *what to extract*.

Shared interface:
    forward(tokens[B,197,1024]) -> (pooled[B,1024], attn[B,197])

NO FC head here (separate swappable module).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Tier2bPool(nn.Module):
    def __init__(self, dim: int = 1024):
        super().__init__()
        self.dim = dim
        self.q = nn.Parameter(torch.randn(dim) * 0.02)   # [1024]
        self.W_k = nn.Linear(dim, dim, bias=False)        # [1024,1024]
        self.W_v = nn.Linear(dim, dim, bias=False)        # [1024,1024]
        self.scale = dim ** 0.5                            # sqrt(1024) = 32

    def forward(self, tokens: torch.Tensor):
        # tokens: [B, N, D]
        K = self.W_k(tokens)                           # [B, N, D]
        V = self.W_v(tokens)                           # [B, N, D]
        scores = (K @ self.q) / self.scale             # [B, N]
        attn = torch.softmax(scores, dim=1)            # [B, N]
        pooled = (attn.unsqueeze(-1) * V).sum(1)       # [B, D]
        return pooled, attn


if __name__ == "__main__":
    x = torch.randn(2, 197, 1024)
    pool = Tier2bPool(1024)
    p, a = pool(x)
    n_params = sum(pp.numel() for pp in pool.parameters())
    print(f"[tier2b]  pooled={tuple(p.shape)}  attn={tuple(a.shape)}  "
          f"attn_sum={a.sum(1).tolist()}  params={n_params}")

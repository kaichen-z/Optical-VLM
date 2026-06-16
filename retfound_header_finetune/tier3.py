"""
Tier 3 — Multi-head single-query attention pooling (Route B, most expressive).

Exactly tier2b done in H parallel heads, each working in a dim/H = 128-d
subspace, then concatenated and passed through an output projection W_o.
This is the Transformer's multi-head attention used as a pooling layer
(a.k.a. PMA in Set Transformer / the Perceiver-style learned-query pool).

Per head h (H = 8, d_head = 1024/8 = 128):
    q^h            slice of the learnable query, [128]
    K^h = W_k^h x  [B, 197, 128]
    V^h = W_v^h x  [B, 197, 128]
    score^h = (q^h . K^h) / sqrt(d_head)   [B, 197]   <- each head its OWN map
    a^h     = softmax(score^h)             [B, 197]
    out^h   = Σ a^h * V^h                  [B, 128]
concat 8 heads -> [B, 1024] -> W_o -> [B, 1024]

Returned `attn` is the per-head attention averaged over heads ([B,197])
so it plugs into the same heat-map viz as the other tiers. (Per-head maps
are available as `attn_per_head` if you want all 8.)

Shared interface:
    forward(tokens[B,197,1024]) -> (pooled[B,1024], attn[B,197])

NO FC head here (separate swappable module).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Tier3Pool(nn.Module):
    def __init__(self, dim: int = 1024, num_heads: int = 8):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.h = num_heads
        self.dh = dim // num_heads                       # 128
        self.q = nn.Parameter(torch.randn(dim) * 0.02)   # [1024] -> view [h, dh]
        self.W_k = nn.Linear(dim, dim, bias=False)        # [1024,1024]
        self.W_v = nn.Linear(dim, dim, bias=False)        # [1024,1024]
        self.W_o = nn.Linear(dim, dim)                    # output projection
        self.scale = self.dh ** 0.5                        # sqrt(128)
        self.attn_per_head = None                          # cached [B,h,N] after fwd

    def forward(self, tokens: torch.Tensor):
        # tokens: [B, N, D]
        B, N, D = tokens.shape
        K = self.W_k(tokens).view(B, N, self.h, self.dh).transpose(1, 2)  # [B,h,N,dh]
        V = self.W_v(tokens).view(B, N, self.h, self.dh).transpose(1, 2)  # [B,h,N,dh]
        q = self.q.view(self.h, self.dh)                                  # [h,dh]

        # per-head scores: contract over dh
        scores = torch.einsum("bhnd,hd->bhn", K, q) / self.scale          # [B,h,N]
        a = torch.softmax(scores, dim=-1)                                  # [B,h,N]
        self.attn_per_head = a                                             # keep for viz

        out = torch.einsum("bhn,bhnd->bhd", a, V)                          # [B,h,dh]
        out = out.reshape(B, D)                                            # concat heads
        pooled = self.W_o(out)                                             # [B, D]

        attn = a.mean(dim=1)                                               # [B, N] avg over heads
        return pooled, attn


if __name__ == "__main__":
    x = torch.randn(2, 197, 1024)
    pool = Tier3Pool(1024, num_heads=8)
    p, a = pool(x)
    n_params = sum(pp.numel() for pp in pool.parameters())
    print(f"[tier3 h=8]  pooled={tuple(p.shape)}  attn={tuple(a.shape)}  "
          f"per_head={tuple(pool.attn_per_head.shape)}  "
          f"attn_sum={a.sum(1).tolist()}  params={n_params}")

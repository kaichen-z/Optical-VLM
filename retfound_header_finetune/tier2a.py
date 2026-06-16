"""
Tier 2a — Gated (MLP-scoring) attention pooling (Route A, medium).

Same skeleton as tier1 (per-token scalar score -> softmax -> weighted sum
of RAW tokens), but the scoring function is a 2-layer MLP with a tanh
nonlinearity instead of a single linear layer:

    score_i = w . tanh(V . x_i)         V: [1024->256], w: [256->1]
    a       = softmax(scores)           [B, 197]
    pooled  = Σ a_i * x_i               [B, 1024]   weighted sum of RAW tokens

The nonlinearity lets the scorer express conditional combinations
("bright AND edge AND not-vessel -> high score"), which a single linear
layer cannot. This is the Attention-MIL scorer (Ilse et al., 2018).

`gated=True` switches to Ilse's *true* gated form (tanh-branch ⊙
sigmoid-branch), which is where the name "Gated Attention" comes from:

    score_i = w . ( tanh(V . x_i) ⊙ sigmoid(U . x_i) )

Default is the plain tanh version (matches the math we walked through).

Shared interface:
    forward(tokens[B,197,1024]) -> (pooled[B,1024], attn[B,197])

NO FC head here (separate swappable module).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Tier2aPool(nn.Module):
    def __init__(self, dim: int = 1024, hidden: int = 256, gated: bool = False):
        super().__init__()
        self.dim = dim
        self.gated = gated
        self.V = nn.Linear(dim, hidden)        # [1024 -> 256]
        if gated:
            self.U = nn.Linear(dim, hidden)    # sigmoid gate branch
        self.w = nn.Linear(hidden, 1)          # [256 -> 1]

    def forward(self, tokens: torch.Tensor):
        # tokens: [B, N, D]
        h = torch.tanh(self.V(tokens))                 # [B, N, hidden]
        if self.gated:
            h = h * torch.sigmoid(self.U(tokens))      # element-wise gate
        scores = self.w(h).squeeze(-1)                 # [B, N]
        attn = torch.softmax(scores, dim=1)            # [B, N]
        pooled = (attn.unsqueeze(-1) * tokens).sum(1)  # [B, D]
        return pooled, attn


if __name__ == "__main__":
    x = torch.randn(2, 197, 1024)
    for g in (False, True):
        pool = Tier2aPool(1024, hidden=256, gated=g)
        p, a = pool(x)
        n_params = sum(pp.numel() for pp in pool.parameters())
        print(f"[tier2a gated={g}]  pooled={tuple(p.shape)}  attn={tuple(a.shape)}  "
              f"attn_sum={a.sum(1).tolist()}  params={n_params}")

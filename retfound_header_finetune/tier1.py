"""
Tier 1 — Linear-scoring attention pooling (Route A, simplest).

Each token gets ONE scalar importance score from a single linear layer
(content-based, NOT position-based: the same learned 1024-vector `w` is
dotted with every token):

    score_i = w . x_i              w is a learned [1024] vector (Linear 1024->1)
    a       = softmax(scores)      [B, 197]   sums to 1
    pooled  = Σ a_i * x_i          [B, 1024]  weighted sum of RAW tokens

No query, no K/V projections (that's Route B / tier2b). Aggregates the
raw tokens directly.

Shared interface:
    forward(tokens[B,197,1024]) -> (pooled[B,1024], attn[B,197])

NO FC head here (separate swappable module).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Tier1Pool(nn.Module):
    def __init__(self, dim: int = 1024):
        super().__init__()
        self.dim = dim
        self.score = nn.Linear(dim, 1)        # w: [1024] (+ bias)

    def forward(self, tokens: torch.Tensor):
        # tokens: [B, N, D]
        scores = self.score(tokens).squeeze(-1)        # [B, N]
        attn = torch.softmax(scores, dim=1)            # [B, N]
        pooled = (attn.unsqueeze(-1) * tokens).sum(1)  # [B, D]
        return pooled, attn


if __name__ == "__main__":
    x = torch.randn(2, 197, 1024)
    pool = Tier1Pool(1024)
    p, a = pool(x)
    n_params = sum(p.numel() for p in pool.parameters())
    print(f"[tier1]  pooled={tuple(p.shape)}  attn={tuple(a.shape)}  "
          f"attn_sum={a.sum(1).tolist()}  params={n_params}")

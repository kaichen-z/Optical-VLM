"""
Swappable FC prediction head for CDR regression.

Sits AFTER the pooling module (baseline/tier1/2a/2b/3), which already
collapsed [B,197,1024] -> [B,1024]. The head maps [B,1024] -> [B,2]
(vertical CDR, horizontal CDR) and applies sigmoid so outputs are in [0,1]
(CDR is physically bounded).

Two variants, picked combinatorially against the 5 poolings:
    "linear" :  Linear(1024 -> 2)                       (simplest, default)
    "mlp"    :  Linear(1024->512) -> GELU -> Dropout -> Linear(512->2)

forward(pooled[B,1024]) -> cdr[B,2]   each entry in (0,1) via sigmoid
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    def __init__(self, in_dim: int = 1024, out_dim: int = 2):
        super().__init__()
        # LayerNorm BEFORE the linear: the pooling modules return raw
        # (unnormalized) RETFound-token aggregates whose magnitude can be
        # large (24 transformer blocks, no final norm). Without this norm the
        # sigmoid saturates -> zero gradient -> loss frozen. Applied here so
        # ALL poolings get identical treatment (fair ablation).
        self.norm = nn.LayerNorm(in_dim)
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(self.norm(x)))      # [B, 2] in (0,1)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int = 1024, hidden: int = 512,
                 out_dim: int = 2, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),             # same reason as LinearHead
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))     # [B, 2] in (0,1)


def build_head(name: str = "linear", in_dim: int = 1024, out_dim: int = 2) -> nn.Module:
    name = name.lower()
    if name == "linear":
        return LinearHead(in_dim, out_dim)
    if name == "mlp":
        return MLPHead(in_dim, out_dim=out_dim)
    raise ValueError(f"unknown head '{name}' (use 'linear' or 'mlp')")


if __name__ == "__main__":
    x = torch.randn(4, 1024)
    for n in ("linear", "mlp"):
        h = build_head(n)
        y = h(x)
        p = sum(pp.numel() for pp in h.parameters())
        print(f"[{n}] in={tuple(x.shape)} -> out={tuple(y.shape)} "
              f"range=({y.min():.3f},{y.max():.3f}) params={p}")

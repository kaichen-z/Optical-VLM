"""Prediction heads that sit on top of a pooled [B, 1024] vector.

Each head has a `linear` and an `mlp` variant. The LayerNorm in front matters:
the pooling modules return raw (unnormalized) token aggregates whose magnitude
is large, so without it the sigmoid/softmax saturates and the loss stalls."""
import torch
import torch.nn as nn


def _head(in_dim, out_dim, kind, hidden=512, dropout=0.2):
    if kind == "linear":
        return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim))
    if kind == "mlp":
        return nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim),
        )
    raise ValueError(f"unknown head kind '{kind}' (use 'linear' or 'mlp')")


class OrderHead(nn.Module):
    """Four raw scores, one per rim sector, ranked to recover the ISNT order."""
    def __init__(self, in_dim=1024, n_quad=4, kind="linear"):
        super().__init__()
        self.net = _head(in_dim, n_quad, kind)

    def forward(self, x):
        return self.net(x)


class MultiClsHead(nn.Module):
    """n_ch independent classifiers sharing the trunk -> [B, n_ch, n_cls].
    Used for per-quadrant rim status and for the glaucomatous signs."""
    def __init__(self, in_dim, n_ch, n_cls, kind="linear"):
        super().__init__()
        self.n_ch, self.n_cls = n_ch, n_cls
        self.net = _head(in_dim, n_ch * n_cls, kind)

    def forward(self, x):
        return self.net(x).view(-1, self.n_ch, self.n_cls)


class CdrRegHead(nn.Module):
    """Vertical/horizontal CDR in [0, 1] via sigmoid."""
    def __init__(self, in_dim, n_cdr=2, kind="linear"):
        super().__init__()
        self.net = _head(in_dim, n_cdr, kind)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class DxHead(nn.Module):
    """Binary glaucoma logits -> [B, 2]."""
    def __init__(self, in_dim, kind="linear"):
        super().__init__()
        self.net = _head(in_dim, 2, kind)

    def forward(self, x):
        return self.net(x)

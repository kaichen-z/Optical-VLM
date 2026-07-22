"""Five pooling modules that collapse the RETFound token sequence
[B, 197, 1024] into a single vector [B, 1024]. Each returns (pooled, attn),
where attn is a per-token weight over the 197 tokens.

    baseline : mean over patch tokens + LayerNorm (RETFound's own global pool)
    tier1    : linear-scored attention pooling
    tier2a   : gated (MLP-scored) attention pooling  (Attention-MIL, Ilse 2018)
    tier2b   : single learnable-query scaled dot-product attention
    tier3    : multi-head learnable-query attention   (PMA, Set Transformer)
"""
import torch
import torch.nn as nn


class BaselinePool(nn.Module):
    def __init__(self, dim=1024, mode="mean"):
        super().__init__()
        assert mode in ("mean", "cls")
        self.mode = mode
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens):
        b, n, _ = tokens.shape
        attn = torch.zeros(b, n, device=tokens.device, dtype=tokens.dtype)
        if self.mode == "mean":
            pooled = self.norm(tokens[:, 1:, :].mean(dim=1))
            attn[:, 1:] = 1.0 / (n - 1)
        else:
            pooled = self.norm(tokens[:, 0])
            attn[:, 0] = 1.0
        return pooled, attn


class Tier1Pool(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, tokens):
        attn = torch.softmax(self.score(tokens).squeeze(-1), dim=1)
        return (attn.unsqueeze(-1) * tokens).sum(1), attn


class Tier2aPool(nn.Module):
    def __init__(self, dim=1024, hidden=256, gated=False):
        super().__init__()
        self.gated = gated
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden) if gated else None
        self.w = nn.Linear(hidden, 1)

    def forward(self, tokens):
        h = torch.tanh(self.V(tokens))
        if self.gated:
            h = h * torch.sigmoid(self.U(tokens))
        attn = torch.softmax(self.w(h).squeeze(-1), dim=1)
        return (attn.unsqueeze(-1) * tokens).sum(1), attn


class Tier2bPool(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.q = nn.Parameter(torch.randn(dim) * 0.02)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** 0.5

    def forward(self, tokens):
        k = self.W_k(tokens)
        v = self.W_v(tokens)
        attn = torch.softmax((k @ self.q) / self.scale, dim=1)
        return (attn.unsqueeze(-1) * v).sum(1), attn


class Tier3Pool(nn.Module):
    def __init__(self, dim=1024, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0
        self.h = num_heads
        self.dh = dim // num_heads
        self.q = nn.Parameter(torch.randn(dim) * 0.02)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim)
        self.scale = self.dh ** 0.5

    def forward(self, tokens):
        b, n, d = tokens.shape
        k = self.W_k(tokens).view(b, n, self.h, self.dh).transpose(1, 2)
        v = self.W_v(tokens).view(b, n, self.h, self.dh).transpose(1, 2)
        q = self.q.view(self.h, self.dh)
        scores = torch.einsum("bhnd,hd->bhn", k, q) / self.scale
        a = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhn,bhnd->bhd", a, v).reshape(b, d)
        return self.W_o(out), a.mean(dim=1)


POOL_NAMES = ["baseline", "tier1", "tier2a", "tier2b", "tier3"]


def make_pool(name, dim=1024):
    if name == "baseline":
        return BaselinePool(dim, mode="mean")
    if name == "tier1":
        return Tier1Pool(dim)
    if name == "tier2a":
        return Tier2aPool(dim, hidden=256, gated=False)
    if name == "tier2b":
        return Tier2bPool(dim)
    if name == "tier3":
        return Tier3Pool(dim, num_heads=8)
    raise ValueError(f"unknown pooling '{name}'")

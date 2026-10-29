"""Velocity field network for Conditional Flow Matching.

A residual MLP with sinusoidal time embedding and FiLM conditioning on the
physically meaningful variables (log10 incident energy, eta, phi).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["VelocityNet"]


def _sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal time embedding, t in [0, 1] -> (batch, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(1000.0) * torch.arange(half, device=t.device) / max(half - 1, 1)
    )
    ang = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class FiLMBlock(nn.Module):
    """Residual block modulated by (time, condition) embeddings (FiLM)."""

    def __init__(self, dim: int, emb_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.film = nn.Linear(emb_dim, 2 * dim)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(emb).chunk(2, dim=-1)
        return x + self.net(torch.nn.functional.silu(x) * (1.0 + scale) + shift)


class VelocityNet(nn.Module):
    """v_theta(x, t, c) with x the voxel vector, t in [0,1], c the condition."""

    def __init__(
        self,
        x_dim: int,
        cond_dim: int = 3,
        hidden: int = 256,
        n_blocks: int = 6,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        emb_dim = 2 * max(hidden // 3, 8)  # even embedding dim for sinusoids
        self.time_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(),
                                      nn.Linear(emb_dim, emb_dim))
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, emb_dim), nn.SiLU(),
                                      nn.Linear(emb_dim, emb_dim))
        self.input = nn.Linear(x_dim, hidden)
        self.blocks = nn.ModuleList(FiLMBlock(hidden, emb_dim) for _ in range(n_blocks))
        self.output = nn.Sequential(nn.SiLU(), nn.Linear(hidden, x_dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(_sinusoidal(t, self.time_mlp[0].in_features)) \
            + self.cond_mlp(cond)
        h = self.input(x)
        for block in self.blocks:
            h = block(h, emb)
        return self.output(h)

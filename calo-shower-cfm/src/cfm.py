"""Conditional Flow Matching core.

Rectified-flow / OT formulation with the linear probability path:

    x_t = (1 - t) * x0 + t * x1,          x0 ~ N(0, I),  x1 = data
    u_t = x1 - x0                          (target velocity)
    L    = E || v_theta(x_t, t, c) - u_t ||^2

Training pairs each data sample x1 with an independent Gaussian noise sample
x0 (Conditional Flow Matching with independent coupling). Sampling integrates
the learned ODE dx/dt = v_theta(x, t, c) from t=0 (noise) to t=1 (showers).
"""

from __future__ import annotations

import torch

__all__ = ["cfm_loss", "ode_sample"]


def sample_xt(x1: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linear interpolation x_t = (1 - t) x0 + t x1 with t of shape (n, 1)."""
    return (1.0 - t) * x0 + t * x1


def cfm_loss(model, x1: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """One-step Conditional Flow Matching loss (MSE to the target velocity)."""
    x0 = torch.randn_like(x1)                      # independent coupling
    t = torch.rand(x1.shape[0], 1, device=x1.device)
    xt = sample_xt(x1, x0, t)
    target = x1 - x0
    pred = model(xt, t.squeeze(-1), cond)
    return torch.mean((pred - target) ** 2)


@torch.no_grad()
def ode_sample(model, cond: torch.Tensor, n_steps: int = 20,
               generator: torch.Generator | None = None) -> torch.Tensor:
    """Integrate dx/dt = v_theta from t=0 to t=1 with explicit Euler."""
    x = torch.randn(cond.shape[0], model.x_dim, device=cond.device,
                    generator=generator)
    ts = torch.linspace(0.0, 1.0, n_steps + 1, device=cond.device)
    for i in range(n_steps):
        dt = ts[i + 1] - ts[i]
        t = torch.full((x.shape[0],), ts[i], device=cond.device)
        x = x + dt * model(x, t, cond)
    return x

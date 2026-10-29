"""Unit tests for the Calo-CFM pipeline (CPU only, fast)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

from src.cfm import cfm_loss, ode_sample, sample_xt  # noqa: E402
from src.data import DEFAULT_LAYERS, generate_showers  # noqa: E402
from src.model import VelocityNet  # noqa: E402

X_DIM = sum(l.n_cells for l in DEFAULT_LAYERS)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def test_data_shapes_and_ranges():
    x, cond = generate_showers(16, seed=0)
    assert x.shape == (16, X_DIM)
    assert cond.shape == (16, 3)
    # showers are strictly positive and normalized
    assert torch.all(x >= 0.0)
    assert torch.allclose(x.sum(dim=1), torch.ones(16), atol=1e-4)


def test_data_deterministic_given_seed():
    a, _ = generate_showers(4, seed=42)
    b, _ = generate_showers(4, seed=42)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------
# CFM path
# --------------------------------------------------------------------------

def test_linear_path_endpoints():
    x1 = torch.randn(5, X_DIM)
    x0 = torch.randn(5, X_DIM)
    t0 = torch.zeros(5, 1)
    t1 = torch.ones(5, 1)
    assert torch.allclose(sample_xt(x1, x0, t0), x0)
    assert torch.allclose(sample_xt(x1, x0, t1), x1)


def test_cfm_loss_finite_and_positive():
    torch.manual_seed(0)
    model = VelocityNet(x_dim=X_DIM, hidden=32, n_blocks=2)
    x, cond = generate_showers(8, seed=0)
    loss = cfm_loss(model, x, cond)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_cfm_loss_gradients_flow():
    torch.manual_seed(0)
    model = VelocityNet(x_dim=X_DIM, hidden=32, n_blocks=2)
    x, cond = generate_showers(8, seed=0)
    cfm_loss(model, x, cond).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


# --------------------------------------------------------------------------
# Sanity: an exact linear velocity field must score ~0 loss
# --------------------------------------------------------------------------

class ExactVelocityField(torch.nn.Module):
    """Memorizes the (x0, x1) coupling so pred == target exactly -> loss 0.

    Verifies that CFM's target u_t = x1 - x0 is the correct velocity for the
    linear path: with buffers x0_buf / x1_buf, if the loss's internal
    torch.randn_like draws exactly x0_buf (same seed & shape), prediction and
    target coincide for every t.
    """

    def __init__(self, x1: torch.Tensor, x0: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("x1_buf", x1.clone())
        self.register_buffer("x0_buf", x0.clone())

    def forward(self, x, t, cond):
        return self.x1_buf - self.x0_buf


def test_cfm_target_is_exact_velocity_for_linear_path():
    torch.manual_seed(0)
    x1, cond = generate_showers(16, seed=1)
    x0 = torch.randn_like(x1)
    model = ExactVelocityField(x1, x0)
    torch.manual_seed(0)  # reseed so loss's internal randn_like redraws x0
    loss = cfm_loss(model, x1, cond)
    assert loss.item() < 1e-6, f"CFM inconsistency: loss {loss.item():.3e}"


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def test_ode_sample_shapes_and_bounds():
    torch.manual_seed(0)
    model = VelocityNet(x_dim=X_DIM, hidden=32, n_blocks=2)
    model.eval()
    _, cond = generate_showers(4, seed=2)
    out = ode_sample(model, cond, n_steps=10)
    assert out.shape == cond.shape[:1] + (X_DIM,)
    assert torch.isfinite(out).all()


def test_vector_field_normalization_changes_norm():
    """A unit-ish velocity field should roughly preserve norms during Euler."""
    torch.manual_seed(0)
    model = VelocityNet(x_dim=X_DIM, hidden=32, n_blocks=2)
    model.eval()
    _, cond = generate_showers(2, seed=3)
    out = ode_sample(model, cond, n_steps=50)
    assert torch.isfinite(out).all() and out.abs().max() > 0


def test_training_reduces_loss():
    """A small model trained briefly must beat its step-1 loss."""
    torch.manual_seed(0)
    x, cond = generate_showers(32, seed=0)
    model = VelocityNet(x_dim=X_DIM, hidden=64, n_blocks=3)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    first = last = None
    for step in range(150):
        idx = torch.randint(0, x.shape[0], (16,))
        loss = cfm_loss(model, x[idx], cond[idx])
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = loss.item()
    assert last < first * 0.8, f"loss did not decrease: {first} -> {last}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))

"""Synthetic calorimeter shower data generator.

Simulates simplified 3-layer ATLAS-style electromagnetic calorimeter showers
(fine / middle / coarse layers). Each shower is produced by a longitudinal
profile that peaks at a depth that grows with (log E), and a transverse
profile with a Moliere-radius-like spread. Cell energies are flattened and
concatenated into a single voxel tensor.

The generator is differentiable-free and fast enough to make training / tests
run on any CPU.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["CaloShowerDataset", "LayerSpec", "DEFAULT_LAYERS"]


class LayerSpec:
    """A single calorimeter layer: grid of (h x w) cells."""

    def __init__(self, h: int, w: int, depth: float) -> None:
        self.h = h
        self.w = w
        self.depth = depth  # characteristic depth of the layer along z
        self.n_cells = h * w

    @property
    def shape(self) -> tuple[int, int]:
        return (self.h, self.w)


# Simplified 3-layer ATLAS ECal geometry (cell counts reduced for CPU demo).
DEFAULT_LAYERS: list[LayerSpec] = [
    LayerSpec(4, 4, depth=1.0),    # fine layer (strip-like)
    LayerSpec(8, 8, depth=4.0),    # middle layer (shower maximum)
    LayerSpec(8, 8, depth=14.0),   # coarse layer (tail)
]


def _gaussian_grid(spec: LayerSpec, center_xy: tuple[float, float],
                  width: float) -> np.ndarray:
    """Normalized 2D Gaussian kernel on the layer grid."""
    ys, xs = np.mgrid[0:spec.h, 0:spec.w].astype(np.float64)
    cy, cx = center_xy
    g = np.exp(-(((ys - cy) ** 2 + (xs - cx) ** 2) / (2.0 * width ** 2)))
    s = g.sum()
    return g / s if s > 0 else np.full((spec.h, spec.w), 1.0 / spec.n_cells)


def generate_showers(
    n: int,
    layers: list[LayerSpec] | None = None,
    seed: int | None = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate `n` synthetic showers.

    Returns
    -------
    x : (n, total_cells) float tensor in [0, 1), normalized cell energies
        (each shower is normalized by its total deposited energy).
    cond : (n, 3) float tensor = (log10(E), eta, phi) conditioning vector.
    """
    layers = layers or DEFAULT_LAYERS
    rng = np.random.default_rng(seed)
    total_cells = sum(l.n_cells for l in layers)

    # Incident energies: log-uniform between 1 and 1000 GeV.
    E = 10.0 ** rng.uniform(0.0, 3.0, size=n)
    eta = rng.uniform(-1.5, 1.5, size=n)
    phi = rng.uniform(-np.pi, np.pi, size=n)
    logE = np.log10(E)

    x = np.zeros((n, total_cells), dtype=np.float64)
    offset = 0
    for spec in layers:
        for i in range(n):
            # Longitudinal profile: gamma-like peak at depth ~ a*log(E)+b
            shower_max = 0.6 * logE[i] + 0.7
            k = max(shower_max / spec.depth, 1e-3)
            frac = rng.gamma(shape=max(k, 0.15)) * spec.depth
            layer_E_i = E[i] * (1.0 / (1.0 + np.exp(-(frac - shower_max))))
            layer_E_i *= (1.0 + 0.15 * rng.standard_normal())
            layer_E_i = max(layer_E_i, 0.0)

            # Transverse profile: Moliere radius does not depend on E, but
            # the spot wanders slightly with direction (eta, phi).
            width = 0.9 + 0.25 * abs(np.sin(eta[i]))
            cy = spec.h / 2.0 + 0.6 * np.sin(phi[i]) * (abs(np.log(abs(eta[i]) + 1e-3)) % 1.0)
            cx = spec.w / 2.0 + 0.6 * np.cos(phi[i]) * (abs(np.log(abs(eta[i]) + 1e-3)) % 1.0)
            kernel = _gaussian_grid(spec, (cy, cx), width)
            cells = layer_E_i * kernel
            cells += cells.max() * 0.02 * rng.random(cells.shape)  # noise floor
            x[i, offset:offset + spec.n_cells] = cells.ravel()
        offset += spec.n_cells

    # Normalize each shower to total deposited energy = 1 -> cells in [0, 1).
    x /= np.clip(x.sum(axis=1, keepdims=True), 1e-8, None)

    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    cond = torch.as_tensor(
        np.stack([logE, eta, phi], axis=1), dtype=torch.float32, device=device
    )
    return x_t, cond

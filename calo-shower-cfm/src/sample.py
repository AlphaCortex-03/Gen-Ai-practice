"""Sample showers from a trained CFM model and plot real vs generated.

Usage (from project root):
    python -m src.sample --out outputs
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .cfm import ode_sample
from .data import DEFAULT_LAYERS, generate_showers
from .model import VelocityNet


def unflatten(x: torch.Tensor):
    """(n, total_cells) -> list of per-layer 2D maps per shower."""
    layers, offset = [], 0
    for spec in DEFAULT_LAYERS:
        layers.append(x[:, offset:offset + spec.n_cells].view(-1, spec.h, spec.w))
        offset += spec.n_cells
    return layers


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sample with Calo-CFM")
    parser.add_argument("--ckpt", type=str, default="outputs/calo_cfm.pt")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--out", type=str, default="outputs")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model = VelocityNet(x_dim=ckpt["x_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    gen = torch.Generator(device=device).manual_seed(123)
    _, cond = generate_showers(8, seed=7, device=device)  # reuse real conditions

    fake = ode_sample(model, cond[: args.n], n_steps=args.steps,
                      generator=gen).clamp_min(0.0)
    fake /= fake.flatten(1).sum(1, keepdim=True).clamp_min(1e-8)

    real, _ = generate_showers(args.n, seed=99, device=device)

    rf = unflatten(fake)
    rr = unflatten(real)
    fig, axes = plt.subplots(
        args.n, 2 * len(DEFAULT_LAYERS) + 1,
        figsize=(3 * (2 * len(DEFAULT_LAYERS) + 1), 2.2 * args.n),
    )
    for i in range(args.n):
        col = 0
        for l, (spec, f_img, r_img) in enumerate(zip(DEFAULT_LAYERS, rf, rr)):
            for img, tag in ((f_img, "gen"), (r_img, "real")):
                ax = axes[i, col] if args.n > 1 else axes[col]
                ax.imshow(img[i].numpy(), cmap="inferno")
                ax.set_title(f"L{l + 1} {tag}", fontsize=8)
                ax.axis("off")
                col += 1
        ax = axes[i, col] if args.n > 1 else axes[col]
        ax.axis("off")
        ax.text(0.1, 0.5, f"cond\nlogE={cond[i, 0]:.2f}\neta={cond[i, 1]:.2f}",
                fontsize=8)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "samples.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Generated {args.n} showers | saved -> {path}")


if __name__ == "__main__":
    main()

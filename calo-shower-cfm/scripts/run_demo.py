"""End-to-end demo: generate data -> train briefly -> sample -> plot.

Run:  python scripts/run_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.cfm import cfm_loss, ode_sample
from src.data import DEFAULT_LAYERS, generate_showers
from src.model import VelocityNet


def main() -> None:
    torch.manual_seed(0)
    device = "cpu"

    print("== Calo-CFM end-to-end demo ==")
    x, cond = generate_showers(4000, seed=0, device=device)
    print(f"data: showers x.shape={tuple(x.shape)}, cond.shape={tuple(cond.shape)}")

    model = VelocityNet(x_dim=x.shape[1], hidden=128, n_blocks=4)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    losses = []
    for step in range(1, 501):
        idx = torch.randint(0, x.shape[0], (128,))
        loss = cfm_loss(model, x[idx], cond[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 100 == 0:
            print(f"  step {step:4d}  loss {loss.item():.4f}")

    print(f"loss first={losses[0]:.4f}  last={losses[-1]:.4f}  "
          f"({(1 - losses[-1] / losses[0]):.0%} reduction)")

    gen = torch.Generator().manual_seed(7)
    fake = ode_sample(model, cond[:2], n_steps=25, generator=gen)
    print(f"sampled showers: shape={tuple(fake.shape)}, "
          f"finite={torch.isfinite(fake).all().item()}")
    print("Downstream (real experiments): plug in CaloChallenge / "
          "CaloGAN h5 datasets in src/data.py.")


if __name__ == "__main__":
    main()

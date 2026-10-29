"""Train the CFM velocity field on synthetic calorimeter showers.

Usage (from project root):
    python -m src.train --steps 3000 --layers 3
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from .cfm import cfm_loss
from .data import DEFAULT_LAYERS, generate_showers
from .model import VelocityNet


def get_dataset(n: int, seed: int, device: str = "cpu"):
    return generate_showers(n, layers=DEFAULT_LAYERS, seed=seed, device=device)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Calo-CFM")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dataset-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="outputs")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Generating {args.dataset_size} synthetic showers on {device}...")
    x, cond = get_dataset(args.dataset_size, seed=args.seed, device=device)
    total_cells = x.shape[1]

    model = VelocityNet(x_dim=total_cells, cond_dim=cond.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    history = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, x.shape[0], (args.batch_size,), device=device)
        loss = cfm_loss(model, x[idx], cond[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        history.append(loss.item())

        if step % max(1, args.steps // 10) == 0 or step == 1:
            print(f"step {step:5d} | loss {loss.item():.5f} | "
                  f"{(time.time() - t0):.1f}s")

        if step == args.steps // 2 and loss.item() > 1.0:
            print("warning: loss abnormally high, consider lowering lr")

    ckpt_path = os.path.join(args.out, "calo_cfm.pt")
    torch.save({"model": model.state_dict(), "x_dim": total_cells}, ckpt_path)
    hist_path = os.path.join(args.out, "loss_history.txt")
    with open(hist_path, "w") as f:
        f.write("\n".join(f"{v:.6f}" for v in history))

    print(f"Saved checkpoint -> {ckpt_path}")
    print(f"Final loss: {history[-1]:.5f} | best: {min(history):.5f}")


if __name__ == "__main__":
    main()

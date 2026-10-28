import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def sample_circle(n, noise=0.05, radius_range=(0.8, 1.2), device="cpu"):
    """noisy ring of radius 1 (that's our "data")"""
    theta = torch.rand(n, device=device) * 2 * math.pi
    r = torch.empty(n, device=device).uniform_(*radius_range)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    pts = torch.stack([x, y], dim=1)
    return pts + noise * torch.randn_like(pts)


def sample_noise(n, device="cpu"):
    """standard 2d gaussian (that's our "noise")"""
    return torch.randn(n, 2, device=device)


class VectorField(nn.Module):
    """small mlp, (x, t) -> velocity vector"""

    def __init__(self, hidden=128, layers=3):
        super().__init__()
        net, d_in = [], 3  # x(2) + t(1)
        for _ in range(layers):
            net += [nn.Linear(d_in, hidden), nn.SiLU()]
            d_in = hidden
        net += [nn.Linear(d_in, 2)]
        self.net = nn.Sequential(*net)

    def forward(self, x, t):
        # t may come in as a scalar, (N,) or (N,1)
        if t.dim() == 0:
            t = t.view(1, 1).expand(x.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        return self.net(torch.cat([x, t], dim=1))


def path_sample(x0, x1, t):
    """straight-line path x_t and its true velocity u = x1 - x0"""
    if t.dim() == 1:
        t = t.unsqueeze(1)
    x_t = (1 - t) * x0 + t * x1
    u = x1 - x0
    return x_t, u


@torch.no_grad()
def euler_sample(model, n, steps=200, device="cpu"):
    """solve dx/dt = v(x, t) from t=0 (noise) to t=1 (data)"""
    x = sample_noise(n, device)
    dt = 1.0 / steps
    traj = [x.clone()]
    for i in range(steps):
        t = torch.full((x.shape[0],), i * dt, device=device)
        x = x + model(x, t) * dt
        if (i + 1) % (steps // 10) == 0:
            traj.append(x.clone())
    return x, torch.stack(traj)  # traj: (11, n, 2)


def train(args, device):
    model = VectorField(args.hidden, args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses = []

    for step in range(1, args.steps + 1):
        x0 = sample_noise(args.batch, device)                     # noise
        x1 = sample_circle(args.batch, device=device)             # circle data
        t = torch.rand(args.batch, 1, device=device)
        x_t, u = path_sample(x0, x1, t)

        loss = (model(x_t, t) - u).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if step % args.log_every == 0 or step == 1:
            print(f"step {step:5d}/{args.steps}  loss = {loss.item():.5f}")
        if args.ckpt_every and step % args.ckpt_every == 0:
            torch.save(model.state_dict(), os.path.join(args.out, "flow_model.pt"))

    print(f"\nfinal loss: {sum(losses[-100:]) / min(100, len(losses)):.5f}")
    return model, losses


def plot_results(args, model, losses, device):
    os.makedirs(args.out, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # loss curve
    ax = axes[0]
    ax.plot(losses, color="tab:blue", lw=0.8)
    ax.set_title("Flow-matching training loss")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.3)

    # data vs generated samples
    ax = axes[1]
    data = sample_circle(args.n_samples, noise=0.0, radius_range=(1, 1), device=device)
    with torch.no_grad():
        x_gen, traj = euler_sample(
            model, args.n_samples, steps=args.euler_steps, device=device
        )
    ax.scatter(data.cpu()[:, 0], data.cpu()[:, 1], s=8,
               alpha=0.4, label="true circle", color="tab:orange")
    ax.scatter(x_gen.cpu()[:, 0].numpy(), x_gen.cpu()[:, 1].numpy(), s=8,
               alpha=0.6, label="flow samples", color="tab:blue")
    ax.set_title("Noise  ->  Circle")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect("equal")

    # ode trajectories from a few noise points
    ax = axes[2]
    d = traj.cpu().numpy()
    k = min(args.n_trajectories, d.shape[1])
    for i in range(k):
        ax.plot(d[:, i, 0], d[:, i, 1], color="steelblue", alpha=0.5, lw=0.9)
        ax.scatter(d[0, i, 0], d[0, i, 1], color="red", s=14, zorder=3)     # start (noise)
        ax.scatter(d[-1, i, 0], d[-1, i, 1], color="green", s=14, zorder=3)  # end (data)
    ax.scatter([], [], color="red", label="start = noise")
    ax.scatter([], [], color="green", label="end = circle")
    ax.set_title(f"{args.n_trajectories} learned ODE trajectories")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal")

    fig.suptitle("Toy Flow Matching: learning the vector field from noise to a circle",
                 fontsize=12)
    fig.tight_layout()
    out_png = os.path.join(args.out, "flow_matching_result.png")
    fig.savefig(out_png, dpi=140)
    print(f"saved plot -> {out_png}")

    # arrows = learned field at the midpoint t=0.5
    grid = torch.stack(torch.meshgrid(
        torch.linspace(-2, 2, 21, device=device),
        torch.linspace(-2, 2, 21, device=device), indexing="xy"), dim=-1
    ).reshape(-1, 2)
    with torch.no_grad():
        t_mid = torch.full((grid.shape[0],), 0.5, device=device)
        v = model(grid, t_mid).cpu().numpy()
    g = grid.cpu().numpy()
    fig2, ax = plt.subplots(figsize=(5.5, 5))
    ax.quiver(g[:, 0], g[:, 1], v[:, 0], v[:, 1], color="tab:purple")
    theta = torch.linspace(0, 2 * torch.pi, 200)
    ax.plot(torch.cos(theta), torch.sin(theta), color="orange", lw=2, label="data")
    ax.set_title("Learned vector field v(x, t=0.5)")
    ax.legend(); ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal")
    quiver_png = os.path.join(args.out, "vector_field.png")
    fig2.savefig(quiver_png, dpi=140)
    print(f"saved plot -> {quiver_png}")


def parse_args():
    p = argparse.ArgumentParser(description="Toy Flow Matching (noise -> circle)")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--euler-steps", type=int, default=200)
    p.add_argument("--n-trajectories", type=int, default=40)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--out", type=str, default="out")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    os.makedirs(args.out, exist_ok=True)

    model, losses = train(args, device)
    torch.save(model.state_dict(), os.path.join(args.out, "flow_model.pt"))
    print(f"saved model -> {os.path.join(args.out, 'flow_model.pt')}")
    plot_results(args, model, losses, device)


if __name__ == "__main__":
    main()

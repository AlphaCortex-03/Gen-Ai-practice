# pytorch-vae-mnist
# A clean Variational Autoencoder on MNIST — the "hello world" of generative models.
#
#   * Encoder: 784 -> 256 -> (mu, logvar)   [latent_dim = 2 by default]
#   * Reparameterization trick:  z = mu + exp(0.5*logvar) * eps
#   * Decoder:  latent_dim -> 256 -> 784 (sigmoid)
#   * Loss: binary cross-entropy reconstruction + KL divergence to N(0, I)
#
# Trains in ~1 minute on CPU. Saves two showcase images:
#   out/vae_reconstructions.png   (originals vs. reconstructions)
#   out/vae_latent_space.png      (2-D latent scatter coloured by digit, latent_dim=2)
#
# Run:  python vae.py
#       python vae.py --epochs 10 --latent-dim 8

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# --------------------------------- model ----------------------------------- #
class VAE(nn.Module):
    def __init__(self, latent_dim=2, hidden=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 784), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar


def vae_loss(recon, x, mu, logvar):
    recon = recon.view(recon.size(0), -1)
    bce = F.binary_cross_entropy(recon, x, reduction="sum") / x.size(0)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return bce, kld, bce + kld


# --------------------------------- training -------------------------------- #
def train(args, device):
    tfm = transforms.ToTensor()
    train_set = datasets.MNIST("data", train=True, download=True, transform=tfm)
    test_set = datasets.MNIST("data", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=0)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

    model = VAE(args.latent_dim, args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, _ in train_loader:
            x = x.view(x.size(0), -1).to(device)
            recon, mu, logvar = model(x)
            bce, kld, loss = vae_loss(recon, x, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # quick validation
        model.eval()
        vl, n = 0.0, 0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.view(x.size(0), -1).to(device)
                recon, mu, logvar = model(x)
                bce, kld, loss = vae_loss(recon, x, mu, logvar)
                vl += loss.item() * x.size(0)
                n += x.size(0)
        print(f"epoch {epoch}/{args.epochs}  bce={bce:.2f}  kld={kld:.2f}  "
              f"val_elbo=-{vl / n:.2f}")

    torch.save(model.state_dict(), os.path.join(args.out, "vae_mnist.pt"))
    print(f"saved model -> {os.path.join(args.out, 'vae_mnist.pt')}")
    return model, test_set


# --------------------------------- plotting -------------------------------- #
@torch.no_grad()
def plot(model, test_set, device, out, latent_dim):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()

    # original vs reconstruction grid
    n_show = 10
    idx = torch.randperm(len(test_set))[:n_show]
    xs, ys = [], []
    for i in idx:
        img, label = test_set[int(i)]
        xs.append(img); ys.append(int(label))
    x = torch.stack(xs).view(n_show, -1).to(device)
    recon, _, _ = model(x)

    fig, ax = plt.subplots(2, n_show, figsize=(n_show * 1.3, 3.0))
    for i in range(n_show):
        ax[0][i].imshow(x[i].view(28, 28).cpu(), cmap="gray")
        ax[1][i].imshow(recon[i].view(28, 28).cpu(), cmap="gray")
        ax[0][i].set_title(f"label {ys[i]}", fontsize=8)
        ax[0][i].axis("off"); ax[1][i].axis("off")
    ax[0][0].text(-4.6, 14, "input", rotation=90, va="center")
    ax[1][0].text(-4.6, 14, "VAE recon", rotation=90, va="center")
    fig.suptitle("Variational Autoencoder on MNIST: originals (top) vs reconstructions (bottom)")
    fig.tight_layout()
    p1 = os.path.join(out, "vae_reconstructions.png")
    fig.savefig(p1, dpi=140)
    print(f"saved plot -> {p1}")

    # latent space scatter (only meaningful when latent_dim == 2)
    if latent_dim != 2:
        print("latent_dim != 2 -> skipping 2-D latent scatter plot")
        return
    zs, labels = [], []
    loader = DataLoader(test_set, batch_size=512, shuffle=False)
    for x, y in loader:
        x = x.view(x.size(0), -1).to(device)
        z = model.encode(x)[0]
        zs.append(z.cpu()); labels.append(y)
    z = torch.cat(zs).numpy()
    y = torch.cat(labels).numpy()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10", s=4, alpha=0.65)
    fig.colorbar(sc, ticks=range(10))
    ax.set_title("MNIST latent space (z ~ N(0, I) learned by the VAE)")
    ax.set_xlabel("z[0]"); ax.set_ylabel("z[1]")
    fig.tight_layout()
    p2 = os.path.join(out, "vae_latent_space.png")
    fig.savefig(p2, dpi=140)
    print(f"saved plot -> {p2}")


# ----------------------------------- main ---------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="PyTorch VAE on MNIST")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--out", type=str, default="out")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}   latent_dim = {args.latent_dim}")
    os.makedirs(args.out, exist_ok=True)

    model, test_set = train(args, device)
    plot(model, test_set, device, args.out, args.latent_dim)


if __name__ == "__main__":
    main()

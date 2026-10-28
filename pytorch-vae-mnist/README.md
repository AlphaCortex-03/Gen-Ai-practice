# pytorch-vae-mnist

Clean, minimal **Variational Autoencoder** on MNIST — the "hello world" of generative models. ~150 lines of readable PyTorch that trains in about a minute on CPU and produces two showcase images.

## What it does

- **Encoder:** `784 → 256 → (μ, logσ²)` with a 2-D latent space by default
- **Reparameterization trick:** `z = μ + exp(0.5·logσ²) · ε`, so sampling stays differentiable
- **Optimizer-free math on show:** the loss is exactly the ELBO,
  `L = BCE(x, x̂) + KL(q(z|x) ‖ N(0, I))`
- **Decoder:** `z → 256 → 784 (sigmoid)` mapping samples back to images

## Files

| File | Purpose |
|---|---|
| `vae.py` | Full model + training + evaluation + plots |
| `out/` | `vae_mnist.pt`, `vae_reconstructions.png`, `vae_latent_space.png` after running |

## Usage

```bash
pip install torch torchvision matplotlib

python vae.py                                  # 5 epochs, latent_dim = 2, ~1 min on CPU
python vae.py --epochs 10 --latent-dim 8       # bigger / longer
```

## Outputs

- `vae_reconstructions.png` — 10 test digits, originals on top row, VAE reconstructions below (blurry = the characteristic VAE blur you want to show interviewers!)
- `vae_latent_space.png` — the famous 2-D latent scatter: all 10 digit classes emerge automatically from the KL regularizer, with no labels used in training.
- `vae_mnist.pt` — trained weights, loadable with `VAE.load_state_dict`.

## Why this project is "simplest but impressive"

It hits all the concepts that come up in ML interviews and ML4SCI-style selections: probabilistic modeling, the reparameterization trick, ELBO/KL divergence, amortized inference, and honest evaluation (validation ELBO per epoch). And because it is a single dependency-light file, anyone can clone and run it immediately.

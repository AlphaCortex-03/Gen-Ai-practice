# Gen-Ai-practice-coder-Army-

A collection of hands-on Gen-AI / generative-modeling projects — every one runs standalone on a laptop CPU.

## Projects

| Project | Main file | What it is |
|---|---|---|
| **[flow-matching-toy](./flow-matching-toy/)** | `train.py` | Toy Flow Matching implementation — learning vector field from noise to circle distribution (CERN GENIE related). The core math behind diffusion/flow models like SD3. |
| **[llm-hallucination-judge](./llm-hallucination-judge/)** | `eval.py` | Lightweight LLM evaluation framework — hallucination detection **without API calls**. Pure-Python claim-level faithfulness judge. |
| **[pytorch-vae-mnist](./pytorch-vae-mnist/)** | `vae.py` | Minimal Variational Autoencoder on MNIST — reparameterization trick, ELBO loss, latent-space visualization. |

## How to run

Each project folder has its own README with instructions:

```bash
# Project 1 — needs torch + matplotlib
cd flow-matching-toy && python train.py

# Project 2 — needs NOTHING (pure stdlib)
cd llm-hallucination-judge && python eval.py

# Project 3 — needs torch + torchvision + matplotlib
cd pytorch-vae-mnist && python vae.py
```

## Topics covered

Flow matching / conditional flow matching • ODE-based generative sampling • hallucination evaluation & faithfulness judging • claim extraction • VAEs • reparameterization trick • ELBO / KL divergence • latent spaces

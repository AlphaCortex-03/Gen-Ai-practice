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

## Tested & verified ✔

All three programs were run end-to-end (no API calls for project 2, CPU-only for 1 and 3) — once in an integrated PowerShell terminal and once inside VS Code using the pre-configured tasks in [`.vscode/tasks.json`](./.vscode/tasks.json). Six runs, six `exit code 0`. Full gallery with per-run descriptions: **[screenshots/](./screenshots/README.md)**

| Project | VS Code terminal run | Result |
|---|---|---|
| flow-matching-toy | ![flow matching training in VS Code](./screenshots/project1_flow_matching_vscode.png) | 4000 steps, `final loss: 1.02470` |
| llm-hallucination-judge | ![hallucination judge in VS Code](./screenshots/project2_hallucination_judge_vscode.png) | 3/5 hallucinations flagged, 0 API calls |
| pytorch-vae-mnist | ![VAE training in VS Code](./screenshots/project3_vae_vscode.png) | ELBO improved `-165.51 → -157.44`, plots saved |

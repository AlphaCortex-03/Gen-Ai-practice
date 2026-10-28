# flow-matching-toy

Toy Flow Matching implementation — learning vector field from noise to circle distribution (CERN GENIE related).

Flow matching is the generative-modeling idea behind modern models like Stable Diffusion 3 — it learns a **velocity field** `v(x, t)` whose ODE carries a simple distribution (Gaussian noise) into a complex one. This repo teaches the concept on the simplest possible 2-D toy problem: **noise → circle**, the exact style of warm-up task used in CERN's *GENIE* / ML4SCI selection exercises.

## What it does

- **Training (Conditional Flow Matching):** draw `x0 ~ N(0, I)` and `x1 ~ circle`, build the linear path `x_t = (1-t)·x0 + t·x1`, and regress the truth velocity `u = x1 − x0`:
  `L = E‖v_θ(x_t, t) − (x1 − x0)‖²`
- **Sampling:** integrate `dx/dt = v_θ(x, t)` from `t = 0` to `t = 1` with Euler steps; points starting as noise converge onto the circle.
- **Visualization:** loss curve, generated samples vs. true ring, learned ODE trajectories, and a quiver plot of the vector field at `t = 0.5`.

## Files

| File | Purpose |
|---|---|
| `train.py` | Full training + sampling + plotting (only needs `torch`, `matplotlib`) |
| `out/` | `flow_model.pt`, `flow_matching_result.png`, `vector_field.png` after running |

## Usage

```bash
pip install torch matplotlib
python train.py                          # defaults: 4000 steps, ~30 s on CPU
python train.py --steps 8000 --hidden 256 --n-trajectories 60
```

## Why this matters for ML4SCI / CERN GENIE

CERN's GENIE task and the ML4SCI deep-learning selections routinely start with *"learn a mapping from a simple noise distribution to a target distribution and visualize the generative trajectory."* Flow matching is the cleanest modern answer to that brief — the same math (ODE integration, vector-field regression, conditional paths) scales to calorimeter showers and particle physics data. This toy shows every ingredient in ~150 lines.

## Results (default settings)

`final loss ≈ 0.02` — generated samples land on the ring and trajectories show the classic "flow field bending noise onto the circle" picture.

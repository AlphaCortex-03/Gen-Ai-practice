# Calo-CFM — Conditional Flow Matching for Calorimeter Showers

Simulation of calorimeter shower images with **Conditional Flow Matching**
(OT / rectified-flow formulation). This is the same modeling objective made
popular by CaloFlow-style generation: instead of an adversarial loss or a
normalizing flow with expensive Jacobians, CFM regresses a velocity field
`v_theta(x, t, c)` toward the probability-flow ODE

```
x_t = (1 - t) x0 + t x1            # linear interpolation path (x0: noise, x1: shower)
u_t = x1 - x0                      # target velocity (independent coupling)
L   = E || v_theta(x_t, t, c) - (x1 - x0) ||^2
```

Sampling integrates `dx/dt = v_theta(x, t, c)` from `t=0` (Gaussian noise) to
`t=1` using a few Euler steps.

## Conditioning

The model conditions `v_theta` on physically meaningful shower variables:

- `log10(E)` — incident energy (controls shower depth)
- `eta`, `phi` — direction of incidence

## Project layout

```
calo-shower-cfm/
├── src/
│   ├── data.py      # synthetic 3-layer calorimeter shower generator
│   ├── model.py     # ResMLP velocity net w/ sinusoidal t + FiLM conditioning
│   ├── cfm.py       # CFM loss + ODE sampler (Euler)
│   ├── train.py     # training CLI
│   └── sample.py    # sampling + real-vs-generated plots
├── scripts/run_demo.py
├── tests/test_cfm.py
├── requirements.txt
└── outputs/          # checkpoints, loss history, sample plots
```

## Quickstart

```bash
pip install -r requirements.txt

# run unit tests
pytest tests -v

# end-to-end demo (train 500 steps on CPU)
python scripts/run_demo.py

# full training
python -m src.train --steps 3000

# sample and plot generated vs real showers
python -m src.sample --n 4
```

Outputs land in `outputs/`: `calo_cfm.pt` (checkpoint), `loss_history.txt`,
and `samples.png` (per-layer generated vs. real shower images).

## Swapping in a real dataset

`src/data.py` is deliberately isolated. To train on real data (e.g. the
CaloChallenge dataset 1: 3 layers of 3x96 / 12x12 / 22x22 cells), replace
`generate_showers` with an HDF5 loader returning `(x_flattened, cond)` and
everything else (model, loss, sampler) works unchanged.

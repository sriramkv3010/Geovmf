# AGG + Neuropalsy: Adaptive Geometric Gaze Estimation under Pathological Eye Conditions

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Research Code** — Group No. 47  
> Appearance-based gaze estimation with uncertainty quantification under neuropathological eye movement disorders using the **Adaptive Geometric Gaze (AGG) framework** combined with **von Mises–Fisher probabilistic heads**.

---

## Overview

This repository implements the **Dual AGG Framework** for robust gaze estimation in the presence of neuropathological gaze disturbances (nystagmus, strabismus, restricted movement, ocular palsy). The pipeline combines:

- **Isomap-based Geometric Propagation Module (GPM)** — maps CNN feature manifolds to a geodesically-aligned gaze sphere via multi-restart Sphere Alignment (SA).
- **Isometric Propagator (IP)** — an invertible neural network that bridges feature space and the projected gaze frame (PGF), enabling geometry-aware training signals.
- **Sphere-Oriented Training (SOT)** — fine-tunes the CNN backbone by backpropagating through the IP inverse path.
- **von Mises–Fisher (vMF) Uncertainty Head** — models directional uncertainty on *S²*, producing calibrated 95%-cone estimates.
- **Neuropalsy Noise Injector** — physically realistic simulation of four conditions for domain-adaptive training.

---

## Architecture

```
Image (224×224)
    │
    ▼
ResNet-18 Backbone  ──────────────────────────────────────────────────────┐
    │ 512-d feature                                                        │
    ├──────────────────────────────────────────────┐                      │
    │                                              ▼                      ▼
    │                               Isometric Propagator (IP)        vMF Head
    │                               (feature → PGF ∈ ℝ³)         (μ ∈ S², κ ∈ ℝ)
    ▼                                              │
Isomap GPM                                        │ Sphere Alignment (SA)
(ℝ^512 → ℝ^3 → S²)   ◄────────────────────────────┘
    │
    ▼
Predicted Gaze Vector (unit vector on S²)
```

---

## Repository Structure

```
agg-neuropalsy-gaze/
├── README.md
├── requirements.txt
├── setup.py
├── config.py                        # All hyperparameters & dataset paths
│
├── data/
│   └── dataset.py                   # MPIIFaceGaze dataset loader
│
├── models/
│   ├── __init__.py
│   ├── backbone.py                  # ResNet-18 feature extractor
│   ├── isometric_propagator.py      # IP: feature → PGF
│   └── vmf_head.py                  # vMF directional uncertainty head
│
├── gpm/
│   ├── __init__.py
│   └── robust_gpm.py                # Isomap + multi-restart Sphere Alignment
│
├── neuropalsy/
│   ├── __init__.py
│   ├── injector.py                  # Physio-realistic noise injector
│   └── dataset.py                   # Augmented neuropalsy dataset wrapper
│
├── training/
│   ├── __init__.py
│   └── framework.py                 # DualAGGFramework: full pipeline
│
├── utils/
│   ├── __init__.py
│   ├── geometry.py                  # Angular math on S²
│   ├── metrics.py                   # MetricLogger + publishable tables
│   └── device.py                    # Device detection & GPU memory utils
│
├── scripts/
│   └── install_torch.sh             # CUDA-compatible torch install
│
├── results/                         # Output JSONs / CSVs (gitignored)
├── notebooks/
│   └── analysis.ipynb               # Results analysis & visualisation
└── main.py                          # Entry point
```

---

## Dataset

**MPIIFaceGaze** — download via KaggleHub:
```python
import kagglehub
path = kagglehub.dataset_download("greninja2006/gazedataset")
```
Set `DATASET_PATH` in `config.py` to `<path>/Data`.

---

## Installation

```bash
# 1. Clone
git clone https://github.com/your-org/agg-neuropalsy-gaze.git
cd agg-neuropalsy-gaze

# 2. Install CUDA-compatible PyTorch (Python 3.12 + CUDA 11.8)
bash scripts/install_torch.sh

# 3. Install remaining dependencies
pip install -r requirements.txt
```

---

## Usage

### Full Pipeline (all phases)
```bash
python main.py
```

### Run a Single Condition
```python
from training.framework import DualAGGFramework
from data.dataset import MPIIFaceGazeDataset
from config import DATASET_PATH, BATCH_SIZE, NUM_WORKERS
from torch.utils.data import DataLoader

train_ds = MPIIFaceGazeDataset(DATASET_PATH, split='train')
val_ds   = MPIIFaceGazeDataset(DATASET_PATH, split='test')

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

agg = DualAGGFramework(condition='nystagmus', severity=0.6)
results = agg.run_full_pipeline(train_loader, val_loader, train_ds, val_ds)
```

---

## Key Fixes (v2 → Final)

| Fix | Problem | Solution |
|-----|---------|----------|
| **FIX-1** | SA error 13.70° (k2=0.295 underfitted) | Multi-restart SA (5×), init k1,k2 ~ U[0.5, 2.0], patience 30→50 |
| **FIX-2** | SOT L1 plateau at 0.35 | IP trained 200ep + 2-phase cosine LR; SOT uses cosine-annealed LR |
| **FIX-3** | Pathological GPM instability | Patho GPM built *before* joint FT on stable stage-i features |
| **FIX-4** | κ explosion | κ clamped to [0.5, 50] via Softplus + clamp |

---

## Publishable Metrics

| Method | Mean (°) | Std (°) | Median (°) |
|--------|----------|---------|-----------|
| FC baseline (healthy) | — | — | — |
| GPM-AGG (healthy) | — | — | — |
| GPM-AGG (pathological) | — | — | — |
| vMF μ vs clean GT | — | — | — |
| vMF μ vs noisy GT | — | — | — |

*Fill with experiment output from `gaze_results_<condition>.json`.*

---

## Citation

```bibtex
@misc{agg_neuropalsy_2024,
  title  = {Adaptive Geometric Gaze Estimation under Neuropathological Conditions},
  author = {Group No. 47},
  year   = {2024},
  note   = {github.com/your-org/agg-neuropalsy-gaze}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).

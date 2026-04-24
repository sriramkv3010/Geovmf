"""
config.py
=========
Centralised hyperparameters, dataset paths, and training schedules for
the AGG + Neuropalsy gaze estimation framework.

Edit DATASET_PATH and toggle USE_CUDA / FORCE_CPU to match your environment.
All other values propagate automatically throughout the codebase.
"""

import torch

# ─── Device auto-detection ────────────────────────────────────────────────────
FORCE_CPU = False   # set True to disable GPU even if available
USE_CUDA: bool = (not FORCE_CPU) and torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET_PATH = "/kaggle/input/datasets/greninja2006/gazedataset/Data"

TRAIN_RATIO  = 0.8   # fraction of participants for training split
RANDOM_SEED  = 42    # controls participant split & data sub-sampling

# ─── DataLoader ───────────────────────────────────────────────────────────────
BATCH_SIZE   = 128 if USE_CUDA else 64
NUM_WORKERS  = 4   if USE_CUDA else 2
PIN_MEMORY   = USE_CUDA
PREFETCH     = 2      # prefetch_factor (only when num_workers > 0)

# ─── Feature / Isomap / GPM ───────────────────────────────────────────────────
N_PRETRAIN_SAMPLES = 9000 if USE_CUDA else 5000   # samples per pretrain epoch
N_SAMPLES          = 1800 if USE_CUDA else 2000   # samples for Isomap / IP
N_NEIGHBORS        = 120  if USE_CUDA else 150    # Isomap k-NN graph

# ─── Training epochs ──────────────────────────────────────────────────────────
PRETRAIN_EP  = 10 if USE_CUDA else 5     # Phase 1: pretrain (L1 regression)
IP_EP        = 200                       # FIX-2: Isometric Propagator epochs
SOT_EP       = 10 if USE_CUDA else 5     # Sphere-Oriented Training epochs
SA_EPOCHS    = 400 if USE_CUDA else 350  # FIX-1: SA epochs per restart
SA_RESTARTS  = 5                         # FIX-1: number of SA random restarts
VMF_EPOCHS   = 10 if USE_CUDA else 5     # Phase 2 stage-i: vMF head (CNN frozen)
JOINT_EPOCHS = 6  if USE_CUDA else 3     # Phase 2 stage-ii: joint CNN + vMF

# ─── Optimiser ────────────────────────────────────────────────────────────────
LR            = 1e-4    # global base learning rate
WEIGHT_DECAY  = 1e-4

# ─── vMF / kappa ──────────────────────────────────────────────────────────────
KAPPA_MIN    = 0.5    # FIX-4: lower clamp to prevent κ collapse
KAPPA_MAX    = 50.0   # FIX-4: upper clamp to prevent κ explosion

# ─── Overfitting guard ────────────────────────────────────────────────────────
OVERFIT_GAP  = 0.5    # warn if val_nll − train_nll > this value

# ─── Neuropalsy simulation ────────────────────────────────────────────────────
# Valid conditions: 'nystagmus', 'strabismus', 'restricted', 'palsy'
DEFAULT_CONDITION = "nystagmus"
DEFAULT_SEVERITY  = 0.6        # in [0.0, 1.0]

# ─── Output paths ─────────────────────────────────────────────────────────────
CHECKPOINT_DIR   = "."         # where .pth files are written
RESULTS_JSON_TPL = "gaze_results_{condition}.json"

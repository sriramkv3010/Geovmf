"""
main.py
=======
Entry point for the AGG + Neuropalsy gaze estimation pipeline.

Usage
-----
    python main.py

Conditions to benchmark are defined in ``CONDITIONS`` below.  Add more
tuples to ``CONDITIONS`` once the first run confirms the pipeline is
working end-to-end.  Results for each condition are written to
``gaze_results_<condition>.json`` and a combined publishable table is
printed at the end.
"""

import os
import numpy as np
from torch.utils.data import DataLoader

import config
from config import (
    DATASET_PATH, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, USE_CUDA,
    TRAIN_RATIO, SA_RESTARTS, SA_EPOCHS, IP_EP, KAPPA_MIN, KAPPA_MAX,
)
from data.dataset       import MPIIFaceGazeDataset
from training.framework import DualAGGFramework
from utils.geometry     import cone_95


# ─── Conditions to evaluate ───────────────────────────────────────────────────
# Each entry is (condition_name, severity).
# Start with a single condition; extend after confirming the pipeline runs.
CONDITIONS = [
    ("nystagmus",   0.6),
    # ("strabismus",  0.5),
    # ("restricted",  0.5),
    # ("palsy",       0.5),
]


def make_loader(ds, shuffle: bool) -> DataLoader:
    """Build a DataLoader with settings from config."""
    nw = NUM_WORKERS if os.name != "nt" else 0
    kw = dict(
        batch_size  = BATCH_SIZE,
        shuffle     = shuffle,
        num_workers = nw,
        pin_memory  = PIN_MEMORY,
        persistent_workers = (nw > 0),
    )
    if USE_CUDA and nw > 0:
        kw["prefetch_factor"] = 2
    return DataLoader(ds, **kw)


def print_header() -> None:
    print("=" * 55)
    print(f"AGG + Neuropalsy | GPU={USE_CUDA} | batch={BATCH_SIZE}")
    print(f"SA: {SA_RESTARTS} restarts × {SA_EPOCHS} ep  | IP: {IP_EP} ep")
    print(f"κ∈[{KAPPA_MIN},{KAPPA_MAX}]")
    print("=" * 55)


def print_final_table(all_results: dict) -> None:
    """Print a publishable LaTeX-ready summary across all conditions."""
    print(
        "\n" + "=" * 70 +
        "\nFINAL PUBLISHABLE TABLE\n" +
        "=" * 70
    )
    header = (
        f"{'Condition':<14} {'FC°':>6} {'GPM_H°':>7} {'GPM_P°':>7} "
        f"{'vMF_c°':>7} {'vMF_n°':>7} {'κ':>6} {'95°cone':>8}"
    )
    print(header)
    print("─" * 70)

    for cond, res in all_results.items():
        fc    = res.get("fc",              np.zeros(1)).mean()
        gpm_h = res.get("gpm_healthy",     np.zeros(1)).mean()
        gpm_p = res.get("gpm_patho",       np.zeros(1)).mean()
        vmf_c = res.get("vmf_err_vs_clean", np.zeros(1)).mean()
        vmf_n = res.get("vmf_err_vs_noisy", np.zeros(1)).mean()
        kappa = res.get("vmf_kappa",       np.zeros(1)).mean()
        print(
            f"{cond:<14} {fc:>5.2f}°  {gpm_h:>6.2f}°  {gpm_p:>6.2f}°  "
            f"{vmf_c:>6.2f}°  {vmf_n:>6.2f}°  {kappa:>6.2f}  "
            f"{cone_95(float(kappa)):>7.1f}°"
        )

    print("─" * 70)
    print("\nColumns: FC = FC-head baseline | GPM_H = GPM-AGG (healthy)")
    print("         GPM_P = GPM-AGG (pathological) | vMF_c = vMF vs clean GT")
    print("         vMF_n = vMF vs noisy GT | κ = mean concentration")
    print("         95°cone = half-angle of 95%% probability cone\n")


def main() -> dict:
    print_header()

    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found at {DATASET_PATH!r}")
        print("Download it with: kagglehub.dataset_download('greninja2006/gazedataset')")
        return {}

    # ── Build datasets ────────────────────────────────────────────────────────
    train_ds = MPIIFaceGazeDataset(DATASET_PATH, split="train", ratio=TRAIN_RATIO)
    val_ds   = MPIIFaceGazeDataset(DATASET_PATH, split="test",  ratio=TRAIN_RATIO)

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader   = make_loader(val_ds,   shuffle=False)

    # ── Run pipeline for each condition ───────────────────────────────────────
    all_results: dict = {}
    for condition, severity in CONDITIONS:
        print(f"\n{'#' * 55}\n# {condition.upper()}  severity={severity}\n{'#' * 55}")
        agg = DualAGGFramework(condition=condition, severity=severity)
        res = agg.run_full_pipeline(train_loader, val_loader, train_ds, val_ds)
        all_results[condition] = res

    # ── Final combined table ──────────────────────────────────────────────────
    if all_results:
        print_final_table(all_results)

    return all_results


if __name__ == "__main__":
    results = main()

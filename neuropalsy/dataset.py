"""
neuropalsy/dataset.py
=====================
NeuropalsyDataset — wraps any base gaze dataset and augments each
sample with a physically simulated neuropathological gaze perturbation.

Each ``__getitem__`` returns a 3-tuple:
    (image, clean_gaze, noisy_gaze)

This allows the training loop to optimise for the perturbed distribution
while retaining the clean label for geometric accuracy evaluation.

The custom collate function ``neuropalsy_collate`` stacks the 3-tuples
into batched tensors suitable for direct use in DataLoader.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .injector import NeuropalsyNoiseInjector


class NeuropalsyDataset(Dataset):
    """
    Augmenting wrapper that adds condition-specific gaze noise.

    Parameters
    ----------
    base      : Dataset  — underlying clean dataset (image, clean_gaze)
    condition : str      — neuropathological condition name
    severity  : float    — perturbation magnitude in [0.0, 1.0]

    Returns (per sample)
    --------------------
    image      : torch.Tensor, shape (3, H, W)
    clean_gaze : torch.Tensor, shape (3,)  — original clean label
    noisy_gaze : torch.Tensor, shape (3,)  — perturbed label
    """

    def __init__(
        self,
        base:      Dataset,
        condition: str   = "nystagmus",
        severity:  float = 0.5,
    ) -> None:
        self.base    = base
        self.injector = NeuropalsyNoiseInjector(condition, severity)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, clean_gaze = self.base[idx]
        noisy_gaze = self.injector(
            clean_gaze.unsqueeze(0), t=idx
        ).squeeze(0)
        return img, clean_gaze, noisy_gaze


def neuropalsy_collate(batch):
    """
    Custom collate function for NeuropalsyDataset.

    Stacks a list of (image, clean_gaze, noisy_gaze) tuples into
    three batched tensors.

    Parameters
    ----------
    batch : list of (Tensor, Tensor, Tensor)

    Returns
    -------
    images      : (B, 3, H, W)
    clean_gazes : (B, 3)
    noisy_gazes : (B, 3)
    """
    return (
        torch.stack([b[0] for b in batch]),
        torch.stack([b[1] for b in batch]),
        torch.stack([b[2] for b in batch]),
    )

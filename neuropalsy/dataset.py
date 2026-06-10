from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .injector import NeuropalsyNoiseInjector


class NeuropalsyDataset(Dataset):

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
    
    return (
        torch.stack([b[0] for b in batch]),
        torch.stack([b[1] for b in batch]),
        torch.stack([b[2] for b in batch]),
    )

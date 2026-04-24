"""
models/isometric_propagator.py
==============================
Isometric Propagator (IP) — a two-hidden-layer MLP with LayerNorm that
maps 512-d CNN features to a 3-D Projected Gaze Frame (PGF).

Role in the pipeline
--------------------
1. **Forward** (feature → PGF):  used to compute the SOT training signal.
2. **Inverse** (PGF ← gaze):     `inverse_predict()` in RobustGPM converts
   a target gaze vector into the ideal PGF, which the CNN must produce.

Training
--------
Trained with L1 loss against Isomap-derived PGF coordinates in two
phases (Phase A warmup LR, Phase B refinement LR) — see FIX-2 in the
training framework.  Target L1 < 0.05 is required for stable SOT.
"""

import torch.nn as nn


class IsometricPropagator(nn.Module):
    """
    MLP: ℝ^{in_dim} → ℝ^{out_dim} (PGF space).

    Architecture
    ------------
    Linear(in_dim → hid) → LayerNorm → ReLU
    Linear(hid    → hid) → LayerNorm → ReLU
    Linear(hid    → out_dim)

    Parameters
    ----------
    in_dim  : int — input feature dimension (default 512, matching ResNet-18)
    hid     : int — hidden layer width (default 512)
    out_dim : int — output PGF dimension (default 3)
    """

    def __init__(
        self,
        in_dim:  int = 512,
        hid:     int = 512,
        out_dim: int = 3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.LayerNorm(hid), nn.ReLU(),
            nn.Linear(hid,    hid), nn.LayerNorm(hid), nn.ReLU(),
            nn.Linear(hid, out_dim),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, in_dim)

        Returns
        -------
        pgf : torch.Tensor, shape (B, out_dim)
        """
        return self.net(x)

"""
models/vmf_head.py
==================
von Mises–Fisher (vMF) probabilistic head for directional uncertainty
estimation on S².

Theory
------
A vMF distribution on the unit sphere S² is parameterised by:
  - μ ∈ S²  — mean direction (predicted gaze)
  - κ > 0   — concentration (inverse of angular spread)

The negative log-likelihood for a single observation y is:

    NLL = −[ κ · (μ·y) + log(κ) − log(2π) − log(sinh κ) ]

For large κ the log-sinh term is computed in a numerically stable way
(see ``_log_sinh``).

FIX-4: κ is clamped to [KAPPA_MIN, KAPPA_MAX] via Softplus + clamp,
preventing both κ collapse (near-uniform distribution) and κ explosion
(overconfident degenerate distribution).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import KAPPA_MIN, KAPPA_MAX


class vMFHead(nn.Module):
    """
    Predicts a vMF distribution (μ, κ) from CNN features.

    Parameters
    ----------
    in_dim : int — feature dimension (default 512)

    Outputs
    -------
    mu    : torch.Tensor, shape (B, 3)  — unit mean direction on S²
    kappa : torch.Tensor, shape (B, 1)  — concentration in [KAPPA_MIN, KAPPA_MAX]
    """

    def __init__(self, in_dim: int = 512) -> None:
        super().__init__()
        self.mu_head = nn.Linear(in_dim, 3)
        self.kappa_head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),   # maps to (0, +∞); clamped below
        )

    def forward(self, features):
        """
        Parameters
        ----------
        features : torch.Tensor, shape (B, in_dim)

        Returns
        -------
        mu    : torch.Tensor, shape (B, 3)
        kappa : torch.Tensor, shape (B, 1)
        """
        mu    = F.normalize(self.mu_head(features), dim=1)
        kappa = torch.clamp(
            self.kappa_head(features) + KAPPA_MIN,
            KAPPA_MIN,
            KAPPA_MAX,
        )
        return mu, kappa

    # ── Loss ──────────────────────────────────────────────────────────────────

    def nll_loss(
        self,
        mu:    torch.Tensor,
        kappa: torch.Tensor,
        y:     torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean negative log-likelihood of observations ``y`` under vMF(μ, κ).

        Parameters
        ----------
        mu    : (B, 3) — predicted mean direction (unit)
        kappa : (B, 1) — predicted concentration
        y     : (B, 3) — target gaze direction (need not be unit)

        Returns
        -------
        loss : scalar torch.Tensor
        """
        y        = F.normalize(y.float(), dim=1)
        dot      = (mu * y).sum(dim=1, keepdim=True)
        log_norm = (
            torch.log(kappa)
            - math.log(2 * math.pi)
            - self._log_sinh(kappa)
        )
        return -(kappa * dot + log_norm).mean()

    # ── Angular error ─────────────────────────────────────────────────────────

    def angular_error_from_mu(
        self,
        mu: torch.Tensor,
        y:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-sample angular error (degrees) between predicted μ and target y.

        Parameters
        ----------
        mu : (B, 3)
        y  : (B, 3)

        Returns
        -------
        errors : (B,)  — degrees
        """
        dot = torch.clamp(
            (F.normalize(mu, dim=1) * F.normalize(y.float(), dim=1)).sum(dim=1),
            -1.0, 1.0,
        )
        return torch.acos(dot) * 180.0 / math.pi

    # ── Numerically stable log(sinh(κ)) ──────────────────────────────────────

    @staticmethod
    def _log_sinh(k: torch.Tensor) -> torch.Tensor:
        """
        Numerically stable computation of log(sinh(κ)).

        For κ > 10: log(sinh k) ≈ k + log(1 − e^{-2k}) − log 2
        For κ ≤ 10: direct computation (avoids overflow in sinh).
        """
        large = k > 10.0
        return torch.where(
            large,
            k + torch.log(1 - torch.exp(-2 * k) + 1e-8) - math.log(2),
            torch.log(torch.sinh(k.clamp(1e-6, 10.0)) + 1e-8),
        )

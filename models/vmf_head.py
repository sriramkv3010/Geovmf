import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import KAPPA_MIN, KAPPA_MAX


class vMFHead(nn.Module):

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
        mu    = F.normalize(self.mu_head(features), dim=1)
        kappa = torch.clamp(
            self.kappa_head(features) + KAPPA_MIN,
            KAPPA_MIN,
            KAPPA_MAX,
        )
        return mu, kappa

    def nll_loss(
        self,
        mu:    torch.Tensor,
        kappa: torch.Tensor,
        y:     torch.Tensor,
    ) -> torch.Tensor:
        y        = F.normalize(y.float(), dim=1)
        dot      = (mu * y).sum(dim=1, keepdim=True)
        log_norm = (
            torch.log(kappa)
            - math.log(2 * math.pi)
            - self._log_sinh(kappa)
        )
        return -(kappa * dot + log_norm).mean()

    def angular_error_from_mu(
        self,
        mu: torch.Tensor,
        y:  torch.Tensor,
    ) -> torch.Tensor:
        
        dot = torch.clamp(
            (F.normalize(mu, dim=1) * F.normalize(y.float(), dim=1)).sum(dim=1),
            -1.0, 1.0,
        )
        return torch.acos(dot) * 180.0 / math.pi


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

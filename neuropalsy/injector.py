"""
neuropalsy/injector.py
======================
NeuropalsyNoiseInjector — physically motivated simulation of four
neuropathological eye movement disorders on gaze direction vectors.

Supported conditions
--------------------
nystagmus   — involuntary oscillatory eye movement (sinusoidal rotation)
strabismus  — constant misalignment along a fixed axis
restricted  — limited range of motion (cone clipping)
palsy       — irregular random-walk deviation (simulated nerve palsy)

Design notes
------------
- All gaze perturbations operate on unit S² vectors via Rodrigues'
  rotation formula.
- The ``palsy`` path is pre-generated (10 000 steps) to ensure
  reproducible per-sample perturbations across epochs.
- ``severity ∈ [0, 1]`` scales the magnitude of each perturbation.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F


class NeuropalsyNoiseInjector:
    """
    Inject condition-specific gaze perturbations.

    Parameters
    ----------
    condition : str   — one of {'nystagmus', 'strabismus', 'restricted', 'palsy'}
    severity  : float — perturbation magnitude in [0.0, 1.0]
    """

    CONDITIONS = ("nystagmus", "strabismus", "restricted", "palsy")

    def __init__(self, condition: str = "nystagmus", severity: float = 0.5) -> None:
        if condition not in self.CONDITIONS:
            raise ValueError(f"condition must be one of {self.CONDITIONS}")
        if not 0.0 <= severity <= 1.0:
            raise ValueError("severity must be in [0.0, 1.0]")

        self.condition = condition
        self.severity  = severity

        rng = np.random.default_rng(42)
        ax  = rng.standard_normal(3).astype(np.float32)
        self._strab_axis = ax / np.linalg.norm(ax)
        self._palsy_path = self._gen_palsy(10_000)

    # ── Palsy path generation ─────────────────────────────────────────────────

    @staticmethod
    def _gen_palsy(n: int) -> np.ndarray:
        """
        Simulate a random-walk on S² representing the irregular deviation
        trajectory of an ocular palsy.
        """
        path    = np.zeros((n, 3), dtype=np.float32)
        path[0] = [0.0, 0.0, 1.0]
        rng     = np.random.default_rng(0)
        for i in range(1, n):
            v       = path[i - 1] + rng.standard_normal(3).astype(np.float32) * 0.02
            path[i] = v / np.linalg.norm(v)
        return path

    # ── Rodrigues rotation ────────────────────────────────────────────────────

    @staticmethod
    def _rot(g: torch.Tensor, axis: torch.Tensor, angle: float) -> torch.Tensor:
        """
        Rotate batch of unit vectors ``g`` around ``axis`` by ``angle`` radians.

        Parameters
        ----------
        g     : (N, 3) — unit input vectors
        axis  : (3,)   — rotation axis (normalised internally)
        angle : float  — rotation angle in radians
        """
        axis = F.normalize(axis.unsqueeze(0), dim=1).squeeze(0)
        kxv  = torch.cross(axis.unsqueeze(0).expand_as(g), g, dim=1)
        kdv  = (axis * g).sum(dim=1, keepdim=True)
        return (
            g * math.cos(angle)
            + kxv * math.sin(angle)
            + axis.unsqueeze(0) * kdv * (1 - math.cos(angle))
        )

    # ── Perturbation dispatch ─────────────────────────────────────────────────

    def __call__(self, gaze: torch.Tensor, t: int = 0) -> torch.Tensor:
        """
        Apply neuropathological perturbation to a batch of gaze vectors.

        Parameters
        ----------
        gaze : torch.Tensor, shape (N, 3)  — input unit gaze vectors
        t    : int — time-step / sample index (used for nystagmus phase and
               palsy path lookup)

        Returns
        -------
        perturbed_gaze : torch.Tensor, shape (N, 3)  — unit vectors
        """
        g = F.normalize(gaze.float(), dim=1)
        d = g.device

        if self.condition == "nystagmus":
            # Sinusoidal oscillation at 5 Hz on the horizontal axis
            amp = math.radians(
                20.0 * self.severity * math.sin(2 * math.pi * 5.0 * t / 1000))
            return F.normalize(
                self._rot(g, torch.tensor([1.0, 0.0, 0.0]).to(d), amp), dim=1)

        elif self.condition == "strabismus":
            # Constant angular offset along a fixed random axis
            return F.normalize(
                self._rot(
                    g,
                    torch.tensor(self._strab_axis).to(d),
                    math.radians(25.0 * self.severity),
                ),
                dim=1,
            )

        elif self.condition == "restricted":
            # Gaze clipped to a cone of half-angle (80 − 60·severity)°
            cone = math.radians(80.0 - 60.0 * self.severity)
            fwd  = torch.tensor([0.0, 0.0, 1.0]).to(d)
            ang  = torch.acos(torch.clamp((g * fwd).sum(dim=1), -1.0, 1.0))
            out  = ang > cone
            if out.any():
                ts   = (cone / (ang + 1e-8)).unsqueeze(1)
                cl   = F.normalize(
                    g * (1 - ts) + fwd.unsqueeze(0) * ts, dim=1)
                mask = out.unsqueeze(1).float()
                g    = g * (1 - mask) + cl * mask
            return F.normalize(g, dim=1)

        elif self.condition == "palsy":
            # Irregular deviation following a pre-computed random-walk path
            dr = torch.tensor(
                self._palsy_path[t % len(self._palsy_path)]).to(d)
            return F.normalize(
                self._rot(g, dr, math.radians(25.0 * self.severity)), dim=1)

        return g   # identity (should not reach here)

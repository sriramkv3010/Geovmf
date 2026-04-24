"""
gpm/robust_gpm.py
=================
RobustGPM — Geometric Propagation Module combining Isomap manifold
embedding with multi-restart Sphere Alignment (SA).

Pipeline
--------
1. ``fit_isomap(features)``        — embed N×512 CNN features to N×3 via Isomap.
2. ``fit_sphere_alignment(pgf, gaze)``
                                   — learn a geometric mapping from PGF ∈ ℝ³
                                     to gaze directions ∈ S² via differentiable
                                     Euler-rotation + scale optimisation.
3. ``predict(pgf)``                — infer gaze from new PGF coordinates.
4. ``inverse_predict(gaze)``       — invert the SA map (used by SOT).
5. ``project_all_to_3d(target)``   — Isomap out-of-sample projection.

Key design decisions (FIX-1)
-----------------------------
- SA_RESTARTS independent random initialisations: k1, k2 ~ U[0.5, 2.0].
  This directly fixes the k2 ≈ 0.295 underfitting observed when starting
  from the origin.
- ReduceLROnPlateau patience raised from 30 → 50 to prevent premature
  LR decay before the optimiser escapes flat regions.
- Global best across all restarts is kept (not the last restart's result).
"""

from __future__ import annotations

import math
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import N_NEIGHBORS, SA_EPOCHS, SA_RESTARTS
from utils.geometry import (
    euler_to_rot,
    angles_to_vector,
    safe_arcsin,
    safe_atan2,
    angular_loss,
)


class RobustGPM:
    """
    Geometric Propagation Module with multi-restart Sphere Alignment.

    Parameters
    ----------
    n_neighbors : int  — Isomap k-NN graph size
    sa_epochs   : int  — SA gradient steps per restart
    n_restarts  : int  — number of random SA initialisations
    """

    def __init__(
        self,
        n_neighbors: int = N_NEIGHBORS,
        sa_epochs:   int = SA_EPOCHS,
        n_restarts:  int = SA_RESTARTS,
    ) -> None:
        self.n_neighbors     = n_neighbors
        self.sa_epochs       = sa_epochs
        self.n_restarts      = n_restarts
        self.source_features: np.ndarray | None = None
        self.source_pgf:      np.ndarray | None = None
        self._pgf_scale      = 1.0
        self.sphere_params:  dict                = {}

    # ── Isomap ────────────────────────────────────────────────────────────────

    def _run_isomap(self, X: np.ndarray) -> np.ndarray:
        """Run Isomap with PCA fallback on failure."""
        k = min(self.n_neighbors, len(X) - 1)
        try:
            from sklearn.manifold import Isomap
            return Isomap(n_components=3, n_neighbors=k, n_jobs=-1).fit_transform(X)
        except Exception as e:
            print(f"\n  ⚠ Isomap failed ({e}). PCA fallback.")
            from sklearn.decomposition import PCA
            return PCA(n_components=3).fit_transform(X)

    def fit_isomap(self, features: np.ndarray) -> "RobustGPM":
        """
        Embed ``features`` (N × 512) into a 3-D PGF via Isomap.
        Normalises by the median norm for scale stability.

        Parameters
        ----------
        features : np.ndarray, shape (N, 512)
        """
        print(f"  Isomap fit: {len(features)} pts ...", end="", flush=True)
        t0  = time.time()
        pgf = self._run_isomap(features)

        med = float(np.median(np.linalg.norm(pgf, axis=1)))
        self._pgf_scale = med if med > 1e-6 else 1.0
        pgf /= self._pgf_scale

        self.source_features = features.copy()
        self.source_pgf      = pgf

        norms = np.linalg.norm(pgf, axis=1)
        print(
            f" {time.time() - t0:.1f}s | "
            f"norm {norms.min():.2f}–{norms.max():.2f} "
            f"(med={np.median(norms):.2f})"
        )
        return self

    def project_all_to_3d(self, target: np.ndarray) -> np.ndarray:
        """
        Out-of-sample Isomap projection: concatenate source + target,
        re-run Isomap, return only the target rows.

        Parameters
        ----------
        target : np.ndarray, shape (M, 512)

        Returns
        -------
        pgf_target : np.ndarray, shape (M, 3)
        """
        combined = np.vstack([self.source_features, target])
        n_src    = len(self.source_features)
        print(f"  Isomap inf: {len(combined)} pts ...", end="", flush=True)
        t0  = time.time()
        out = self._run_isomap(combined)

        med = float(np.median(np.linalg.norm(out[:n_src], axis=1)))
        if med > 1e-6:
            out /= med
        print(f" {time.time() - t0:.1f}s")
        return out[n_src:]

    # ── Sphere Alignment ──────────────────────────────────────────────────────

    def fit_sphere_alignment(
        self,
        pgf:  np.ndarray,
        gaze: np.ndarray,
    ) -> "RobustGPM":
        """
        Learn the SA mapping PGF → gaze via multi-restart gradient descent.

        Parameters
        ----------
        pgf  : np.ndarray, shape (N, 3)
        gaze : np.ndarray, shape (N, 3)  — unit gaze vectors
        """
        pgf_t  = torch.FloatTensor(pgf)
        gaze_t = torch.FloatTensor(gaze)

        print(f"  SA: {self.n_restarts} restarts × {self.sa_epochs} epochs ...")
        global_best_loss = float("inf")
        global_best_p    = None

        for restart in range(self.n_restarts):
            rng   = np.random.default_rng(restart * 42)
            k1_0  = float(rng.uniform(0.5, 2.0))  # FIX-1: wider init range
            k2_0  = float(rng.uniform(0.5, 2.0))

            best_p, best_loss = self._single_sa(
                pgf_t, gaze_t,
                k1_init=k1_0, k2_init=k2_0,
                lr=0.01, label=f"R{restart + 1}",
            )
            print(
                f"    restart {restart + 1}/{self.n_restarts} | "
                f"best={best_loss:.4f}°  "
                f"k1={best_p['k1']:.3f}  k2={best_p['k2']:.3f}"
            )
            if best_loss < global_best_loss and self._valid(best_p):
                global_best_loss = best_loss
                global_best_p    = best_p

        if global_best_p is None or not self._valid(global_best_p):
            print("  ⚠ All SA restarts failed → identity fallback")
            global_best_p    = self._identity()
            global_best_loss = 90.0

        self.sphere_params = global_best_p
        flag = "✓" if global_best_loss < 10 else ("⚠" if global_best_loss < 14 else "✗")
        print(
            f"  SA final best: {global_best_loss:.4f}°  "
            f"k1={global_best_p['k1']:.3f}  k2={global_best_p['k2']:.3f}  "
            f"b1={global_best_p['b1']:.3f}  b2={global_best_p['b2']:.3f}  "
            f"{flag}"
        )
        return self

    def _single_sa(
        self,
        pgf_t:   torch.Tensor,
        gaze_t:  torch.Tensor,
        k1_init: float,
        k2_init: float,
        lr:      float,
        label:   str,
    ) -> tuple[dict, float]:
        """One SA run from a single initialisation."""
        center = nn.Parameter(torch.zeros(3))
        euler  = nn.Parameter(torch.zeros(3))
        k1     = nn.Parameter(torch.tensor([k1_init]))
        k2     = nn.Parameter(torch.tensor([k2_init]))
        b1     = nn.Parameter(torch.zeros(1))
        b2     = nn.Parameter(torch.zeros(1))
        params = [center, euler, k1, k2, b1, b2]

        opt = torch.optim.Adam(params, lr=lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=50, factor=0.5, min_lr=1e-6,   # FIX-1: patience 30→50
        )

        best_loss = float("inf")
        best_p    = None
        rep       = max(1, self.sa_epochs // 4)

        for ep in range(self.sa_epochs):
            opt.zero_grad()
            R     = euler_to_rot(euler)
            e     = F.normalize((pgf_t - center) @ R.T, dim=1)
            yaw   = safe_atan2(e[:, 0], e[:, 2])
            pitch = safe_arcsin(e[:, 1])
            pred  = angles_to_vector(k1 * yaw + b1, k2 * pitch + b2)

            if torch.isnan(pred).any():
                continue
            loss = angular_loss(pred, gaze_t)
            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sch.step(loss)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_p = dict(
                    center=center.detach().numpy().copy(),
                    R=R.detach().numpy().copy(),
                    k1=k1.item(), k2=k2.item(),
                    b1=b1.item(), b2=b2.item(),
                )

            if (ep + 1) % rep == 0:
                print(
                    f"    {label} ep{ep + 1}/{self.sa_epochs} | "
                    f"{loss.item():.4f}°  k1={k1.item():.3f}  k2={k2.item():.3f}"
                )

        return best_p or self._identity(), best_loss

    # ── Predict / Inverse ─────────────────────────────────────────────────────

    def predict(self, pgf: np.ndarray) -> np.ndarray:
        """
        Map PGF coordinates to gaze directions.

        Parameters
        ----------
        pgf : np.ndarray, shape (N, 3)

        Returns
        -------
        gaze_pred : np.ndarray, shape (N, 3)
        """
        p = self.sphere_params
        e = F.normalize(
            (torch.FloatTensor(pgf) - torch.FloatTensor(p["center"]))
            @ torch.FloatTensor(p["R"]).T,
            dim=1,
        )
        return angles_to_vector(
            p["k1"] * safe_atan2(e[:, 0], e[:, 2]) + p["b1"],
            p["k2"] * safe_arcsin(e[:, 1])           + p["b2"],
        ).detach().numpy()

    def inverse_predict(self, gaze: torch.Tensor) -> torch.Tensor:
        """
        Invert the SA map: gaze direction → ideal PGF (used in SOT).

        Parameters
        ----------
        gaze : torch.Tensor, shape (N, 3)

        Returns
        -------
        pgf_ideal : torch.Tensor, shape (N, 3)
        """
        p = self.sphere_params if self._valid(self.sphere_params) else self._identity()
        g = F.normalize(gaze.float(), dim=1)
        yr = (safe_atan2(g[:, 0], g[:, 2]) - p["b1"]) / (p["k1"] + 1e-8)
        pr = (safe_arcsin(g[:, 1])           - p["b2"]) / (p["k2"] + 1e-8)
        return (
            angles_to_vector(yr, pr)
            @ torch.FloatTensor(p["R"])
            + torch.FloatTensor(p["center"])
        )

    # ── Validity helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _valid(p: dict | None) -> bool:
        if p is None:
            return False
        return (
            all(
                math.isfinite(p.get(k, float("nan"))) and abs(p.get(k, 0)) < 100
                for k in ["k1", "k2", "b1", "b2"]
            )
            and abs(p.get("k1", 0)) > 1e-3
            and abs(p.get("k2", 0)) > 1e-3
        )

    @staticmethod
    def _identity() -> dict:
        return dict(center=np.zeros(3), R=np.eye(3),
                    k1=1.0, k2=1.0, b1=0.0, b2=0.0)

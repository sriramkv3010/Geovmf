"""
utils/geometry.py
=================
Angular geometry primitives on the unit sphere S².

All functions operate on PyTorch tensors unless suffixed with `_np`
(NumPy variants, used for post-hoc metric computation).
"""

import math
import numpy as np
import torch
import torch.nn.functional as F


# ─── Rotation helpers ─────────────────────────────────────────────────────────

def euler_to_rot(angles: torch.Tensor) -> torch.Tensor:
    """
    Construct a 3×3 rotation matrix from Euler angles (rx, ry, rz).

    Parameters
    ----------
    angles : torch.Tensor, shape (3,)
        Euler angles in radians: [roll, pitch, yaw].

    Returns
    -------
    R : torch.Tensor, shape (3, 3)
        Rotation matrix R = Rz @ Ry @ Rx.
    """
    rx, ry, rz = angles[0], angles[1], angles[2]
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)

    z = torch.zeros(1, dtype=angles.dtype, device=angles.device).squeeze()
    o = torch.ones(1,  dtype=angles.dtype, device=angles.device).squeeze()

    Rx = torch.stack([torch.stack([o,  z,   z]),
                      torch.stack([z,  cx, -sx]),
                      torch.stack([z,  sx,  cx])])
    Ry = torch.stack([torch.stack([ cy, z, sy]),
                      torch.stack([  z, o,  z]),
                      torch.stack([-sy, z, cy])])
    Rz = torch.stack([torch.stack([cz, -sz, z]),
                      torch.stack([sz,  cz, z]),
                      torch.stack([ z,   z, o])])
    return Rz @ Ry @ Rx


def angles_to_vector(yaw: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    """
    Convert (yaw, pitch) angles to unit 3-D gaze vectors.

    Parameters
    ----------
    yaw   : torch.Tensor, shape (N,)
    pitch : torch.Tensor, shape (N,)

    Returns
    -------
    v : torch.Tensor, shape (N, 3)  — unit vectors on S²
    """
    x = torch.cos(pitch) * torch.sin(yaw)
    y = torch.sin(pitch)
    z = torch.cos(pitch) * torch.cos(yaw)
    return F.normalize(torch.stack([x, y, z], dim=1), dim=1)


# ─── Safe trig (avoids NaN at poles) ─────────────────────────────────────────

def safe_arcsin(y: torch.Tensor) -> torch.Tensor:
    """arcsin clamped to (-1+ε, 1-ε) to avoid NaN gradients at poles."""
    return torch.asin(torch.clamp(y, -1.0 + 1e-4, 1.0 - 1e-4))


def safe_atan2(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """atan2 with a small epsilon added to z to avoid the z=0 discontinuity."""
    return torch.atan2(x, z + 1e-6)


# ─── Loss / error functions ───────────────────────────────────────────────────

def angular_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean angular error (degrees) between predicted and target gaze vectors.

    Parameters
    ----------
    pred   : torch.Tensor, shape (N, 3)  — not necessarily unit
    target : torch.Tensor, shape (N, 3)  — not necessarily unit

    Returns
    -------
    loss : torch.Tensor, scalar
    """
    dot = torch.clamp(
        (F.normalize(pred, dim=1) * F.normalize(target, dim=1)).sum(dim=1),
        -1.0, 1.0)
    return (torch.acos(dot) * 180.0 / math.pi).mean()


def angular_error_np(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Per-sample angular error in degrees (NumPy, for evaluation).

    Parameters
    ----------
    pred   : np.ndarray, shape (N, 3)
    target : np.ndarray, shape (N, 3)

    Returns
    -------
    errors : np.ndarray, shape (N,)  — degrees
    """
    p = pred   / (np.linalg.norm(pred,   axis=1, keepdims=True) + 1e-8)
    t = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8)
    return np.degrees(np.arccos(np.clip((p * t).sum(axis=1), -1.0, 1.0)))


# ─── vMF utilities ────────────────────────────────────────────────────────────

def cone_95(kappa: float) -> float:
    """
    Half-angle (degrees) of the 95% probability cone for a vMF distribution
    with concentration parameter κ.

    Parameters
    ----------
    kappa : float  — concentration (> 0)

    Returns
    -------
    angle : float  — degrees
    """
    return float(np.degrees(np.arccos(
        np.clip(1.0 + np.log(0.05) / (kappa + 1e-6), -1.0, 1.0))))

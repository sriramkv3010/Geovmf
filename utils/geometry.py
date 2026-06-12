import math
import numpy as np
import torch
import torch.nn.functional as F

def euler_to_rot(angles: torch.Tensor) -> torch.Tensor:
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
    x = torch.cos(pitch) * torch.sin(yaw)
    y = torch.sin(pitch)
    z = torch.cos(pitch) * torch.cos(yaw)
    return F.normalize(torch.stack([x, y, z], dim=1), dim=1)

def safe_arcsin(y: torch.Tensor) -> torch.Tensor:
    """arcsin clamped to (-1+ε, 1-ε) to avoid NaN gradients at poles."""
    return torch.asin(torch.clamp(y, -1.0 + 1e-4, 1.0 - 1e-4))

def safe_atan2(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """atan2 with a small epsilon added to z to avoid the z=0 discontinuity."""
    return torch.atan2(x, z + 1e-6)


def angular_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dot = torch.clamp(
        (F.normalize(pred, dim=1) * F.normalize(target, dim=1)).sum(dim=1),
        -1.0, 1.0)
    return (torch.acos(dot) * 180.0 / math.pi).mean()


def angular_error_np(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    p = pred   / (np.linalg.norm(pred,   axis=1, keepdims=True) + 1e-8)
    t = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8)
    return np.degrees(np.arccos(np.clip((p * t).sum(axis=1), -1.0, 1.0)))

def cone_95(kappa: float) -> float:
    return float(np.degrees(np.arccos(
        np.clip(1.0 + np.log(0.05) / (kappa + 1e-6), -1.0, 1.0))))

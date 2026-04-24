"""
utils/device.py
===============
Device detection and GPU memory utilities.
"""

import torch
from config import USE_CUDA, DEVICE


def get_device() -> tuple[torch.device, bool]:
    """
    Detect and validate the compute device.

    Performs a small matrix multiply on CUDA to confirm the GPU is functional
    before committing to it.

    Returns
    -------
    device   : torch.device
    use_cuda : bool
    """
    if not torch.cuda.is_available():
        print("[Device] CPU only — CUDA not available.")
        return torch.device("cpu"), False

    try:
        t = torch.zeros(256, 256, device="cuda")
        _ = (t @ t).sum().item()
        del t
        torch.cuda.empty_cache()
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Device] GPU → {name}  ({mem:.1f} GB VRAM)")
        return torch.device("cuda"), True
    except Exception as e:
        print(f"[Device] GPU smoke-test failed → CPU  ({e})")
        return torch.device("cpu"), False


def gpu_mem() -> None:
    """Print current GPU memory allocation (no-op on CPU)."""
    if USE_CUDA:
        allocated = torch.cuda.memory_allocated() / 1e9
        total     = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  [GPU] {allocated:.2f}/{total:.1f} GB allocated")

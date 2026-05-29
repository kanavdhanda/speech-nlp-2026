"""
device.py — Hardware-aware device selection for D-JEPA
Priority: MPS (Apple Silicon) → CUDA (T5/A100/consumer GPUs) → CPU
"""

from __future__ import annotations
import os
import torch


def get_device(verbose: bool = True) -> torch.device:
    """
    Returns the best available device.
    Override with env var DJEPA_DEVICE=cpu|cuda|mps if needed.
    """
    override = os.environ.get("DJEPA_DEVICE", "").lower().strip()
    if override:
        device = torch.device(override)
        if verbose:
            print(f"[device] Forced via DJEPA_DEVICE: {device}")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        name   = torch.cuda.get_device_name(0)
        if verbose:
            print(f"[device] CUDA — {name}")
        return device

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("[device] MPS — Apple Silicon GPU")
        return device

    if verbose:
        print("[device] CPU (no GPU detected)")
    return torch.device("cpu")


def device_info(device: torch.device) -> dict:
    """Return a dict of device metadata for README/logging."""
    info: dict = {"device": str(device)}
    if device.type == "cuda":
        info["name"]      = torch.cuda.get_device_name(0)
        info["vram_gb"]   = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        info["t5"]        = "T5" in info["name"] or "Tesla" in info["name"]
    elif device.type == "mps":
        info["name"]      = "Apple MPS"
        info["vram_gb"]   = "unified"
    else:
        import platform
        info["name"]      = platform.processor() or "CPU"
        info["vram_gb"]   = "N/A"
    return info

"""
Small runtime helpers for multi-GPU data parallelism and optional
torch.compile acceleration.
"""
import torch
import torch.nn as nn

from .config import device


def parallelize(model):
    """Wrap model with DataParallel if multiple GPUs available."""
    if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        return nn.DataParallel(model)
    return model


def maybe_compile(model, name=""):
    """
    Safe wrap with torch.compile for faster CNN inference.
    """
    if device != "cuda" or not hasattr(torch, "compile"):
        return model
    try:
        print(f"[INFO] Compiling model")
        return torch.compile(model, backend="cudagraphs")
    except Exception as e:
        print(f"  [WARN] torch.compile failed for {name}, using eager model: {e}")
        return model

import torch
import torch.nn as nn

"""
Not in use currently.
"""

class FakeIntQuantSTE(torch.autograd.Function):
    """
    Straight-Through Estimator for integer quantization.

    Forward: rounds and clamps the input to the representable integer range.
    Backward: passes the gradient through unchanged (STE).

    Args:
        bits: Number of quantization bits (e.g. 4 or 8).
    """

    def __init__(self, bits):
        if bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {bits}")
        self.bits = bits
        self.qmax = 2 ** (bits - 1) - 1
        self.qmin = -self.qmax - 1

    @staticmethod
    def forward(ctx, x):
        # Quantize: scale to [qmin, qmax], round, then rescale
        qmax = 2 ** (x.dim() // 2)  # placeholder; actual scaling done by caller
        x_scaled = x.clamp(-8, 7)  # approximate; use quantize_tensor wrapper for full behavior
        return torch.round(x_scaled)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: no clipping, no scaling in the backward pass
        return grad_output


class FakeIntQuantLayer(nn.Module):
    """
    Wraps FakeIntQuantSTE into an nn.Module that can be inserted into
    a model's module hierarchy.

    Note: this is the original (non-torchao) fake-quantization layer.
    The torchao-based Model class uses its own prepare/convert workflow
    via ``torchao.quantization.quantize_`` and does not rely on this layer.
    """

    def __init__(self, bits=8):
        super().__init__()
        self.bits = bits
        self.qmax = 2 ** (bits - 1) - 1
        self.qmin = -self.qmax - 1

    def forward(self, x):
        return FakeIntQuantSTE.apply(x)

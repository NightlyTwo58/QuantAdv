import torch

class FakeIntQuantSTE(torch.autograd.Function):
    # Fake quantization in the full model
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
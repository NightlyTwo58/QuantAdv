"""
Adversarial attack implementations (FGSM, PGD, BPDA-style PGD, random-noise
baseline, transfer attack) and small helpers (AMP context, gradient-masking
tolerance) shared across the evaluation and diagnostics modules.
"""
from contextlib import nullcontext, contextmanager

import torch
import torch.nn.functional as F

from .config import device, USE_AMP, CLIP_MIN_DEV, CLIP_MAX_DEV


def amp_ctx():
    if USE_AMP:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@contextmanager
def tolerate_masked_gradients():
    real_grad = torch.autograd.grad

    def _patched_grad(outputs, inputs, *args, **kwargs):
        kwargs.setdefault("allow_unused", True)
        grads = real_grad(outputs, inputs, *args, **kwargs)
        inputs_seq = [inputs] if isinstance(inputs, torch.Tensor) else list(inputs)
        return tuple(
            torch.zeros_like(inp) if g is None else g
            for g, inp in zip(grads, inputs_seq)
        )

    torch.autograd.grad = _patched_grad
    try:
        yield
    finally:
        torch.autograd.grad = real_grad


def pgd_step(model, x_adv, x, y, eps, alpha, clip_min=CLIP_MIN_DEV, clip_max=CLIP_MAX_DEV, return_grad=False):
    x_adv = x_adv.clone().requires_grad_(True)
    with amp_ctx():
        loss = F.cross_entropy(model(x_adv), y)
    grad = torch.autograd.grad(loss, x_adv, allow_unused=True)[0]
    if grad is None:
        grad = torch.zeros_like(x_adv)
    x_adv = x_adv.detach() + alpha * grad.sign()
    x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
    x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
    if return_grad:
        return x_adv, grad
    return x_adv


def fgsm_attack(model, x, y, eps, clip_min=CLIP_MIN_DEV, clip_max=CLIP_MAX_DEV):
    """Single-step FGSM, built on the same allow_unused-safe gradient logic
    as pgd_step, in place of torchattacks.FGSM (which hard-crashes instead
    of tolerating a fully-masked gradient)."""
    x_adv = x.clone().detach().requires_grad_(True)
    with amp_ctx():
        loss = F.cross_entropy(model(x_adv), y)
    grad = torch.autograd.grad(loss, x_adv, allow_unused=True)[0]
    if grad is None:
        grad = torch.zeros_like(x_adv)
    x_adv = x_adv.detach() + eps * grad.sign()
    x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
    return x_adv


def pgd_attack(model, x, y, eps, alpha=2 / 255, steps=20, random_start=True,
               clip_min=CLIP_MIN_DEV, clip_max=CLIP_MAX_DEV):
    """Multi-step PGD built on pgd_step, in place of torchattacks.PGD."""
    if random_start:
        x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
        x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
    else:
        x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv = pgd_step(model, x_adv, x, y, eps, alpha, clip_min=clip_min, clip_max=clip_max)
    return x_adv


def bpda_pgd_attack(model, x, y, eps=8 / 255, alpha=2 / 255, steps=20):
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.max(torch.min(x_adv, CLIP_MAX_DEV), CLIP_MIN_DEV).detach()
    for _ in range(steps):
        x_adv = pgd_step(model, x_adv, x, y, eps, alpha)
    return x_adv.detach()


def evaluate_under_attack(model, loader, attack_fn):
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with amp_ctx():
            x_adv = attack_fn(x, y)
            with torch.no_grad():
                pred = model(x_adv).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def random_noise_attack(model, loader, eps=8 / 255, n_restarts=1, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            worst_correct = torch.ones(y.size(0), dtype=torch.bool, device=device)
            for _ in range(n_restarts):
                noise = torch.empty_like(x).uniform_(-eps, eps)
                x_adv = torch.max(torch.min(x + noise, CLIP_MAX_DEV), CLIP_MIN_DEV)
                with amp_ctx():
                    pred = model(x_adv).argmax(dim=1)
                worst_correct &= (pred == y)
            correct += worst_correct.sum().item()
            total += y.size(0)
    return correct / total


def transfer_attack(source_model, target_model, loader, eps=8 / 255):
    return evaluate_under_attack(
        target_model, loader,
        lambda x, y: pgd_attack(source_model, x, y, eps, alpha=2 / 255, steps=20, random_start=True))

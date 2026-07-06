"""
Adversarial attack implementations (FGSM, PGD, BPDA-style PGD, random-noise
baseline, transfer attack) and small helpers (AMP context, gradient-masking
tolerance) shared across the evaluation and diagnostics modules.
"""
import warnings
from contextlib import nullcontext, contextmanager

import torch
import torch.nn.functional as F

from .config import device, USE_AMP, CLIP_MIN_DEV, CLIP_MAX_DEV

# Static loss-scaling factor used only to reduce FP16 gradient underflow when
# USE_AMP is on. autocast computes the forward pass in FP16, and a small
# cross-entropy loss can underflow to zero before autograd ever gets to the
# input -- which is indistinguishable, downstream, from a genuinely masked
# gradient. Scaling the loss up before differentiating and dividing the
# resulting gradient back down mitigates that without requiring a full
# GradScaler/optimizer-step integration (there's no optimizer step here).
AMP_GRAD_SCALE = 1024.0


def amp_ctx():
    if USE_AMP:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def attack_precision_ctx():
    """
    Attack generation is intentionally full precision.

    The archived reference generated FGSM/PGD/BPDA gradients without AMP.
    Autocast can change gradients enough to mask failures, especially around
    hard quantization. Keep AMP for cheap inference paths, but not for the
    optimization loop that constructs adversarial examples.
    """
    return nullcontext()


def set_ste_mode(model, flag):
    """
    Toggle straight-through fake-quantization modules when the model exposes
    the archived `use_ste` convention. Returns the number of modules toggled.
    """
    toggled = 0
    for mod in model.modules():
        if hasattr(mod, "use_ste"):
            mod.use_ste = flag
            toggled += 1
    return toggled


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


def _compute_gradient(loss, x_adv, *, name="attack step", scale=1.0):
    """
    Compute d(loss)/d(x_adv), tolerating a fully masked (`None`) gradient.
    """
    if scale != 1.0:
        loss = loss * scale
    try:
        grad = torch.autograd.grad(loss, x_adv, allow_unused=True)[0]
    except RuntimeError as e:
        warnings.warn(
            f"Gradient computation raised during {name} ({e}). This usually means "
            "the forward pass contains an op with no autograd backward defined at "
            "all -- common with true integer/quantized kernels -- rather than just "
            "a masked gradient. Falling back to a zero-gradient step, which will "
            "UNDERSTATE the true attack strength. Use bpda_pgd_attack with an "
            "explicit `backward_model` (e.g. the pre-quantization fp32 model) "
            "instead of trusting plain PGD/FGSM results here.",
            RuntimeWarning,
            stacklevel=3,
        )
        return torch.zeros_like(x_adv)
    if grad is None:
        warnings.warn(
            f"Gradient masked/unavailable during {name}: "
            "torch.autograd.grad returned None. Falling back to a "
            "zero-gradient step, which will UNDERSTATE the true attack "
            "strength against this model. If this is expected (e.g. a "
            "defense with a non-differentiable component), use "
            "bpda_pgd_attack with an explicit `backward_model` "
            "approximation instead of trusting plain PGD/FGSM results.",
            RuntimeWarning,
            stacklevel=3,
        )
        return torch.zeros_like(x_adv)
    if scale != 1.0:
        grad = grad / scale
    return grad


def _worst_of_restarts(x, y, model, candidate, best_adv, best_score):
    """
    Given a newly-generated `candidate` adversarial batch, keep the
    per-example worst case (highest loss, with any misclassification always
    beating any correct classification) against `best_adv`/`best_score` so
    far. Used to implement multi-restart PGD/BPDA.
    """
    with torch.no_grad(), amp_ctx():
        logits = model(candidate)
        loss_per_example = F.cross_entropy(logits, y, reduction="none")
        correct = logits.argmax(dim=1) == y
    score = loss_per_example + (~correct).float() * 1e6
    if best_adv is None:
        return candidate, score
    improve = score > best_score
    view_shape = (-1,) + (1,) * (x.dim() - 1)
    best_adv = torch.where(improve.view(view_shape), candidate, best_adv)
    best_score = torch.where(improve, score, best_score)
    return best_adv, best_score


def pgd_step(model, x_adv, x, y, eps, alpha, clip_min=CLIP_MIN_DEV, clip_max=CLIP_MAX_DEV, return_grad=False):
    """One PGD update: gradient-sign ascent step, L_inf projection onto the
    eps-ball around `x`, and a clip to the valid data range."""
    x_adv = x_adv.clone().requires_grad_(True)
    with attack_precision_ctx():
        loss = F.cross_entropy(model(x_adv), y)
    grad = _compute_gradient(loss, x_adv, name="pgd_step")
    x_adv = x_adv.detach() + alpha * grad.sign()
    x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
    x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
    if return_grad:
        return x_adv, grad
    return x_adv


def fgsm_attack(model, x, y, eps, clip_min=CLIP_MIN_DEV, clip_max=CLIP_MAX_DEV):
    """Single-step FGSM."""
    x_adv = x.clone().detach().requires_grad_(True)
    with attack_precision_ctx():
        loss = F.cross_entropy(model(x_adv), y)
    grad = _compute_gradient(loss, x_adv, name="fgsm_attack")
    x_adv = x_adv.detach() + eps * grad.sign()
    x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
    return x_adv


def pgd_attack(model, x, y, eps, alpha=2 / 255, steps=20, random_start=True,
               clip_min=CLIP_MIN_DEV, clip_max=CLIP_MAX_DEV, n_restarts=1):
    best_adv, best_score = None, None
    for _ in range(n_restarts):
        if random_start:
            x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
            x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
        else:
            x_adv = x.clone().detach()
        for _ in range(steps):
            x_adv = pgd_step(model, x_adv, x, y, eps, alpha, clip_min=clip_min, clip_max=clip_max)
        if n_restarts == 1:
            return x_adv
        best_adv, best_score = _worst_of_restarts(x, y, model, x_adv, best_adv, best_score)
    return best_adv


def bpda_pgd_attack(model, x, y, eps=8 / 255, alpha=2 / 255, steps=20, n_restarts=1,
                     backward_model=None, clip_min=CLIP_MIN_DEV, clip_max=CLIP_MAX_DEV):
    """
    BPDA-style PGD (Athalye et al., 2018, "Obfuscated Gradients Give a
    False Sense of Security").

    Plain PGD/FGSM fail against defenses with non-differentiable or
    gradient-masking components (quantization, JPEG-style compression,
    randomized/stochastic transforms, hard thresholds, etc): the gradient
    through those components comes back None or near-zero, the attack
    silently does nothing, and the defense looks robust without ever
    actually being tested. BPDA addresses this by computing gradients
    through a differentiable *approximation* of the offending component,
    while still measuring real attack success (loss/misclassification)
    against the true, non-differentiable `model`.

    Args:
        model: the real model under attack; its forward pass is what
            defines success/failure and what the final adversarial batch
            is evaluated against. May be non-differentiable or contain
            gradient-masking components.
        backward_model: a differentiable stand-in used ONLY to compute
            gradients on the backward pass -- e.g. `model` with a hard
            quantization/thresholding op swapped for a smooth
            approximation, or wrapped with a straight-through estimator
            (`y = nondiff_op(x).detach() + x - x.detach()`). This is the
            piece that makes BPDA meaningfully different from, and
            stronger than, plain PGD against a masked-gradient defense.
            If omitted, defaults to `model` itself: gradients are then
            computed directly through the real model, which is
            equivalent to `pgd_attack` when it is fully differentiable,
            and will emit a `RuntimeWarning` via `_compute_gradient`
            (rather than silently zero-filling) if it still is not.
        n_restarts: see `pgd_attack`; worst case is selected by evaluating
            each restart's final adversarial batch against the true
            `model`, not `backward_model`.

    Note: if you don't have a specific differentiable approximation to
    supply and the model is fully differentiable, use `pgd_attack`
    instead -- this function only earns its name when `backward_model`
    is a genuine substitute for a non-differentiable component.
    """
    # Prefer the archived behavior when available: use the true quantized model
    # for both forward and backward, but turn on its STE path during gradient
    # construction. Torchao converted models do not expose `use_ste`, so those
    # fall back to the supplied differentiable fp32 shadow model.
    use_model_ste = set_ste_mode(model, True) > 0
    if use_model_ste:
        backward_model = model
    else:
        backward_model = backward_model if backward_model is not None else model

    def _bpda_step(x_adv):
        x_adv = x_adv.clone().requires_grad_(True)
        with attack_precision_ctx():
            loss = F.cross_entropy(backward_model(x_adv), y)
        grad = _compute_gradient(loss, x_adv, name="bpda_pgd_attack")
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
        x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
        return x_adv

    try:
        best_adv, best_score = None, None
        for _ in range(n_restarts):
            x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
            x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
            for _ in range(steps):
                x_adv = _bpda_step(x_adv)
            x_adv = x_adv.detach()
            if n_restarts == 1:
                return x_adv
            # Success is judged against the real model, not the backward surrogate.
            best_adv, best_score = _worst_of_restarts(x, y, model, x_adv, best_adv, best_score)
        return best_adv.detach()
    finally:
        if use_model_ste:
            set_ste_mode(model, False)


def evaluate_under_attack(model, loader, attack_fn):
    """Run `attack_fn(x, y) -> x_adv` over `loader` and report clean-label
    accuracy of `model` on the resulting adversarial batches."""
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = attack_fn(x, y)
        with torch.no_grad(), amp_ctx():
            pred = model(x_adv).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def random_noise_attack(model, loader, eps=8 / 255, n_restarts=1, seed=None):
    """
    Non-adaptive uniform-noise baseline, no gradient is used at all.
    """
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


def transfer_attack(source_model, target_model, loader, eps=8 / 255, n_restarts=1):
    """Black-box transfer attack: craft adversarial examples with PGD
    against `source_model` only, then evaluate them against `target_model`
    (which is never queried for gradients)."""
    return evaluate_under_attack(
        target_model, loader,
        lambda x, y: pgd_attack(source_model, x, y, eps, alpha=2 / 255, steps=20,
                                 random_start=True, n_restarts=n_restarts))

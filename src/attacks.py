"""
Adversarial attack implementations (FGSM, PGD, BPDA-style PGD, random-noise
baseline, transfer attack) and small helpers (AMP context, gradient-masking
tolerance) shared across the evaluation and diagnostics modules.
"""
import warnings
from contextlib import nullcontext, contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks

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


def make_torchattack(attack_cls, model, *args, **kwargs):
    """
    Build a torchattacks attack instance configured with identity
    normalization (mean=0, std=1), so it operates directly on this
    pipeline's already-normalized input tensors and clip bounds
    (CLIP_MIN_DEV/CLIP_MAX_DEV), matching fgsm_attack/pgd_attack above
    rather than assuming raw [0, 1] pixel inputs.
    """
    attack = attack_cls(model, *args, **kwargs)
    attack.set_normalization_used(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
    return attack


def accuracy_under_torchattack(model, loader, attack, target_model=None, max_images=None):
    """Same contract as evaluate_under_attack, but for torchattacks-based
    attacks built via make_torchattack, with optional cross-model transfer
    evaluation and an optional cap on the number of images attacked (useful
    for query-heavy attacks like JSMA)."""
    target = target_model if target_model is not None else model
    correct, total, n_seen = 0, 0, 0
    for x, y in loader:
        if max_images is not None:
            if n_seen >= max_images:
                break
            remaining = max_images - n_seen
            x, y = x[:remaining], y[:remaining]
        x, y = x.to(device), y.to(device)
        x_adv = attack(x, y)
        with torch.no_grad(), amp_ctx():
            pred = target(x_adv).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        n_seen += y.size(0)
    return correct / total if total else None


def transfer_attack_mim(source_model, target_model, loader, eps=8 / 255):
    """Black-box transfer attack using MI-FGSM (momentum iterative FGSM)
    instead of plain PGD as the source attack."""
    mim = make_torchattack(torchattacks.MIFGSM, source_model, eps=eps, alpha=2 / 255, steps=20, decay=1.0)
    return accuracy_under_torchattack(source_model, loader, mim, target_model=target_model)


def build_uap(model, loader, eps=8 / 255, delta=0.2, max_iter=10, deepfool_steps=20,
              overshoot=0.02, max_images=1000):
    """
    Universal Adversarial Perturbation (Moosavi-Dezfooli et al., 2017): a
    single perturbation `v`, shared across all inputs, built by repeatedly
    running DeepFool against still-correctly-classified examples and
    accumulating the minimal per-example perturbation into `v`, clipped to
    the L_inf eps-ball. Iterates until the fooling rate exceeds `1 - delta`
    or `max_iter` rounds are exhausted.
    """
    model.eval()
    sample_x, _ = next(iter(loader))
    v = torch.zeros(1, *sample_x.shape[1:], device=device)
    deepfool = make_torchattack(torchattacks.DeepFool, model, steps=deepfool_steps, overshoot=overshoot)

    fooling_rate, it = 0.0, 0
    while fooling_rate < (1 - delta) and it < max_iter:
        n_seen, n_fooled = 0, 0
        for x, y in loader:
            if n_seen >= max_images:
                break
            x, y = x.to(device), y.to(device)
            x_pert = torch.max(torch.min(x + v, CLIP_MAX_DEV), CLIP_MIN_DEV)
            with torch.no_grad():
                pred_orig = model(x).argmax(dim=1)
                pred_pert = model(x_pert).argmax(dim=1)
            still_correct = pred_pert == pred_orig
            if still_correct.any():
                xs, ys = x_pert[still_correct], pred_orig[still_correct]
                x_adv = deepfool(xs, ys)
                v = torch.clamp(v + (x_adv - xs).mean(dim=0, keepdim=True), -eps, eps)
            n_fooled += (pred_pert != pred_orig).sum().item()
            n_seen += y.size(0)
        fooling_rate = n_fooled / max(n_seen, 1)
        it += 1
    return v.detach()


def run_uap_attack(model, loader, eps=8 / 255, max_images=1000):
    """White-box UAP: build the perturbation against `model` itself and
    report `model`'s accuracy on the perturbed inputs."""
    v = build_uap(model, loader, eps=eps, max_images=max_images)
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = torch.max(torch.min(x + v, CLIP_MAX_DEV), CLIP_MIN_DEV)
        with torch.no_grad(), amp_ctx():
            pred = model(x_adv).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def transfer_uap_attack(source_model, target_model, loader, eps=8 / 255, max_images=1000):
    """Black-box UAP transfer: build the perturbation against `source_model`
    and evaluate it against `target_model`."""
    v = build_uap(source_model, loader, eps=eps, max_images=max_images)
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = torch.max(torch.min(x + v, CLIP_MAX_DEV), CLIP_MIN_DEV)
        with torch.no_grad(), amp_ctx():
            pred = target_model(x_adv).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def nes_estimate_gradient(model, x, y, n_samples=20, sigma=1e-3, query_chunk=512):
    """
    Antithetic NES (Natural Evolution Strategies) gradient estimate of
    d(loss)/dx, computed from forward passes only -- no backprop through
    the model. This is the query-only, ZOO/NES-style estimate used as the
    basis of a genuinely black-box attack, unaffected by gradient masking
    since no autograd call is made through `model` at all.
    """
    if n_samples % 2 != 0:
        n_samples += 1
    n_pairs = n_samples // 2
    B = x.size(0)
    grad_acc = torch.zeros_like(x)

    remaining = n_pairs
    while remaining > 0:
        chunk = max(1, min(remaining, query_chunk // max(B, 1)))
        u = torch.randn(chunk, *x.shape, device=x.device)
        x_plus = (x.unsqueeze(0) + sigma * u).view(chunk * B, *x.shape[1:])
        x_minus = (x.unsqueeze(0) - sigma * u).view(chunk * B, *x.shape[1:])
        y_rep = y.repeat(chunk)

        with torch.no_grad(), amp_ctx():
            loss_plus = F.cross_entropy(model(x_plus), y_rep, reduction="none").view(chunk, B)
            loss_minus = F.cross_entropy(model(x_minus), y_rep, reduction="none").view(chunk, B)

        weight = (loss_plus - loss_minus).view(chunk, B, 1, 1, 1)
        grad_acc += (weight * u).sum(dim=0)
        remaining -= chunk

    return grad_acc / (2 * n_pairs * sigma)


def nes_pgd_attack(model, x, y, eps=8 / 255, alpha=2 / 255, steps=10, n_samples=20,
                    sigma=1e-3, query_chunk=512, clip_min=CLIP_MIN_DEV, clip_max=CLIP_MAX_DEV):
    """Black-box L_inf PGD that substitutes the NES gradient estimate above
    for the true gradient; otherwise identical in structure to pgd_attack."""
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()

    for _ in range(steps):
        grad = nes_estimate_gradient(model, x_adv, y, n_samples=n_samples, sigma=sigma, query_chunk=query_chunk)
        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
        x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
    return x_adv.detach()


def nes_attack(model, loader, eps=8 / 255, n_samples=20, sigma=1e-3, alpha=2 / 255,
               steps=10, seed=None, query_chunk=512):
    """Single-seed black-box NES attack accuracy over the whole loader."""
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = nes_pgd_attack(model, x, y, eps=eps, alpha=alpha, steps=steps,
                                n_samples=n_samples, sigma=sigma, query_chunk=query_chunk)
        with torch.no_grad(), amp_ctx():
            pred = model(x_adv).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


class SubstituteCNN(nn.Module):
    """Small CNN used as the substitute/surrogate model trained via Jacobian-
    based dataset augmentation for the black-box surrogate attack below."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_substitute(target_model, seed_x, rounds=6, epochs_per_round=10, lr=1e-3,
                      lam=0.1, batch_size=128):
    """
    Trains a SubstituteCNN to mimic `target_model`'s predicted labels
    (Papernot et al., 2017, "Practical Black-Box Attacks against Machine
    Learning"), using Jacobian-based dataset augmentation: after each
    training round (except the last), the seed set is doubled with FGSM-
    perturbed copies (perturbed w.r.t. the substitute's own gradient) so
    later rounds probe the target's decision boundary more finely.
    """
    target_model.eval()
    substitute = SubstituteCNN().to(device)
    opt = torch.optim.Adam(substitute.parameters(), lr=lr)
    x = seed_x.clone().to(device)

    for r in range(rounds):
        with torch.no_grad():
            y = target_model(x).argmax(dim=1)

        substitute.train()
        ds = torch.utils.data.TensorDataset(x, y)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        for _ in range(epochs_per_round):
            for xb, yb in dl:
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(substitute(xb), yb)
                loss.backward()
                opt.step()

        if r < rounds - 1:
            substitute.eval()
            x_in = x.clone().requires_grad_(True)
            loss = F.cross_entropy(substitute(x_in), y)
            grad = torch.autograd.grad(loss, x_in)[0]
            x_aug = (x + lam * grad.sign()).detach()
            x = torch.cat([x, x_aug], dim=0)

    substitute.eval()
    return substitute


def run_surrogate_attack(model, loader, eps=8 / 255, seed_n=500, rounds=6):
    """Trains a substitute model against `model`'s hard labels, crafts PGD
    adversarial examples against the substitute, and reports `model`'s
    accuracy on those examples: a fully black-box (query-only for training,
    then zero-query for attack generation) robustness check."""
    x_seed, n = [], 0
    for x, _ in loader:
        x_seed.append(x)
        n += x.size(0)
        if n >= seed_n:
            break
    x_seed = torch.cat(x_seed, dim=0)[:seed_n]

    substitute = train_substitute(model, x_seed, rounds=rounds)
    return transfer_attack(substitute, model, loader, eps=eps)


def _predict_batch(model, x):
    with torch.no_grad():
        return model(x).argmax(dim=1)


def boundary_attack_single(model, x_orig, y_true, clip_min, clip_max, steps=200,
                            spherical_step=1e-2, source_step=1e-2, step_adapt=1.5,
                            init_tries=200, init_chunk=25):
    """
    Decision-based Boundary Attack (Brendel, Bethge, 2018) for a single
    already-correctly-classified image: starts from a large random
    perturbation that is already misclassified, then walks along the
    decision boundary while shrinking distance to `x_orig`, using only
    hard-label predictions (no gradients, no confidence scores at all).
    """
    clip_min = clip_min.to(x_orig.device)
    clip_max = clip_max.to(x_orig.device)

    x_adv = None
    tries_left = init_tries
    while tries_left > 0 and x_adv is None:
        chunk = min(init_chunk, tries_left)
        cand = torch.rand(chunk, *x_orig.shape, device=x_orig.device)
        cand = cand * (clip_max - clip_min) + clip_min
        preds = _predict_batch(model, cand)
        mismatch = (preds != y_true).nonzero(as_tuple=True)[0]
        if mismatch.numel() > 0:
            x_adv = cand[mismatch[0]].clone()
        tries_left -= chunk

    if x_adv is None:
        return x_orig.clone()

    sph_step, src_step = spherical_step, source_step
    sph_hist, src_hist = [], []

    for i in range(steps):
        diff = x_orig - x_adv
        dist = diff.norm()
        if dist.item() < 1e-12:
            break

        perturb = torch.randn_like(x_adv)
        perturb = perturb - (perturb * diff).sum() / (dist ** 2) * diff
        perturb = perturb / (perturb.norm() + 1e-12) * dist * sph_step
        cand = x_adv + perturb
        new_diff = x_orig - cand
        cand = x_orig - new_diff / (new_diff.norm() + 1e-12) * dist
        cand = torch.clamp(cand, clip_min, clip_max)

        sph_ok = (_predict_batch(model, cand.unsqueeze(0))[0] != y_true).item()
        sph_hist.append(sph_ok)

        if sph_ok:
            cand2 = torch.clamp(cand + src_step * (x_orig - cand), clip_min, clip_max)
            src_ok = (_predict_batch(model, cand2.unsqueeze(0))[0] != y_true).item()
            src_hist.append(src_ok)
            if src_ok:
                x_adv = cand2

        if (i + 1) % 10 == 0:
            if sph_hist:
                rate = np.mean(sph_hist[-10:])
                sph_step *= step_adapt if rate > 0.5 else (1 / step_adapt if rate < 0.2 else 1.0)
            if src_hist:
                rate = np.mean(src_hist[-10:])
                src_step *= step_adapt if rate > 0.5 else (1 / step_adapt if rate < 0.2 else 1.0)

    return x_adv.detach()


def run_boundary_attack(model, loader, eps=8 / 255, max_images=50, steps=200, seed=0,
                         clip_min=CLIP_MIN_DEV, clip_max=CLIP_MAX_DEV):
    """
    Runs the Boundary Attack on up to `max_images` correctly-classified
    examples (inherently per-sample and query-heavy, so the full eval set
    is not used) and reports:
      - Boundary_acc: fraction of attacked images whose minimal L_inf
        perturbation exceeds eps, i.e. would still be correctly classified
        within an eps budget -- directly comparable to the other robust-
        accuracy columns produced elsewhere in this pipeline.
      - Boundary_mean_Linf: mean minimal L_inf distance to the boundary
        found.
    """
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    clip_min, clip_max = clip_min.squeeze(0), clip_max.squeeze(0)

    dists = []
    n_seen = 0
    for x, y in loader:
        if n_seen >= max_images:
            break
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x).argmax(dim=1)
        for i in range(x.size(0)):
            if n_seen >= max_images:
                break
            if pred[i] != y[i]:
                continue
            x_adv = boundary_attack_single(model, x[i], y[i], clip_min, clip_max, steps=steps)
            dists.append((x_adv - x[i]).abs().max().item())
            n_seen += 1

    if not dists:
        return {"Boundary_acc": None, "Boundary_mean_Linf": None}

    dists = np.array(dists)
    return {
        "Boundary_acc": float((dists > eps).mean()),
        "Boundary_mean_Linf": float(dists.mean()),
    }

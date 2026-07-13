"""Defense wrappers and defense-training utilities for QuantAdv experiments.

The main runner uses this module for input transformations, randomized
smoothing, rejection-style guardrails, adversarial-example detectors, and
optional adversarial training.  The lightweight quantized copy used for
adversarial training intentionally mirrors the public attributes of the main
QuantAdv fake-quantized layers so attacks can switch between hard rounding and
straight-through gradients.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchattacks
from torch.amp import autocast, GradScaler

from config import *


def _cfg(name, default):
    """Return a config value while keeping this module importable in isolation."""
    return globals().get(name, default)


def _device():
    """Return the configured torch device as a ``torch.device`` instance."""
    return device if isinstance(device, torch.device) else torch.device(device)


def normalize_pixels(x):
    """Normalize pixel-space tensors with the configured dataset constants."""
    return (x - DATASET_MEAN.to(x.device)) / DATASET_STD.to(x.device)


def denormalize_inputs(x):
    """Map normalized dataset tensors back to pixel space."""
    return x * DATASET_STD.to(x.device) + DATASET_MEAN.to(x.device)


def make_attack(attack_cls, model, *args, **kwargs):
    """Construct a torchattacks object for normalized configured-dataset inputs."""
    attack = attack_cls(model, *args, **kwargs)
    attack.set_normalization_used(mean=DATASET_MEAN_VALUES, std=DATASET_STD_VALUES)
    return attack


class _FakeQuantSTE(torch.autograd.Function):
    """Round in the forward pass and use identity gradients in the backward pass."""

    @staticmethod
    def forward(ctx, x):
        """Round values during the forward pass of the STE quantizer."""
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        """Pass gradients through the STE quantizer unchanged."""
        return grad_output


def _quantize_tensor(t, bits, use_ste=False):
    """Symmetric per-tensor fake quantization used by defense-side copies."""
    if bits is None:
        return t
    qmax = 2 ** (bits - 1) - 1
    scale = torch.clamp(
        t.detach().abs().max() / qmax, min=_cfg("QUANT_SCALE_MIN", 1e-8)
    )
    scaled = t / scale
    q = (_FakeQuantSTE.apply(scaled) if use_ste else torch.round(scaled)).clamp(
        -qmax - 1, qmax
    )
    return q * scale


class _QuantConv2d(nn.Conv2d):
    """Conv2d with fake-quantized weights and/or activations."""

    def forward(self, x):
        """Run convolution with optional fake-quantized weights and activations."""
        w = (
            _quantize_tensor(self.weight, self.bits, self.use_ste)
            if self.quant_weight
            else self.weight
        )
        out = self._conv_forward(x, w, self.bias)
        return _quantize_tensor(out, self.bits, self.use_ste) if self.quant_act else out


class _QuantLinear(nn.Linear):
    """Linear layer with fake-quantized weights and/or activations."""

    def forward(self, x):
        """Run a linear projection with optional fake-quantized weights and activations."""
        w = (
            _quantize_tensor(self.weight, self.bits, self.use_ste)
            if self.quant_weight
            else self.weight
        )
        out = F.linear(x, w, self.bias)
        return _quantize_tensor(out, self.bits, self.use_ste) if self.quant_act else out


def _to_quant_module(module, bits):
    """Convert supported leaf layers to defense-side fake-quantized layers."""
    if isinstance(module, nn.Conv2d):
        new = _QuantConv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.bias is not None,
            module.padding_mode,
        )
        new.weight = module.weight
        if module.bias is not None:
            new.bias = module.bias
        new.bits = bits
        new.use_ste = _cfg("QUANT_DEFAULT_USE_STE", False)
        new.quant_weight = _cfg("QUANT_DEFAULT_WEIGHT", True)
        new.quant_act = _cfg("QUANT_DEFAULT_ACT", True)
        return new

    if isinstance(module, nn.Linear):
        new = _QuantLinear(
            module.in_features, module.out_features, bias=module.bias is not None
        )
        new.weight = module.weight
        if module.bias is not None:
            new.bias = module.bias
        new.bits = bits
        new.use_ste = _cfg("QUANT_DEFAULT_USE_STE", False)
        new.quant_weight = _cfg("QUANT_DEFAULT_WEIGHT", True)
        new.quant_act = _cfg("QUANT_DEFAULT_ACT", True)
        return new

    return None


def _replace_quant_modules(module, bits):
    """Recursively replace Conv2d/Linear children with quantized equivalents."""
    for name, child in list(module.named_children()):
        replacement = _to_quant_module(child, bits)
        if replacement is None:
            _replace_quant_modules(child, bits)
        else:
            setattr(module, name, replacement)


def quantized_copy(model, bits):
    """Return a model copy with Conv2d/Linear layers fake-quantized."""
    m = copy.deepcopy(model)
    _replace_quant_modules(m, bits)
    return m.to(_device())


def _set_quant_ste(model, flag):
    """Toggle straight-through gradients on defense-side quantized modules."""
    toggled = 0
    for module in model.modules():
        if hasattr(module, "use_ste") and hasattr(module, "bits"):
            module.use_ste = flag
            toggled += 1
    return toggled


def prepare_adversarial_training(base_model, loader, bits=None):
    """Fine-tune a copy of ``base_model`` on PGD adversarial examples."""
    dev = _device()
    model = (
        quantized_copy(base_model, bits)
        if bits is not None
        else copy.deepcopy(base_model).to(dev)
    )
    model.train()

    epochs = _cfg("DEFENSE_AT_EPOCHS", 1)
    lr = _cfg("DEFENSE_AT_LR", 1e-3)
    eps = _cfg("DEFAULT_EPS", 8 / 255)
    alpha = _cfg("PGD_ALPHA", 2 / 255)
    steps = _cfg("DEFENSE_AT_PGD_STEPS", 3)

    opt = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=_cfg("QAT_MOMENTUM", 0.9),
        weight_decay=_cfg("QAT_WEIGHT_DECAY", 5e-4),
    )
    scaler = GradScaler(device=dev.type)
    attack = make_attack(
        torchattacks.PGD, model, eps=eps, alpha=alpha, steps=steps, random_start=True
    )

    for _ in range(epochs):
        for x, y in loader:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            try:
                model.eval()
                _set_quant_ste(model, True)
                x_adv = attack(x, y).detach()
                model.train()

                opt.zero_grad(set_to_none=True)
                with autocast(device_type=dev.type):
                    loss = F.cross_entropy(model(x_adv), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            finally:
                _set_quant_ste(model, False)

    return model.eval()


class SanitizedModel(nn.Module):
    def __init__(self, model):
        """Wrap a classifier with deterministic input sanitization settings."""
        super().__init__()
        self.model = model
        self.resize = _cfg("DEFENSE_SANITIZE_SIZE", 28)
        self.bits = _cfg("DEFENSE_SANITIZE_BITS", 6)

    def forward(self, x):
        """Sanitize inputs in pixel space before normalized model inference."""
        pixels = denormalize_inputs(x).clamp(0.0, 1.0)
        pixels = T.functional.gaussian_blur(pixels, kernel_size=3)
        pixels = F.interpolate(
            pixels,
            size=(self.resize, self.resize),
            mode="bilinear",
            align_corners=False,
        )
        pixels = F.interpolate(
            pixels,
            size=(DATASET_IMAGE_SIZE, DATASET_IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        levels = float(2**self.bits - 1)
        pixels = torch.round(pixels * levels).div(levels)
        return self.model(normalize_pixels(pixels.clamp(0.0, 1.0)))


class SmoothedModel(nn.Module):
    def __init__(self, model):
        """Wrap a classifier with randomized smoothing parameters."""
        super().__init__()
        self.model = model
        self.sigma = _cfg("DEFENSE_SMOOTH_SIGMA", 0.12)
        self.samples = _cfg("DEFENSE_SMOOTH_SAMPLES", 8)

    def forward(self, x):
        """Average noisy predictions at evaluation time and sample once during training."""
        if not self.training:
            logits = 0.0
            for _ in range(self.samples):
                noise = torch.randn_like(x) * self.sigma
                logits = logits + self.model(
                    (x + noise).clamp(CLIP_MIN.to(x.device), CLIP_MAX.to(x.device))
                )
            return logits / self.samples
        noise = torch.randn_like(x) * self.sigma
        return self.model(
            (x + noise).clamp(CLIP_MIN.to(x.device), CLIP_MAX.to(x.device))
        )


class GuardrailModel(nn.Module):
    def __init__(self, model):
        """Wrap a classifier with confidence-threshold rejection settings."""
        super().__init__()
        self.model = model
        self.conf_threshold = _cfg("DEFENSE_GUARD_CONF_THRESHOLD", 0.55)
        self.reject_logit = _cfg("DEFENSE_REJECT_LOGIT", -1e4)

    def forward(self, x):
        """Replace low-confidence predictions with the rejection logit."""
        logits = self.model(x)
        conf = F.softmax(logits, dim=1).max(dim=1).values
        reject = conf < self.conf_threshold
        guarded = logits.clone()
        guarded[reject] = self.reject_logit
        return guarded


class DetectorCNN(nn.Module):
    def __init__(self):
        """Build the convolutional binary adversarial-example detector."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(DATASET_INPUT_CHANNELS, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        """Return clean-versus-adversarial detector logits for a batch."""
        return self.net(x)


def train_adversarial_detector(base_model, loader):
    """Train a binary detector to distinguish clean and adversarial inputs."""
    dev = _device()
    base_model = base_model.to(dev).eval()
    detector = DetectorCNN().to(dev).train()

    epochs = _cfg("DEFENSE_DETECTOR_EPOCHS", 1)
    lr = _cfg("DEFENSE_DETECTOR_LR", 1e-3)
    eps = _cfg("DEFAULT_EPS", 8 / 255)
    alpha = _cfg("PGD_ALPHA", 2 / 255)
    steps = _cfg("DEFENSE_DETECTOR_PGD_STEPS", 3)

    opt = torch.optim.AdamW(
        detector.parameters(),
        lr=lr,
        weight_decay=_cfg("DEFENSE_DETECTOR_WEIGHT_DECAY", 1e-4),
    )
    attack = make_attack(
        torchattacks.PGD,
        base_model,
        eps=eps,
        alpha=alpha,
        steps=steps,
        random_start=True,
    )

    for _ in range(epochs):
        for x, y in loader:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)

            x_adv = attack(x, y).detach()

            xb = torch.cat([x, x_adv], dim=0)
            yb = torch.cat(
                [
                    torch.zeros(x.size(0), dtype=torch.long, device=dev),
                    torch.ones(x_adv.size(0), dtype=torch.long, device=dev),
                ]
            )

            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(detector(xb), yb)
            loss.backward()
            opt.step()

    return detector.eval()


class DetectGuardModel(nn.Module):
    def __init__(self, model, detector):
        """Wrap a classifier with a learned detector-based rejection guard."""
        super().__init__()
        self.model = model
        self.detector = detector
        self.threshold = _cfg("DEFENSE_DETECTOR_THRESHOLD", 0.5)
        self.reject_logit = _cfg("DEFENSE_REJECT_LOGIT", -1e4)

    def forward(self, x):
        """Reject inputs that the detector classifies as adversarial."""
        logits = self.model(x)
        detector_prob = F.softmax(self.detector(x), dim=1)[:, 1]
        reject = detector_prob > self.threshold
        guarded = logits.clone()
        guarded[reject] = self.reject_logit
        return guarded


def run_certified_accuracy(model, loader):
    """Estimate smoothed-model accuracy and a radius proxy."""
    dev = _device()
    model.eval()
    correct = 0
    total = 0
    radius_sum = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            top2 = probs.topk(2, dim=1).values
            pred = logits.argmax(dim=1)
            margin = top2[:, 0] - top2[:, 1]
            correct += (pred == y).sum().item()
            total += y.size(0)
            radius_sum += (model.sigma * margin).sum().item()

    return {
        "certified_acc": correct / total if total else None,
        "mean_cert_radius_proxy": radius_sum / total if total else None,
    }


def run_guardrail_flagging_rate(model, loader, attack=None):
    """Measure how often a confidence guardrail flags clean and attacked inputs."""
    return _run_flagging_rate(model, loader, attack, detector_attr=None)


def run_detector_catch_rate(model, loader, attack=None):
    """Measure how often the learned detector catches adversarial inputs."""
    return _run_flagging_rate(model, loader, attack, detector_attr="detector")


def _run_flagging_rate(model, loader, attack=None, detector_attr=None):
    """Compute clean and adversarial flagging rates for a rejection model."""
    dev = _device()
    model.eval()
    clean_flags = 0
    adv_flags = 0
    total = 0

    for x, y in loader:
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)

        with torch.no_grad():
            clean_flags += _flag_mask(model, x, detector_attr).sum().item()

        if attack is not None:
            x_adv = attack(x, y).detach()
            with torch.no_grad():
                adv_flags += _flag_mask(model, x_adv, detector_attr).sum().item()

        total += y.size(0)

    return {
        "clean_flag_rate": clean_flags / total if total else None,
        "adv_flag_rate": adv_flags / total if total and attack is not None else None,
    }


def _flag_mask(model, x, detector_attr):
    """Return the boolean mask of inputs rejected by a guard or detector."""
    if detector_attr is not None and hasattr(model, detector_attr):
        prob = F.softmax(model.detector(x), dim=1)[:, 1]
        return prob > model.threshold

    logits = model.model(x) if hasattr(model, "model") else model(x)
    conf = F.softmax(logits, dim=1).max(dim=1).values
    return conf < getattr(model, "conf_threshold", 0.55)

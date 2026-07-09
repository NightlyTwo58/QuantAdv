import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchattacks
from torch.amp import autocast, GradScaler

from .config import *


def _cfg(name, default):
    return globals().get(name, default)


def _device():
    return device if isinstance(device, torch.device) else torch.device(device)


def normalize_pixels(x):
    return (x - CIFAR_MEAN.to(x.device)) / CIFAR_STD.to(x.device)


def denormalize_inputs(x):
    return x * CIFAR_STD.to(x.device) + CIFAR_MEAN.to(x.device)


def make_attack(attack_cls, model, *args, **kwargs):
    attack = attack_cls(model, *args, **kwargs)
    attack.set_normalization_used(mean=CIFAR_MEAN_VALUES, std=CIFAR_STD_VALUES)
    return attack


def _quantize_tensor(t, bits):
    if bits is None:
        return t
    qmax = 2 ** (bits - 1) - 1
    scale = torch.clamp(
        t.detach().abs().max() / qmax, min=_cfg("QUANT_SCALE_MIN", 1e-8)
    )
    q = torch.round(t / scale).clamp(-qmax - 1, qmax)
    return q * scale


class _QuantConv2d(nn.Conv2d):
    def forward(self, x):
        w = _quantize_tensor(self.weight, self.bits)
        out = self._conv_forward(x, w, self.bias)
        return _quantize_tensor(out, self.bits)


class _QuantLinear(nn.Linear):
    def forward(self, x):
        w = _quantize_tensor(self.weight, self.bits)
        out = F.linear(x, w, self.bias)
        return _quantize_tensor(out, self.bits)


def _to_quant_module(module, bits):
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
        return new

    if isinstance(module, nn.Linear):
        new = _QuantLinear(
            module.in_features, module.out_features, bias=module.bias is not None
        )
        new.weight = module.weight
        if module.bias is not None:
            new.bias = module.bias
        new.bits = bits
        return new

    return None


def _replace_quant_modules(module, bits):
    for name, child in list(module.named_children()):
        replacement = _to_quant_module(child, bits)
        if replacement is None:
            _replace_quant_modules(child, bits)
        else:
            setattr(module, name, replacement)


def quantized_copy(model, bits):
    m = copy.deepcopy(model)
    _replace_quant_modules(m, bits)
    return m.to(_device())


def prepare_adversarial_training(base_model, loader, bits=None):
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
            model.eval()
            x_adv_pixel = attack(denormalize_inputs(x).clamp(0.0, 1.0), y)
            x_adv = normalize_pixels(x_adv_pixel).detach()
            model.train()

            opt.zero_grad(set_to_none=True)
            with autocast(device_type=dev.type):
                loss = F.cross_entropy(model(x_adv), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

    return model.eval()


class SanitizedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.resize = _cfg("DEFENSE_SANITIZE_SIZE", 28)
        self.bits = _cfg("DEFENSE_SANITIZE_BITS", 6)

    def forward(self, x):
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
            size=(CIFAR_IMAGE_SIZE, CIFAR_IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        levels = float(2**self.bits - 1)
        pixels = torch.round(pixels * levels).div(levels)
        return self.model(normalize_pixels(pixels.clamp(0.0, 1.0)))


class SmoothedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.sigma = _cfg("DEFENSE_SMOOTH_SIGMA", 0.12)
        self.samples = _cfg("DEFENSE_SMOOTH_SAMPLES", 8)

    def forward(self, x):
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
        super().__init__()
        self.model = model
        self.conf_threshold = _cfg("DEFENSE_GUARD_CONF_THRESHOLD", 0.55)
        self.reject_logit = _cfg("DEFENSE_REJECT_LOGIT", -1e4)

    def forward(self, x):
        logits = self.model(x)
        conf = F.softmax(logits, dim=1).max(dim=1).values
        reject = conf < self.conf_threshold
        guarded = logits.clone()
        guarded[reject] = self.reject_logit
        return guarded


class DetectorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
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
        return self.net(x)


def train_adversarial_detector(base_model, loader):
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

            x_adv_pixel = attack(denormalize_inputs(x).clamp(0.0, 1.0), y)
            x_adv = normalize_pixels(x_adv_pixel).detach()

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
        super().__init__()
        self.model = model
        self.detector = detector
        self.threshold = _cfg("DEFENSE_DETECTOR_THRESHOLD", 0.5)
        self.reject_logit = _cfg("DEFENSE_REJECT_LOGIT", -1e4)

    def forward(self, x):
        logits = self.model(x)
        detector_prob = F.softmax(self.detector(x), dim=1)[:, 1]
        reject = detector_prob > self.threshold
        guarded = logits.clone()
        guarded[reject] = self.reject_logit
        return guarded


def run_certified_accuracy(model, loader):
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
    return _run_flagging_rate(model, loader, attack, detector_attr=None)


def run_detector_catch_rate(model, loader, attack=None):
    return _run_flagging_rate(model, loader, attack, detector_attr="detector")


def _run_flagging_rate(model, loader, attack=None, detector_attr=None):
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
            x_adv_pixel = attack(denormalize_inputs(x).clamp(0.0, 1.0), y)
            x_adv = normalize_pixels(x_adv_pixel).detach()
            with torch.no_grad():
                adv_flags += _flag_mask(model, x_adv, detector_attr).sum().item()

        total += y.size(0)

    return {
        "clean_flag_rate": clean_flags / total if total else None,
        "adv_flag_rate": adv_flags / total if total and attack is not None else None,
    }


def _flag_mask(model, x, detector_attr):
    if detector_attr is not None and hasattr(model, detector_attr):
        prob = F.softmax(model.detector(x), dim=1)[:, 1]
        return prob > model.threshold

    logits = model.model(x) if hasattr(model, "model") else model(x)
    conf = F.softmax(logits, dim=1).max(dim=1).values
    return conf < getattr(model, "conf_threshold", 0.55)

#!/usr/bin/env python
# coding: utf-8
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
import pandas as pd
import copy
import os
import json
import traceback
import sys
import warnings
import time
import threading
import psutil
from torch.amp import autocast, GradScaler

from torchao.quantization.qat import IntxFakeQuantizeConfig, IntxFakeQuantizer
from pytorchcv.model_provider import get_model as ptcv_get_model

import torchattacks
from autoattack import AutoAttack

import defense as dfn
import data as report_data
import stats as qstats

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import *
from ResourceMonitor import ResourceMonitor
from attack import *

"""QuantAdv experiment runner for quantization and adversarial robustness.

This module builds FP32, post-training quantized, quantization-aware trained,
and optional defended CIFAR-10 models, then evaluates them with white-box,
transfer, adaptive, black-box, and diagnostic attacks.  Quantized layers are
fake-quantized float modules: they simulate integer rounding during forward
passes while optionally using straight-through gradients for attacks and QAT.
"""


def csv_path(model_name, type):
    """Return the per-model CSV path for an experiment artifact family."""
    return os.path.join(DATA_DIR, f"{type}_{model_name}.csv")


def json_path(model_name, type):
    """Return the per-model JSON path for an experiment artifact family."""
    return os.path.join(DATA_DIR, f"{type}_{model_name}.json")


def defense_summary_csv_path():
    """Return the aggregate defense-summary CSV path."""
    return os.path.join(DATA_DIR, "defense_summary.csv")


def check_environment():
    """Validate runtime packages and local CIFAR-10 data before a full run."""
    missing = [
        pkg
        for pkg in ("torchattacks", "autoattack", "pytorchcv", "torchao")
        if importlib.util.find_spec(pkg) is None
    ]
    if missing:
        raise ImportError(
            f"Missing packages: {missing}.\nInstall via: pip install -r requirements.txt"
        )
    print("All required packages are available.")
    print("device:", device)
    if not os.path.isdir(CIFAR10_DIR):
        raise FileNotFoundError(f"Expected extracted CIFAR-10 at {CIFAR10_DIR!r}")


def get_dataloaders(
    batch_size=DEFAULT_BATCH_SIZE, eval_n=DEFAULT_EVAL_N, finetune_n=DEFAULT_FINETUNE_N
):
    """Build CIFAR-10 fine-tuning and evaluation loaders from fixed subsets."""
    transform_train = T.Compose(
        [
            T.RandomCrop(CIFAR_IMAGE_SIZE, padding=CIFAR_RANDOM_CROP_PADDING),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=CIFAR_MEAN_VALUES, std=CIFAR_STD_VALUES),
        ]
    )
    transform_test = T.Compose(
        [T.ToTensor(), T.Normalize(mean=CIFAR_MEAN_VALUES, std=CIFAR_STD_VALUES)]
    )
    train_full = torchvision.datasets.CIFAR10(
        root=PROJECT_ROOT,
        train=True,
        download=CIFAR_DOWNLOAD,
        transform=transform_train,
    )
    test_full = torchvision.datasets.CIFAR10(
        root=PROJECT_ROOT,
        train=False,
        download=CIFAR_DOWNLOAD,
        transform=transform_test,
    )
    finetune_subset = torch.utils.data.Subset(train_full, list(range(finetune_n)))
    eval_subset = torch.utils.data.Subset(test_full, list(range(eval_n)))
    workers = min(MAX_DATA_WORKERS, os.cpu_count() or 1)
    finetune_loader = torch.utils.data.DataLoader(
        finetune_subset,
        batch_size=batch_size,
        shuffle=TRAIN_SHUFFLE,
        num_workers=workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=workers > 0,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_subset,
        batch_size=DEFAULT_EVAL_BATCH_SIZE,
        shuffle=EVAL_SHUFFLE,
        num_workers=workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=workers > 0,
    )
    return finetune_loader, eval_loader


def load_pretrained(arch_key):
    """Load a supported pretrained CIFAR-10 architecture onto the active device."""
    if arch_key != "ResNet56":
        raise ValueError(f"Unsupported architecture {arch_key!r}; expected 'ResNet56'.")
    if ptcv_get_model is None:
        raise ImportError(
            "Missing package 'pytorchcv'. Install via: pip install -r requirements.txt"
        )
    model_name = PRETRAINED_NAMES[arch_key]
    model = ptcv_get_model(model_name, pretrained=True, root=PYTORCHCV_MODEL_DIR)
    return model.to(device).eval()


def sanity_check_accuracy(model, loader):
    """Compute clean accuracy with the same evaluation path as attack metrics."""
    model.eval()
    return accuracy_from_adv_fn(model, loader, use_autocast=True)


def _torchao_int_dtype(bits):
    """Return TorchAO's integer dtype corresponding to the requested bit width."""
    if bits == 8:
        return torch.int8
    dtype = getattr(torch, f"int{bits}", None)
    if dtype is None:
        raise RuntimeError(
            f"This PyTorch build does not expose torch.int{bits}; upgrade PyTorch/TorchAO "
            f"to use {bits}-bit TorchAO fake quantization."
        )
    return dtype


def _make_torchao_fake_quantizer(bits, *, role):
    """Build TorchAO fake quantization for a weight or activation tensor.
    """
    if role == "activation":
        config = IntxFakeQuantizeConfig(
            dtype=_torchao_int_dtype(bits),
            granularity="per_token",
            is_symmetric=False,
            is_dynamic=True,
            eps=QUANT_SCALE_MIN,
        )
    elif role == "weight":
        config = IntxFakeQuantizeConfig(
            dtype=_torchao_int_dtype(bits),
            granularity="per_channel",
            is_symmetric=True,
            is_dynamic=True,
            eps=QUANT_SCALE_MIN,
        )
    else:
        raise ValueError(f"Unknown fake-quantizer role: {role!r}")
    return IntxFakeQuantizer(config)


def _hard_or_ste(fake_quantizer, tensor, use_ste):
    """Use TorchAO numerics while selecting STE or true hard-round gradients."""
    quantized = fake_quantizer(tensor)
    return quantized if use_ste else quantized.detach()

def quantize_tensor(t: torch.Tensor, bits: int, alpha: int | None = None):
    """
    Quantizes a tensor t to a given number of bits through
    scale quantization. See https://arxiv.org/pdf/2004.09602
    """
    max: int = 2 ** (bits - 1) - 1

    # Rescale to 0-1 range by default if no alpha is specified.
    if alpha is None:
        alpha = max

    scale: float = max / alpha
    return torch.clamp(torch.round(scale * t), min=-max, max=max)

def chaotic_sequence_like(t, seed=CHAOTIC_QUANT_SEED, map_name=CHAOTIC_QUANT_MAP):
    """Create a deterministic chaotic sequence with the same shape as ``t``."""
    n = t.numel()
    if n == 0:
        return torch.empty_like(t)
    dtype = torch.float32 if t.dtype in (torch.float16, torch.bfloat16) else t.dtype
    idx = torch.arange(n, device=t.device, dtype=dtype)
    z = torch.frac(seed + (idx + 1.0) * 0.6180339887498949)
    z = torch.clamp(z, 1e-6, 1.0 - 1e-6)
    if map_name == "tent":
        for _ in range(CHAOTIC_QUANT_WARMUP):
            z = torch.where(
                z < 0.5, CHAOTIC_QUANT_MU * z * 0.5, CHAOTIC_QUANT_MU * (1.0 - z) * 0.5
            )
    else:
        for _ in range(CHAOTIC_QUANT_WARMUP):
            z = CHAOTIC_QUANT_R * z * (1.0 - z)
    return z.view_as(t).to(dtype=t.dtype)


def chaotic_quantize_tensor(
    t,
    bits,
    use_ste,
    fake_quantizer,
    quantize=True,
    dither_amplitude=CHAOTIC_QUANT_DITHER,
):
    """Apply subtractive chaotic dither around a TorchAO fake quantizer."""
    if bits is None or not quantize:
        return t
    chaos = chaotic_sequence_like(t)
    # Dither is expressed in units of the tensor's dynamic range. TorchAO
    # selects the actual scale/zero-point according to the configured scheme.
    amplitude = t.detach().abs().amax().clamp_min(QUANT_SCALE_MIN)
    dither = (chaos - 0.5) * dither_amplitude * amplitude
    quantized = fake_quantizer(t + dither) - dither
    return quantized if use_ste else quantized.detach()


class _QuantizedLayerMixin:
    """Shared TorchAO fake-quantization controls for Conv2d/Linear wrappers."""

    chaotic = False

    def _init_quantizers(self, bits):
        self.bits = bits
        self.weight_fake_quantizer = _make_torchao_fake_quantizer(bits, role="weight")
        self.activation_fake_quantizer = _make_torchao_fake_quantizer(bits, role="activation")

    def _quant_params(self):
        return (
            getattr(self, "bits", None),
            getattr(self, "use_ste", QUANT_DEFAULT_USE_STE),
            getattr(self, "quant_weight", QUANT_DEFAULT_WEIGHT),
            getattr(self, "quant_act", QUANT_DEFAULT_ACT),
        )

    def _quantize_weight(self, tensor, enabled):
        if not enabled:
            return tensor
        original_shape = tensor.shape
        flattened = tensor.reshape(tensor.shape[0], -1)
        if self.chaotic:
            quantized = chaotic_quantize_tensor(
                flattened,
                self.bits,
                self.use_ste,
                self.weight_fake_quantizer,
                True,
                getattr(self, "dither_amplitude", CHAOTIC_QUANT_DITHER),
            )
        else:
            quantized = _hard_or_ste(
                self.weight_fake_quantizer, flattened, self.use_ste
            )
        return quantized.reshape(original_shape)

    def _quantize_activation(self, tensor, enabled):
        if not enabled:
            return tensor
        if self.chaotic:
            return chaotic_quantize_tensor(
                tensor,
                self.bits,
                self.use_ste,
                self.activation_fake_quantizer,
                True,
                getattr(self, "dither_amplitude", CHAOTIC_QUANT_DITHER),
            )
        return _hard_or_ste(self.activation_fake_quantizer, tensor, self.use_ste)


class QuantConv2d(_QuantizedLayerMixin, nn.Conv2d):
    """Conv2d using TorchAO fake quantization for weights and output activations."""

    def forward(self, x):
        _, _, quant_weight, quant_act = self._quant_params()
        weight = self._quantize_weight(self.weight, quant_weight)
        out = self._conv_forward(x, weight, self.bias)
        return self._quantize_activation(out, quant_act)


class ChaoticQuantConv2d(QuantConv2d):
    chaotic = True


class QuantLinear(_QuantizedLayerMixin, nn.Linear):
    """Linear using TorchAO fake quantization for weights and output activations."""

    def forward(self, x):
        _, _, quant_weight, quant_act = self._quant_params()
        weight = self._quantize_weight(self.weight, quant_weight)
        out = F.linear(x, weight, self.bias)
        return self._quantize_activation(out, quant_act)


class ChaoticQuantLinear(QuantLinear):
    chaotic = True


def _to_quant_module(
    mod,
    bits,
    quant_weight=True,
    quant_act=True,
    chaotic=False,
    dither_amplitude=CHAOTIC_QUANT_DITHER,
):
    """Convert Conv2d/Linear to TorchAO-backed fake-quantized wrappers."""
    if isinstance(mod, nn.Conv2d):
        cls = ChaoticQuantConv2d if chaotic else QuantConv2d
        new = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            mod.stride,
            mod.padding,
            mod.dilation,
            mod.groups,
            mod.bias is not None,
            mod.padding_mode,
            device=mod.weight.device,
            dtype=mod.weight.dtype,
        )
    elif isinstance(mod, nn.Linear):
        cls = ChaoticQuantLinear if chaotic else QuantLinear
        new = cls(
            mod.in_features,
            mod.out_features,
            bias=mod.bias is not None,
            device=mod.weight.device,
            dtype=mod.weight.dtype,
        )
    else:
        return None

    new.weight = mod.weight
    if mod.bias is not None:
        new.bias = mod.bias
    new._init_quantizers(bits)
    new.use_ste = QUANT_DEFAULT_USE_STE
    new.quant_weight = quant_weight
    new.quant_act = quant_act
    new.dither_amplitude = dither_amplitude
    return new


def _replace_recursive(
    module,
    bits,
    quant_weight=True,
    quant_act=True,
    chaotic=False,
    dither_amplitude=CHAOTIC_QUANT_DITHER,
):
    for name, child in list(module.named_children()):
        replacement = _to_quant_module(
            child,
            bits,
            quant_weight,
            quant_act,
            chaotic=chaotic,
            dither_amplitude=dither_amplitude,
        )
        if replacement is not None:
            setattr(module, name, replacement)
        else:
            _replace_recursive(
                child,
                bits,
                quant_weight,
                quant_act,
                chaotic=chaotic,
                dither_amplitude=dither_amplitude,
            )


def convert_to_quant(
    model,
    bits,
    quant_weight=True,
    quant_act=True,
    chaotic=False,
    dither_amplitude=CHAOTIC_QUANT_DITHER,
):
    """Return a deep-copied model backed by TorchAO fake quantizers."""
    m = copy.deepcopy(model)
    _replace_recursive(
        m,
        bits,
        quant_weight,
        quant_act,
        chaotic=chaotic,
        dither_amplitude=dither_amplitude,
    )
    return m


def convert_to_chaotic_quant(
    model,
    bits,
    quant_weight=True,
    quant_act=True,
    dither_amplitude=CHAOTIC_QUANT_DITHER,
):
    return convert_to_quant(
        model,
        bits,
        quant_weight=quant_weight,
        quant_act=quant_act,
        chaotic=True,
        dither_amplitude=dither_amplitude,
    )


def quantizable_layer_names(model):
    quant_types = (QuantConv2d, QuantLinear, ChaoticQuantConv2d, ChaoticQuantLinear)
    return [
        name
        for name, mod in model.named_modules()
        if isinstance(mod, (nn.Conv2d, nn.Linear)) and not isinstance(mod, quant_types)
    ]


def set_child_module(root, module_name, new_module):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent._modules[part]
    parent._modules[parts[-1]] = new_module


def convert_layer_chunk_to_quant(
    model, layer_names, bits, quant_weight=True, quant_act=True, chaotic=False
):
    m = copy.deepcopy(model)
    targets = set(layer_names)
    for name, mod in list(m.named_modules()):
        if name not in targets:
            continue
        new_mod = _to_quant_module(mod, bits, quant_weight, quant_act, chaotic=chaotic)
        if new_mod is not None:
            set_child_module(m, name, new_mod)
    return m.to(device).eval()


def quant_layer_chunks(layer_names, n_chunks):
    if not layer_names:
        return []
    n_chunks = max(1, min(n_chunks, len(layer_names)))
    return [
        list(chunk)
        for chunk in np.array_split(np.array(layer_names, dtype=object), n_chunks)
        if len(chunk) > 0
    ]


def count_quant_layers(model):
    return sum(
        isinstance(mod, (QuantConv2d, QuantLinear, ChaoticQuantConv2d, ChaoticQuantLinear))
        for mod in model.modules()
    )


def verify_quantization_layers(
    arch_key, fp32_model, quant_model, label, fp32_layer_names=None
):
    fp32_layer_names = (
        quantizable_layer_names(fp32_model)
        if fp32_layer_names is None
        else fp32_layer_names
    )
    if not fp32_layer_names:
        raise RuntimeError(f"{arch_key} exposes zero FP32 nn.Conv2d/nn.Linear layers.")
    quant_count = count_quant_layers(quant_model)
    threshold = int(np.ceil(0.8 * len(fp32_layer_names)))
    print(f"  {label} quantized layers: {quant_count}", flush=True)
    if quant_count < threshold:
        raise RuntimeError(
            f"{arch_key} {label} replaced {quant_count}/{len(fp32_layer_names)} "
            f"quantizable layers; expected at least {threshold}."
        )
    return quant_count


def set_quant_components(model, quant_weight, quant_act):
    for mod in model.modules():
        if isinstance(mod, (QuantConv2d, QuantLinear, ChaoticQuantConv2d, ChaoticQuantLinear)):
            mod.quant_weight = quant_weight
            mod.quant_act = quant_act


def prepare_qat(
    fp32_model,
    bits,
    finetune_loader,
    epochs=QAT_EPOCHS_DEFAULT,
    lr=QAT_LR,
    chaotic=False,
):
    """Fine-tune a TorchAO fake-quantized copy with STE enabled."""
    m = (
        convert_to_chaotic_quant(fp32_model, bits, quant_weight=True, quant_act=True)
        if chaotic
        else convert_to_quant(fp32_model, bits, quant_weight=True, quant_act=True)
    )
    if torch.cuda.device_count() > 1:
        m = nn.DataParallel(m)
    set_ste_mode(m, True)
    m.train()
    opt = torch.optim.SGD(
        m.parameters(),
        lr=lr,
        momentum=QAT_MOMENTUM,
        weight_decay=QAT_WEIGHT_DECAY,
    )
    scaler = GradScaler(device=device.type, enabled=device.type == "cuda")
    for epoch in range(epochs):
        running = 0.0
        for batch_idx, (x, y) in enumerate(finetune_loader):
            x = x.to(device, non_blocking=NON_BLOCKING_TRANSFER)
            y = y.to(device, non_blocking=NON_BLOCKING_TRANSFER)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = F.cross_entropy(m(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            if batch_idx == 0 or (batch_idx + 1) % QAT_LOG_EVERY_BATCHES == 0:
                print(
                    f"  QAT epoch {epoch + 1}/{epochs} batch "
                    f"{batch_idx + 1}/{len(finetune_loader)} loss {loss.item():.4f}",
                    flush=True,
                )
        print(
            f"  QAT epoch {epoch + 1}/{epochs} avg loss "
            f"{running / len(finetune_loader):.4f}",
            flush=True,
        )
    set_ste_mode(m, False)
    return m.eval()


class ImageCompressionSTE(torch.autograd.Function):
    """Quantize pixel values with identity gradients for compression defenses."""

    @staticmethod
    def forward(ctx, x, bits):
        """Apply image compression in the forward pass for input preprocessing."""
        levels = float(2**bits - 1)
        return torch.round(x * levels).div(levels)

    @staticmethod
    def backward(ctx, grad_output):
        """Pass gradients through image compression unchanged."""
        return grad_output, None


class CompressedInputModel(nn.Module):
    """Wrap a classifier with resize and pixel-bit-depth input compression."""

    def __init__(
        self,
        model,
        size=COMPRESS_IMAGE_SIZE,
        bits=COMPRESS_IMAGE_BITS,
        mode=COMPRESS_IMAGE_MODE,
    ):
        """Wrap a model with differentiable image-compression preprocessing."""
        super().__init__()
        self.model = model
        self.size = size
        self.bits = bits
        self.mode = mode

    def forward(self, x):
        """Compress inputs before forwarding them to the wrapped model."""
        pixels = denormalize_inputs(x).clamp(0.0, 1.0)
        if self.size and self.size < pixels.shape[-1]:
            kwargs = {"mode": self.mode}
            if self.mode in ("linear", "bilinear", "bicubic", "trilinear"):
                kwargs["align_corners"] = COMPRESS_IMAGE_ALIGN_CORNERS
            pixels = F.interpolate(pixels, size=(self.size, self.size), **kwargs)
            pixels = F.interpolate(
                pixels, size=(CIFAR_IMAGE_SIZE, CIFAR_IMAGE_SIZE), **kwargs
            )
        if self.bits is not None:
            pixels = ImageCompressionSTE.apply(pixels, self.bits)
        return self.model(normalize_pixels(pixels.clamp(0.0, 1.0)))


def with_image_compression(
    model, size=COMPRESS_IMAGE_SIZE, bits=COMPRESS_IMAGE_BITS, mode=COMPRESS_IMAGE_MODE
):
    """Return an eval-mode copy of ``model`` behind input compression."""
    return (
        CompressedInputModel(copy.deepcopy(model), size=size, bits=bits, mode=mode)
        .to(device)
        .eval()
    )


def add_paired_fp32_mcnemar_tests(df_results):
    """Post-hoc paired tests between each variant and its architecture's FP32 model."""
    return qstats.add_paired_mcnemar_tests(
        df_results, baseline_name=qstats.fp32_baseline_name
    )


def gradient_diagnostics(
    model, loader, fp32_ref=None, max_batches=GRAD_DIAG_MAX_BATCHES
):
    """Ground-truth masking check: compare hard-round vs. STE input gradients.
    """
    frac_zero_hard, norm_hard = [], []
    frac_zero_ste, norm_ste = [], []
    cos_sims = []
    for bi, (x, y) in enumerate(loader):
        if bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with ste_mode(model, False):
            x_in = x.clone().requires_grad_(True)
            loss = F.cross_entropy(model(x_in), y)
            g_hard = torch.autograd.grad(loss, x_in)[0].flatten()
        frac_zero_hard.append(
            (g_hard.abs() < GRAD_ZERO_THRESHOLD).float().mean().item()
        )
        norm_hard.append(g_hard.norm().item())
        with ste_mode(model, True):
            x_in2 = x.clone().requires_grad_(True)
            loss2 = F.cross_entropy(model(x_in2), y)
            g_ste = torch.autograd.grad(loss2, x_in2)[0].flatten()
        frac_zero_ste.append((g_ste.abs() < GRAD_ZERO_THRESHOLD).float().mean().item())
        norm_ste.append(g_ste.norm().item())
        if fp32_ref is not None:
            fp32_ref.eval()
            x_ref = x.clone().requires_grad_(True)
            loss_ref = F.cross_entropy(fp32_ref(x_ref), y)
            g_ref = torch.autograd.grad(loss_ref, x_ref)[0].flatten()
            cos_sims.append(
                F.cosine_similarity(g_ste.unsqueeze(0), g_ref.unsqueeze(0)).item()
            )
    diagnostics = {
        "frac_zero_grad_hard": float(np.mean(frac_zero_hard)),
        "frac_zero_grad_ste": float(np.mean(frac_zero_ste)),
        "grad_norm_hard": float(np.mean(norm_hard)),
        "grad_norm_ste": float(np.mean(norm_ste)),
    }
    if cos_sims:
        diagnostics["grad_cosine_sim_with_FP32"] = float(np.mean(cos_sims))
    return diagnostics


def layerwise_grad_profile(model, loader, use_ste, max_batches=LAYERWISE_MAX_BATCHES):
    """Collect per-layer gradient statistics for a model and loader."""
    quant_layers = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, (QuantConv2d, QuantLinear))
    ]
    norms = {n: [] for n, _ in quant_layers}
    handles = []

    def make_hook(name):
        """Create a backward hook that records gradient statistics for one layer."""
        def hook(module, grad_input, grad_output):
            """Record gradient statistics emitted by a backward hook."""
            gi = grad_input[0]
            if gi is not None:
                norms[name].append(gi.flatten(1).norm(dim=1).mean().item())

        return hook

    try:
        for n, m in quant_layers:
            handles.append(m.register_full_backward_hook(make_hook(n)))
        model.eval()
        with ste_mode(model, use_ste):
            for bi, (x, y) in enumerate(loader):
                if bi >= max_batches:
                    break
                x, y = x.to(device), y.to(device)
                x = x.clone().requires_grad_(True)
                loss = F.cross_entropy(model(x), y)
                model.zero_grad(set_to_none=True)
                loss.backward()
    finally:
        for h in handles:
            h.remove()
    ordered_names = [n for n, _ in quant_layers]
    return {
        n: (float(np.mean(norms[n])) if len(norms[n]) else None) for n in ordered_names
    }


def run_quant_component_ablation(model, loader, name, eps=DEFAULT_EPS):
    """Evaluate which quantized components are responsible for observed effects.

    The rows distinguish three interpretations of "testing quantized":
    weight-only quantization, activation-only quantization, and both together.
    For each interpretation the experiment reports ordinary hard-round PGD and
    a budget-matched STE/BPDA PGD.  A large hard-PGD vs STE-PGD gap is evidence
    of gradient masking, not evidence that quantization itself improves
    robustness.

    Weight quantization alone cannot produce this gap: rounding a weight
    doesn't touch the gradient path back to the input (the conv/linear op is
    still linear in ``x`` for whatever rounded weight value it has), so
    ``weight_only`` should show ``frac_zero_grad_hard`` near the background
    rate and a small hard-vs-STE gap. Only ``act_only``/``both`` quantize a
    tensor that ``x`` actually flows through, so only those can mask.
    """
    configs = [
        ("weight_only", True, False),
        ("act_only", False, True),
        ("both", True, True),
    ]
    rows = []
    for label, qw, qa in configs:
        set_quant_components(model, qw, qa)
        clean_acc = sanity_check_accuracy(model, loader)

        with ste_mode(model, False):
            torch.manual_seed(0)
            pgd_hard = make_torchattack(
                torchattacks.PGD,
                model,
                eps=eps,
                alpha=PGD_ALPHA,
                steps=PGD_STEPS,
                random_start=PGD_RANDOM_START,
            )
            pgd_hard_acc = accuracy_under_attack(model, loader, pgd_hard)

            x, y = next(iter(loader))
            x, y = x.to(device), y.to(device)
            x_in = x.clone().requires_grad_(True)
            loss = F.cross_entropy(model(x_in), y)
            g_hard = torch.autograd.grad(loss, x_in)[0].flatten()
            frac_zero = (g_hard.abs() < GRAD_ZERO_THRESHOLD).float().mean().item()

        with ste_mode(model, True):
            torch.manual_seed(0)
            pgd_ste = make_torchattack(
                torchattacks.PGD,
                model,
                eps=eps,
                alpha=PGD_ALPHA,
                steps=PGD_STEPS,
                random_start=PGD_RANDOM_START,
            )
            pgd_ste_acc = accuracy_under_attack(model, loader, pgd_ste)

        bpda = run_bpda(
            model,
            loader,
            eps=eps,
            n_restarts=1,
            seeds=SEEDS[:1],
        )
        rows.append(
            {
                "model": name,
                "config": label,
                "quant_weight": qw,
                "quant_act": qa,
                "clean_acc": clean_acc,
                "PGD_acc": pgd_hard_acc,
                "PGD_hard_acc": pgd_hard_acc,
                "PGD_ste_acc": pgd_ste_acc,
                "BPDA_acc": bpda["BPDA_PGD"],
                "BPDA_mean": bpda["BPDA_PGD_mean"],
                "BPDA_std": bpda["BPDA_PGD_std"],
                "PGD_minus_STE": (
                    pgd_hard_acc - pgd_ste_acc
                    if pgd_hard_acc is not None and pgd_ste_acc is not None
                    else None
                ),
                "PGD_minus_BPDA": (
                    pgd_hard_acc - bpda["BPDA_PGD"]
                    if pgd_hard_acc is not None and bpda["BPDA_PGD"] is not None
                    else None
                ),
                "frac_zero_grad_hard": frac_zero,
            }
        )
    # restore original (both quantized) state
    set_quant_components(model, True, True)
    return rows


def run_chunk_quantization_attacks(
    fp32_model,
    loader,
    name,
    bits=8,
    n_chunks=CHUNK_QUANT_NUM_CHUNKS,
    eps=DEFAULT_EPS,
):
    """Evaluate attacks as contiguous chunks of layers are quantized."""
    layer_names = quantizable_layer_names(fp32_model)
    chunks = quant_layer_chunks(layer_names, n_chunks)
    rows = []
    for i, chunk in enumerate(chunks, start=1):
        chunk_model = convert_layer_chunk_to_quant(
            fp32_model, chunk, bits=bits, quant_weight=True, quant_act=True
        )
        row = {
            "model": name,
            "bits": bits,
            "chunk_id": i,
            "chunk_count": len(chunks),
            "chunk_label": f"{i}/{len(chunks)}",
            "chunk_size": len(chunk),
            "first_layer": chunk[0],
            "last_layer": chunk[-1],
            "layers": json.dumps(chunk),
        }
        try:
            row["clean_acc"] = sanity_check_accuracy(chunk_model, loader)
        except Exception as e:
            print(
                f"  [WARN] chunk clean_acc failed for {name} {row['chunk_label']}: {e}"
            )
            row["clean_acc"] = None
        try:
            pgd = make_torchattack(
                torchattacks.PGD,
                chunk_model,
                eps=eps,
                alpha=PGD_ALPHA,
                steps=PGD_STEPS,
                random_start=PGD_RANDOM_START,
            )
            row["PGD_acc"] = accuracy_under_attack(chunk_model, loader, pgd)
        except Exception as e:
            print(f"  [WARN] chunk PGD failed for {name} {row['chunk_label']}: {e}")
            row["PGD_acc"] = None
        rows.append(row)
        del chunk_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def safe_call(
    fn, warning, *, context=None, default=None, show_traceback=False, level="WARN"
):
    """Run an optional experiment stage with consistent warning and fallback handling."""
    try:
        return fn()
    except Exception as exc:
        suffix = f" for {context}" if context else ""
        print(f"  [{level}] {warning}{suffix}: {exc}")
        if show_traceback:
            traceback.print_exc()
        return default() if callable(default) else default


def safe_set(target, key, fn, warning, *, context=None, default=None):
    """Run a metric function and store its value with warning-based fallback."""
    target[key] = safe_call(fn, warning, context=context, default=default)
    return target[key]


def safe_update(target, fn, warning, *, context=None, defaults=None):
    """Merge a metric dictionary into a target dictionary with warning fallback."""
    update = safe_call(fn, warning, context=context, default=None)
    if update is not None:
        target.update(update)
        return True
    if defaults:
        target.update(
            {
                key: value() if callable(value) else value
                for key, value in defaults.items()
            }
        )
    return False


def safe_set_vector(results, vectors, metric, fn, warning, *, context=None):
    """Run a vector-producing metric and store values plus correctness vectors."""
    pair = safe_call(fn, warning, context=context, default=None)
    if pair is None:
        results[metric] = None
        return False
    results[metric], vectors[metric] = pair
    return True


def safe_update_vectors(results, vectors, fn, warning, *, context=None, defaults=None):
    """Merge metric and vector outputs into their destination dictionaries."""
    update = safe_call(fn, warning, context=context, default=None)
    if update is None:
        if defaults:
            results.update(defaults)
        return False
    vectors.update(update.pop("_vectors", {}))
    results.update(update)
    return True


def save_json(path, data, *, indent=None):
    """Serialize a Python object as pretty-printed JSON."""
    with open(path, "w") as handle:
        json.dump(data, handle, indent=indent)


def run_suite(model, loader, name, fp32_ref=None, eps=DEFAULT_EPS):
    """Run the full attack and diagnostic suite for one model."""
    model.eval()
    results = {"model": name}
    vectors = {}
    safe_set_vector(
        results,
        vectors,
        "clean_acc",
        lambda: accuracy_from_adv_fn(
            model, loader, use_autocast=True, return_vector=True
        ),
        "clean_acc failed",
        context=name,
    )
    safe_update_vectors(
        results,
        vectors,
        lambda: run_fgsm_pgd(
            model, loader, eps=eps, return_vectors=True, use_ste=False
        ),
        "FGSM/PGD failed",
        context=name,
        defaults={"FGSM": None, "PGD": None},
    )
    safe_set_vector(
        results,
        vectors,
        "AutoAttack",
        lambda: run_autoattack(
            model, loader, eps=eps, return_vector=True, use_ste=False
        ),
        "AutoAttack failed",
        context=name,
    )
    if RUN_EXTRA_WHITEBOX_ATTACKS:
        safe_update(
            results,
            lambda: run_extra_whitebox_attacks(
                model, loader, eps=eps, use_ste=False
            ),
            "CW/DeepFool/JSMA failed",
            context=name,
        )
    if RUN_UAP_ATTACKS:
        safe_set(
            results,
            "UAP",
            lambda: run_uap_attack(model, loader, eps=eps),
            "UAP attack failed",
            context=name,
        )
    if RUN_SURROGATE_ATTACK:
        safe_set(
            results,
            "Surrogate_Transfer",
            lambda: run_surrogate_attack(model, loader, eps=eps),
            "surrogate attack failed",
            context=name,
        )
    if fp32_ref is not None:
        safe_set_vector(
            results,
            vectors,
            "Transfer_from_FP32",
            lambda: transfer_attack(
                fp32_ref,
                model,
                loader,
                eps=eps,
                return_vector=True,
                use_ste=False,
            ),
            "transfer_attack failed",
            context=name,
        )
        safe_set_vector(
            results,
            vectors,
            "MIM_Transfer",
            lambda: transfer_attack_mim(
                fp32_ref,
                model,
                loader,
                eps=eps,
                return_vector=True,
                use_ste=False,
            ),
            "MIM transfer_attack failed",
            context=name,
        )
        if RUN_UAP_ATTACKS:
            safe_set(
                results,
                "UAP_Transfer",
                lambda: transfer_uap_attack(fp32_ref, model, loader, eps=eps),
                "UAP transfer_attack failed",
                context=name,
            )
        if RUN_REVERSE_TRANSFERS and count_quant_layers(model) > 0:
            safe_set(
                results,
                "Transfer_to_FP32",
                lambda: transfer_attack(
                    model, fp32_ref, loader, eps=eps, use_ste=False
                ),
                "reverse transfer_attack failed",
                context=name,
            )
            safe_set(
                results,
                "MIM_Transfer_to_FP32",
                lambda: transfer_attack_mim(
                    model, fp32_ref, loader, eps=eps, use_ste=False
                ),
                "reverse MIM transfer_attack failed",
                context=name,
            )
            safe_set(
                results,
                "UAP_Transfer_to_FP32",
                lambda: transfer_uap_attack(model, fp32_ref, loader, eps=eps),
                "reverse UAP transfer_attack failed",
                context=name,
            )
    safe_update_vectors(
        results,
        vectors,
        lambda: run_random_noise_seeded(model, loader, eps=eps, return_vector=True),
        "random_noise_attack failed",
        context=name,
        defaults={"Random_Noise": None},
    )
    defense_model = unwrap_model(model)
    adaptive_defaults = {}
    if isinstance(defense_model, dfn.SanitizedModel):
        adaptive_defaults["BPDA_Adaptive"] = None
    elif isinstance(defense_model, dfn.SmoothedModel):
        adaptive_defaults["EOT_PGD"] = None
    elif isinstance(defense_model, dfn.GuardrailModel):
        adaptive_defaults["Adaptive_Guardrail"] = None
    elif isinstance(defense_model, dfn.DetectGuardModel):
        adaptive_defaults["Adaptive_DetectGuard"] = None
    safe_update(
        results,
        lambda: run_defense_adaptive_attacks(model, loader, eps=eps),
        "adaptive defense attack failed",
        context=name,
        defaults=adaptive_defaults,
    )
    if count_quant_layers(model) > 0:
        safe_update_vectors(
            results,
            vectors,
            lambda: run_bpda(
                model,
                loader,
                eps=eps,
                n_restarts=1,
                return_vector=True,
            ),
            "BPDA failed",
            context=name,
            defaults={"BPDA_PGD": None},
        )
        safe_update(
            results,
            lambda: gradient_diagnostics(
                model, loader, fp32_ref=fp32_ref, max_batches=GRAD_DIAG_MAX_BATCHES
            ),
            "gradient_diagnostics failed",
            context=name,
        )
        safe_update(
            results,
            lambda: staircase_diagnostic(model, loader),
            "staircase_diagnostic failed",
            context=name,
        )
        boundary_defaults = {
            "Boundary_acc": None,
            "Boundary_mean_Linf": None,
            "Boundary_median_Linf": None,
            "Boundary_min_Linf": None,
            "Boundary_max_Linf": None,
            "Boundary_std_Linf": None,
            "Boundary_n": 0,
            "Boundary_init_failed": None,
            "Boundary_init_failed_rate": None,
        }
        if RUN_BOUNDARY_ATTACK:
            safe_update(
                results,
                lambda: run_boundary_attack(
                    model,
                    loader,
                    eps=eps,
                    max_images=BOUNDARY_MAX_IMAGES_SUITE,
                    steps=BOUNDARY_STEPS_SUITE,
                    seed=BOUNDARY_SEED,
                ),
                "boundary_attack failed",
                context=name,
                defaults=boundary_defaults,
            )
        else:
            results.update(boundary_defaults)
        if RUN_NES_ATTACK:
            safe_update(
                results,
                lambda: run_nes_attack(
                    model,
                    loader,
                    eps=eps,
                    seeds=SEEDS,
                    n_samples=NES_SAMPLES_SUITE,
                    query_chunk=NES_QUERY_CHUNK,
                ),
                "NES attack failed",
                context=name,
                defaults={"NES": None},
            )

        def save_pgd_ablation():
            """Persist PGD step-ablation diagnostics for the current model."""
            ablation = pgd_steps_ablation(model, loader, eps=eps)
            pd.DataFrame(
                [{"model": name, "steps": k, "acc": v} for k, v in ablation.items()]
            ).to_csv(csv_path(name, "ablation"), index=False)

        safe_call(save_pgd_ablation, "pgd_steps_ablation failed", context=name)
        if RUN_PGD_TRAJECTORY:

            def save_trajectory():
                """Persist PGD trajectory diagnostics for the current model."""
                traj = pgd_trajectory_diagnostics(
                    model, loader, eps=eps, max_batches=TRAJECTORY_MAX_BATCHES
                )
                save_json(json_path(name, "trajectory"), traj, indent=2)

            safe_call(
                save_trajectory, "pgd_trajectory_diagnostics failed", context=name
            )
        if RUN_LAYERWISE_PROFILE:

            def save_layerwise_profile():
                """Persist layerwise gradient diagnostics for the current model."""
                prof_hard = layerwise_grad_profile(model, loader, use_ste=False)
                prof_ste = layerwise_grad_profile(model, loader, use_ste=True)
                rows = [
                    {
                        "model": name,
                        "layer": n,
                        "grad_norm_hard": prof_hard.get(n),
                        "grad_norm_ste": prof_ste.get(n),
                    }
                    for n in prof_hard
                ]
                pd.DataFrame(rows).to_csv(csv_path(name, "layerwise"), index=False)

            safe_call(
                save_layerwise_profile, "layerwise_grad_profile failed", context=name
            )
        # weight-only / activation-only / both ablation
        if RUN_COMPONENT_ABLATION:

            def save_component_ablation():
                """Persist quantization component-ablation diagnostics for the current model."""
                rows = run_quant_component_ablation(model, loader, name, eps=eps)
                pd.DataFrame(rows).to_csv(
                    csv_path(name, "component_ablation"), index=False
                )

            safe_call(
                save_component_ablation,
                "run_quant_component_ablation failed",
                context=name,
            )
        if RUN_CONFIDENCE_MARGIN:

            def save_confidence_margins():
                """Persist confidence-margin diagnostics for the current model."""
                margins = confidence_margin_diagnostic(
                    model, loader, eps=eps, max_batches=MARGIN_MAX_BATCHES
                )
                save_json(json_path(name, "margin"), margins)

            safe_call(
                save_confidence_margins,
                "confidence_margin_diagnostic failed",
                context=name,
            )
    for metric, vector in vectors.items():
        qstats.add_binomial_statistics(
            results, metric, vector, confidence=CI_CONFIDENCE
        )
    if "PGD" in vectors and "BPDA_PGD" in vectors:
        test = qstats.mcnemar_exact(vectors["PGD"], vectors["BPDA_PGD"])
        results.update(
            {f"McNemar_PGD_vs_BPDA_{key}": value for key, value in test.items()}
        )
    if vectors:
        results["correctness_vectors_path"] = qstats.save_correctness_vectors(
            name, vectors, PER_EXAMPLE_DIR
        )
    return results


def run_epsilon_sweep_for_model_wrapped(model, loader, name, epsilons):
    """Run and annotate an epsilon sweep for one named model."""
    return run_epsilon_sweep_for_model(
        model,
        loader,
        name,
        epsilons,
        count_quant_layers_fn=count_quant_layers,
        safe_set=safe_set,
    )


def run_defense_suite(model_registry, finetune_loader, eval_loader):
    """Build defended models and evaluate them under adaptive attacks."""
    summary_rows = []
    arch_keys = sorted(
        {name.split("_FP32")[0] for name in model_registry if name.endswith("_FP32")}
    )
    for arch_key in arch_keys:
        fp32_entry = model_registry.get(f"{arch_key}_FP32")
        int8_qat_entry = model_registry.get(f"{arch_key}_int8_QAT")
        int4_qat_entry = model_registry.get(f"{arch_key}_int4_QAT")
        if fp32_entry is None:
            continue
        fp32_model = fp32_entry[0]

        def add_fp32_at():
            """Add the adversarially trained FP32 defense to the registry when available."""
            fp32_at = dfn.prepare_adversarial_training(
                fp32_model, finetune_loader, bits=None
            )
            model_registry[f"{arch_key}_FP32_AT"] = (fp32_at, fp32_model)

        safe_call(
            add_fp32_at,
            "adversarial training (FP32) failed",
            context=arch_key,
            show_traceback=True,
            level="FAIL",
        )

        def add_int8_at():
            """Add the adversarially trained INT8 defense to the registry when available."""
            int8_at = dfn.prepare_adversarial_training(
                fp32_model, finetune_loader, bits=8
            )
            model_registry[f"{arch_key}_int8_QAT_AT"] = (int8_at, fp32_model)

        safe_call(
            add_int8_at,
            "adversarial training (int8) failed",
            context=arch_key,
            show_traceback=True,
            level="FAIL",
        )
        wrap_targets = [("FP32", fp32_model)]
        if int8_qat_entry is not None:
            wrap_targets.append(("int8_QAT", int8_qat_entry[0]))
        if int4_qat_entry is not None:
            wrap_targets.append(("int4_QAT", int4_qat_entry[0]))
        detector = safe_call(
            lambda: dfn.train_adversarial_detector(fp32_model, finetune_loader),
            "adversarial detector training failed",
            context=arch_key,
            show_traceback=True,
            level="FAIL",
        )
        for tag, base_model in wrap_targets:
            entry_name = f"{arch_key}_{tag}"

            def add_sanitized():
                """Add the input-sanitization defense to the registry."""
                sanitized = dfn.SanitizedModel(base_model).to(device).eval()
                model_registry[f"{entry_name}_Sanitized"] = (sanitized, fp32_model)

            safe_call(
                add_sanitized, "SanitizedModel failed", context=entry_name, level="FAIL"
            )

            def add_smoothed():
                """Add the randomized-smoothing defense to the registry."""
                smoothed = dfn.SmoothedModel(base_model).to(device).eval()
                model_registry[f"{entry_name}_Smoothed"] = (smoothed, fp32_model)
                cert_stats = dfn.run_certified_accuracy(smoothed, eval_loader)
                summary_rows.append(
                    {
                        "model": entry_name,
                        "defense": "randomized_smoothing",
                        **cert_stats,
                    }
                )

            safe_call(
                add_smoothed,
                "SmoothedModel/certification failed",
                context=entry_name,
                show_traceback=True,
                level="FAIL",
            )

            def add_guardrail():
                """Add the confidence guardrail defense to the registry."""
                guardrail = dfn.GuardrailModel(base_model).to(device).eval()
                model_registry[f"{entry_name}_Guardrail"] = (guardrail, fp32_model)
                pgd_for_flagging = make_torchattack(
                    torchattacks.PGD,
                    guardrail,
                    eps=DEFAULT_EPS,
                    alpha=PGD_ALPHA,
                    steps=PGD_STEPS,
                    random_start=PGD_RANDOM_START,
                )
                flag_stats = dfn.run_guardrail_flagging_rate(
                    guardrail, eval_loader, attack=pgd_for_flagging
                )
                summary_rows.append(
                    {"model": entry_name, "defense": "guardrail", **flag_stats}
                )

            safe_call(
                add_guardrail,
                "GuardrailModel failed",
                context=entry_name,
                show_traceback=True,
                level="FAIL",
            )
            if detector is not None:

                def add_detect_guard():
                    """Add the detector-based guardrail defense to the registry."""
                    detect_guard = (
                        dfn.DetectGuardModel(base_model, detector).to(device).eval()
                    )
                    model_registry[f"{entry_name}_DetectGuard"] = (
                        detect_guard,
                        fp32_model,
                    )
                    pgd_for_detect = make_torchattack(
                        torchattacks.PGD,
                        detect_guard,
                        eps=DEFAULT_EPS,
                        alpha=PGD_ALPHA,
                        steps=PGD_STEPS,
                        random_start=PGD_RANDOM_START,
                    )
                    catch_stats = dfn.run_detector_catch_rate(
                        detect_guard, eval_loader, attack=pgd_for_detect
                    )
                    summary_rows.append(
                        {"model": entry_name, "defense": "detector", **catch_stats}
                    )

                safe_call(
                    add_detect_guard,
                    "DetectGuardModel failed",
                    context=entry_name,
                    show_traceback=True,
                    level="FAIL",
                )
    df_defense = pd.DataFrame(summary_rows)
    if not df_defense.empty:
        df_defense.to_csv(defense_summary_csv_path(), index=False)
    return model_registry, df_defense


def _palette_for(values):
    """Choose a plotting palette sized to the provided values."""
    return {v: ATTACK_PALETTE[v] for v in values if v in ATTACK_PALETTE}


def parallelize(model):
    """Wrap a model in DataParallel when multiple CUDA devices are available."""
    if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        return nn.DataParallel(model)
    return model


def run_chaotic_dither_sweep(fp32_model, loader, arch_key, bits=8):
    """Evaluate chaotic quantization over dither-amplitude settings."""
    rows = []
    for amplitude in CHAOTIC_DITHER_AMPLITUDES:
        model = (
            convert_to_chaotic_quant(
                fp32_model,
                bits=bits,
                quant_weight=True,
                quant_act=True,
                dither_amplitude=amplitude,
            )
            .to(device)
            .eval()
        )
        clean = sanity_check_accuracy(model, loader)
        attack = make_torchattack(
            torchattacks.PGD,
            model,
            eps=DEFAULT_EPS,
            alpha=PGD_ALPHA,
            steps=PGD_STEPS,
            random_start=PGD_RANDOM_START,
        )
        torch.manual_seed(SEEDS[0])
        pgd = accuracy_under_attack(model, loader, attack)
        rows.append(
            {
                "model": arch_key,
                "bits": bits,
                "dither_amplitude": amplitude,
                "clean_acc": clean,
                "PGD_acc": pgd,
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def add_general_scores(df_results):
    """Compute comparable aggregate scores solely from already-produced metrics."""
    robust_candidates = [
        c
        for c in (
            "AutoAttack",
            "BPDA_PGD",
            "Transfer_from_FP32",
            "MIM_Transfer",
            "BPDA_Adaptive",
            "EOT_PGD",
            "Adaptive_Guardrail",
            "Adaptive_DetectGuard",
        )
        if c in df_results.columns
    ]
    if robust_candidates:
        df_results["General_Robustness"] = df_results[robust_candidates].min(
            axis=1, skipna=True
        )
    if {"clean_acc", "General_Robustness"}.issubset(df_results.columns):
        denom = df_results["clean_acc"].replace(0, np.nan)
        df_results["General_Robustness_Retention"] = (
            df_results["General_Robustness"] / denom
        )
    if {"PGD", "General_Robustness"}.issubset(df_results.columns):
        gap = (df_results["PGD"] - df_results["General_Robustness"]).clip(lower=0)
        denom = (
            df_results["clean_acc"].replace(0, np.nan)
            if "clean_acc" in df_results
            else 1.0
        )
        df_results["General_Masking_Score"] = (gap / denom).clip(lower=0, upper=1)
    return df_results


def main():
    """Run the command-line entry point for this module."""
    check_environment()
    finetune_loader, eval_loader = get_dataloaders()
    model_registry = {}
    for arch_key in PRETRAINED_NAMES:
        print(f"\n>>> {arch_key} <<<")
        try:
            fp32 = load_pretrained(arch_key)
            fp32_layer_names = quantizable_layer_names(fp32)
            print(
                f"  FP32 quantizable nn.Conv2d/nn.Linear layers: {len(fp32_layer_names)}"
            )
            print(f"  first quantizable layers: {fp32_layer_names[:8]}")
            if not fp32_layer_names:
                raise RuntimeError(
                    f"{arch_key} exposes zero FP32 nn.Conv2d/nn.Linear layers."
                )
            acc = sanity_check_accuracy(fp32, eval_loader)
            print(f"  loaded pretrained {arch_key}, clean acc: {acc:.3f}")
            model_registry[f"{arch_key}_FP32"] = (fp32, None)
        except Exception as e:
            print(f"  [FAIL] could not load {arch_key}: {e}")
            traceback.print_exc()
            continue
        try:
            int8_ptq = convert_to_quant(
                fp32, bits=8, quant_weight=True, quant_act=True
            )
            verify_quantization_layers(
                arch_key, fp32, int8_ptq, "int8 PTQ", fp32_layer_names
            )
            model_registry[f"{arch_key}_int8_PTQ"] = (int8_ptq, fp32)
        except Exception as e:
            print(f"  [FAIL] int8 PTQ for {arch_key}: {e}")
            traceback.print_exc()
            raise
        try:
            int4_ptq = convert_to_quant(fp32, bits=4, quant_weight=True, quant_act=True)
            verify_quantization_layers(
                arch_key, fp32, int4_ptq, "int4 PTQ", fp32_layer_names
            )
            model_registry[f"{arch_key}_int4_PTQ"] = (int4_ptq, fp32)
        except Exception as e:
            print(f"  [FAIL] int4 PTQ for {arch_key}: {e}")
            traceback.print_exc()
            raise
        try:
            int8_qat = prepare_qat(
                fp32,
                bits=8,
                finetune_loader=finetune_loader,
                epochs=QAT_MAIN_EPOCHS,
            )
            verify_quantization_layers(
                arch_key, fp32, int8_qat, "int8 QAT", fp32_layer_names
            )
            model_registry[f"{arch_key}_int8_QAT"] = (int8_qat, fp32)
        except Exception as e:
            print(f"  [FAIL] int8 QAT for {arch_key}: {e}")
            traceback.print_exc()
            raise
        try:
            int4_qat = prepare_qat(
                fp32,
                bits=4,
                finetune_loader=finetune_loader,
                epochs=QAT_MAIN_EPOCHS,
            )
            verify_quantization_layers(
                arch_key, fp32, int4_qat, "int4 QAT", fp32_layer_names
            )
            model_registry[f"{arch_key}_int4_QAT"] = (int4_qat, fp32)
        except Exception as e:
            print(f"  [FAIL] int4 QAT for {arch_key}: {e}")
            traceback.print_exc()
            raise
        if QUANTIZATION_DEBUG_ONLY:
            print(
                "\nQUANTIZATION_DEBUG_ONLY=True; exiting before defenses, attacks, sweeps, and plots."
            )
            print("Registry built:", list(model_registry.keys()))
            return
        try:
            model_registry[f"{arch_key}_FP32_Compressed"] = (
                with_image_compression(fp32),
                fp32,
            )
        except Exception as e:
            print(f"  [FAIL] compressed FP32 for {arch_key}: {e}")
        if RUN_CHAOTIC_COMPRESS:
            try:
                chaotic_int8_ptq = convert_to_chaotic_quant(
                    fp32, bits=8, quant_weight=True, quant_act=True
                )
                model_registry[f"{arch_key}_chaotic_int8_PTQ"] = (
                    chaotic_int8_ptq,
                    fp32,
                )
            except Exception as e:
                print(f"  [FAIL] chaotic int8 PTQ for {arch_key}: {e}")
            try:
                chaotic_int4_ptq = convert_to_chaotic_quant(
                    fp32, bits=4, quant_weight=True, quant_act=True
                )
                model_registry[f"{arch_key}_chaotic_int4_PTQ"] = (
                    chaotic_int4_ptq,
                    fp32,
                )
            except Exception as e:
                print(f"  [FAIL] chaotic int4 PTQ for {arch_key}: {e}")
            try:
                compressed_chaotic_int8 = with_image_compression(
                    convert_to_chaotic_quant(
                        fp32, bits=8, quant_weight=True, quant_act=True
                    )
                )
                model_registry[f"{arch_key}_chaotic_int8_PTQ_Compressed"] = (
                    compressed_chaotic_int8,
                    fp32,
                )
            except Exception as e:
                print(f"  [FAIL] compressed chaotic int8 PTQ for {arch_key}: {e}")
            try:
                chaotic_int8_qat = prepare_qat(
                    fp32,
                    bits=8,
                    finetune_loader=finetune_loader,
                    epochs=QAT_MAIN_EPOCHS,
                    chaotic=True,
                )
                model_registry[f"{arch_key}_chaotic_int8_QAT"] = (
                    chaotic_int8_qat,
                    fp32,
                )
            except Exception as e:
                print(f"  [FAIL] chaotic int8 QAT for {arch_key}: {e}")
                traceback.print_exc()
    if RUN_DEFENSE_SUITE:
        try:
            model_registry, df_defense_summary = run_defense_suite(
                model_registry, finetune_loader, eval_loader
            )
            if not df_defense_summary.empty:
                print(
                    "\nDefense summary (guardrail/detector flag rates, certified accuracy):"
                )
                print(df_defense_summary.to_string(index=False))
        except Exception as e:
            print(f"  [FAIL] run_defense_suite failed: {e}")
            traceback.print_exc()
    if RUN_CHAOTIC_DITHER_SWEEP:
        dither_rows = []
        for arch_key in PRETRAINED_NAMES:
            entry = model_registry.get(f"{arch_key}_FP32")
            if entry is not None:
                dither_rows.extend(
                    run_chaotic_dither_sweep(entry[0], eval_loader, arch_key)
                )
        pd.DataFrame(dither_rows).to_csv(CHAOTIC_DITHER_SWEEP_CSV, index=False)
    chunk_model_names = []
    if RUN_CHUNK_QUANTIZATION:
        for arch_key in PRETRAINED_NAMES:
            entry = model_registry.get(f"{arch_key}_FP32")
            if entry is None:
                continue
            chunk_model_names.append(arch_key)
            out_path = csv_path(arch_key, "chunk_quant")
            if os.path.exists(out_path):
                print(
                    f"Skipping chunk quantization for {arch_key} (already in {out_path})"
                )
                continue
            print(f"\nChunk quantization sweep for {arch_key} ...")
            try:
                rows = run_chunk_quantization_attacks(
                    entry[0],
                    eval_loader,
                    arch_key,
                    bits=8,
                    n_chunks=CHUNK_QUANT_NUM_CHUNKS,
                    eps=DEFAULT_EPS,
                )
                pd.DataFrame(rows).to_csv(out_path, index=False)
                print(f"Chunk quantization results saved to {out_path}")
            except Exception as e:
                print(f"  [FAIL] chunk quantization sweep failed for {arch_key}: {e}")
                traceback.print_exc()
    print("\nRegistry built:", list(model_registry.keys()))
    for k in model_registry:
        m, r = model_registry[k]
        model_registry[k] = (parallelize(m), parallelize(r) if r else None)
    if os.path.exists(RESULTS_CSV):
        df_results = report_data.read_table(RESULTS_CSV)
        done = set(
            df_results.loc[
                df_results.get(
                    "clean_acc_n", pd.Series(index=df_results.index)
                ).notna(),
                "model",
            ].astype(str)
        )
    else:
        df_results = pd.DataFrame(columns=["model"])
        done = set()
    if os.path.exists(PERFORMANCE_CSV):
        existing_run_metrics = pd.read_csv(PERFORMANCE_CSV)
        run_metrics = (
            existing_run_metrics.to_dict("records")
            if "run_seconds" in existing_run_metrics.columns
            else []
        )
    else:
        run_metrics = []
    if RUN_EPSILON_SWEEP and os.path.exists(SWEEP_CSV):
        df_sweep = pd.read_csv(SWEEP_CSV)
        sweep_done = set(
            zip(df_sweep["model"].astype(str), df_sweep["epsilon"].round(6))
        )
    else:
        df_sweep = pd.DataFrame()
        sweep_done = set()

    def run_pending_epsilon_sweep(name, model):
        """Run an epsilon sweep for a model that still needs one."""
        nonlocal df_sweep, sweep_done
        if not RUN_EPSILON_SWEEP:
            return
        pending_eps = [
            eps for eps in SWEEP_EPSILONS if (name, round(eps, 6)) not in sweep_done
        ]
        if not pending_eps:
            print(f"  Skipping epsilon sweep for {name} (already done)")
            return

        print(f"\nSweeping {name} ...")

        def save_epsilon_sweep():
            """Persist epsilon-sweep diagnostics for pending models."""
            nonlocal df_sweep, sweep_done
            rows = run_epsilon_sweep_for_model_wrapped(
                model, eval_loader, name, pending_eps
            )
            if rows:
                new_sweep = pd.DataFrame(rows)
                df_sweep = report_data.upsert_table(
                    SWEEP_CSV, new_sweep, ["model", "epsilon"]
                )
                sweep_done = set(
                    zip(df_sweep["model"].astype(str), df_sweep["epsilon"].round(6))
                )

        safe_call(
            save_epsilon_sweep,
            "epsilon sweep failed",
            context=name,
            show_traceback=True,
        )

    for name, (model, ref) in list(model_registry.items()):
        if name in done:
            print(f"Skipping {name} (already in {RESULTS_CSV})")
            run_pending_epsilon_sweep(name, model)
            continue
        print(f"\nEvaluating {name} ...")
        monitor = ResourceMonitor(model, name)
        try:
            if RECORD_RUN_METRICS:
                with monitor:
                    res = run_suite(model, eval_loader, name, fp32_ref=ref)
            else:
                res = run_suite(model, eval_loader, name, fp32_ref=ref)
        except Exception as e:
            print(f"  [FAIL] run_suite failed for {name}: {e}")
            traceback.print_exc()
            res = {"model": name}
        finally:
            if RECORD_RUN_METRICS and monitor.metrics is not None:
                run_metrics = [
                    row for row in run_metrics if str(row.get("model")) != name
                ]
                run_metrics.append(monitor.metrics)
                report_data.upsert_table(
                    PERFORMANCE_CSV, pd.DataFrame([monitor.metrics]), ["model"]
                )
                print("Run metrics:")
                print(pd.DataFrame([monitor.metrics]).to_string(index=False))
        new_row = pd.DataFrame([res])
        df_results = report_data.upsert_table(RESULTS_CSV, new_row, ["model"])
        print("Result:")
        print(new_row.to_string(index=False))
        run_pending_epsilon_sweep(name, model)
        print("-" * 100)
    print("\nFinal results:")
    print(df_results)
    adaptive_cols = [
        c
        for c in [
            "BPDA_PGD",
            "BPDA_Adaptive",
            "EOT_PGD",
            "Adaptive_Guardrail",
            "Adaptive_DetectGuard",
            "NES",
            "Boundary_acc",
            "AutoAttack",
        ]
        if c in df_results.columns
    ]
    if adaptive_cols:
        df_results["Worst_Robust_Acc"] = df_results[adaptive_cols].min(
            axis=1, skipna=True
        )
    if {"PGD", "Worst_Robust_Acc"}.issubset(df_results.columns):
        df_results["Gradient_Masking_Gap"] = (
            df_results["PGD"] - df_results["Worst_Robust_Acc"]
        )
    fp32_baseline = (
        df_results[df_results["model"].str.endswith("_FP32")]
        .assign(Architecture=lambda d: d["model"].str.replace("_FP32", "", regex=False))
        .set_index("Architecture")["Worst_Robust_Acc"]
    )
    df_results["Architecture"] = df_results["model"].str.replace(
        r"_(FP32|int8_PTQ|int4_PTQ|int8_QAT|int4_QAT).*", "", regex=True
    )
    df_results["FP32_Worst_Robust_Acc"] = df_results["Architecture"].map(fp32_baseline)
    if {
        "Worst_Robust_Acc",
        "FP32_Worst_Robust_Acc",
    }.issubset(df_results.columns):
        df_results["True_Robustness_Gain"] = (
            df_results["Worst_Robust_Acc"] - df_results["FP32_Worst_Robust_Acc"]
        )
    df_results = report_data.add_paired_tests(
        report_data.add_derived_metrics(df_results)
    )
    df_results.to_csv(RESULTS_CSV, index=False)
    if RUN_EPSILON_SWEEP:
        print("\nEpsilon sweep completed. Results saved to", SWEEP_CSV)

    def build_reports():
        """Combine result files and generate final report artifacts."""
        tables = report_data.combine_all(report_data.DATA_DIR)
        report_data.plot_all(tables, report_data.DATA_DIR)
        report_data.print_report(tables)

    safe_call(build_reports, "report generation failed", show_traceback=True)
    print("All done.")


if __name__ == "__main__":
    main()

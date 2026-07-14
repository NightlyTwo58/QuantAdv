"""Fake-quantization layers, conversion helpers, and hard/STE mode controls."""

import copy
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torchao.quantization.qat import IntxFakeQuantizeConfig, IntxFakeQuantizer

from config import *

WEIGHT = "weight"
ACTIVATION = "activation"


def _dtype(bits):
    dtype = torch.int8 if bits == 8 else getattr(torch, f"int{bits}", None)
    if dtype is None:
        raise RuntimeError(f"This PyTorch build does not expose torch.int{bits}")
    return dtype


def make_fake_quantizer(bits, role):
    if role not in (WEIGHT, ACTIVATION):
        raise ValueError(f"Unknown quantizer role: {role!r}")
    return IntxFakeQuantizer(
        IntxFakeQuantizeConfig(
            dtype=_dtype(bits),
            granularity="per_channel" if role == WEIGHT else "per_token",
            is_symmetric=role == WEIGHT,
            is_dynamic=True,
            eps=QUANT_SCALE_MIN,
        )
    )


def hard_fake_quantize(tensor, bits, role):
    """Match TorchAO forward numerics using native hard ``torch.round``."""
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    with torch.no_grad():
        zero = tensor.new_zeros(())
        lo = tensor.amin(dim=-1, keepdim=True).minimum(zero)
        hi = tensor.amax(dim=-1, keepdim=True).maximum(zero)
        if role == WEIGHT:
            scale = (2 * torch.maximum(-lo, hi) / (qmax - qmin)).clamp_min(
                QUANT_SCALE_MIN
            )
            zero_point = torch.zeros_like(scale, dtype=torch.int32)
        elif role == ACTIVATION:
            scale = ((hi - lo) / (qmax - qmin)).clamp_min(QUANT_SCALE_MIN)
            zero_point = (qmin - torch.round(lo / scale)).clamp(qmin, qmax).int()
        else:
            raise ValueError(f"Unknown quantizer role: {role!r}")
    integer = torch.clamp(
        torch.round(tensor * (1.0 / scale)) + zero_point, qmin, qmax
    )
    return (integer - zero_point) * scale


def chaotic_sequence_like(tensor, seed=CHAOTIC_QUANT_SEED, map_name=CHAOTIC_QUANT_MAP):
    if tensor.numel() == 0:
        return torch.empty_like(tensor)
    dtype = torch.float32 if tensor.dtype in (torch.float16, torch.bfloat16) else tensor.dtype
    z = torch.frac(seed + (torch.arange(tensor.numel(), device=tensor.device, dtype=dtype) + 1) * 0.6180339887498949)
    z = z.clamp(1e-6, 1 - 1e-6)
    for _ in range(CHAOTIC_QUANT_WARMUP):
        z = (torch.where(z < .5, CHAOTIC_QUANT_MU * z * .5, CHAOTIC_QUANT_MU * (1 - z) * .5)
             if map_name == "tent" else CHAOTIC_QUANT_R * z * (1 - z))
    return z.view_as(tensor).to(tensor.dtype)


def fake_quantize(tensor, bits, role, quantizer, use_ste, dither_amplitude=None):
    """Uniform hard/STE entry point, optionally with subtractive chaotic dither."""
    dither = 0
    if dither_amplitude is not None:
        with torch.no_grad():
            amplitude = tensor.abs().amax().clamp_min(QUANT_SCALE_MIN)
        dither = (chaotic_sequence_like(tensor) - .5) * dither_amplitude * amplitude
    value = tensor + dither
    quantized = quantizer(value) if use_ste else hard_fake_quantize(value, bits, role)
    return quantized - dither


class _QuantizedMixin:
    chaotic = False

    def _init_quantization(self, bits, quant_weight, quant_act, dither_amplitude):
        self.bits, self.use_ste = bits, QUANT_DEFAULT_USE_STE
        self.quant_weight, self.quant_act = quant_weight, quant_act
        self.dither_amplitude = dither_amplitude
        self.weight_fake_quantizer = make_fake_quantizer(bits, WEIGHT)
        self.activation_fake_quantizer = make_fake_quantizer(bits, ACTIVATION)

    def _quantize(self, tensor, role, enabled):
        if not enabled:
            return tensor
        shape = tensor.shape
        value = tensor.reshape(tensor.shape[0], -1) if role == WEIGHT else tensor
        quantizer = self.weight_fake_quantizer if role == WEIGHT else self.activation_fake_quantizer
        result = fake_quantize(
            value, self.bits, role, quantizer, self.use_ste,
            self.dither_amplitude if self.chaotic else None,
        )
        return result.reshape(shape)


class QuantConv2d(_QuantizedMixin, nn.Conv2d):
    def forward(self, x):
        weight = self._quantize(self.weight, WEIGHT, self.quant_weight)
        return self._quantize(self._conv_forward(x, weight, self.bias), ACTIVATION, self.quant_act)


class ChaoticQuantConv2d(QuantConv2d):
    chaotic = True


class QuantLinear(_QuantizedMixin, nn.Linear):
    def forward(self, x):
        weight = self._quantize(self.weight, WEIGHT, self.quant_weight)
        return self._quantize(F.linear(x, weight, self.bias), ACTIVATION, self.quant_act)


class ChaoticQuantLinear(QuantLinear):
    chaotic = True


QUANT_TYPES = (QuantConv2d, QuantLinear, ChaoticQuantConv2d, ChaoticQuantLinear)


def _convert_module(module, bits, quant_weight, quant_act, chaotic, dither_amplitude):
    if isinstance(module, nn.Conv2d):
        cls = ChaoticQuantConv2d if chaotic else QuantConv2d
        new = cls(module.in_channels, module.out_channels, module.kernel_size,
                  module.stride, module.padding, module.dilation, module.groups,
                  module.bias is not None, module.padding_mode,
                  device=module.weight.device, dtype=module.weight.dtype)
    elif isinstance(module, nn.Linear):
        cls = ChaoticQuantLinear if chaotic else QuantLinear
        new = cls(module.in_features, module.out_features, module.bias is not None,
                  device=module.weight.device, dtype=module.weight.dtype)
    else:
        return None
    new.weight = module.weight
    if module.bias is not None:
        new.bias = module.bias
    new._init_quantization(bits, quant_weight, quant_act, dither_amplitude)
    return new


def _replace(module, bits, quant_weight, quant_act, chaotic, dither_amplitude):
    for name, child in list(module.named_children()):
        replacement = _convert_module(child, bits, quant_weight, quant_act, chaotic, dither_amplitude)
        if replacement is None:
            _replace(child, bits, quant_weight, quant_act, chaotic, dither_amplitude)
        else:
            setattr(module, name, replacement)


def convert_to_quant(model, bits, quant_weight=True, quant_act=True, chaotic=False,
                     dither_amplitude=CHAOTIC_QUANT_DITHER):
    result = copy.deepcopy(model)
    _replace(result, bits, quant_weight, quant_act, chaotic, dither_amplitude)
    return result


def convert_to_chaotic_quant(model, bits, quant_weight=True, quant_act=True,
                             dither_amplitude=CHAOTIC_QUANT_DITHER):
    return convert_to_quant(model, bits, quant_weight, quant_act, True, dither_amplitude)


def quantizable_layer_names(model):
    return [name for name, mod in model.named_modules()
            if isinstance(mod, (nn.Conv2d, nn.Linear)) and not isinstance(mod, QUANT_TYPES)]


def convert_layer_chunk_to_quant(model, layer_names, bits, quant_weight=True,
                                 quant_act=True, chaotic=False):
    result, targets = copy.deepcopy(model), set(layer_names)
    for name, module in list(result.named_modules()):
        if name not in targets:
            continue
        replacement = _convert_module(module, bits, quant_weight, quant_act, chaotic, CHAOTIC_QUANT_DITHER)
        if replacement is not None:
            parent_name, _, child_name = name.rpartition(".")
            parent = result.get_submodule(parent_name) if parent_name else result
            setattr(parent, child_name, replacement)
    return result.to(device).eval()


def quant_layer_chunks(layer_names, n_chunks):
    if not layer_names:
        return []
    return [list(x) for x in np.array_split(layer_names, min(max(1, n_chunks), len(layer_names))) if len(x)]


def count_quant_layers(model):
    return sum(isinstance(module, QUANT_TYPES) for module in model.modules())


def verify_quantization_layers(arch_key, fp32_model, quant_model, label, fp32_layer_names=None):
    names = quantizable_layer_names(fp32_model) if fp32_layer_names is None else fp32_layer_names
    if not names:
        raise RuntimeError(f"{arch_key} exposes zero quantizable layers")
    count, threshold = count_quant_layers(quant_model), int(np.ceil(.8 * len(names)))
    print(f"  {label} quantized layers: {count}", flush=True)
    if count < threshold:
        raise RuntimeError(f"{arch_key} {label} replaced {count}/{len(names)} layers; expected at least {threshold}")
    return count


def set_quant_components(model, quant_weight, quant_act):
    for module in model.modules():
        if isinstance(module, QUANT_TYPES):
            module.quant_weight, module.quant_act = quant_weight, quant_act


def prepare_qat(fp32_model, bits, finetune_loader, epochs=QAT_EPOCHS_DEFAULT,
                lr=QAT_LR, chaotic=False):
    """Fine-tune a quantized copy through TorchAO's STE path."""
    model = convert_to_quant(fp32_model, bits, chaotic=chaotic).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    set_ste_mode(model, True)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=QAT_MOMENTUM,
                                weight_decay=QAT_WEIGHT_DECAY)
    scaler = GradScaler(device=device.type, enabled=device.type == "cuda")
    for epoch in range(epochs):
        running = 0.0
        for batch_idx, (x, y) in enumerate(finetune_loader):
            x = x.to(device, non_blocking=NON_BLOCKING_TRANSFER)
            y = y.to(device, non_blocking=NON_BLOCKING_TRANSFER)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            if batch_idx == 0 or (batch_idx + 1) % QAT_LOG_EVERY_BATCHES == 0:
                print(f"  QAT epoch {epoch + 1}/{epochs} batch {batch_idx + 1}/{len(finetune_loader)} loss {loss.item():.4f}", flush=True)
        print(f"  QAT epoch {epoch + 1}/{epochs} avg loss {running / len(finetune_loader):.4f}", flush=True)
    set_ste_mode(model, False)
    return model.eval()


def set_ste_mode(model, flag):
    # Keep the mode protocol structural so defense wrappers and any external
    # quantized modules exposing the same controls participate as well.
    modules = [m for m in model.modules() if hasattr(m, "bits") and hasattr(m, "use_ste")]
    for module in modules:
        module.use_ste = flag
    return len(modules)


def get_ste_mode(model):
    values = {m.use_ste for m in model.modules()
              if hasattr(m, "bits") and hasattr(m, "use_ste")}
    return None if not values else next(iter(values)) if len(values) == 1 else "mixed"


@contextmanager
def ste_mode(model, flag):
    previous = get_ste_mode(model)
    set_ste_mode(model, flag)
    try:
        yield
    finally:
        set_ste_mode(model, previous if isinstance(previous, bool) else False)

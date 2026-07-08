"""Custom Conv2d/Linear fake quantization and QAT helpers."""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from .config import *


class FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x).clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def quantize_tensor(t, bits, use_ste):
    if bits is None:
        return t
    qmax = 2 ** (bits - 1) - 1
    scale = torch.clamp(t.detach().abs().max() / qmax, min=QUANT_SCALE_MIN)
    t_scaled = t / scale
    t_round = FakeQuantSTE.apply(t_scaled) if use_ste else torch.round(t_scaled)
    t_round = torch.clamp(t_round, -qmax - 1, qmax)
    return t_round * scale


def chaotic_sequence_like(t, seed=CHAOTIC_QUANT_SEED, map_name=CHAOTIC_QUANT_MAP):
    n = t.numel()
    if n == 0:
        return torch.empty_like(t)
    dtype = torch.float32 if t.dtype in (torch.float16, torch.bfloat16) else t.dtype
    idx = torch.arange(n, device=t.device, dtype=dtype)
    z = torch.frac(seed + (idx + 1.0) * 0.6180339887498949)
    z = torch.clamp(z, 1e-6, 1.0 - 1e-6)
    if map_name == "tent":
        for _ in range(CHAOTIC_QUANT_WARMUP):
            z = torch.where(z < 0.5, CHAOTIC_QUANT_MU * z * 0.5, CHAOTIC_QUANT_MU * (1.0 - z) * 0.5)
    else:
        for _ in range(CHAOTIC_QUANT_WARMUP):
            z = CHAOTIC_QUANT_R * z * (1.0 - z)
    return z.view_as(t).to(dtype=t.dtype)


def chaotic_quantize_tensor(t, bits, use_ste, quantize=True):
    if bits is None or not quantize:
        return t
    qmax = 2 ** (bits - 1) - 1
    scale = torch.clamp(t.detach().abs().max() / qmax, min=QUANT_SCALE_MIN)
    chaos = chaotic_sequence_like(t)
    dither = (chaos - 0.5) * CHAOTIC_QUANT_DITHER
    t_scaled = t / scale + dither
    t_round = FakeQuantSTE.apply(t_scaled) if use_ste else torch.round(t_scaled)
    t_round = torch.clamp(t_round, -qmax - 1, qmax)
    return (t_round - dither) * scale


class _QuantizedLayerMixin:
    chaotic = False

    def _quant_params(self):
        bits = getattr(self, 'bits', None)
        use_ste = getattr(self, 'use_ste', QUANT_DEFAULT_USE_STE)
        quant_weight = getattr(self, 'quant_weight', QUANT_DEFAULT_WEIGHT)
        quant_act = getattr(self, 'quant_act', QUANT_DEFAULT_ACT)
        return bits, use_ste, quant_weight, quant_act

    def _quantize(self, t, bits, use_ste, enabled=True):
        if self.chaotic:
            return chaotic_quantize_tensor(t, bits, use_ste, enabled)
        return quantize_tensor(t, bits, use_ste) if enabled else t


class QuantConv2d(_QuantizedLayerMixin, nn.Conv2d):
    def forward(self, x):
        bits, use_ste, quant_weight, quant_act = self._quant_params()
        w = self._quantize(self.weight, bits, use_ste, quant_weight)
        out = self._conv_forward(x, w, self.bias)
        if quant_act:
            out = self._quantize(out, bits, use_ste, quant_act)
        return out.clone()


class ChaoticQuantConv2d(QuantConv2d):
    chaotic = True


class QuantLinear(_QuantizedLayerMixin, nn.Linear):
    def forward(self, x):
        bits, use_ste, quant_weight, quant_act = self._quant_params()
        w = self._quantize(self.weight, bits, use_ste, quant_weight)
        out = F.linear(x, w, self.bias)
        if quant_act:
            out = self._quantize(out, bits, use_ste, quant_act)
        return out.clone()


class ChaoticQuantLinear(QuantLinear):
    chaotic = True


def _to_quant_module(mod, bits, quant_weight=True, quant_act=True, chaotic=False):
    if isinstance(mod, nn.Conv2d):
        cls = ChaoticQuantConv2d if chaotic else QuantConv2d
        new = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                  mod.stride, mod.padding, mod.dilation, mod.groups,
                  mod.bias is not None, mod.padding_mode)
        new.weight = mod.weight
        if mod.bias is not None:
            new.bias = mod.bias
        new.bits, new.use_ste, new.quant_weight, new.quant_act = bits, QUANT_DEFAULT_USE_STE, quant_weight, quant_act
        return new
    if isinstance(mod, nn.Linear):
        cls = ChaoticQuantLinear if chaotic else QuantLinear
        new = cls(mod.in_features, mod.out_features, bias=mod.bias is not None)
        new.weight = mod.weight
        if mod.bias is not None:
            new.bias = mod.bias
        new.bits, new.use_ste, new.quant_weight, new.quant_act = bits, QUANT_DEFAULT_USE_STE, quant_weight, quant_act
        return new
    return None


def _replace_recursive(module, bits, quant_weight=True, quant_act=True, chaotic=False):
    for name, child in list(module.named_children()):
        nc = _to_quant_module(child, bits, quant_weight, quant_act, chaotic=chaotic)
        if nc is not None:
            setattr(module, name, nc)
        else:
            _replace_recursive(child, bits, quant_weight, quant_act, chaotic=chaotic)


def convert_to_quant(model, bits, quant_weight=True, quant_act=True, chaotic=False):
    m = copy.deepcopy(model)
    _replace_recursive(m, bits, quant_weight, quant_act, chaotic=chaotic)
    return m


def convert_to_chaotic_quant(model, bits, quant_weight=True, quant_act=True):
    return convert_to_quant(model, bits, quant_weight=quant_weight, quant_act=quant_act, chaotic=True)


def quantizable_layer_names(model):
    quant_types = (QuantConv2d, QuantLinear, ChaoticQuantConv2d, ChaoticQuantLinear)
    return [n for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear)) and not isinstance(m, quant_types)]


def set_child_module(root, module_name, new_module):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent._modules[part]
    parent._modules[parts[-1]] = new_module


def convert_layer_chunk_to_quant(model, layer_names, bits, quant_weight=True, quant_act=True, chaotic=False):
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
    return [list(chunk) for chunk in np.array_split(np.array(layer_names, dtype=object), n_chunks) if len(chunk) > 0]


def set_ste_mode(model, flag):
    toggled = 0
    for mod in model.modules():
        if isinstance(mod, (QuantConv2d, QuantLinear, ChaoticQuantConv2d, ChaoticQuantLinear)):
            mod.use_ste = flag
            toggled += 1
    return toggled


def count_quant_layers(model):
    return sum(1 for m in model.modules() if isinstance(m, (QuantConv2d, QuantLinear, ChaoticQuantConv2d, ChaoticQuantLinear)))


def verify_quantization_layers(arch_key, fp32_model, quant_model, label, fp32_layer_names=None):
    fp32_layer_names = quantizable_layer_names(fp32_model) if fp32_layer_names is None else fp32_layer_names
    if not fp32_layer_names:
        raise RuntimeError(f"{arch_key} exposes zero FP32 nn.Conv2d/nn.Linear layers.")
    quant_count = count_quant_layers(quant_model)
    threshold = int(np.ceil(0.8 * len(fp32_layer_names)))
    print(f"  {label} quantized layers: {quant_count}")
    if quant_count < threshold:
        raise RuntimeError(
            f"{arch_key} {label} replaced {quant_count}/{len(fp32_layer_names)} quantizable layers; expected at least {threshold}."
        )
    return quant_count


def set_quant_components(model, quant_weight, quant_act):
    for mod in model.modules():
        if isinstance(mod, (QuantConv2d, QuantLinear, ChaoticQuantConv2d, ChaoticQuantLinear)):
            mod.quant_weight = quant_weight
            mod.quant_act = quant_act


def prepare_qat(fp32_model, bits, finetune_loader, epochs=QAT_EPOCHS_DEFAULT, lr=QAT_LR, chaotic=False):
    m = convert_to_chaotic_quant(fp32_model, bits, quant_weight=True, quant_act=True) if chaotic else convert_to_quant(fp32_model, bits, quant_weight=True, quant_act=True)
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
    scaler = GradScaler(device=device.type)

    for epoch in range(epochs):
        running = 0.0
        for x, y in finetune_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)

            with autocast(device_type=device.type):
                loss = F.cross_entropy(m(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
        print(f"  QAT epoch {epoch+1}/{epochs} avg loss {running/len(finetune_loader):.4f}")
    set_ste_mode(m, False)
    return m.eval()


CUSTOM_QUANT_MODULES = (QuantConv2d, QuantLinear, ChaoticQuantConv2d, ChaoticQuantLinear)


def iter_quant_modules(model):
    return (mod for mod in model.modules() if isinstance(mod, CUSTOM_QUANT_MODULES))

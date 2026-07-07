import copy
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler

from .config import (
    CHAOTIC_QUANT_DITHER,
    CHAOTIC_QUANT_MAP,
    CHAOTIC_QUANT_MU,
    CHAOTIC_QUANT_R,
    CHAOTIC_QUANT_SEED,
    CHAOTIC_QUANT_WARMUP,
    QAT_EPOCHS_DEFAULT,
    QAT_LR,
    QAT_MOMENTUM,
    QAT_WEIGHT_DECAY,
    QUANT_DEFAULT_ACT,
    QUANT_DEFAULT_USE_STE,
    QUANT_DEFAULT_WEIGHT,
    QUANT_SCALE_MIN,
    USE_AMP,
    device,
)


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
            z = torch.where(
                z < 0.5,
                CHAOTIC_QUANT_MU * z * 0.5,
                CHAOTIC_QUANT_MU * (1.0 - z) * 0.5,
            )
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


class QuantConv2d(nn.Conv2d):
    def forward(self, x):
        bits = getattr(self, "bits", None)
        use_ste = getattr(self, "use_ste", QUANT_DEFAULT_USE_STE)
        quant_weight = getattr(self, "quant_weight", QUANT_DEFAULT_WEIGHT)
        quant_act = getattr(self, "quant_act", QUANT_DEFAULT_ACT)
        w = quantize_tensor(self.weight, bits, use_ste) if quant_weight else self.weight
        out = self._conv_forward(x, w, self.bias)
        return quantize_tensor(out, bits, use_ste).clone() if quant_act else out.clone()


class ChaoticQuantConv2d(QuantConv2d):
    def forward(self, x):
        bits = getattr(self, "bits", None)
        use_ste = getattr(self, "use_ste", QUANT_DEFAULT_USE_STE)
        quant_weight = getattr(self, "quant_weight", QUANT_DEFAULT_WEIGHT)
        quant_act = getattr(self, "quant_act", QUANT_DEFAULT_ACT)
        w = chaotic_quantize_tensor(self.weight, bits, use_ste, quant_weight) if quant_weight else self.weight
        out = self._conv_forward(x, w, self.bias)
        return chaotic_quantize_tensor(out, bits, use_ste, quant_act).clone() if quant_act else out.clone()


class QuantLinear(nn.Linear):
    def forward(self, x):
        bits = getattr(self, "bits", None)
        use_ste = getattr(self, "use_ste", QUANT_DEFAULT_USE_STE)
        quant_weight = getattr(self, "quant_weight", QUANT_DEFAULT_WEIGHT)
        quant_act = getattr(self, "quant_act", QUANT_DEFAULT_ACT)
        w = quantize_tensor(self.weight, bits, use_ste) if quant_weight else self.weight
        out = F.linear(x, w, self.bias)
        return quantize_tensor(out, bits, use_ste).clone() if quant_act else out.clone()


class ChaoticQuantLinear(QuantLinear):
    def forward(self, x):
        bits = getattr(self, "bits", None)
        use_ste = getattr(self, "use_ste", QUANT_DEFAULT_USE_STE)
        quant_weight = getattr(self, "quant_weight", QUANT_DEFAULT_WEIGHT)
        quant_act = getattr(self, "quant_act", QUANT_DEFAULT_ACT)
        w = chaotic_quantize_tensor(self.weight, bits, use_ste, quant_weight) if quant_weight else self.weight
        out = F.linear(x, w, self.bias)
        return chaotic_quantize_tensor(out, bits, use_ste, quant_act).clone() if quant_act else out.clone()


CUSTOM_QUANT_MODULES = (QuantConv2d, QuantLinear, ChaoticQuantConv2d, ChaoticQuantLinear)


def _copy_common_module_state(src, dst):
    dst.weight = src.weight
    if src.bias is not None:
        dst.bias = src.bias
    dst.training = src.training
    return dst


def _to_quant_module(mod, bits, quant_weight=True, quant_act=True, chaotic=False):
    if isinstance(mod, CUSTOM_QUANT_MODULES):
        mod.bits = bits
        mod.quant_weight = quant_weight
        mod.quant_act = quant_act
        return mod
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
        )
        _copy_common_module_state(mod, new)
        new.bits = bits
        new.use_ste = QUANT_DEFAULT_USE_STE
        new.quant_weight = quant_weight
        new.quant_act = quant_act
        return new
    if isinstance(mod, nn.Linear):
        cls = ChaoticQuantLinear if chaotic else QuantLinear
        new = cls(mod.in_features, mod.out_features, bias=mod.bias is not None)
        _copy_common_module_state(mod, new)
        new.bits = bits
        new.use_ste = QUANT_DEFAULT_USE_STE
        new.quant_weight = quant_weight
        new.quant_act = quant_act
        return new
    return None


def _replace_recursive(module, bits, quant_weight=True, quant_act=True, chaotic=False):
    for name, child in list(module.named_children()):
        new_child = _to_quant_module(child, bits, quant_weight, quant_act, chaotic=chaotic)
        if new_child is not None:
            setattr(module, name, new_child)
        else:
            _replace_recursive(child, bits, quant_weight, quant_act, chaotic=chaotic)


def convert_to_quant(model, bits, quant_weight=True, quant_act=True):
    quantized = copy.deepcopy(model)
    _replace_recursive(quantized, bits, quant_weight, quant_act)
    return quantized


def convert_to_chaotic_quant(model, bits, quant_weight=True, quant_act=True):
    quantized = copy.deepcopy(model)
    _replace_recursive(quantized, bits, quant_weight, quant_act, chaotic=True)
    return quantized


def set_ste_mode(model, flag):
    toggled = 0
    for mod in model.modules():
        if isinstance(mod, CUSTOM_QUANT_MODULES):
            mod.use_ste = flag
            toggled += 1
    return toggled


def set_quant_components(model, quant_weight, quant_act):
    for mod in model.modules():
        if isinstance(mod, CUSTOM_QUANT_MODULES):
            mod.quant_weight = quant_weight
            mod.quant_act = quant_act


def iter_quant_modules(model):
    modules = getattr(model, "modules", None)
    if modules is None:
        return iter(())
    return (mod for mod in modules() if isinstance(mod, CUSTOM_QUANT_MODULES))


def count_quant_layers(model):
    return sum(1 for _ in iter_quant_modules(model))


def _amp_ctx():
    if USE_AMP:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def prepare_qat(fp32_model, bits, finetune_loader, epochs=QAT_EPOCHS_DEFAULT, lr=QAT_LR, chaotic=False):
    model = (
        convert_to_chaotic_quant(fp32_model, bits, quant_weight=True, quant_act=True)
        if chaotic
        else convert_to_quant(fp32_model, bits, quant_weight=True, quant_act=True)
    )
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    set_ste_mode(model, True)
    model.train()

    opt = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=QAT_MOMENTUM,
        weight_decay=QAT_WEIGHT_DECAY,
    )
    scaler = GradScaler(device="cuda", enabled=USE_AMP)

    for epoch in range(epochs):
        running = 0.0
        for x, y in finetune_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with _amp_ctx():
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
        print(f"  QAT epoch {epoch + 1}/{epochs} avg loss {running / len(finetune_loader):.4f}")

    set_ste_mode(model, False)
    return model.eval()

"""
Quantization wrapper around a plain PyTorch model, exposing torchao-based
int8 PTQ (post-training quantization) and int8 QAT (quantization-aware
training) variants through a single `Model` class.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchao.quantization import (
    quantize_,
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
)
from torchao.quantization.granularity import PerGroup
from torchao.quantization.qat import QATConfig
from torchao.quantization.qat.linear import FakeQuantizedLinear
import copy
from config import *
import time

def _get_device():
    """Return the device to use for model operations."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# Mapping from bitwidth string to torchao base config & QAT config

_BITWIDTH_TO_CONFIG = {
    8: {
        "ptq": lambda: Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int8, weight_granularity=PerGroup(32)
        ),
        "qat_prepare": lambda: QATConfig(
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int8, weight_granularity=PerGroup(32)
            ),
            step="prepare",
        ),
        "qat_convert": lambda: QATConfig(
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int8, weight_granularity=PerGroup(32)
            ),
            step="convert",
        ),
        "weight_dtype": torch.int8,
        "activation_dtype": torch.int8,
    },
}


def _config_for_bits(bits):
    """Return (ptq_config, qat_prepare_config, qat_convert_config) for the given bitwidth."""
    if bits not in _BITWIDTH_TO_CONFIG:
        raise ValueError(
            f"Unsupported bitwidth {bits}. Supported: {list(_BITWIDTH_TO_CONFIG.keys())}"
        )
    cfg = _BITWIDTH_TO_CONFIG[bits]
    return (
        cfg["ptq"](),
        cfg["qat_prepare"](),
        cfg["qat_convert"](),
    )


# Model


class Model:
    """
    OOP wrapper around a PyTorch model that manages quantization state:

      * self.base_model        – original FP32 model (kept pristine)
      * self.int8_PTQ          – post-training int8-quantized model
      * self.int8_QAT          – quantization-aware int8 model (after train_QAT)

    The model follows torchao's QAT lifecycle:

        prepare()  →  train()  →  convert()

    where `prepare()` injects fake quantization, `train()` runs the training loop
    with STE, and `convert()` replaces fake-quantized modules with actual
    quantized ones.
    """

    def __init__(self, model):
        """
        Args:
            model: A PyTorch ``nn.Module`` (e.g. ResNet20, MobileNetV2, …).
                   Must be on ``device`` before calling.
        """
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected nn.Module, got {type(model).__name__}")

        self.model = model
        self._device = _get_device()

        self.int8_PTQ = copy.deepcopy(model)
        quantize_(
            self.int8_PTQ,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int8, weight_granularity=PerGroup(32)
            ),
        )

        self.int8_PTQ_weight_only = copy.deepcopy(model)
        quantize_(
            self.int8_PTQ_weight_only,
            IntxWeightOnlyConfig(weight_dtype=torch.int8, granularity=PerGroup(32)),
        )

        #- QAT models (in "untrained" state)-------------------------
        # These start as plain deepcopies; they become "QAT" after prepare_qat().
        self.int8_QAT_untrained = copy.deepcopy(model)
        self._qat_initialized = {8: False}  # tracks whether prepare+convert has run

    # Internal helpers

    def _apply_qat_prepare(self, qat_model, bits):
        """
        Apply the QAT *prepare* step: injects fake quantization into the model.
        """
        _, qat_prepare_config, _ = _config_for_bits(bits)
        quantize_(qat_model, qat_prepare_config)

    def _apply_qat_convert(self, qat_model, bits):
        """
        Apply the QAT *convert* step: replaces fake-quantized modules with
        actual quantized modules.
        """
        _, _, qat_convert_config = _config_for_bits(bits)
        quantize_(qat_model, qat_convert_config)

    @staticmethod
    def _count_quant_layers(model):
        """Count quantized layers in a model.

        Detects:
          - FakeQuantizedLinear / FakeQuantizedConv2d (QAT prepare state)
          - Modules with _quantized_op (older torchao QAT-convert representation)
          - Modules whose .weight has been replaced by a torchao tensor-subclass
            (current torchao's PTQ and QAT-convert representation, e.g.
            IntxUnpackedToInt8Tensor for Int8DynamicActivationIntxWeightConfig).
            Without this branch, PTQ and post-convert QAT models silently report
            0 quantized layers even though they are quantized.
        """
        count = 0
        for mod in model.modules():
            if isinstance(mod, (FakeQuantizedLinear,)):
                count += 1
                continue
            if hasattr(mod, "_quantized_op") and mod._quantized_op is not None:
                count += 1
                continue
            weight = getattr(mod, "weight", None)
            if weight is not None and type(weight).__module__.startswith("torchao"):
                count += 1
        return count

    def _get_qat_model(self, bits):
        """Return the QAT model object (only 8-bit supported)."""
        if bits != 8:
            raise ValueError(f"bits must be 8, got {bits}")
        return self.int8_QAT_untrained

    # QAT preparation

    def prepare_qat(self, bits):
        """
        Prepare a QAT model for training.

        Steps:
            1. Take a fresh deep-copy of the base model (if not already prepared).
            2. Apply torchao's QAT *prepare* step (injects fake quantization).

        Args:
            bits: Integer, 8.

        Returns:
            The prepared model object for in-place training.
        """
        if bits != 8:
            raise ValueError(f"bits must be 8, got {bits}")

        qat_model = self._get_qat_model(bits)

        # If already prepared, return as-is.
        if self._qat_initialized[bits]:
            print(f"[qat] QAT model ({bits}-bit) already prepared, returning.")
            return qat_model

        self._apply_qat_prepare(qat_model, bits)
        self._qat_initialized[bits] = True

        n_quant = self._count_quant_layers(qat_model)
        print(
            f"[qat] QAT prepare complete ({bits}-bit): "
            f"{n_quant} fake-quantized layers."
        )
        return qat_model

    # Standard QAT training (clean data)

    def train_qat(
        self,
        finetune_loader,
        epochs=5,
        lr=1e-3,
        weight_decay=5e-4,
        bits=8,
    ):
        """
        Standard (clean) QAT fine-tuning loop.

        The full lifecycle is:
            1. prepare_qat(bits)    – inject fake quantization
            2. train_qat()          – SGD on clean data through fake-quantized layers
            3. (implicit convert)   – convert() is called on return

        Args:
            finetune_loader: DataLoader yielding (images, labels).
            epochs: Number of training epochs.
            lr: Learning rate.
            weight_decay: L2 weight decay.
            bits: Target bitwidth, 8.

        Returns:
            ``self.int8_QAT`` – the QAT model ready for inference
            (i.e. after the convert step).
        """
        if bits != 8:
            raise ValueError(f"bits must be 8, got {bits}")

        name = f"int{bits}_QAT"
        print(f"[qat] Starting standard QAT training ({name}, bits={bits}, "
              f"epochs={epochs}, lr={lr})...")

        qat_model = self.prepare_qat(bits)
        device = _get_device()

        qat_model.train()
        opt = torch.optim.SGD(
            qat_model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )

        n_batches = len(finetune_loader)
        for epoch in range(epochs):
            epoch_start = time.time()
            running_loss = 0.0
            for bi, (x, y) in enumerate(finetune_loader):
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = F.cross_entropy(qat_model(x), y)
                loss.backward()
                opt.step()
                running_loss += loss.item()

                if (bi + 1) % max(n_batches // 5, 1) == 0 or (bi + 1) == n_batches:
                    avg = running_loss / (bi + 1)
                    print(
                        f"  [qat] {name} epoch {epoch + 1}/{epochs} "
                        f"batch {bi + 1}/{n_batches} loss={loss.item():.4f} "
                        f"avg_loss={avg:.4f}"
                    )

            elapsed = time.time() - epoch_start
            avg_loss = running_loss / n_batches
            print(
                f"  [qat] {name} epoch {epoch + 1}/{epochs} "
                f"avg_loss={avg_loss:.4f} ({elapsed:.1f}s)"
            )

        # Convert to final quantized model
        self._apply_qat_convert(qat_model, bits)

        # Store result on self
        setattr(self, name, qat_model.eval())
        print(f"[qat] Standard QAT training complete ({name}).")
        return getattr(self, name)

    # Inference helpers

    @staticmethod
    def clean_accuracy(model, loader):
        """
        Evaluate clean accuracy on *loader*.

        Args:
            model: PyTorch model (any quantization state).
            loader: DataLoader yielding (images, labels).

        Returns:
            Float accuracy in [0, 1].
        """
        model.eval()
        device = _get_device()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        return correct / total if total > 0 else 0.0

    def evaluate(
        self,
        loader,
        model_name="",
        bits=None,
        qat_mode="clean",
    ):
        """
        Run a full evaluation suite on a specific quantized model.

        Args:
            loader: DataLoader yielding (images, labels).
            model_name: Display name for logging.
            bits: Bitwidth of the model to evaluate (8 only).
            qat_mode: One of "ptq", "qat_clean".

        Returns:
            dict with accuracy results.
        """
        if bits != 8:
            raise ValueError(f"bits must be 8, got {bits}")

        if qat_mode == "ptq":
            target_model = self.int8_PTQ
            label = f"int{bits}_PTQ"
        elif qat_mode == "qat_clean":
            target_model = getattr(self, f"int{bits}_QAT", None)
            label = f"int{bits}_QAT"
            if target_model is None:
                raise ValueError(
                    f"int{bits}_QAT not trained yet. Call train_qat() first."
                )
        else:
            raise ValueError(f"Unknown qat_mode: {qat_mode}")

        acc = self.clean_accuracy(target_model, loader)
        print(
            f"[eval] {model_name or label} clean_acc={acc:.4f} "
            f"(bits={bits}, mode={qat_mode})"
        )

        return {
            "model": model_name or label,
            "bits": bits,
            "mode": qat_mode,
            "clean_acc": acc,
        }

    # Summary / inspection

    def summary(self):
        """
        Print a summary of all built models and their QAT state.
        """
        device = _get_device()

        def _num_params(m):
            return sum(p.numel() for p in m.parameters())

        lines = [
            "===== Model Summary =====",
            f"Device: {device}",
            f"Base model: {type(self.model).__name__}",
            f"Base model parameters: {_num_params(self.model):,}",
        ]

        for bits in (8,):
            label_prefix = f"int{bits}"
            lines.append(f"--- {label_prefix}")

            ptq_ref = getattr(self, f"{label_prefix}_PTQ", None)
            lines.append(f"  PTQ (activation+weight) built: {ptq_ref is not None}")
            if ptq_ref is not None:
                lines.append(f"    quantized layers: {self._count_quant_layers(ptq_ref)}")

            ptq_wo_ref = getattr(self, f"{label_prefix}_PTQ_weight_only", None)
            lines.append(f"  PTQ (weight-only) built: {ptq_wo_ref is not None}")
            if ptq_wo_ref is not None:
                lines.append(f"    quantized layers: {self._count_quant_layers(ptq_wo_ref)}")

            qat_prepared = self._qat_initialized[bits]
            lines.append(f"  QAT prepared: {qat_prepared}")
            qat_untrained_ref = getattr(self, f"{label_prefix}_QAT_untrained", None)
            if qat_prepared and qat_untrained_ref is not None:
                lines.append(f"    fake-quantized layers: {self._count_quant_layers(qat_untrained_ref)}")

            qat_trained_ref = getattr(self, f"{label_prefix}_QAT", None)
            lines.append(f"  QAT (clean, trained+converted) built: {qat_trained_ref is not None}")
            if qat_trained_ref is not None:
                lines.append(f"    quantized layers: {self._count_quant_layers(qat_trained_ref)}")

            lines.append("")

        return "\n".join(lines)

    def __repr__(self):
        return self.summary()

# ===== Ported chaotic quantization logic from archive/QuantAdv.py =====
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


class QuantConv2d(nn.Conv2d):
    def forward(self, x):
        bits = getattr(self, 'bits', None)
        use_ste = getattr(self, 'use_ste', QUANT_DEFAULT_USE_STE)
        quant_weight = getattr(self, 'quant_weight', QUANT_DEFAULT_WEIGHT)
        quant_act = getattr(self, 'quant_act', QUANT_DEFAULT_ACT)

        w = quantize_tensor(self.weight, bits, use_ste) if quant_weight else self.weight
        out = self._conv_forward(x, w, self.bias)
        if quant_act:
            out = quantize_tensor(out, bits, use_ste)
        return out.clone()


class ChaoticQuantConv2d(QuantConv2d):
    def forward(self, x):
        bits = getattr(self, 'bits', None)
        use_ste = getattr(self, 'use_ste', QUANT_DEFAULT_USE_STE)
        quant_weight = getattr(self, 'quant_weight', QUANT_DEFAULT_WEIGHT)
        quant_act = getattr(self, 'quant_act', QUANT_DEFAULT_ACT)

        w = chaotic_quantize_tensor(self.weight, bits, use_ste, quant_weight) if quant_weight else self.weight
        out = self._conv_forward(x, w, self.bias)
        if quant_act:
            out = chaotic_quantize_tensor(out, bits, use_ste, quant_act)
        return out.clone()


class QuantLinear(nn.Linear):
    def forward(self, x):
        bits = getattr(self, 'bits', None)
        use_ste = getattr(self, 'use_ste', QUANT_DEFAULT_USE_STE)
        quant_weight = getattr(self, 'quant_weight', QUANT_DEFAULT_WEIGHT)
        quant_act = getattr(self, 'quant_act', QUANT_DEFAULT_ACT)

        w = quantize_tensor(self.weight, bits, use_ste) if quant_weight else self.weight
        out = F.linear(x, w, self.bias)
        if quant_act:
            out = quantize_tensor(out, bits, use_ste)
        return out.clone()


class ChaoticQuantLinear(QuantLinear):
    def forward(self, x):
        bits = getattr(self, 'bits', None)
        use_ste = getattr(self, 'use_ste', QUANT_DEFAULT_USE_STE)
        quant_weight = getattr(self, 'quant_weight', QUANT_DEFAULT_WEIGHT)
        quant_act = getattr(self, 'quant_act', QUANT_DEFAULT_ACT)

        w = chaotic_quantize_tensor(self.weight, bits, use_ste, quant_weight) if quant_weight else self.weight
        out = F.linear(x, w, self.bias)
        if quant_act:
            out = chaotic_quantize_tensor(out, bits, use_ste, quant_act)
        return out.clone()


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


def convert_to_quant(model, bits, quant_weight=True, quant_act=True):
    m = copy.deepcopy(model)
    _replace_recursive(m, bits, quant_weight, quant_act)
    return m


def convert_to_chaotic_quant(model, bits, quant_weight=True, quant_act=True):
    m = copy.deepcopy(model)
    _replace_recursive(m, bits, quant_weight, quant_act, chaotic=True)
    return m


def chaotic_set_ste_mode(model, flag):
    for mod in model.modules():
        if isinstance(mod, (QuantConv2d, QuantLinear, ChaoticQuantConv2d, ChaoticQuantLinear)):
            mod.use_ste = flag


def count_quant_layers(model):
    return sum(1 for m in model.modules() if isinstance(m, (QuantConv2d, QuantLinear, ChaoticQuantConv2d, ChaoticQuantLinear)))


# flip weight/activation quantization on an already-built quantized model
# without rebuilding it, so one model object can be reused for all three
# ablation configs instead of re-running convert_to_quant/QAT.
def set_quant_components(model, quant_weight, quant_act):
    for mod in model.modules():
        if isinstance(mod, (QuantConv2d, QuantLinear, ChaoticQuantConv2d, ChaoticQuantLinear)):
            mod.quant_weight = quant_weight
            mod.quant_act = quant_act


def chaotic_prepare_qat(fp32_model, bits, finetune_loader, epochs=QAT_EPOCHS_DEFAULT, lr=QAT_LR, chaotic=False):
    m = convert_to_chaotic_quant(fp32_model, bits, quant_weight=True, quant_act=True) if chaotic else convert_to_quant(fp32_model, bits, quant_weight=True, quant_act=True)
    if torch.cuda.device_count() > 1:
        m = nn.DataParallel(m)

    chaotic_set_ste_mode(m, True)
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
    chaotic_set_ste_mode(m, False)
    return m.eval()

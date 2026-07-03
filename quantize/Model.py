import torch
import torch.nn as nn
import torch.nn.functional as F
from torchao.quantization import (
    quantize_,
    Int8DynamicActivationIntxWeightConfig,
)
from torchao.quantization.granularity import PerGroup
from torchao.quantization.qat import QATConfig
import copy
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

    def _count_quant_layers(self, model):
        """Count quantized layers in a model (modules with quantized params)."""
        count = 0
        for mod in model.modules():
            if hasattr(mod, "_quantized_op"):
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
        epochs=3,
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
        lines = [
            "===== Model Summary =====",
            f"Device: {device}",
            f"Base model: {type(self.model).__name__}",
            "",
        ]

        for bits in (8,):
            label_prefix = f"int{bits}"
            lines.append(f"--- {label_prefix}")
            ptq_ref = getattr(self, f"{label_prefix}_PTQ", None)
            lines.append(f"  PTQ built: {ptq_ref is not None}")
            lines.append(f"  QAT prepared: {self._qat_initialized[bits]}")

            lines.append(f"  QAT (clean) built: {getattr(self, f'{label_prefix}_QAT', None) is not None}")
            lines.append("")

        return "\n".join(lines)

    def __repr__(self):
        return self.summary()
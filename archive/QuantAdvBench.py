#!/usr/bin/env python
# coding: utf-8
"""
QuantAdvBench.py

Cheap benchmarking harness derived from QuantAdv.py.
Goal: figure out which stage of the pipeline (data loading, model loading,
PTQ conversion, QAT fine-tuning, clean-acc eval, PGD, BPDA, or raw inference)
is the time hog NOT to produce research-grade robustness numbers.

Everything is deliberately tiny (few images, few steps, 1 QAT epoch, 1 seed)
so the whole thing runs in a couple minutes and the timing breakdown is what
matters, not the accuracy values.

All outputs (CSV logs, timing breakdown, plots) and all detected inputs
(dataset root, device, package versions) go under /benchmarking.
"""

import importlib.util
import logging
import os
import sys
import time
import traceback
from contextlib import contextmanager
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import seaborn as sns

# Paths -- everything lives under /benchmarking
BENCH_DIR = "/benchmarking"
os.makedirs(BENCH_DIR, exist_ok=True)

LOG_PATH = os.path.join(BENCH_DIR, "benchmark.log")
TIMING_CSV = os.path.join(BENCH_DIR, "stage_timings.csv")
TIMING_PNG = os.path.join(BENCH_DIR, "stage_timings.png")
RESULTS_CSV = os.path.join(BENCH_DIR, "results_gutted.csv")
SPEED_CSV = os.path.join(BENCH_DIR, "inference_speed.csv")
SPEED_PNG = os.path.join(BENCH_DIR, "inference_speed.png")

# Logging -- console + file, timestamps down to ms
#
# NOTE: this is a function, not top-level code, and is only called from
# main() under the `if __name__ == "__main__":` guard below. On Windows
# (and anywhere DataLoader uses num_workers>0 with the 'spawn' start
# method), worker processes re-import this file. If logger setup lived at
# module level it would re-run in every worker -> duplicate handlers,
# duplicate "Logging to console..." lines, and a garbled/truncated log
# file from concurrent writers. Keeping it inside main() means workers
# only pick up function/class definitions, never execute this.
def setup_logging():
    logger = logging.getLogger("quantadvbench")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(LOG_PATH, mode="w")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    logger.info(f"Logging to console + {LOG_PATH}")
    return logger

logger = logging.getLogger("quantadvbench")  # module-level handle, handlers attached later in main()

# Timing infrastructure
TIMINGS = []  # list of dicts: {stage, detail, seconds}

@contextmanager
def timed(stage, detail=""):
    logger.debug(f"START  {stage} {('(' + detail + ')') if detail else ''}")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        TIMINGS.append({"stage": stage, "detail": detail, "seconds": dt})
        logger.info(f"DONE   {stage} {('(' + detail + ')') if detail else ''} -- {dt:.3f}s")

def dump_timings():
    df = pd.DataFrame(TIMINGS)
    df.to_csv(TIMING_CSV, index=False)
    logger.info(f"Wrote stage timings -> {TIMING_CSV}")

    agg = df.groupby("stage")["seconds"].sum().sort_values(ascending=False)
    logger.info("Slowest stages (summed across all calls):")
    for stage, secs in agg.items():
        logger.info(f"    {stage:<30s} {secs:8.3f}s")

    plt.figure(figsize=(10, max(4, 0.4 * len(agg))))
    sns.barplot(x=agg.values, y=agg.index, orient="h", color="steelblue")
    plt.xlabel("Total seconds")
    plt.title("Time spent per pipeline stage (summed)")
    plt.tight_layout()
    plt.savefig(TIMING_PNG, dpi=200, bbox_inches="tight")
    logger.info(f"Wrote timing chart -> {TIMING_PNG}")
    return agg

# Detected inputs / environment -- also only run once, from main()
def detect_environment():
    with timed("detect_environment"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        CIFAR10_ROOT = os.environ.get("CIFAR10_ROOT", "./")
        expected = os.path.join(CIFAR10_ROOT, "cifar-10-batches-py")
        dataset_found = os.path.isdir(expected)

        missing = [pkg for pkg in ("torchattacks", "autoattack") if importlib.util.find_spec(pkg) is None]

        logger.info(f"device               = {device}")
        if device == "cuda":
            logger.info(f"cuda device name     = {torch.cuda.get_device_name(0)}")
            logger.info(f"cuda device count    = {torch.cuda.device_count()}")
        logger.info(f"CIFAR10_ROOT (env)   = {CIFAR10_ROOT!r}")
        logger.info(f"dataset dir expected = {expected!r}")
        logger.info(f"dataset dir found    = {dataset_found}")
        logger.info(f"torch version        = {torch.__version__}")
        logger.info(f"torchvision version  = {torchvision.__version__}")
        logger.info(f"missing packages     = {missing}")

        if missing:
            logger.error(f"Missing packages: {missing}. Install via: pip install -r requirements.txt")
            raise ImportError(f"Missing packages: {missing}")
        if not dataset_found:
            logger.error(f"Expected extracted CIFAR-10 at {expected!r}")
            raise FileNotFoundError(f"Expected extracted CIFAR-10 at {expected!r}")

    return device, CIFAR10_ROOT

# Placeholders -- real values are assigned inside main() via detect_environment(),
# never at import time, so spawned DataLoader worker processes never trigger
# device probing / dataset-path checks / logging setup a second time.
device = None
CIFAR10_ROOT = None

import torchattacks

# GUTTED settings -- small on purpose
EVAL_N = 100          # vs 500 in full script
FINETUNE_N = 200      # vs 4000
QAT_EPOCHS = 1        # vs 3
PGD_STEPS = 5         # vs 20
BPDA_RESTARTS = 1
SEEDS = [0]           # vs [0,1,2]
SPEED_BATCHES = 10    # number of batches used to time raw inference
ARCHES = ["ResNet20", "MobileNetV2"]  # trimmed from all 6 to keep it cheap

PRETRAINED_NAMES = {
    "ResNet20": "cifar10_resnet20",
    "ResNet56": "cifar10_resnet56",
    "MobileNetV2": "cifar10_mobilenetv2_x1_0",
    "VGG16_BN": "cifar10_vgg16_bn",
    "ShuffleNetV2": "cifar10_shufflenetv2_x1_0",
    "RepVGG_A0": "cifar10_repvgg_a0",
}

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
CIFAR_MEAN_T = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
CIFAR_STD_T = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
CLIP_MIN = ((0.0 - CIFAR_MEAN_T) / CIFAR_STD_T)
CLIP_MAX = ((1.0 - CIFAR_MEAN_T) / CIFAR_STD_T)

# Data
def get_dataloaders(batch_size=64, eval_n=EVAL_N, finetune_n=FINETUNE_N):
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
    ])

    train_full = torchvision.datasets.CIFAR10(root=CIFAR10_ROOT, train=True, download=False, transform=transform_train)
    test_full = torchvision.datasets.CIFAR10(root=CIFAR10_ROOT, train=False, download=False, transform=transform_test)

    finetune_subset = torch.utils.data.Subset(train_full, list(range(finetune_n)))
    eval_subset = torch.utils.data.Subset(test_full, list(range(eval_n)))

    workers = min(4, os.cpu_count() or 1)

    finetune_loader = torch.utils.data.DataLoader(
        finetune_subset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_subset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True
    )
    logger.debug(f"finetune_n={finetune_n} eval_n={eval_n} batch_size={batch_size} workers={workers}")
    return finetune_loader, eval_loader

def load_pretrained(arch_key):
    hub_name = PRETRAINED_NAMES[arch_key]
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", hub_name, pretrained=True)
    return model.to(device).eval()

def sanity_check_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# Quantization infra (kept in full -- includes int4, same as QuantAdv.py)
class FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def quantize_tensor(t, bits, use_ste):
    if bits is None:
        return t
    qmax = 2 ** (bits - 1) - 1
    scale = torch.clamp(t.detach().abs().max() / qmax, min=1e-8)
    t_scaled = t / scale
    t_round = FakeQuantSTE.apply(t_scaled) if use_ste else torch.round(t_scaled)
    t_round = torch.clamp(t_round, -qmax - 1, qmax)
    return t_round * scale

class QuantConv2d(nn.Conv2d):
    def forward(self, x):
        bits = getattr(self, 'bits', None)
        use_ste = getattr(self, 'use_ste', False)
        quant_weight = getattr(self, 'quant_weight', True)
        quant_act = getattr(self, 'quant_act', True)
        w = quantize_tensor(self.weight, bits, use_ste) if quant_weight else self.weight
        out = self._conv_forward(x, w, self.bias)
        if quant_act:
            out = quantize_tensor(out, bits, use_ste)
        return out

class QuantLinear(nn.Linear):
    def forward(self, x):
        bits = getattr(self, 'bits', None)
        use_ste = getattr(self, 'use_ste', False)
        quant_weight = getattr(self, 'quant_weight', True)
        quant_act = getattr(self, 'quant_act', True)
        w = quantize_tensor(self.weight, bits, use_ste) if quant_weight else self.weight
        out = F.linear(x, w, self.bias)
        if quant_act:
            out = quantize_tensor(out, bits, use_ste)
        return out

def _to_quant_module(mod, bits, quant_weight=True, quant_act=True):
    if isinstance(mod, nn.Conv2d):
        new = QuantConv2d(mod.in_channels, mod.out_channels, mod.kernel_size,
                          mod.stride, mod.padding, mod.dilation, mod.groups,
                          mod.bias is not None, mod.padding_mode)
        new.weight = mod.weight
        if mod.bias is not None:
            new.bias = mod.bias
        new.bits, new.use_ste, new.quant_weight, new.quant_act = bits, False, quant_weight, quant_act
        return new
    if isinstance(mod, nn.Linear):
        new = QuantLinear(mod.in_features, mod.out_features, bias=mod.bias is not None)
        new.weight = mod.weight
        if mod.bias is not None:
            new.bias = mod.bias
        new.bits, new.use_ste, new.quant_weight, new.quant_act = bits, False, quant_weight, quant_act
        return new
    return None

def _replace_recursive(module, bits, quant_weight=True, quant_act=True):
    for name, child in list(module.named_children()):
        nc = _to_quant_module(child, bits, quant_weight, quant_act)
        if nc is not None:
            setattr(module, name, nc)
        else:
            _replace_recursive(child, bits, quant_weight, quant_act)

def convert_to_quant(model, bits, quant_weight=True, quant_act=True):
    m = copy.deepcopy(model)
    _replace_recursive(m, bits, quant_weight, quant_act)
    return m

def set_ste_mode(model, flag):
    for mod in model.modules():
        if isinstance(mod, (QuantConv2d, QuantLinear)):
            mod.use_ste = flag

def count_quant_layers(model):
    return sum(1 for m in model.modules() if isinstance(m, (QuantConv2d, QuantLinear)))

def prepare_qat(fp32_model, bits, finetune_loader, epochs=QAT_EPOCHS, lr=1e-3):
    m = convert_to_quant(fp32_model, bits, quant_weight=True, quant_act=True)
    set_ste_mode(m, True)
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for epoch in range(epochs):
        running = 0.0
        for x, y in finetune_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(m(x), y)
            loss.backward()
            opt.step()
            running += loss.item()
        logger.debug(f"    QAT epoch {epoch+1}/{epochs} avg loss {running/len(finetune_loader):.4f}")
    set_ste_mode(m, False)
    return m.eval()

# Gutted attacks
def run_pgd(model, loader, eps=8/255, steps=PGD_STEPS, seeds=SEEDS):
    model.eval()
    accs = []
    for seed in seeds:
        torch.manual_seed(seed)
        pgd = torchattacks.PGD(model, eps=eps, alpha=2/255, steps=steps, random_start=True)
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_adv = pgd(x, y)
            with torch.no_grad():
                pred = model(x_adv).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        accs.append(correct / total)
    return float(np.mean(accs))

def bpda_pgd_attack(model, x, y, eps=8/255, alpha=2/255, steps=PGD_STEPS):
    set_ste_mode(model, True)
    clip_min, clip_max = CLIP_MIN.to(device), CLIP_MAX.to(device)
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
        x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
    set_ste_mode(model, False)
    return x_adv.detach()

def run_bpda(model, loader, eps=8/255, n_restarts=BPDA_RESTARTS, seeds=SEEDS):
    accs = []
    for seed in seeds:
        torch.manual_seed(seed)
        correct_masks = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            worst_correct = torch.ones(y.size(0), dtype=torch.bool, device=device)
            for _ in range(n_restarts):
                x_adv = bpda_pgd_attack(model, x, y, eps=eps)
                with torch.no_grad():
                    pred = model(x_adv).argmax(dim=1)
                worst_correct &= (pred == y)
            correct_masks.append(worst_correct)
        accs.append(torch.cat(correct_masks).float().mean().item())
    return float(np.mean(accs))

# Raw inference speed benchmark (FP32 vs int8 vs int4)
def benchmark_inference_speed(model, loader, name, n_batches=SPEED_BATCHES, warmup=2):
    model.eval()
    batches = []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches + warmup:
            break
        batches.append(x.to(device))
    if not batches:
        raise RuntimeError("No batches available for speed benchmark")

    batch_size = batches[0].size(0)

    # warmup (not timed) -- lets cudnn autotune / caches warm before we measure
    with torch.no_grad():
        for x in batches[:warmup]:
            _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    timed_batches = batches[warmup:] if len(batches) > warmup else batches
    t0 = time.perf_counter()
    with torch.no_grad():
        for x in timed_batches:
            _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    n = len(timed_batches)
    ms_per_batch = (total_time / n) * 1000
    images_per_sec = (n * batch_size) / total_time

    logger.info(f"    [speed] {name:<20s} {ms_per_batch:8.2f} ms/batch  {images_per_sec:9.1f} img/s")
    return {"model": name, "ms_per_batch": ms_per_batch, "images_per_sec": images_per_sec,
            "n_batches": n, "batch_size": batch_size}

# Main
def main():
    global logger, device, CIFAR10_ROOT

    logger = setup_logging()
    device, CIFAR10_ROOT = detect_environment()

    run_t0 = time.perf_counter()
    logger.info("=" * 70)
    logger.info("QuantAdvBench -- gutted timing/benchmark run")
    logger.info(f"ARCHES={ARCHES}  EVAL_N={EVAL_N}  FINETUNE_N={FINETUNE_N}  "
                f"QAT_EPOCHS={QAT_EPOCHS}  PGD_STEPS={PGD_STEPS}")
    logger.info("=" * 70)

    with timed("get_dataloaders"):
        finetune_loader, eval_loader = get_dataloaders()

    results_rows = []
    speed_rows = []

    for arch_key in ARCHES:
        logger.info(f"\n>>> {arch_key} <<<")

        try:
            with timed("load_pretrained", arch_key):
                fp32 = load_pretrained(arch_key)
        except Exception as e:
            logger.error(f"[FAIL] could not load {arch_key}: {e}")
            traceback.print_exc()
            continue

        variants = {f"{arch_key}_FP32": fp32}

        try:
            with timed("convert_to_quant_int8", arch_key):
                variants[f"{arch_key}_int8_PTQ"] = convert_to_quant(fp32, bits=8)
        except Exception as e:
            logger.error(f"[FAIL] int8 PTQ for {arch_key}: {e}")

        try:
            with timed("convert_to_quant_int4", arch_key):
                variants[f"{arch_key}_int4_PTQ"] = convert_to_quant(fp32, bits=4)
        except Exception as e:
            logger.error(f"[FAIL] int4 PTQ for {arch_key}: {e}")

        try:
            with timed("prepare_qat_int8", arch_key):
                variants[f"{arch_key}_int8_QAT"] = prepare_qat(fp32, bits=8, finetune_loader=finetune_loader)
        except Exception as e:
            logger.error(f"[FAIL] int8 QAT for {arch_key}: {e}")
            traceback.print_exc()

        for name, model in variants.items():
            logger.info(f"  -- evaluating {name} --")
            row = {"model": name}

            try:
                with timed("clean_acc", name):
                    row["clean_acc"] = sanity_check_accuracy(model, eval_loader)
            except Exception as e:
                logger.warning(f"clean_acc failed for {name}: {e}")
                row["clean_acc"] = None

            try:
                with timed("pgd_attack", name):
                    row["PGD_acc"] = run_pgd(model, eval_loader)
            except Exception as e:
                logger.warning(f"PGD failed for {name}: {e}")
                row["PGD_acc"] = None

            if count_quant_layers(model) > 0:
                try:
                    with timed("bpda_attack", name):
                        row["BPDA_acc"] = run_bpda(model, eval_loader)
                except Exception as e:
                    logger.warning(f"BPDA failed for {name}: {e}")
                    row["BPDA_acc"] = None

            results_rows.append(row)

            # --- inference speed test (FP32 vs int8 vs int4 vs QAT) ---
            try:
                with timed("inference_speed_bench", name):
                    speed_rows.append(benchmark_inference_speed(model, eval_loader, name))
            except Exception as e:
                logger.warning(f"speed benchmark failed for {name}: {e}")

    # ---- save gutted results ----
    with timed("save_results_csv"):
        df_results = pd.DataFrame(results_rows)
        df_results.to_csv(RESULTS_CSV, index=False)
        logger.info(f"Wrote {RESULTS_CSV}")
        logger.info("\n" + df_results.to_string(index=False))

    # ---- save + plot speed comparison ----
    with timed("save_speed_csv_and_plot"):
        df_speed = pd.DataFrame(speed_rows)
        df_speed.to_csv(SPEED_CSV, index=False)
        logger.info(f"Wrote {SPEED_CSV}")
        logger.info("\n" + df_speed.to_string(index=False))

        if len(df_speed):
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            sns.barplot(data=df_speed, x="model", y="ms_per_batch", ax=axes[0], color="indianred")
            axes[0].set_title("Inference latency (ms / batch) -- lower is better")
            axes[0].tick_params(axis="x", rotation=45)
            for label in axes[0].get_xticklabels():
                label.set_ha("right")

            sns.barplot(data=df_speed, x="model", y="images_per_sec", ax=axes[1], color="seagreen")
            axes[1].set_title("Inference throughput (images / sec) -- higher is better")
            axes[1].tick_params(axis="x", rotation=45)
            for label in axes[1].get_xticklabels():
                label.set_ha("right")

            plt.tight_layout()
            plt.savefig(SPEED_PNG, dpi=200, bbox_inches="tight")
            logger.info(f"Wrote {SPEED_PNG}")

    total_time = time.perf_counter() - run_t0
    logger.info(f"\nTotal wall-clock time: {total_time:.2f}s")

    agg = dump_timings()
    top = agg.index[0] if len(agg) else None
    if top is not None:
        logger.info(f"\n>>> Slowest stage overall: '{top}' ({agg.iloc[0]:.2f}s total) <<<")

    logger.info("All benchmark outputs are in " + BENCH_DIR)

if __name__ == '__main__':
    main()
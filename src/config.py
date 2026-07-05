"""
Central configuration and constants for the quantadv package.

Includes device selection, CIFAR-10 normalization/clipping statistics, the
project/data directory locations, and the set of pretrained CIFAR-10
architectures used throughout the quantization + adversarial-robustness
pipeline.
"""
import logging
import os
from pathlib import Path

import torch

# Silence a noisy torchao/torch.utils._pytree warning that otherwise clutters
# every run's output.
logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)

device = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = torch.cuda.is_available()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(DATA_DIR, "results.csv")
SWEEP_CSV = os.path.join(DATA_DIR, "results_sweep.csv")
PLOT_PNG = os.path.join(DATA_DIR, "accuracy_plot.png")

SEEDS = [0, 1, 2]

PRETRAINED_NAMES = {
    "ResNet20": "cifar10_resnet20",
    "ResNet56": "cifar10_resnet56",
    "MobileNetV2": "cifar10_mobilenetv2_x1_0",
    "VGG16_BN": "cifar10_vgg16_bn",
    "ShuffleNetV2": "cifar10_shufflenetv2_x1_0",
    "RepVGG_A0": "cifar10_repvgg_a0",
}

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
CLIP_MIN = (0.0 - CIFAR_MEAN) / CIFAR_STD
CLIP_MAX = (1.0 - CIFAR_MEAN) / CIFAR_STD
CLIP_MIN_DEV = CLIP_MIN.to(device)
CLIP_MAX_DEV = CLIP_MAX.to(device)

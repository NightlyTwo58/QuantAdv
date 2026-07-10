"""Data loading utilities for CIFAR-10 and pytorchcv pretrained models."""

import os

import torch
import torchvision
import torchvision.transforms as T
from pytorchcv.model_provider import get_model as ptcv_get_model

from .config import *


def load_pretrained(arch_key):
    if arch_key != "ResNet56":
        raise ValueError(f"Unsupported architecture {arch_key!r}; expected 'ResNet56'.")
    if ptcv_get_model is None:
        raise ImportError(
            "Missing package 'pytorchcv'. Install via: pip install -r requirements.txt"
        )
    model_name = PRETRAINED_NAMES[arch_key]
    model = ptcv_get_model(model_name, pretrained=True, root=PYTORCHCV_MODEL_DIR)
    return model.to(device).eval()


def get_dataloaders(
    batch_size=DEFAULT_BATCH_SIZE, eval_n=DEFAULT_EVAL_N, finetune_n=DEFAULT_FINETUNE_N
):
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
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_subset,
        batch_size=DEFAULT_EVAL_BATCH_SIZE,
        shuffle=EVAL_SHUFFLE,
        num_workers=workers,
        pin_memory=PIN_MEMORY,
    )

    return finetune_loader, eval_loader

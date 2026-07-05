"""
Data loading utilities: pretrained CIFAR-10 model loading and DataLoader
construction for fine-tuning / evaluation.
"""
import os

import torch
import torchvision
import torchvision.transforms as T

from .config import PROJECT_ROOT, PRETRAINED_NAMES, device


def load_pretrained(arch_key):
    hub_name = PRETRAINED_NAMES[arch_key]
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", hub_name, pretrained=True)
    return model.to(device).eval()


def get_dataloaders(batch_size=100, eval_n=500, finetune_n=4000):
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])

    train_full = torchvision.datasets.CIFAR10(root=PROJECT_ROOT, train=True, download=False, transform=transform_train)
    test_full = torchvision.datasets.CIFAR10(root=PROJECT_ROOT, train=False, download=False, transform=transform_test)

    finetune_subset = torch.utils.data.Subset(train_full, list(range(finetune_n)))
    eval_subset = torch.utils.data.Subset(test_full, list(range(eval_n)))

    workers = min(4, os.cpu_count() or 1)

    finetune_loader = torch.utils.data.DataLoader(
        finetune_subset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_subset, batch_size=100, shuffle=False, num_workers=workers, pin_memory=True
    )

    return finetune_loader, eval_loader

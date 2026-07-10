"""Clean accuracy and epsilon-sweep evaluation routines."""

import os
import traceback

import pandas as pd
import torch
import torchattacks

from .config import *
from .quantization import count_quant_layers
from .attacks import (
    make_torchattack,
    accuracy_under_attack,
    random_noise_attack,
    _run_bpda_once,
)


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


def safe_run(fn, name, label):
    try:
        return fn(), None
    except Exception as e:
        print(f"  [WARN] {label} failed for {name}: {e}")
        traceback.print_exc()
        return None, e


def run_epsilon_sweep_for_model(model, loader, name, epsilons):
    rows = []
    is_quant = count_quant_layers(model) > 0
    for eps in epsilons:
        row = {"model": name, "epsilon": eps}
        try:
            pgd = make_torchattack(
                torchattacks.PGD,
                model,
                eps=eps,
                alpha=PGD_ALPHA,
                steps=PGD_STEPS,
                random_start=PGD_RANDOM_START,
            )
            row["PGD_acc"] = accuracy_under_attack(model, loader, pgd)
        except Exception as e:
            print(f"  [WARN] PGD sweep failed for {name} eps={eps:.4f}: {e}")
            row["PGD_acc"] = None

        try:
            row["Random_Noise_acc"] = random_noise_attack(model, loader, eps=eps)
        except Exception as e:
            print(f"  [WARN] random_noise sweep failed for {name} eps={eps:.4f}: {e}")
            row["Random_Noise_acc"] = None

        if is_quant:
            try:
                row["BPDA_acc"] = _run_bpda_once(
                    model, loader, eps=eps, n_restarts=BPDA_RESTARTS_SWEEP
                )
            except Exception as e:
                print(f"  [WARN] BPDA sweep failed for {name} eps={eps:.4f}: {e}")
                row["BPDA_acc"] = None
        rows.append(row)
    return rows

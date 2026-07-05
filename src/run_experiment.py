#!/usr/bin/env python
# coding: utf-8
"""
CLI entrypoint for the quantization + adversarial-robustness experiment.

Single Python invocation analysis of metrics of attacks per model and
epsilon. Uses torchao-based quantization via the quantadv.model.Model class.

Usage:
    python src/run_experiment.py                        # run all architectures
    python src/run_experiment.py --arch-key ResNet20     # run a single architecture
                                                      # (used internally by
                                                      # dispatch_multi_gpu for
                                                      # one-process-per-GPU runs)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.experiment import _ARGS, _startup_checks, main, dispatch_multi_gpu

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if _ARGS.arch_key is None and torch.cuda.device_count() > 1:
        print("device:", device)
        _startup_checks()
        dispatch_multi_gpu()
    else:
        print("device:", device)
        _startup_checks()
        main()

#!/usr/bin/env python
# coding: utf-8
"""
Combines the per-model CSVs/JSON written by run_experiment.py into master
files and produces summary plots.

Run this after run_experiment.py (or its multi-GPU dispatch) finishes all
subprocesses:

    python src/combine_results.py

Safe to re-run at any time since it only reads whatever per-model files
currently exist and never mutates them.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.combine import main

if __name__ == "__main__":
    main()

"""
Experiment entrypoint: orchestrates loading pretrained CIFAR-10 architectures,
building PTQ/QAT-quantized variants via the Model wrapper, running the full
adversarial-robustness evaluation suite and epsilon sweep for each, and
(optionally) dispatching one worker process per GPU for parallel evaluation
across architectures.
"""
import argparse
import gc
import importlib.util
import os
import subprocess
import sys
import time
import traceback

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from . import config
from .config import PROJECT_ROOT, DATA_DIR, PRETRAINED_NAMES, device
from .data import get_dataloaders, load_pretrained
from .model import Model as QuantModel
from .compute import parallelize, maybe_compile
from .evaluation import sanity_check_accuracy, run_epsilon_sweep_for_model
from .suite import run_suite, _model_to_qat_instance
from .paths import results_csv_path, sweep_csv_path

PLOT_PNG = config.PLOT_PNG


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arch-key", default=None,
                        help="Internal: restrict this run to a single architecture.")
    args, _ = parser.parse_known_args()
    return args


_ARGS = _parse_args()

RESULTS_CSV = config.RESULTS_CSV
SWEEP_CSV = config.SWEEP_CSV

if _ARGS.arch_key is not None:
    if _ARGS.arch_key not in PRETRAINED_NAMES:
        raise ValueError(f"Unknown --arch-key {_ARGS.arch_key!r}, expected one of {list(PRETRAINED_NAMES)}")
    PRETRAINED_NAMES = {_ARGS.arch_key: PRETRAINED_NAMES[_ARGS.arch_key]}
    # Per-arch paths are now derived from model names; keep these for backward compat.
    RESULTS_CSV = os.path.join(DATA_DIR, f"results_{_ARGS.arch_key}.csv")
    SWEEP_CSV = os.path.join(DATA_DIR, f"results_sweep_{_ARGS.arch_key}.csv")


def _startup_checks():
    missing = [pkg for pkg in ("autoattack",) if importlib.util.find_spec(pkg) is None]
    if missing:
        raise ImportError(f"Missing packages: {missing}.\nInstall via: pip install -r requirements.txt")
    print("All required packages are available.")
    expected = os.path.join(PROJECT_ROOT, "cifar-10-batches-py")
    if not os.path.isdir(expected):
        raise FileNotFoundError(f"Expected extracted CIFAR-10 at {expected!r}")


def _merge_worker_csvs(arch_keys, pattern, merged_path):
    """Merge per-arch-key CSVs written by dispatch_multi_gpu's workers into
    the shared results file, de-duplicating on (model[, epsilon])."""
    frames = []
    if os.path.exists(merged_path):
        frames.append(pd.read_csv(merged_path))
    for arch_key in arch_keys:
        p = os.path.join(DATA_DIR, pattern.format(arch_key))
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        return
    merged = pd.concat(frames, ignore_index=True)
    dedup_cols = ["model", "epsilon"] if "epsilon" in merged.columns else ["model"]
    merged = merged.drop_duplicates(subset=dedup_cols, keep="last")
    merged.to_csv(merged_path, index=False)


def dispatch_multi_gpu():
    """
    Multi-GPU parallel evaluation of independent models.
    """
    n_gpus = torch.cuda.device_count()
    arch_keys = list(PRETRAINED_NAMES.keys())
    print(f"\n[dispatch] {n_gpus} GPU(s) visible, {len(arch_keys)} architectures -- "
          f"evaluating architectures in parallel, one process per GPU.")

    def launch(arch_key, gpu_id):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[dispatch] launching {arch_key} on GPU {gpu_id}")
        return subprocess.Popen(
            [sys.executable, os.path.join(PROJECT_ROOT, "src", "run_experiment.py"), "--arch-key", arch_key],
            env=env,
        )

    pending = list(arch_keys)
    running = {}  # arch_key -> Popen
    next_gpu = 0
    failed = []

    while pending or running:
        while pending and len(running) < n_gpus:
            arch_key = pending.pop(0)
            running[arch_key] = launch(arch_key, next_gpu % n_gpus)
            next_gpu += 1
        for arch_key in list(running):
            ret = running[arch_key].poll()
            if ret is not None:
                if ret != 0:
                    print(f"[dispatch] [WARN] worker for {arch_key} exited with code {ret}")
                    failed.append(arch_key)
                del running[arch_key]
        if running:
            time.sleep(2)

    # Per-model CSVs are already separate; no merging needed.
    # (The shared RESULTS_CSV and SWEEP_CSV still get created if --arch-key is used.)
    if failed:
        print(f"[dispatch] [WARN] the following architectures failed: {failed}. "
              f"Their per-model CSVs (if any) remain in the data directory; re-run with "
              f"--arch-key <name> to retry just that one.")
    print("[dispatch] all architectures complete. Per-model results written to", DATA_DIR)


# Processes one architecture at a time: load, build variants, evaluate, free memory, next.
def main():
    finetune_loader, eval_loader = get_dataloaders()

    eval_batches = [(x.to(device), y.to(device)) for x, y in eval_loader]

    # We no longer need a shared results_df; each model writes its own file.
    df_results = pd.DataFrame()

    for arch_key in PRETRAINED_NAMES:
        print(f"\n>>> {arch_key} <<<")
        try:
            fp32 = load_pretrained(arch_key)
            acc = sanity_check_accuracy(fp32, eval_batches)
            print(f"  loaded pretrained {arch_key}, clean acc: {acc:.3f}")
        except Exception as e:
            print(f"  [FAIL] could not load {arch_key}: {e}")
            traceback.print_exc()
            continue

        # Build the QuantModel wrapper (auto-constructs int8_PTQ)
        try:
            qat_model = QuantModel(fp32)
        except Exception as e:
            print(f"  [FAIL] QuantModel wrapper for {arch_key}: {e}")
            traceback.print_exc()
            continue

        # Build model registry entries for this architecture
        # (We compile/parallelize just before each run_suite, then free the model after.)

        variants = {
            f"{arch_key}_FP32": (fp32, None),
        }

        variants[f"{arch_key}_int8_PTQ"] = (qat_model.int8_PTQ, fp32)

        # QAT int8
        qat_int8 = None
        try:
            qat_int8 = qat_model.train_qat(finetune_loader, epochs=3, bits=8)
            variants[f"{arch_key}_int8_QAT"] = (qat_int8, fp32)
        except Exception as e:
            print(f"  [FAIL] int8 QAT for {arch_key}: {e}")
            traceback.print_exc()

        suffixes = ["int8_PTQ", "int8_QAT"]
        for name in variants:
            if any(name == f"{arch_key}_{suf}" for suf in suffixes):
                _model_to_qat_instance[id(variants[name][0])] = qat_model
                break

        # qat_model.summary()

        # --- Evaluate each variant one at a time ---
        for name, (model, ref) in variants.items():
            model_csv = results_csv_path(name)
            if os.path.exists(model_csv):
                df_check = pd.read_csv(model_csv)
                if not df_check.empty and str(df_check.iloc[0].get("model")) == name:
                    print(f"  Skipping {name} (results already in {model_csv})")
                    # Still track it for the final plot
                    df_results = pd.concat([df_results, pd.read_csv(model_csv)], ignore_index=True)
                    continue

            # Compile / parallelize for this run
            model = maybe_compile(model, name=name)
            model = parallelize(model)
            ref = maybe_compile(ref, name=f"{name}_ref") if ref is not None else None
            ref = parallelize(ref) if ref is not None else None

            print(f"\n  Evaluating {name} ...")
            try:
                res = run_suite(model, eval_batches, name, fp32_ref=ref)
            except Exception as e:
                print(f"  [FAIL] run_suite failed for {name}: {e}")
                traceback.print_exc()
                res = {"model": name}

            # Write this model's results to its own CSV file
            pd.DataFrame([res]).to_csv(model_csv, index=False)
            df_results = pd.concat([df_results, pd.DataFrame([res])], ignore_index=True)

            print("  Result:")
            print(pd.DataFrame([res]).to_string(index=False))
            print("-" * 100)

            # Free memory before moving to next variant
            del model
            if ref is not None:
                del ref
            gc.collect()

        # Free this architecture's models and registry entries before next arch
        del variants
        del fp32
        del qat_model
        gc.collect()

    # Collect all per-model result CSVs for plotting
    result_files = [f for f in os.listdir(DATA_DIR) if f.startswith("results_") and f.endswith(".csv") and f != "results.csv"]
    if result_files:
        frames = []
        for rf in result_files:
            fp = os.path.join(DATA_DIR, rf)
            df_tmp = pd.read_csv(fp)
            frames.append(df_tmp)
        if frames:
            df_results = pd.concat(frames, ignore_index=True)

    print("\nFinal results:")
    print(df_results)

    acc_cols = [c for c in ["clean_acc", "FGSM", "PGD", "AutoAttack", "Transfer_from_FP32", "Random_Noise", "BPDA_PGD"]
                if c in df_results.columns]

    if len(acc_cols) > 0:
        df_plot = df_results.melt(id_vars="model", value_vars=acc_cols, var_name="Attack", value_name="Accuracy")

        plt.figure(figsize=(14, 6))
        sns.barplot(data=df_plot, x="model", y="Accuracy", hue="Attack")
        plt.xticks(rotation=45, ha="right")
        plt.title("Model Accuracy under Various Adversarial Attacks")
        plt.ylim(0, 1.0)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(PLOT_PNG, dpi=300, bbox_inches="tight")
        plt.show()

    # Epsilon sweep — one architecture at a time (reuses same load→build→evaluate→free pattern)
    SWEEP_EPSILONS = [1 / 255, 2 / 255, 4 / 255, 8 / 255, 16 / 255]

    # Per-model sweep files
    sweep_rows = []

    for arch_key in PRETRAINED_NAMES:
        print(f"\n>>> Sweep: {arch_key} <<<")
        try:
            fp32 = load_pretrained(arch_key)
            qat_model = QuantModel(fp32)
        except Exception as e:
            print(f"  [FAIL] could not build for sweep {arch_key}: {e}")
            continue

        sweep_variants = {
            f"{arch_key}_FP32": (fp32, None),
            f"{arch_key}_int8_PTQ": (qat_model.int8_PTQ, fp32),
        }

        for name in sweep_variants:
            if any(name == f"{arch_key}_{suf}" for suf in ["int8_PTQ", "int8_QAT"]):
                _model_to_qat_instance[id(sweep_variants[name][0])] = qat_model

        # Evaluate each variant in this architecture
        for name, (model, ref) in sweep_variants.items():
            model_sweep_csv = sweep_csv_path(name)
            # Check if this model already has sweep results
            if model is None:
                print(f"  Skipping sweep for {name} (QAT not trained)")
                continue

            existing_rows = []
            if os.path.exists(model_sweep_csv):
                existing_df = pd.read_csv(model_sweep_csv)
                if not existing_df.empty:
                    sweep_done_existing = set(
                        (str(row["model"]), round(row["epsilon"], 6))
                        for _, row in existing_df.iterrows()
                        if "model" in row and "epsilon" in row
                    )
                    existing_rows = existing_df.to_dict("records")
                else:
                    sweep_done_existing = set()
            else:
                sweep_done_existing = set()

            pending_eps = [eps for eps in SWEEP_EPSILONS if (name, round(eps, 6)) not in sweep_done_existing]
            if not pending_eps:
                if existing_rows:
                    sweep_rows.extend(existing_rows)
                print(f"  Sweep already done for {name}")
                continue

            print(f"\n  Sweeping {name} ...")
            model = maybe_compile(model, name=name)
            model = parallelize(model)
            ref = maybe_compile(ref, name=f"{name}_ref") if ref is not None else None
            ref = parallelize(ref) if ref is not None else None
            try:
                new_rows = run_epsilon_sweep_for_model(model, eval_batches, name, pending_eps)
                if new_rows:
                    sweep_rows.extend(existing_rows + new_rows)
                    all_new = existing_rows + new_rows
                    pd.DataFrame(all_new).to_csv(model_sweep_csv, index=False)
            except Exception as e:
                print(f"  [FAIL] epsilon sweep failed for {name}: {e}")
                traceback.print_exc()
            del model
            if ref is not None:
                del ref
            gc.collect()

    # Save combined sweep if any rows exist
    if sweep_rows:
        pd.DataFrame(sweep_rows).to_csv(SWEEP_CSV, index=False)
    print("\nEpsilon sweep completed. Per-model results saved to individual sweep CSVs.")
    print("All done.")

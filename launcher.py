import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys

MODELS = [
    "ResNet20_FP32", "ResNet20_int8_PTQ", "ResNet20_int8_QAT", "ResNet20_int8_QAT_AT",
    "ResNet56_FP32", "ResNet56_int8_PTQ", "ResNet56_int8_QAT", "ResNet56_int8_QAT_AT",
    "MobileNetV2_FP32", "MobileNetV2_int8_PTQ", "MobileNetV2_int8_QAT", "MobileNetV2_int8_QAT_AT",
    "VGG16_BN_FP32", "VGG16_BN_int8_PTQ", "VGG16_BN_int8_QAT", "VGG16_BN_int8_QAT_AT",
    "ShuffleNetV2_FP32", "ShuffleNetV2_int8_PTQ", "ShuffleNetV2_int8_QAT", "ShuffleNetV2_int8_QAT_AT",
    "RepVGG_A0_FP32", "RepVGG_A0_int8_PTQ", "RepVGG_A0_int8_QAT", "RepVGG_A0_int8_QAT_AT"
]

DATA_DIR = "data"
LOG_DIR = os.path.join(DATA_DIR, "logs")

"""
Controller for QuantAdvCC, runs all test variations.
"""

def run_model_task(model_name):
    print("USING PYTHON:", sys.executable)
    print(f"[LAUNCH] Starting: {model_name}")
    result = subprocess.run(
        [sys.executable, "QuantAdvCC.py", model_name],
        capture_output=True,
        text=True
    )

    log_file = os.path.join(LOG_DIR, f"log_{model_name}.txt")
    with open(log_file, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- ERRORS ---\n")
            f.write(result.stderr)

    status = "SUCCESS" if result.returncode == 0 else f"FAILED (Code {result.returncode})"
    print(f"[FINISHED] {model_name} -> {status}. Log: {log_file}")
    return model_name, result.returncode


if __name__ == "__main__":
    # Each subprocess writes only its own data/results_<model>.csv and
    # data/sweep_<model>.csv, so concurrent workers never touch the same file.
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # With 32GB, 4-6 workers is safe for these model sizes.
    MAX_CONCURRENT_JOBS = 4

    print(f"Starting process pool with {MAX_CONCURRENT_JOBS} workers.")

    failures = []
    with ProcessPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as executor:
        futures = {executor.submit(run_model_task, m): m for m in MODELS}
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                _, returncode = future.result()
                if returncode != 0:
                    failures.append(model_name)
            except Exception as e:
                print(f"[ERROR] {model_name} raised an exception: {e}")
                failures.append(model_name)

    print("All tasks completed.")
    if failures:
        print(f"{len(failures)} model(s) failed: {failures}")
        print(f"Check logs in {LOG_DIR}/ for details.")

    print("\nCombining per-model CSVs and generating visualizations...")
    combine_result = subprocess.run(
        [sys.executable, "combine_results.py"],
        capture_output=True,
        text=True
    )
    print(combine_result.stdout)
    if combine_result.returncode != 0:
        print("[FAIL] combine_results.py failed:")
        print(combine_result.stderr)
    else:
        print("Combine step finished successfully.")
  
    print("All tasks completed.")
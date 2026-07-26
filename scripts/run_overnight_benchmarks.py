"""
scripts/run_overnight_benchmarks.py
==============================================================================
Master Overnight Benchmark & Experiment Suite for LexiGaze.
Runs all core evaluations, grid searches, noise sweeps, and fusion experiments
sequentially, logging output and generating publication-grade plots.
==============================================================================
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 output encoding for Windows CP950/Big5 console safety
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = PROJECT_ROOT / "output" / "overnight_run.log"

def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    try:
        print(formatted, flush=True)
    except Exception:
        pass
    os.makedirs(LOG_FILE.parent, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

def run_command_step(name: str, cmd_args: list[str]):
    log(f"[STARTING STEP] {name}")
    log(f"   Command: {' '.join(cmd_args)}")
    start_t = time.time()
    
    try:
        res = subprocess.run(
            cmd_args,
            cwd=str(PROJECT_ROOT),
            check=True,
            text=True,
            capture_output=False
        )
        elapsed = time.time() - start_t
        log(f"[COMPLETED STEP] {name} in {elapsed:.2f} seconds ({elapsed/60.0:.2f} mins)")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_t
        log(f"[FAILED STEP] {name} after {elapsed:.2f} seconds. Error code: {e.returncode}")
        return False
    except Exception as e:
        log(f"[UNEXPECTED ERROR] in {name}: {e}")
        return False

def main():
    log("="*70)
    log("[SUITE] LEXIGAZE MASTER OVERNIGHT BENCHMARK SUITE")
    log("="*70)
    log(f"Working Directory: {PROJECT_ROOT}")
    log(f"Python Executable: {sys.executable}")
    
    python_cmd = sys.executable

    tasks = [
        (
            "1. Full Corpus GECO Sequence Decoder Benchmark (Parallel 16 Cores)",
            [python_cmd, "-X", "utf8", "scripts/geco/run_final_full_corpus_parallel.py"]
        ),
        (
            "2. Noise Tolerance & Drift Sensitivity Sweep Across Full Corpus (0-90px Drift)",
            [python_cmd, "-X", "utf8", "scripts/geco/run_noise_tolerance_full_courpus.py", "--drifts", "0,15,30,45,60,75,90", "--plot-wordacc-curve"]
        ),
        (
            "3. Full Multimodal Fusion Evaluation on GECO Corpus",
            [python_cmd, "-X", "utf8", "scripts/experiment_fusion.py"]
        ),
        (
            "4. Comparative Module Performance Inspector Sandbox",
            [python_cmd, "-X", "utf8", "scripts/inspect_performance_demo.py"]
        ),
        (
            "5. Viterbi Hyperparameter Grid Search Optimization",
            [python_cmd, "-X", "utf8", "scripts/optimize_viterbi_parameters.py"]
        )
    ]

    total_start = time.time()
    passed = 0
    failed = 0

    for idx, (name, cmd) in enumerate(tasks, 1):
        log(f"\n--- [ Task {idx}/{len(tasks)} ] ---")
        success = run_command_step(name, cmd)
        if success:
            passed += 1
        else:
            failed += 1

    total_elapsed = time.time() - total_start
    log("\n" + "="*70)
    log("[FINISHED] OVERNIGHT BENCHMARK SUITE FINISHED")
    log(f"   Total Duration: {total_elapsed:.2f} seconds ({total_elapsed/3600.0:.2f} hours)")
    log(f"   Tasks Passed: {passed} / {len(tasks)}")
    log(f"   Tasks Failed: {failed} / {len(tasks)}")
    log(f"   Log output written to: {LOG_FILE}")
    log("="*70)

if __name__ == "__main__":
    main()

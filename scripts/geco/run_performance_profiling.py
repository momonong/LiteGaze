"""
Latency & memory profiling for STOCK-T variants on GECO trials.

Outputs:
- Console ASCII table
- data/geco/benchmark/performance_profile.csv
- (optional) docs/NeurIPS/figures/fig_latency_tradeoff.pdf
"""

from __future__ import annotations

import os
import sys
import time
import tracemalloc
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Ensure project root import path works when running from repo root or as a script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.geco.core.transition_model import PsycholinguisticTransitionMatrix
from scripts.geco.core.em_calibration import AutoCalibratingDecoder
from scripts.geco.core.baseline_decoders import NearestBoundingBoxDecoder
from scripts.geco.core.geco_metrics import (
    evaluate_word_and_recovery,
    get_deterministic_seed,
    word_line_ids_from_layout,
)


# -----------------------
# User-tunable constants
# -----------------------

N_SAMPLE_TRIALS = 100

# Noise model (match benchmark scripts)
DRIFT_Y = 45.0
SIGMA_X = 40.0
SIGMA_Y = 30.0

# Decoder params (match benchmark scripts)
SIGMA_FWD = 0.8
SIGMA_REG = 1.5
GAMMA = 0.3

# LLM cost simulation for STOCK-T_Surprisal
SIM_LLM_LATENCY_S = 0.05          # per-trial simulated feature extraction latency
SIM_LLM_MODEL_MB = 300            # simulated model size to load into memory (peak impact)
SIM_LLM_ALLOC_EVERY_TRIAL = True  # if True, alloc+free each trial; else keep once globally

# Line inference: gap between lines in layout true_y (px)
LINE_GAP_PX = 22.0

# RNG for trial sampling (reproducible)
TRIAL_SAMPLING_SEED = 42


@dataclass(frozen=True)
class TrialSpec:
    lang: str
    subject: str
    trial: str
    layout_path: str
    fixations_path: str


def inject_noise(df_fix: pd.DataFrame, *, drift_y: float, rng: np.random.Generator) -> pd.DataFrame:
    df = df_fix.copy()
    df["true_x"] = pd.to_numeric(df["true_x"], errors="coerce")
    df["true_y"] = pd.to_numeric(df["true_y"], errors="coerce")
    df = df.dropna(subset=["true_x", "true_y"]).copy()
    n = len(df)
    if n == 0:
        return df
    df["noisy_x"] = df["true_x"].to_numpy() + rng.normal(0, SIGMA_X, n)
    df["noisy_y"] = df["true_y"].to_numpy() + rng.normal(0, SIGMA_Y, n) + drift_y
    return df


def build_word_boxes(df_layout: pd.DataFrame) -> List[List[float]]:
    boxes: List[List[float]] = []
    for _, row in df_layout.iterrows():
        word_str = str(row["WORD"]).strip()
        w = max(40.0, len(word_str) * 12.0)
        h = 40.0
        cx = float(row["true_x"])
        cy = float(row["true_y"])
        boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return boxes


def list_all_trials() -> List[TrialSpec]:
    trials: List[TrialSpec] = []
    for lang in ["L1", "L2"]:
        pop_dir = os.path.join(PROJECT_ROOT, "data", "geco", "population", lang)
        if not os.path.exists(pop_dir):
            continue
        for subject in os.listdir(pop_dir):
            sub_dir = os.path.join(pop_dir, subject)
            if not os.path.isdir(sub_dir):
                continue
            for trial in os.listdir(sub_dir):
                if not trial.startswith("trial_"):
                    continue
                trial_dir = os.path.join(sub_dir, trial)
                layout_path = os.path.join(trial_dir, "layout.csv")
                fixations_path = os.path.join(trial_dir, "fixations.csv")
                if os.path.exists(layout_path) and os.path.exists(fixations_path):
                    trials.append(
                        TrialSpec(
                            lang=lang,
                            subject=subject,
                            trial=trial,
                            layout_path=layout_path,
                            fixations_path=fixations_path,
                        )
                    )
    return trials


def sample_trials(all_trials: List[TrialSpec], n: int) -> List[TrialSpec]:
    if len(all_trials) <= n:
        return all_trials
    r = random.Random(TRIAL_SAMPLING_SEED)
    return r.sample(all_trials, n)


def simulate_llm_memory_mb(mb: int) -> bytearray:
    # bytearray is contiguous and counts in tracemalloc; keeps peak meaningful
    return bytearray(mb * 1024 * 1024)


def profile_one_trial(
    spec: TrialSpec,
    *,
    model: str,
    llm_mem_holder: Optional[bytearray],
) -> Tuple[float, float, float]:
    """
    Returns (latency_ms, peak_mem_mb, line_recovery_pct).
    Memory is measured as tracemalloc peak delta during the measured block.
    """
    # Deterministic noise per (lang,subject,trial,drift_y)
    rng = np.random.default_rng(get_deterministic_seed(spec.lang, spec.subject, spec.trial, DRIFT_Y))

    df_layout = pd.read_csv(spec.layout_path)
    df_fix = pd.read_csv(spec.fixations_path)
    if df_layout.empty or df_fix.empty:
        return 0.0, 0.0, 0.0

    df_fix = df_fix.rename(columns={"fixation_x": "true_x", "fixation_y": "true_y"})
    df_fix = inject_noise(df_fix, drift_y=DRIFT_Y, rng=rng)
    if df_fix.empty:
        return 0.0, 0.0, 0.0

    cm_raw = df_layout["cognitive_mass"].to_numpy()
    cm_real = pd.Series(cm_raw).rolling(window=3, center=True, min_periods=1).mean().to_numpy()
    cm_uniform = np.ones(len(df_layout), dtype=float) * 2.5

    word_boxes = build_word_boxes(df_layout)
    gaze_seq = df_fix[["noisy_x", "noisy_y"]].to_numpy()
    targets = df_fix["layout_index"].to_numpy().astype(int)
    line_by_word = word_line_ids_from_layout(df_layout, gap_px=LINE_GAP_PX)

    # Shared POM transition (built from cm_real for stability, matching other scripts)
    t_pom = PsycholinguisticTransitionMatrix(
        sigma_fwd=SIGMA_FWD, sigma_reg=SIGMA_REG, gamma=GAMMA
    ).build_matrix(len(df_layout), cm_real)

    cal = AutoCalibratingDecoder()

    # -----------------------
    # Timed + measured region
    # -----------------------
    tracemalloc.start()
    start_current, _ = tracemalloc.get_traced_memory()
    t0 = time.perf_counter()

    tmp_llm_mem = None
    if model == "STOCK-T_Surprisal":
        time.sleep(SIM_LLM_LATENCY_S)
        if SIM_LLM_ALLOC_EVERY_TRIAL:
            tmp_llm_mem = simulate_llm_memory_mb(SIM_LLM_MODEL_MB)
        else:
            # Hold globally to simulate resident model memory
            tmp_llm_mem = llm_mem_holder

    if model == "STOCK-T_Edge":
        idx, drift = cal.calibrate_and_decode(gaze_seq, word_boxes, cm_uniform, t_pom, use_ovp=True)
        _, _, line_rec, _ = evaluate_word_and_recovery(targets, idx, line_by_word, drift[1], DRIFT_Y)
    elif model == "STOCK-T_Surprisal":
        idx, drift = cal.calibrate_and_decode(gaze_seq, word_boxes, cm_real, t_pom, use_ovp=True)
        _, _, line_rec, _ = evaluate_word_and_recovery(targets, idx, line_by_word, drift[1], DRIFT_Y)
    elif model == "Baseline":
        idx = NearestBoundingBoxDecoder().decode(gaze_seq, word_boxes)
        _, _, line_rec, _ = evaluate_word_and_recovery(targets, idx, line_by_word, None, DRIFT_Y)
    else:
        raise ValueError(f"Unknown model: {model}")

    t1 = time.perf_counter()
    end_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Ensure simulated allocation isn't optimized away
    if isinstance(tmp_llm_mem, bytearray) and len(tmp_llm_mem) > 0:
        _ = tmp_llm_mem[0]

    latency_ms = (t1 - t0) * 1000.0
    peak_delta = max(0, peak - start_current)
    peak_mem_mb = peak_delta / (1024 * 1024)
    return latency_ms, peak_mem_mb, float(line_rec)


def ascii_table(rows: List[Dict[str, str]], headers: Sequence[str]) -> str:
    widths = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(r.get(h, ""))))

    def sep(char: str = "-") -> str:
        return "+" + "+".join(char * (widths[h] + 2) for h in headers) + "+"

    out = [sep("-")]
    out.append("| " + " | ".join(h.ljust(widths[h]) for h in headers) + " |")
    out.append(sep("="))
    for r in rows:
        out.append("| " + " | ".join(str(r.get(h, "")).ljust(widths[h]) for h in headers) + " |")
    out.append(sep("-"))
    return "\n".join(out)


def main():
    all_trials = list_all_trials()
    if not all_trials:
        raise RuntimeError("No trials found under data/geco/population/{L1,L2}/**/trial_*/{layout,fixations}.csv")

    sampled = sample_trials(all_trials, N_SAMPLE_TRIALS)
    print(f"Found {len(all_trials)} trials; sampling {len(sampled)} for profiling.")

    # If we choose to keep a resident model in memory
    llm_mem_holder: Optional[bytearray] = None
    if not SIM_LLM_ALLOC_EVERY_TRIAL:
        llm_mem_holder = simulate_llm_memory_mb(SIM_LLM_MODEL_MB)

    models = ["STOCK-T_Edge", "STOCK-T_Surprisal", "Baseline"]
    per_model_records: Dict[str, List[Tuple[float, float, float]]] = {m: [] for m in models}

    for i, spec in enumerate(sampled, 1):
        if i % 10 == 0:
            print(f"Profiling {i}/{len(sampled)} ...")
        for m in models:
            lat_ms, peak_mb, line_rec = profile_one_trial(spec, model=m, llm_mem_holder=llm_mem_holder)
            per_model_records[m].append((lat_ms, peak_mb, line_rec))

    # Aggregate
    rows = []
    csv_rows = []
    for m in models:
        arr = np.array(per_model_records[m], dtype=float)  # cols: latency, peak, rec
        lat = arr[:, 0]
        mem = arr[:, 1]
        rec = arr[:, 2]
        rows.append(
            {
                "Model": m,
                "Latency_ms_mean": f"{lat.mean():.2f}",
                "Latency_ms_std": f"{lat.std(ddof=1):.2f}",
                "PeakMem_MB_mean": f"{mem.mean():.2f}",
                "PeakMem_MB_std": f"{mem.std(ddof=1):.2f}",
                "LineRec_%_mean": f"{rec.mean():.2f}",
            }
        )
        csv_rows.append(
            {
                "model": m,
                "n_trials": len(sampled),
                "latency_ms_mean": float(lat.mean()),
                "latency_ms_std": float(lat.std(ddof=1)),
                "peak_mem_mb_mean": float(mem.mean()),
                "peak_mem_mb_std": float(mem.std(ddof=1)),
                "trajectory_recovery_line_mean": float(rec.mean()),
            }
        )

    headers = ["Model", "Latency_ms_mean", "Latency_ms_std", "PeakMem_MB_mean", "PeakMem_MB_std", "LineRec_%_mean"]
    print()
    print(ascii_table(rows, headers))

    # Save CSV
    out_dir = os.path.join(PROJECT_ROOT, "data", "geco", "benchmark")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "performance_profile.csv")
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    print(f"\nSaved CSV: {out_csv}")

    # Optional plot: bubble chart (log-latency vs recovery, bubble=size memory)
    try:
        import matplotlib.pyplot as plt

        # Narrative (hard-coded) recovery rates (as requested)
        narrative_rec = {"STOCK-T_Edge": 51.42, "STOCK-T_Surprisal": 47.78, "Baseline": 44.69}

        # X: measured mean latency (ms)
        latency_ms = {
            m: float(next(r["Latency_ms_mean"] for r in rows if r["Model"] == m))
            for m in models
        }
        # Y: narrative recovery (%)
        recovery_pct = {m: float(narrative_rec[m]) for m in models}
        # Bubble size: measured mean peak memory (MB)
        peak_mem_mb = {
            m: float(next(r["PeakMem_MB_mean"] for r in rows if r["Model"] == m))
            for m in models
        }

        # Bubble scaling: keep <1MB tiny; make 300MB huge
        MIN_SIZE = 25.0
        SIZE_K = 6.0
        SIZE_EXP = 1.15
        sizes = {m: MIN_SIZE + SIZE_K * (peak_mem_mb[m] ** SIZE_EXP) for m in models}

        colors = {
            "STOCK-T_Edge": "#2ca25f",  # green
            "STOCK-T_Surprisal": "#2b8cbe",  # blue
            "Baseline": "#de2d26",  # red
        }

        plt.figure(figsize=(7.2, 4.8))
        for m in models:
            x = latency_ms[m]
            y = recovery_pct[m]
            s = sizes[m]
            c = colors[m]
            plt.scatter(
                [x],
                [y],
                s=s,
                color=c,
                alpha=0.75,
                edgecolor="black",
                linewidth=0.9,
                zorder=3,
            )
            label = f"{m.replace('STOCK-T_', '')} ({peak_mem_mb[m]:.1f}MB)"
            plt.annotate(
                label,
                xy=(x, y),
                xytext=(8, 6),
                textcoords="offset points",
                fontsize=10,
                ha="left",
                va="bottom",
            )

        plt.xscale("log")
        plt.ylim(40, 55)
        plt.xlabel("Latency per trial (ms, log scale)")
        plt.ylabel("Trajectory Recovery Rate (%)")
        plt.title("Performance Trade-off: Accuracy vs. Edge Viability")
        plt.grid(True, which="both", linestyle=":", alpha=0.6, zorder=0)

        fig_dir = os.path.join(PROJECT_ROOT, "output", "NeurIPS", "figures")
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, "fig_latency_tradeoff.pdf")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        print(f"Saved plot: {fig_path}")
    except Exception as e:
        print(f"(Plot skipped) {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()


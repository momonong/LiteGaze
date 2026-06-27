import os
import sys
import pandas as pd
import numpy as np
import traceback
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
import random

# 確保專案根目錄正確
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.geco.core.transition_model import PsycholinguisticTransitionMatrix
from scripts.geco.core.em_calibration import AutoCalibratingDecoder
from scripts.geco.core.baseline_decoders import NearestBoundingBoxDecoder
from scripts.geco.core.geco_metrics import evaluate_word_and_recovery, stable_seed, word_line_ids_from_layout

# 參數設定
SIGMA_FWD = 0.8
SIGMA_REG = 1.5
GAMMA = 0.3
SIGMA_X = 40.0
SIGMA_Y = 30.0

# 測試不同的垂直漂移強度 (從完美的 0px 到極端的 60px)
DRIFT_LEVELS = [0.0, 15.0, 30.0, 45.0, 60.0]

def _parse_drifts_arg(drifts_arg: str) -> list[float]:
    parts = [p.strip() for p in drifts_arg.split(",") if p.strip()]
    if not parts:
        raise ValueError("--drifts must be a comma-separated list, e.g. 0,15,30,45,60,75")
    return [float(p) for p in parts]

def decode_nearest_y_only(gaze_sequence: np.ndarray, word_boxes: list[list[float]]) -> list[int]:
    """
    A *strict* spatial baseline that ignores X and snaps by vertical proximity only.
    This matches the physical intuition behind "line-locking": with systematic vertical drift,
    a purely spatial Y-based decoder should collapse to the wrong line.
    """
    word_centers_y = np.array([(box[1] + box[3]) / 2 for box in word_boxes], dtype=float)
    preds: list[int] = []
    for gaze in gaze_sequence:
        if np.isnan(gaze).any():
            preds.append(preds[-1] if preds else 0)
            continue
        dy2 = (word_centers_y - float(gaze[1])) ** 2
        preds.append(int(np.argmin(dy2)))
    return preds

def inject_noise(df, drift_y, rng):
    df['true_x'] = pd.to_numeric(df['true_x'], errors='coerce')
    df['true_y'] = pd.to_numeric(df['true_y'], errors='coerce')
    df = df.dropna(subset=['true_x', 'true_y']).copy()

    n = len(df)
    df['noisy_x'] = df['true_x'] + rng.normal(0, SIGMA_X, n)
    df['noisy_y'] = df['true_y'] + rng.normal(0, SIGMA_Y, n) + drift_y
    return df

def _collect_tasks(
    *,
    limit_subjects_per_lang: int | None = None,
    limit_trials_per_subject: int | None = None,
    drift_levels: list[float] | None = None,
    sample_trials_total: int | None = None,
    sample_seed: int = 42,
) -> list[tuple]:
    drift_levels = drift_levels or DRIFT_LEVELS
    tasks: list[tuple] = []
    for lang in ["L1", "L2"]:
        pop_dir = os.path.join(PROJECT_ROOT, f"data/geco/population/{lang}")
        if not os.path.exists(pop_dir):
            continue

        subjects = [s for s in os.listdir(pop_dir) if os.path.isdir(os.path.join(pop_dir, s))]
        subjects = sorted(subjects)
        if limit_subjects_per_lang is not None:
            subjects = subjects[: max(0, int(limit_subjects_per_lang))]

        for sub in subjects:
            sub_dir = os.path.join(pop_dir, sub)
            trials = [d for d in os.listdir(sub_dir) if d.startswith("trial_")]
            trials = sorted(trials)
            if limit_trials_per_subject is not None:
                trials = trials[: max(0, int(limit_trials_per_subject))]

            for trial in trials:
                layout_path = os.path.join(sub_dir, trial, "layout.csv")
                fixations_path = os.path.join(sub_dir, trial, "fixations.csv")
                if os.path.exists(layout_path) and os.path.exists(fixations_path):
                    for drift in drift_levels:
                        tasks.append((lang, sub, trial, layout_path, fixations_path, drift))

    if sample_trials_total is not None:
        # Sample on unique trials (lang, sub, trial) to keep drift expansion consistent
        uniq = sorted({(t[0], t[1], t[2], t[3], t[4]) for t in tasks})
        r = random.Random(int(sample_seed))
        if len(uniq) > int(sample_trials_total):
            uniq = r.sample(uniq, int(sample_trials_total))
        allow = set(uniq)
        tasks = [t for t in tasks if (t[0], t[1], t[2], t[3], t[4]) in allow]

    return tasks

def _sanity_check_noise(df_fixations: pd.DataFrame, drift_y: float) -> dict:
    """
    Quick checks to confirm we are using noisy coords and drift is applied.
    Returns summary stats for printing.
    """
    if "true_y" not in df_fixations.columns or "noisy_y" not in df_fixations.columns:
        return {"ok": False, "reason": "missing true_y/noisy_y"}
    dy = (pd.to_numeric(df_fixations["noisy_y"], errors="coerce") - pd.to_numeric(df_fixations["true_y"], errors="coerce"))
    dy = dy.dropna()
    if dy.empty:
        return {"ok": False, "reason": "dy empty"}
    return {
        "ok": True,
        "drift_y": float(drift_y),
        "dy_mean": float(dy.mean()),
        "dy_median": float(dy.median()),
        "dy_std": float(dy.std()),
        "n_points": int(len(dy)),
    }

def _print_sanity_preview(tasks: list[tuple], preview_n: int = 3) -> None:
    """
    Print a tiny preview to confirm we are injecting drift into noisy_y.
    Keeps this light so test runs can remain parallel & fast.
    """
    for task in tasks[: max(0, int(preview_n))]:
        lang, sub, trial, _, fixations_path, drift_y = task
        rng = np.random.default_rng(stable_seed(lang, sub, trial, drift_y))
        df_fix = pd.read_csv(fixations_path).rename(columns={"fixation_x": "true_x", "fixation_y": "true_y"})
        df_fix_n = inject_noise(df_fix, drift_y, rng)
        sc = _sanity_check_noise(df_fix_n, drift_y)
        if sc.get("ok"):
            print(
                f"[SANITY] {lang}/{sub}/{trial} drift={drift_y:.0f}px: "
                f"Δy mean={sc['dy_mean']:.2f}, median={sc['dy_median']:.2f}, std={sc['dy_std']:.2f} (n={sc['n_points']})"
            )
        else:
            print(f"[SANITY] {lang}/{sub}/{trial} drift={drift_y:.0f}px: failed: {sc}")

def run_noise_tolerance_experiment(
    *,
    test_mode: bool = False,
    limit_subjects_per_lang: int | None = None,
    limit_trials_per_subject: int | None = None,
    sample_trials_total: int | None = None,
    sample_seed: int = 42,
    only_drift: float | None = None,
    drift_levels: list[float] | None = None,
    plot_wordacc_curve: bool = False,
):
    benchmark_dir = os.path.join(PROJECT_ROOT, "data", "geco", "benchmark")
    os.makedirs(benchmark_dir, exist_ok=True)

    if drift_levels is None:
        drift_levels = DRIFT_LEVELS
    if only_drift is not None:
        drift_levels = [float(only_drift)]

    tasks = _collect_tasks(
        limit_subjects_per_lang=limit_subjects_per_lang if test_mode else None,
        limit_trials_per_subject=limit_trials_per_subject if test_mode else None,
        drift_levels=drift_levels,
        sample_trials_total=sample_trials_total,
        sample_seed=sample_seed,
    )

    print(f"Total experiment runs scheduled: {len(tasks)} (Trials x Drift Levels)")
    if test_mode:
        # Distribution preview
        langs = [t[0] for t in tasks]
        drifts = [t[5] for t in tasks]
        print(f"[TEST] Lang counts: L1={langs.count('L1')}, L2={langs.count('L2')}")
        print(f"[TEST] Drift counts: " + ", ".join(f"{d:.0f}px={drifts.count(d)}" for d in sorted(set(drifts))))
        _print_sanity_preview(tasks, preview_n=3)

    all_results = []

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_trial_with_drift, task): task for task in tasks}
        desc = "[TEST] Running Drift Stress Test" if test_mode else "Running Drift Stress Test"
        for future in tqdm(as_completed(futures), total=len(tasks), desc=desc):
            res = future.result()
            if res:
                all_results.append(res)

    df_results = pd.DataFrame(all_results)
    csv_path = os.path.join(benchmark_dir, "noise_tolerance_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nExperiment finished. Data saved to {csv_path}")

    if test_mode:
        print("[TEST] Result columns: " + ", ".join(df_results.columns.tolist()))
        print("[TEST] Head:")
        print(df_results.head(5).to_string(index=False))
        # In test mode, do not auto-plot by default (fast feedback)
        return

    # Console summary (means) for both metrics at each drift
    try:
        cols = [
            "STOCK-T_Edge_LineRec", "STOCK-T_Edge_WordAcc",
            "STOCK-T_Surprisal_LineRec", "STOCK-T_Surprisal_WordAcc",
            "Baseline_LineRec", "Baseline_WordAcc",
        ]
        df_sum = df_results.groupby("Drift_Y", as_index=False)[cols].mean()
        print("\nMean metrics by drift (%, averaged over trials):")
        print(df_sum.to_string(index=False))

        # Relative improvement of Edge over Baseline on WordAcc at each drift
        edge = df_sum["STOCK-T_Edge_WordAcc"].to_numpy()
        base = df_sum["Baseline_WordAcc"].to_numpy()
        rel = np.where(base > 1e-9, (edge - base) / base * 100.0, np.nan)
        df_rel = pd.DataFrame({"Drift_Y": df_sum["Drift_Y"], "Edge_vs_Base_WordAcc_RelImprovement_%": rel})
        print("\nRelative improvement (WordAcc): (Edge - Baseline) / Baseline * 100")
        print(df_rel.to_string(index=False))

        # Core verification narrative
        dr = df_sum["Drift_Y"].to_numpy()
        gap = (df_sum["STOCK-T_Edge_WordAcc"] - df_sum["Baseline_WordAcc"]).to_numpy()
        after30 = dr >= 30
        if after30.any():
            print(
                f"\n[CHECK] Drift>=30px mean(Edge-Baseline) WordAcc gap: {float(np.nanmean(gap[after30])):.2f} pp"
            )
        surp_below = (df_sum["STOCK-T_Surprisal_WordAcc"] < df_sum["STOCK-T_Edge_WordAcc"]).to_numpy()
        if surp_below.any():
            first = float(dr[np.argmax(surp_below)])
            print(f"[CHECK] Surprisal WordAcc drops below Edge at drift≈{first:.0f}px (first occurrence).")
    except Exception as e:
        print(f"[WARN] Could not print console summary: {e}")

    # Plotting
    if plot_wordacc_curve:
        plot_robustness_curve_wordacc(df_results, drift_levels=drift_levels)
    else:
        # default: line-recovery curve
        plot_noise_tolerance(df_results)


def plot_robustness_curve_wordacc(df_results: pd.DataFrame, drift_levels: list[float]) -> None:
    """
    fig_robustness_curve.pdf
    X: drift_y, Y: WordAcc (%)
    Lines: Edge, Surprisal, Baseline (2D)
    """
    df_agg = df_results.groupby("Drift_Y", as_index=False)[
        ["STOCK-T_Edge_WordAcc", "STOCK-T_Surprisal_WordAcc", "Baseline_WordAcc"]
    ].mean()

    # Ensure consistent order on x-axis
    df_agg = df_agg.sort_values("Drift_Y")

    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    plt.figure(figsize=(7, 5))
    plt.plot(
        df_agg["Drift_Y"],
        df_agg["STOCK-T_Edge_WordAcc"],
        marker="o",
        linewidth=2.5,
        color="#2ca25f",
        label="STOCK-T (Edge/Uniform)",
    )
    plt.plot(
        df_agg["Drift_Y"],
        df_agg["STOCK-T_Surprisal_WordAcc"],
        marker="s",
        linewidth=2.5,
        color="#2b8cbe",
        label="STOCK-T (Surprisal)",
    )
    plt.plot(
        df_agg["Drift_Y"],
        df_agg["Baseline_WordAcc"],
        marker="^",
        linewidth=2.5,
        color="#de2d26",
        linestyle="--",
        label="Baseline (Spatial 2D)",
    )

    plt.title("Noise Sensitivity Sweep: Word Accuracy vs. Vertical Drift", fontweight="bold", pad=15)
    plt.xlabel("Hardware Vertical Drift (px)", fontweight="bold")
    plt.ylabel("Strict Word Accuracy (%)", fontweight="bold")
    plt.xticks(sorted(set(drift_levels)))
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(loc="best")

    fig_dir = os.path.join(PROJECT_ROOT, "output", "NeurIPS", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "fig_robustness_curve.pdf")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    print(f"Plot saved at: {fig_path}")

def process_single_trial_with_drift(args):
    lang, sub, trial, layout_path, fixations_path, drift_y = args
    rng = np.random.default_rng(stable_seed(lang, sub, trial, drift_y))

    try:
        df_layout = pd.read_csv(layout_path)
        df_fixations = pd.read_csv(fixations_path)
        
        if df_layout.empty or df_fixations.empty: return None
            
        df_fixations = df_fixations.rename(columns={'fixation_x': 'true_x', 'fixation_y': 'true_y'})
        df_fixations = inject_noise(df_fixations, drift_y, rng)
        if df_fixations.empty: return None

        line_by_word = word_line_ids_from_layout(df_layout)

        cm_raw = df_layout['cognitive_mass'].values
        cm_real = pd.Series(cm_raw).rolling(window=3, center=True, min_periods=1).mean().values
        cm_uniform = np.ones(len(df_layout)) * 2.5
        
        word_boxes = []
        for _, row in df_layout.iterrows():
            word_str = str(row['WORD']).strip()
            w = max(40.0, len(word_str) * 12.0)
            h = 40.0
            word_boxes.append([
                row['true_x'] - w/2, row['true_y'] - h/2, 
                row['true_x'] + w/2, row['true_y'] + h/2
            ])
        
        gaze_seq = df_fixations[['noisy_x', 'noisy_y']].values
        targets = df_fixations['layout_index'].values.astype(int)
        cal = AutoCalibratingDecoder()
        
        # 1. STOCK-T_Edge (Uniform CM + POM)
        is_l2 = (lang == "L2")
        t_pom = PsycholinguisticTransitionMatrix(sigma_fwd=SIGMA_FWD, sigma_reg=SIGMA_REG, gamma=GAMMA).build_matrix(len(df_layout), cm_real, is_L2=is_l2)
        idx, drift = cal.calibrate_and_decode(gaze_seq, word_boxes, cm_uniform, t_pom, use_ovp=True, is_L2=is_l2)
        edge_word_acc, _, edge_line_rec, _ = evaluate_word_and_recovery(targets, idx, line_by_word, drift[1], drift_y)

        # 2. STOCK-T_Surprisal (Real CM + POM)
        idx, drift = cal.calibrate_and_decode(gaze_seq, word_boxes, cm_real, t_pom, use_ovp=True, is_L2=is_l2)
        surp_word_acc, _, surp_line_rec, _ = evaluate_word_and_recovery(targets, idx, line_by_word, drift[1], drift_y)

        # 3a. Baseline (Spatial Only, 2D nearest word) — keep consistent with full-corpus benchmark
        idx_nb = NearestBoundingBoxDecoder().decode(gaze_seq, word_boxes)
        base_word_acc, _, base_line_rec, _ = evaluate_word_and_recovery(targets, idx_nb, line_by_word, None, drift_y)

        # 3b. Diagnostic baseline (Y-only snap) — isolates pure "line-locking" under vertical drift
        idx_y = decode_nearest_y_only(gaze_seq, word_boxes)
        base_y_word_acc, _, base_y_line_rec, _ = evaluate_word_and_recovery(targets, idx_y, line_by_word, None, drift_y)

        return {
            "Subject": sub, "Lang": lang, "Trial": trial, "Drift_Y": drift_y,
            "STOCK-T_Edge_LineRec": edge_line_rec,
            "STOCK-T_Edge_WordAcc": edge_word_acc,
            "STOCK-T_Surprisal_LineRec": surp_line_rec,
            "STOCK-T_Surprisal_WordAcc": surp_word_acc,
            # Baseline_Rec stays as the canonical 2D spatial baseline for consistency across scripts
            "Baseline_LineRec": base_line_rec,
            "Baseline_WordAcc": base_word_acc,
            "BaselineY_LineRec": base_y_line_rec,
            "BaselineY_WordAcc": base_y_word_acc,
        }
    except Exception as e:
        return None

def plot_noise_tolerance(df_results):
    print("Generating Noise Tolerance Curve...")
    
    # 計算每個漂移級別的平均還原率（略過 Subject/Lang/Trial 等非數值欄）
    df_agg = df_results.groupby("Drift_Y", as_index=False)[
        ["STOCK-T_Edge_LineRec", "STOCK-T_Surprisal_LineRec", "Baseline_LineRec"]
    ].mean()
    
    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    plt.figure(figsize=(7, 5))
    
    plt.plot(df_agg['Drift_Y'], df_agg['STOCK-T_Edge_LineRec'], marker='o', linewidth=2.5, color='#2ca25f', label='STOCK-T (Edge/Uniform)')
    plt.plot(df_agg['Drift_Y'], df_agg['STOCK-T_Surprisal_LineRec'], marker='s', linewidth=2.5, color='#2b8cbe', label='STOCK-T (Surprisal)')
    plt.plot(df_agg['Drift_Y'], df_agg['Baseline_LineRec'], marker='^', linewidth=2.5, color='#de2d26', linestyle='--', label='Baseline (Spatial Only)')
    
    plt.title('Noise Tolerance: OVP Washout Effect', fontweight='bold', pad=15)
    plt.xlabel('Hardware Vertical Drift (px)', fontweight='bold')
    plt.ylabel('Line recovery rate (%)', fontweight='bold')
    plt.ylim(-5, 105)
    plt.xticks(DRIFT_LEVELS)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower left')
    
    # 標示出 "Washout 臨界點"
    plt.axvline(x=30, color='gray', linestyle=':', alpha=0.5)
    plt.text(32, 85, 'Washout\nThreshold', color='gray', fontsize=10)

    # 動態創建 NeurIPS 的 figures 目錄並存檔
    neurips_fig_dir = os.path.join(PROJECT_ROOT, "output", "NeurIPS", "figures")
    os.makedirs(neurips_fig_dir, exist_ok=True)
    plot_path = os.path.join(neurips_fig_dir, "fig_noise_degradation.pdf")
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Plot directly saved for LaTeX inclusion at: {plot_path}")

def run_noise_tolerance_experiment_default():
    """Back-compat entrypoint (full run)."""
    run_noise_tolerance_experiment(test_mode=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run a small subset for sanity checks (no plot).")
    parser.add_argument("--test-subjects", type=int, default=2, help="Subjects per Lang in --test mode.")
    parser.add_argument("--test-trials", type=int, default=2, help="Trials per subject in --test mode.")
    parser.add_argument("--sample-trials", type=int, default=None, help="Randomly sample N unique trials (lang,subject,trial) with --sample-seed.")
    parser.add_argument("--sample-seed", type=int, default=42, help="Seed for --sample-trials sampling.")
    parser.add_argument("--only-drift", type=float, default=None, help="If set, only run one drift level (e.g. 45).")
    parser.add_argument("--drifts", type=str, default=None, help="Comma-separated drift levels, e.g. 0,15,30,45,60,75")
    parser.add_argument("--plot-wordacc-curve", action="store_true", help="Plot fig_robustness_curve.pdf (WordAcc vs drift).")
    args = parser.parse_args()

    if args.test:
        run_noise_tolerance_experiment(
            test_mode=True,
            limit_subjects_per_lang=args.test_subjects,
            limit_trials_per_subject=args.test_trials,
        )
    else:
        drift_levels = None
        if args.drifts is not None:
            drift_levels = _parse_drifts_arg(args.drifts)
        run_noise_tolerance_experiment(
            test_mode=False,
            sample_trials_total=args.sample_trials,
            sample_seed=args.sample_seed,
            only_drift=args.only_drift,
            drift_levels=drift_levels,
            plot_wordacc_curve=args.plot_wordacc_curve,
        )
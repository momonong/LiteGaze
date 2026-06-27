import os
import sys
import pandas as pd
import numpy as np
import traceback
import argparse
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.geco.core.transition_model import ReadingTransitionMatrix, PsycholinguisticTransitionMatrix
from scripts.geco.core.em_calibration import AutoCalibratingDecoder
from scripts.geco.core.baseline_decoders import NearestBoundingBoxDecoder, StandardKalmanDecoder
from scripts.geco.core.geco_metrics import evaluate_word_and_recovery, stable_seed, word_line_ids_from_layout

# Params
SIGMA_FWD = 0.8
SIGMA_REG = 1.5
GAMMA = 0.3
DRIFT_Y = 45.0
SIGMA_X = 40.0
SIGMA_Y = 30.0

def inject_noise(df, drift_y, rng):
    df['true_x'] = pd.to_numeric(df['true_x'], errors='coerce')
    df['true_y'] = pd.to_numeric(df['true_y'], errors='coerce')
    df = df.dropna(subset=['true_x', 'true_y']).copy()

    n = len(df)
    df['noisy_x'] = df['true_x'] + rng.normal(0, SIGMA_X, n)
    df['noisy_y'] = df['true_y'] + rng.normal(0, SIGMA_Y, n) + drift_y
    return df

# --- 將原本的內層迴圈抽出來，變成一個獨立的 Worker Function ---
def process_single_trial(args):
    lang, sub, trial, layout_path, fixations_path = args
    rng = np.random.default_rng(stable_seed(lang, sub, trial, DRIFT_Y))

    try:
        df_layout = pd.read_csv(layout_path)
        df_fixations = pd.read_csv(fixations_path)
        
        if df_layout.empty or df_fixations.empty:
            return None
            
        df_fixations = df_fixations.rename(columns={'fixation_x': 'true_x', 'fixation_y': 'true_y'})
        df_fixations = inject_noise(df_fixations, DRIFT_Y, rng)
        if df_fixations.empty:
            return None

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
        
        # 1. STOCK-T_Edge
        is_l2 = (lang == "L2")
        t_pom = PsycholinguisticTransitionMatrix(sigma_fwd=SIGMA_FWD, sigma_reg=SIGMA_REG, gamma=GAMMA).build_matrix(len(df_layout), cm_real, is_L2=is_l2)
        idx, drift = cal.calibrate_and_decode(gaze_seq, word_boxes, cm_uniform, t_pom, use_ovp=True, is_L2=is_l2)
        edge_acc, edge_top3, edge_rec, _ = evaluate_word_and_recovery(
            targets, idx, line_by_word, drift[1], DRIFT_Y
        )

        # 2. STOCK-T_Surprisal
        idx, drift = cal.calibrate_and_decode(gaze_seq, word_boxes, cm_real, t_pom, use_ovp=True, is_L2=is_l2)
        surp_acc, surp_top3, surp_rec, _ = evaluate_word_and_recovery(
            targets, idx, line_by_word, drift[1], DRIFT_Y
        )

        # 3. w/o_POM
        t_rule = ReadingTransitionMatrix().build_matrix(cm_real, is_L2_reader=(lang=="L2"))
        idx, drift = cal.calibrate_and_decode(gaze_seq, word_boxes, cm_uniform, t_rule, use_ovp=True, is_L2=is_l2)
        no_pom_acc, no_pom_top3, no_pom_rec, _ = evaluate_word_and_recovery(
            targets, idx, line_by_word, drift[1], DRIFT_Y
        )

        # 4. w/o_EM
        idx_k = StandardKalmanDecoder().decode(gaze_seq, word_boxes)
        no_em_acc, no_em_top3, no_em_rec, _ = evaluate_word_and_recovery(
            targets, idx_k, line_by_word, None, DRIFT_Y
        )

        # 5. w/o_Temp
        idx_nb = NearestBoundingBoxDecoder().decode(gaze_seq, word_boxes)
        no_temp_acc, no_temp_top3, no_temp_rec, _ = evaluate_word_and_recovery(
            targets, idx_nb, line_by_word, None, DRIFT_Y
        )

        return {
            "Subject": sub, "Lang": lang, "Trial": trial,
            "STOCK-T_Edge_Acc": edge_acc, "STOCK-T_Edge_Top3": edge_top3, "STOCK-T_Edge_Rec": edge_rec,
            "STOCK-T_Surprisal_Acc": surp_acc, "STOCK-T_Surprisal_Top3": surp_top3, "STOCK-T_Surprisal_Rec": surp_rec,
            "w/o_POM_Acc": no_pom_acc, "w/o_POM_Top3": no_pom_top3, "w/o_POM_Rec": no_pom_rec,
            "w/o_EM_Acc": no_em_acc, "w/o_EM_Top3": no_em_top3, "w/o_EM_Rec": no_em_rec,
            "w/o_Temp_Acc": no_temp_acc, "w/o_Temp_Top3": no_temp_top3, "w/o_Temp_Rec": no_temp_rec
        }
        
    except Exception as e:
        with open("error.log", "a") as f:
            f.write(f"Error in {sub}/{trial}: {traceback.format_exc()}\n")
        return None

def run_full_corpus():
    benchmark_dir = "data/geco/benchmark"
    os.makedirs(benchmark_dir, exist_ok=True)
    
    tasks = []
    # 建立所有任務清單
    for lang in ["L1", "L2"]:
        pop_dir = f"data/geco/population/{lang}"
        if not os.path.exists(pop_dir): continue
            
        subjects = sorted(os.listdir(pop_dir))
        for sub in subjects:
            sub_dir = os.path.join(pop_dir, sub)
            if not os.path.isdir(sub_dir): continue
            
            trials = [d for d in os.listdir(sub_dir) if d.startswith("trial_")]
            for trial in trials:
                trial_dir = os.path.join(sub_dir, trial)
                layout_path = os.path.join(trial_dir, "layout.csv")
                fixations_path = os.path.join(trial_dir, "fixations.csv")
                if os.path.exists(layout_path) and os.path.exists(fixations_path):
                    tasks.append((lang, sub, trial, layout_path, fixations_path))
                    
    print(f"Total trials to process: {len(tasks)}")
    
    return tasks

def _sample_tasks(tasks, sample_trials_total: int, sample_seed: int = 42):
    # Sample on unique (lang, sub, trial, layout_path, fixations_path)
    uniq = sorted({t for t in tasks})
    r = random.Random(int(sample_seed))
    if len(uniq) > int(sample_trials_total):
        uniq = r.sample(uniq, int(sample_trials_total))
    return uniq

def run_full_corpus_parallel(*, sample_trials_total: int | None = None, sample_seed: int = 42):
    tasks = run_full_corpus()
    if sample_trials_total is not None:
        tasks = _sample_tasks(tasks, sample_trials_total=sample_trials_total, sample_seed=sample_seed)
        print(f"[SAMPLE] Using {len(tasks)} trials (seed={sample_seed})")

    all_trial_results = []
    # 啟動多進程池 (自動抓取你的 Ultra 9 所有核心)
    with ProcessPoolExecutor() as executor:
        # 使用 tqdm 顯示進度條
        futures = {executor.submit(process_single_trial, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing Trials"):
            result = future.result()
            if result is not None:
                all_trial_results.append(result)

    df_results = pd.DataFrame(all_trial_results)
    df_results.to_csv("data/geco/benchmark/full_corpus_results.csv", index=False)
    print("Evaluation completed. Saved to data/geco/benchmark/full_corpus_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run a quick sampled subset (parallel).")
    parser.add_argument("--sample-trials", type=int, default=50, help="Trials to sample in --test mode.")
    parser.add_argument("--sample-seed", type=int, default=42, help="Seed for --sample-trials sampling.")
    args = parser.parse_args()

    if args.test:
        run_full_corpus_parallel(sample_trials_total=args.sample_trials, sample_seed=args.sample_seed)
    else:
        run_full_corpus_parallel(sample_trials_total=None)
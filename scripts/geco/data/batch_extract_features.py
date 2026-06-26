import os
import sys
import pandas as pd
import torch
import math
import numpy as np
import json
import re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

# ── Path bootstrap ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.cognition import CognitiveLoadPipeline

# Fix HF_HOME if it's set to a Windows drive path on a non-Windows OS
if os.name != 'nt' and os.environ.get("HF_HOME"):
    _hf_home = os.environ["HF_HOME"]
    if ":" in _hf_home or _hf_home.startswith("D:") or _hf_home.startswith("C:"):
        del os.environ["HF_HOME"]

def clean_word(w):
    return re.sub(r'[^a-zA-Z0-9]', '', w.lower())

def align_words(geco_words, pipeline_words):
    aligned = []
    p_idx = 0
    for gw in geco_words:
        cgw = clean_word(gw)
        if not cgw:
            aligned.append(None)
            continue
            
        found = False
        # Look ahead for exact match first
        for k in range(p_idx, min(p_idx + 15, len(pipeline_words))):
            cpw = clean_word(pipeline_words[k]["word"])
            if cgw == cpw:
                aligned.append(pipeline_words[k])
                p_idx = k + 1
                found = True
                break
                
        if not found:
            # Look ahead for prefix/prefix match (min length 2)
            for k in range(p_idx, min(p_idx + 15, len(pipeline_words))):
                cpw = clean_word(pipeline_words[k]["word"])
                if len(cpw) >= 2 and (cgw.startswith(cpw) or cpw.startswith(cgw)):
                    aligned.append(pipeline_words[k])
                    p_idx = k + 1
                    found = True
                    break
                    
        if not found:
            # Search globally in a window around p_idx for exact match
            for k in range(max(0, p_idx - 10), min(len(pipeline_words), p_idx + 30)):
                cpw = clean_word(pipeline_words[k]["word"])
                if cgw == cpw:
                    aligned.append(pipeline_words[k])
                    p_idx = k + 1
                    found = True
                    break
                    
        if not found:
            # Search globally in a window around p_idx for prefix match (min length 2)
            for k in range(max(0, p_idx - 10), min(len(pipeline_words), p_idx + 30)):
                cpw = clean_word(pipeline_words[k]["word"])
                if len(cpw) >= 2 and (cgw.startswith(cpw) or cpw.startswith(cgw)):
                    aligned.append(pipeline_words[k])
                    p_idx = k + 1
                    found = True
                    break
                    
        if not found:
            aligned.append(None)
            
    return aligned

# Global variable for the worker process pipeline instance
_worker_pipeline = None

def init_worker(lang_label):
    global _worker_pipeline
    pipeline_lang = 'nl' if lang_label == "L1" else 'en'
    _worker_pipeline = CognitiveLoadPipeline(model_type='bert', lang=pipeline_lang)

def process_trial_worker(task_args):
    global _worker_pipeline
    trial_key, sorted_word_indices, trial_layout = task_args
    
    sentence_words = [trial_layout[str(i)]['word'].strip() for i in sorted_word_indices]
    full_sentence = " ".join(sentence_words)
    
    # Run the pipeline
    cog_result = _worker_pipeline.run(full_sentence)
    word_analysis = cog_result.get("word_analysis", [])
    
    # Align words
    aligned = align_words(sentence_words, word_analysis)
    
    num_words = len(sentence_words)
    word_attn = np.zeros((num_words, num_words)) # Mock/unused attention matrix
    
    layout_results = []
    for i, word_idx in enumerate(sorted_word_indices):
        match = aligned[i]
        surprisal = match["surprisal"] if match else 5.0
        entropy = match["entropy"] if match else 0.5
        load_score = match["load_score"] if match else 0.5
        
        word_info = trial_layout[str(word_idx)]
            
        layout_results.append({
            "WORD_ID_WITHIN_TRIAL": word_idx,
            "WORD": word_info['word'],
            "true_x": round(word_info['x'], 1),
            "true_y": round(word_info['y'], 1),
            "surprisal_score": round(surprisal, 4),
            "attention_score": round(entropy, 4),
            "cognitive_mass": round(load_score, 4)
        })
        
    return trial_key, pd.DataFrame(layout_results), word_attn

def extract_features(input_path, lang_label):
    csv_path = input_path.replace(".xlsx", ".csv")
    layout_json_path = os.path.join(PROJECT_ROOT, "data", "geco", "meta", f"{lang_label.lower()}_consensus_layout.json")
    
    if not os.path.exists(layout_json_path):
        print(f"❌ Consensus layout not found: {layout_json_path}")
        return

    with open(layout_json_path, 'r', encoding='utf-8') as f:
        consensus_layouts = json.load(f)

    full_csv_path = os.path.join(PROJECT_ROOT, csv_path)
    if os.path.exists(full_csv_path):
        print(f"⏳ Reading CSV data from {full_csv_path}...")
        try:
            df_all = pd.read_csv(full_csv_path)
        except Exception as e:
            print(f"❌ Failed to read {full_csv_path}: {e}")
            return
    else:
        print(f"❌ {full_csv_path} not found. Please convert .xlsx to .csv first.")
        return
            
    subjects = df_all['PP_NR'].unique()
    print(f"👥 Found {len(subjects)} subjects in {lang_label} dataset.")
    
    # 1. Gather all unique trial layouts
    unique_trial_keys = set()
    for sub in subjects:
        sub_df = df_all[df_all['PP_NR'] == sub]
        trials = sub_df['TRIAL'].unique()
        for t in trials:
            if str(t) in consensus_layouts:
                unique_trial_keys.add(str(t))
    unique_trial_keys = sorted(list(unique_trial_keys))
    
    worker_tasks = []
    for trial_key in unique_trial_keys:
        trial_layout = consensus_layouts[trial_key]
        sorted_word_indices = sorted([int(k) for k in trial_layout.keys()])
        worker_tasks.append((trial_key, sorted_word_indices, trial_layout))
        
    # 2. Parallel Processing of BERT/Ridge Cognitive Load Pipeline
    print(f"🔥 Starting parallel extraction of {len(worker_tasks)} unique trials on {lang_label}...")
    num_workers = min(multiprocessing.cpu_count(), 8)
    print(f"   Using {num_workers} parallel workers.")
    
    trial_metadata = {}
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(lang_label,)) as executor:
        futures = {executor.submit(process_trial_worker, task): task for task in worker_tasks}
        for future in tqdm(as_completed(futures), total=len(worker_tasks), desc="Extracting features"):
            try:
                trial_key, df_layout, word_attn = future.result()
                trial_metadata[trial_key] = {
                    "df_layout": df_layout,
                    "word_attn": word_attn
                }
            except Exception as e:
                print(f"❌ Error extracting features for trial: {e}")
                
    # 3. Write CSV Layouts and Fixations sequentially
    for sub in tqdm(subjects, desc=f"Writing subject files for {lang_label}"):
        sub_df = df_all[df_all['PP_NR'] == sub]
        trials = sub_df['TRIAL'].unique()
        
        for trial_id in trials:
            trial_key = str(trial_id)
            if trial_key not in trial_metadata:
                continue
                
            out_dir = os.path.join(PROJECT_ROOT, "data", "geco", "population", lang_label, str(sub), f"trial_{trial_id}")
            os.makedirs(out_dir, exist_ok=True)
            out_layout = os.path.join(out_dir, "layout.csv")
            out_fixations = os.path.join(out_dir, "fixations.csv")
            out_attn = os.path.join(out_dir, "attention.npy")
            
            trial_data = trial_metadata[trial_key]
            trial_data["df_layout"].to_csv(out_layout, index=False)
            np.save(out_attn, trial_data["word_attn"])
                
            # Process Fixations
            df_sub_trial = sub_df[sub_df['TRIAL'] == trial_id]
            fixation_results = []
            sorted_word_indices = sorted([int(k) for k in consensus_layouts[trial_key].keys()])
            
            for row in df_sub_trial.itertuples():
                if pd.notna(row.WORD_FIRST_FIXATION_X) and pd.notna(row.WORD_FIRST_FIXATION_Y):
                    try:
                        layout_idx = sorted_word_indices.index(int(row.WORD_ID_WITHIN_TRIAL))
                        fixation_results.append({
                            "layout_index": layout_idx,
                            "WORD_ID_WITHIN_TRIAL": row.WORD_ID_WITHIN_TRIAL,
                            "fixation_x": row.WORD_FIRST_FIXATION_X,
                            "fixation_y": row.WORD_FIRST_FIXATION_Y,
                            "reading_time": getattr(row, 'WORD_TOTAL_READING_TIME', 0)
                        })
                    except ValueError:
                        continue
                        
            pd.DataFrame(fixation_results).to_csv(out_fixations, index=False)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    extract_features("data/geco/L1ReadingData.xlsx", "L1")
    extract_features("data/geco/L2ReadingData.xlsx", "L2")

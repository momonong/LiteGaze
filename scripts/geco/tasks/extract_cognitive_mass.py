import os
import sys
import pandas as pd
import torch
import math
import numpy as np
import re

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

def process_file(input_file, output_file, cog_pipeline, attn_pipeline):
    if not os.path.exists(input_file):
        print(f"❌ 找不到輸入檔案: {input_file}")
        return
        
    print(f"⏳ 正在處理 GECO 資料集: {input_file} ...")
    df = pd.read_csv(input_file)
    sentence_words = df['WORD'].tolist()
    full_sentence = " ".join(sentence_words)
    
    # 1. 執行 Pipeline 獲取完整認知特徵 (使用 GPT-2 獲得心理語言學特徵與正確的 XGBoost / Ridge 分數)
    cog_result = cog_pipeline.run(full_sentence)
    word_analysis = cog_result.get("word_analysis", [])
    
    # 2. 進行雙向對齊
    aligned = align_words(sentence_words, word_analysis)
    print(f"   GECO 行數: {len(df)}, 對齊成功行數: {len(aligned) - aligned.count(None)}, 未對齊: {aligned.count(None)}")
    
    # 3. 計算經典 BERT Attention 矩陣 (使用 BERT)
    tokenizer = attn_pipeline.calculator.tokenizer
    model = attn_pipeline.calculator.model
    device = attn_pipeline.calculator.device
    
    inputs = tokenizer(full_sentence, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attention_matrix = outputs.attentions[-1][0].mean(dim=0)
        
    word_token_indices = []
    for idx, row in df.iterrows():
        target_word = str(row['WORD']).lower().strip()
        token_index = -1
        for i, token in enumerate(tokens):
            if token.replace("##", "") == target_word:
                token_index = i
                break
        word_token_indices.append(token_index)
        
    num_words = len(df)
    word_attention_matrix = np.zeros((num_words, num_words))
    for i in range(num_words):
        ti = word_token_indices[i]
        for j in range(num_words):
            tj = word_token_indices[j]
            if ti != -1 and tj != -1:
                word_attention_matrix[i, j] = attention_matrix[ti, tj].item()
                
    # 4. 寫入結果
    results = []
    for i, row in df.iterrows():
        match = aligned[i]
        surprisal = match["surprisal"] if match else 5.0
        entropy = match["entropy"] if match else 0.5
        load_score = match["load_score"] if match else 0.5
        
        # 經典 attention centrality 度量
        attention_score = word_attention_matrix[:, i].sum()
        
        results.append({
            "WORD_ID": row["WORD_ID"],
            "WORD": row["WORD"],
            "true_x": row["true_x"],
            "true_y": row["true_y"],
            "surprisal_score": round(surprisal, 4),
            "attention_score": round(attention_score, 4),
            # 使用 XGBoost/Ridge 的多特徵融合認知負荷分 (0.0-1.0) 做為 cognitive_mass
            "cognitive_mass": round(load_score, 4)
        })
        
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"   ✅ 特徵 CSV 已儲存至: {output_file}")
    
    # 5. 儲存 Attention Matrix
    attention_output_file = output_file.replace(".csv", "_attention.npy")
    np.save(attention_output_file, word_attention_matrix)
    print(f"   ✅ Attention Matrix 已儲存至: {attention_output_file}")

def main():
    print("=========================================================")
    print("    LexiGaze GECO Cognitive Mass Feature Extractor       ")
    print("=========================================================")
    
    # 初始化 Pipeline: 用於認知特徵的 GPT-2 與用於 Attention Matrix 的 BERT
    cog_pipeline = CognitiveLoadPipeline(model_type='gpt2', lang='en')
    attn_pipeline = CognitiveLoadPipeline(model_type='bert', lang='en')
    
    # 處理 L2 ESL 資料
    l2_input = os.path.join(PROJECT_ROOT, "archive", "data", "geco", "geco_pp01_trial5_clean.csv")
    l2_output = os.path.join(PROJECT_ROOT, "archive", "data", "geco", "geco_pp01_cognitive_mass.csv")
    process_file(l2_input, l2_output, cog_pipeline, attn_pipeline)
    
    # 同步寫入主資料庫以防萬一
    l2_output_main = os.path.join(PROJECT_ROOT, "data", "geco", "geco_pp01_cognitive_mass.csv")
    os.makedirs(os.path.dirname(l2_output_main), exist_ok=True)
    process_file(l2_input, l2_output_main, cog_pipeline, attn_pipeline)
    
    # 處理 L1 母語者資料
    l1_input = os.path.join(PROJECT_ROOT, "archive", "data", "geco", "geco_l1_pp01_trial5_clean.csv")
    l1_output = os.path.join(PROJECT_ROOT, "archive", "data", "geco", "geco_l1_pp01_cognitive_mass.csv")
    process_file(l1_input, l1_output, cog_pipeline, attn_pipeline)
    
    l1_output_main = os.path.join(PROJECT_ROOT, "data", "geco", "geco_l1_pp01_cognitive_mass.csv")
    process_file(l1_input, l1_output_main, cog_pipeline, attn_pipeline)

if __name__ == "__main__":
    main()
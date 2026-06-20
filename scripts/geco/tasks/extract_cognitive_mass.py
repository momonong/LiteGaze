import os
from dotenv import load_dotenv

# 1. 載入 .env 檔案中的環境變數
load_dotenv()

# Fix HF_HOME if it's set to a Windows drive path on a non-Windows OS
if os.name != 'nt' and os.environ.get("HF_HOME"):
    _hf_home = os.environ["HF_HOME"]
    if ":" in _hf_home or _hf_home.startswith("D:") or _hf_home.startswith("C:"):
        del os.environ["HF_HOME"]

# 2. (可選) 印出來確認一下有沒有成功讀到
print(f"📦 Hugging Face 模型存放路徑: {os.getenv('HF_HOME')}")

import pandas as pd
import torch
import math
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM

# 1. 載入我們清洗好的乾淨 ESL 閱讀資料
input_file = "data/geco/geco_pp01_trial5_clean.csv"
output_file = "data/geco/geco_pp01_cognitive_mass.csv"

print("⏳ 載入資料集...")
df = pd.read_csv(input_file)

# 把所有的單字串成一個完整的句子，供 BERT 閱讀上下文
sentence_words = df['WORD'].tolist()
full_sentence = " ".join(sentence_words)

# 2. 載入模型 (RTX 5090 準備發威！)
# 雖然 bert-tiny 很小，但使用 GPU 還是能顯著加速
MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
print(f"📦 載入神經網路模型: {MODEL_NAME}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用運算設備: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, output_attentions=True).to(device)
model.eval()

# 3. 核心運算：萃取 Surprisal 與 Attention
print("🧠 正在計算神經符號特徵 (這可能需要幾秒鐘)...")

inputs = tokenizer(full_sentence, return_tensors="pt").to(device)
input_ids = inputs["input_ids"][0]
tokens = tokenizer.convert_ids_to_tokens(input_ids)

results = []

with torch.no_grad():
    outputs = model(**inputs)
    # 提取最後一層的 Attention Matrix (Layer -1), 取所有 Head 的平均值
    # 形狀: [batch_size, num_heads, seq_length, seq_length]
    attention_matrix = outputs.attentions[-1][0].mean(dim=0) 

    # 針對每一個被 GECO 記錄下來的單字計算特徵
    word_token_indices = []
    for idx, row in df.iterrows():
        target_word = str(row['WORD']).lower().strip()
        
        # 尋找該單字在 BERT token 列表中的對應位置 (簡單匹配)
        token_index = -1
        for i, token in enumerate(tokens):
            if token.replace("##", "") == target_word:
                token_index = i
                break
        word_token_indices.append(token_index)
                
    # 建立 Word-to-Word Attention Matrix (Skill 8)
    num_words = len(df)
    word_attention_matrix = np.zeros((num_words, num_words))
    
    for i in range(num_words):
        ti = word_token_indices[i]
        for j in range(num_words):
            tj = word_token_indices[j]
            if ti != -1 and tj != -1:
                word_attention_matrix[i, j] = attention_matrix[ti, tj].item()

    # 計算各單字的特徵
    for i, row in df.iterrows():
        token_index = word_token_indices[i]
        if token_index != -1 and token_index > 0 and token_index < len(input_ids) - 1:
            gold_id = input_ids[token_index].item()
            
            # A. 計算 Surprisal (局部難度)
            masked_ids = input_ids.clone()
            masked_ids[token_index] = tokenizer.mask_token_id
            
            mask_outputs = model(masked_ids.unsqueeze(0))
            logits = mask_outputs.logits[0, token_index]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            word_prob = probs[gold_id].item()
            surprisal = -math.log2(word_prob) if word_prob > 0 else 15.0
            
            # B. 計算 Attention Centrality (使用已提取的矩陣)
            attention_score = word_attention_matrix[:, i].sum()
            
            # C. 融合為 Cognitive Mass 認知質量
            cognitive_mass = surprisal * attention_score
            
            results.append({
                "WORD_ID": row["WORD_ID"],
                "WORD": row["WORD"],
                "true_x": row["true_x"],
                "true_y": row["true_y"],
                "surprisal_score": round(surprisal, 4),
                "attention_score": round(attention_score, 4),
                "cognitive_mass": round(cognitive_mass, 4)
            })
        else:
             results.append({
                "WORD_ID": row["WORD_ID"],
                "WORD": row["WORD"],
                "true_x": row["true_x"],
                "true_y": row["true_y"],
                "surprisal_score": 5.0,
                "attention_score": 0.5,
                "cognitive_mass": 2.5
            })

# 4. 儲存結果與 Attention 矩陣
df_results = pd.DataFrame(results)
df_results.to_csv(output_file, index=False)

# 儲存 N x N Attention Matrix 供 Skill 8 使用
attention_output_file = output_file.replace(".csv", "_attention.npy")
np.save(attention_output_file, word_attention_matrix)

print(f"✅ 運算完成！特徵檔已儲存為 {output_file}")
print(f"✅ Attention Matrix 已儲存為 {attention_output_file} (Skill 8 Ready)")
print("\n預覽前 3 筆最高質量的單字 (這些字會產生最強的引力)：")
print(df_results.sort_values(by="cognitive_mass", ascending=False).head(3))
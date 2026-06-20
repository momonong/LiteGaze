# %% [markdown]
# # IntelligentGaze: Neuro-Symbolic 認知引擎 (Word Surprisal 萃取)
# 
# 本腳本為 IntelligentGaze 專案的「神經符號 (Neuro-Symbolic) 引擎」核心實作。
# 我們將使用輕量級的預訓練語言模型 (`bert-tiny`) 來分析一段閱讀文本，
# 模擬人類大腦的認知過程：當遇到不符合上下文預期的「生難字」時，模型會給出較高的**驚奇度 (Word Surprisal)**。
# 
# 接著，我們將這個驚奇度轉換為 UI 上的**引力半徑 (Gravity Radius)**，
# 這個半徑將在後續的邊緣運算校準中，像引力一樣將漂移的眼動儀軌跡「吸附」到正確的單字上！
# 
# **此腳本設計為可在 Google Colab 上「一鍵執行」。**

# %%
# ==========================================
# 1. 環境建置 (Environment Setup)
# ==========================================
print("📦 [1/4] 正在安裝與載入必要的套件...")
# 安裝 transformers 庫 (Hugging Face) 與 PyTorch
!pip install -q transformers torch pandas

import os
import math
import json
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM
import pandas as pd

# ==========================================
# 2. 參數設定與模型載入 (Configuration & Model Loading)
# ==========================================
print("🧠 [2/4] 正在初始化神經符號引擎參數與載入 bert-tiny 模型...")

# 定義測試文本
TEST_SENTENCE = "The ubiquitous phenomenon completely bewildered the inexperienced researcher."

# 設定模型名稱 (使用極度輕量的 bert-tiny，適合邊緣運算與快速展示)
MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"

# 載入 Tokenizer 與 Masked Language Model
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
lm_model = BertForMaskedLM.from_pretrained(MODEL_NAME)
lm_model.eval() # 設定為評估模式，不計算梯度

# UI 視覺化參數設定 (定義引力半徑的範圍)
MIN_GRAVITY_RADIUS_PX = 10  # 最小引力半徑 (像素)
MAX_GRAVITY_RADIUS_PX = 50  # 最大引力半徑 (像素)
# 設定預期的驚奇度上下限 (用於將無限大的 Surprisal 歸一化到 0~1 之間)
MIN_EXPECTED_SURPRISAL = 1.0  
MAX_EXPECTED_SURPRISAL = 15.0 

# 定義要過濾的常見功能詞與標點符號 (停用詞)，我們不計算這些字的難度
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    ".", ",", "!", "?", ";", ":", '"', "'"
}

# ==========================================
# 3. 核心邏輯：計算驚奇度與引力半徑 (Core Logic)
# ==========================================
print("⚙️ [3/4] 正在逐字計算 Word Surprisal 與 Gravity Radius...")

def calculate_word_surprisal(word: str, context_sentence: str) -> float:
    """
    計算單字在句子中的驚奇度 (Surprisal = -log2(P(word|context)))
    """
    # 1. 將目標單字轉換為 tokens
    word_tokens = tokenizer.tokenize(word)
    if not word_tokens:
        return MAX_EXPECTED_SURPRISAL # 若無法分詞，視為極難字

    # 2. 構造 Masked 句子 (將目標單字替換為 [MASK])
    # 注意：這裡使用簡單的字串替換，實際嚴謹實作應基於 token index
    masked_sentence = context_sentence.replace(word, tokenizer.mask_token, 1)

    # 3. 將句子轉換為模型可接受的 Tensor 格式
    inputs = tokenizer(masked_sentence, return_tensors="pt")
    
    # 4. 找出 [MASK] token 在張量中的位置 (index)
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    if mask_token_index.numel() == 0:
        return MAX_EXPECTED_SURPRISAL
    mask_token_index = mask_token_index[0]

    # 5. 模型推論：預測 [MASK] 位置的字彙機率分佈
    with torch.no_grad():
        outputs = lm_model(**inputs)
        predictions = outputs.logits
        
    # 取出 [MASK] 位置的 logits
    masked_token_logits = predictions[0, mask_token_index, :]
    
    # 將 logits 轉換為機率 (Softmax)
    probabilities = torch.softmax(masked_token_logits, dim=-1)

    # 6. 取得目標單字 (第一個 subword) 的真實 Token ID
    target_token_id = tokenizer.convert_tokens_to_ids(word_tokens[0])
    if target_token_id == tokenizer.unk_token_id:
         return MAX_EXPECTED_SURPRISAL
         
    # 7. 取得模型認為該位置出現這個字的「機率」
    target_word_probability = probabilities[target_token_id].item()
    
    # 防止 log(0) 錯誤
    target_word_probability = max(target_word_probability, 1e-10)

    # 8. 核心公式：計算 Surprisal (-log2(P))
    surprisal = -math.log2(target_word_probability)
    return surprisal

def map_surprisal_to_radius(surprisal: float) -> float:
    """
    將驚奇度轉換為 UI 上的引力半徑 (Pixels)
    依照專案規格公式：gravity_radius_px = 50 + (surprisal * 10)
    """
    # 根據規格的公式直接轉換
    gravity_radius_px = 50 + (surprisal * 10)
    
    # 防止半徑無限制放大，這裡設定一個合理的上限值 (例如 200px)
    return min(round(gravity_radius_px, 2), 200.0)

# 開始遍歷句子中的每個單字
results = []
# 簡單的單字分割 (移除句點)
words = TEST_SENTENCE.replace('.', '').split()

for w in words:
    clean_word = w.lower()
    if clean_word in STOP_WORDS:
        continue # 跳過無意義的功能詞
        
    # 執行計算
    surp_score = calculate_word_surprisal(w, TEST_SENTENCE)
    grav_radius = map_surprisal_to_radius(surp_score)
    
    results.append({
        "word": w,
        "surprisal_score": round(surp_score, 4),
        "gravity_radius_px": grav_radius
    })

# ==========================================
# 4. 資料輸出與預覽 (Data Output & Preview)
# ==========================================
print("\n💾 [4/4] 正在輸出結果與生成 cognitive_weights.json...")

# 將結果轉換為 DataFrame 方便排序與預覽
df = pd.DataFrame(results)

# 依照驚奇度 (難度) 由高到低排序
df_sorted = df.sort_values(by="surprisal_score", ascending=False).reset_index(drop=True)

# 匯出 JSON 檔案
output_data = {
    "sentence": TEST_SENTENCE,
    "model_used": MODEL_NAME,
    "cognitive_weights": df_sorted.to_dict(orient="records")
}

with open("cognitive_weights.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

print("\n✅ 執行成功！已生成檔案：`cognitive_weights.json`\n")

# 印出前三個最難的單字供使用者預覽
print("🏆 系統判定前 3 名「最難單字 (Top 3 Hardest Words)」：")
for index, row in df_sorted.head(3).iterrows():
    print(f"  {index+1}. {row['word']:<15} | 驚奇度: {row['surprisal_score']:>7.4f} bits | 引力半徑: {row['gravity_radius_px']:>5.2f} px")

import os
import re
import math
import json
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ GPU 已啟用: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️ 未偵測到 GPU，將使用 CPU 執行。")

print("🚀 啟動 IntelligentGaze 神經符號融合引擎 (Neuro-Symbolic Engine)...")

print(f"\n📦 [1/3] 正在載入語言模型 ({MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

text = (
    "The ubiquitous phenomenon completely bewildered the inexperienced researcher.\n"
    "Despite rigorous analysis, the underlying mechanisms remained enigmatic, defying conventional explanation."
)
target_word = "phenomenon"
word_x, word_y = 850, 500


def find_word_char_span(text, target_word):
    pattern = re.compile(rf"\b{re.escape(target_word)}\b", flags=re.IGNORECASE)
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"找不到 target_word='{target_word}' 在句子中的字元位置。")
    return match.start(), match.end()


def get_target_token_positions(text, target_word, tokenizer):
    char_start, char_end = find_word_char_span(text, target_word)

    enc = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
    )

    offsets = enc["offset_mapping"][0].tolist()
    input_ids = enc["input_ids"][0].tolist()

    token_positions = []
    token_strings = []

    for i, ((s, e), tid) in enumerate(zip(offsets, input_ids)):
        if s == e:
            continue
        overlap = max(s, char_start) < min(e, char_end)
        if overlap:
            token_positions.append(i)
            token_strings.append(tokenizer.convert_ids_to_tokens(tid))

    if not token_positions:
        raise ValueError(f"target_word='{target_word}' 沒有對齊到任何 token。")

    return enc, token_positions, token_strings, (char_start, char_end)


def compute_word_surprisal_pll(text, target_word, tokenizer, model, device):
    enc, token_positions, token_strings, char_span = get_target_token_positions(text, target_word, tokenizer)

    input_ids = enc["input_ids"][0]
    attention_mask = enc["attention_mask"][0]

    total_nll = 0.0
    token_details = []

    for pos in token_positions:
        masked_ids = input_ids.clone()
        gold_id = int(masked_ids[pos].item())
        masked_ids[pos] = tokenizer.mask_token_id

        batch = {
            "input_ids": masked_ids.unsqueeze(0).to(device),
            "attention_mask": attention_mask.unsqueeze(0).to(device),
        }

        with torch.no_grad():
            logits = model(**batch).logits[0, pos]
            log_probs = F.log_softmax(logits, dim=-1)

        token_log_prob = float(log_probs[gold_id].item())
        token_surprisal = -token_log_prob
        total_nll += token_surprisal

        token_details.append({
            "position": int(pos),
            "token": tokenizer.convert_ids_to_tokens(gold_id),
            "log_prob": round(token_log_prob, 6),
            "surprisal": round(token_surprisal, 6),
        })

    return {
        "target_word": target_word,
        "char_span": {"start": char_span[0], "end": char_span[1]},
        "token_positions": token_positions,
        "token_pieces": token_strings,
        "num_subtokens": len(token_positions),
        "word_surprisal": total_nll,
        "avg_subtoken_surprisal": total_nll / len(token_positions),
        "token_details": token_details,
    }


print(f"📖 分析文本: '{text}'")
surprisal_result = compute_word_surprisal_pll(text, target_word, tokenizer, model, device)

surprisal = surprisal_result["word_surprisal"]
alpha_weight = min(surprisal / 15.0, 0.95)

knowledge_base = {
    target_word: {
        "x": word_x,
        "y": word_y,
        "surprisal": round(surprisal, 4),
        "avg_subtoken_surprisal": round(surprisal_result["avg_subtoken_surprisal"], 4),
        "token_pieces": surprisal_result["token_pieces"],
        "num_subtokens": surprisal_result["num_subtokens"],
        "alpha": round(alpha_weight, 4),
    }
}

kb_path = os.path.join(DATA_DIR, "cognitive_weights.json")
with open(kb_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "text": text,
            "target_analysis": surprisal_result,
            "knowledge_base": knowledge_base,
        },
        f,
        indent=4,
        ensure_ascii=False,
    )

print(
    f"✅ 驚奇度計算完成: {target_word} | "
    f"subtokens={surprisal_result['token_pieces']} | "
    f"word_surprisal={surprisal:.4f} | alpha={alpha_weight:.2f}"
)
print(f"✅ 已儲存至 {kb_path}")

print("\n📊 [2/3] 正在載入硬體眼動數據...")
csv_path = os.path.join(DATA_DIR, "raw.csv")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"✅ 成功載入真實數據: {len(df)} 筆。")
else:
    print(f"⚠️ 找不到 {csv_path}，啟用防護機制：自動生成模擬雜訊數據。")
    np.random.seed(42)
    df = pd.DataFrame({
        "x_px": np.random.normal(loc=430, scale=80, size=50),
        "y_px": np.random.normal(loc=350, scale=60, size=50)
    })

df["baseline_x"] = df["x_px"].rolling(window=5, min_periods=1).mean()
df["baseline_y"] = df["y_px"].rolling(window=5, min_periods=1).mean()

print("\n🧲 [3/3] 執行引力吸附演算法...")

def apply_gravity(row):
    px, py = row["baseline_x"], row["baseline_y"]
    dist = np.sqrt((px - word_x) ** 2 + (py - word_y) ** 2)

    if dist < 150:
        calib_x = px + (word_x - px) * alpha_weight
        calib_y = py + (word_y - py) * alpha_weight
        return pd.Series([calib_x, calib_y])
    return pd.Series([px, py])

df[["vibe_x", "vibe_y"]] = df.apply(apply_gravity, axis=1)

# 輸出 Baseline 對照組
baseline_output = os.path.join(DATA_DIR, "baseline.csv")
df_baseline = df[['timestamp', 'baseline_x', 'baseline_y']].rename(columns={'baseline_x': 'x_px', 'baseline_y': 'y_px'})
df_baseline.to_csv(baseline_output, index=False)

# 輸出最終 Calibrated 結果
calibrated_output = os.path.join(DATA_DIR, "calibrated.csv")
df_calibrated = df[['timestamp', 'vibe_x', 'vibe_y']].rename(columns={'vibe_x': 'x_px', 'vibe_y': 'y_px'})
df_calibrated.to_csv(calibrated_output, index=False)

print(f"✅ 校準完成！")
print(f"  - 基線對照 (Baseline) 匯出至 {baseline_output}")
print(f"  - 最終校準 (Calibrated) 匯出至 {calibrated_output}")

plt.figure(figsize=(10, 6))
plt.scatter(df["x_px"], df["y_px"], color="lightcoral", alpha=0.3, label="1. Raw Webcam Noise")
plt.plot(df["baseline_x"], df["baseline_y"], color="orange", linewidth=1.5, linestyle="--", label="2. Baseline (Moving Average)")
plt.scatter(df["vibe_x"], df["vibe_y"], color="darkblue", alpha=0.8, s=60, label="3. Neuro-Symbolic Gaze")

plt.scatter(word_x, word_y, color="green", marker="*", s=300, edgecolor="black", zorder=5)
plt.text(
    word_x + 15,
    word_y - 15,
    f"{target_word}\n(Alpha: {alpha_weight:.2f})\nSubtokens: {surprisal_result['num_subtokens']}",
    fontsize=11,
    fontweight="bold",
    color="green",
)
circle = plt.Circle((word_x, word_y), 150, color="green", fill=False, linestyle=":", alpha=0.5)
plt.gca().add_patch(circle)

plt.title("Neuro-Symbolic Engine: PLL-Based Word Surprisal Calibration", fontsize=16)
plt.xlabel("Screen X")
plt.ylabel("Screen Y")
plt.gca().invert_yaxis()
plt.legend(loc="lower left")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "dashboard.png"), dpi=200)
plt.show()
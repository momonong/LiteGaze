import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 設定路徑
# ==========================================
DATA_DIR = r"tutorial\data"
RAW_FILE = os.path.join(DATA_DIR, "raw.csv")

# ==========================================
# 模組 1：資料讀取 (寫死，學生不動)
# ==========================================
try:
    df = pd.read_csv(RAW_FILE)
    print(f"成功載入 {len(df)} 筆原始眼動數據！")
except FileNotFoundError:
    print(f"錯誤：找不到 {RAW_FILE}")
    exit()

# ==========================================
# 模組 2：對照組 Baseline - 基礎移動平均 (已提供)
# ==========================================
# 這是業界最無腦的解法，讓學生看到數學的基礎力量
window_size = 5
df['baseline_x'] = df['x_px'].rolling(window=window_size, min_periods=1).mean()
df['baseline_y'] = df['y_px'].rolling(window=window_size, min_periods=1).mean()


# ==========================================
# 🌟 模組 3：VIBE CODING 優化區 (挖空給 AI)
# ==========================================
# 學生任務：請修改 knowledge_base.md，並讓 AI 在這裡寫出更強的校準演算法
# 目標：結合 NLP 先驗引力 (例如 'phenomenon' 座標 450, 320)，產出 'vibe_x' 與 'vibe_y'

# --- [AI 產出的程式碼貼在這裡] ---
# (為了防止報錯，預設先讓 vibe 等於 baseline)
df['vibe_x'] = df['baseline_x']
df['vibe_y'] = df['baseline_y']
# ---------------------------------


# ==========================================
# 模組 4：輸出校準結果 (資料工程解耦)
# ==========================================
# 輸出 Baseline 對照組
baseline_output = os.path.join(DATA_DIR, 'baseline.csv')
df_baseline = df[['timestamp', 'baseline_x', 'baseline_y']].rename(columns={'baseline_x': 'x_px', 'baseline_y': 'y_px'})
df_baseline.to_csv(baseline_output, index=False)

# 輸出最終 Calibrated 結果
calibrated_output = os.path.join(DATA_DIR, 'calibrated.csv')
df_calibrated = df[['timestamp', 'vibe_x', 'vibe_y']].rename(columns={'vibe_x': 'x_px', 'vibe_y': 'y_px'})
df_calibrated.to_csv(calibrated_output, index=False)

print(f"✅ 校準完成！")
print(f"  - 基線對照 (Baseline) 匯出至：{baseline_output}")
print(f"  - 最終校準 (Calibrated) 匯出至：{calibrated_output}")


# ==========================================
# 模組 5：視覺化成效驗證 (寫死，學生不動)
# ==========================================
plt.figure(figsize=(12, 7))

# 1. 原始雜訊
plt.scatter(df['x_px'], df['y_px'], color='lightcoral', alpha=0.3, label='1. Raw Webcam Noise')

# 2. 對照組：無腦移動平均 (沒有認知科學邏輯)
plt.plot(df['baseline_x'], df['baseline_y'], color='orange', linewidth=1.5, linestyle='--', label='2. Baseline (Moving Average)')

# 3. 實驗組：Vibe Coding 產出的神經符號校準
plt.scatter(df['vibe_x'], df['vibe_y'], color='darkblue', alpha=0.8, label='3. Vibe-Coded Gaze (Neuro-Symbolic)')

# 標示引力目標 (這裡先寫死示範)
plt.scatter(850, 500, color='green', marker='*', s=300, edgecolor='black', zorder=5)
plt.text(865, 515, 'phenomenon (alpha=0.8)', fontsize=12, fontweight='bold', color='green')

plt.title('IntelligentGaze: Baseline vs. Vibe Coding Optimization', fontsize=16)
plt.xlabel('Screen X')
plt.ylabel('Screen Y')
plt.gca().invert_yaxis()
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig(os.path.join(BASE_DIR, "figures", "result.png"))
plt.show().savefig("tutorial/figures/result.png")
plt.show()
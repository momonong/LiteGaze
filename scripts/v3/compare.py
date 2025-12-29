import tensorflow as tf
import h5py
import numpy as np
import os
from models import build_teacher_v3

# === ⚙️ 設定區 ===
LAPTOP_MODEL_PATH = 'models/teacher_v3_best_5090.h5' # 請改成你的筆電模型檔名
A100_MODEL_PATH = 'models/teacher_v3_best_a100.h5'     # 請改成你從 A100 抓下來的檔名
DATA_PATH = 'data/teacher_224.h5'          # 資料集路徑
TEST_SAMPLES = 1000                        # 考 1000 題就好

def evaluate_model(model_path, x_test, y_test, name):
    if not os.path.exists(model_path):
        print(f"⚠️ 找不到 {name} ({model_path})，跳過。")
        return

    print(f"\n🔍 正在載入 {name}...")
    model = build_teacher_v3()
    model.load_weights(model_path)
    
    print(f"📝 {name} 正在考試...")
    # 預測
    preds = model.predict(x_test, verbose=1)
    # preds[0] 是 gaze vector
    pred_gaze = preds[0]
    
    # 計算 MAE (平均絕對誤差)
    mae = np.mean(np.abs(pred_gaze - y_test))
    print(f"🏆 {name} 成績單:")
    print(f"   MAE (Error): {mae:.4f}")
    return mae

def main():
    # 1. 準備考試題目 (隨機抽 1000 張圖)
    print("📚 正在準備測試資料...")
    with h5py.File(DATA_PATH, 'r') as hf:
        total = hf['images'].shape[0]
        indices = np.random.choice(total, TEST_SAMPLES, replace=False)
        indices = np.sort(indices)
        
        x_test = hf['images'][indices].astype(np.float32) / 255.0
        # 標籤我們只要 gaze_out (index 0~2)
        y_test = hf['labels'][indices][:, 0:2] 

    # 2. 評估筆電模型
    score_laptop = evaluate_model(LAPTOP_MODEL_PATH, x_test, y_test, "筆電老師")
    
    # 3. 評估 A100 模型
    score_a100 = evaluate_model(A100_MODEL_PATH, x_test, y_test, "A100 老師")

    # 4. 宣布獲勝者
    if score_laptop and score_a100:
        print("\n========== 最終結果 ==========")
        if score_a100 < score_laptop:
            print(f"🎉 建議使用 [A100 老師] (誤差少 {(score_laptop - score_a100):.4f})")
        else:
            print(f"🎉 建議使用 [筆電老師] (誤差少 {(score_a100 - score_laptop):.4f})")

if __name__ == "__main__":
    main()
import tensorflow as tf
import numpy as np
import os
import glob
import random

# === 設定 ===
TFLITE_PATH = 'models/litegaze_student.tflite'
DATA_DIR = './data/processed'

def verify_model():
    # 1. 載入 TFLite 模型
    print(f"📥 載入 TFLite 模型: {TFLITE_PATH}")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()

    # 取得輸入輸出的詳細資訊
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 2. 隨機抓一筆資料
    files = glob.glob(os.path.join(DATA_DIR, '*', '*.npz'))
    if not files:
        print("❌ 找不到測試資料！")
        return

    test_file = random.choice(files)
    print(f"🕵️‍♂️ 測試樣本: {test_file}")
    
    with np.load(test_file) as data:
        # 注意：TFLite 接受的是 float32，且要歸一化 (0~1)
        # 我們抓第一張圖來測
        img_raw = data['student'][0] # (60, 60, 3) uint8
        label = data['label'][0]     # [Pitch, Yaw]
    
    # 前處理 (跟訓練時一模一樣)
    input_data = img_raw.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0) # 增加 Batch 維度 -> (1, 60, 60, 3)

    # 3. 執行推論 (Inference)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # 取得結果
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0]

    # 4. 比對結果
    print("\n--- 🎯 測試結果 ---")
    print(f"正確答案 (Label):     Pitch={label[0]:.4f}, Yaw={label[1]:.4f}")
    print(f"模型預測 (Prediction): Pitch={prediction[0]:.4f}, Yaw={prediction[1]:.4f}")
    
    # 計算誤差
    mae = np.mean(np.abs(label - prediction))
    print(f"📉 平均誤差 (MAE):     {mae:.4f} (約 {mae * 180 / np.pi:.2f} 度)")

    if mae < 0.1:
        print("\n✅ 通過！模型運作正常且準確。")
    else:
        print("\n⚠️ 警告：誤差有點大，建議多測幾次確認。")

if __name__ == "__main__":
    verify_model()
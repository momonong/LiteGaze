import tensorflow as tf
import os
from models import build_student_v2

# === ⚙️ 設定 ===
# 確保路徑對應到您 Windows 的檔案位置
WEIGHTS_PATH = "models/litegaze_v2_best.h5" 
OUTPUT_PATH = "models/litegaze_v2_win.tflite"

def main():
    print(f"🖥️ 目前使用的 TensorFlow 版本 (Windows): {tf.__version__}")
    
    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ 找不到權重檔: {WEIGHTS_PATH}")
        return

    print("⏳ 重建模型架構...")
    # 這裡會用 Windows 版的 TF 來建立模型
    model = build_student_v2(input_shape=(60, 60, 3))
    
    print("📥 載入 WSL 訓練好的權重...")
    # HDF5 格式通常跨版本相容性很好，應該能順利載入
    model.load_weights(WEIGHTS_PATH)
    
    print("🔄 轉換為 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 這裡不需要特別設定什麼，因為 Converter 本身就是舊版的
    # 它自然會轉出舊版 Runtime 看得懂的格式
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    with open(OUTPUT_PATH, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ 成功！已使用 Windows 環境轉出: {OUTPUT_PATH}")
    print("👉 現在請修改 demo_v2.py 的 MODEL_PATH 指向這個新檔案！")

if __name__ == "__main__":
    main()
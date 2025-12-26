import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers, models, applications
from student_model import build_student_model

# 1. 匯入 Distiller 類別
try:
    from train_distil import Distiller
except ImportError:
    try:
        from train_distillation import Distiller
    except ImportError:
        print("❌ 錯誤：找不到 train_distil.py 或 train_distillation.py")
        exit()

# === 🔥 動態修補 Distiller (加上 call 方法) ===
def dummy_call(self, inputs):
    return self.student(inputs["student_input"])
Distiller.call = dummy_call
# ============================================

# === 設定 ===
MODEL_PATH = 'models/student_mobilenet_distilled.keras'
TFLITE_PATH = 'models/litegaze_student.tflite'

# === 關鍵：重建老師模型架構 (為了讓 load_weights 結構吻合) ===
def build_dummy_teacher():
    print("🏗️ 重建 Teacher 架構 (ResNet50V2)...")
    # 不需要載入 ImageNet 權重 (weights=None)，反正會被覆蓋
    # 結構必須跟訓練時一模一樣
    base_model = applications.ResNet50V2(
        include_top=False, 
        weights=None, 
        input_shape=(224, 224, 3)
    )
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, name='gaze_output')(x)
    return models.Model(inputs=base_model.input, outputs=outputs, name='Teacher_ResNet50')

def convert_model():
    if not os.path.exists(MODEL_PATH):
        print("❌ 找不到模型檔案，請確認路徑。")
        return

    print("🚀 開始模型轉換流程...")
    
    # 1. 重建學生
    student_model = build_student_model()
    
    # 2. 重建老師 (關鍵修正！)
    teacher_model = build_dummy_teacher()
    
    # 3. 組合蒸餾器
    distiller = Distiller(student=student_model, teacher=teacher_model)

    # 4. 初始化 (Build)
    print("🔧 初始化模型變數...")
    dummy_input = {
        "teacher_input": tf.zeros((1, 224, 224, 3)),
        "student_input": tf.zeros((1, 60, 60, 3))
    }
    distiller(dummy_input) 
    
    # 5. 載入權重 (現在結構完美對應，應該會成功)
    print(f"📥 從 {MODEL_PATH} 載入權重...")
    distiller.load_weights(MODEL_PATH)

    # 6. 取出學生
    print("💎 提取學生模型核心...")
    target_model = distiller.student

    # --- 轉換 TFLite ---
    print("🔄 轉換為 TFLite (FP16 量化)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(target_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    # 儲存
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
        
    print(f"✅ 轉換成功！檔案: {TFLITE_PATH}")
    size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
    print(f"📏 模型大小: {size_mb:.2f} MB")

if __name__ == "__main__":
    convert_model()
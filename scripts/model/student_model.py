import tensorflow as tf
from tensorflow.keras import layers, models, applications

# === 設定 ===
# 學生吃的是低解析度圖片
STUDENT_INPUT_SHAPE = (60, 60, 3) 

def build_student_model():
    """
    建立 LiteGaze 的學生模型 (基於 MobileNetV3-Small)
    修正：配合 Keras 預訓練權重的限制
    """
    print("🏗️ 正在建立 Student Model (MobileNetV3-Small)...")
    
    base_model = applications.MobileNetV3Small(
        input_shape=STUDENT_INPUT_SHAPE,
        include_top=False,
        weights='imagenet',
        alpha=0.75,        # 保持 0.75 以極致輕量化
        minimalistic=False # <--- 改成 False (標準版才有 0.75 的預訓練權重)
    )
    
    base_model.trainable = True

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(2, name='gaze_output')(x)
    
    model = models.Model(inputs=base_model.input, outputs=outputs, name='Student_MobileNetV3')
    
    return model

if __name__ == "__main__":
    # 測試架構並查看參數量 (應該要很小)
    model = build_student_model()
    model.summary()
    
    # 計算參數量與 ResNet50 (Teacher) 的差異
    # ResNet50 約 23M 參數
    # MobileNetV3-Small 約 1-2M 參數 -> 壓縮比 > 10x 達標！
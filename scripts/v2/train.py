import tensorflow as tf
import numpy as np
import os
from models import build_student_v2
from data_utils import load_data, augment_image

# === ⚙️ 設定區 ===
DATA_PATH = "data/mpiigaze_60x60.npz"  # 請確認您的 npz 路徑
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
CLIPNORM = 1.0
NUM_BINS = 90
# 角度範圍 (Radians) -1.57 ~ 1.57 (約 -90度 ~ 90度)
BIN_MIN = -1.57 
BIN_MAX = 1.57

def process_targets(image, label):
    """
    將單一的 label (Pitch, Yaw) 轉換成模型需要的三種輸出：
    1. gaze_out: 原本的連續數值 (用來算 MSE)
    2. pitch_logits: Pitch 屬於哪一個 Bin (用來算分類 Loss)
    3. yaw_logits: Yaw 屬於哪一個 Bin (用來算分類 Loss)
    """
    pitch = label[0]
    yaw = label[1]

    # 將連續角度映射到 0 ~ (NUM_BINS-1) 的整數索引
    # Normalize to 0.0 ~ 1.0
    p_norm = (pitch - BIN_MIN) / (BIN_MAX - BIN_MIN)
    y_norm = (yaw - BIN_MIN) / (BIN_MAX - BIN_MIN)
    
    # Scale to index
    p_idx = tf.cast(p_norm * (NUM_BINS - 1), tf.int32)
    y_idx = tf.cast(y_norm * (NUM_BINS - 1), tf.int32)
    
    # 限制範圍 (Clip) 避免超出去
    p_idx = tf.clip_by_value(p_idx, 0, NUM_BINS - 1)
    y_idx = tf.clip_by_value(y_idx, 0, NUM_BINS - 1)

    return image, {
        'gaze_out': label,     # 回歸任務
        'pitch_logits': p_idx, # 分類任務 (Pitch)
        'yaw_logits': y_idx    # 分類任務 (Yaw)
    }

def main():
    # 1. 載入資料
    print("📥 Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: 找不到 {DATA_PATH}，請把之前的 .npz 檔複製過來！")
        return

    images, labels = load_data(DATA_PATH)
    print(f"✅ Data loaded: {images.shape}, {labels.shape}")

    # 簡單切分 Train/Val
    split_idx = int(len(images) * 0.9)
    train_imgs, val_imgs = images[:split_idx], images[split_idx:]
    train_lbls, val_lbls = labels[:split_idx], labels[split_idx:]

    # 2. 建立 Pipeline
    # Train Set (加入 Augmentation + Target Processing)
    train_ds = tf.data.Dataset.from_tensor_slices((train_imgs, train_lbls))
    train_ds = train_ds.shuffle(1000)
    train_ds = train_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE) # 先做影像增強
    train_ds = train_ds.map(process_targets, num_parallel_calls=tf.data.AUTOTUNE) # 再做標籤轉換
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Val Set (不做 Augmentation，但要做 Target Processing)
    val_ds = tf.data.Dataset.from_tensor_slices((val_imgs, val_lbls))
    val_ds = val_ds.map(lambda x, y: (tf.image.resize(x, (60,60)), y)) # 確保尺寸正確
    val_ds = val_ds.map(process_targets, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # 3. 建立模型
    print("🏗️ Building LiteGaze V2 Model...")
    model = build_student_v2(input_shape=(60, 60, 3))
    model.summary()

    # 4. 編譯 (Compile)
    # L2CS 核心：同時優化分類準確度 (CrossEntropy) 和 回歸準確度 (MSE)
    losses = {
        'gaze_out': 'mse',
        'pitch_logits': 'sparse_categorical_crossentropy',
        'yaw_logits': 'sparse_categorical_crossentropy'
    }
    
    # 權重分配：通常分類 Loss 比較大，給它 1.0，MSE 給小一點或相等
    loss_weights = {
        'gaze_out': 1.0,     # 回歸權重
        'pitch_logits': 1.0, # Pitch 分類權重
        'yaw_logits': 1.0    # Yaw 分類權重
    }

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIPNORM),
        loss=losses,
        loss_weights=loss_weights,
        metrics={'gaze_out': 'mae'} # 我們主要看 MAE (平均絕對誤差)
    )

    # 5. 開始訓練
    print("🚀 Start Training...")
    callbacks = [
        # ✅ 加入 mode='min'，告訴它誤差越小越好
        tf.keras.callbacks.ModelCheckpoint(
            "litegaze_v2_best.h5", 
            save_best_only=True, 
            monitor='val_gaze_out_mae', 
            mode='min' 
        ),
        # ✅ 這裡也要加
        tf.keras.callbacks.EarlyStopping(
            patience=5, 
            monitor='val_gaze_out_mae', 
            restore_best_weights=True, 
            mode='min'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, monitor='val_gaze_out_mae', mode='min')
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # 6. 轉出 TFLite
    print("📦 Exporting to TFLite...")
    # 載入最好的權重
    model.load_weights("litegaze_v2_best.h5")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT] # 預設量化
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    
    tflite_model = converter.convert()
    
    with open('litegaze_v2.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("🎉 Done! Model saved to 'litegaze_v2.tflite'")

if __name__ == '__main__':
    main()
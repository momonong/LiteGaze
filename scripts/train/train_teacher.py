import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
import random

# === [🔥 5090 修正: 開啟 GPU 記憶體增長] ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ 5090 記憶體動態增長已開啟")
    except RuntimeError as e:
        print(e)

# === [設定區] ===
PROCESSED_DIR = './data/processed'
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
INPUT_SHAPE = (224, 224, 3)

# === [1. 新版資料管線: 使用 tf.data] ===
def gaze_data_generator():
    """
    這是一個 Python Generator，負責從硬碟隨機讀取 npz
    """
    # 搜尋所有檔案
    files = glob.glob(os.path.join(PROCESSED_DIR, '*', '*.npz'))
    if not files:
        raise ValueError("找不到資料！請確認 preprocess.py 是否執行完成。")
        
    random.shuffle(files) # 每個 epoch 開始前洗牌
    
    for file_path in files:
        try:
            with np.load(file_path) as data:
                # 讀取 Teacher 圖片 (uint8) 和 Label
                images = data['teacher'] 
                labels = data['label']
            
            # 這裡我們一次 yield 一張圖，讓 tf.data 去負責組裝 batch
            # 這樣更靈活，且能利用 tf.data 的並行優勢
            for i in range(len(images)):
                yield images[i], labels[i]
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

def create_dataset():
    """
    將 Python Generator 轉換為高效能的 tf.data.Dataset
    """
    # 定義輸出的資料格式 (圖片 224x224x3, 標籤 2)
    output_signature = (
        tf.TensorSpec(shape=(224, 224, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(2,), dtype=tf.float32)
    )

    # 1. 建立 Dataset
    dataset = tf.data.Dataset.from_generator(
        gaze_data_generator,
        output_signature=output_signature
    )

    # 2. 資料增強與預處理 (這裡可以開多核心並行!)
    def preprocess(img, label):
        # 轉 float 並歸一化 (0~1)
        img = tf.cast(img, tf.float32) / 255.0
        # 確保形狀正確
        img = tf.ensure_shape(img, INPUT_SHAPE)
        return img, label

    # === 🚀 效能全開關鍵 ===
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) # 多核心同時處理
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE) # GPU 算圖時，CPU 預先讀下一批
    
    return dataset

# === [2. 模型定義] ===
def build_teacher_model():
    print("正在建立 ResNet50V2 模型...")
    base_model = applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=INPUT_SHAPE
    )
    base_model.trainable = True 

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, name='gaze_output')(x)
    
    model = models.Model(inputs=base_model.input, outputs=outputs, name='Teacher_ResNet50')
    return model

# === [3. 主程式] ===
if __name__ == "__main__":
    # 建立 Dataset
    print("🚀 正在建構 tf.data 高速管線...")
    train_ds = create_dataset()
    
    # 簡單算一下步數 (為了顯示進度條)
    # 估計：檔案數 * 1000張 / 32
    num_files = len(glob.glob(os.path.join(PROCESSED_DIR, '*', '*.npz')))
    steps_per_epoch = (num_files * 500) // BATCH_SIZE # 保守估計每檔500張
    print(f"預估每個 Epoch 需要跑 {steps_per_epoch} 步")

    # 建立模型
    model = build_teacher_model()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath='models/teacher_resnet50_best.keras',
        save_best_only=True,
        monitor='loss',
        mode='min'
    )

    # 開始訓練
    # 注意：這裡不再需要 workers 參數，因為 tf.data 自動搞定了
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks=[checkpoint_cb]
    )
    
    print("訓練完成！")
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses
import random
from student_model import build_student_model # 匯入我們定義好的學生模型

# === [🔥 5090 記憶體設定] ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# === [設定區] ===
PROCESSED_DIR = './data/processed'
TEACHER_MODEL_PATH = 'models/teacher_resnet50_best.keras' # 假設組員傳給你的檔名
BATCH_SIZE = 64 # 學生模型比較小，Batch 可以開大一點
EPOCHS = 30
LEARNING_RATE = 1e-3

# 蒸餾權重 (關鍵參數)
ALPHA = 0.5  # 0.5 表示：一半聽老師的，一半看標準答案

# === [1. 雙輸入資料管線] ===
def distillation_generator():
    files = glob.glob(os.path.join(PROCESSED_DIR, '*', '*.npz'))
    random.shuffle(files)
    
    for file_path in files:
        try:
            with np.load(file_path) as data:
                img_t = data['teacher'] # 224x224
                img_s = data['student'] # 60x60
                labels = data['label']
            
            # 同時吐出 (Teacher圖, Student圖, 標籤)
            for i in range(len(img_t)):
                yield (img_t[i], img_s[i]), labels[i]
                
        except Exception:
            continue

def create_distillation_dataset():
    # 定義格式: ((T_img, S_img), Label)
    output_signature = (
        (
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(60, 60, 3), dtype=tf.uint8)
        ),
        tf.TensorSpec(shape=(2,), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(
        distillation_generator,
        output_signature=output_signature
    )

    def preprocess(inputs, label):
        t_img, s_img = inputs
        # 歸一化
        t_img = tf.cast(t_img, tf.float32) / 255.0
        s_img = tf.cast(s_img, tf.float32) / 255.0
        
        # 回傳字典格式，讓 Model 比較好讀
        return {"teacher_input": t_img, "student_input": s_img}, label

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# === [2. 定義蒸餾模型 (Distiller Class)] ===
class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.1):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha

    def train_step(self, data):
        # Unpack data
        x, y_true = data # x 是一個字典 {'teacher_input': ..., 'student_input': ...}
        
        # Teacher 只做預測，不訓練 (Forward pass only)
        # 注意：Teacher 在訓練模式下通常會關閉 Dropout，這裡我們設 training=False
        teacher_predictions = self.teacher(x['teacher_input'], training=False)

        with tf.GradientTape() as tape:
            # Student 進行預測
            student_predictions = self.student(x['student_input'], training=True)

            # 計算兩種 Loss
            # 1. Student vs Ground Truth (標準答案)
            loss_student = self.student_loss_fn(y_true, student_predictions)
            
            # 2. Student vs Teacher (老師的指導)
            loss_distillation = self.distillation_loss_fn(teacher_predictions, student_predictions)

            # 3. 總 Loss (加權平均)
            total_loss = self.alpha * loss_student + (1 - self.alpha) * loss_distillation

        # 計算梯度並更新 Student 的權重
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # 更新監控指標
        self.compiled_metrics.update_state(y_true, student_predictions)
        
        # 回傳當下的 Loss 給進度條顯示
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": loss_student, "dist_loss": loss_distillation})
        return results

# === [3. 主程式] ===
if __name__ == "__main__":
    print("🚀 準備開始知識蒸餾 (Knowledge Distillation)...")
    
    # 1. 檢查是否有老師模型
    if not os.path.exists(TEACHER_MODEL_PATH):
        print(f"⚠️  警告：找不到老師模型 {TEACHER_MODEL_PATH}")
        print("請等待組員傳送 'teacher_resnet50_best.keras' 後再執行此程式。")
        # 這裡為了不報錯退出，我們先用假的 Teacher 代替 (僅供測試流程)
        # print(">>> 測試模式：使用未訓練的 Teacher 進行模擬 <<<")
        # teacher_model = tf.keras.applications.ResNet50V2(input_shape=(224,224,3), classes=2, weights=None, classifier_activation=None)
        exit() # 正式執行請把這行留著，沒老師不能跑
    else:
        print("✅ 載入老師模型...")
        teacher_model = models.load_model(TEACHER_MODEL_PATH)
        # 凍結老師，不讓他更新 (他已經出師了)
        teacher_model.trainable = False 

    # 2. 建立全新的學生模型
    student_model = build_student_model()
    
    # 3. 建立蒸餾器
    distiller = Distiller(student=student_model, teacher=teacher_model)
    
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=['mae'],
        student_loss_fn=losses.MeanSquaredError(),
        distillation_loss_fn=losses.MeanSquaredError(),
        alpha=ALPHA
    )

    # 4. 準備資料
    print("📥 建立資料管線...")
    train_ds = create_distillation_dataset()
    
    # 計算步數
    num_files = len(glob.glob(os.path.join(PROCESSED_DIR, '*', '*.npz')))
    steps_per_epoch = (num_files * 800) // BATCH_SIZE 

    # 5. 開始訓練
    print("🔥 開始蒸餾訓練 (Teacher -> Student)...")
    
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath='models/student_mobilenet_distilled.keras',
        save_best_only=True,
        monitor='mae', # 監控學生本身的準確度
        mode='min'
    )

    distiller.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks=[checkpoint_cb]
    )
    
    print("🎉 蒸餾完成！學生模型已儲存。")
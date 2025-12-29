import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
import numpy as np
import h5py
import os
import sys  # <--- 新增這行

sys.path.append(os.getcwd()) 
# === 引入您的模型建構函式 ===
# 這樣就不會報錯了
from models import build_teacher_v3, build_student_v2

# === ⚙️ 設定區 (Rapid Mode) ===
TEACHER_WEIGHTS_PATH = 'models/teacher_v3_best_5090.h5' # 剛剛練好的老師權重檔名 (請確認您的檔名)
DATA_PATH = 'data/teacher_224.h5'      # 繼續用這份資料
BATCH_SIZE = 16                        # 5090 跑 16 很穩
EPOCHS = 3                             # 學生比較笨，給他 3 輪 (約 3-4 小時)
LR = 1e-4                              # 學生是從頭練，學習率可以正常點
ALPHA = 0.5                            # 0.5 學老師，0.5 學標準答案
TEMPERATURE = 3.0                      # 蒸餾溫度

# === 準備模型類別 (Distiller) ===
class DistillModel(models.Model):
    def __init__(self, student, teacher):
        super(DistillModel, self).__init__()
        self.student = student
        self.teacher = teacher

    def compile(self, optimizer, metrics, student_loss_fn, distill_loss_fn, alpha=0.1, temperature=3):
        super(DistillModel, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distill_loss_fn = distill_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        x, y = data
        
        # 1. 準備輸入
        # 老師看高清 RGB
        teacher_input = x 
        
        # 學生看低清，並且轉成黑白！
        student_input_resize = tf.image.resize(x, (60, 60))
        student_input = tf.image.rgb_to_grayscale(student_input_resize)

        # 2. 老師先看 (Training=False)
        teacher_pred = self.teacher(teacher_input, training=False)
        t_pitch_logits = teacher_pred[1]
        t_yaw_logits = teacher_pred[2]

        # 3. 學生學習
        with tf.GradientTape() as tape:
            # 這裡餵進去的 student_input 現在是 (Batch, 60, 60, 1) 了，模型就不會報錯
            student_pred = self.student(student_input, training=True)
            
            s_gaze = student_pred[0]
            s_pitch_logits = student_pred[1]
            s_yaw_logits = student_pred[2]

            # Loss A: Gaze Vector MSE
            gaze_loss = self.student_loss_fn(y['gaze_out'], s_gaze)

            # Loss B: Distillation KL
            t_pitch_soft = tf.nn.softmax(t_pitch_logits / self.temperature)
            s_pitch_soft = tf.nn.softmax(s_pitch_logits / self.temperature)
            t_yaw_soft = tf.nn.softmax(t_yaw_logits / self.temperature)
            s_yaw_soft = tf.nn.softmax(s_yaw_logits / self.temperature)

            distill_loss_pitch = self.distill_loss_fn(t_pitch_soft, s_pitch_soft)
            distill_loss_yaw = self.distill_loss_fn(t_yaw_soft, s_yaw_soft)
            
            total_distill_loss = distill_loss_pitch + distill_loss_yaw

            # 總損失
            loss = (1 - self.alpha) * gaze_loss + (self.alpha) * total_distill_loss

        # 4. 更新權重
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # 5. 更新 Metrics
        self.compiled_metrics.update_state(y['gaze_out'], s_gaze)
        
        return {
            "loss": loss, 
            "gaze_loss": gaze_loss, 
            "distill_loss": total_distill_loss
        }

# === 資料生成器 (跟之前一樣) ===
def hdf5_generator(path, batch_size):
    while True:
        with h5py.File(path, 'r') as hf:
            images_dset = hf['images']
            labels_dset = hf['labels']
            total_len = images_dset.shape[0]
            indices = np.arange(total_len)
            np.random.shuffle(indices)
            
            for i in range(0, total_len, batch_size):
                batch_idx = np.sort(indices[i : i + batch_size])
                # 讀取 224 圖 (之後在 Model 內縮放)
                batch_imgs = images_dset[batch_idx].astype(np.float32) / 255.0
                batch_lbls = labels_dset[batch_idx]
                
                yield batch_imgs, {'gaze_out': batch_lbls}

def main():
    print("🚀 Loading Teacher Model weights...")
    # 1. 建立老師並載入權重
    teacher_model = build_teacher_v3()
    # 確保這裡的檔名跟您剛剛存的一樣 (可能是 model.save 預設的 teacher_v3.h5 或其他)
    if os.path.exists(TEACHER_WEIGHTS_PATH):
        teacher_model.load_weights(TEACHER_WEIGHTS_PATH)
        print("✅ Teacher weights loaded!")
    else:
        print(f"❌ Error: Cannot find {TEACHER_WEIGHTS_PATH}")
        return
    
    # 凍結老師 (不訓練他)
    teacher_model.trainable = False

    # 2. 建立全新的學生
    print("👶 Creating Student Model...")
    student_model = build_student_v2()

    # 3. 準備資料
    # 使用跟剛剛一樣的邏輯，只取部分資料加速
    limit_samples = 42000 # 跟剛剛一樣 10%
    
    output_signature = (
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        {'gaze_out': tf.TensorSpec(shape=(None, 2), dtype=tf.float32)}
    )
    
    ds = tf.data.Dataset.from_generator(
        lambda: hdf5_generator(DATA_PATH, BATCH_SIZE),
        output_signature=output_signature
    )
    ds = ds.take(limit_samples).prefetch(tf.data.AUTOTUNE)

    # 4. 建立蒸餾器
    distiller = DistillModel(student=student_model, teacher=teacher_model)
    
    distiller.compile(
        optimizer=optimizers.Adam(learning_rate=LR),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
        student_loss_fn=losses.MeanSquaredError(),
        distill_loss_fn=losses.KLDivergence(),
        alpha=ALPHA,
        temperature=TEMPERATURE
    )

    # 5. 開始蒸餾訓練！
    print(f"🔥 Start Distillation for {EPOCHS} epochs...")
    distiller.fit(ds, epochs=EPOCHS, steps_per_epoch=limit_samples // BATCH_SIZE)

    # ... (前面的訓練 fit 程式碼) ...

    # 6. 正確存檔流程
    print("💾 Saving Distilled Student...")
    
    # 步驟 A: 先存成 Keras H5 格式 (這是最重要的，有了這個隨時可以轉 TFLite)
    # include_optimizer=False 可以讓檔案小一點，預測時不需要優化器
    h5_path = "models/litegaze_distilled_final.h5"
    student_model.save(h5_path, include_optimizer=False)
    print(f"✅ H5 model saved: {h5_path}")
    
    # 步驟 B: 手動轉換成 TFLite
    print("⚙️ Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(student_model)
    
    # (選用) 針對 TF operator 的額外設定，防止某些層轉不過去
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # Enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS    # Enable TensorFlow ops.
    ]
    
    tflite_model = converter.convert()
    
    tflite_path = 'models/litegaze_v2_distilled.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ TFLite model generated: {tflite_path}")

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import random

# === 設定 ===
PROCESSED_DIR = './data/processed'

def inspect_data():
    # 1. 隨機抓一個檔案
    files = glob.glob(os.path.join(PROCESSED_DIR, '*', '*.npz'))
    if not files:
        print("❌ 找不到任何資料！")
        return

    target_file = random.choice(files)
    print(f"🕵️‍♂️ 正在檢查檔案: {target_file}")

    # 2. 讀取內容
    data = np.load(target_file)
    teacher_imgs = data['teacher'] # uint8
    student_imgs = data['student'] # uint8
    labels = data['label']         # float32

    print(f"📊 該檔案包含樣本數: {len(teacher_imgs)}")
    print(f"   Teacher Shape: {teacher_imgs.shape} (預期: N, 224, 224, 3)")
    print(f"   Student Shape: {student_imgs.shape} (預期: N, 60, 60, 3)")
    print(f"   Label Shape:   {labels.shape}       (預期: N, 2)")
    print(f"   Label 範圍:    Min {labels.min():.4f} / Max {labels.max():.4f}")

    # 3. 隨機畫出一張圖來看看
    idx = random.randint(0, len(teacher_imgs) - 1)
    
    img_t = teacher_imgs[idx]
    img_s = student_imgs[idx]
    label = labels[idx] # [Pitch, Yaw]

    # 畫圖
    plt.figure(figsize=(10, 5))
    
    # 左邊：Teacher
    plt.subplot(1, 2, 1)
    plt.imshow(img_t)
    plt.title(f"Teacher (224x224)\nPitch: {label[0]:.2f}, Yaw: {label[1]:.2f}")
    plt.axis('off')

    # 右邊：Student
    plt.subplot(1, 2, 2)
    plt.imshow(img_s)
    plt.title(f"Student (60x60)\nLow Res")
    plt.axis('off')

    plt.show()
    print("✅ 檢查完畢！如果圖片看起來像眼睛，那就沒問題了。")

if __name__ == "__main__":
    inspect_data()
import os
import glob
import numpy as np
import cv2
import scipy.io as sio
from multiprocessing import Pool, cpu_count, current_process
from tqdm import tqdm

# === [設定區] ===
DATA_ROOT = './data/MPIIGaze/Data/Normalized'
OUTPUT_DIR = './data/processed'
TEACHER_SIZE = (224, 224)
STUDENT_SIZE = (60, 60)

# 輔助函式：將 3D 向量轉 2D
def vector_to_angle(gaze_vector):
    x, y, z = gaze_vector
    pitch = np.arcsin(-y)
    yaw = np.arctan2(-x, -z)
    return np.array([pitch, yaw])

# === [核心工作函式] ===
# 這是每一個 CPU 核心具體要做的工作
def process_single_mat(args):
    mat_file, save_path = args
    
    # 如果檔案已存在，直接跳過
    if os.path.exists(save_path):
        return 0

    try:
        # 讀取 .mat
        mat = sio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
        data = mat['data']

        images_teacher = []
        images_student = []
        labels = []

        eyes = []
        if hasattr(data, 'left'): eyes.append(('left', data.left))
        if hasattr(data, 'right'): eyes.append(('right', data.right))

        for side, eye_data in eyes:
            imgs = eye_data.image
            gazes = eye_data.gaze
            
            # 處理單張圖的情況
            if len(imgs.shape) == 2:
                imgs = imgs[np.newaxis, :, :]
                gazes = gazes[np.newaxis, :]

            for i in range(len(imgs)):
                img = imgs[i]
                gaze = gazes[i]

                # 鏡像與角度處理
                if side == 'right':
                    img = cv2.flip(img, 1)
                    gaze_angle = vector_to_angle(gaze)
                    gaze_angle[1] = -gaze_angle[1]
                else:
                    gaze_angle = vector_to_angle(gaze)

                # 轉 RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                # Resize (uint8)
                img_t = cv2.resize(img_rgb, TEACHER_SIZE)
                img_s = cv2.resize(img_rgb, STUDENT_SIZE)
                
                images_teacher.append(img_t)
                images_student.append(img_s)
                labels.append(gaze_angle)

        # 存檔
        if len(images_teacher) > 0:
            np.savez_compressed(
                save_path,
                teacher=np.array(images_teacher, dtype=np.uint8),
                student=np.array(images_student, dtype=np.uint8),
                label=np.array(labels, dtype=np.float32)
            )
            return 1 # 成功處理一個檔案
        return 0

    except Exception as e:
        # 多工模式下 print 比較亂，通常建議 pass 或寫 log，這裡簡單印出
        print(f"Error in {mat_file}: {e}")
        return 0

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 收集所有任務
    print("🔍 正在掃描檔案列表...")
    subjects = [f'p{i:02d}' for i in range(15)]
    tasks = []

    for subject_id in subjects:
        # 預先建立好資料夾，避免多核心同時建立導致衝突
        save_dir = os.path.join(OUTPUT_DIR, subject_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        subject_path = os.path.join(DATA_ROOT, subject_id)
        mat_files = glob.glob(os.path.join(subject_path, '*.mat'))
        
        for f in mat_files:
            base_name = os.path.splitext(os.path.basename(f))[0]
            save_path = os.path.join(save_dir, f"{base_name}.npz")
            tasks.append((f, save_path))

    print(f"📋 總共發現 {len(tasks)} 個檔案，準備開始多核心處理...")
    
    # 2. 開啟多核心處理 (Pool)
    # cpu_count() 會自動抓您有幾個核心
    workers = cpu_count()
    print(f"🔥 火力全開！啟動 {workers} 個核心同時運算...")

    with Pool(processes=workers) as pool:
        # 使用 tqdm 顯示進度條
        results = list(tqdm(pool.imap_unordered(process_single_mat, tasks), total=len(tasks)))

    print(f"\n✅ 全部完成！")

if __name__ == '__main__':
    # Windows/WSL 多工必須放在 main 區塊下
    main()
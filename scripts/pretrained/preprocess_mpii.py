import cv2
import os
import glob
from pathlib import Path
import numpy as np

# ================= ⚙️ 設定區 =================
# 你的原始資料路徑
SOURCE_ROOT = r'data\MPIIGaze\Data\Original'
# 輸出路徑
OUTPUT_DIR = 'data/distill_images'
# ============================================

def main():
    # 1. 建立輸出資料夾
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. 初始化 OpenCV Haar Cascade (內建的人臉偵測器)
    # OpenCV 通常自帶這些 xml 模型檔
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    if face_cascade.empty():
        print("❌ Error: 無法載入 Haar Cascade 模型，請確認 OpenCV 安裝完整。")
        return

    # 3. 搜尋所有圖片
    print(f"🔍 Scanning {SOURCE_ROOT}...")
    image_paths = list(Path(SOURCE_ROOT).rglob("*.jpg")) + list(Path(SOURCE_ROOT).rglob("*.png"))
    
    print(f"📊 Found {len(image_paths)} images. Switching to OpenCV detection...")
    
    count = 0
    
    for i, img_path in enumerate(image_paths):
        try:
            # 讀取圖片
            frame = cv2.imread(str(img_path))
            if frame is None: continue

            h, w, _ = frame.shape
            
            # Haar 需要轉灰階才能偵測
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 偵測人臉 (參數調整以減少誤判)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # 如果有偵測到臉
            if len(faces) > 0:
                # 假設最大的那個是主角
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                x, y, w_face, h_face = faces[0]
                
                # 擴大一點 (Padding)
                k = 0.2
                x_min = max(0, x - int(w_face * k))
                y_min = max(0, y - int(h_face * k))
                x_max = min(w, x + w_face + int(w_face * 2 * k))
                y_max = min(h, y + h_face + int(h_face * 2 * k))
                
                # 裁切
                face_img = frame[y_min:y_max, x_min:x_max]
                
                if face_img.size > 0:
                    # 檔名處理 (p00_day01_0001.jpg)
                    # 使用 parents 來確保跨平台路徑相容
                    p_folder = img_path.parent.parent.name
                    day_folder = img_path.parent.name
                    file_name = img_path.name
                    
                    save_name = f"{p_folder}_{day_folder}_{file_name}"
                    cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), face_img)
                    count += 1
                        
        except Exception as e:
            # 遇到壞圖就跳過，不中斷
            print(f"⚠️ Error processing {img_path}: {e}")
            
        if i % 100 == 0:
            print(f"⏳ Processed {i}/{len(image_paths)} | Saved: {count} faces", end='\r')

    print(f"\n✅ Done! Saved {count} face images to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
import torch
import cv2
import os
import numpy as np
from l2cs import Pipeline, render
import sys
import time

# ================= ⚙️ 設定 =================
OUTPUT_DIR = 'data/selfmade_sisi'  # 這是我們的終極資料集
MODEL_PATH = 'models/L2CSNet_gaze360.pkl'
DEVICE = torch.device('cuda')
TARGET_COUNT = 3000  # 目標收集 3000 張
# =======================================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"🚀 Loading Teacher Pipeline...")
    # 這是為了確保「裁切邏輯」跟老師一模一樣
    gaze_pipeline = Pipeline(
        weights=MODEL_PATH, arch='ResNet50', device=DEVICE
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n🎮 【完美資料採集模式】")
    print("1. 請按 [SPACE] 開始/暫停 錄製。")
    print("2. 請做各種動作：轉頭、抬頭、低頭、靠近、遠離。")
    print("3. 眼睛請盯著螢幕上的不同位置，或者跟著手指動。")
    print("⚠️ 只有當綠色箭頭準確時，才讓它錄製！")
    
    count = 0
    recording = False
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. 讓老師看
        # 我們需要修改 pipeline 讓它回傳 logits 嗎？
        # 其實不用，我們直接存圖，訓練時再讓老師即時算 Logits 就好
        # 這樣可以省硬碟空間，而且可以做 Data Augmentation
        
        # 這裡我們只用 pipeline 來取得 "BBox" 以便裁切
        try:
            results = gaze_pipeline.step(frame)
        except: continue

        frame_vis = render(frame.copy(), results)
        
        # 2. 裁切邏輯 (這是關鍵！必須跟 Teacher 一致)
        if results.bboxes is not None and len(results.bboxes) > 0:
            bbox = results.bboxes[0]
            x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # 安全邊界
            h, w, _ = frame.shape
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)
            
            # 取得裁切圖
            face_img = frame[y_min:y_max, x_min:x_max]
            
            if face_img.size > 0:
                if recording:
                    # 存檔
                    filename = f"{OUTPUT_DIR}/img_{count:05d}.jpg"
                    cv2.imwrite(filename, face_img)
                    count += 1
                    
                    # 錄影指示燈 (紅點)
                    cv2.circle(frame_vis, (50, 50), 20, (0, 0, 255), -1)
                    cv2.putText(frame_vis, "REC", (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # UI
        cv2.putText(frame_vis, f"Count: {count}/{TARGET_COUNT}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if not recording:
            cv2.putText(frame_vis, "Press SPACE to Record", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
        cv2.imshow("Perfect Dataset Collector", frame_vis)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32: # SPACE
            recording = not recording
        elif key == ord('q'):
            break
        
        if count >= TARGET_COUNT:
            print("✅ 收集完成！")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
import cv2
import numpy as np
import tensorflow as tf
import os
import sys

# ================= 🔍 路徑設定 =================
# 自動抓取 v3 資料夾路徑
current_script_path = os.path.abspath(__file__)
demo_dir = os.path.dirname(current_script_path)
scripts_dir = os.path.dirname(demo_dir)
v3_dir = os.path.join(scripts_dir, 'v3')
project_root = os.path.dirname(scripts_dir)

if v3_dir not in sys.path:
    sys.path.insert(0, v3_dir)

try:
    from models import build_teacher_v3
except ImportError:
    print("❌ Error: Cannot import models.py")
    exit()

# ================= ⚙️ 設定 =================
TEACHER_MODEL_PATH = "models/teacher_v3_best_A100.h5" 
INPUT_SIZE = 224
# ==========================================

def load_and_predict(model, image_path):
    if not os.path.exists(image_path):
        print(f"⚠️ 找不到圖片: {image_path}")
        return

    # 1. 讀取圖片
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ 無法讀取: {image_path}")
        return

    h, w, _ = img.shape

    # 2. 暴力裁切中間 (模擬人臉偵測)
    # 假設臉在正中間，裁切 50% 的區域
    center_y, center_x = h // 2, w // 2
    crop_h, crop_w = h // 2, w // 2 # 裁切大小
    y1 = max(0, center_y - crop_h // 2)
    y2 = min(h, center_y + crop_h // 2)
    x1 = max(0, center_x - crop_w // 2)
    x2 = min(w, center_x + crop_w // 2)
    
    face_crop = img[y1:y2, x1:x2]

    # 3. 預處理 (Resize -> RGB -> Normalize)
    face_resized = cv2.resize(face_crop, (INPUT_SIZE, INPUT_SIZE))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    input_tensor = face_rgb.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # 4. 推論
    print(f"🔍 分析 {image_path} 中...")
    preds = model.predict(input_tensor, verbose=0)
    gaze = preds[0][0]
    pitch, yaw = gaze[0], gaze[1]

    print(f"   👉 結果: Pitch(上下)={pitch:.4f}, Yaw(左右)={yaw:.4f}")
    return pitch, yaw

def main():
    # 1. 載入模型
    print("👨‍🏫 正在載入老師模型 (CPU)...")
    model = build_teacher_v3()
    full_model_path = os.path.join(project_root, TEACHER_MODEL_PATH)
    
    if not os.path.exists(full_model_path):
        print(f"❌ 模型檔案不存在: {full_model_path}")
        exit()
        
    model.load_weights(full_model_path)
    print("✅ 模型載入完成！")
    print("-" * 30)

    # 2. 測試三張圖
    # 請確保你有拍這三張照片並放在專案根目錄 (或是修改這裡的路徑)
    img_center = "pictures/center.png"
    img_left = "pictures/left.png"
    img_right = "pictures/right.png"

    # 執行測試
    p_c, y_c = load_and_predict(model, img_center) or (0,0)
    p_l, y_l = load_and_predict(model, img_left) or (0,0)
    p_r, y_r = load_and_predict(model, img_right) or (0,0)

    print("-" * 30)
    print("📊 【最終診斷報告】")
    
    # 計算差異 (Range)
    yaw_diff = abs(y_l - y_r)
    
    if yaw_diff < 0.1:
        print("🔴 結果: [FAIL] 模型幾乎沒有反應 (Mode Collapse)")
        print("   原因: 老師可能沒練好，或是只學會猜平均值。")
    else:
        print(f"🟢 結果: [PASS] 模型有反應！(左右差異 {yaw_diff:.2f})")
        print("   建議: 模型是好的！問題出在 Demo 的座標映射或校正。")
        
        # 簡單的方向判斷
        if y_l < y_r:
            print("   觀測: 數值隨視線向右而變大 (正相關)")
        else:
            print("   觀測: 數值隨視線向右而變小 (負相關)")

if __name__ == "__main__":
    main()
from l2cs import Pipeline, render
import cv2
import torch
import os
import numpy as np
import sys

# ================= ⚙️ 設定區 =================
OUTPUT_DIR = 'data/official_calibration'
CWD = os.getcwd()
MODEL_PATH = os.path.join(CWD, 'models', 'L2CSNet_gaze360.pkl')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# ============================================

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 錯誤：找不到模型 {MODEL_PATH}")
        return
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"🚀 Loading Official Pipeline on {DEVICE}...")
    gaze_pipeline = Pipeline(
        weights=MODEL_PATH,
        arch='ResNet50',
        device=DEVICE
    )
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 動作指令
    actions = ["LOOK UP (看上面)", "LOOK DOWN (看下面)", "LOOK LEFT (看左邊)", "LOOK RIGHT (看右邊)", "CENTER (看中間)"]
    action_idx = 0
    samples_per_action = 100
    current_samples = 0
    collecting = False

    print("\n🎮 【官方邏輯採集模式】")
    print("這一次，我們用官方的演算法來抓臉，保證準！")
    print("請按【空白鍵】開始/暫停收集。")

    while action_idx < len(actions):
        ret, frame = cap.read()
        if not ret: break
        
        # 1. 讓官方 Pipeline 幫我們算 (包含偵測臉 + 預測視線)
        # results 包含: pitch, yaw, bboxes, landmarks, scores
        try:
            results = gaze_pipeline.step(frame)
        except Exception as e:
            cv2.imshow("Collector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue
            
        # 繪製預覽 (讓我們知道現在準不準)
        frame_vis = render(frame.copy(), results)
        
        # 如果有偵測到臉
        if results.bboxes is not None and len(results.bboxes) > 0:
            # 抓出最大的臉
            bbox = results.bboxes[0] # [x_min, y_min, x_max, y_max]
            pitch = results.pitch[0]
            yaw = results.yaw[0]
            
            # 取得座標 (官方的 Bbox 座標)
            x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # 安全邊界檢查
            h, w, _ = frame.shape
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)
            
            # 裁切出這張「官方認證」的臉
            face_img = frame[y_min:y_max, x_min:x_max]
            
            if collecting and face_img.size > 0:
                # 存檔
                filename = f"{OUTPUT_DIR}/act{action_idx}_{current_samples:04d}.jpg"
                cv2.imwrite(filename, face_img)
                current_samples += 1
                
                # 在畫面上給個紅點回饋
                cv2.circle(frame_vis, (30, 30), 15, (0, 0, 255), -1)
                
                if current_samples >= samples_per_action:
                    collecting = False
                    action_idx += 1
                    current_samples = 0
                    print(f"✅ 完成動作: {actions[action_idx-1]}")

        # UI 顯示
        if action_idx < len(actions):
            msg = f"DO: {actions[action_idx]}"
            status = f"Collected: {current_samples}/{samples_per_action}"
            cv2.putText(frame_vis, msg, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame_vis, status, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.putText(frame_vis, "ALL DONE! Press Q", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Collector", frame_vis)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32: # Space
            collecting = not collecting
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"🎉 資料收集完成！請到 {OUTPUT_DIR} 確認。")

if __name__ == '__main__':
    main()
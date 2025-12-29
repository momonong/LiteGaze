import onnxruntime as ort
import cv2
import numpy as np
import time
from l2cs import Pipeline
import torch

# ================= ⚙️ 設定 =================
ONNX_MODEL_PATH = 'models/litegaze_student_fp32.onnx'
TEACHER_PATH = 'models/L2CSNet_gaze360.pkl' 
# ==========================================

# 模擬螢幕解析度
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

# 熱圖參數
HEATMAP_DECAY_RATE = 0.95 # 每幀熱圖衰減速度
HEATMAP_BRIGHTNESS = 25  # 每個新視線點的亮度
HEATMAP_RADIUS = 30      # 每個新視線點的半徑

def compute_gaze_np(logits):
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    idx = np.arange(90)
    gaze = np.sum(probs * idx) * 4 - 180
    return gaze

# 全局變量，用於熱圖
heatmap_data = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.float32)

def main():
    # 1. 優化 ONNX Session 設定
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(ONNX_MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    
    # 2. 初始化偵測器
    print("👀 啟動 GPU 人臉偵測器...")
    detector = Pipeline(weights=TEACHER_PATH, arch='ResNet50', device=torch.device('cuda'))

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n🚀 熱圖 Demo 啟動！按下 'q' 退出。")
    
    fps_time = time.time()
    frame_count = 0
    
    bbox = None 

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        if time.time() - fps_time > 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = time.time()

        # 每 2 幀偵測一次臉部
        if frame_count % 2 == 0 or bbox is None:
            try:
                results = detector.step(frame)
                if results.bboxes is not None and len(results.bboxes) > 0:
                    bbox = results.bboxes[0]
            except: pass

        gaze_x_screen, gaze_y_screen = None, None

        if bbox is not None:
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            h, w, _ = frame.shape
            x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), min(w, x_max), min(h, y_max)
            
            face_img = frame[y_min:y_max, x_min:x_max]
            
            if face_img.size > 0:
                # 推理預處理
                img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img = img.astype(np.float32) / 255.0
                img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                img = img.transpose(2, 0, 1)[np.newaxis, ...]

                # 視線推理
                outputs = sess.run(None, {input_name: img.astype(np.float32)})
                pitch = compute_gaze_np(outputs[0][0])
                yaw = compute_gaze_np(outputs[1][0])

                # 🔥 修正：垂直偏置，讓你眼睛往上看時能準確反應
                pitch = pitch + 8.0 

                # 將視線角度轉換為螢幕上的座標 (這需要一些經驗法則的映射)
                # 這裡假設你的頭基本正對螢幕，且螢幕約在前方 60 公分處
                # 這個轉換需要根據你的實際使用情境微調！
                # 簡化映射：pitch 和 yaw 在 -180 到 180 度之間
                # 我們假設 -45 ~ +45 度是螢幕範圍
                
                # 將 pitch/yaw 映射到 0~1 之間
                # 假設螢幕水平視角約 60 度，垂直約 40 度
                norm_x = (yaw + 30) / 60 # 將 yaw 映射到 0~1
                norm_y = (pitch + 20) / 40 # 將 pitch 映射到 0~1

                gaze_x_screen = int(np.clip(norm_x * SCREEN_WIDTH, 0, SCREEN_WIDTH - 1))
                gaze_y_screen = int(np.clip(norm_y * SCREEN_HEIGHT, 0, SCREEN_HEIGHT - 1))

                # 更新 Webcam 影像上的箭頭 (可選，用於比對)
                cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
                dx = -100 * np.sin(yaw * np.pi / 180)
                dy = -100 * np.sin(pitch * np.pi / 180)
                cv2.arrowedLine(frame, (cx, cy), (int(cx + dx), int(cy + dy)), (0, 255, 0), 3)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"P:{pitch:.0f} Y:{yaw:.0f}", 
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 熱圖處理
        global heatmap_data
        heatmap_data *= HEATMAP_DECAY_RATE # 熱圖衰減
        
        if gaze_x_screen is not None and gaze_y_screen is not None:
            # 增加新的熱點
            cv2.circle(heatmap_data, (gaze_x_screen, gaze_y_screen), HEATMAP_RADIUS, HEATMAP_BRIGHTNESS, -1)
        
        # 將熱圖數據轉換為彩色圖像
        heatmap_display = np.uint8(np.clip(heatmap_data * (255 / HEATMAP_BRIGHTNESS), 0, 255))
        heatmap_colored = cv2.applyColorMap(heatmap_display, cv2.COLORMAP_JET)

        # 顯示 FPS
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 顯示 Webcam 畫面
        cv2.imshow("Webcam Gaze Demo", frame)
        # 顯示模擬螢幕熱圖
        cv2.imshow("Simulated Screen Heatmap", heatmap_colored)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
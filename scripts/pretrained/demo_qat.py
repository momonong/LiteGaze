import onnxruntime as ort
import cv2
import numpy as np
import time
from l2cs import Pipeline
import torch

# ================= ⚙️ 設定 =================
ONNX_MODEL_PATH = 'models/litegaze_student_fp32.onnx'
TEACHER_PATH = 'models/L2CSNet_gaze360.pkl' # 僅用於偵測臉部
# ==========================================

def compute_gaze_np(logits):
    # 在 NumPy 中實作 Softmax Expectation
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    idx = np.arange(90)
    gaze = np.sum(probs * idx) * 4 - 180
    return gaze

def draw_gaze(image, pitch, yaw, bbox, color=(0, 255, 0)):
    x_min, y_min, x_max, y_max = bbox
    cx, cy = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
    face_w = x_max - x_min
    length = face_w / 2.0
    dx = -length * np.sin(yaw) * np.cos(pitch)
    dy = -length * np.sin(pitch)
    cv2.arrowedLine(image, (cx, cy), (int(cx + dx), int(cy + dy)), color, 4, cv2.LINE_AA, tipLength=0.2)

def main():
    print(f"🚀 啟動 ONNX 推理引擎: {ONNX_MODEL_PATH}")
    
    # 1. 初始化 ONNX Session
    # 我們強制使用 CPUExecutionProvider 來測試 CPU 極限速度
    sess = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    
    # 2. 初始化人臉偵測器 (放在 GPU 以節省 CPU 資源給推理)
    print("👀 啟動 GPU 人臉偵測器...")
    detector = Pipeline(weights=TEACHER_PATH, arch='ResNet50', device=torch.device('cuda'))

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n✅ 極速版 Demo 已準備就緒！按下 'q' 退出。")
    
    fps_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        if time.time() - fps_time > 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = time.time()

        # 1. 偵測人臉 (GPU)
        try:
            results = detector.step(frame)
        except: continue

        if results.bboxes is not None and len(results.bboxes) > 0:
            bbox = results.bboxes[0]
            x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            h, w, _ = frame.shape
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)
            face_img = frame[y_min:y_max, x_min:x_max]
            
            if face_img.size > 0:
                # 2. 預處理 (使用 NumPy 替代 Torch 以求最快速度)
                img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img = img.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img = (img - mean) / std
                img = img.transpose(2, 0, 1)
                img = np.expand_dims(img, axis=0)

                # 3. ONNX 推理 (CPU)
                outputs = sess.run(None, {input_name: img})
                s_pitch = compute_gaze_np(outputs[0][0])
                s_yaw = compute_gaze_np(outputs[1][0])

                # 4. 畫圖
                pitch_rad = s_pitch * np.pi / 180
                yaw_rad = s_yaw * np.pi / 180
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                draw_gaze(frame, pitch_rad, yaw_rad, [x_min, y_min, x_max, y_max], color=(0, 255, 0))
                
                cv2.putText(frame, f"ONNX FP32 | P:{s_pitch:.0f} Y:{s_yaw:.0f}", 
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"FPS: {fps}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("LiteGaze High-Speed Demo", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
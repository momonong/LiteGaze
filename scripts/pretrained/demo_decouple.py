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

def compute_gaze_np(logits):
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    idx = np.arange(90)
    gaze = np.sum(probs * idx) * 4 - 180
    return gaze

def main():
    # 1. 優化 ONNX Session 設定
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4  # 限制執行緒避免爭搶
    sess = ort.InferenceSession(ONNX_MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    
    # 2. 初始化偵測器
    print("👀 啟動 GPU 人臉偵測器...")
    detector = Pipeline(weights=TEACHER_PATH, arch='ResNet50', device=torch.device('cuda'))

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 降低解析度以加速偵測
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n🚀 極速優化啟動！")
    fps_time = time.time()
    frame_count = 0
    fps = 0
    
    bbox = None # 緩存臉部位置

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        if time.time() - fps_time > 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = time.time()

        # 每 2 幀才偵測一次臉部，大幅減輕 CPU/GPU 負擔
        if frame_count % 2 == 0 or bbox is None:
            try:
                results = detector.step(frame)
                if results.bboxes is not None and len(results.bboxes) > 0:
                    bbox = results.bboxes[0]
            except: pass

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

                # 畫圖
                cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
                dx = -100 * np.sin(yaw * np.pi / 180)
                dy = -100 * np.sin(pitch * np.pi / 180)
                cv2.arrowedLine(frame, (cx, cy), (int(cx + dx), int(cy + dy)), (0, 255, 0), 3)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Optimized Gaze Demo", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
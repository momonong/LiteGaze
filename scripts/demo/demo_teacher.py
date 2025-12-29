import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from screeninfo import get_monitors
import os
import sys
import time

# ================= 🔍 路徑修正 =================
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

# ================= ⚙️ 設定區 =================
TEACHER_MODEL_PATH = "models/teacher_v3_best_batch64_H100.h5" 
INPUT_SIZE = 224        
GAZE_SENSITIVITY = 1500 # 調高靈敏度，讓它更容易動
SMOOTHING_RATIO = 0.7   # 稍微降低平滑，讓反應快一點
# ============================================

class TeacherDemoGPU:
    def __init__(self, model_path):
        # 1. 🔥 GPU 檢查與設定
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU Detected: {gpus[0].name}")
                print("🚀 Accelerating with RTX 5090...")
            except RuntimeError as e:
                print(e)
        else:
            print("⚠️ Warning: No GPU detected. Running on CPU (will be slow).")

        # 2. 螢幕設定
        try:
            monitor = get_monitors()[0]
            self.screen_w = monitor.width
            self.screen_h = monitor.height
        except:
            self.screen_w = 1920
            self.screen_h = 1080
        self.screen_cx = self.screen_w // 2
        self.screen_cy = self.screen_h // 2

        # 3. 建構並載入模型
        print(f"👨‍🏫 Building Teacher Model...")
        self.model = build_teacher_v3()
        
        full_model_path = os.path.join(project_root, model_path)
        print(f"⚖️ Loading weights from: {full_model_path}")
        
        if not os.path.exists(full_model_path):
            print(f"❌ Error: Model file not found at {full_model_path}")
            exit()
            
        self.model.load_weights(full_model_path)
        print("✅ Teacher Model Loaded!")

        # 4. Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.smooth_pitch = 0
        self.smooth_yaw = 0
        self.calib_pitch = 0
        self.calib_yaw = 0

    def preprocess(self, face_img):
        try:
            img = cv2.resize(face_img, (INPUT_SIZE, INPUT_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            return img
        except:
            return None

    def map_gaze_to_screen(self, pitch, yaw):
        corrected_pitch = pitch - self.calib_pitch
        corrected_yaw = yaw - self.calib_yaw
        dx = -corrected_yaw * GAZE_SENSITIVITY
        dy = -corrected_pitch * GAZE_SENSITIVITY
        return int(np.clip(self.screen_cx + dx, 0, self.screen_w)), int(np.clip(self.screen_cy + dy, 0, self.screen_h))

    def run(self):
        cap = cv2.VideoCapture(0)
        # 設定 WebCam 解析度 (降低一點可以跑更快)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        cv2.namedWindow('Teacher GPU Demo', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Teacher GPU Demo', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        print("📷 Demo Started! Look at center and press 'c' to calibrate.")
        
        fps_time = time.time()
        frame_count = 0
        fps = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 計算 FPS
            frame_count += 1
            if time.time() - fps_time > 1.0:
                fps = frame_count
                frame_count = 0
                fps_time = time.time()

            canvas = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
            cv2.line(canvas, (self.screen_cx, 0), (self.screen_cx, self.screen_h), (0, 255, 0), 2)
            cv2.line(canvas, (0, self.screen_cy), (self.screen_w, self.screen_cy), (0, 255, 0), 2)

            frame_flipped = cv2.flip(frame, 1)
            h, w, _ = frame_flipped.shape
            rgb_frame = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    x_coords = [lm.x for lm in face_landmarks.landmark]
                    y_coords = [lm.y for lm in face_landmarks.landmark]
                    x_min, x_max = int(min(x_coords)*w), int(max(x_coords)*w)
                    y_min, y_max = int(min(y_coords)*h), int(max(y_coords)*h)
                    
                    # 擴大一點
                    margin_x, margin_y = int((x_max-x_min)*0.3), int((y_max-y_min)*0.4)
                    face_roi = frame_flipped[
                        max(0, y_min-margin_y):min(h, y_max+margin_y),
                        max(0, x_min-margin_x):min(w, x_max+margin_x)
                    ]
                    
                    if face_roi.size > 0:
                        input_tensor = self.preprocess(face_roi)
                        if input_tensor is not None:
                            # 🔥🔥🔥 加速關鍵：直接呼叫模型，不走 .predict() 🔥🔥🔥
                            preds = self.model(input_tensor, training=False)
                            
                            # 轉回 numpy
                            gaze = preds[0][0].numpy()
                            pitch, yaw = gaze[0], gaze[1]
                            
                            self.smooth_pitch = SMOOTHING_RATIO * self.smooth_pitch + (1 - SMOOTHING_RATIO) * pitch
                            self.smooth_yaw = SMOOTHING_RATIO * self.smooth_yaw + (1 - SMOOTHING_RATIO) * yaw
                            
                            gx, gy = self.map_gaze_to_screen(self.smooth_pitch, self.smooth_yaw)
                            
                            # 畫紅點
                            cv2.circle(canvas, (gx, gy), 30, (0, 0, 255), -1)
                            
                            status = f"P: {self.smooth_pitch:.2f} Y: {self.smooth_yaw:.2f}"
                            cv2.putText(canvas, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # 顯示 FPS
            cv2.putText(canvas, f"FPS: {fps}", (self.screen_w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cam_small = cv2.resize(frame_flipped, (320, 240))
            canvas[self.screen_h-240:self.screen_h, 0:320] = cam_small
            cv2.putText(canvas, "Press 'c' to Calibrate", (self.screen_cx - 150, self.screen_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Teacher GPU Demo', canvas)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('c'):
                self.calib_pitch = self.smooth_pitch
                self.calib_yaw = self.smooth_yaw
                print("🎯 Calibrated!")
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    TeacherDemoGPU(TEACHER_MODEL_PATH).run()
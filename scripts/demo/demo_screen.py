import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from screeninfo import get_monitors

# ================= ⚙️ 設定區 =================
MODEL_PATH = "models/litegaze_v2_distilled.tflite"
INPUT_SIZE = 60
SMOOTHING_RATIO = 0.85   # 稍微調高一點點，讓紅點更穩
GAZE_SENSITIVITY = 1200  # 視線靈敏度
# ============================================

class LiteGazeScreenDemo:
    def __init__(self, model_path):
        # 1. 螢幕設定
        try:
            monitor = get_monitors()[0]
            self.screen_w = monitor.width
            self.screen_h = monitor.height
        except:
            self.screen_w = 1920
            self.screen_h = 1080
        self.screen_cx = self.screen_w // 2
        self.screen_cy = self.screen_h // 2

        # 2. 模型載入
        print(f"🚀 Loading model from: {model_path}")
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        except Exception as e:
            print(f"❌ Error: {e}")
            exit()

        # 3. Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, # 🔥 開啟 Refine landmarks 以獲得更準的虹膜點
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.smooth_pitch = 0
        self.smooth_yaw = 0

        # 🔥 定義眼睛的特徵點索引 (MediaPipe 標準)
        self.LEFT_EYE_IDX = [33, 133, 160, 159, 158, 144, 145, 153]
        self.RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 373, 374, 380]

    def preprocess(self, face_img):
        try:
            img = cv2.resize(face_img, (INPUT_SIZE, INPUT_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 轉黑白
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=-1)
            return img
        except:
            return None

    def predict(self, input_tensor):
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])[0]

    def map_gaze_to_screen(self, pitch, yaw):
        dx = -yaw * GAZE_SENSITIVITY
        dy = -pitch * GAZE_SENSITIVITY
        return int(np.clip(self.screen_cx + dx, 0, self.screen_w)), int(np.clip(self.screen_cy + dy, 0, self.screen_h))

    def draw_eye_boxes(self, frame, landmarks, w, h):
        """ 🔥 新增功能：畫出眼睛的框框 """
        for eye_idx, color in [(self.LEFT_EYE_IDX, (0, 255, 255)), (self.RIGHT_EYE_IDX, (0, 255, 255))]:
            # 取得該眼睛所有點的座標
            eye_points = []
            for idx in eye_idx:
                lm = landmarks[idx]
                eye_points.append([int(lm.x * w), int(lm.y * h)])
            
            eye_points = np.array(eye_points)
            
            # 計算外接矩形 (Bounding Box)
            x, y, ew, eh = cv2.boundingRect(eye_points)
            
            # 稍微外擴一點點，比較好看
            margin = 5
            x = max(0, x - margin)
            y = max(0, y - margin)
            ew += margin * 2
            eh += margin * 2
            
            # 畫框 (黃色)
            cv2.rectangle(frame, (x, y), (x + ew, y + eh), color, 2)

    def run(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('LiteGaze', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('LiteGaze', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 準備畫布
            canvas = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
            cv2.line(canvas, (self.screen_cx, 0), (self.screen_cx, self.screen_h), (0, 100, 0), 1)
            cv2.line(canvas, (0, self.screen_cy), (self.screen_w, self.screen_cy), (0, 100, 0), 1)

            # 處理影像
            frame_flipped = cv2.flip(frame, 1)
            h, w, _ = frame_flipped.shape
            rgb_frame = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            detected = False
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 1. 取得人臉 ROI (給模型用)
                    x_coords = [lm.x for lm in face_landmarks.landmark]
                    y_coords = [lm.y for lm in face_landmarks.landmark]
                    x_min, x_max = int(min(x_coords)*w), int(max(x_coords)*w)
                    y_min, y_max = int(min(y_coords)*h), int(max(y_coords)*h)
                    
                    # 擴大人臉框
                    margin_x, margin_y = int((x_max-x_min)*0.25), int((y_max-y_min)*0.35)
                    face_roi = frame_flipped[
                        max(0, y_min-margin_y):min(h, y_max+margin_y),
                        max(0, x_min-margin_x):min(w, x_max+margin_x)
                    ]
                    
                    if face_roi.size > 0:
                        input_tensor = self.preprocess(face_roi)
                        if input_tensor is not None:
                            # 2. 推論與平滑
                            prediction = self.predict(input_tensor)
                            if len(prediction) >= 2:
                                pitch, yaw = prediction[0], prediction[1]
                            else:
                                continue
                            self.smooth_pitch = SMOOTHING_RATIO * self.smooth_pitch + (1 - SMOOTHING_RATIO) * pitch
                            self.smooth_yaw = SMOOTHING_RATIO * self.smooth_yaw + (1 - SMOOTHING_RATIO) * yaw
                            
                            # 3. 視覺化
                            gx, gy = self.map_gaze_to_screen(self.smooth_pitch, self.smooth_yaw)
                            
                            # 畫紅點 (視線)
                            cv2.circle(canvas, (gx, gy), 25, (0, 0, 255), -1)
                            # 畫光暈 (讓紅點看起來像雷射)
                            cv2.circle(canvas, (gx, gy), 40, (0, 0, 255), 2)
                            
                            # 🔥 4. 畫框框：人臉 (綠色) + 眼睛 (黃色)
                            # 畫人臉框
                            cv2.rectangle(frame_flipped, 
                                        (max(0, x_min-margin_x), max(0, y_min-margin_y)), 
                                        (min(w, x_max+margin_x), min(h, y_max+margin_y)), 
                                        (0, 255, 0), 2)
                            
                            # 畫眼睛框 (新增的函式)
                            self.draw_eye_boxes(frame_flipped, face_landmarks.landmark, w, h)
                            
                            detected = True

            # 顯示左下角小畫面
            cam_small = cv2.resize(frame_flipped, (320, 240))
            canvas[self.screen_h-240:self.screen_h, 0:320] = cam_small
            
            # 加入文字資訊
            status = f"Pitch: {self.smooth_pitch:.2f} Yaw: {self.smooth_yaw:.2f}"
            cv2.putText(canvas, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

            cv2.imshow('LiteGaze', canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    demo = LiteGazeScreenDemo(MODEL_PATH)
    demo.run()
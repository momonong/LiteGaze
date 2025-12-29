import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

# ================= ⚙️ 設定區 =================
MODEL_PATH = "models/litegaze_v2_distilled.tflite"  # TFLite 模型路徑
INPUT_SIZE = 60                                     # 模型輸入大小 (60x60)
SMOOTHING_RATIO = 0.7                               # 平滑係數 (0~1)，越高越靈敏，越低越穩
SENSITIVITY = 100                                   # 視線箭頭長度
# =============================================

class LiteGazeDemo:
    def __init__(self, model_path):
        # 1. 初始化 TFLite 解譯器
        print(f"🚀 Loading model from: {model_path}")
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # 檢查輸入形狀 (預期: [1, 60, 60, 1])
            input_shape = self.input_details[0]['shape']
            print(f"✅ Model Input Shape: {input_shape}")
            
            # 簡單檢查是否為單通道 (黑白)
            if input_shape[-1] != 1:
                print("⚠️ Warning: Model expects RGB input? Check your training script.")
            else:
                print("✅ Model expects Grayscale input (Correct!)")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            exit()

        # 2. 初始化 MediaPipe Face Mesh (用來抓臉)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 平滑用的變數
        self.prev_pitch = 0
        self.prev_yaw = 0

    def preprocess(self, face_img):
        """
        關鍵步驟：將圖片轉為模型看得懂的格式
        1. Resize -> 60x60
        2. BGR -> Grayscale (重要!)
        3. Normalize -> 0~1
        4. Expand Dims -> (1, 60, 60, 1)
        """
        try:
            # Resize
            img = cv2.resize(face_img, (INPUT_SIZE, INPUT_SIZE))
            
            # 🔥 轉為灰階 (配合學生模型)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Normalize (0~1)
            img = img.astype(np.float32) / 255.0
            
            # 增加維度: (60, 60) -> (1, 60, 60, 1)
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=-1)
            
            return img
        except Exception as e:
            return None

    def predict(self, input_tensor):
        """ 執行推論 """
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        
        # 取得輸出 (假設 index 0 是 gaze vector，如果不是要檢查 output_details)
        # 我們的模型輸出順序通常是: [gaze_xy, pitch_logits, yaw_logits] 或類似
        # 但通常主要輸出 (Gaze) 會在第一個
        gaze_vector = self.interpreter.get_tensor(self.output_details[0]['index'])
        return gaze_vector[0] # [pitch, yaw]

    def draw_gaze(self, frame, landmarks, pitch, yaw):
        """ 畫出視線箭頭 """
        h, w, c = frame.shape
        
        # 找鼻頭位置 (Index 1 or 4) 作為起點
        nose_idx = 4
        nose_x = int(landmarks[nose_idx].x * w)
        nose_y = int(landmarks[nose_idx].y * h)
        
        # 計算終點 (將 Pitch/Yaw 轉換為 2D 向量)
        # Pitch (上下): 負值往上
        # Yaw (左右): 負值往右 (視角不同可能要反轉)
        
        dx = -np.sin(yaw) * SENSITIVITY
        dy = -np.sin(pitch) * SENSITIVITY
        
        end_x = int(nose_x + dx)
        end_y = int(nose_y + dy)
        
        # 畫線
        cv2.arrowedLine(frame, (nose_x, nose_y), (end_x, end_y), (0, 0, 255), 4)
        
        # 顯示數值
        cv2.putText(frame, f"Pitch: {pitch:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Yaw:   {yaw:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def run(self):
        cap = cv2.VideoCapture(0) # 開啟 Webcam
        
        print("📷 Starting Webcam... Press 'q' to exit.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 翻轉鏡頭 (像照鏡子一樣)
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 1. 偵測人臉
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 2. 裁切人臉區域 (簡單版：取 bounding box)
                    x_min, y_min = w, h
                    x_max, y_max = 0, 0
                    
                    for lm in face_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        if x < x_min: x_min = x
                        if x > x_max: x_max = x
                        if y < y_min: y_min = y
                        if y > y_max: y_max = y
                    
                    # 稍微擴大一點範圍，包含整個頭
                    margin_x = int((x_max - x_min) * 0.2)
                    margin_y = int((y_max - y_min) * 0.2)
                    x_min = max(0, x_min - margin_x)
                    x_max = min(w, x_max + margin_x)
                    y_min = max(0, y_min - margin_y)
                    y_max = min(h, y_max + margin_y)
                    
                    face_img = frame[y_min:y_max, x_min:x_max]
                    
                    if face_img.size == 0: continue

                    # 畫出人臉框
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
                    
                    # 3. 預處理
                    input_tensor = self.preprocess(face_img)
                    
                    if input_tensor is not None:
                        # 4. 推論
                        pred = self.predict(input_tensor)
                        pitch, yaw = pred[0], pred[1]
                        
                        # 5. 平滑處理 (Exponential Moving Average)
                        pitch = SMOOTHING_RATIO * pitch + (1 - SMOOTHING_RATIO) * self.prev_pitch
                        yaw = SMOOTHING_RATIO * yaw + (1 - SMOOTHING_RATIO) * self.prev_yaw
                        
                        self.prev_pitch = pitch
                        self.prev_yaw = yaw
                        
                        # 6. 畫出視線
                        self.draw_gaze(frame, face_landmarks.landmark, pitch, yaw)

            cv2.imshow('LiteGaze Final Demo', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 請確保檔名正確
    demo = LiteGazeDemo(MODEL_PATH)
    demo.run()
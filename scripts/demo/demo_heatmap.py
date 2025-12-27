import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

# === ⚙️ 設定區 ===
MODEL_PATH = 'models/litegaze_v2_win.tflite'
INPUT_SIZE = (60, 60)

# 熱圖設定
HEATMAP_ALPHA = 0.6    # 透明度 (0.0~1.0)
SMOOTHING_FRAME = 5    # 平滑化幀數，數值越大越穩定但延遲越高
GAZE_SENSITIVITY = 800 # 靈敏度，數值越大熱點跑得越遠

class GazeHeatmapDemo:
    def __init__(self):
        # 載入模型
        print("⏳ Loading TFLite model...")
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.output_index = self.output_details[0]['index'] 

        # 初始化 MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 用於平滑化注視點的佇列
        self.gaze_history = deque(maxlen=SMOOTHING_FRAME)
        self.heatmap_canvas = None

    def get_gaze_point(self, pitch, yaw, img_h, img_w):
        """將 Pitch/Yaw 角度轉換為螢幕上的粗略 (x, y) 坐標"""
        # 假設相機在螢幕中心上方，簡單線性映射
        # Yaw (左右) -> X 軸, Pitch (上下) -> Y 軸
        # 負號可能需要根據實際體驗微調
        dx = -np.sin(yaw) * GAZE_SENSITIVITY
        dy = np.sin(pitch) * GAZE_SENSITIVITY
        
        center_x, center_y = img_w // 2, img_h // 2
        gaze_x = int(center_x + dx)
        gaze_y = int(center_y + dy)
        
        return gaze_x, gaze_y

    def draw_heatmap(self, image, gaze_point):
        h, w, _ = image.shape
        if self.heatmap_canvas is None:
            self.heatmap_canvas = np.zeros((h, w), dtype=np.float32)

        # 1. 在畫布上繪製一個新的熱點 (高斯分佈)
        gx, gy = gaze_point
        # 確保坐標在圖像範圍內，並留邊界給高斯模糊
        gx = np.clip(gx, 50, w - 50)
        gy = np.clip(gy, 50, h - 50)
        
        # 創建一個局部的高斯遮罩
        kernel_size = 201 # 熱點大小
        sigma = 50        # 熱點擴散程度
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = kernel * kernel.T
        # Normalize 到 0~1 並增強強度
        kernel = kernel / kernel.max()
        
        # 將熱點疊加到當前位置
        x1, y1 = gx - kernel_size // 2, gy - kernel_size // 2
        x2, y2 = x1 + kernel_size, y1 + kernel_size
        self.heatmap_canvas[y1:y2, x1:x2] = np.maximum(self.heatmap_canvas[y1:y2, x1:x2], kernel)

        # 2. 讓舊的熱度慢慢消退 (Decay)
        self.heatmap_canvas *= 0.92

        # 3. 產生彩色熱圖
        heatmap_img = (self.heatmap_canvas * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
        
        # 將黑色背景變透明
        mask = heatmap_img > 10
        overlay = image.copy()
        overlay[mask] = cv2.addWeighted(image[mask], 1 - HEATMAP_ALPHA, heatmap_color[mask], HEATMAP_ALPHA, 0)
        
        return overlay

    def run(self):
        cap = cv2.VideoCapture(0)
        print("🚀 Starting Heatmap Demo... Look around!")
        
        while cap.isOpened():
            success, image = cap.read()
            if not success: break
            
            image = cv2.flip(image, 1) # 鏡像
            h, w, _ = image.shape
            
            # 處理影像
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)

            current_gaze = None

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 抓臉與裁切 (與之前相同)
                    x_coords = [lm.x for lm in face_landmarks.landmark]
                    y_coords = [lm.y for lm in face_landmarks.landmark]
                    x_min, x_max = int(min(x_coords)*w), int(max(x_coords)*w)
                    y_min, y_max = int(min(y_coords)*h), int(max(y_coords)*h)
                    pad = 30
                    x_min, y_min = max(0, x_min-pad), max(0, y_min-pad)
                    x_max, y_max = min(w, x_max+pad), min(h, y_max+pad)
                    
                    face_img = image[y_min:y_max, x_min:x_max]
                    if face_img.size == 0: continue

                    # 推論
                    input_img = cv2.resize(face_img, INPUT_SIZE)
                    input_data = input_img.astype(np.float32) / 255.0
                    input_data = np.expand_dims(input_data, axis=0)
                    
                    self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                    self.interpreter.invoke()
                    pred = self.interpreter.get_tensor(self.output_index)[0]
                    
                    # 計算注視點
                    gaze_point = self.get_gaze_point(pred[0], pred[1], h, w)
                    self.gaze_history.append(gaze_point)
                    
                    # 平滑化
                    avg_x = int(np.mean([p[0] for p in self.gaze_history]))
                    avg_y = int(np.mean([p[1] for p in self.gaze_history]))
                    current_gaze = (avg_x, avg_y)

            # 繪製熱圖 (如果沒有偵測到臉，熱圖會慢慢消退)
            if current_gaze:
                output_image = self.draw_heatmap(image, current_gaze)
            elif self.heatmap_canvas is not None:
                 # 沒有人時，讓熱圖持續消退
                 self.heatmap_canvas *= 0.9
                 heatmap_img = (self.heatmap_canvas * 255).astype(np.uint8)
                 heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
                 mask = heatmap_img > 10
                 output_image = image.copy()
                 output_image[mask] = cv2.addWeighted(image[mask], 1 - HEATMAP_ALPHA, heatmap_color[mask], HEATMAP_ALPHA, 0)
            else:
                output_image = image

            cv2.imshow('LiteGaze V2 - Heatmap Visualization', output_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    demo = GazeHeatmapDemo()
    demo.run()
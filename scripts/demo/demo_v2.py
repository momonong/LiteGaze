import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

# === ⚙️ 設定區 ===
MODEL_PATH = 'models/litegaze_v2_win.tflite' 
INPUT_SIZE = (60, 60)

# 視線繪製長度
AXIS_LENGTH = 200 

def draw_gaze(image_in, pitchyaw, thickness=2, color=(0, 0, 255)):
    """畫出視線向量"""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = AXIS_LENGTH
    
    # 模型的輸出是 Radians，我們轉成向量
    pitch, yaw = pitchyaw[0], pitchyaw[1]
    
    # 數學轉換 (Spherical to Cartesian)
    # 注意：這裡的坐標系可能需要根據模型訓練時的定義微調
    # 假設：X向右, Y向下, Z向後 (OpenCV Standard)
    dx = -length * np.sin(yaw) * np.cos(pitch)
    dy = -length * np.sin(pitch)
    
    center = (w // 2, h // 2)
    end_point = (int(center[0] + dx), int(center[1] + dy))
    
    cv2.arrowedLine(image_out, center, end_point, color, thickness, tipLength=0.2)
    return image_out

def main():
    # 1. 載入 TFLite 模型
    print("⏳ Loading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 找到視線輸出的 Index (通常是第一個，但也可能是分類的 Logits)
    # 我們在 export 時 output 順序是 [gaze, pitch_logits, yaw_logits]
    output_index = output_details[0]['index'] 

    # 2. 初始化 Face Mesh (用來抓臉)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    
    print("🚀 Starting Demo... Press 'q' to quit.")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # 翻轉圖片 (像照鏡子一樣)
        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 偵測人臉
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 簡單取臉部邊界 (Bounding Box)
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x < x_min: x_min = x
                    if y < y_min: y_min = y
                    if x > x_max: x_max = x
                    if y > y_max: y_max = y
                
                # 稍微擴大框框 (Padding) 以包含整張臉
                pad_x = int((x_max - x_min) * 0.1)
                pad_y = int((y_max - y_min) * 0.1)
                x_min = max(0, x_min - pad_x)
                y_min = max(0, y_min - pad_y)
                x_max = min(w, x_max + pad_x)
                y_max = min(h, y_max + pad_y)

                # 裁切臉部
                face_img = image[y_min:y_max, x_min:x_max]
                
                if face_img.size == 0: continue

                # === 核心：前處理 & 推論 ===
                try:
                    # Resize to 60x60
                    input_img = cv2.resize(face_img, INPUT_SIZE)
                    
                    # Normalize (0~1 或 0~255 取決於訓練數據)
                    # 假設訓練時是 float 0-1 (因為用了 tf.image.convert_image_dtype 或 clip 0-1)
                    input_data = input_img.astype(np.float32) / 255.0
                    input_data = np.expand_dims(input_data, axis=0) # Add Batch dim

                    # 推論
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    
                    # 取得結果 (Pitch, Yaw)
                    pred_gaze = interpreter.get_tensor(output_index)[0]
                    
                    # 繪製結果
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # 在臉部中心畫箭頭
                    # 我們把箭頭畫在臉的框框上，比較清楚
                    center_x = (x_min + x_max) // 2
                    center_y = (y_min + y_max) // 2
                    
                    # 顯示數值
                    text = f"P: {pred_gaze[0]:.2f}, Y: {pred_gaze[1]:.2f}"
                    cv2.putText(image, text, (x_min, y_min - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # 為了視覺化，我們創一個小視窗專門畫箭頭，或者直接畫在臉上
                    # 這裡簡單畫在臉上
                    length = 100
                    pitch, yaw = pred_gaze[0], pred_gaze[1]
                    dx = -length * np.sin(yaw)
                    dy = -length * np.sin(pitch)
                    cv2.arrowedLine(image, (center_x, center_y), 
                                  (int(center_x + dx), int(center_y + dy)), 
                                  (0, 0, 255), 4)
                                  
                except Exception as e:
                    print(f"Inference Error: {e}")

        cv2.imshow('LiteGaze V2 Demo', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
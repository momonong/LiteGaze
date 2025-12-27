import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# === 設定區 ===
TFLITE_MODEL_PATH = 'models/litegaze_student.tflite'
INPUT_SIZE = (60, 60)

# 靈敏度參數 (等確認數值會動了再來調這個)
GAIN_X = 1500
GAIN_Y = 1500
OFFSET_X = 0
OFFSET_Y = 0

# 平滑參數
history_pitch = []
history_yaw = []
SMOOTH_WINDOW = 4

# 初始化
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True, max_num_faces=1,
    min_detection_confidence=0.6, min_tracking_confidence=0.6
)

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

def moving_average(new_val, history):
    history.append(new_val)
    if len(history) > SMOOTH_WINDOW: history.pop(0)
    return sum(history) / len(history)

# 🔥 關鍵函式：模擬訓練時的圖像處理
def preprocess_for_model(eye_crop_bgr):
    # 1. 轉灰階
    gray = cv2.cvtColor(eye_crop_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. 直方圖均衡化 (Histogram Equalization) - 增強對比
    gray_eq = cv2.equalizeHist(gray)
    
    # 3. 轉回 RGB (因為模型輸入層是 3 Channel)
    img_rgb = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2RGB)
    
    # 4. Resize 到 60x60
    img_resized = cv2.resize(img_rgb, INPUT_SIZE)
    
    # 5. 歸一化
    input_data = img_resized.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    
    return input_data, img_resized # 回傳處理好的小圖以便顯示

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened(): cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("🚀 LiteGaze V3 (Debug 版) 啟動！")
print("👀 請觀察左上角的小圖，那是模型真正看到的樣子。")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    debug_eye_view = None # 用來存要顯示的小圖

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            pts = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) for p in face_landmarks.landmark])
            
            gaze_preds = []
            
            for eye_idxs in [LEFT_EYE, RIGHT_EYE]:
                eye_pts = pts[eye_idxs]
                x_min, y_min = np.min(eye_pts, axis=0)
                x_max, y_max = np.max(eye_pts, axis=0)
                
                # 擴大範圍：如果不夠大，眼睛會太小
                margin_x = int((x_max - x_min) * 0.5) 
                margin_y = int((y_max - y_min) * 0.8)
                
                eye_img = frame[max(0, y_min-margin_y):min(h, y_max+margin_y), 
                                max(0, x_min-margin_x):min(w, x_max+margin_x)]
                
                if eye_img.size > 0:
                    # 🔥 使用新的前處理
                    input_tensor, debug_img = preprocess_for_model(eye_img)
                    debug_eye_view = debug_img # 存起來等等畫
                    
                    interpreter.set_tensor(input_details[0]['index'], input_tensor)
                    interpreter.invoke()
                    pred = interpreter.get_tensor(output_details[0]['index'])[0]
                    gaze_preds.append(pred)

            if gaze_preds:
                avg_pitch = np.mean([g[0] for g in gaze_preds])
                avg_yaw = np.mean([g[1] for g in gaze_preds])
                
                smooth_p = moving_average(avg_pitch, history_pitch)
                smooth_y = moving_average(avg_yaw, history_yaw)

                # 畫圖邏輯 (跟之前一樣)
                screen_cx, screen_cy = w // 2, h // 2
                dx = -smooth_y * GAIN_X # 負號可能需要根據您的測試調整
                dy = -smooth_p * GAIN_Y 
                gaze_x = int(screen_cx + dx + OFFSET_X)
                gaze_y = int(screen_cy + dy + OFFSET_Y)
                gaze_x = max(0, min(w, gaze_x))
                gaze_y = max(0, min(h, gaze_y))

                cv2.circle(frame, (gaze_x, gaze_y), 15, (0, 0, 255), -1)
                nose = pts[1]
                cv2.line(frame, tuple(nose), (gaze_x, gaze_y), (0, 255, 255), 2)
                
                cv2.putText(frame, f"P: {smooth_p:.2f}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Y: {smooth_y:.2f}", (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # === 畫出「模型看到的眼睛」 (Debug View) ===
    if debug_eye_view is not None:
        # 把 60x60 放大到 120x120 比較好人眼觀察
        disp_eye = cv2.resize(debug_eye_view, (120, 120))
        frame[0:120, 0:120] = disp_eye
        cv2.rectangle(frame, (0,0), (120,120), (0,255,0), 2)
        cv2.putText(frame, "Model Input", (5, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

    cv2.imshow('LiteGaze V3 - Histogram Eq', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
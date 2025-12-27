import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import traceback

# === 設定區 ===
TFLITE_MODEL_PATH = 'models/litegaze_student.tflite'
INPUT_SIZE = (60, 60)
SMOOTH_WINDOW = 5

# 🔥 視線映射參數 (這是控制紅球怎麼跑的關鍵)
X_SENSITIVITY = 1000   # 水平靈敏度 (越大跑越快)
Y_SENSITIVITY = 1200   # 垂直靈敏度

# 🔥 校正偏移 (如果你看正中間時紅球不在中間，改這裡)
# 負值代表紅球會往上/左修，正值往下/右修
OFFSET_PITCH = -0.15 
OFFSET_YAW = 0.0

# 穩定化歷史紀錄
history_pitch = []
history_yaw = []

def moving_average(new_val, history):
    history.append(new_val)
    if len(history) > SMOOTH_WINDOW:
        history.pop(0)
    return sum(history) / len(history)

def draw_debug_text(img, text, line_num, color=(0, 255, 0)):
    cv2.putText(img, text, (10, 30 + line_num * 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def main():
    cap = None
    try:
        # --- Step 1: 模型載入 ---
        print("[Step 1] 正在載入 TFLite 模型...")
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("✅ TFLite 模型載入完成")

        # --- Step 2: MediaPipe 初始化 ---
        print("[Step 2] 正在初始化 MediaPipe...")
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            refine_landmarks=True, max_num_faces=1,
            min_detection_confidence=0.6, min_tracking_confidence=0.6
        )

        LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

        # --- Step 3: 開啟攝影機 ---
        print("[Step 3] 正在開啟攝影機...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        
        # 設定解析度
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 取得實際畫面大小 (用於映射座標)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print("🚀 LiteGaze Gaze Tracking 啟動！(按 'q' 離開)")

        while True:
            # 這裡不需要額外的 try，因為外層已經有了，除非你想捕捉單一幀的錯誤
            success, frame = cap.read()
            if not success:
                print("⚠️ 掉幀中...")
                continue

            # 鏡像翻轉 (讓操作比較直覺)
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe 推論
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    pts = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) for p in face_landmarks.landmark])
                    
                    eye_centers = []
                    gaze_results = []

                    # 裁切眼睛並推論
                    for eye_idxs in [LEFT_EYE, RIGHT_EYE]:
                        eye_pts = pts[eye_idxs]
                        x_min, y_min = np.min(eye_pts, axis=0)
                        x_max, y_max = np.max(eye_pts, axis=0)
                        
                        # 安全邊界
                        x1, y1 = max(0, x_min-5), max(0, y_min-5)
                        x2, y2 = min(w, x_max+5), min(h, y_max+5)

                        eye_img = frame[y1:y2, x1:x2]
                        
                        # 檢查眼睛圖片是否有效
                        if eye_img.size > 0 and eye_img.shape[0] > 5 and eye_img.shape[1] > 5:
                            eye_input = cv2.resize(cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB), INPUT_SIZE)
                            eye_input = (eye_input.astype(np.float32) / 255.0)[np.newaxis, :]
                            
                            interpreter.set_tensor(input_details[0]['index'], eye_input)
                            interpreter.invoke()
                            gaze = interpreter.get_tensor(output_details[0]['index'])[0]
                            
                            eye_centers.append(((x1+x2)//2, (y1+y2)//2))
                            gaze_results.append(gaze)

                    if gaze_results:
                        # 1. 計算平均角度
                        avg_pitch = np.mean([g[0] for g in gaze_results])
                        avg_yaw = np.mean([g[1] for g in gaze_results])
                        
                        # 2. 平滑化
                        smooth_p = moving_average(avg_pitch, history_pitch)
                        smooth_y = moving_average(avg_yaw, history_yaw)

                        # 3. 🔥 視線映射核心邏輯 🔥
                        # 公式: 偏移量 = tan(角度 - 校正值) * 靈敏度
                        delta_x = np.tan(smooth_y - OFFSET_YAW) * X_SENSITIVITY
                        delta_y = np.tan(smooth_p - OFFSET_PITCH) * Y_SENSITIVITY
                        
                        # 算出螢幕座標 (假設畫面中心 = 螢幕中心)
                        gaze_x = int(frame_w / 2 + delta_x)
                        gaze_y = int(frame_h / 2 + delta_y) # Pitch 負值往上，但在影像座標 Y 往上是變小，這裡直接加即可 (視模型定義而定)

                        # 繪製紅球 (代表視線落點)
                        cv2.circle(frame, (gaze_x, gaze_y), 15, (0, 0, 255), -1)
                        # 畫一條線連到眼睛 (視覺輔助)
                        for center in eye_centers:
                             cv2.line(frame, center, (gaze_x, gaze_y), (0, 255, 255), 1)

                        # 顯示數值
                        draw_debug_text(frame, f"Pitch: {smooth_p:.2f}", 0)
                        draw_debug_text(frame, f"Yaw:   {smooth_y:.2f}", 1)
                        draw_debug_text(frame, f"Gaze: ({gaze_x}, {gaze_y})", 2, (0, 255, 255))

            cv2.imshow('LiteGaze - Eye Tracking', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print("❌ 發生錯誤:")
        traceback.print_exc()
    finally:
        print("[Cleanup] 釋放資源...")
        if cap and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
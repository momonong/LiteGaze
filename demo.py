import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import sys
import traceback  # 🔥 新增：用於印出詳細錯誤堆疊

# === 設定區 ===
TFLITE_MODEL_PATH = 'models/litegaze_student.tflite'
INPUT_SIZE = (60, 60)
SMOOTH_WINDOW = 5

# 穩定化參數
history_pitch = []
history_yaw = []

def moving_average(new_val, history):
    history.append(new_val)
    if len(history) > SMOOTH_WINDOW:
        history.pop(0)
    return sum(history) / len(history)

# 用於在畫面上印字的輔助函式
def draw_debug_text(img, text, line_num, color=(0, 255, 0)):
    cv2.putText(img, text, (10, 30 + line_num * 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

try:
    # --- Step 1: 模型載入 ---
    print("\n[Step 1] 正在載入 TFLite 模型...")
    if not tf.io.gfile.exists(TFLITE_MODEL_PATH):
        raise FileNotFoundError(f"❌ 找不到模型檔案: {TFLITE_MODEL_PATH}")
        
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 🔥 印出模型資訊，幫助除錯
    print(f"   👉 模型輸入形狀: {input_details[0]['shape']}")
    print(f"   👉 模型輸入類型: {input_details[0]['dtype']}")
    print("✅ TFLite 模型載入完成")

    # --- Step 2: MediaPipe 初始化 ---
    print("[Step 2] 正在初始化 MediaPipe...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    print("✅ MediaPipe 初始化完成")

    # 眼睛關鍵點索引
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

    # --- Step 3: 開啟攝影機 ---
    print("[Step 3] 正在開啟攝影機...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("⚠️ 無法打開 Camera 0，嘗試 Camera 1...")
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError("❌ 錯誤：找不到任何攝影機！請檢查裝置連線。")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("🚀 LiteGaze 啟動成功！")

    fps_time = time.time()
    frame_count = 0
    fail_count = 0  # 計算連續失敗次數

    while True:
        try:
            success, frame = cap.read()
            if not success:
                fail_count += 1
                print(f"⚠️ 無法讀取影像 ({fail_count}/10)")
                if fail_count > 10:
                    raise RuntimeError("❌ 攝影機訊號中斷，程式強制結束。")
                continue
            
            fail_count = 0 # 重置失敗計數
            
            # FPS 計算
            frame_count += 1
            fps = 0
            if time.time() - fps_time > 1.0:
                fps = frame_count
                frame_count = 0
                fps_time = time.time()

            # 影像前處理
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe 推論
            results = face_mesh.process(rgb_frame)

            # 畫面儀表板 (HUD)
            draw_debug_text(frame, f"FPS: {fps}", 0, (0, 255, 255))

            if results.multi_face_landmarks:
                draw_debug_text(frame, "Face: Detected", 1, (0, 255, 0))
                
                for face_landmarks in results.multi_face_landmarks:
                    pts = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) for p in face_landmarks.landmark])
                    
                    eye_centers = []
                    gaze_results = []

                    for i, eye_idxs in enumerate([LEFT_EYE, RIGHT_EYE]):
                        eye_pts = pts[eye_idxs]
                        
                        # 🔥 安全邊界檢查：防止裁切超出畫面
                        x_min, y_min = np.min(eye_pts, axis=0)
                        x_max, y_max = np.max(eye_pts, axis=0)
                        
                        # 擴大一點範圍，但限制在 0~w, 0~h 之間
                        x1 = max(0, x_min - 5)
                        y1 = max(0, y_min - 5)
                        x2 = min(w, x_max + 5)
                        y2 = min(h, y_max + 5)

                        # 繪製眼睛框框 (除錯用)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

                        eye_img = frame[y1:y2, x1:x2]
                        
                        if eye_img.size > 0 and eye_img.shape[0] > 5 and eye_img.shape[1] > 5:
                            # 模型推論
                            eye_input = cv2.resize(cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB), INPUT_SIZE)
                            eye_input = (eye_input.astype(np.float32) / 255.0)[np.newaxis, :]
                            
                            interpreter.set_tensor(input_details[0]['index'], eye_input)
                            interpreter.invoke()
                            gaze = interpreter.get_tensor(output_details[0]['index'])[0]
                            
                            eye_centers.append(((x1+x2)//2, (y1+y2)//2))
                            gaze_results.append(gaze)
                        else:
                            print(f"⚠️ 跳過過小的眼睛區域: {eye_img.shape}")

                    if gaze_results:
                        avg_pitch = np.mean([g[0] for g in gaze_results])
                        avg_yaw = np.mean([g[1] for g in gaze_results])
                        
                        smooth_p = moving_average(avg_pitch, history_pitch)
                        smooth_y = moving_average(avg_yaw, history_yaw)

                        # 顯示數值
                        draw_debug_text(frame, f"Pitch: {smooth_p:.2f}", 2)
                        draw_debug_text(frame, f"Yaw:   {smooth_y:.2f}", 3)

                        # 繪製視線箭頭
                        for center in eye_centers:
                            dx = -150 * np.sin(smooth_y)
                            dy = -150 * np.sin(smooth_p)
                            end_pt = (int(center[0] + dx), int(center[1] + dy))
                            cv2.arrowedLine(frame, center, end_pt, (0, 0, 255), 2, tipLength=0.3)
            else:
                draw_debug_text(frame, "Face: Searching...", 1, (0, 0, 255))

            cv2.imshow('LiteGaze Enhanced Debug', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👉 使用者按下 'q' 離開")
                break
        
        except KeyboardInterrupt:
            print("\n🛑 強制中斷 (KeyboardInterrupt)")
            break
        except Exception as e:
            # 🔥 這是最重要的部分：捕捉迴圈內的任何錯誤並印出
            print("\n❌ 執行期間發生錯誤！")
            print("==========================================")
            traceback.print_exc()
            print("==========================================")
            break

except Exception as e:
    # 捕捉初始化階段的錯誤
    print("\n❌ 初始化失敗！")
    print("==========================================")
    traceback.print_exc()
    print("==========================================")

finally:
    # 確保資源釋放 (即使報錯也會執行)
    print("\n[Cleanup] 正在釋放資源...")
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    print("👋 程式已安全結束")
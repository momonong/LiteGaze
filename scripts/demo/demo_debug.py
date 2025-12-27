import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

# === ⚙️ 設定區 ===
MODEL_PATH = 'models/litegaze_v2_win.tflite' # 確保您的檔名正確
INPUT_SIZE = (60, 60)

def main():
    print("Loading model...")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    output_index = output_details[0]['index'] 

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    cap = cv2.VideoCapture(0)
    
    print("\n========== DEBUG MODE ==========")
    print("請觀察終端機輸出的數值，試著大幅度轉動頭部")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success: break

        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 裁切臉部
                x_coords = [lm.x for lm in face_landmarks.landmark]
                y_coords = [lm.y for lm in face_landmarks.landmark]
                x_min, x_max = int(min(x_coords)*w), int(max(x_coords)*w)
                y_min, y_max = int(min(y_coords)*h), int(max(y_coords)*h)
                
                # Padding (這很重要，如果太擠模型會看不懂)
                pad = 30
                x_min, y_min = max(0, x_min-pad), max(0, y_min-pad)
                x_max, y_max = min(w, x_max+pad), min(h, y_max+pad)
                
                face_img = image[y_min:y_max, x_min:x_max]
                
                if face_img.size > 0:
                    # 1. 顯示模型看到的圖片 (檢查是不是黑的)
                    debug_face = cv2.resize(face_img, (200, 200)) # 放大給人眼看
                    cv2.imshow('What Model Sees', debug_face)

                    # 2. 推論
                    # 2. 推論
                    input_img = cv2.resize(face_img, INPUT_SIZE)
                    
                    # ✅ 嘗試新的寫法 (-1 ~ 1)
                    input_data = (input_img.astype(np.float32) / 127.5) - 1.0
                    
                    # ⚠️ 檢查這裡：只能有一行 expand_dims
                    input_data = np.expand_dims(input_data, axis=0) 

                    # 如果下面還有 input_data = np.expand_dims(...) 請刪除它！

                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    pred = interpreter.get_tensor(output_index)[0]
                    
                    pitch, yaw = pred[0], pred[1]

                    # 3. 印出數值
                    # 用顏色區分數值變化
                    bar_len = 50
                    p_bar = int((pitch + 1.5) / 3 * bar_len) # 視覺化 Pitch
                    y_bar = int((yaw + 1.5) / 3 * bar_len)   # 視覺化 Yaw
                    
                    print(f"Pitch: {pitch:+.4f} [{'#'*p_bar:<{bar_len}}] | Yaw: {yaw:+.4f} [{'#'*y_bar:<{bar_len}}]", end='\r')

        cv2.imshow('Main', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
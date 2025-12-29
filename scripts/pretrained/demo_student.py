import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
import time
from math import cos, sin
import os

# ================= ⚙️ 設定區 =================
MODEL_PATH = 'models/student_mobilenet_v3.pth'
INPUT_SIZE = 224
# ============================================

# 定義學生模型結構
class L2CS_MobileNetV3(nn.Module):
    def __init__(self, num_bins=90):
        super(L2CS_MobileNetV3, self).__init__()
        self.numOfLabels = num_bins
        self.backbone = models.mobilenet_v3_large(weights=None)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_bins * 2)

    def forward(self, x):
        x = self.backbone(x)
        pitch = x[:, :self.numOfLabels]
        yaw = x[:, self.numOfLabels:]
        return pitch, yaw

def draw_gaze(image, pitch, yaw, center_x, center_y, face_width):
    # 箭頭長度設為臉寬的一半
    length = face_width / 2.0
    
    dx = -length * sin(yaw) * cos(pitch)
    dy = -length * sin(pitch)
    
    # 畫箭頭
    cv2.arrowedLine(image, (int(center_x), int(center_y)), 
                   (int(center_x + dx), int(center_y + dy)), 
                   (0, 0, 255), 4, cv2.LINE_AA, tipLength=0.2)

def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Loading Student Model on {device}...")

    # 1. 載入模型
    model = L2CS_MobileNetV3()
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print("✅ Student Model Loaded!")
    except FileNotFoundError:
        print(f"❌ Error: 找不到 {MODEL_PATH}")
        return

    # 2. 準備 Haar Cascade (取代 MediaPipe)
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        print("❌ Error: 無法載入 OpenCV Haar Cascade！")
        return

    # 3. 預處理
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    idx_tensor = torch.FloatTensor([idx for idx in range(90)]).to(device)
    cap = cv2.VideoCapture(0)
    
    # 計算 FPS 用
    prev_time = 0
    
    print("📷 Demo Started! (Student Model - No MediaPipe)")

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 鏡像翻轉
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # 轉灰階給 Haar 使用
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 偵測人臉
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                # 找最大的臉
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                x, y, w_face, h_face = faces[0]
                
                # 擴大裁切 (跟訓練時保持一致的 Padding)
                k = 0.5 
                # (注意：這裡要小心不要切出邊界)
                x_min = max(0, x - int(w_face * k))
                y_min = max(0, y - int(h_face * k))
                x_max = min(w, x + w_face + int(w_face * k)) # 修正寬度算法
                y_max = min(h, y + h_face + int(h_face * k)) # 修正高度算法
                
                face_img = frame[y_min:y_max, x_min:x_max]
                
                if face_img.size > 0:
                    # 轉 PIL
                    from PIL import Image
                    img_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    
                    # 推論
                    img_tensor = transform(img_pil).unsqueeze(0).to(device)
                    pitch_out, yaw_out = model(img_tensor)
                    
                    # 解碼
                    pitch_pred = softmax_temperature(pitch_out, 1)
                    yaw_pred = softmax_temperature(yaw_out, 1)
                    
                    pitch_deg = torch.sum(pitch_pred * idx_tensor, 1) * 4 - 180
                    yaw_deg = torch.sum(yaw_pred * idx_tensor, 1) * 4 - 180
                    
                    pitch_rad = pitch_deg[0].item() * np.pi / 180
                    yaw_rad = yaw_deg[0].item() * np.pi / 180
                    
                    # 估算鼻尖位置 (臉中心再稍微下面一點)
                    nose_x = x + w_face / 2
                    nose_y = y + h_face * 0.6 
                    
                    draw_gaze(frame, pitch_rad, yaw_rad, nose_x, nose_y, w_face)
                    
                    # 畫框框
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # 顯示數值
                    text = f"P: {pitch_rad:.2f} Y: {yaw_rad:.2f}"
                    cv2.putText(frame, text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 顯示 FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "Student (MobileNet)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Student Demo', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
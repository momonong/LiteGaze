import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import sys
import os
from math import cos, sin

# ================= ⚙️ 設定區 =================
# 請確認下載下來的檔名是這個
MODEL_PATH = "L2CSNet_gaze360.pkl" 
# ============================================

# === 1. 定義 L2CS-Net 模型結構 (這是為了載入權重) ===
class L2CS(nn.Module):
    def __init__(self, block, layers, num_bins):
        super(L2CS, self).__init__()
        import torchvision.models as models
        self.numOfLabels = num_bins
        # 官方用 ResNet50
        self.model = models.resnet50(weights=None) # 不下載預訓練權重，因為我們要載入自己的
        self.model.fc = nn.Linear(2048, self.numOfLabels * 2)

    def forward(self, x):
        x = self.model(x)
        pitch = x[:, :self.numOfLabels]
        yaw = x[:, self.numOfLabels:]
        return pitch, yaw

def get_pitch_yaw(pitch_predicted, yaw_predicted):
    # 官方的 Softmax 轉換邏輯
    pitch_predicted = torch.softmax(pitch_predicted, dim=1)
    yaw_predicted = torch.softmax(yaw_predicted, dim=1)
    
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda()
    
    pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 4 - 180
    yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 4 - 180
    
    pitch = pitch_predicted[0] * np.pi / 180
    yaw = yaw_predicted[0] * np.pi / 180
    return pitch.item(), yaw.item()

def draw_gaze(image, pitch, yaw, landmarks):
    h, w, c = image.shape
    length = w / 2
    
    # 鼻頭 index 4
    nose_x = int(landmarks[4].x * w)
    nose_y = int(landmarks[4].y * h)
    
    dx = -length * sin(yaw) * cos(pitch)
    dy = -length * sin(pitch)
    
    cv2.arrowedLine(image, (nose_x, nose_y), (int(nose_x + dx), int(nose_y + dy)), (0, 0, 255), 4)

def main():
    # 1. 檢查 GPU
    if not torch.cuda.is_available():
        print("❌ 警告: PyTorch 抓不到 GPU，會很慢！")
        device = torch.device('cpu')
    else:
        print(f"✅ PyTorch 使用 GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')

    # 2. 載入模型
    print("🚀 正在載入 L2CS-Net 官方模型...")
    model = L2CS(None, [3, 4, 6, 3], 90)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到權重檔: {MODEL_PATH}")
        print("請去下載: https://github.com/Ahmednull/L2CS-Net/raw/main/models/L2CSNet_gaze360.pkl")
        return

    # 載入權重 (Map location 確保在 GPU/CPU 正確載入)
    saved_state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(saved_state_dict)
    model.to(device)
    model.eval()
    print("✅ 模型載入成功！")

    # 3. MediaPipe 設定
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    
    # 4. 預處理 (官方建議的 Normalize 參數)
    transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(0)
    
    print("📷 Demo 啟動！請看鏡頭！(按 q 離開)")
    
    with torch.no_grad(): # 推論模式，不算梯度
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 抓臉框
                    x_coords = [lm.x for lm in face_landmarks.landmark]
                    y_coords = [lm.y for lm in face_landmarks.landmark]
                    x_min, x_max = int(min(x_coords)*w), int(max(x_coords)*w)
                    y_min, y_max = int(min(y_coords)*h), int(max(y_coords)*h)
                    
                    # 稍微擴大框框
                    margin_x, margin_y = int((x_max-x_min)*0.2), int((y_max-y_min)*0.2)
                    x_min, x_max = max(0, x_min-margin_x), min(w, x_max+margin_x)
                    y_min, y_max = max(0, y_min-margin_y), min(h, y_max+margin_y)
                    
                    face_img = frame[y_min:y_max, x_min:x_max]
                    
                    if face_img.size > 0:
                        # 預處理並送入 GPU
                        img_tensor = transformations(face_img).unsqueeze(0).to(device)
                        
                        # 推論
                        pitch_out, yaw_out = model(img_tensor)
                        pitch, yaw = get_pitch_yaw(pitch_out, yaw_out)
                        
                        # 畫圖
                        draw_gaze(frame, pitch, yaw, face_landmarks.landmark)
                        
                        # 顯示數值
                        cv2.putText(frame, f"Pitch: {pitch:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(frame, f"Yaw:   {yaw:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            cv2.imshow('Rescue Demo (PyTorch)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
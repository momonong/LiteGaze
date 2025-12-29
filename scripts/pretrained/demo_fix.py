import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# ================= ⚙️ 設定區 =================
MODEL_PATH = 'models\student_mobilenet_final_fix.pth'
INPUT_SIZE = 224
SMOOTHING_WINDOW = 5
# ============================================

class Stabilizer:
    def __init__(self, window_size=5):
        self.maxlen = window_size
        self.q_pitch = deque(maxlen=window_size)
        self.q_yaw = deque(maxlen=window_size)
    def update(self, pitch, yaw):
        self.q_pitch.append(pitch)
        self.q_yaw.append(yaw)
        return np.mean(self.q_pitch), np.mean(self.q_yaw)

class L2CS_MobileNetV3(nn.Module):
    def __init__(self, num_bins=90):
        super(L2CS_MobileNetV3, self).__init__()
        self.backbone = models.mobilenet_v3_large(weights=None)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_bins * 2)
    def forward(self, x):
        x = self.backbone(x)
        return x[:, :90], x[:, 90:]

def compute_gaze(logits):
    softmax = nn.Softmax(dim=1)
    prob = softmax(logits)
    idx = torch.arange(90, dtype=torch.float32).to(logits.device)
    gaze = torch.sum(prob * idx, dim=1) * 4 - 180
    return gaze * np.pi / 180

def draw_gaze(image, pitch, yaw, nose_x, nose_y, face_width):
    length = face_width / 2.0
    dx = -length * np.sin(yaw) * np.cos(pitch)
    dy = -length * np.sin(pitch)
    cv2.arrowedLine(image, (int(nose_x), int(nose_y)), 
                   (int(nose_x + dx), int(nose_y + dy)), 
                   (0, 0, 255), 4, cv2.LINE_AA, tipLength=0.2)

def get_square_crop(frame, x_min, y_min, x_max, y_max):
    # 1. 計算目前的中心點和長寬
    h, w, _ = frame.shape
    cw, ch = x_max - x_min, y_max - y_min
    cx = x_min + cw // 2
    cy = y_min + ch // 2
    
    # 2. 取長寬的最大值，作為正方形的邊長
    side = max(cw, ch)
    # 稍微再擴大一點 (1.2倍)，確保臉都在裡面，不要切到下巴
    side = int(side * 1.2)
    
    # 3. 重新計算左上角 (確保是正方形)
    new_x_min = max(0, cx - side // 2)
    new_y_min = max(0, cy - side // 2)
    new_x_max = min(w, cx + side // 2)
    new_y_max = min(h, cy + side // 2)
    
    return frame[new_y_min:new_y_max, new_x_min:new_x_max], side

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Loading Student Model on {device}...")

    model = L2CS_MobileNetV3()
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    cap = cv2.VideoCapture(0)
    stabilizer = Stabilizer(SMOOTHING_WINDOW)

    # 🎛️ 校正參數 (預設加強靈敏度)
    flip_frame = True
    mul_pitch = 1.5   # 加強上下
    mul_yaw = -1.5    # 加強左右
    offset_pitch = 0.0

    print("✅ Demo Started with SQUARE CROP debug.")

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if flip_frame: frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe Detection
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # 取得臉部範圍
                x_c = [lm.x * w for lm in landmarks]
                y_c = [lm.y * h for lm in landmarks]
                x_min, x_max = int(min(x_c)), int(max(x_c))
                y_min, y_max = int(min(y_c)), int(max(y_c))
                
                # 鼻尖
                nose_x, nose_y = int(landmarks[4].x * w), int(landmarks[4].y * h)
                
                # 🔥 關鍵修正：強制正方形裁切
                face_img, face_side = get_square_crop(frame, x_min, y_min, x_max, y_max)
                
                if face_img.size > 0:
                    # 顯示模型看到的畫面 (Debug View)
                    debug_view = cv2.resize(face_img, (150, 150))
                    frame[0:150, 0:150] = debug_view
                    cv2.putText(frame, "Model Input", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.rectangle(frame, (0,0), (150,150), (0,255,0), 2)

                    # 推論
                    img_pil = transforms.ToPILImage()(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    img_tensor = transform(img_pil).unsqueeze(0).to(device)
                    p_out, y_out = model(img_tensor)
                    
                    raw_pitch = compute_gaze(p_out).item()
                    raw_yaw = compute_gaze(y_out).item()
                    
                    # 校正
                    final_pitch = (raw_pitch * mul_pitch) + offset_pitch
                    final_yaw = raw_yaw * mul_yaw
                    
                    # 平滑
                    s_pitch, s_yaw = stabilizer.update(final_pitch, final_yaw)
                    
                    draw_gaze(frame, s_pitch, s_yaw, nose_x, nose_y, face_side)
                    
                    # 畫正方形框
                    side_half = face_side // 2
                    cx, cy = (x_min + x_max)//2, (y_min + y_max)//2
                    cv2.rectangle(frame, (cx-side_half, cy-side_half), (cx+side_half, cy+side_half), (0, 255, 0), 2)

            # 顯示
            status = f"Pitch Offset (W/S): {offset_pitch:.2f}"
            cv2.putText(frame, status, (160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Final Debug Demo', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('w'): offset_pitch -= 0.05
            elif key == ord('s'): offset_pitch += 0.05
            elif key == ord('x'): mul_yaw *= -1 # 反轉左右

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
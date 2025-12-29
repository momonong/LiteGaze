import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import mediapipe as mp
import numpy as np

# ================= ⚙️ 設定 =================
# 你的學生模型 (請確認是 Production 或最新的那個)
STUDENT_PATH = 'models/student_mobilenet_production.pth'
# 你的老師模型 (你確認是準的那個)
TEACHER_PATH = 'models/L2CSNet_gaze360.pkl'
# ==========================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 模型定義 ===
class L2CS_ResNet50(nn.Module):
    def __init__(self, num_bins=90):
        super(L2CS_ResNet50, self).__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(2048, num_bins * 2)
    def forward(self, x):
        x = self.model(x)
        return x[:, :90], x[:, 90:]

class L2CS_MobileNetV3(nn.Module):
    def __init__(self, num_bins=90):
        super(L2CS_MobileNetV3, self).__init__()
        self.backbone = models.mobilenet_v3_large(weights=None)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_bins * 2)
    def forward(self, x):
        x = self.backbone(x)
        return x[:, :90], x[:, 90:]

# === 載入模型 ===
def load_models():
    print("Loading Teacher...")
    teacher = L2CS_ResNet50().to(device)
    ckpt = torch.load(TEACHER_PATH, map_location=device)
    # 萬能載入 Teacher
    state = {}
    for k, v in ckpt.items():
        if 'fc_pitch' in k or 'fc_yaw' in k: continue
        nk = 'model.'+k if not k.startswith('model.') else k
        state[nk] = v
    if 'fc_pitch_gaze.weight' in ckpt:
        state['model.fc.weight'] = torch.cat((ckpt['fc_pitch_gaze.weight'], ckpt['fc_yaw_gaze.weight']), 0)
        state['model.fc.bias'] = torch.cat((ckpt['fc_pitch_gaze.bias'], ckpt['fc_yaw_gaze.bias']), 0)
    teacher.load_state_dict(state, strict=False)
    teacher.eval()

    print("Loading Student...")
    student = L2CS_MobileNetV3().to(device)
    try:
        student.load_state_dict(torch.load(STUDENT_PATH, map_location=device))
    except:
        print("❌ 學生模型載入失敗！路徑對嗎？")
    student.eval()
    return teacher, student

def compute_gaze(logits):
    softmax = nn.Softmax(dim=1)
    prob = softmax(logits)
    idx = torch.arange(90, dtype=torch.float32).to(logits.device)
    gaze = torch.sum(prob * idx, dim=1) * 4 - 180
    return gaze.item()

def main():
    teacher, student = load_models()
    
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    cap = cv2.VideoCapture(0)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\n🧐 診斷開始：請做這三個動作")
    print("1. 抬頭看天花板 (測試 Pitch+)")
    print("2. 低頭看地板 (測試 Pitch-)")
    print("3. 看螢幕正中央 (測試歸零)")
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = mp_face_mesh.process(rgb)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                # 簡單裁切
                x_c = [lm.x * w for lm in landmarks]
                y_c = [lm.y * h for lm in landmarks]
                x_min, x_max = min(x_c), max(x_c)
                y_min, y_max = min(y_c), max(y_c)
                ww, hh = x_max-x_min, y_max-y_min
                
                # 給一點 padding
                k = 0.5
                x_min = max(0, int(x_min - ww*k))
                y_min = max(0, int(y_min - hh*k))
                x_max = min(w, int(x_max + ww*k))
                y_max = min(h, int(y_max + hh*k))
                
                face_img = frame[y_min:y_max, x_min:x_max]
                
                if face_img.size > 0:
                    inp_pil = transforms.ToPILImage()(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    inp_tensor = transform(inp_pil).unsqueeze(0).to(device)
                    
                    # 雙模型推論
                    tp, ty = teacher(inp_tensor)
                    sp, sy = student(inp_tensor)
                    
                    t_pitch = compute_gaze(tp)
                    s_pitch = compute_gaze(sp)
                    
                    # 顯示數值
                    # 綠色 = 老師, 紅色 = 學生
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # 繪製條狀圖 (Bar Chart) 讓你直觀看到數值差距
                    bar_x = 50
                    cv2.putText(frame, f"Teacher Pitch: {t_pitch:.2f}", (bar_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Student Pitch: {s_pitch:.2f}", (bar_x, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # 畫一條線代表 0 度
                    cv2.line(frame, (bar_x, 120), (bar_x+200, 120), (200, 200, 200), 2)
                    
                    # 畫 Teacher 點 (綠)
                    t_pos = int(120 - t_pitch * 2) # 放大顯示
                    cv2.circle(frame, (bar_x + 50, t_pos), 10, (0, 255, 0), -1)
                    
                    # 畫 Student 點 (紅)
                    s_pos = int(120 - s_pitch * 2)
                    cv2.circle(frame, (bar_x + 100, s_pos), 10, (0, 0, 255), -1)

            cv2.imshow("Diagnosis", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
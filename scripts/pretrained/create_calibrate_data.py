import cv2
import torch
import os
import numpy as np
from torchvision import transforms, models
import torch.nn as nn
import mediapipe as mp

# ================= 設定 =================
OUTPUT_DIR = 'data/final_calibration'  # 換一個新資料夾
TEACHER_PATH = 'models/L2CSNet_gaze360.pkl'
# =======================================

class L2CS_ResNet50(nn.Module):
    def __init__(self, num_bins=90):
        super(L2CS_ResNet50, self).__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(2048, num_bins * 2)
    def forward(self, x):
        x = self.model(x)
        return x[:, :90], x[:, 90:]

def compute_gaze(logits):
    softmax = nn.Softmax(dim=1)
    prob = softmax(logits)
    idx = torch.arange(90, dtype=torch.float32).to(logits.device)
    gaze = torch.sum(prob * idx, dim=1) * 4 - 180
    return gaze * np.pi / 180

def draw_gaze(image, pitch, yaw, nose_x, nose_y, face_width, color):
    length = face_width / 2.0
    dx = -length * np.sin(yaw) * np.cos(pitch)
    dy = -length * np.sin(pitch)
    cv2.arrowedLine(image, (int(nose_x), int(nose_y)), 
                   (int(nose_x + dx), int(nose_y + dy)), 
                   color, 4, cv2.LINE_AA, tipLength=0.2)

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    device = torch.device('cuda')
    print("載入 Teacher...")
    teacher = L2CS_ResNet50().to(device)
    ckpt = torch.load(TEACHER_PATH, map_location=device)
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

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    cap = cv2.VideoCapture(0)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 動作列表
    actions = ["LOOK UP (看上面)", "LOOK DOWN (看下面)", "LOOK LEFT (看左邊)", "LOOK RIGHT (看右邊)", "LOOK CENTER (看中間)"]
    action_idx = 0
    samples_per_action = 100 # 每個動作 100 張
    current_samples = 0
    collecting = False
    
    print("\n🚀 最終校正數據收集")
    print("請按【空白鍵】切換 收集/暫停")
    print("⚠️ 只有當綠色箭頭方向正確時，才收集！")

    while action_idx < len(actions):
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = mp_face_mesh.process(rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            x_c = [lm.x * w for lm in landmarks]
            y_c = [lm.y * h for lm in landmarks]
            x_min, x_max = min(x_c), max(x_c)
            y_min, y_max = min(y_c), max(y_c)
            ww, hh = x_max-x_min, y_max-y_min
            k = 0.3
            x_min = max(0, int(x_min - ww*k))
            y_min = max(0, int(y_min - hh*k))
            x_max = min(w, int(x_max + ww*k))
            y_max = min(h, int(y_max + hh*k))
            
            face_img = frame[y_min:y_max, x_min:x_max]
            
            if face_img.size > 0:
                nose_x, nose_y = landmarks[4].x * w, landmarks[4].y * h
                
                # 讓老師預測
                inp = transform(transforms.ToPILImage()(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
                with torch.no_grad():
                    tp, ty = teacher(inp)
                
                tp_rad = compute_gaze(tp).item()
                ty_rad = compute_gaze(ty).item()
                
                # 畫圖
                color = (0, 0, 255) if not collecting else (0, 255, 0)
                draw_gaze(frame, tp_rad, ty_rad, nose_x, nose_y, ww, color)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                
                if collecting:
                    # 儲存
                    filename = f"{OUTPUT_DIR}/act{action_idx}_{current_samples:04d}.jpg"
                    cv2.imwrite(filename, face_img)
                    current_samples += 1
                    if current_samples >= samples_per_action:
                        collecting = False
                        action_idx += 1
                        current_samples = 0
                        print("✅ 下一個動作...")

        # UI
        if action_idx < len(actions):
            cv2.putText(frame, f"DO: {actions[action_idx]}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"{current_samples}/{samples_per_action}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Done! Press Q", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 32: collecting = not collecting
        elif key == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
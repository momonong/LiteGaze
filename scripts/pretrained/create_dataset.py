import cv2
import torch
import os
import time
from torchvision import transforms, models
import torch.nn as nn

# ================= 設定 =================
OUTPUT_DIR = 'data/my_gaze'
TEACHER_PATH = 'models/L2CSNet_gaze360.pkl'
FRAMES_TO_CAPTURE = 2000 
# =======================================

class L2CS_ResNet50(nn.Module):
    def __init__(self, num_bins=90):
        super(L2CS_ResNet50, self).__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(2048, num_bins * 2)
    def forward(self, x):
        x = self.model(x)
        return x[:, :90], x[:, 90:]

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    device = torch.device('cuda')
    print("載入老師模型中...")
    teacher = L2CS_ResNet50().to(device)
    
    # 載入權重 (萬能載入法)
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
    
    cap = cv2.VideoCapture(0)
    # 確保使用 OpenCV 內建的人臉偵測
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("📷 開始錄製！請在鏡頭前做各種眼神動作。收集 2000 張。")
    print("⚠️ 請點擊視窗一下，確保視窗有吃到焦點，按 'q' 可提早結束。")
    
    count = 0
    while count < FRAMES_TO_CAPTURE:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) > 0:
            # 找最大的臉
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w, h = faces[0]
            
            # 擴大裁切 (跟 Demo 一致的 0.5 padding)
            k = 0.5
            x_min = max(0, x - int(w*k))
            y_min = max(0, y - int(h*k))
            x_max = min(frame.shape[1], x + w + int(w*k))
            y_max = min(frame.shape[0], y + h + int(h*k))
            
            face_img = frame[y_min:y_max, x_min:x_max]
            
            if face_img.size > 0:
                filename = f"{OUTPUT_DIR}/frame_{count:05d}.jpg"
                cv2.imwrite(filename, face_img)
                count += 1
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        cv2.putText(frame, f"Collected: {count}/{FRAMES_TO_CAPTURE}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Data Collection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 資料收集完畢！")

if __name__ == "__main__":
    main()
import torch
import cv2
import glob
import numpy as np
import random
from torchvision import transforms, models
import torch.nn as nn
from math import cos, sin
from PIL import Image

# ================= ⚙️ 設定 =================
TEACHER_PATH = 'models/L2CSNet_gaze360.pkl'
STUDENT_PATH = 'models/student_mobilenet_v3.pth'
DATA_DIR = 'data/distill_images'
# ==========================================

# (省略重複的模型定義 Class，請確保這裡有 L2CS_ResNet50 和 L2CS_MobileNetV3)
# 為了節省版面，請直接複製 train_distill_final.py 裡面的那兩個 Class 定義貼在這邊
# ... [Class L2CS_ResNet50] ...
# ... [Class L2CS_MobileNetV3] ...
# ⬇️ 臨時貼上方便你執行，請確保縮排正確 ⬇️
class L2CS_ResNet50(nn.Module):
    def __init__(self, num_bins=90):
        super(L2CS_ResNet50, self).__init__()
        self.numOfLabels = num_bins
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(2048, num_bins * 2)
    def forward(self, x):
        x = self.model(x)
        return x[:, :90], x[:, 90:]

class L2CS_MobileNetV3(nn.Module):
    def __init__(self, num_bins=90):
        super(L2CS_MobileNetV3, self).__init__()
        self.numOfLabels = num_bins
        self.backbone = models.mobilenet_v3_large(weights=None)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_bins * 2)
    def forward(self, x):
        x = self.backbone(x)
        return x[:, :90], x[:, 90:]
# ⬆️ 結束 Class 定義 ⬆️

def softmax_and_decode(pitch_out, yaw_out):
    idx_tensor = torch.arange(90, dtype=torch.float32).to(pitch_out.device)
    
    pitch_prob = torch.softmax(pitch_out, dim=1)
    yaw_prob = torch.softmax(yaw_out, dim=1)
    
    pitch_deg = torch.sum(pitch_prob * idx_tensor, 1) * 4 - 180
    yaw_deg = torch.sum(yaw_prob * idx_tensor, 1) * 4 - 180
    
    return pitch_deg.item() * np.pi / 180, yaw_deg.item() * np.pi / 180

def draw_arrow(img, pitch, yaw, color, text):
    h, w, _ = img.shape
    length = w / 2
    cx, cy = w // 2, h // 2
    dx = -length * sin(yaw) * cos(pitch)
    dy = -length * sin(pitch)
    cv2.arrowedLine(img, (cx, cy), (int(cx+dx), int(cy+dy)), color, 3, tipLength=0.3)
    cv2.putText(img, text, (10, h-10 if color==(0,0,255) else 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    device = torch.device('cuda')
    
    # 1. 載入老師
    print("Load Teacher...")
    teacher = L2CS_ResNet50().to(device)
    # 這裡用之前的萬能載入法邏輯簡化版
    ckpt = torch.load(TEACHER_PATH, map_location=device)
    state = {}
    for k, v in ckpt.items():
        if 'fc_pitch' in k or 'fc_yaw' in k: continue
        nk = 'model.'+k if not k.startswith('model.') else k
        state[nk] = v
    # 合併 FC
    if 'fc_pitch_gaze.weight' in ckpt:
        state['model.fc.weight'] = torch.cat((ckpt['fc_pitch_gaze.weight'], ckpt['fc_yaw_gaze.weight']), 0)
        state['model.fc.bias'] = torch.cat((ckpt['fc_pitch_gaze.bias'], ckpt['fc_yaw_gaze.bias']), 0)
    teacher.load_state_dict(state, strict=False)
    teacher.eval()

    # 2. 載入學生
    print("Load Student...")
    student = L2CS_MobileNetV3().to(device)
    student.load_state_dict(torch.load(STUDENT_PATH, map_location=device))
    student.eval()

    # 3. 隨機抓圖
    img_paths = glob.glob(f"{DATA_DIR}/*.jpg")
    if not img_paths:
        print("No images found!")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Checking random images... Press space for next, q to quit.")

    while True:
        path = random.choice(img_paths)
        img_origin = cv2.imread(path)
        if img_origin is None: continue
        
        # 轉 tensor
        img_pil = Image.fromarray(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB))
        inp = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            tp, ty = teacher(inp)
            sp, sy = student(inp)
        
        t_pitch, t_yaw = softmax_and_decode(tp, ty)
        s_pitch, s_yaw = softmax_and_decode(sp, sy)

        # 畫圖 (綠色是老師，紅色是學生)
        display = cv2.resize(img_origin, (400, 400))
        draw_arrow(display, t_pitch, t_yaw, (0, 255, 0), f"Teacher P:{t_pitch:.2f}")
        draw_arrow(display, s_pitch, s_yaw, (0, 0, 255), f"Student P:{s_pitch:.2f}")

        cv2.imshow("Diagnosis (Green=Teacher, Red=Student)", display)
        key = cv2.waitKey(0)
        if key == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
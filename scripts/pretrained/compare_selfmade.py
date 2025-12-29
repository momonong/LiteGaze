import torch
import torch.nn as nn
from torchvision import transforms, models
import os
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm

# ================= ⚙️ 設定 =================
DATA_DIR = 'data/selfmade_combined'  # 你的 9000 張圖
TEACHER_PATH = 'models/L2CSNet_gaze360.pkl'
STUDENT_PATH = 'models/student_mobilenet_3people_9k.pth'
DEVICE = torch.device('cuda') # 評估時用 GPU 跑比較快
BATCH_SIZE = 128
# ==========================================

# 模型定義 (省略重複部分，保持一致)
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

def compute_gaze(logits):
    softmax = nn.Softmax(dim=1)
    prob = softmax(logits)
    idx = torch.arange(90, dtype=torch.float32).to(logits.device)
    gaze = torch.sum(prob * idx, dim=1) * 4 - 180
    return gaze

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def main():
    print("🚀 開始評估模型指標...")
    
    # 1. 載入模型
    teacher = L2CS_ResNet50().to(DEVICE)
    ckpt = torch.load(TEACHER_PATH, map_location=DEVICE)
    # (載入 Teacher 權重代碼省略，同前)
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
    
    student = L2CS_MobileNetV3().to(DEVICE)
    student.load_state_dict(torch.load(STUDENT_PATH, map_location=DEVICE))
    student.eval()

    # 2. 計算模型大小
    t_size = get_model_size(teacher)
    s_size = get_model_size(student)
    print(f"\n📊 模型大小比較:")
    print(f"Teacher (ResNet50)   : {t_size:.2f} MB")
    print(f"Student (MobileNetV3): {s_size:.2f} MB")
    print(f"👉 壓縮率: {t_size/s_size:.1f}x (縮小了 {t_size/s_size:.1f} 倍)")

    # 3. 準備資料
    files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 4. 開始計算誤差
    pitch_errors = []
    yaw_errors = []
    
    print(f"\nrunning evaluation on {len(files)} images...")
    
    # 為了方便，我們手動做 batch
    batch_imgs = []
    for i, f in enumerate(tqdm(files)):
        try:
            img = transform(Image.open(f).convert('RGB'))
            batch_imgs.append(img)
        except: continue
        
        if len(batch_imgs) == BATCH_SIZE or i == len(files)-1:
            if not batch_imgs: break
            
            inp = torch.stack(batch_imgs).to(DEVICE)
            
            with torch.no_grad():
                tp, ty = teacher(inp)
                sp, sy = student(inp)
                
                t_pitch = compute_gaze(tp)
                t_yaw = compute_gaze(ty)
                s_pitch = compute_gaze(sp)
                s_yaw = compute_gaze(sy)
                
                # 計算絕對誤差 (Absolute Error)
                p_err = torch.abs(t_pitch - s_pitch)
                y_err = torch.abs(t_yaw - s_yaw)
                
                pitch_errors.extend(p_err.cpu().numpy())
                yaw_errors.extend(y_err.cpu().numpy())
            
            batch_imgs = []

    # 5. 總結報告
    mae_pitch = np.mean(pitch_errors)
    mae_yaw = np.mean(yaw_errors)
    
    print("\n" + "="*40)
    print("📈 準確度評估報告 (Accuracy Report)")
    print("="*40)
    print(f"MAE Pitch : {mae_pitch:.2f}°")
    print(f"MAE Yaw   : {mae_yaw:.2f}°")
    print(f"Avg Error : {(mae_pitch + mae_yaw)/2:.2f}°")
    print("="*40)
    print("💡 解讀：")
    print("- < 3.0° : 完美蒸餾 (Perfect)")
    print("- 3.0°~5.0° : 優良 (Good)")
    print("- > 5.0° : 尚可 (Acceptable)")

if __name__ == "__main__":
    main()
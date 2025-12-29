import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import glob
import os
import cv2
import numpy as np
from PIL import Image

# ================= ⚙️ 極速設定 =================
DATA_DIR = 'data/selfmade_combined'
TEACHER_PATH = 'models/L2CSNet_gaze360.pkl'
STUDENT_SAVE_PATH = 'models/student_mobilenet_3people_9k.pth'

DEVICE = torch.device('cuda')

# 🔥 5090 專屬優化參數
BATCH_SIZE = 512        # 直接開 256 或 512 (MobileNet 很小，放心開)
EPOCHS = 50             # 既然跑得快了，就練滿 50 輪，讓它徹底學會
LR = 1e-3               # Batch 變大，學習率通常也可以稍微調大一點點
TEMP = 5.0
NUM_WORKERS = 8         # 開 8 個 CPU 核心幫忙讀圖
# ==========================================

# Teacher Definition
class L2CS_ResNet50(nn.Module):
    def __init__(self, num_bins=90):
        super(L2CS_ResNet50, self).__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(2048, num_bins * 2)
    def forward(self, x):
        x = self.model(x)
        return x[:, :90], x[:, 90:]

# Student Definition
class L2CS_MobileNetV3(nn.Module):
    def __init__(self, num_bins=90):
        super(L2CS_MobileNetV3, self).__init__()
        # 載入 ImageNet 預訓練權重 (這對防止過擬合很有幫助)
        self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_bins * 2)
    def forward(self, x):
        x = self.backbone(x)
        return x[:, :90], x[:, 90:]

class PerfectDataset(Dataset):
    def __init__(self, root):
        self.files = glob.glob(os.path.join(root, "*.jpg"))
        print(f"📊 載入資料: {len(self.files)} 張")
        
        # 🔥 強力數據增強：防止過擬合到你的個人特徵
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # 隨機改變亮度、對比、飽和度 (讓它認不出是同一個房間/光線)
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            # 隨機灰階
            transforms.RandomGrayscale(p=0.1),
            # 隨機模糊 (模擬動態模糊)
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        try:
            img = Image.open(self.files[i]).convert('RGB')
            return self.transform(img)
        except: return torch.zeros(3, 224, 224)

def distillation_loss(student_logits, teacher_logits, T):
    soft_targets = nn.functional.softmax(teacher_logits / T, dim=1)
    soft_prob = nn.functional.log_softmax(student_logits / T, dim=1)
    return nn.KLDivLoss(reduction='batchmean')(soft_prob, soft_targets) * (T**2)

def main():
    print("🚀 啟動完美蒸餾程序...")
    
    # 1. Teacher
    teacher = L2CS_ResNet50().to(DEVICE)
    ckpt = torch.load(TEACHER_PATH, map_location=DEVICE)
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
    
    # 2. Student
    student = L2CS_MobileNetV3().to(DEVICE)
    student.train()
    
    dataset = PerfectDataset(DATA_DIR)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS,      # 多工讀圖
        pin_memory=True,              # 加速傳輸
        persistent_workers=True,      # 讓工人待命，不要一直重啟
        prefetch_factor=4             # 每個工人預先多讀 4 個 Batch
    )
    optimizer = optim.Adam(student.parameters(), lr=LR)
    
    print("🔥 開始訓練...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for images in dataloader:
            images = images.to(DEVICE)
            
            # Teacher 產生 Logits
            with torch.no_grad():
                tp, ty = teacher(images)
            
            # Student 產生 Logits
            sp, sy = student(images)
            
            loss = distillation_loss(sp, tp, TEMP) + distillation_loss(sy, ty, TEMP)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f}")
        
    torch.save(student.state_dict(), STUDENT_SAVE_PATH)
    print(f"✅ 模型已儲存: {STUDENT_SAVE_PATH}")

if __name__ == '__main__':
    main()
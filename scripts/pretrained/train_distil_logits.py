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

# ================= ⚙️ 設定區 =================
# 請指向你原本那個最大的 MPII 資料集
DATA_DIR = 'data/distill_images' 
TEACHER_PATH = 'models/L2CSNet_gaze360.pkl'
STUDENT_SAVE_PATH = 'models/student_mobilenet_mpii_logits.pth'

DEVICE = torch.device('cuda')
BATCH_SIZE = 64      # 5090 可以開大一點，跑快一點
EPOCHS = 20          # MPII 資料多，20 輪就很強了
LR = 1e-4            # 標準學習率
TEMP = 5.0           # 蒸餾溫度 (讓分佈更平滑，更好學)
# ============================================

# === 1. 模型定義 ===
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
        # 載入 ImageNet 預訓練，確保骨幹有基礎視覺能力
        self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_bins * 2)
    def forward(self, x):
        x = self.backbone(x)
        return x[:, :90], x[:, 90:]

# === 2. 資料集 (直接讀取切好的圖) ===
class MPIIDataset(Dataset):
    def __init__(self, root):
        # 支援 jpg 和 png
        self.files = glob.glob(os.path.join(root, "*.jpg")) + glob.glob(os.path.join(root, "*.png"))
        print(f"📊 載入 MPII 資料集: 共 {len(self.files)} 張圖片")
        
        # 訓練時加入一點點增強，讓模型更強壯
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.files)
    
    def __getitem__(self, i):
        try:
            img = Image.open(self.files[i]).convert('RGB')
            return self.transform(img)
        except:
            return torch.zeros(3, 224, 224)

# === 3. 蒸餾 Loss (KL Divergence) ===
def distillation_loss(student_logits, teacher_logits, T):
    # Logits -> Softmax 分佈
    soft_targets = nn.functional.softmax(teacher_logits / T, dim=1)
    soft_prob = nn.functional.log_softmax(student_logits / T, dim=1)
    # 計算分佈差異
    loss = nn.KLDivLoss(reduction='batchmean')(soft_prob, soft_targets) * (T**2)
    return loss

def main():
    print(f"🚀 啟動 MPII Logit Distillation on {DEVICE}...")
    
    # A. 準備老師
    print("👨‍🏫 Loading Teacher...")
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
    teacher.eval() # 老師不訓練
    
    # B. 準備學生
    print("👶 Initializing Student (MobileNetV3)...")
    student = L2CS_MobileNetV3().to(DEVICE)
    student.train()
    
    # C. 準備資料
    dataset = MPIIDataset(DATA_DIR)
    if len(dataset) == 0:
        print("❌ 錯誤：找不到資料！請確認 datasets/distill_images 是否存在。")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    optimizer = optim.Adam(student.parameters(), lr=LR)
    
    print(f"🔥 開始訓練 (Temp={TEMP})...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        batch_count = 0
        
        for i, images in enumerate(dataloader):
            images = images.to(DEVICE)
            
            # 1. 老師看圖 -> 產生 Logits (不只是一個角度，而是90個信心分數)
            with torch.no_grad():
                t_pitch_logits, t_yaw_logits = teacher(images)
            
            # 2. 學生看圖 -> 產生 Logits
            s_pitch_logits, s_yaw_logits = student(images)
            
            # 3. 計算 KL Loss
            loss_p = distillation_loss(s_pitch_logits, t_pitch_logits, TEMP)
            loss_y = distillation_loss(s_yaw_logits, t_yaw_logits, TEMP)
            loss = loss_p + loss_y
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if i % 20 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(dataloader)} | Loss: {loss.item():.4f}", end='\r')
                
        avg_loss = total_loss / batch_count
        print(f"\n✅ Epoch {epoch+1} Done. Avg Distill Loss: {avg_loss:.4f}")
        
        # 每 5 輪存一次，以防萬一
        if (epoch+1) % 5 == 0:
            torch.save(student.state_dict(), STUDENT_SAVE_PATH)

    print(f"🎉 訓練完成！模型已存為: {STUDENT_SAVE_PATH}")
    print("👉 這個模型擁有 MPII 的大數據知識，以及老師的 Logit 判斷邏輯。")

if __name__ == '__main__':
    main()
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import glob
from PIL import Image
import sys
import copy

# ================= ⚙️ 生產級設定區 =================
TEACHER_PATH = 'models/L2CSNet_gaze360.pkl'
DATA_DIR = 'data/distill_images'
# 存檔名稱：加上 production 以示區別
STUDENT_SAVE_PATH = 'models/student_mobilenet_production.pth'

# 參數調整：拉長訓練時間
BATCH_SIZE = 64
EPOCHS = 50          # 增加到 50 輪 (預估 15 分鐘內跑完)
LR = 1e-4
# ==================================================

# === 1. 輔助函式：從 Softmax 算出角度 ===
def compute_gaze(logits):
    softmax = nn.Softmax(dim=1)
    prob = softmax(logits)
    idx = torch.arange(90, dtype=torch.float32).to(logits.device)
    gaze = torch.sum(prob * idx, dim=1) * 4 - 180
    return gaze

# === 2. 模型定義 ===
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
        self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_bins * 2)
    def forward(self, x):
        x = self.backbone(x)
        return x[:, :90], x[:, 90:]

# === 3. 資料集 (加入隨機增強) ===
class DistillDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_paths = glob.glob(os.path.join(img_dir, "*.jpg")) + \
                         glob.glob(os.path.join(img_dir, "*.png"))
        self.transform = transform
        print(f"📊 Production Training: Found {len(self.img_paths)} images.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.img_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except:
            return torch.zeros((3, 224, 224))

# === 4. 混合損失函數 ===
def loss_fn(s_pitch, s_yaw, t_pitch, t_yaw, T=1.0):
    # KL Divergence
    loss_kl = nn.KLDivLoss(reduction='batchmean')(
        nn.functional.log_softmax(s_pitch/T, dim=1),
        nn.functional.softmax(t_pitch/T, dim=1)
    ) * (T**2) + \
    nn.KLDivLoss(reduction='batchmean')(
        nn.functional.log_softmax(s_yaw/T, dim=1),
        nn.functional.softmax(t_yaw/T, dim=1)
    ) * (T**2)

    # MSE Loss (角度誤差)
    s_p_deg = compute_gaze(s_pitch)
    s_y_deg = compute_gaze(s_yaw)
    t_p_deg = compute_gaze(t_pitch)
    t_y_deg = compute_gaze(t_yaw)
    
    loss_mse = nn.MSELoss()(s_p_deg, t_p_deg) + nn.MSELoss()(s_y_deg, t_y_deg)
    
    # 權重分配：讓 MSE 佔比稍微重一點，強迫學生準確
    return 1.0 * loss_kl + 0.5 * loss_mse

# === 5. 主程式 ===
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting Production Distillation on {device}...")
    
    # A. 載入老師
    print("👨‍🏫 Loading Teacher...")
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
    for p in teacher.parameters(): p.requires_grad = False

    # B. 學生
    print("👶 Initializing Student...")
    student = L2CS_MobileNetV3().to(device)
    student.train()

    # C. 資料增強
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2), # 增強魯棒性
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = DistillDataset(DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    
    optimizer = optim.Adam(student.parameters(), lr=LR)
    
    # 🔥 新增：學習率排程器 (Loss 卡住時自動降低 LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print(f"🔥 Training Start! Epochs: {EPOCHS} | Best Model Checkpointing Enabled")
    
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(student.state_dict())

    for epoch in range(EPOCHS):
        total_loss = 0
        batch_cnt = 0
        
        for i, images in enumerate(dataloader):
            images = images.to(device)
            
            with torch.no_grad():
                tp, ty = teacher(images)
            
            sp, sy = student(images)
            
            loss = loss_fn(sp, sy, tp, ty, T=1.0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_cnt += 1
            
            if i % 20 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {i} | Loss: {loss.item():.4f}", end='\r')
        
        avg_loss = total_loss / batch_cnt
        print(f"\n✅ Epoch {epoch+1} Done. Avg Loss: {avg_loss:.4f}")
        
        # 更新 Scheduler
        scheduler.step(avg_loss)
        
        # 🔥 只存最好的模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_wts = copy.deepcopy(student.state_dict())
            torch.save(student.state_dict(), STUDENT_SAVE_PATH)
            print(f"⭐ New Best Model Saved! (Loss: {best_loss:.4f})")
        else:
            print(f"   (Best Loss so far: {best_loss:.4f})")

    print(f"🎉 Training Complete! Best Loss: {best_loss:.4f}")
    # 確保最後存的是最好的權重
    torch.save(best_model_wts, STUDENT_SAVE_PATH)

if __name__ == '__main__':
    main()
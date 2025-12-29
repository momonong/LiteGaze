import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import glob
from PIL import Image

# ================= ⚙️ 設定區 =================
# 指向剛剛那個數據很漂亮的資料夾
DATA_DIR = 'data/official_calibration' 
# 老師模型 (維持不變)
TEACHER_PATH = 'models/L2CSNet_gaze360.pkl'
# 這是我們要拯救的學生模型 (載入你原本最好的那個 production)
STUDENT_LOAD_PATH = 'models/student_mobilenet_production.pth'
# 這是最終成品
STUDENT_SAVE_PATH = 'models/student_mobilenet_final_fix.pth'

BATCH_SIZE = 16  # 小 Batch 讓它學得更細
EPOCHS = 20      # 20 輪暴力矯正
LR = 0.001       # 較大的學習率 (1e-3)
# ============================================

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

class SimpleDataset(Dataset):
    def __init__(self, root):
        self.files = glob.glob(os.path.join(root, "*.jpg"))
        # ⚠️ 關鍵：這裡不做任何 ColorJitter，我們要它死記硬背你的環境
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        try:
            img = Image.open(self.files[i]).convert('RGB')
            return self.transform(img)
        except: return torch.zeros(3, 224, 224)

def main():
    device = torch.device('cuda')
    print("🚀 啟動最終微調 (Final Finetune)...")

    # 1. 載入老師
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

    # 2. 載入學生
    print(f"📥 載入學生: {STUDENT_LOAD_PATH}")
    student = L2CS_MobileNetV3().to(device)
    try:
        student.load_state_dict(torch.load(STUDENT_LOAD_PATH, map_location=device))
    except:
        print("⚠️ 警告：找不到舊學生模型，將從頭開始訓練 (這也沒問題)")
        # 如果找不到舊的，就讓它用 ImageNet 權重重新學這 500 張
        student.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        in_features = student.backbone.classifier[3].in_features
        student.backbone.classifier[3] = nn.Linear(in_features, 180)
        student.to(device)
        
    student.train()
    
    # 3. 準備訓練
    dataset = SimpleDataset(DATA_DIR)
    if len(dataset) == 0:
        print("❌ 錯誤：找不到訓練圖片！請檢查路徑。")
        return
    print(f"📊 訓練資料: {len(dataset)} 張 (高品質官方認證圖)")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam(student.parameters(), lr=LR)
    
    # 4. 開始訓練
    print("🔥 開始訓練...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for imgs in dataloader:
            imgs = imgs.to(device)
            
            with torch.no_grad():
                tp, ty = teacher(imgs)
            
            sp, sy = student(imgs)
            
            # 使用 MSE 強力矯正
            loss = nn.MSELoss()(sp, tp) + nn.MSELoss()(sy, ty)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f}")

    # 5. 存檔
    torch.save(student.state_dict(), STUDENT_SAVE_PATH)
    print(f"\n✅✅✅ 最終模型已儲存: {STUDENT_SAVE_PATH}")
    print("👉 下一步：請使用 demo_final_stable.py 載入這個新模型進行測試！")

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
# 使用量化版模型庫
from torchvision.models.quantization import mobilenet_v3_large
import os
import glob
from PIL import Image
from tqdm import tqdm
import numpy as np

# ================= ⚙️ QAT 設定 =================
DATA_DIR = 'data/selfmade_combined'
TEACHER_PATH = 'models/L2CSNet_gaze360.pkl' # 依然需要老師來指導
PRETRAINED_STUDENT = 'models/student_mobilenet_3people_9k.pth'
QAT_SAVE_PATH = 'models/student_mobilenet_qat.pth'

DEVICE = torch.device('cuda') # QAT 訓練可以用 GPU 加速
BATCH_SIZE = 64
LR = 1e-5             # 🔥 非常小的學習率，只是微調
EPOCHS = 5            # 不用多，幾輪就夠適應了
# ===============================================

# 1. 定義 QAT 模型結構
class L2CS_MobileNetV3_QAT(nn.Module):
    def __init__(self, num_bins=90):
        super(L2CS_MobileNetV3_QAT, self).__init__()
        # quantize=False: 先載入 FP32 權重
        self.backbone = mobilenet_v3_large(weights=None, quantize=False)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_bins * 2)
        
        # QAT 需要 Stub
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.backbone(x)
        x = self.dequant(x)
        return x[:, :90], x[:, 90:]

# 2. 老師模型 (固定)
class L2CS_ResNet50(nn.Module):
    def __init__(self, num_bins=90):
        super(L2CS_ResNet50, self).__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(2048, num_bins * 2)
    def forward(self, x):
        x = self.model(x)
        return x[:, :90], x[:, 90:]

# 3. 資料集 (簡單版)
class GazeDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = glob.glob(os.path.join(root_dir, "*.jpg"))
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def main():
    print("🚀 啟動 QAT (量化感知訓練)...")

    # A. 準備資料
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = GazeDataset(DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # B. 準備老師 (Freeze)
    print("🎓 載入老師模型...")
    teacher = L2CS_ResNet50().to(DEVICE)
    ckpt = torch.load(TEACHER_PATH, map_location=DEVICE)
    # (權重載入邏輯省略，假設你之前代碼有，這邊簡化)
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

    # C. 準備學生 (QAT Setup)
    print("👶 載入並準備 QAT 學生模型...")
    student = L2CS_MobileNetV3_QAT().to(DEVICE)
    
    # C1. 載入之前的訓練成果 (FP32)
    # 注意：因為結構略有不同 (Quantizable Backbone)，使用 strict=False
    saved_state = torch.load(PRETRAINED_STUDENT, map_location=DEVICE)
    student.load_state_dict(saved_state, strict=False)
    
    # C2. 設定 QAT 配置
    student.train()
    # 使用與 Static Quantize 相同的後端
    student.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
    
    # C3. 融合算子 (Fusion) - 這是關鍵！
    # 這會把 Conv+BN+ReLU 合併成一個層，讓量化更準
    student.backbone.fuse_model(is_qat=True)
    
    # C4. 準備 (Prepare QAT) - 插入 FakeQuant 節點
    torch.ao.quantization.prepare_qat(student, inplace=True)
    
    # D. 開始微調訓練
    optimizer = optim.Adam(student.parameters(), lr=LR)
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    T = 5.0 # Temperature

    student = student.to(DEVICE) # 確保在 GPU 上訓練

    print(f"🔥 開始 {EPOCHS} Epochs 的微調...")
    for epoch in range(EPOCHS):
        total_loss = 0
        student.train() # QAT 必須在 train 模式下更新 FakeQuant 參數
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images in loop:
            images = images.to(DEVICE)
            
            # 老師預測
            with torch.no_grad():
                t_pitch, t_yaw = teacher(images)
            
            # 學生預測 (帶有量化噪聲)
            s_pitch, s_yaw = student(images)
            
            # 蒸餾 Loss
            loss_pitch = kl_loss(torch.log_softmax(s_pitch/T, dim=1), torch.softmax(t_pitch/T, dim=1)) * (T**2)
            loss_yaw = kl_loss(torch.log_softmax(s_yaw/T, dim=1), torch.softmax(t_yaw/T, dim=1)) * (T**2)
            loss = loss_pitch + loss_yaw
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(dataloader):.4f}")

    # E. 轉換為真正的 INT8 模型 (Convert)
    print("🔄 正在轉換為 INT8 模型 (CPU)...")
    student.eval()
    student.to('cpu') # 轉換必須在 CPU
    torch.ao.quantization.convert(student, inplace=True)
    
    torch.save(student.state_dict(), QAT_SAVE_PATH)
    print(f"✅ QAT 模型已儲存: {QAT_SAVE_PATH}")
    print(f"📊 模型大小: {os.path.getsize(QAT_SAVE_PATH)/1024**2:.2f} MB")

if __name__ == "__main__":
    main()
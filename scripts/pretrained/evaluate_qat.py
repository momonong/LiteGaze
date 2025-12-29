import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.quantization import mobilenet_v3_large
from torchvision.models import resnet50
import os
import glob
from PIL import Image
from tqdm import tqdm
import numpy as np

# ================= ⚙️ 設定 =================
DATA_DIR = 'data/selfmade_combined'
QAT_MODEL_PATH = 'models/student_mobilenet_qat.pth'
TEACHER_PATH = 'models/L2CSNet_gaze360.pkl'
DEVICE = torch.device('cpu') 
BATCH_SIZE = 32
# ==========================================

# 🔥 修正 1: 全局設置後端引擎為 onednn
torch.backends.quantized.engine = 'onednn'

# Dataset 定義在最外層
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, files, transform):
        self.files = files
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.files[idx]).convert('RGB')
            return self.transform(img)
        except:
            return torch.zeros(3, 224, 224)

# 推論模型結構 (移除外層 QuantStub)
class L2CS_MobileNetV3_QAT_Inference(nn.Module):
    def __init__(self, num_bins=90):
        super().__init__()
        self.backbone = mobilenet_v3_large(weights=None, quantize=False)
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

def load_qat_model(path):
    print("🏗️ 重建 QAT 模型結構 (推論模式)...")
    model = L2CS_MobileNetV3_QAT_Inference()
    
    # 🔥 修正 2: 使用 onednn 的量化配置
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('onednn')
    
    # 融合
    model.backbone.fuse_model(is_qat=True)
    
    # 準備與轉換
    torch.ao.quantization.prepare_qat(model, inplace=True)
    torch.ao.quantization.convert(model, inplace=True)
    
    print(f"📥 載入權重: {path}")
    state_dict = torch.load(path, map_location='cpu')
    
    try:
        # strict=False 忽略多餘的外層 quant 權重
        model.load_state_dict(state_dict, strict=False)
        print("✅ 載入成功！")
    except Exception as e:
        print(f"❌ 載入失敗: {e}")
        return None
        
    return model

# Teacher 定義
class L2CS_ResNet50(nn.Module):
    def __init__(self, num_bins=90):
        super().__init__()
        self.model = resnet50(weights=None)
        self.model.fc = nn.Linear(2048, num_bins * 2)
    def forward(self, x):
        x = self.model(x)
        return x[:, :90], x[:, 90:]

def main():
    # A. 載入模型
    model = load_qat_model(QAT_MODEL_PATH)
    if model is None: return
    model.to(DEVICE)
    model.eval()

    # B. 載入 Teacher
    print("🎓 載入 Teacher (ResNet50)...")
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

    # C. 評估
    print(f"🚀 開始評估 9000 張圖片...")
    files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    ds = SimpleDataset(files, transform)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4) 
    
    pitch_diffs = []
    yaw_diffs = []
    
    with torch.no_grad():
        for imgs in tqdm(dl):
            imgs = imgs.to(DEVICE)
            
            # Teacher
            tp, ty = teacher(imgs)
            t_pitch = compute_gaze(tp)
            t_yaw = compute_gaze(ty)
            
            # Student (QAT INT8)
            sp, sy = model(imgs)
            s_pitch = compute_gaze(sp)
            s_yaw = compute_gaze(sy)
            
            pitch_diffs.extend(torch.abs(t_pitch - s_pitch).numpy())
            yaw_diffs.extend(torch.abs(t_yaw - s_yaw).numpy())

    mae_pitch = np.mean(pitch_diffs)
    mae_yaw = np.mean(yaw_diffs)
    
    print("\n" + "="*40)
    print(f"📊 QAT 模型 (INT8) 最終驗收報告")
    print("="*40)
    print(f"Pitch MAE : {mae_pitch:.4f}°")
    print(f"Yaw   MAE : {mae_yaw:.4f}°")
    print(f"Avg Error : {(mae_pitch + mae_yaw)/2:.4f}°")
    print("="*40)

if __name__ == "__main__":
    main()
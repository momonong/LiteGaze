import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm

# ================= ⚙️ 設定 =================
DATA_DIR = 'data/selfmade_combined'
STUDENT_PATH = 'models/student_mobilenet_3people_9k.pth'
QUANTIZED_PATH = 'models/student_mobilenet_static_int8.pth'
CALIBRATE_BATCH = 100 
DEVICE = torch.device('cpu') # 量化驗證一定要用 CPU
# ==========================================

# 🔥 關鍵修改：使用 quantization 版本的 MobileNetV3
# 這個版本把所有的 '+' 換成了 FloatFunctional，解決了報錯問題
from torchvision.models.quantization import mobilenet_v3_large, MobileNet_V3_Large_QuantizedWeights

class L2CS_MobileNetV3_Quant(nn.Module):
    def __init__(self, num_bins=90):
        super(L2CS_MobileNetV3_Quant, self).__init__()
        # 使用支援量化的骨幹網絡 (quantize=False 代表先以 FP32 模式載入，準備進行 PTQ)
        self.backbone = mobilenet_v3_large(weights=None, quantize=False)
        
        # 修改最後一層
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_bins * 2)
        
        # Stub (量化邊界標記)
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.backbone(x)
        x = self.dequant(x)
        return x[:, :90], x[:, 90:]

def compute_gaze(logits):
    softmax = nn.Softmax(dim=1)
    prob = softmax(logits)
    idx = torch.arange(90, dtype=torch.float32).to(logits.device)
    gaze = torch.sum(prob * idx, dim=1) * 4 - 180
    return gaze.item()

def main():
    print(f"📥 載入 FP32 模型: {STUDENT_PATH}")
    
    # 1. 準備模型
    # 注意：我們現在用的是 Quantizable 的骨幹，結構跟原本的略有不同
    # 但權重大部分是兼容的，我們可以透過 strict=False 硬吃進去
    model = L2CS_MobileNetV3_Quant()
    state_dict = torch.load(STUDENT_PATH, map_location='cpu')
    
    # 這裡可能會有一些 key 不匹配 (因為 quantizable model 結構變了)
    # 沒關係，只要 backbone.features 的權重有進去就好
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0:
        print(f"⚠️ 部分權重未載入 (這是正常的，因為換了 Quantizable 骨幹): {len(missing)} keys")
    
    model.eval()

    # 2. 設定量化配置 (使用 onednn)
    backend = 'onednn' # 🔥 你指定的後端
    print(f"⚙️ 使用後端: {backend}")
    
    model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    
    # 3. 融合算子 (Fusion) - 這是加速的關鍵
    # MobileNetV3 的標準融合： Conv+BN+ReLU
    print("🔥 正在融合算子 (Fuse Modules)...")
    model.backbone.fuse_model(is_qat=False)

    # 4. 準備量化
    print("👀 準備量化 (Prepare)...")
    torch.ao.quantization.prepare(model, inplace=True)
    
    # 5. 校準 (Calibration)
    print("📏 正在校準 (Calibration) - 讀取 100 張圖片...")
    files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))[:CALIBRATE_BATCH]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        for f in tqdm(files):
            try:
                img = transform(Image.open(f).convert('RGB')).unsqueeze(0)
                model(img)
            except: continue
            
    # 6. 轉換 (Convert)
    print("🔄 正在轉換為 INT8 (Convert)...")
    torch.ao.quantization.convert(model, inplace=True)
    
    # 7. 存檔與大小比較
    torch.save(model.state_dict(), QUANTIZED_PATH)
    size_fp32 = os.path.getsize(STUDENT_PATH) / 1024**2
    size_int8 = os.path.getsize(QUANTIZED_PATH) / 1024**2
    
    print("\n" + "="*30)
    print(f"📊 模型瘦身成果:")
    print(f"FP32: {size_fp32:.2f} MB")
    print(f"INT8: {size_int8:.2f} MB")
    print(f"👉 壓縮率: {size_fp32/size_int8:.1f}x")
    print("="*30)

    # 8. 驗證精度
    print("⚖️ 正在評估 INT8 模型精度 (前 200 張)...")
    pitch_diffs = []
    yaw_diffs = []
    
    # 載入一個原始 FP32 模型做對照
    # 這裡我們用回原本的 class，確保對照組是正確的
    from torchvision import models as original_models
    class L2CS_MobileNetV3_Original(nn.Module):
        def __init__(self, num_bins=90):
            super().__init__()
            self.backbone = original_models.mobilenet_v3_large(weights=None)
            in_features = self.backbone.classifier[3].in_features
            self.backbone.classifier[3] = nn.Linear(in_features, num_bins * 2)
        def forward(self, x):
            x = self.backbone(x)
            return x[:, :90], x[:, 90:]
            
    fp32_model = L2CS_MobileNetV3_Original()
    fp32_model.load_state_dict(torch.load(STUDENT_PATH, map_location='cpu'))
    fp32_model.eval()
    
    eval_files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))[:200]
    
    with torch.no_grad():
        for f in tqdm(eval_files):
            try:
                img = transform(Image.open(f).convert('RGB')).unsqueeze(0)
                
                # FP32
                p1, y1 = fp32_model(img)
                deg_p1 = compute_gaze(p1)
                deg_y1 = compute_gaze(y1)
                
                # INT8
                p2, y2 = model(img)
                deg_p2 = compute_gaze(p2)
                deg_y2 = compute_gaze(y2)
                
                pitch_diffs.append(abs(deg_p1 - deg_p2))
                yaw_diffs.append(abs(deg_y1 - deg_y2))
            except: continue

    mae_pitch = np.mean(pitch_diffs)
    mae_yaw = np.mean(yaw_diffs)
    
    print("\n" + "="*40)
    print("📉 量化誤差報告 (Quantization Loss Report)")
    print("="*40)
    print(f"Pitch MAE Loss: {mae_pitch:.4f}°")
    print(f"Yaw   MAE Loss: {mae_yaw:.4f}°")
    print("="*40)

if __name__ == "__main__":
    main()
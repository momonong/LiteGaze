import torch
import torch.nn as nn
from torchvision import models
import os

# ================= ⚙️ 設定 =================
STUDENT_PATH = 'models/student_mobilenet_3people_9k.pth'
QUANTIZED_PATH = 'models/student_mobilenet_quantized.pth'
# ==========================================

class L2CS_MobileNetV3(nn.Module):
    def __init__(self, num_bins=90):
        super(L2CS_MobileNetV3, self).__init__()
        self.backbone = models.mobilenet_v3_large(weights=None)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_bins * 2)
    def forward(self, x):
        x = self.backbone(x)
        return x[:, :90], x[:, 90:]

def main():
    print(f"📥 載入 FP32 模型: {STUDENT_PATH}")
    model = L2CS_MobileNetV3()
    model.load_state_dict(torch.load(STUDENT_PATH, map_location='cpu')) # 量化要在 CPU 上做
    model.eval()

    print("🔄 正在進行動態量化 (Dynamic Quantization)...")
    
    # 針對 Linear 和 LSTM/RNN 層進行 int8 量化 (MobileNetV3 主要是 Conv，但在 CPU 上 PyTorch 也能優化)
    # 注意：PyTorch 的 Dynamic Quantization 主要對 Linear 有效。
    # 如果要對 Conv 量化，通常需要 Static Quantization (QAT)，比較複雜。
    # 這裡我們先做簡單版，看看能壓多少。
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear},  # 指定量化 Linear 層
        dtype=torch.qint8
    )
    
    print(f"💾 儲存量化模型: {QUANTIZED_PATH}")
    torch.save(quantized_model.state_dict(), QUANTIZED_PATH)
    
    # 比較大小
    size_fp32 = os.path.getsize(STUDENT_PATH) / 1024**2
    size_int8 = os.path.getsize(QUANTIZED_PATH) / 1024**2
    
    print(f"\n📊 大小比較:")
    print(f"FP32 Model: {size_fp32:.2f} MB")
    print(f"INT8 Model: {size_int8:.2f} MB")
    print(f"👉 壓縮率: {size_fp32/size_int8:.1f}x")

if __name__ == "__main__":
    main()
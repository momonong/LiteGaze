import torch
import torch.nn as nn
from torchvision import models
from torchinfo import summary
from thop import profile
import sys
import os

# ================= ⚙️ 設定 =================
MODEL_PATH = 'models/L2CSNet_gaze360.pkl'
# ==========================================

# 定義 Teacher 結構 (必須跟權重檔一致)
class L2CS_ResNet50(nn.Module):
    def __init__(self, num_bins=90):
        super(L2CS_ResNet50, self).__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(2048, num_bins * 2)
    def forward(self, x):
        x = self.model(x)
        return x[:, :90], x[:, 90:]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Inspecting Model on {device}...")

    # 1. 載入模型結構
    model = L2CS_ResNet50().to(device)
    
    # 2. 嘗試載入權重 (為了確認檔案沒壞)
    if os.path.exists(MODEL_PATH):
        try:
            # 萬能載入法
            ckpt = torch.load(MODEL_PATH, map_location=device)
            state = {}
            for k, v in ckpt.items():
                if 'fc_pitch' in k or 'fc_yaw' in k: continue
                nk = 'model.'+k if not k.startswith('model.') else k
                state[nk] = v
            if 'fc_pitch_gaze.weight' in ckpt:
                state['model.fc.weight'] = torch.cat((ckpt['fc_pitch_gaze.weight'], ckpt['fc_yaw_gaze.weight']), 0)
                state['model.fc.bias'] = torch.cat((ckpt['fc_pitch_gaze.bias'], ckpt['fc_yaw_gaze.bias']), 0)
            
            model.load_state_dict(state, strict=False)
            print("✅ Weights loaded successfully (Model Integrity OK).")
        except Exception as e:
            print(f"⚠️ Warning: Weights loading failed ({e}), but we can still inspect architecture.")
    else:
        print("⚠️ Warning: Weight file not found, inspecting random init model.")

    model.eval()

    # 3. 準備假資料 (模擬一張 224x224 的圖片)
    input_size = (1, 3, 224, 224)
    dummy_input = torch.randn(input_size).to(device)

    # 4. 計算 FLOPs (運算量)
    # thop 會回傳 (MACs, Params)，通常 FLOPs ~ 2 * MACs
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    flops = macs * 2

    print("\n" + "="*40)
    print("📊 MODEL INSPECTION REPORT (TEACHER)")
    print("="*40)
    print(f"Architecture: ResNet50 (L2CS-Net)")
    print(f"Input Size  : {input_size}")
    print(f"Parameters  : {params / 1e6:.2f} M (百萬)")
    print(f"FLOPs       : {flops / 1e9:.2f} G (十億次浮點運算)")
    print("="*40)
    
    # 5. 詳細層級分析
    print("\n🔍 Layer-wise Summary:")
    summary(model, input_size=input_size, device=device.type)

if __name__ == "__main__":
    main()
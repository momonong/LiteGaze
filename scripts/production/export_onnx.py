import torch
import torch.nn as nn
from torchvision import models
import onnx
import os
import shutil

# ================= ⚙️ 設定 =================
STUDENT_PATH = 'models/student_mobilenet_3people_9k.pth'
ONNX_SAVE_PATH = 'models/litegaze_student_fp32.onnx'
DEVICE = torch.device('cpu') 
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
    print(f"📥 載入 PyTorch 模型: {STUDENT_PATH}")
    model = L2CS_MobileNetV3()
    try:
        model.load_state_dict(torch.load(STUDENT_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"❌ 載入失敗: {e}")
        return
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, device=DEVICE)

    print(f"🔄 嘗試導出 ONNX (強制 Legacy 模式)...")
    
    # 刪除舊檔案以免誤判
    if os.path.exists(ONNX_SAVE_PATH):
        os.remove(ONNX_SAVE_PATH)

    try:
        # 🔥 嘗試方案 A: 顯式傳入 dynamo=False
        print("👉 方案 A: 嘗試傳入 dynamo=False 參數...")
        torch.onnx.export(
            model,
            dummy_input,
            ONNX_SAVE_PATH,
            export_params=True,
            opset_version=11,          # Legacy 模式最喜歡 Opset 11
            do_constant_folding=True,
            input_names=['input'],
            output_names=['pitch_logits', 'yaw_logits'],
            dynamic_axes={'input': {0: 'batch_size'},
                          'pitch_logits': {0: 'batch_size'},
                          'yaw_logits': {0: 'batch_size'}},
            dynamo=False  # 🚫 強制關閉新引擎
        )
    except TypeError:
        # 如果 pytorch 版本舊到不認識 dynamo 參數，那它原本就是 legacy，直接跑
        print("⚠️ 方案 A 失敗 (不支援 dynamo 參數)，轉為方案 B (預設導出)...")
        try:
            torch.onnx.export(
                model,
                dummy_input,
                ONNX_SAVE_PATH,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['pitch_logits', 'yaw_logits'],
                dynamic_axes={'input': {0: 'batch_size'},
                              'pitch_logits': {0: 'batch_size'},
                              'yaw_logits': {0: 'batch_size'}}
            )
        except Exception as e:
            print(f"❌ 方案 B 也失敗: {e}")
            return
    except Exception as e:
        print(f"❌ 導出發生未預期錯誤: {e}")
        return

    # 驗收
    if os.path.exists(ONNX_SAVE_PATH):
        size_mb = os.path.getsize(ONNX_SAVE_PATH) / 1024**2
        print(f"✅ FP32 ONNX 已儲存: {ONNX_SAVE_PATH}")
        print(f"📊 檔案大小: {size_mb:.2f} MB")
        
        if size_mb < 5.0:
            print("❌ 警告：檔案依然是空的！請檢查你的 PyTorch 安裝是否損壞。")
        else:
            print("🎉 成功！這才是包含權重的完整模型。")
            print("👉 下一步：python scripts/pretrained/quantize_onnx.py")
    else:
        print("❌ 導出後找不到檔案！")

if __name__ == "__main__":
    main()
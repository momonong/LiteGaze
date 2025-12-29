import torch
import torch.nn as nn
from torchvision import transforms, models
import os
import glob
from PIL import Image
import numpy as np

# ================= 設定 =================
DATA_DIR = 'data/official_calibration' # 請確認跟採集時的路徑一致
TEACHER_PATH = 'models/L2CSNet_gaze360.pkl'
# =======================================

class L2CS_ResNet50(nn.Module):
    def __init__(self, num_bins=90):
        super(L2CS_ResNet50, self).__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(2048, num_bins * 2)
    def forward(self, x):
        x = self.model(x)
        return x[:, :90], x[:, 90:]

def compute_gaze(logits):
    softmax = nn.Softmax(dim=1)
    prob = softmax(logits)
    idx = torch.arange(90, dtype=torch.float32).to(logits.device)
    gaze = torch.sum(prob * idx, dim=1) * 4 - 180
    return gaze.item()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Loading Teacher on {device}...")
    
    # 載入老師模型
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
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 準備統計
    actions = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "CENTER"}
    stats = {k: {'pitch': [], 'yaw': []} for k in actions.keys()}
    
    files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
    print(f"📂 Found {len(files)} images. Analyzing...")
    
    for f in files:
        filename = os.path.basename(f)
        try:
            # 檔名格式 act0_0001.jpg
            act_idx = int(filename.split('_')[0].replace('act', ''))
        except: continue
        
        img = Image.open(f).convert('RGB')
        inp = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            tp, ty = teacher(inp)
            p_val = compute_gaze(tp)
            y_val = compute_gaze(ty)
            
            stats[act_idx]['pitch'].append(p_val)
            stats[act_idx]['yaw'].append(y_val)
            
    print("\n" + "="*50)
    print(f"{'ACTION':<10} | {'AVG PITCH':<15} | {'AVG YAW':<15} | {'STATUS'}")
    print("="*50)
    
    # 顯示結果
    for k, name in actions.items():
        if len(stats[k]['pitch']) == 0:
            print(f"{name:<10} | {'No Data':<15} | {'No Data':<15}")
            continue
            
        avg_p = np.mean(stats[k]['pitch'])
        avg_y = np.mean(stats[k]['yaw'])
        
        # 簡單判定狀態
        status = "✅ OK"
        if name == "UP" and avg_p > -5: status = "⚠️ Weak (Not high enough)" # 假設負是上
        if name == "DOWN" and avg_p < 5: status = "⚠️ Weak (Not low enough)"
        
        print(f"{name:<10} | {avg_p:>10.2f}°    | {avg_y:>10.2f}°    |")

    print("="*50)
    print("💡 判斷標準：")
    print("1. UP 的 Pitch 應該要是 負數 (例如 -15 ~ -30)")
    print("2. DOWN 的 Pitch 應該要是 正數 (例如 +10 ~ +30)")
    print("   (或者反過來，重點是兩個數值要差很遠！)")
    print("3. LEFT 和 RIGHT 的 Yaw 也要差很遠。")

if __name__ == "__main__":
    main()
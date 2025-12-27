import numpy as np
import glob
import os
from tqdm import tqdm

# === ⚙️ 設定區 ===
SOURCE_ROOT = "/mnt/d/projects/LiteGaze/data/processed"
OUTPUT_FILE = "data/mpiigaze_224.npz"

# ✅ 根據您的檢查結果設定正確的 Keys
KEY_IMAGES = 'teacher'
KEY_LABELS = 'label'

def aggregate():
    print(f"🔍 正在搜尋 {SOURCE_ROOT} 下的所有 .npz 檔案...")
    file_list = glob.glob(os.path.join(SOURCE_ROOT, "**", "*.npz"), recursive=True)
    
    if not file_list:
        print("❌ 找不到檔案！")
        return

    print(f"📦 找到了 {len(file_list)} 個檔案，使用 Key: ['{KEY_IMAGES}', '{KEY_LABELS}']")

    all_images = []
    all_labels = []

    for fpath in tqdm(file_list):
        try:
            with np.load(fpath) as data:
                if KEY_IMAGES not in data or KEY_LABELS not in data:
                    continue
                
                imgs = data[KEY_IMAGES]
                lbls = data[KEY_LABELS]
                
                # 🛠️ 確保形狀正確：如果是 (N, 60, 60) 少了 channel，補上它
                if len(imgs.shape) == 3: 
                    imgs = np.expand_dims(imgs, axis=-1)
                
                all_images.append(imgs.astype(np.float32))
                all_labels.append(lbls.astype(np.float32))
                
        except Exception as e:
            print(f"❌ 讀取 {fpath} 失敗: {e}")

    if not all_images:
        print("❌ 錯誤：沒有成功載入任何資料，請檢查 Key 是否正確。")
        return

    print("🔄 正在堆疊資料 (這可能需要幾秒鐘)...")
    full_images = np.concatenate(all_images, axis=0)
    full_labels = np.concatenate(all_labels, axis=0)

    print(f"✅ 合併完成！")
    print(f"   總圖片數: {full_images.shape}")
    print(f"   總標籤數: {full_labels.shape}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    np.savez(OUTPUT_FILE, images=full_images, labels=full_labels)
    print(f"💾 已儲存至 {OUTPUT_FILE}")
    print("🚀 準備就緒！請執行: python scripts/v2/train.py")

if __name__ == "__main__":
    aggregate()
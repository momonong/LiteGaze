import os
import shutil
from tqdm import tqdm

# ================= ⚙️ 設定區 =================
# 你的三個來源資料夾名稱 (根據你的 ls 結果)
SOURCE_DIRS = {
    'morris': 'data/selfmade_morris',
    'dunnie': 'data/selfmade_dunnie',
    'sisi':   'data/selfmade_sisi'
}

# 目標資料夾
TARGET_DIR = 'data/selfmade_combined'
# ============================================

def main():
    # 1. 建立目標資料夾
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"📂 建立目標資料夾: {TARGET_DIR}")
    else:
        print(f"⚠️ 目標資料夾已存在: {TARGET_DIR} (新檔案將會加入其中)")

    total_files = 0
    
    # 2. 開始搬運
    print(f"🚀 開始合併數據...")
    
    for prefix, src_path in SOURCE_DIRS.items():
        if not os.path.exists(src_path):
            print(f"❌ 找不到資料夾: {src_path}，跳過。")
            continue
            
        files = [f for f in os.listdir(src_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"📥 正在處理 {prefix} ({len(files)} 張)...")
        
        for filename in tqdm(files):
            # 原始路徑
            src_file = os.path.join(src_path, filename)
            
            # 新檔名：加上前綴，避免衝突 (例如 morris_img_00001.jpg)
            new_filename = f"{prefix}_{filename}"
            dst_file = os.path.join(TARGET_DIR, new_filename)
            
            # 複製檔案
            shutil.copy2(src_file, dst_file)
            total_files += 1

    print("\n" + "="*40)
    print(f"🎉 合併完成！")
    print(f"📊 總共圖片數: {total_files} 張")
    print(f"📂 儲存位置: {TARGET_DIR}")
    print("="*40)
    print("👉 下一步：請修改 train_perfect_distill.py 的 DATA_DIR 指向這裡！")

if __name__ == "__main__":
    main()
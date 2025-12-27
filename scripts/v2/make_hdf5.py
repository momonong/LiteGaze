import numpy as np
import glob
import os
import h5py
import gc  # 垃圾回收機制
from tqdm import tqdm

# === ⚙️ 設定區 ===
SOURCE_ROOT = "data/processed"
OUTPUT_FILE = "data/teacher_224.h5"

# Teacher 專用 Keys
KEY_IMAGES = 'teacher'
KEY_LABELS = 'label'

def make_h5():
    # 1. 搜尋檔案
    file_list = glob.glob(os.path.join(SOURCE_ROOT, "**", "*.npz"), recursive=True)
    if not file_list:
        print("❌ 找不到檔案！請確認路徑。")
        return
    print(f"📦 找到了 {len(file_list)} 個檔案，準備製作 HDF5 (高效能模式)...")

    # 2. 讀取第一個檔案來獲取形狀 (但不讀入全部數據)
    with np.load(file_list[0]) as first_data:
        sample_img = first_data[KEY_IMAGES]
        # 確保形狀是 (H, W, 3)
        if len(sample_img.shape) == 3: 
            img_dim = (sample_img.shape[1], sample_img.shape[2], 1)
        elif len(sample_img.shape) == 4:
            img_dim = sample_img.shape[1:] # (224, 224, 3)
        
        lbl_dim = first_data[KEY_LABELS].shape[1:] # (2,)

    print(f"ℹ️ 圖片尺寸: {img_dim}, 儲存格式: uint8 (節省空間)")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 3. 開啟 HDF5 檔案
    with h5py.File(OUTPUT_FILE, 'w') as hf:
        # 🔥 關鍵修改 1: dtype='uint8' (整數)，這比 float32 小 4 倍
        dset_img = hf.create_dataset('images', shape=(0, *img_dim), maxshape=(None, *img_dim), dtype='uint8', chunks=True)
        # 標籤通常還是要 float32
        dset_lbl = hf.create_dataset('labels', shape=(0, *lbl_dim), maxshape=(None, *lbl_dim), dtype='float32', chunks=True)

        total_count = 0
        
        # 4. 逐檔寫入
        pbar = tqdm(file_list)
        for fpath in pbar:
            try:
                with np.load(fpath) as data:
                    if KEY_IMAGES not in data or KEY_LABELS not in data:
                        continue
                    
                    # 🔥 關鍵修改 2: 保持原始格式 (通常是 uint8)，不要 .astype('float32')
                    # 這樣就不會發生記憶體膨脹
                    imgs = data[KEY_IMAGES] 
                    lbls = data[KEY_LABELS].astype(np.float32)

                    # 補齊 Channel 維度如果需要
                    if len(imgs.shape) == 3: imgs = np.expand_dims(imgs, axis=-1)
                    
                    n_current = imgs.shape[0]
                    
                    # 擴充 HDF5
                    dset_img.resize(total_count + n_current, axis=0)
                    dset_lbl.resize(total_count + n_current, axis=0)
                    
                    # 寫入
                    dset_img[total_count : total_count + n_current] = imgs
                    dset_lbl[total_count : total_count + n_current] = lbls
                    
                    total_count += n_current
                    
                    # 更新進度條資訊
                    pbar.set_description(f"Count: {total_count}")

                # 🔥 關鍵修改 3: 強制釋放記憶體
                del imgs, lbls, data
                gc.collect() 
                    
            except Exception as e:
                print(f"\n⚠️ 讀取 {fpath} 失敗: {e}")
                # 遇到壞檔不要斷掉，繼續下一個

    print(f"\n✅ HDF5 製作完成！")
    print(f"   位置: {OUTPUT_FILE}")
    print(f"   總張數: {total_count}")
    print(f"   注意: 訓練時請記得將 uint8 除以 255.0 轉回 float！")

if __name__ == "__main__":
    make_h5()
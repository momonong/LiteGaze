import h5py
import cv2
import numpy as np
import os

# 修改成你的檔案路徑
H5_PATH = r'D:\projects\LiteGaze\data\teacher_224.h5'

def main():
    if not os.path.exists(H5_PATH):
        print(f"❌ 找不到檔案: {H5_PATH}")
        return

    print(f"📂 Opening {H5_PATH}...")
    
    with h5py.File(H5_PATH, 'r') as f:
        # 1. 列出所有的 Keys (看看裡面叫什麼名字，通常是 'images', 'data' 之類的)
        print("🔑 Keys inside H5:", list(f.keys()))
        
        # 假設 key 叫做 'images' 或 'data' (根據你之前的習慣)
        # 我們嘗試抓第一個 key
        key = list(f.keys())[0]
        data = f[key]
        
        print(f"📊 Shape of dataset '{key}': {data.shape}")
        
        # 2. 抓第一張圖出來看看
        # 注意：H5 裡的圖片格式可能是 (N, H, W, 3) 或是 (N, 3, H, W)
        img_raw = data[0]
        
        # 如果是 (3, 224, 224) 這種 PyTorch 格式，要轉成 (224, 224, 3)
        if img_raw.shape[0] == 3:
            img_raw = np.transpose(img_raw, (1, 2, 0))
            
        # 如果數值是 0~1 (Float)，要轉回 0~255 (Int)
        if img_raw.max() <= 1.0:
            img_raw = (img_raw * 255).astype(np.uint8)
        else:
            img_raw = img_raw.astype(np.uint8)
            
        # 3. 顯示圖片 (轉回 BGR 讓 OpenCV 顯示正確顏色)
        # 假設存的時候是 RGB
        img_show = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("Check H5 Content", img_show)
        print("👀 按任意鍵關閉視窗...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
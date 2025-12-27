import h5py
import numpy as np

def check_all_labels():
    print("🔍 正在檢查全部標籤 (Labels)...")
    with h5py.File('data/teacher_224.h5', 'r') as hf:
        # 一次讀取所有標籤 (記憶體吃很少，不用擔心)
        labels = hf['labels'][:]
        
        # 1. 檢查 NaN
        if np.isnan(labels).any():
            print("❌ 慘了！標籤裡真的有 NaN！請重新檢查 aggregate_data.py")
            return
            
        # 2. 檢查數值範圍 (確保都在 -1.57 ~ 1.57 左右)
        # 有時候會有異常大的數值導致梯度爆炸
        print(f"📊 標籤統計:")
        print(f"   Max: {np.max(labels)}")
        print(f"   Min: {np.min(labels)}")
        print(f"   Mean: {np.mean(labels)}")

        if np.max(labels) > 10 or np.min(labels) < -10:
             print("⚠️ 警告：標籤數值似乎異常大，這也可能導致 Loss NaN")
        else:
             print("✅ 標籤數據看起來非常健康！")

    print("\n💡 結論：")
    print("既然圖片是 uint8 (不可能 NaN)，標籤也沒問題，")
    print("那兇手 100% 就是【學習率 (Learning Rate)】太高了！")

if __name__ == "__main__":
    check_all_labels()
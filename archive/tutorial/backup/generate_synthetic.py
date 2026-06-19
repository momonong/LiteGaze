import pandas as pd
import numpy as np
import os

def generate_synthetic_gaze(output_path="tutorial/data/raw_backup.csv"):
    gaze_points = []
    
    # 初始系統性偏移 (Drift)：隨著閱讀時間累積，不會歸零
    drift_x = 0
    drift_y = 0
    
    # 對應你圖表上的三行文字的大致範圍與 Y 座標
    lines = [
        {"start_x": 150, "end_x": 1800, "base_y": 450}, # 第一行: The phenomenon...
        {"start_x": 250, "end_x": 1600, "base_y": 550}, # 第二行: particles can be...
        {"start_x": 400, "end_x": 1400, "base_y": 650}  # 第三行: challenging our...
    ]
    
    for line_idx, line in enumerate(lines):
        current_x = line["start_x"]
        
        while current_x < line["end_x"]:
            # 1. 模擬「注視 (Fixation)」：眼球在同一個單字群附近震顫停留
            fixation_duration = np.random.randint(4, 10) # 停留 4~10 個 frame
            
            for _ in range(fixation_duration):
                # Webcam 特有的高頻抖動雜訊
                noise_x = np.random.normal(0, 25)
                noise_y = np.random.normal(0, 30)
                
                # 模擬頭部逐漸下垂或姿勢跑掉的「累積偏移」
                drift_x += np.random.normal(0.2, 0.8) 
                drift_y += np.random.normal(0.5, 1.5) 
                
                # 模擬極端雜訊 (Webcam 瞬間抓錯位置的 Outlier)
                if np.random.rand() < 0.02:
                    noise_y += np.random.choice([-80, 80])
                
                gaze_points.append({
                    'x_px': current_x + noise_x + drift_x,
                    'y_px': line["base_y"] + noise_y + drift_y,
                    'timestamp': len(gaze_points) * 33 # 模擬 30 FPS
                })
            
            # 2. 模擬「掃視 (Saccade)」：讀完這個字，快速跳到下一個字群
            current_x += np.random.randint(60, 150)
            
            # 3. 模擬「回視 (Regression)」：大約 15% 機率，看不懂往回跳
            if np.random.rand() < 0.15:
                current_x -= np.random.randint(40, 100)
        
        # 4. 模擬「換行與眨眼」：換行時通常伴隨一個巨大的跳躍與幾次抓不到資料 (NaN)
        if line_idx < len(lines) - 1:
            for _ in range(5):
                gaze_points.append({'x_px': np.nan, 'y_px': np.nan, 'timestamp': len(gaze_points)*33})
                
    df = pd.DataFrame(gaze_points)
    
    # 確保資料夾存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ 真實感黃金備用資料已生成: {output_path} (包含 3 行文字與閱讀動態雜訊)")

if __name__ == "__main__":
    generate_synthetic_gaze()
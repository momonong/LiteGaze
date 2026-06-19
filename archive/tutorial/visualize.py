import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import textwrap
import argparse

# ==========================================
# 1. 設定路徑與環境
# ==========================================
# 自動抓取腳本所在目錄，確保相對路徑正確
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAVE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(SAVE_DIR, exist_ok=True)

SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080 

# ==========================================
# 2. 核心繪圖與動畫邏輯
# ==========================================
def draw_background_text(ax, text, x_center, y_center):
    """模擬 index.html 的排版：置中且支援多行"""
    paragraphs = text.split('\n')
    formatted_text = ""
    for p in paragraphs:
        formatted_text += textwrap.fill(p, width=50) + "\n\n"
    
    ax.text(
        x_center, y_center, formatted_text.strip(), 
        fontsize=20, color='gray', alpha=0.2,
        ha='center', va='center', fontfamily='serif',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
    )

def create_trajectory_gif(csv_path, output_gif_path, sample_text, title="Gaze Trajectory"):
    if not os.path.exists(csv_path):
        print(f"⚠️ 找不到檔案: {os.path.basename(csv_path)}")
        return

    print(f"🎬 正在為 {os.path.basename(csv_path)} 製作動畫，請稍候...")
    df = pd.read_csv(csv_path)
    
    # 支援原始資料與校正後資料的欄位名稱
    x_col = 'vibe_x' if 'vibe_x' in df.columns else 'x_px'
    y_col = 'vibe_y' if 'vibe_y' in df.columns else 'y_px'
    df = df.dropna(subset=[x_col, y_col])
    
    # 過濾不合理的極端值 (確保點在螢幕範圍內)
    df = df[(df[x_col] >= 0) & (df[x_col] <= SCREEN_WIDTH)]
    df = df[(df[y_col] >= 0) & (df[y_col] <= SCREEN_HEIGHT)]
    
    if len(df) == 0:
        print(f"⚠️ {os.path.basename(csv_path)} 沒有有效的資料點，跳過製作。")
        return

    # 稍微降採樣，以免 GIF 檔案太大或畫太久 (假設我們畫大約 150 個 frame)
    step = max(1, len(df) // 150)
    df_sampled = df.iloc[::step].copy()
    
    xs = df_sampled[x_col].values
    ys = df_sampled[y_col].values
    
    # 設定畫布 (採用 16:9 比例)
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=100)
    
    ax.set_xlim(0, SCREEN_WIDTH)
    ax.set_ylim(SCREEN_HEIGHT, 0) # Y 軸反轉 (螢幕座標是上方為 0)
    ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("Screen X (px)")
    ax.set_ylabel("Screen Y (px)")
    
    # 畫背景字
    draw_background_text(ax, sample_text, SCREEN_WIDTH/2, SCREEN_HEIGHT/2)
    
    # 初始化軌跡線和目前注視點 (空資料)
    line, = ax.plot([], [], color='blue', alpha=0.5, linewidth=2, label='Trajectory')
    point, = ax.plot([], [], color='red', marker='o', markersize=12, alpha=0.8, label='Current Gaze')
    
    # 加入圖例
    ax.legend(loc='lower right')

    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    def update(frame):
        # 更新軌跡與紅點到目前所在的 frame
        line.set_data(xs[:frame+1], ys[:frame+1])
        # plot 需要傳入 sequence, 所以用 list 包裝
        point.set_data([xs[frame]], [ys[frame]]) 
        return line, point

    # 製作動畫
    ani = animation.FuncAnimation(
        fig, update, frames=len(xs), init_func=init, blit=True, interval=40
    )
    
    # 儲存 GIF
    # Pillow writer 是 matplotlib 內建的，不需額外安裝 ffmpeg 或 imageio
    ani.save(output_gif_path, writer='pillow')
    print(f"✅ 動畫已儲存至: {output_gif_path}")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成眼動軌跡的 GIF 動畫")
    parser.add_argument(
        "--target", 
        type=str, 
        choices=['raw', 'baseline', 'calibrated', 'raw-backup', 'calibrated-backup', 'all'],
        default='all',
        help="指定要視覺化的階段 (預設為 all)"
    )
    args = parser.parse_args()

    # 與 index.html 保持一致的測試文本
    sample_text = (
        "The ubiquitous phenomenon completely bewildered the inexperienced researcher.\n"
        "Despite rigorous analysis, the underlying mechanisms remained enigmatic, defying conventional explanation."
    )
    
    # 定義所有可能的階段檔案對應
    all_files_mapping = {
        'raw': (os.path.join(DATA_DIR, "raw.csv"), "Raw Gaze Trajectory", "raw_trajectory.gif"),
        'baseline': (os.path.join(DATA_DIR, "baseline.csv"), "Baseline (Moving Average) Trajectory", "baseline_trajectory.gif"),
        'calibrated': (os.path.join(DATA_DIR, "calibrated.csv"), "Neuro-Symbolic Calibrated Trajectory", "calibrated_trajectory.gif"),
        'raw-backup': (os.path.join(DATA_DIR, "raw_backup.csv"), "Backup: Raw Gaze Trajectory", "raw_backup_trajectory.gif"),
        'calibrated-backup': (os.path.join(DATA_DIR, "calibrated_backup.csv"), "Backup: Calibrated Trajectory", "calibrated_backup_trajectory.gif"),
    }
    
    if args.target == 'all':
        files_to_process = list(all_files_mapping.values())
    else:
        files_to_process = [all_files_mapping[args.target]]
    
    print(f"🚀 開始產出眼動軌跡 GIF 動畫 (目標: {args.target})...")
    for csv_file, title, out_filename in files_to_process:
        # 只處理實際存在的檔案，避免報錯中斷
        if os.path.exists(csv_file):
            out_path = os.path.join(SAVE_DIR, out_filename)
            create_trajectory_gif(csv_file, out_path, sample_text, title=title)
        else:
            if args.target != 'all':
                print(f"⚠️ 指定的檔案不存在: {csv_file}")
    print("🎉 視覺化處理完畢！")

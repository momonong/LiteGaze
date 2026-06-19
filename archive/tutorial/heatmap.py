import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import textwrap

# ==========================================
# 1. 設定路徑與環境
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_FILE = os.path.join(DATA_DIR, "raw.csv")
CALIB_FILE = os.path.join(DATA_DIR, "calibrated.csv")
SAVE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(SAVE_DIR, exist_ok=True)

SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080 

# ==========================================
# 2. 核心繪圖邏輯
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

def plot_kde_on_ax(ax, csv_path, title, color_map, text_content):
    """在指定的子圖 (ax) 上繪製 KDE 熱力圖"""
    if not os.path.exists(csv_path):
        ax.set_title(f"⚠️ 找不到檔案: {os.path.basename(csv_path)}")
        ax.axis('off')
        return

    df = pd.read_csv(csv_path)
    
    # [優化] 防呆機制：過濾掉超出螢幕的極端雜訊與 NaN，否則 KDE 會變形
    x_col = 'vibe_x' if 'vibe_x' in df.columns else 'x_px'
    y_col = 'vibe_y' if 'vibe_y' in df.columns else 'y_px'
    df = df.dropna(subset=[x_col, y_col])
    df = df[(df[x_col] >= 0) & (df[x_col] <= SCREEN_WIDTH)]
    df = df[(df[y_col] >= 0) & (df[y_col] <= SCREEN_HEIGHT)]

    # 繪製背景文字
    draw_background_text(ax, text_content, SCREEN_WIDTH/2, SCREEN_HEIGHT/2)

    # [優化] bw_adjust 控制平滑度 (預設 1.0 太散，0.5 較適合眼動的注視點)
    sns.kdeplot(
        data=df, x=x_col, y=y_col, 
        fill=True, thresh=0.05, levels=15, 
        cmap=color_map, alpha=0.75, ax=ax, bw_adjust=0.5
    )

    # 設定坐標軸
    ax.set_xlim(0, SCREEN_WIDTH)
    ax.set_ylim(SCREEN_HEIGHT, 0) # 反轉 Y 軸
    ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("Screen X (px)")
    ax.set_ylabel("Screen Y (px)")

# ==========================================
# 3. 產出 Before/After 對照儀表板
# ==========================================
def generate_dashboard():
    print("🎨 正在生成 Neuro-Symbolic 成效對照儀表板...")
    
    # 與 index.html 保持一致的測試文本
    sample_text = (
        "The ubiquitous phenomenon completely bewildered the inexperienced researcher.\n"
        "Despite rigorous analysis, the underlying mechanisms remained enigmatic, defying conventional explanation."
    )
    
    # 建立 1x2 的並排畫布
    fig, axes = plt.subplots(1, 2, figsize=(24, 9), dpi=120)
    
    # 左圖：純神經網路感知 (雜訊多、發散)
    plot_kde_on_ax(axes[0], RAW_FILE, "Phase 1: Raw Neural Perception (Noisy)", "Reds", sample_text)
    
    # 右圖：神經符號融合 (收斂在生難字上)
    plot_kde_on_ax(axes[1], CALIB_FILE, "Phase 3: Neuro-Symbolic Calibrated", "Blues", sample_text)
    
    plt.tight_layout(pad=3.0)
    
    # 儲存與顯示
    output_img = os.path.join(SAVE_DIR, "neuro_symbolic_dashboard.png")
    plt.savefig(output_img, bbox_inches='tight')
    print(f"✅ 對照圖已儲存至: {output_img}")
    plt.show()

if __name__ == "__main__":
    generate_dashboard()
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_from_csv():
    # 1. 讀取你剛剛花 43 分鐘跑完並存好的寶貴數據！
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    csv_path = os.path.join(PROJECT_ROOT, "data", "geco", "benchmark", "noise_tolerance_results.csv")
    print(f"Loading data from {csv_path}...")
    df_results = pd.read_csv(csv_path)
    
    # 2. 解決剛剛的 Bug：加上 numeric_only=True，不要把文字拿去算平均
    df_agg = df_results.groupby('Drift_Y').mean(numeric_only=True).reset_index()
    
    # 3. 開始畫圖
    DRIFT_LEVELS = [0.0, 15.0, 30.0, 45.0, 60.0]
    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    plt.figure(figsize=(7, 5))
    
    plt.plot(df_agg['Drift_Y'], df_agg['STOCK-T_Edge_Rec'], marker='o', linewidth=2.5, color='#2ca25f', label='STOCK-T (Edge/Uniform)')
    plt.plot(df_agg['Drift_Y'], df_agg['STOCK-T_Surprisal_Rec'], marker='s', linewidth=2.5, color='#2b8cbe', label='STOCK-T (Surprisal)')
    plt.plot(df_agg['Drift_Y'], df_agg['Baseline_Rec'], marker='^', linewidth=2.5, color='#de2d26', linestyle='--', label='Baseline (Spatial Only)')
    
    plt.title('Noise Tolerance: OVP Washout Effect', fontweight='bold', pad=15)
    plt.xlabel('Hardware Vertical Drift (px)', fontweight='bold')
    plt.ylabel('Line recovery rate (%)', fontweight='bold')
    plt.ylim(-5, 105)
    plt.xticks(DRIFT_LEVELS)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower left')
    
    # 標示出 "Washout 臨界點"
    plt.axvline(x=30, color='gray', linestyle=':', alpha=0.5)
    plt.text(32, 85, 'Washout\nThreshold', color='gray', fontsize=10)

    # 4. 存檔到 NeurIPS 資料夾
    neurips_fig_dir = os.path.join(PROJECT_ROOT, "output", "NeurIPS", "figures")
    os.makedirs(neurips_fig_dir, exist_ok=True)
    plot_path = os.path.join(neurips_fig_dir, "fig_noise_degradation.pdf")
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Success! Plot directly saved for LaTeX inclusion at: {plot_path}")
    plt.show()

if __name__ == "__main__":
    plot_from_csv()
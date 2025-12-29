import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# ================= 數據區 (從你的 Log 提取) =================
epochs = np.arange(1, 51)
loss_values = [
    1058.99, 788.87, 679.16, 614.58, 556.03, 487.93, 471.07, 428.04, 406.67, 403.25,
    372.97, 405.17, 375.13, 343.15, 345.46, 321.34, 353.59, 328.04, 320.36, 309.11,
    331.05, 287.48, 314.65, 306.67, 296.08, 265.75, 259.01, 248.56, 261.52, 274.23,
    250.51, 255.29, 256.24, 216.01, 239.47, 232.44, 218.42, 212.63, 217.17, 217.49,
    200.39, 223.72, 210.40, 222.18, 215.65, 193.22, 194.53, 195.31, 197.69, 192.61
]
# ==========================================================

def plot_curve():
    plt.figure(figsize=(12, 6))
    
    # 1. 繪製原始數據 (淺色線)
    plt.plot(epochs, loss_values, alpha=0.3, color='gray', linestyle='--', label='Raw Loss')
    
    # 2. 繪製平滑曲線 (深色主線，看起來更專業)
    # 使用 Spline 插值讓線條變圓滑
    X_Y_Spline = make_interp_spline(epochs, loss_values)
    X_ = np.linspace(epochs.min(), epochs.max(), 500)
    Y_ = X_Y_Spline(X_)
    plt.plot(X_, Y_, color='#1f77b4', linewidth=2.5, label='Smoothed Trend')

    # 3. 標註關鍵點
    # 起點
    plt.scatter(1, loss_values[0], color='red', zorder=5)
    plt.text(1, loss_values[0]+20, f'Start: {loss_values[0]:.0f}', fontsize=10, fontweight='bold')
    
    # 終點 (Best Model)
    best_epoch = np.argmin(loss_values) + 1
    best_loss = min(loss_values)
    plt.scatter(best_epoch, best_loss, color='green', s=100, zorder=5, edgecolors='white', linewidth=2)
    plt.text(best_epoch-5, best_loss-60, f'Best: {best_loss:.2f}', fontsize=12, fontweight='bold', color='green')

    # 4. 美化圖表
    plt.title('LiteGaze Production Training Curve (50 Epochs)', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Distillation Loss (MSE + KL)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # 加入說明文字框
    text_str = (
        f"Teacher: ResNet50\n"
        f"Student: MobileNetV3\n"
        f"Dataset: MPIIGaze (Filtered)\n"
        f"Reduction: -{100*(1-best_loss/loss_values[0]):.1f}%"
    )
    props = dict(boxstyle='round', facecolor='aliceblue', alpha=0.5)
    plt.text(0.75, 0.85, text_str, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=props)

    # 存檔
    plt.savefig('plots/production_training_curve.png', dpi=300, bbox_inches='tight')
    print("✅ 圖表已產生：production_training_curve.png")
    plt.show()

if __name__ == "__main__":
    plot_curve()
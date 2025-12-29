import matplotlib.pyplot as plt
import numpy as np

# ================= 數據區 (來自你的 Training Log) =================
epochs = np.arange(1, 11) # Epoch 1 到 10
loss_values = [
    18.8695,  # Epoch 1
    7.5916,   # Epoch 2
    5.8558,   # Epoch 3
    4.8919,   # Epoch 4
    4.0343,   # Epoch 5
    3.6055,   # Epoch 6
    2.9821,   # Epoch 7
    2.6091,   # Epoch 8
    2.2916,   # Epoch 9
    2.1644    # Epoch 10
]
# ==============================================================

def plot_training_curve():
    plt.figure(figsize=(10, 6))
    
    # 畫線
    plt.plot(epochs, loss_values, marker='o', linestyle='-', color='#2ca02c', linewidth=2, label='Distillation Loss')
    
    # 裝飾圖表
    plt.title('LiteGaze Distillation Training Curve', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (KL Divergence)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(epochs) # 確保 X 軸顯示整數 1~10
    
    # 標示起點和終點數值
    plt.text(1, loss_values[0]+0.5, f'{loss_values[0]:.2f}', ha='center', fontsize=10)
    plt.text(10, loss_values[-1]+0.5, f'{loss_values[-1]:.2f}', ha='center', fontsize=10, fontweight='bold', color='red')
    
    # 加上說明框
    text_str = (
        f"Teacher: L2CS-Net (ResNet50)\n"
        f"Student: MobileNetV3-Large\n"
        f"Final Loss: {loss_values[-1]:.4f}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    plt.text(0.65, 0.85, text_str, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    plt.legend()
    plt.tight_layout()
    
    # 存檔
    save_path = 'plots/training_curve.png'
    plt.savefig(save_path, dpi=300)
    print(f"✅ 圖表已儲存為: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_training_curve()
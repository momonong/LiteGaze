import matplotlib.pyplot as plt
import numpy as np

# 你剛剛跑出來的數據
losses = [
    24.8483, 9.1075, 6.7035, 5.4061, 4.8935, 4.5383, 4.2722, 4.0021, 3.7181, 3.5121,
    3.5390, 3.4481, 3.0824, 2.9958, 3.0201, 2.9454, 2.8358, 2.7838, 2.7784, 2.6329,
    2.5160, 2.6887, 2.5165, 2.4423, 2.5503, 2.4287, 2.5272, 2.3779, 2.4201, 2.2647,
    2.2604, 2.1501, 2.2572, 2.0995, 2.0905, 2.0811, 2.0939, 2.0437, 1.9964, 1.9756,
    1.9510, 1.9505, 1.8897, 1.9118, 1.9139, 1.8773, 1.8321, 1.8577, 1.9879, 2.0090
]

epochs = range(1, len(losses) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, 'b-', linewidth=2, label='Distillation Loss (KL Divergence)')
plt.scatter(epochs, losses, color='red', s=15) # 標出數據點

# 標註開始與結束
plt.annotate(f'Start: {losses[0]:.2f}', xy=(1, losses[0]), xytext=(3, losses[0]),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate(f'End: {losses[-1]:.2f}', xy=(len(losses), losses[-1]), xytext=(len(losses)-10, losses[-1]+5),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.title('L2CS-Net Knowledge Distillation Training Curve (3 People / 9k Images)', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# 存檔並顯示
plt.savefig('plots/training_curve_selfmade.png', dpi=300)
print("✅ 訓練曲線圖已儲存為 training_curve_final.png")
plt.show()
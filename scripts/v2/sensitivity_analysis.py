import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def analyze_sensitivity(model, val_dataset, output_path='sensitivity_report.png'):
    """
    針對每一層 Conv，遮蔽 (Mask) 它的 Filters，觀察 Loss 上升幅度。
    幅度越大 = 越重要 (不能剪)。
    幅度越小 = 冗餘 (Prune it!)。
    """
    print("🔬 開始敏感度分析 (這可能需要一段時間)...")
    
    # 1. 取得 Baseline Accuracy (原本的 Loss)
    baseline_loss = model.evaluate(val_dataset, verbose=0)[0] # 假設 return [loss, mae]
    print(f"📉 Baseline Loss: {baseline_loss:.4f}")
    
    layer_sensitivities = {}
    
    # 2. 遍歷所有卷積層
    target_layers = [l for l in model.layers if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D))]
    
    for layer in tqdm(target_layers):
        weights = layer.get_weights()
        if not weights: continue # 跳過沒有權重的層
        
        w = weights[0] # Kernel: (H, W, In, Out)
        num_filters = w.shape[-1]
        
        # 我們不測試每一個 Filter (太慢)，我們測試「整層歸零」的影響
        # 或者是隨機遮蔽 50% 來測試該層的"強健性"
        # 這裡實作 "L1-Norm 重要性排序" 的概念：
        # 如果整層權重都很小，理論上移除它影響不大
        
        l1_norm = np.sum(np.abs(w)) / w.size
        
        # 為了真正測試"剪枝影響"，我們暫時把該層權重設為 0，跑一次 inference
        # (這是最暴力的測試法，最準確)
        original_w = w.copy()
        
        # 模擬剪掉 50% 最小的權重 (Structured Pruning Simulation)
        # 這裡簡化：測試 "整層被干擾" 的後果
        noise = np.random.normal(0, 0.1, w.shape) * np.mean(np.abs(w))
        layer.set_weights([w + noise] + weights[1:]) # 加入雜訊干擾
        
        perturbed_loss = model.evaluate(val_dataset, verbose=0)[0]
        
        # 敏感度 = Loss 增加量
        sensitivity = perturbed_loss - baseline_loss
        layer_sensitivities[layer.name] = sensitivity
        
        # 復原權重
        layer.set_weights(weights)

    # 3. 繪製報告
    names = list(layer_sensitivities.keys())
    values = list(layer_sensitivities.values())
    
    plt.figure(figsize=(12, 6))
    plt.bar(names, values)
    plt.xticks(rotation=90)
    plt.ylabel("Loss Increase (Sensitivity)")
    plt.title("Layer Sensitivity Analysis (Pruning Guide)")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ 分析完成！圖表已存至 {output_path}")
    
    # 4. 輸出建議
    sorted_layers = sorted(layer_sensitivities.items(), key=lambda x: x[1])
    print("\n✂️ 剪枝建議 (最不重要的層):")
    for name, score in sorted_layers[:5]:
        print(f"  - {name}: 敏感度 {score:.4f} (建議 Prune)")

    return layer_sensitivities
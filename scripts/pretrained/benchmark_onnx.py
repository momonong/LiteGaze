import onnxruntime as ort
import time
import numpy as np
import os
import glob
from PIL import Image
from tqdm import tqdm

# ================= ⚙️ 設定 =================
FP32_MODEL = 'models/litegaze_student_fp32.onnx'
INT8_MODEL = 'models/litegaze_student_int8.onnx'
DATA_DIR = 'data/selfmade_combined'
TEST_IMAGES = 200 # 測試幾張圖來算精度
# ==========================================

def compute_gaze(logits):
    # 手動實作 Softmax Expectation
    exp_logits = np.exp(logits - np.max(logits)) 
    probs = exp_logits / np.sum(exp_logits)
    idx = np.arange(90)
    gaze = np.sum(probs * idx) * 4 - 180
    return gaze

def benchmark(model_path, name):
    if not os.path.exists(model_path):
        print(f"❌ 找不到 {model_path}")
        return None, None

    print(f"\n🚀 測試 {name} ({os.path.basename(model_path)})...")
    
    # 使用 CPU 跑
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # 1. 測速
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # 預熱
    print("🔥 正在預熱...")
    for _ in range(20): session.run(None, {input_name: dummy})
    
    # 正式測速
    print("⏱️ 正在測速 (跑 200 次)...")
    start = time.time()
    iters = 200
    for _ in range(iters): session.run(None, {input_name: dummy})
    end = time.time()
    
    avg_time = (end - start) / iters * 1000
    fps = 1000 / avg_time
    print(f"✅ 平均延遲: {avg_time:.2f} ms")
    print(f"⚡ FPS: {fps:.1f}")
    
    return session, input_name

def evaluate_accuracy(fp32_sess, int8_sess, inp_name):
    print(f"\n⚖️ 正在比對精度 (FP32 vs INT8) - 使用 {TEST_IMAGES} 張圖片...")
    files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))[:TEST_IMAGES]
    
    pitch_errs = []
    yaw_errs = []
    
    for f in tqdm(files):
        try:
            # 預處理 (必須跟量化時一致)
            img = Image.open(f).convert('RGB').resize((224, 224))
            img = np.array(img).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, axis=0)
            
            # 推論
            fp32_out = fp32_sess.run(None, {inp_name: img})
            int8_out = int8_sess.run(None, {inp_name: img})
            
            # 算角度
            p1 = compute_gaze(fp32_out[0][0])
            y1 = compute_gaze(fp32_out[1][0])
            p2 = compute_gaze(int8_out[0][0])
            y2 = compute_gaze(int8_out[1][0])
            
            pitch_errs.append(abs(p1 - p2))
            yaw_errs.append(abs(y1 - y2))
        except: continue
        
    mae_p = np.mean(pitch_errs)
    mae_y = np.mean(yaw_errs)
    
    print("\n" + "="*40)
    print("📉 量化損耗報告 (Quantization Loss)")
    print("="*40)
    print(f"Pitch MAE Loss: {mae_p:.4f}°")
    print(f"Yaw   MAE Loss: {mae_y:.4f}°")
    print("="*40)
    
    if mae_p < 1.0 and mae_y < 1.0:
        print("🏆 完美結果！誤差小於 1 度，量化幾乎無損。")
    else:
        print("👌 結果可接受 (通常 < 2 度都算好)。")

def main():
    sess_fp32, name = benchmark(FP32_MODEL, "FP32 ONNX")
    if sess_fp32:
        sess_int8, _ = benchmark(INT8_MODEL, "INT8 ONNX")
        if sess_int8:
            evaluate_accuracy(sess_fp32, sess_int8, name)

if __name__ == "__main__":
    main()
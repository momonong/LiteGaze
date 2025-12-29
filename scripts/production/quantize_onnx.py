import onnx
from onnxruntime.quantization import quantize_dynamic, quantize_static, CalibrationDataReader, QuantType, QuantFormat
import os
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm

# ================= ⚙️ 設定 =================
FP32_MODEL = 'models/litegaze_student_fp32.onnx'
INT8_MODEL = 'models/litegaze_student_int8.onnx'
DATA_DIR = 'data/selfmade_combined'
CALIBRATE_SIZE = 100
# ==========================================

class ImageDataReader(CalibrationDataReader):
    def __init__(self, image_folder, model_path):
        self.image_folder = image_folder
        self.enum_data = None
        
        # 讀取輸入層名稱
        session = onnx.load(model_path)
        self.input_name = session.graph.input[0].name
        
        self.datas = self._load_data()

    def _load_data(self):
        files = glob.glob(os.path.join(self.image_folder, "*.jpg"))[:CALIBRATE_SIZE]
        print(f"📂 讀取 {len(files)} 張圖片進行校準...")
        
        batch_data = []
        for f in tqdm(files):
            try:
                # 1. 讀取與 Resize
                img = Image.open(f).convert('RGB').resize((224, 224))
                
                # 2. 轉成 NumPy 並強制指定 float32
                img = np.array(img).astype(np.float32) / 255.0
                
                # 3. Normalize (關鍵修正：mean/std 也要是 float32)
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img = (img - mean) / std
                
                # 4. Transpose & Expand Dims
                img = img.transpose(2, 0, 1)
                img = np.expand_dims(img, axis=0)
                
                # 5. 🔥 最終保險：再次強制轉型，確保絕對是 float32
                img = img.astype(np.float32)
                
                batch_data.append({self.input_name: img})
            except: continue
            
        return iter(batch_data)

    def get_next(self):
        return next(self.datas, None)

def main():
    if not os.path.exists(FP32_MODEL):
        print(f"❌ 找不到 {FP32_MODEL}")
        return

    print(f"🚀 開始 ONNX 量化 (修正版)...")
    
    dr = ImageDataReader(DATA_DIR, FP32_MODEL)
    
    print("🔄 正在量化模型 (Static Quantization)...")
    quantize_static(
        model_input=FP32_MODEL,
        model_output=INT8_MODEL,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ, 
        per_channel=False, 
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8
    )
    
    print(f"✅ INT8 ONNX 模型已儲存: {INT8_MODEL}")
    
    size_fp32 = os.path.getsize(FP32_MODEL) / 1024**2
    size_int8 = os.path.getsize(INT8_MODEL) / 1024**2
    print(f"\n📊 瘦身成果:")
    print(f"FP32 ONNX: {size_fp32:.2f} MB")
    print(f"INT8 ONNX: {size_int8:.2f} MB")
    print(f"👉 壓縮率: {size_fp32/size_int8:.1f}x")

if __name__ == "__main__":
    main()
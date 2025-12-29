from l2cs import Pipeline, render
import cv2
import torch
import os
import sys

# ================= ⚙️ 設定區 =================
# 這裡指向你剛剛搬進去 models 資料夾的權重檔
CWD = os.getcwd()
MODEL_PATH = os.path.join(CWD, 'models', 'L2CSNet_gaze360.pkl')

# 設定使用的裝置 (你有 5090，當然用 gpu)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# ============================================

def main():
    # 1. 檢查模型檔案是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 錯誤：找不到模型檔案 {MODEL_PATH}")
        print("請確認你已經建立 'models' 資料夾並把 .pkl 檔放進去")
        return

    print(f"🚀 Loading L2CS-Net Pipeline on {DEVICE}...")
    
    # 2. 初始化官方 Pipeline
    gaze_pipeline = Pipeline(
        weights=MODEL_PATH,
        arch='ResNet50',
        device=DEVICE
    )
    print("✅ Model Loaded!")

    # 3. 開啟 Webcam
    cap = cv2.VideoCapture(0)
    
    # 設定解析度 (可以根據電腦性能調整)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ 無法開啟攝影機")
        return

    print("📷 Demo Started! Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取影像")
            break

        # 4. 官方核心步驟：一步完成預測
        # step() 會幫你做人臉偵測 + 視線預測
        results = gaze_pipeline.step(frame)

        # 5. 官方核心步驟：渲染結果
        # render() 會幫你畫出漂亮的箭頭和框框
        frame = render(frame, results)

        cv2.imshow("L2CS-Net Official Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
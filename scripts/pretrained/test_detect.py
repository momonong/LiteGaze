from l2cs import Pipeline, render
import cv2
import torch
import sys

# 1. 設定裝置 (你有 5090，一定要用 cuda)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"🚀 Using device: {device}")

# 2. 載入模型
try:
    gaze_pipeline = Pipeline(
        weights='models/L2CSNet_gaze360.pkl',
        arch='ResNet50',
        device=device
    )
except FileNotFoundError:
    print("❌ 找不到模型檔案！請確認 models/L2CSNet_gaze360.pkl 存在。")
    sys.exit()

# 3. 開啟 WebCam
cam = 0
cap = cv2.VideoCapture(cam)

# 設定解析度 (選用，為了畫質)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("📷 Demo Started! Press 'q' to exit.")

# 4. 🔥 重點：加上 while 迴圈連續讀取
while True:
    ret, frame = cap.read()
    if not ret:
        print("無法讀取影像")
        break

    # 處理並繪製
    results = gaze_pipeline.step(frame)
    frame = render(frame, results)

    # 5. 🔥 重點：顯示視窗
    cv2.imshow("L2CS-Net Demo", frame)

    # 6. 🔥 重點：等待按鍵 (每 1 毫秒檢查一次，按 q 離開)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
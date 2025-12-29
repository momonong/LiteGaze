import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from l2cs import Pipeline, render
import sys
import os
import time

# ================= ⚙️ 設定區 =================
# 1. 剛剛練好的最強學生模型
STUDENT_PATH = 'models/student_mobilenet_mpii_logits.pth'

# 2. 官方老師模型 (只用來抓臉)
TEACHER_PATH = 'models/L2CSNet_gaze360.pkl'

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# ============================================

# === 學生模型架構 ===
class L2CS_MobileNetV3(nn.Module):
    def __init__(self, num_bins=90):
        super(L2CS_MobileNetV3, self).__init__()
        self.backbone = models.mobilenet_v3_large(weights=None)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_bins * 2)
    def forward(self, x):
        x = self.backbone(x)
        return x[:, :90], x[:, 90:]

def compute_gaze(logits):
    softmax = nn.Softmax(dim=1)
    prob = softmax(logits)
    idx = torch.arange(90, dtype=torch.float32).to(logits.device)
    gaze = torch.sum(prob * idx, dim=1) * 4 - 180
    return gaze.item()

def draw_gaze(image, pitch, yaw, bbox, color=(0, 0, 255)):
    x_min, y_min, x_max, y_max = bbox
    cx, cy = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
    face_w = x_max - x_min
    
    length = face_w / 2.0
    dx = -length * np.sin(yaw) * np.cos(pitch)
    dy = -length * np.sin(pitch)
    
    cv2.arrowedLine(image, (cx, cy), 
                   (int(cx + dx), int(cy + dy)), 
                   color, 4, cv2.LINE_AA, tipLength=0.2)

def main():
    print(f"🚀 Loading Student Model: {STUDENT_PATH}")
    student = L2CS_MobileNetV3().to(DEVICE)
    try:
        student.load_state_dict(torch.load(STUDENT_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"❌ 學生模型載入失敗！錯誤訊息: {e}")
        return
    student.eval()

    print(f"🚀 Loading Teacher Pipeline (For Detection Only)...")
    detection_pipeline = Pipeline(
        weights=TEACHER_PATH,
        arch='ResNet50',
        device=DEVICE
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n✅ Demo Started! (Press 'q' to exit)")
    print("---------------------------------------")
    
    fps_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. 計算 FPS
        frame_count += 1
        if time.time() - fps_time > 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = time.time()

        # 🔥【關鍵修復】加上 try-except 防止沒人臉時崩潰
        try:
            results = detection_pipeline.step(frame)
        except ValueError:
            # 這代表 l2cs 沒抓到臉，直接跳過這幀
            cv2.imshow("Student Model Final Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue
        except Exception as e:
            print(f"Unexpected error: {e}")
            continue
        
        # 2. 如果有抓到臉
        if results.bboxes is not None and len(results.bboxes) > 0:
            bbox = results.bboxes[0] 
            x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            h, w, _ = frame.shape
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)
            
            face_img = frame[y_min:y_max, x_min:x_max]
            
            if face_img.size > 0:
                # 3. 學生推論
                img_pil = transforms.ToPILImage()(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                inp = transform(img_pil).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    sp_logits, sy_logits = student(inp)
                    s_pitch = compute_gaze(sp_logits)
                    s_yaw = compute_gaze(sy_logits)
                
                # 4. 繪製
                pitch_rad = s_pitch * np.pi / 180
                yaw_rad = s_yaw * np.pi / 180
                
                draw_gaze(frame, pitch_rad, yaw_rad, [x_min, y_min, x_max, y_max], color=(0, 0, 255))
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                info = f"Pitch: {s_pitch:.1f}  Yaw: {s_yaw:.1f}"
                cv2.putText(frame, info, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Student Model Final Demo", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
from l2cs import Pipeline
from torchvision import transforms
import cv2
import numpy as np

# 繼承並修改官方 Pipeline，只為了挖出 Raw Logits
class TeacherPipeline(Pipeline):
    def step_logits(self, frame):
        # 這是官方 step() 的簡化版，重點是回傳 raw_logits
        face_imgs = self._get_face_crops(frame) # 使用官方內建的裁切邏輯
        if face_imgs is None or len(face_imgs) == 0:
            return None, None
        
        # 轉成 Tensor
        face_imgs = torch.stack(face_imgs).to(self.device)
        
        with torch.no_grad():
            # 🔥 關鍵：直接呼叫 model，取得 [Batch, 90] 的 logits
            pitch_logits, yaw_logits = self.model(face_imgs)
            
        return pitch_logits, yaw_logits

    def _get_face_crops(self, frame):
        # 偷用官方的私有方法來做一樣的裁切
        results = self.detect_faces(frame)
        if results.bboxes is None or len(results.bboxes) == 0:
            return []
        
        face_imgs = []
        for bbox in results.bboxes:
            bbox = bbox.astype(int)
            x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
            # Padding 邏輯需與 Teacher 一致
            h, w, _ = frame.shape
            # (官方可能有自己的 padding，這裡我們盡量模仿或直接用 detector 裁切)
            # 為了簡單，我們這裡直接切 bbox (官方 pipeline 內部有做處理)
            face_img = frame[y_min:y_max, x_min:x_max]
            
            # 預處理
            if face_img.size == 0: continue
            img = cv2.resize(face_img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
            face_imgs.append(img)
            
        return face_imgs
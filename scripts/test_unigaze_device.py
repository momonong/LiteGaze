"""
scripts/test_unigaze_device.py
==============================================================================
Tests UniGaze ViT inference on CUDA vs CPU to verify sm_120 kernel execution.
==============================================================================
"""

import sys
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.unigaze_personalization.model import load_unigaze_b16, UniGazeFeatureWrapper
from core.unigaze_personalization.transforms import to_unigaze_tensor

def main():
    print("Testing UniGaze Device Loading...")
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())

    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    
    for dev in ["cuda", "cpu"]:
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        print(f"\n--- Testing Device: {dev} ---")
        try:
            model = UniGazeFeatureWrapper(load_unigaze_b16(dev)).to(dev).eval()
            img_tensor = to_unigaze_tensor(dummy_img).unsqueeze(0).to(dev)
            with torch.no_grad():
                pred = model(img_tensor)
            print(f"✅ Success on {dev}! Output shape: {pred.shape}, Pred: {pred.cpu().numpy()}")
        except Exception as e:
            print(f"❌ Failed on {dev}: {e}")

if __name__ == "__main__":
    main()

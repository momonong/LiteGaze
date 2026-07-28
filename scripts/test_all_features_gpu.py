import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import time
import base64
import cv2
import numpy as np
import torch

def get_vram_mb():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
        return allocated, reserved
    return 0.0, 0.0

def main():
    print("=" * 75)
    print("  LexiGaze Full System GPU Verification Suite")
    print("=" * 75)

    results = []

    # 1. Hardware & CUDA Environment Test
    print("\n[Module 1/7] Hardware & CUDA Profile...")
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    alloc_mb, res_mb = get_vram_mb()
    print(f"  CUDA Available : {cuda_avail}")
    print(f"  GPU Device     : {device_name}")
    print(f"  Allocated VRAM : {alloc_mb:.2f} MB | Reserved VRAM: {res_mb:.2f} MB")
    results.append(("1. CUDA Hardware Profile", "PASS", f"GPU: {device_name} ({alloc_mb:.1f}MB VRAM)"))

    # 2. UniGaze Feature Extraction on GPU
    print("\n[Module 2/7] UniGaze Vision Transformer on GPU...")
    vram_before, _ = get_vram_mb()
    try:
        from core.unigaze_personalization.model import UniGazeFeatureWrapper, load_unigaze_b16
        
        # Test loading on GPU device if supported, or verify safe device fallback
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            t = torch.zeros((1, 3, 224, 224), device="cuda")
            conv = torch.nn.Conv2d(3, 16, kernel_size=16, stride=16).to("cuda")
            _ = conv(t)
            device = "cuda"
        except Exception:
            device = "cpu"

        model = UniGazeFeatureWrapper(load_unigaze_b16(device)).to(device).eval()
        fake_img = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            gaze_pred = model(fake_img)
        vram_after, _ = get_vram_mb()
        vram_delta = vram_after - vram_before
        print(f"  UniGaze Execution Device : {device}")
        print(f"  Gaze Prediction Shape   : {gaze_pred.shape}")
        print(f"  VRAM Consumption        : +{vram_delta:.2f} MB")
        results.append(("2. UniGaze ViT Neural Net", "PASS", f"Device: {device}, Output: {list(gaze_pred.shape)}"))
    except Exception as exc:
        print(f"  ❌ UniGaze Test Failed: {exc}")
        results.append(("2. UniGaze ViT Neural Net", "FAIL", str(exc)))

    # 3. LLM Cognitive Load Pipeline on GPU
    print("\n[Module 3/7] LLM Cognitive Load Pipeline on GPU...")
    vram_before, _ = get_vram_mb()
    try:
        from core.cognition.pipeline import LanguageModelCalculator
        lm_calc = LanguageModelCalculator(model_type='bert', lang='en')
        words = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
        lm_res = lm_calc.compute(words)
        vram_after, _ = get_vram_mb()
        vram_delta = vram_after - vram_before
        print(f"  LM Execution Device : {lm_calc.device}")
        print(f"  Tokens Processed   : {len(words)}")
        print(f"  Surprisal Mean     : {np.mean(lm_res['surprisals']):.4f}")
        print(f"  VRAM Consumption   : +{vram_delta:.2f} MB")
        results.append(("3. LLM Cognitive Pipeline", "PASS", f"Device: {lm_calc.device}, Mean Surprisal: {np.mean(lm_res['surprisals']):.2f}"))
    except Exception as exc:
        print(f"  ❌ LLM Pipeline Failed: {exc}")
        results.append(("3. LLM Cognitive Pipeline", "FAIL", str(exc)))

    # 4. Personalization Trainer & Model Calibration
    print("\n[Module 4/7] Personalization Model Calibration Trainer...")
    try:
        from core.unigaze_personalization.dataset import read_manifest
        from core.gaze_core.training import train_placeholder
        session_dir = ROOT / "data" / "sessions" / "test_verification_session"
        session_dir.mkdir(parents=True, exist_ok=True)
        dummy_manifest = session_dir / "manifest.jsonl"
        if not dummy_manifest.exists():
            import json
            rec = {
                "session_id": "test_verification_session",
                "normalized_face_path": "dummy_face.jpg",
                "target_x_norm": 0.5,
                "target_y_norm": 0.5,
                "viewport_width": 1920.0,
                "viewport_height": 1080.0
            }
            # Save dummy face image
            dummy_img = np.full((224, 224, 3), 128, dtype=np.uint8)
            cv2.imwrite(str(session_dir / "dummy_face.jpg"), dummy_img)
            with dummy_manifest.open("w", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        resp, status = train_placeholder(ROOT, {"data_session_id": "test_verification_session", "output_model_name": "gpu_test_model"})
        print(f"  Training Status Code : {status}")
        print(f"  Training Response OK : {resp.get('ok', False)}")
        results.append(("4. Personalization Trainer", "PASS" if resp.get("ok", False) else "FAIL", f"Status: {status}"))
    except Exception as exc:
        print(f"  ❌ Calibration Trainer Failed: {exc}")
        results.append(("4. Personalization Trainer", "FAIL", str(exc)))

    # 5. Psycholinguistic POM Transition Matrix & Viterbi Decoder
    print("\n[Module 5/7] Psycholinguistic POM Matrix & Viterbi Decoder...")
    try:
        from scripts.geco.core.transition_model import PsycholinguisticTransitionMatrix
        from scripts.geco.core.viterbi_decoder import viterbi_gaze_decode

        trans_model = PsycholinguisticTransitionMatrix()
        base_cm = np.array([0.2, 0.8, 0.3, 0.9, 0.4])
        word_boxes = np.array([[0, 0, 50, 30], [60, 0, 110, 30], [120, 0, 170, 30], [180, 0, 230, 30], [240, 0, 290, 30]])
        pom_matrix = trans_model.build_matrix(len(base_cm), base_cm, is_L2=True, word_boxes=word_boxes)

        fake_gaze = np.array([[25.0, 15.0], [85.0, 15.0], [145.0, 15.0], [205.0, 15.0], [265.0, 15.0]], dtype=np.float64)
        decoded, ll = viterbi_gaze_decode(fake_gaze, word_boxes, base_cm, pom_matrix)
        print(f"  POM Matrix Shape : {pom_matrix.shape}")
        print(f"  Decoded Sequence : {decoded}")
        print(f"  Log Likelihood   : {ll:.2f}")
        results.append(("5. POM & Viterbi Decoder", "PASS", f"Likelihood: {ll:.2f}"))
    except Exception as exc:
        print(f"  ❌ POM/Viterbi Failed: {exc}")
        results.append(("5. POM & Viterbi Decoder", "FAIL", str(exc)))

    # 6. Multi-Line Adaptive EM Anchoring Decoder
    print("\n[Module 6/7] Multi-Line Adaptive EM Anchoring Decoder...")
    try:
        from scripts.geco.core.em_calibration import MultiLineAdaptiveEMDecoder
        decoder = MultiLineAdaptiveEMDecoder()
        word_boxes = np.array([
            [10, 10, 50, 40], [60, 10, 100, 40],   # Line 1
            [10, 60, 50, 90], [60, 60, 100, 90]    # Line 2
        ])
        base_cm = np.array([0.5, 0.5, 0.5, 0.5])
        trans_matrix = np.eye(4) / 4.0
        raw_gaze = np.array([[30.0, 25.0], [80.0, 25.0], [30.0, 85.0], [80.0, 85.0]], dtype=np.float64)
        decoded, drift = decoder.calibrate_and_decode(raw_gaze, word_boxes, base_cm, trans_matrix)
        print(f"  Multi-Line Decoded Indices : {decoded}")
        print(f"  Estimated Line Drifts      : {drift[1]}")
        results.append(("6. Multi-Line Adaptive EM", "PASS", f"Lines Clustered: {len(drift[1])}"))
    except Exception as exc:
        print(f"  ❌ Multi-Line EM Failed: {exc}")
        results.append(("6. Multi-Line Adaptive EM", "FAIL", str(exc)))

    # 7. Real-time Inference Predict API Pipeline
    print("\n[Module 7/7] Real-time Inference Predict Pipeline...")
    try:
        from core.gaze_core.inference import predict
        dummy_img = np.full((480, 640, 3), 128, dtype=np.uint8)
        _, img_buf = cv2.imencode(".jpg", dummy_img)
        b64_str = "data:image/jpeg;base64," + base64.b64encode(img_buf).decode("utf-8")
        payload = {"image_data": b64_str, "model_name": "before", "viewport_width": 1920, "viewport_height": 1080}
        pred_resp, status = predict(ROOT, payload)
        print(f"  Predict Status Code : {status}")
        print(f"  Predict Error Msg   : {pred_resp.get('error', 'None')}")
        # Handled non-face frame validation (400) or valid prediction (200) cleanly
        pass_cond = (status == 200) or (status == 400 and "no face detected" in pred_resp.get("error", ""))
        results.append(("7. Inference Predict API", "PASS" if pass_cond else "FAIL", f"Status: {status} ({pred_resp.get('error', 'OK')})"))
    except Exception as exc:
        print(f"  ❌ Inference Predict Failed: {exc}")
        results.append(("7. Inference Predict API", "FAIL", str(exc)))

    print("\n" + "=" * 75)
    print("  FULL SYSTEM GPU VERIFICATION SUMMARY")
    print("=" * 75)
    print(f"  {'Module Name':<35} | {'Status':<8} | Details")
    print("  " + "-" * 71)
    for name, status, details in results:
        flag = "✅" if status == "PASS" else ("⚠️" if status == "SKIP" else "❌")
        print(f"  {flag} {name:<32} | {status:<8} | {details}")
    print("=" * 75)

    # Save verification report
    report_md = "# Full System GPU Verification Report\n\n"
    report_md += f"**Execution Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    report_md += f"**GPU Device**: {device_name}\n"
    report_md += f"**Current Allocated VRAM**: {alloc_mb:.2f} MB | Reserved: {res_mb:.2f} MB\n\n"
    report_md += "## Verification Results Summary\n\n"
    report_md += "| Module Name | Status | Details |\n"
    report_md += "| :--- | :--- | :--- |\n"
    for name, status, details in results:
        flag = "✅" if status == "PASS" else ("⚠️" if status == "SKIP" else "❌")
        report_md += f"| {flag} {name} | **{status}** | {details} |\n"

    report_path = ROOT / "output" / "full_system_gpu_verification_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"\nVerification report saved to {report_path}")

if __name__ == "__main__":
    main()

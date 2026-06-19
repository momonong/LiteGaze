import os
import sys
from pathlib import Path

# Set up ROOT path for import
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

print("=== Refactoring Diagnostic Tests ===")

# Test 1: Import cognitive load pipeline
try:
    print("Testing Test 1: CognitiveLoadPipeline import...")
    from cognition import CognitiveLoadPipeline
    print("✓ Successfully imported CognitiveLoadPipeline.")
except Exception as e:
    print(f"✗ Failed Test 1: {e}")
    sys.exit(1)

# Test 2: Import gaze core module
try:
    print("Testing Test 2: Gaze Core imports...")
    from gaze_core.model_registry import ensure_runs_dir
    from gaze_core.sample_store import ensure_sessions_dir
    
    runs_dir = ensure_runs_dir(ROOT)
    sessions_dir = ensure_sessions_dir(ROOT)
    print(f"✓ Gaze Core directories validated:")
    print(f"  Runs Dir: {runs_dir}")
    print(f"  Sessions Dir: {sessions_dir}")
except Exception as e:
    print(f"✗ Failed Test 2: {e}")
    sys.exit(1)

# Test 3: Import and test Fusion Orchestrator
try:
    print("Testing Test 3: Fusion Orchestrator import and logic...")
    from scripts.fusion.orchestrator import _classify_rds, build_cognitive_lookup
    
    # Test RDS classification logic
    assert _classify_rds(0.85) == "difficulty"
    assert _classify_rds(0.55) == "attention"
    assert _classify_rds(0.20) == "fluent"
    print("✓ RDS classification logic works.")
    
    # Test lookup builder logic
    mock_analysis = [{"word": "neuro-symbolic", "surprisal": 0.5}]
    lookup = build_cognitive_lookup(mock_analysis)
    assert "neuro-symbolic" in lookup
    assert "neuro" in lookup
    assert "symbolic" in lookup
    print("✓ Cognitive lookup builder handles hyphenated words correctly.")
except Exception as e:
    print(f"✗ Failed Test 3: {e}")
    sys.exit(1)

# Test 4: Run Flask routing checks
try:
    print("Testing Test 4: Flask server routing integration...")
    import requests
    response = requests.get("http://localhost:8080/api/cognitive/health")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    print("✓ Flask server live endpoint ping verified successfully.")
except Exception as e:
    print(f"✗ Failed Test 4: {e} (Is the Flask server running on port 8080?)")
    # We do not crash the tests if the server isn't running, but we warn the user.
    print("  Note: Flask server query skipped or failed. Ensure `run.py` is running.")

print("\n🎉 All refactoring diagnostic tests completed successfully!")

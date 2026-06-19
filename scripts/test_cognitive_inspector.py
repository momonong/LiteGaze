import sys
from pathlib import Path

# Setup root path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.cognitive_inspector import CognitiveInspector, generate_markdown_report

def main():
    print("=== Testing Cognitive Inspector Backend ===")
    
    # 1. Simulate reading timeline (sequence of words and indices)
    # A fluent sequence: 0, 1, 2, 3, 4
    # With a regression: 4 -> 2
    # With a reread: 2 is visited again
    gaze_history = [
        # Word 0: "The" (index 0)
        {"word": "The", "index": 0, "confidence": "high", "timestamp_ms": 1000},
        {"word": "The", "index": 0, "confidence": "high", "timestamp_ms": 1120},
        {"word": "The", "index": 0, "confidence": "medium", "timestamp_ms": 1240},
        
        # Word 1: "neuro-symbolic" (index 1)
        {"word": "neuro-symbolic", "index": 1, "confidence": "high", "timestamp_ms": 1360},
        {"word": "neuro-symbolic", "index": 1, "confidence": "medium", "timestamp_ms": 1480},
        {"word": "neuro-symbolic", "index": 1, "confidence": "low", "timestamp_ms": 1600},
        
        # Word 2: "method" (index 2)
        {"word": "method", "index": 2, "confidence": "high", "timestamp_ms": 1720},
        
        # Word 3: "improves" (index 3)
        {"word": "improves", "index": 3, "confidence": "high", "timestamp_ms": 1840},
        {"word": "improves", "index": 3, "confidence": "high", "timestamp_ms": 1960},
        
        # Word 4: "accuracy" (index 4)
        {"word": "accuracy", "index": 4, "confidence": "high", "timestamp_ms": 2080},
        {"word": "accuracy", "index": 4, "confidence": "high", "timestamp_ms": 2200},
        {"word": "accuracy", "index": 4, "confidence": "high", "timestamp_ms": 2320},
        {"word": "accuracy", "index": 4, "confidence": "medium", "timestamp_ms": 2440}, # Long dwell (total 4 ticks = ~480ms)
        
        # Regression back to Word 2: "method" (index 2)
        {"word": "method", "index": 2, "confidence": "high", "timestamp_ms": 2600},
        {"word": "method", "index": 2, "confidence": "high", "timestamp_ms": 2720},
        
        # Move forward again to Word 5: "significantly" (index 5)
        {"word": "significantly", "index": 5, "confidence": "high", "timestamp_ms": 2900},
    ]

    inspector = CognitiveInspector(sample_rate_hz=8)
    
    # Run aggregation
    fixations = inspector.group_fixations(gaze_history)
    print(f"Aggregated Fixations: {len(fixations)}")
    for i, f in enumerate(fixations):
        print(f"  [{i}] Word: {f.word:<15} Index: {f.index} Dwell: {f.duration_ms}ms Conf: {f.confidence}")
    
    assert len(fixations) > 0, "Fixations should be aggregated"
    
    # Run analysis
    result = inspector.analyze(gaze_history, lang="en")
    print("\nAnalysis Result User Profile:")
    for k, v in result["user_profile"].items():
        print(f"  {k}: {v}")
        
    print("\nAnalysis Result Summary:")
    for k, v in result["summary"].items():
        print(f"  {k}: {v}")

    # Check metrics
    summary = result["summary"]
    profile = result["user_profile"]
    
    assert summary["regression_count"] == 1, f"Expected 1 regression, got {summary['regression_count']}"
    assert summary["reread_count"] == 1, f"Expected 1 reread, got {summary['reread_count']}"
    assert profile["reading_ability_score"] > 0, "Reading ability score should be positive"
    assert profile["english_proficiency_score"] > 0, "English proficiency score should be positive"
    assert profile["fatigue_ratio"] > 0, "Fatigue ratio should be calculated"

    # 2. Test markdown report generation
    report_md = generate_markdown_report(result, "test_agent_user")
    print("\n=== Generated Report Preview ===")
    lines = report_md.strip().split("\n")
    for line in lines[:25]:
        print(line)
    print("...")
    
    print("\n✓ All Cognitive Inspector tests passed successfully!")

if __name__ == "__main__":
    main()

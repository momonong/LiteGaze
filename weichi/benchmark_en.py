import sys
import io
import json

# 修正 Windows 終端機顯示問題
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from cognitive_load_pipeline import CognitiveLoadPipeline

# 定義英文測試組 (取材自 GECO 或常見認知心理學難題)
EN_BENCHMARK_SUITE = [
    {
        "text": "The ubiquitous phenomenon completely bewildered the researchers during the observation.",
        "expected_high": ["ubiquitous", "phenomenon", "bewildered"],
    },
    {
        "text": "The complex financial derivative products led to an unprecedented global economic crisis.",
        "expected_high": ["derivative", "unprecedented", "crisis"],
    },
    {
        "text": "If you don't mind, we should investigate the mysterious laboratory immediately.",
        "expected_high": ["investigate", "mysterious", "immediately"],
    }
]

def run_en_benchmark():
    # 切換到英文模式
    pipeline = CognitiveLoadPipeline(model_type='bert', lang='en')
    hits = 0
    total_expected = 0
    total_predicted = 0
    
    print("\n" + "!"*20 + " English Cognitive Load Validation " + "!"*20)
    
    for i, test in enumerate(EN_BENCHMARK_SUITE):
        result = pipeline.run(test["text"])
        high_load_found = result["high_load_words"]
        word_analysis = result["word_analysis"]
        
        print(f"\n[Test {i+1}] {test['text']}")
        print(f"  Human Ground Truth: {test['expected_high']}")
        
        print(f"  Model Predictions (All Scores):")
        for w in word_analysis:
            star = "⭐" if w['word'] in test['expected_high'] else "  "
            mark = " [HIGH]" if w['load_level'] == "high" else ""
            print(f"    {star} {w['word']:<12} | score: {w['load_score']:.3f} | surp: {w['surprisal']:>6}{mark}")
        
        current_hits = len(set(high_load_found) & set(test["expected_high"]))
        hits += current_hits
        total_expected += len(test["expected_high"])
        total_predicted += len(high_load_found)

    precision = hits / total_predicted if total_predicted > 0 else 0
    recall = hits / total_expected if total_expected > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*60)
    print(f"Final English Results:")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 Score:  {f1:.2%}")
    print("="*60)

if __name__ == "__main__":
    run_en_benchmark()

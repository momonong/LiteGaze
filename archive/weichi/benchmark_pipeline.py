
import json
import sys
import io

# 修正 Windows 終端機顯示中文問題
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from cognitive_load_pipeline import CognitiveLoadPipeline

# 定義測試組
BENCHMARK_SUITE = [
    {
        "text": "日本要想在會上與西方國家保持協調，必須得到美國的支持和理解。",
        "expected_high": ["協調", "支持", "理解"],
    },
    {
        "text": "江蘇隊的核心隊員是國手胡衛東，全隊的戰術是以他為中心制定的。",
        "expected_high": ["胡衛東", "國手", "戰術"],
    },
    {
        "text": "有超過半數的在港德國公司表示，會繼續擴大其在亞洲地區的業務。",
        "expected_high": ["業務", "擴大", "德國"],
    },
    {
        "text": "大部分日子都是早晨六點起床，晚上二十四時左右睡覺。",
        "expected_high": ["睡覺", "二十四時"],
    },
    {
        "text": "由於這些原因，他決定放棄這次出國深造的機會。",
        "expected_high": ["深造", "放棄", "機會"],
    }
]

def run_benchmark():
    pipeline = CognitiveLoadPipeline(model_type='bert', lang='zh')
    hits = 0
    total_expected = 0
    total_predicted = 0
    
    print("\n" + "!"*20 + " 認知負荷 Pipeline 效能驗證 (Precision/Recall/F1) " + "!"*20)
    
    for i, test in enumerate(BENCHMARK_SUITE):
        result = pipeline.run(test["text"])
        high_load_found = result["high_load_words"]
        word_analysis = result["word_analysis"]
        
        print(f"\n[測試 {i+1}] {test['text']}")
        print(f"  人類基準高負荷：{test['expected_high']}")
        
        # 顯示高負荷單詞及其分數
        print(f"  模型預測高負荷 (score >= 0.65)：")
        for w in word_analysis:
            if w['load_level'] == 'high':
                star = "⭐" if w['word'] in test['expected_high'] else "  "
                print(f"    {star} {w['word']:<6} | score: {w['load_score']:.3f} | surp: {w['surprisal']:>6} | pos: {w['pos']}")
        
        current_hits = len(set(high_load_found) & set(test["expected_high"]))
        hits += current_hits
        total_expected += len(test["expected_high"])
        total_predicted += len(high_load_found)
        
        if current_hits > 0:
            p_batch = current_hits / len(high_load_found) if len(high_load_found) > 0 else 0
            r_batch = current_hits / len(test["expected_high"])
            print(f"  ✅ 命中 {current_hits}/{len(test['expected_high'])} 個點 (P: {p_batch:.2f}, R: {r_batch:.2f})")
        else:
            print(f"  ❌ 未能捕捉到預期關鍵點")

    # 計算整體指標
    precision = hits / total_predicted if total_predicted > 0 else 0
    recall = hits / total_expected if total_expected > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*60)
    print(f"最終驗證結果：")
    print(f"總預測數量：{total_predicted}")
    print(f"總命中數量：{hits}")
    print(f"總預期數量：{total_expected}")
    print("-" * 30)
    print(f"精準率 (Precision): {precision:.2%}")
    print(f"召回率 (Recall):    {recall:.2%}")
    print(f"F1 分數 (F1 Score): {f1:.2%}")
    print("="*60)

if __name__ == "__main__":
    run_benchmark()

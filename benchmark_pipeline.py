
import json
from core.cognition import CognitiveLoadPipeline

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
    
    print("\n" + "!"*20 + " 認知負荷 Pipeline 效能驗證 " + "!"*20)
    
    for i, test in enumerate(BENCHMARK_SUITE):
        result = pipeline.run(test["text"])
        high_load_found = result["high_load_words"]
        
        print(f"\n[測試 {i+1}] {test['text']}")
        print(f"  模型預測高負荷：{high_load_found}")
        print(f"  人類基準高負荷：{test['expected_high']}")
        
        current_hits = len(set(high_load_found) & set(test["expected_high"]))
        hits += current_hits
        total_expected += len(test["expected_high"])
        
        if current_hits > 0:
            print(f"  ✅ 成功捕捉到 {current_hits} 個關鍵負荷點")
        else:
            print(f"  ❌ 未能捕捉到預期關鍵點")

    accuracy = (hits / total_expected) * 100
    print("\n" + "="*60)
    print(f"最終驗證結果：")
    print(f"總預期關鍵詞數：{total_expected}")
    print(f"模型成功命中數：{hits}")
    print(f"初步命中率 (Hit Rate)：{accuracy:.2f}%")
    print("="*60)

if __name__ == "__main__":
    run_benchmark()

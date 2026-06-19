
import sys
import io
import os
# 確保能找到 local modules
sys.path.append(os.getcwd())

from lexigaze.weichi.cognitive_load_pipeline import CognitiveLoadPipeline

def test():
    texts = [
        ('zh', '雖然今天天氣不錯，但複雜的量子力學概念依然讓人感到困惑。'),
        ('zh', '大型語言模型的發展代表了人工智慧領域的一個重要里程碑。透過神經網絡架構的持續優化，模型能夠在處理自然語言理解與生成任務時展現出驚人的準確度。'),
        ('en', 'The detective carefully examined the fingerprints left on the cold glass surface.'),
        ('en', 'The emergence of neuro-symbolic AI signifies a paradigm shift in machine learning.')
    ]
    
    for lang, text in texts:
        p = CognitiveLoadPipeline(model_type='bert', lang=lang)
        res = p.run(text)
        print(f"\n[{lang.upper()} Result]")
        print(f"Text: {text}")
        print(f"High Load: {res['high_load_words']}")
        print(f"Time: {res['process_time_ms']}ms")

if __name__ == "__main__":
    test()

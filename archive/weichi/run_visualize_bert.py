import json
from cognitive_load_pipeline import CognitiveLoadPipeline
from visualize_load import generate_html_heatmap

DEMO_TEXTS = [
    "今天天氣很好，我想去公園散步。",
    "量子糾纏是一種物理現象，描述兩個粒子之間的非局域關聯。",
    "學生在課堂上認真聽老師講解微積分的基本概念。",
]

if __name__ == "__main__":
    # 使用 BERT 模型
    pipeline = CognitiveLoadPipeline(model_type='bert', lang='zh')
    
    for i, text in enumerate(DEMO_TEXTS):
        result = pipeline.run(text)
        
        # 儲存 JSON
        json_path = f"bert_output_{i}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        # 生成 HTML
        generate_html_heatmap(json_path, f"bert_heatmap_{i}.html")
        print(f"BERT Heatmap {i} generated.")

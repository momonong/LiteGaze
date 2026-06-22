import json
from core.cognition import CognitiveLoadPipeline
from visualize_load import generate_html_heatmap

ENGLISH_TEXTS = [
    "The mystery was finally solved by the clever detective.",
    "Quantum mechanics is a fundamental theory in physics describing nature at the scale of atoms.",
    "The quick brown fox jumps over the lazy dog."
]

if __name__ == "__main__":
    # 使用 BERT 英文模型
    pipeline_en = CognitiveLoadPipeline(model_type='bert', lang='en')
    
    print("\n" + "="*20 + " English Cognitive Load Analysis " + "="*20)
    
    for i, text in enumerate(ENGLISH_TEXTS):
        result = pipeline_en.run(text)
        
        print(f"\n[Text {i+1}]: {text}")
        print(f"High Load Words: {result['high_load_words']}")
        
        # 儲存 JSON
        json_path = f"en_output_{i}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            
        # 生成 HTML
        generate_html_heatmap(json_path, f"en_heatmap_{i}.html")
        print(f"Generated Heatmap: en_heatmap_{i}.html")



import os
import json
from cognitive_load_pipeline import CognitiveLoadPipeline
from visualize_load import generate_html_heatmap

def main():
    # 1. 初始化 Pipeline
    pipeline = CognitiveLoadPipeline(model_type='bert', lang='zh')
    
    # 2. 指定 PDF 路徑
    pdf_path = "tutorial/knowledge/text_model/paper/2502.10378v1.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return
        
    # 3. 執行分析
    output_json = "output_analysis.json"
    print(f"Starting analysis of {pdf_path}...")
    pipeline.process_file(pdf_path, output_path=output_json, max_chars=3000)
    
    # 4. 產生熱點圖
    output_html = "heatmap_analysis.html"
    print(f"Generating heatmap to {output_html}...")
    generate_html_heatmap(output_json, output_html)
    
    print("Done!")

if __name__ == "__main__":
    main()


import json
import os

def generate_html_heatmap(json_path, output_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Cognitive Load Heatmap</title>
        <style>
            body {{ font-family: "Microsoft JhengHei", sans-serif; padding: 40px; line-height: 1.8; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; text-align: center; }}
            .heatmap-container {{ font-size: 24px; margin-top: 30px; display: flex; flex-wrap: wrap; gap: 4px; }}
            .word-box {{ 
                padding: 4px 8px; 
                border-radius: 4px; 
                position: relative; 
                cursor: help;
                transition: transform 0.2s;
            }}
            .word-box:hover {{ transform: scale(1.1); z-index: 10; }}
            .tooltip {{
                visibility: hidden;
                width: 140px;
                background-color: #333;
                color: #fff;
                text-align: center;
                border-radius: 6px;
                padding: 8px;
                position: absolute;
                z-index: 100;
                bottom: 125%;
                left: 50%;
                margin-left: -70px;
                opacity: 0;
                transition: opacity 0.3s;
                font-size: 14px;
            }}
            .word-box:hover .tooltip {{ visibility: visible; opacity: 1; }}
            .legend {{ margin-top: 40px; display: flex; align-items: center; justify-content: center; gap: 10px; font-size: 14px; color: #666; }}
            .gradient-bar {{ width: 200px; height: 10px; background: linear-gradient(to right, hsla(200, 80%, 70%, 0.8), hsla(0, 80%, 70%, 0.8)); border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>認知負荷熱點圖 (Cognitive Load Heatmap)</h1>
            <div class="heatmap-container">
                {content}
            </div>
            <div class="legend">
                <span>低負荷</span>
                <div class="gradient-bar"></div>
                <span>高負荷</span>
            </div>
        </div>
    </body>
    </html>
    """
    
    def get_color(score):
        # Blue (200) to Red (0)
        hue = 200 - (score * 200)
        return f"hsla({hue}, 80%, 70%, 0.8)"

    word_html = []
    for w in data['word_analysis']:
        color = get_color(w['load_score'])
        html_item = f"""
        <div class="word-box" style="background-color: {color};">
            {w['word']}
            <span class="tooltip">
                Score: {w['load_score']:.4f}<br>
                Surprisal: {w['surprisal']:.2f}
            </span>
        </div>
        """
        word_html.append(html_item)
    
    final_html = html_template.format(content="".join(word_html))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_html)
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    for i in range(3):
        json_file = f"/home/wei-chi/Eyetracking/output_{i}.json"
        if os.path.exists(json_file):
            generate_html_heatmap(json_file, f"/home/wei-chi/Eyetracking/heatmap_{i}.html")

import os
import shutil
import tempfile
import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# 導入先前的 Pipeline
from cognitive_load_pipeline import CognitiveLoadPipeline, DocumentLoader

app = FastAPI(title="LexiGaze Cognitive Load API")

# 獲取專案根目錄 (現在是在 weichi 目錄下，所以根目錄是上一層)
WEICHI_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(WEICHI_DIR)
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive", "analysis_results")

# 確保目錄存在
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# 設定 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 生產環境應限制為特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 Pipeline (預設使用 BERT)
pipeline_zh = CognitiveLoadPipeline(model_type='bert', lang='zh')
pipeline_en = CognitiveLoadPipeline(model_type='gpt2', lang='en')

class TextAnalysisRequest(BaseModel):
    text: str
    lang: str = "zh"
    domain: str = "auto"  # "auto" | "academic" | "general"

@app.get("/")
async def root():
    return {"message": "Welcome to LexiGaze API"}

@app.post("/api/analyze/text")
async def analyze_text(request: TextAnalysisRequest):
    p = pipeline_zh if request.lang == "zh" else pipeline_en
    try:
        result = p.run(request.text, domain=request.domain)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/pdf")
async def analyze_pdf(
    file: UploadFile = File(...),
    lang: str = Form("zh"),
    domain: str = Form("auto"),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # 建立臨時文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    
    try:
        p = pipeline_zh if lang == "zh" else pipeline_en
        
        # 準備存檔路徑
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = "".join([c if c.isalnum() else "_" for c in file.filename])
        archive_name = f"{timestamp}_{safe_filename}.json"
        archive_path = os.path.join(ARCHIVE_DIR, archive_name)
        
        # 處理 PDF 並自動存檔
        result = p.process_file(tmp_path, output_path=archive_path, domain=domain)

        result["domain"] = domain
        result["archive_file"] = archive_name
        word_count = len(result.get("word_analysis", []))
        print(f"[API] PDF 分析完成: {file.filename}, 總詞數: {word_count}")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/api/evaluate")
async def evaluate_results(analysis_result: dict, ground_truth_words: List[str]):
    """對比模型預測結果與人工標註，計算指標。

    修正點：
    1. 大小寫正規化 (carefully == Carefully)
    2. 多詞短語展開 ("paradigm shift" → "paradigm" + "shift" 各自比對)
    3. 英文模式過濾非 ASCII 詞，排除中文混排產生的 False Positive
    """
    lang = analysis_result.get("lang", "en")
    raw_predicted = analysis_result.get("high_load_words", [])

    # Step 1: 正規化 predicted_high，英文模式只保留 ASCII 詞
    if lang == "en":
        predicted_high = set(
            w.lower().strip() for w in raw_predicted
            if w.strip() and all(ord(c) < 128 for c in w.strip())
        )
    else:
        predicted_high = set(w.strip() for w in raw_predicted if w.strip())

    # Step 2: 展開 ground truth 多詞短語
    # "paradigm shift" → ["paradigm", "shift"]，各自成為獨立比對單元
    expanded_gt: List[str] = []
    original_phrases = []
    for phrase in ground_truth_words:
        phrase = phrase.strip()
        if not phrase:
            continue
        original_phrases.append(phrase)
        tokens = [t.lower().strip() for t in phrase.split() if t.strip()]
        expanded_gt.extend(tokens)
    actual_high = set(expanded_gt)

    # Step 3: 計算指標
    hits = predicted_high & actual_high
    false_positives = predicted_high - actual_high
    misses = actual_high - predicted_high

    precision = len(hits) / len(predicted_high) if predicted_high else 0
    recall = len(hits) / len(actual_high) if actual_high else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "hits": sorted(hits),
        "misses": sorted(misses),
        "false_positives": sorted(false_positives)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

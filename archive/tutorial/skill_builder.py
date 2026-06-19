import os
import asyncio
import yaml
import csv
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# 載入環境變數
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("找不到 API Key，請確認 .env 檔案中是否設定了 GEMINI_API_KEY 或 GOOGLE_API_KEY")
    exit(1)

# 初始化 Client
client = genai.Client(api_key=api_key)

def log_usage(module_name: str, section: str, usage_metadata):
    """記錄 API Token 使用量到 CSV 檔案"""
    if not usage_metadata:
        return
    
    base_dir = Path(__file__).parent.resolve()
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)
    usage_file = data_dir / "api_usage.csv"
    
    file_exists = usage_file.exists()
    
    try:
        with open(usage_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                writer.writerow(['timestamp', 'module', 'section', 'prompt_tokens', 'candidate_tokens', 'total_tokens'])
            
            prompt_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
            candidate_tokens = getattr(usage_metadata, 'candidates_token_count', 0)
            total_tokens = getattr(usage_metadata, 'total_token_count', 0)

            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                module_name,
                section,
                prompt_tokens,
                candidate_tokens,
                total_tokens
            ])
    except Exception as e:
        print(f"無法記錄使用量: {e}")

# 從環境變數讀取 MODEL_NAME，若無則預設為 gemini-2.5-flash
model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")

# 限制最高併發數為 1，強迫問題「乖乖排隊」一個一個問
sem = asyncio.Semaphore(1)

async def process_query(module_name: str, section: str, query: str, papers_content: str, max_retries: int = 3) -> tuple[str, str]:
    """非同步呼叫 Gemini API 處理單一問題 (加入排隊與防爆冷卻機制)"""
    prompt = (
        f"你現在是 {module_name} 領域的專家。請根據以下文獻：\n"
        f"{papers_content}\n\n"
        f"回答學生問題：\n{query}\n\n"
        f"請以精確、可直接輔助 Vibe Coding 的 Markdown 格式輸出，包含具體的參數、公式或演算法邏輯。"
    )
    
    # 使用 async with sem 獲取鎖，確保同一時間只有一個請求發送
    async with sem:
        for attempt in range(max_retries):
            try:
                # 使用 async 版本的 API 呼叫
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                
                # 記錄 Token 使用量
                log_usage(module_name, section, response.usage_metadata)
                
                # 成功呼叫後，強制讓程式睡 4 秒 (確保符合 15 RPM 的免費額度限制)
                await asyncio.sleep(4)
                
                return section, response.text
                
            except Exception as e:
                error_msg = str(e)
                # 如果真的不小心撞到 429 或 503，等待 30 秒再試
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "503" in error_msg:
                    print(f"[速限保護] API 需要冷卻 ({section})，等待 30 秒後進行第 {attempt + 1} 次重試...")
                    await asyncio.sleep(30)
                else:
                    print(f"API 呼叫失敗 ({module_name} - {section}): {e}")
                    return section, f"API 錯誤: {e}"
        
        return section, f"{section} 重試 {max_retries} 次後依然失敗。"

def extract_text_from_pdf(pdf_path):
    """增強版 PDF 萃取：整合顯性表格與文字流"""
    try:
        import pdfplumber
    except ImportError:
        print(f"缺少 pdfplumber 套件，請先執行 `pip install pdfplumber`")
        return ""
        
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # 1. 萃取顯性表格
                tables = page.extract_tables()
                table_texts = []
                for table in tables:
                    formatted_rows = []
                    for row in table:
                        if row and any(row):
                            cells = [str(c).replace('\n', ' ').strip() if c else "" for c in row]
                            formatted_rows.append("| " + " | ".join(cells) + " |")
                    if formatted_rows:
                        num_cols = formatted_rows[0].count('|') - 1
                        separator = "|" + "---|" * num_cols
                        table_texts.append("\n" + formatted_rows[0] + "\n" + separator + "\n" + "\n".join(formatted_rows[1:]) + "\n")
                
                # 2. 萃取純文字 (調優 tolerance 以避免文字粘連)
                extracted = page.extract_text(x_tolerance=2, y_tolerance=3)
                
                # 3. 整合
                if extracted:
                    text += extracted + "\n"
                if table_texts:
                    text += "\n" + "\n".join(table_texts) + "\n"
                text += "\n--- PAGE BREAK ---\n"
    except Exception as e:
        print(f"  [警告] PDF 解析失敗 {os.path.basename(pdf_path)}: {e}")
    return text

async def process_module(module_path: Path, template_content: str):
    """處理單一知識模組 (讀取文獻、YAML，並呼叫 API 產出 Markdown)"""
    module_name = module_path.name
    yaml_path = module_path / "questions.yaml"
    
    # 支援 'paper' 或 'papers' 目錄名稱
    papers_dir = module_path / "papers"
    if not papers_dir.exists():
        papers_dir = module_path / "paper"

    # 防呆機制：檢查 YAML 與 papers 目錄
    if not yaml_path.exists():
        print(f"跳過 {module_name}: 找不到 questions.yaml")
        return
    
    if not papers_dir.exists() or not any(papers_dir.iterdir()):
        print(f"跳過 {module_name}: {papers_dir.name}/ 目錄為空或不存在")
        return

    print(f"開始處理模組: {module_name}")

    # 讀取 YAML 檔案
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            questions = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"讀取 {yaml_path} 失敗: {e}")
        return

    # 讀取所有 paper 內容 (支援純文本、Markdown 與 PDF)
    papers_content = ""
    for file_path in papers_dir.glob("*"):
        if file_path.is_file():
            if file_path.suffix.lower() == '.pdf':
                extracted_text = extract_text_from_pdf(file_path)
                if extracted_text:
                    papers_content += f"\n--- {file_path.name} ---\n{extracted_text}\n"
            else:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        papers_content += f"\n--- {file_path.name} ---\n{f.read()}\n"
                except Exception as e:
                    print(f"無法讀取檔案 {file_path.name}: {e}")

    # 建立所有 section 的非同步任務
    tasks = [
        process_query(module_name, section, query, papers_content)
        for section, query in questions.items()
    ]

    # 等待所有 API 呼叫完成
    results = await asyncio.gather(*tasks)
    
    # 組合輸出，替換佔位符
    final_content = template_content
    for section, answer in results:
        placeholder = f"{{{section}}}"
        final_content = final_content.replace(placeholder, answer)

    # 建立 skills 目錄並儲存結果
    output_dir = Path("tutorial/skills")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"skill_{module_name}.md"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_content)
        
    print(f"成功產出 {output_path}")

async def main(target_module=None):
    base_dir = Path(__file__).parent.resolve()
    knowledge_dir = base_dir / "knowledge"
    template_path = base_dir / "default_template.md"

    # 確認基礎檔案與目錄存在
    if not template_path.exists():
        print(f"找不到 {template_path}，請確認檔案存在。")
        return

    if not knowledge_dir.exists():
        print(f"找不到 {knowledge_dir} 目錄。")
        return

    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()

    # 決定要處理的模組
    if target_module:
        modules = [target_module]
    else:
        modules = ["web_cam", "text_model", "calibration"]

    for module_name in modules:
        module_path = knowledge_dir / module_name
        if module_path.exists():
            await process_module(module_path, template_content)
        else:
            print(f"跳過 {module_name}: 目錄不存在")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Skill Builder - 知識轉譯工具")
    parser.add_argument("--module", "-m", type=str, help="指定要建置的特定模組名稱 (例如: web_cam)")
    args = parser.parse_args()

    asyncio.run(main(target_module=args.module))
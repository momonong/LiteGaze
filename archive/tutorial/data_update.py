import os
import re
import json
import hashlib
import argparse
import psycopg2
from pgvector.psycopg2 import register_vector
from tqdm import tqdm

# --- 設定路徑 ---
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
HASH_REGISTRY_FILE = "data/file_hash_registry.json"

# --- 資料庫設定 ---
DB_PARAMS = {
    "dbname": "ragdb",
    "user": "raguser",
    "password": "ragpassword",
    "host": "localhost",
    "port": "5432"
}

def get_file_hash(filepath):
    """計算檔案 MD5 Hash，實作冪等性與增量更新"""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def repair_broken_text(text):
    """【資深優化】修復 PDF 解析常見的粘連或過度分散文字"""
    if not text: return ""
    
    # 1. 修復分散的字母 (如 "A N N U A L" -> "ANNUAL")
    text = re.sub(r'(?<=[A-Z]) (?=[A-Z](?: |$))', '', text)
    
    # 2. 修復常見的粘連單字 (基於標點符號與大小寫特徵)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # 3. 移除 XBRL 標籤與 SEC 元數據噪音
    text = re.sub(r'[a-z\-]+:[A-Za-z0-9\-_]+', '', text)
    # 移除 URL
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    return text

def is_potential_header(line):
    """判斷一行是否應該被提升為 Markdown 標題 (針對學術論文優化)"""
    line = line.strip()
    if not line: return False
    # 學術論文常見標題特徵
    keywords = ["Abstract", "Introduction", "Method", "Methodology", "Results", "Discussion", "Conclusion", "References", "Literature Review"]
    if len(line) < 80 and (line.isupper() or any(k in line for k in keywords)) and not line.endswith(('.', ',', ';')):
        return True
    return False

def clean_text_pipeline(raw_text, filename=""):
    """【精華淨化邏輯】深度清洗、注入 YAML Metadata、並優化結構 (針對學術論文)"""
    if not raw_text: return ""

    # 移除 NUL 字元
    text = raw_text.replace('\x00', '')
    
    # 基礎清洗
    text = repair_broken_text(text)
    text = re.sub(r'[ \t]+', ' ', text)

    # 移除學術論文常見的雜訊
    blacklist = ["Downloaded from", "Copyright", "All rights reserved", "ISSN", "DOI:", "Published by", "Skip to main content"]
    
    lines = text.split('\n')
    cleaned_lines = []
    
    # 提取 Category
    category = filename.split('_')[0] if '_' in filename else "Academic"
    
    # 注入 YAML Frontmatter
    cleaned_lines.append("---")
    cleaned_lines.append(f"title: \"{filename.rsplit('.', 1)[0]}\"")
    cleaned_lines.append(f"category: \"{category}\"")
    cleaned_lines.append(f"source: \"{filename}\"")
    cleaned_lines.append("---\n")

    for line in lines:
        stripped = line.strip()
        if not stripped: 
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        
        # 檢查黑名單
        if any(token in stripped for token in blacklist): continue
        
        # 結構化優化：自動提升標題
        if is_potential_header(stripped):
            cleaned_lines.append(f"\n### {stripped}\n")
        else:
            cleaned_lines.append(stripped)
            
    # 移除過多重複空行
    result = '\n'.join(cleaned_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result

def extract_text_from_pdf(pdf_path):
    """增強版 PDF 萃取：整合顯性表格與文字流"""
    import pdfplumber
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

def extract_text_from_html(html_path):
    """HTML 處理：優化標題轉換與表格保留"""
    from bs4 import BeautifulSoup
    try:
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_html = f.read()
            soup = BeautifulSoup(raw_html, 'html.parser')
            
            # 移除無效區塊
            for tag in ["script", "style", "nav", "footer", "header", "aside", "form", "button", "iframe", "noscript"]:
                for s in soup.find_all(tag):
                    s.decompose()
            
            # 移除隱藏數據
            for div in soup.find_all(lambda t: t.name == 'div' and t.get('style') and ('display:none' in t.get('style').replace(' ', ''))):
                div.decompose()

            # 轉換標題
            for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                level = min(int(h.name[1]), 3) # RAG 最佳實踐：不超過 3 級
                h.replace_with(f"\n\n{'#' * level} {h.get_text(strip=True)}\n\n")
            
            # 轉換表格
            for table in soup.find_all('table'):
                rows = []
                for tr in table.find_all('tr'):
                    cells = [td.get_text(strip=True).replace('\n', ' ').replace('|', '\|') for td in tr.find_all(['th', 'td'])]
                    if any(cells):
                        rows.append("| " + " | ".join(cells) + " |")
                
                if rows:
                    num_cols = rows[0].count('|') - 1
                    separator = "|" + "---|" * num_cols
                    markdown_table = "\n\n" + rows[0] + "\n" + separator + "\n" + "\n".join(rows[1:]) + "\n\n"
                    table.replace_with(markdown_table)
            
            text = soup.get_text(separator='\n')
            return text
    except Exception as e:
        print(f"  [警告] HTML 解析失敗 {os.path.basename(html_path)}: {e}")
        return ""

def smart_chunk_text(text, chunk_size=800, overlap=100):
    """語意感知的切塊策略：增加大小以保留完整財務對比"""
    # 優先按雙換行切分
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        p = p.strip()
        if not p: continue
        
        if len(current_chunk) + len(p) <= chunk_size:
            current_chunk += p + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            if len(p) > chunk_size:
                # 段落過長，按字數切
                p_start = 0
                while p_start < len(p):
                    chunks.append(p[p_start : p_start + chunk_size].strip())
                    p_start += (chunk_size - overlap)
                current_chunk = ""
            else:
                current_chunk = p + "\n\n"
                
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def init_db():
    """初始化 PostgreSQL 與 pgvector"""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id bigserial PRIMARY KEY,
            doc_name text,
            category text,
            content text,
            embedding vector(384)
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def process_pipeline(rebuild=False):
    from sentence_transformers import SentenceTransformer
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    registry = {}
    
    if rebuild:
        print("🧹 啟動全量重建 (--rebuild)...")
        if os.path.exists(HASH_REGISTRY_FILE):
            os.remove(HASH_REGISTRY_FILE)
        for f in os.listdir(PROCESSED_DIR):
            os.remove(os.path.join(PROCESSED_DIR, f))
            
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS documents;")
        conn.commit()
        cur.close()
        conn.close()
    else:
        if os.path.exists(HASH_REGISTRY_FILE):
            with open(HASH_REGISTRY_FILE, 'r') as f:
                registry = json.load(f)

    init_db()
    
    print("🧠 載入 Embedding 模型...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    conn = psycopg2.connect(**DB_PARAMS)
    register_vector(conn)
    cur = conn.cursor()

    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith(('.pdf', '.html', '.htm'))]
    print(f"📦 準備處理 {len(raw_files)} 份高價值檔案...\n" + "-"*40)

    for filename in raw_files:
        filepath = os.path.join(RAW_DIR, filename)
        current_hash = get_file_hash(filepath)
        
        if not rebuild and registry.get(filename) == current_hash:
            continue
            
        print(f"⚙️ 優化處理中: {filename[:50]}...")
        
        # 1. 萃取
        raw_text = extract_text_from_pdf(filepath) if filename.endswith('.pdf') else extract_text_from_html(filepath)
        
        # 2. 淨化
        clean_text = clean_text_pipeline(raw_text, filename=filename)
        
        if len(clean_text) < 150:
            print(f"  [略過] 有效內文不足。")
            continue
            
        # 儲存高品質純文字版 (符合 HW3 規範)
        txt_filename = filename.rsplit('.', 1)[0] + ".txt"
        with open(os.path.join(PROCESSED_DIR, txt_filename), 'w', encoding='utf-8') as f:
            f.write(clean_text)
            
        # 3. 切塊
        chunks = smart_chunk_text(clean_text)
        
        # 4. Metadata
        category = filename.split('_')[0] if '_' in filename else "Unknown"
        
        # 5. 寫入資料庫
        for chunk in chunks:
            vector = model.encode(chunk).tolist()
            cur.execute(
                "INSERT INTO documents (doc_name, category, content, embedding) VALUES (%s, %s, %s, %s)",
                (filename, category, chunk, vector)
            )
            
        registry[filename] = current_hash
        conn.commit()

    with open(HASH_REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=4)
        
    cur.close()
    conn.close()
    print("\n✅ 索引優化完成！結構化與文字密度已顯著提升。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="強制清空並重建")
    args = parser.parse_args()
    process_pipeline(rebuild=args.rebuild)

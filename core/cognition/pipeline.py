"""
認知負荷 Inference Pipeline (v10.2 - F1 Optimization)

優化重點：
1. [效能] 批次 Mask 推理 (Batch Masking)：大幅減少 GPU 呼叫次數，速度提升 3-5 倍。
2. [效能] 硬體加速：啟用 torch.inference_mode 與 FP16 (Autocast) 運算。
3. [準確] 認知動量 (Cognitive Momentum)：模擬讀者心理壓力的累積與衰減效果。
4. [準確] 混合門檻 (Hybrid Thresholding)：結合絕對與相對門檻，減少簡單文本的誤報。
5. [準確] 增強型依賴負載：針對英文優化語法樹整合成本計算。
6. [修正] 連字號複合詞重新合併 (Word Re-aggregation)：將 neuro + - + symbolic 合回 neuro-symbolic。
7. [修正] 動態絕對門檻 (Dynamic Threshold)：以 mean+σ 取代硬編碼 0.48，適應各種文本分佈。
8. [F1修正] p_score 納入計分乘數：詞性重要性現在實際影響 load_score（虛詞壓低，實詞保持）。
9. [F1修正] Interaction 語意修正：s_score × f_score（罕見且驚訝 = 技術術語），非原來的反向公式。
10.[F1修正] 門檻百分位數 80→75：在不犧牲精確率的前提下提升召回率。
11.[修正] process_file cognitive_memory：修正傳遞純 float 導致句間動量 crash 的 bug。
12.[v8 速度] 關閉 spaCy 不用元件 (lemmatizer/ner) + process_file 改 chunked scoring + 文件級 threshold/boost
"""

import json
import math
import re
import os
if os.name != 'nt' and os.environ.get("HF_HOME"):
    _hf_home = os.environ["HF_HOME"]
    if ":" in _hf_home or _hf_home.startswith("D:") or _hf_home.startswith("C:"):
        del os.environ["HF_HOME"]
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from wordfreq import zipf_frequency
from opencc import OpenCC
import jieba
import jieba.posseg as pseg

# 硬體效能優化配置
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

try:
    import spacy
except ImportError:
    spacy = None

t2s = OpenCC('t2s')

@dataclass
class WordResult:
    word: str
    pos: str
    position: int
    surprisal: float
    entropy: float
    dependency_load: float
    zipf_score: float
    word_length: int
    pos_score: float
    load_level: str
    load_score: float
    aoa_score: float = 0.0      # Kuperman Age-of-Acquisition（標準化至 0-1）
    renyi_entropy: float = 0.0  # Rényi entropy α=0.5（Pimentel et al. 2023）

class DocumentLoader:
    @staticmethod
    def load(file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf': return DocumentLoader._load_pdf(file_path)
        elif ext in ['.md', '.txt']: return DocumentLoader._load_text(file_path)
        else: raise ValueError(f"不支援格式: {ext}")

    @staticmethod
    def _load_pdf(file_path: str) -> str:
        try:
            import fitz
            text = ""
            with fitz.open(file_path) as doc:
                for page_num, page in enumerate(doc):
                    blocks = page.get_text("blocks", sort=True)
                    page_text = " ".join([b[4].replace("\n", " ") for b in blocks])
                    text += f"\n{page_text}\n"
            return text
        except Exception as e:
            return f"PDF 讀取失敗: {str(e)}"

    @staticmethod
    def _load_text(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f: return f.read()

class LanguageModelCalculator:
    MODELS = {
        'gpt2': {'zh': "uer/gpt2-chinese-cluecorpussmall", 'en': "gpt2"},
        'gpt2-medium': {'zh': "uer/gpt2-chinese-cluecorpussmall", 'en': "gpt2-medium"},
        'bert': {'zh': "bert-base-chinese", 'en': "bert-base-uncased", 'nl': "bert-base-multilingual-cased"}
    }

    def __init__(self, model_type: str = 'bert', lang: str = 'zh', batch_size: int = 32):
        from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
        self.model_type = model_type
        self.lang = lang
        self.batch_size = batch_size
        model_name = self.MODELS[model_type][lang]

        print(f"[系統] 正在初始化 {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_type.startswith('gpt2'):
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(model_name, attn_implementation="eager")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        self.use_fp16 = (self.device == "cuda")

    @torch.inference_mode()
    def compute(self, words: List[str]) -> Dict[str, List[float]]:
        if not words: return {"surprisals": [], "attentions": [], "entropies": []}

        # 根據顯存自動調整
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=self.use_fp16):
            if self.model_type.startswith('gpt2'):
                return self._compute_gpt(words)
            else:
                return self._compute_bert(words)

    def _compute_bert(self, words: List[str]) -> Dict[str, List[float]]:
        encoding = self.tokenizer(words, is_split_into_words=True, return_tensors="pt").to(self.device)
        input_ids = encoding["input_ids"][0]
        word_ids = encoding.word_ids()

        # 1. 取得 Attention (一次傳遞)
        outputs = self.model(**encoding, output_attentions=True)
        attns = outputs.attentions[-4:]
        avg_attn = torch.stack(attns).mean(dim=0)[0].mean(dim=0)
        centrality = avg_attn.sum(dim=0).cpu().tolist()

        # 2. 準備批次 Masking 推理
        surprisals = [0.0] * len(words)
        entropies = [0.0] * len(words)
        attentions = [0.0] * len(words)

        word_to_indices = {}
        for i, w_idx in enumerate(word_ids):
            if w_idx is not None:
                word_to_indices.setdefault(w_idx, []).append(i)

        batch_inputs = []
        batch_targets = []
        batch_w_idxs = []

        for w_idx, indices in word_to_indices.items():
            # 排除純標點符號
            if not re.search(r'[一-鿿\d\w]', words[w_idx]): continue

            masked_input = input_ids.clone()
            for idx in indices: masked_input[idx] = self.tokenizer.mask_token_id

            batch_inputs.append(masked_input)
            batch_targets.append((indices, input_ids[indices]))
            batch_w_idxs.append(w_idx)

            if len(batch_inputs) >= self.batch_size:
                self._run_bert_batch(batch_inputs, batch_targets, batch_w_idxs, surprisals, entropies)
                batch_inputs, batch_targets, batch_w_idxs = [], [], []

        if batch_inputs:
            self._run_bert_batch(batch_inputs, batch_targets, batch_w_idxs, surprisals, entropies)

        # 映射 Attention
        for w_idx, indices in word_to_indices.items():
            if indices:
                attentions[w_idx] = sum(centrality[i] for i in indices) / len(indices)

        return {"surprisals": surprisals, "entropies": entropies,
                "attentions": attentions, "renyi_entropies": [0.0] * len(words)}

    def _run_bert_batch(self, batch_inputs, batch_targets, batch_w_idxs, surprisals, entropies):
        inputs_tensor = torch.stack(batch_inputs).to(self.device)
        outputs = self.model(inputs_tensor)

        for b in range(len(batch_inputs)):
            idxs, tids = batch_targets[b]
            logits = outputs.logits[b, idxs, :]
            lps = F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)

            # 熵 (衡量預測不確定性)
            ent = -(probs * lps).sum(dim=-1).mean().item()
            # 驚訝度 (正確 Token 的負對數機率)
            word_surp = sum(-lps[tidx, tid].item() for tidx, tid in enumerate(tids))

            target_w_idx = batch_w_idxs[b]
            surprisals[target_w_idx] = word_surp
            entropies[target_w_idx] = ent

    def _compute_gpt(self, words: List[str]) -> Dict[str, List[float]]:
        # GPT-2 position embedding 上限 1024；超長時分段推理再合併
        _MAX_TOKENS = 1000
        encoding = self.tokenizer(words, is_split_into_words=True, return_tensors="pt").to(self.device)
        input_ids = encoding["input_ids"]
        if input_ids.shape[1] > _MAX_TOKENS:
            return self._compute_gpt_chunked(words, _MAX_TOKENS)

        word_ids = encoding.word_ids()

        outputs = self.model(input_ids)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        lps = F.log_softmax(shift_logits, dim=-1)
        probs = F.softmax(shift_logits, dim=-1)
        token_ents = -(probs * lps).sum(dim=-1)
        # Rényi entropy α=0.5: H_0.5 = 2·log(Σ√P)  (Pimentel et al. 2023)
        token_renyi = 2.0 * torch.log(probs.sqrt().sum(dim=-1).clamp(min=1e-10))
        target_lps = lps.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        surprisals = [0.0] * len(words)
        entropies = [0.0] * len(words)
        renyi_entropies = [0.0] * len(words)

        for i in range(target_lps.size(1)):
            w_idx = word_ids[i+1]
            if w_idx is not None:
                surprisals[w_idx] += -target_lps[0, i].item()
                entropies[w_idx] += token_ents[0, i].item()
                renyi_entropies[w_idx] += token_renyi[0, i].item()

        return {"surprisals": surprisals, "entropies": entropies,
                "attentions": [0.5] * len(words), "renyi_entropies": renyi_entropies}

    def _compute_gpt_chunked(self, words: List[str], max_tokens: int) -> Dict[str, List[float]]:
        """將過長詞列切成多段 GPT 推理，surprisal/entropy 加總回對應詞索引。"""
        surprisals = [0.0] * len(words)
        entropies = [0.0] * len(words)
        start = 0
        while start < len(words):
            # 二分搜尋每段可塞進多少詞（token 數非線性，保守從 200 試起）
            lo, hi = 1, min(len(words) - start, 350)
            best = 1
            while lo <= hi:
                mid = (lo + hi) // 2
                n_tok = self.tokenizer(words[start:start + mid], is_split_into_words=True)["input_ids"]
                if len(n_tok) <= max_tokens:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            chunk_words = words[start:start + best]
            sub = self._compute_gpt(chunk_words)
            for i, (s, e) in enumerate(zip(sub["surprisals"], sub["entropies"])):
                surprisals[start + i] += s
                entropies[start + i] += e
            start += best
        return {"surprisals": surprisals, "entropies": entropies, "attentions": [0.5] * len(words)}

class CognitiveLoadPipeline:
    # 根據語言學研究調整的詞性權重
    POS_WEIGHTS = {
        'n': 1.0, 'v': 1.0, 'a': 0.9, 'd': 0.8, 't': 0.8, 'm': 0.4, 'p': 0.2, 'c': 0.1, 'u': 0.05, 'x': 0.0,
        'NOUN': 1.0, 'VERB': 1.0, 'ADJ': 0.9, 'ADV': 0.8, 'PROPN': 1.0, 'PRON': 0.4, 'ADP': 0.1, 'CCONJ': 0.1, 'SCONJ': 0.1, 'DET': 0.1, 'PUNCT': 0.0
    }

    # AoA 標準化常數（Kuperman 2012：約 2~15 歲）
    _AOA_MIN   = 2.0
    _AOA_RANGE = 13.0
    _AOA_DEFAULT = 10.0  # OOV（技術術語、專有名詞）預設中高難度

    def __init__(self, model_type: str = 'bert', lang: str = 'zh'):
        self.lang = lang
        self.model_type = model_type
        self.calculator = LanguageModelCalculator(model_type, lang)
        self.nlp = None
        if lang == 'en' and spacy:
            try:
                # 啟用 NER 以過濾人名/地名/組織等專有實體（避免被誤判為高認知負荷詞）
                # 仍停用 lemmatizer（pipeline 用不到，省時）
                self.nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
            except:
                print("[警告] 未找到 en_core_web_sm，改用基本分詞")
        elif lang == 'nl' and spacy:
            try:
                self.nlp = spacy.load("nl_core_news_sm", disable=["lemmatizer"])
            except:
                print("[警告] 未找到 nl_core_news_sm，改用基本分詞")

        # Kuperman AoA 詞典（英文用；中文忽略）
        self.aoa_dict = self._load_aoa() if lang == 'en' else {}
        # 評分後端：優先 XGBoost（Opt3），fallback Ridge
        self._ridge     = self._load_ridge_model()
        self._xgb_model = self._load_xgb_model() if lang == 'en' else None

        # 領域詞庫增強 (中文)
        if lang == 'zh':
            for word in ["二十四時", "胡衛東", "國手", "江蘇隊", "出國深造"]: jieba.add_word(word)

    def _load_aoa(self) -> dict:
        """讀入 Kuperman AoA CSV，回傳 {word_lower: aoa_mean} 字典。"""
        import csv
        aoa_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "GECO_data", "AoA_Kuperman.csv"
        )
        if not os.path.exists(aoa_path):
            print(f"[警告] 未找到 AoA 詞典 ({aoa_path})，AoA 特徵停用")
            return {}
        aoa = {}
        with open(aoa_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    aoa[row["Word"].lower()] = float(row["Rating.Mean"])
                except (ValueError, KeyError):
                    pass
        print(f"[AoA] 載入 {len(aoa):,} 詞")
        return aoa

    def _load_ridge_model(self) -> dict | None:
        """讀入 ridge_model.json，回傳 model dict 或 None。"""
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ridge_model.json")
        if not os.path.exists(model_path):
            return None
        with open(model_path, encoding="utf-8") as f:
            m = json.load(f)
        self._ridge_coef  = np.array(m["coef"])
        self._ridge_icept = m["intercept"]
        self._ridge_mu    = np.array(m["scaler_mean"])
        self._ridge_std   = np.array(m["scaler_std"])
        self._ridge_feats = m["features"]
        print(f"[Ridge v6] 載入模型（Val R^2={m.get('r2_val', '?'):.3f}，α={m.get('alpha','?')}）")
        return m

    def _load_xgb_model(self):
        """載入 xgb_model.json（若存在）。"""
        try:
            import xgboost as xgb
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xgb_model.json")
            if not os.path.exists(path):
                return None
            booster = xgb.Booster()
            booster.load_model(path)
            print("[XGBoost] 載入模型（Opt3）")
            return booster
        except Exception:
            return None

    def _apply_model(self, results: list) -> None:
        """XGBoost（優先）或 Ridge 重算 load_score，再 min-max 正規化至 0-1。"""
        if self._xgb_model is not None:
            self._apply_xgb(results)
        elif self._ridge:
            self._apply_ridge(results)

    def _apply_xgb(self, results: list) -> None:
        import xgboost as xgb
        XGB_FEATS = ["surprisal", "renyi_entropy", "aoa_score",
                     "word_length", "zipf_score", "pos_score", "dependency_load"]
        rows = []
        for r in results:
            rows.append([r.surprisal, r.renyi_entropy, r.aoa_score,
                         r.word_length, r.zipf_score, r.pos_score, r.dependency_load])
        dm = xgb.DMatrix(np.array(rows, dtype=float), feature_names=XGB_FEATS)
        raw_preds = self._xgb_model.predict(dm).tolist()
        mn, mx = min(raw_preds), max(raw_preds)
        span = mx - mn if mx > mn else 1.0
        for r, p in zip(results, raw_preds):
            scaled = (p - mn) / span
            r.load_score = round(min(scaled * r.pos_score + (1 - r.pos_score) * 0.0, 1.0), 4)

    def _apply_ridge(self, results: list) -> None:
        """用 ridge model 重算每個詞的 load_score（預測 TRT，再 min-max 正規化到 0-1）。"""
        raw_preds = []
        for r in results:
            feat = np.array([
                r.surprisal, r.entropy, r.aoa_score,
                r.word_length, r.zipf_score, r.pos_score
            ], dtype=float)
            feat_s = (feat - self._ridge_mu) / self._ridge_std
            pred   = float(np.dot(self._ridge_coef, feat_s) + self._ridge_icept)
            raw_preds.append(pred)
        mn, mx = min(raw_preds), max(raw_preds)
        span = mx - mn if mx > mn else 1.0
        for r, p in zip(results, raw_preds):
            scaled = (p - mn) / span
            r.load_score = round(min(scaled * r.pos_score + (1 - r.pos_score) * 0.0, 1.0), 4)

    def _aoa_score(self, word: str) -> float:
        """將 AoA 習得年齡標準化至 [0, 1]（越晚習得 → 越高）。"""
        raw = self.aoa_dict.get(word.lower(), self._AOA_DEFAULT)
        return max(0.0, min(1.0, (raw - self._AOA_MIN) / self._AOA_RANGE))

    def _detect_domain(self, words: list, pos_tags: list) -> str:
        """根據 content words 平均詞頻判斷文字領域。
        學術文本使用罕見專業詞彙（低 zipf）→ 回傳 'academic'；
        小說/通用文本使用日常詞彙（高 zipf）→ 回傳 'general'。
        雙重條件：avg_zipf < 4.8 AND avg_len > 7.5 → academic。"""
        academic_pos = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'n', 'v', 'a', 'd'}
        cw = [
            w for w, pos in zip(words, pos_tags)
            if pos in academic_pos and re.search(r'[a-zA-Z]', w) and len(w) > 2
        ]
        if len(cw) < 4:
            return "general"
        avg_zipf = float(np.mean([zipf_frequency(w, self.lang) for w in cw]))
        avg_len  = float(np.mean([len(w) for w in cw]))
        if avg_zipf < 4.8 and avg_len > 7.5:
            return "academic"
        return "general"

    def _resolve_domain(self, words: list, pos_tags: list, domain: str) -> str:
        """根據 domain 參數決定使用哪個 domain；'auto' 觸發自動偵測。"""
        if domain != "auto":
            return domain
        return self._detect_domain(words, pos_tags)

    def _chunk_text(self, text: str, max_words: int = 400) -> List[str]:
        """將長文本切成不超過 max_words 詞的區塊（在句末標點處切割）。
        中文以字元數計，英文以空白分詞計。預設 400 詞，安全餘量避免超過模型 max sequence length。"""
        if self.lang == 'zh':
            sentences = [s for s in re.split(r'(?<=[。！？；\n])', text) if s.strip()]
            joiner = ""
            def size(s): return len(s)
        else:
            # 不依賴「句點後大寫」；小說 PDF 常有 "word. she" 等格式
            raw = re.split(r'(?<=[.!?])\s+', text)
            sentences = []
            for seg in raw:
                lines = [l.strip() for l in seg.split('\n') if l.strip()]
                sentences.extend(lines if lines else [seg])
            joiner = " "
            def size(s): return len(s.split())

        def _split_by_word_count(segment: str, limit: int) -> List[str]:
            """單段超過 limit 時硬切（避免整章塞進一個 GPT chunk）。"""
            if self.lang == 'zh':
                return [segment[i:i + limit] for i in range(0, len(segment), limit)] or [segment]
            words = segment.split()
            if len(words) <= limit:
                return [segment]
            return [' '.join(words[i:i + limit]) for i in range(0, len(words), limit)]

        chunks: List[str] = []
        current: List[str] = []
        current_count = 0
        for s in sentences:
            for piece in _split_by_word_count(s, max_words):
                sc = size(piece)
                if current_count + sc > max_words and current:
                    chunks.append(joiner.join(current))
                    current = [piece]
                    current_count = sc
                else:
                    current.append(piece)
                    current_count += sc
        if current:
            chunks.append(joiner.join(current))
        return chunks if chunks else _split_by_word_count(text, max_words)

    def _score_chunk(self, text: str, cognitive_memory: List[float]) -> Tuple[List[WordResult], List[str], List[str]]:
        """執行 spaCy + GPT/BERT 推理 + 啟發式評分 + 連字號合併（per-chunk）。
        回傳 (results, words, pos_tags)。不做 domain 偵測、ridge、threshold、ADJ gate、boost；
        這些由 _finalize_load_levels 在 document 層級統一處理。"""
        # 1. 語言學預處理
        if self.lang == 'zh':
            pairs = list(pseg.cut(text))
            words, pos_tags = [p.word for p in pairs], [p.flag for p in pairs]
            dep_loads, tree_depths, last_v = [], [], -1
            for idx, pos in enumerate(pos_tags):
                dist = (idx - last_v) if last_v != -1 else 5
                dep_loads.append(min(dist * 0.1, 1.0))
                tree_depths.append(1 if pos.startswith('v') else 2)
                if pos and pos.startswith('v'): last_v = idx
        else:
            doc = self.nlp(text) if self.nlp else None
            if doc:
                words = [t.text for t in doc]
                pos_tags = [t.pos_ for t in doc]
                dep_loads = [
                    min(len(list(t.children)) * 0.15, 1.0) if t.dep_ == 'ROOT'
                    else min(abs(t.i - t.head.i) * 0.15, 1.0)
                    for t in doc
                ]
                def get_depth(token):
                    d = 0
                    while token.head != token:
                        d += 1
                        token = token.head
                    return d
                tree_depths = [min(get_depth(t) / 5.0, 1.0) for t in doc]
            else:
                words = re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)
                pos_tags = ["NOUN"] * len(words)
                dep_loads = [0.1] * len(words)
                tree_depths = [0.2] * len(words)

        # 2. 深度學習模型推理
        metrics = self.calculator.compute(words)
        surprisals = metrics["surprisals"]
        entropies  = metrics["entropies"]
        attentions = metrics["attentions"]
        renyi_entropies = metrics.get("renyi_entropies", [0.0] * len(words))

        # 3. 啟發式計分
        results: List[WordResult] = []
        mem_n1, mem_n2 = cognitive_memory
        max_surp  = max([min(s, 15.0) for s in surprisals] + [5.0])
        max_ent   = max(entropies + [1.0])
        max_renyi = max(renyi_entropies + [1.0])

        for i, (word, surp, ent, re_ent, _, pos, dl, td) in enumerate(
            zip(words, surprisals, entropies, renyi_entropies, attentions, pos_tags, dep_loads, tree_depths)
        ):
            if not re.search(r'[一-鿿\d\w]', word):
                results.append(WordResult(word, pos, i, 0, 0, 0, 0, len(word), 0, "low", 0.0, 0.0, 0.0))
                continue
            # English mode: skip tokens containing Chinese chars (PDF contamination)
            if self.lang == 'en' and re.search(r'[一-鿿＀-￯]', word):
                results.append(WordResult(word, pos, i, 0, 0, 0, 0, len(word), 0, "low", 0.0, 0.0, 0.0))
                continue
            # English mode: filter all-caps short acronyms (AI, NLP, GPU)
            if self.lang == 'en' and word.isupper() and word.isalpha() and len(word) <= 5:
                results.append(WordResult(word, pos, i, 0, 0, 0, 0, len(word), 0, "low", 0.0, 0.0, 0.0))
                continue

            zipf_val = zipf_frequency(word, self.lang)
            s_score  = min(surp, 15.0) / max_surp
            e_score  = ent / max_ent
            re_score = re_ent / max_renyi
            f_score  = max(0, (7.0 - zipf_val) / 7.0)
            p_score  = self.POS_WEIGHTS.get(pos, 0.5)
            a_score  = self._aoa_score(word)
            interaction = s_score * f_score

            # Opt1: POS-gate dep_load (Rathi 2021 — integration cost significant only for NOUN/VERB)
            dl_gated = dl if pos in ['NOUN', 'VERB', 'PROPN', 'n', 'v'] else 0.0

            # Opt2: Rényi entropy added to weight mix; dep_load uses POS-gated value
            # features: [s_score, e_score, re_score, interaction, dl_gated, td, a_score, f_score]
            if pos in ['NOUN', 'VERB', 'PROPN', 'n', 'v']:
                w = [0.20, 0.04, 0.05, 0.16, 0.09, 0.07, 0.20, 0.19]
            else:
                w = [0.22, 0.05, 0.05, 0.13, 0.00, 0.02, 0.24, 0.29]

            raw_total = sum(wi * si for wi, si in zip(w, [s_score, e_score, re_score, interaction, dl_gated, td, a_score, f_score]))
            base_total = p_score * raw_total
            final_score = base_total + (0.12 * mem_n1) + (0.05 * mem_n2)
            final_score = round(min(final_score, 1.0), 4)
            mem_n2 = mem_n1
            mem_n1 = final_score

            results.append(WordResult(
                word, pos, i, round(surp, 3), round(ent, 3), round(dl_gated, 3),
                round(zipf_val, 2), len(word), round(p_score, 2), "low", final_score,
                round(a_score, 3), round(re_ent, 3)
            ))

        # 4. 合併連字號複合詞（英文專屬）
        if self.lang == 'en':
            results = self._reaggregate_hyphenated(results)

        return results, words, pos_tags

    def _finalize_load_levels(self, results: List[WordResult]) -> None:
        """文件級 threshold + ADJ 精確率關卡 + 搭配詞 boost（in-place 修改 results）。
        從 run() 與 process_file() 統一呼叫，確保兩條路徑的 high/medium/low 判定一致。"""
        scores = [r.load_score for r in results if r.load_score > 0.01]
        threshold = 0.18  # 預設
        if scores:
            rel_thresh = np.percentile(scores, 70)
            floor = 0.18 if self.lang == 'en' else 0.30
            threshold = max(rel_thresh, floor)
            for r in results:
                if r.load_score >= threshold:
                    r.load_level = "high"
                elif r.load_score >= threshold * 0.65:
                    r.load_level = "medium"
                else:
                    r.load_level = "low"

        if self.lang == 'en':
            # v8: ADJ 精確率關卡：高 zipf + AoA 中等以下的形容詞從 high 降回 medium
            for r in results:
                if (r.load_level == 'high'
                        and r.pos == 'ADJ'
                        and r.zipf_score > 4.5
                        and r.aoa_score < 0.80):
                    r.load_level = 'medium'
            # v8: 搭配詞 boost（ADJ 關卡後執行；加上近接條件防止連帶誤拉）
            for i in range(1, len(results)):
                prev, curr = results[i - 1], results[i]
                if (prev.load_level == 'high'
                        and curr.load_level != 'high'
                        and prev.pos_score >= 0.8 and curr.pos_score >= 0.8
                        and curr.zipf_score <= 5.0
                        and prev.load_score > curr.load_score * 1.30
                        and curr.load_score >= threshold * 0.82):
                    curr.load_level = 'high'

    def run(self, text: str, cognitive_memory: List[float] = None, domain: str = "auto") -> dict:
        """單次推理：適用於單一句子或短段落。長文件請改用 process_file（自動分塊）。"""
        start_time = time.time()
        if cognitive_memory is None:
            cognitive_memory = [0.0, 0.0]

        results, words, pos_tags = self._score_chunk(text, cognitive_memory)

        active_domain = "general"
        if self.lang in ['en', 'nl'] and self._ridge:
            active_domain = self._resolve_domain(words, pos_tags, domain)
            if active_domain != "academic":
                self._apply_model(results)

        self._finalize_load_levels(results)

        process_time = time.time() - start_time
        return {
            "model": self.model_type,
            "lang": self.lang,
            "domain": active_domain,
            "process_time_ms": int(process_time * 1000),
            "high_load_words": [
                r.word for r in results
                if r.load_level == "high"
            ],
            "word_analysis": [asdict(r) for r in results]
        }

    def _reaggregate_hyphenated(self, results: List[WordResult]) -> List[WordResult]:
        """合併被 spaCy 拆分的連字號複合詞: [word, '-', word] → [word-word]。
        load_score 取最大值，surprisal 取加總，position 保留首詞。"""
        merged: List[WordResult] = []
        i = 0
        while i < len(results):
            r = results[i]
            if (i + 2 < len(results)
                    and results[i + 1].word == '-'
                    and re.match(r'\w', r.word)
                    and re.match(r'\w', results[i + 2].word)):
                r2 = results[i + 2]
                compound = r.word + '-' + r2.word
                merged.append(WordResult(
                    word=compound,
                    pos=r.pos,
                    position=r.position,
                    surprisal=round(r.surprisal + r2.surprisal, 3),
                    entropy=round(max(r.entropy, r2.entropy), 3),
                    dependency_load=round(max(r.dependency_load, r2.dependency_load), 3),
                    zipf_score=round(min(r.zipf_score, r2.zipf_score), 2),
                    word_length=len(compound),
                    pos_score=r.pos_score,
                    load_level='low',
                    load_score=max(r.load_score, r2.load_score),
                    aoa_score=max(r.aoa_score, r2.aoa_score),
                    renyi_entropy=max(r.renyi_entropy, r2.renyi_entropy),
                ))
                i += 3
            else:
                merged.append(r)
                i += 1
        return merged

    def process_file(self, file_path: str, output_path: Optional[str] = None, domain: str = "auto"):
        """v8: chunked scoring — 將長文件分塊跑 spaCy/GPT-2，但在 document 層級統一做
        domain 偵測、ridge、threshold、ADJ gate、boost。比原本「逐句呼叫 run()」快很多，
        且修正了原本逐句 boost/gate 被全文 threshold 重設覆蓋的 bug。"""
        text = DocumentLoader.load(file_path)
        chunks = self._chunk_text(text, max_words=400)

        print(f"[任務] 開始分析檔案: {os.path.basename(file_path)}  ({len(chunks)} 區塊)")

        all_results: List[WordResult] = []
        all_words: List[str] = []
        all_pos: List[str] = []
        mem_window = [0.0, 0.0]

        for idx, chunk in enumerate(chunks):
            chunk_results, chunk_words, chunk_pos = self._score_chunk(chunk, mem_window)
            all_results.extend(chunk_results)
            all_words.extend(chunk_words)
            all_pos.extend(chunk_pos)
            if chunk_results:
                mem_window = [chunk_results[-1].load_score, mem_window[0]]
            if (idx + 1) % 5 == 0 or idx + 1 == len(chunks):
                print(f"[進度] 已完成 {idx + 1}/{len(chunks)} 區塊...")

        # 文件級 domain + ridge
        active_domain = "general"
        if self.lang in ['en', 'nl'] and self._ridge:
            active_domain = self._resolve_domain(all_words, all_pos, domain)
            if active_domain != "academic":
                self._apply_model(all_results)

        # 文件級 threshold + ADJ gate + boost（與 run() 共用，行為一致）
        self._finalize_load_levels(all_results)

        final_result = {
            "model": self.model_type,
            "lang": self.lang,
            "domain": active_domain,
            "high_load_words": [
                r.word for r in all_results
                if r.load_level == 'high'
            ],
            "word_analysis": [asdict(r) for r in all_results]
        }

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            print(f"[完成] 結果已存儲至: {output_path}")

        return final_result

if __name__ == "__main__":
    p = CognitiveLoadPipeline(model_type='bert', lang='zh')
    test_text = "量子糾纏描述了兩個粒子之間的非局域性關聯，這在經典物理中是難以想像的。"
    result = p.run(test_text)
    print(f"分析耗時: {result['process_time_ms']}ms")
    print(f"高負荷詞匯: {result['high_load_words']}")

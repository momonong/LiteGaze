"""
認知負荷 Inference Pipeline (v9.2 - Final Alignment & English Fix)

優化點：
1. 修正 BERT Surprisal 計算：確保 Token 對齊邏輯在英文模式下完全正確。
2. 自動門檻調整：針對不同語言的得分分佈自動調整門檻。
3. 移除冗餘規則：全模型驅動，利用 spaCy 進行深層語言分析。
"""

import json
import math
import re
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

import torch
import torch.nn.functional as F
import numpy as np
from wordfreq import zipf_frequency
from opencc import OpenCC
import jieba
import jieba.posseg as pseg

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
                print(f"[PDF 讀取] 檔案: {file_path}, 總頁數: {len(doc)}")
                for page_num, page in enumerate(doc):
                    # 使用 blocks 模式並手動拼接，確保空白字元被保留
                    blocks = page.get_text("blocks", sort=True)
                    page_text = ""
                    for b in blocks:
                        page_text += b[4].replace("\n", " ") + " "
                    text += f"\n[第 {page_num + 1} 頁]\n" + page_text + "\n"
            print(f"[PDF 讀取] 提取完成，總字數: {len(text)}")
            return text
        except Exception as e:
            print(f"[PDF 錯誤] {str(e)}")
            return f"錯誤: {str(e)}"

    @staticmethod
    def _load_text(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f: return f.read()

class LanguageModelCalculator:
    MODELS = {
        'gpt2': {'zh': "uer/gpt2-chinese-cluecorpussmall", 'en': "gpt2"},
        'bert': {'zh': "bert-base-chinese", 'en': "bert-base-uncased"}
    }

    def __init__(self, model_type: str = 'bert', lang: str = 'zh'):
        from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
        self.model_type = model_type
        self.lang = lang
        model_name = self.MODELS[model_type][lang]
        print(f"[載入模型] {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_type == 'gpt2':
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(model_name, attn_implementation="eager")
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def compute(self, words: List[str]) -> Dict[str, List[float]]:
        if not words: return {"surprisals": [0.0], "attentions": [0.0], "entropies": [0.0]}
        if self.model_type == 'gpt2': 
            metrics = self._compute_gpt(words)
        else:
            metrics = self._compute_bert(words)
        return metrics

    def _compute_bert(self, words: List[str]) -> Dict[str, List[float]]:
        # 使用更穩健的對齊方式
        encoding = self.tokenizer(words, is_split_into_words=True, return_tensors="pt").to(self.device)
        input_ids = encoding["input_ids"][0]
        word_ids = encoding.word_ids()
        
        # 計算 Attention
        with torch.no_grad():
            outputs = self.model(**encoding, output_attentions=True)
        
        # Attention Centrality (最後四層)
        attns = outputs.attentions[-4:] 
        avg_attn = torch.stack(attns).mean(dim=0)[0].mean(dim=0)
        centrality = avg_attn.sum(dim=0).cpu().tolist()

        # 計算 Surprisal (逐個單詞 Mask)
        surprisals = [0.0] * len(words)
        entropies = [0.0] * len(words)
        attentions = [0.0] * len(words)
        
        # 建立 Word -> Token mapping
        word_to_indices = {}
        for i, w_idx in enumerate(word_ids):
            if w_idx is not None:
                if w_idx not in word_to_indices: word_to_indices[w_idx] = []
                word_to_indices[w_idx].append(i)
        
        # 批次 Masking
        batch_inputs = []
        batch_targets = []
        batch_w_idxs = []
        
        for w_idx, indices in word_to_indices.items():
            if not re.search(r'[\u4e00-\u9fff\d\w]', words[w_idx]): continue
            
            masked_input = input_ids.clone()
            for idx in indices: masked_input[idx] = self.tokenizer.mask_token_id
            
            batch_inputs.append(masked_input)
            batch_targets.append((indices, input_ids[indices]))
            batch_w_idxs.append(w_idx)
            
            if len(batch_inputs) >= 16:
                self._process_bert_batch(batch_inputs, batch_targets, batch_w_idxs, surprisals, entropies)
                batch_inputs, batch_targets, batch_w_idxs = [], [], []

        # 處理最後一輪剩餘的單詞
        if batch_inputs:
            self._process_bert_batch(batch_inputs, batch_targets, batch_w_idxs, surprisals, entropies)

        # 映射 Attention
        for w_idx, indices in word_to_indices.items():
            if indices:
                attentions[w_idx] = sum(centrality[i] for i in indices) / len(indices)

        return {"surprisals": surprisals, "entropies": entropies, "attentions": attentions}

    def _process_bert_batch(self, batch_inputs, batch_targets, batch_w_idxs, surprisals, entropies):
        inputs_tensor = torch.stack(batch_inputs).to(self.device)
        with torch.no_grad():
            b_outputs = self.model(inputs_tensor)
        
        for b in range(len(batch_inputs)):
            idxs, tids = batch_targets[b]
            logits = b_outputs.logits[b, idxs, :]
            lps = F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            
            ent = -(probs * lps).sum(dim=-1).mean().item()
            word_surp = 0.0
            for tidx, tid in enumerate(tids):
                word_surp -= lps[tidx, tid].item()
            
            target_w_idx = batch_w_idxs[b]
            surprisals[target_w_idx] = word_surp
            entropies[target_w_idx] = ent

    def _compute_gpt(self, words: List[str]) -> Dict[str, List[float]]:
        encoding = self.tokenizer(words, is_split_into_words=True, return_tensors="pt").to(self.device)
        input_ids = encoding["input_ids"]
        word_ids = encoding.word_ids()
        
        with torch.no_grad():
            outputs = self.model(input_ids)
        
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        lps = F.log_softmax(shift_logits, dim=-1)
        probs = F.softmax(shift_logits, dim=-1)
        token_ents = -(probs * lps).sum(dim=-1)
        target_lps = lps.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        surprisals = [0.0] * len(words)
        entropies = [0.0] * len(words)
        
        for i in range(target_lps.size(1)):
            w_idx = word_ids[i+1]
            if w_idx is not None:
                surprisals[w_idx] -= target_lps[0, i].item()
                entropies[w_idx] += token_ents[0, i].item()
        
        return {"surprisals": surprisals, "entropies": entropies, "attentions": [0.5] * len(words)}

class CognitiveLoadPipeline:
    POS_WEIGHTS = {
        'n': 1.0, 'v': 1.0, 'a': 0.9, 'd': 0.8, 't': 0.8, 'm': 0.3, 'p': 0.1, 'c': 0.1, 'u': 0.05, 'x': 0.0,
        'NOUN': 1.0, 'VERB': 1.0, 'ADJ': 0.9, 'ADV': 0.8, 'PROPN': 1.0, 'PRON': 0.4, 'ADP': 0.1, 'CCONJ': 0.1, 'SCONJ': 0.1, 'DET': 0.05, 'PUNCT': 0.0
    }

    def __init__(self, model_type: str = 'bert', lang: str = 'zh'):
        self.lang = lang
        self.model_type = model_type
        self.calculator = LanguageModelCalculator(model_type, lang)
        self.nlp = None
        if lang == 'en' and spacy:
            print("[載入 NLP] spaCy (en_core_web_sm) ...")
            self.nlp = spacy.load("en_core_web_sm")
        if lang == 'zh':
            for word in ["二十四時", "胡衛東", "國手", "江蘇隊", "出國深造"]: jieba.add_word(word)

    def run(self, text: str, prev_results: Optional[List[WordResult]] = None) -> dict:
        # 1. 解析文本
        if self.lang == 'zh':
            pairs = list(pseg.cut(text))
            words, pos_tags = [p.word for p in pairs], [p.flag for p in pairs]
            dep_loads, last_verb_idx = [], -1
            for idx, pos in enumerate(pos_tags):
                if pos and pos.startswith('v'):
                    load = (idx - last_verb_idx) * 0.12 if last_verb_idx != -1 else 0.1
                    dep_loads.append(min(load, 1.0)); last_verb_idx = idx
                else:
                    load = (idx - last_verb_idx) * 0.08 if last_verb_idx != -1 else 0.15
                    dep_loads.append(min(load, 1.0))
        else:
            doc = self.nlp(text) if self.nlp else None
            if doc:
                words = [token.text for token in doc]
                pos_tags = [token.pos_ for token in doc]
                dep_loads = [min(abs(token.i - token.head.i) * 0.15, 1.0) for token in doc]
            else:
                words = re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)
                pos_tags = ["NOUN"] * len(words)
                dep_loads = [0.1] * len(words)

        # 2. 模型推理
        metrics = self.calculator.compute(words)
        surprisals, entropies, attentions = metrics["surprisals"], metrics["entropies"], metrics["attentions"]

        # 3. 正規化與計分 (使用 Capping 防止極端值 squashing)
        surp_capped = [min(s, 20.0) for s in surprisals] # Cap Surprisal at 20.0
        max_s = max(surp_capped + [1.0])
        max_e = max(entropies + [1.0])
        max_a = max(attentions + [1.0])
        
        results = []
        last_load = prev_results[-1].load_score if prev_results else 0.0
        
        for i, (word, surp, ent, attn, pos, dl) in enumerate(zip(words, surprisals, entropies, attentions, pos_tags, dep_loads)):
            if not re.search(r'[\u4e00-\u9fff\d\w]', word):
                results.append(WordResult(word, pos, i, 0, 0, 0, 0, len(word), 0, "low", 0.0))
                continue
            
            s_score = min(surp, 20.0) / max_s
            e_score = ent / max_e
            a_score = attn / max_a
            f_score = min(max(0, (7.0 - zipf_frequency(word, self.lang)) / 5.0), 1.0)
            wl_score = min(len(word) / (8.0 if self.lang == 'zh' else 12.0), 1.0)
            p_score = self.POS_WEIGHTS.get(pos, 0.5)
            
            weights = [0.40, 0.20, 0.05, 0.10, 0.10, 0.05, 0.10]
            base_total = sum(w * s for w, s in zip(weights, [s_score, e_score, a_score, dl, f_score, wl_score, p_score]))
            
            final_score = base_total + (0.15 * last_load)
            final_score = round(min(max(final_score, 0.0), 1.0), 4)
            last_load = final_score
            results.append(WordResult(word, pos, i, round(surp, 3), round(ent, 3), round(dl, 3), round(zipf_frequency(word, self.lang), 2), len(word), round(p_score, 2), "low", final_score))

        # 4. 動態門檻 (Top 25% for better recall)
        valid_scores = [r.load_score for r in results if r.load_score > 0.1]
        if valid_scores:
            threshold = np.percentile(valid_scores, 75)
            # 自動適配英文較低的得分區間
            threshold = max(threshold, 0.25 if self.lang == 'en' else 0.55)
            for r in results:
                if r.load_score >= threshold: r.load_level = "high"
                elif r.load_score >= threshold * 0.6: r.load_level = "medium"

        return {"model": self.model_type, "lang": self.lang, "high_load_words": [r.word for r in results if r.load_level == "high"], "word_analysis": [asdict(r) for r in results]}

    def process_file(self, file_path: str, output_path: Optional[str] = None):
        text = DocumentLoader.load(file_path)
        sentences = [s for s in re.split(r'([。！？；\n])', text) if s.strip()]
        all_analysis, last_res = [], None
        
        total_chunks = len(sentences) // 2 + 1
        print(f"[文件處理] 開始分析，預計處理 {total_chunks} 個句段...")
        
        for i in range(0, len(sentences), 2):
            s = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")
            if not s.strip(): continue
            
            res_dict = self.run(s, prev_results=last_res)
            all_analysis.extend(res_dict['word_analysis'])
            last_res = [WordResult(**w) for w in res_dict['word_analysis']]
            
            if (i // 2 + 1) % 10 == 0:
                print(f"[文件處理] 進度: {i // 2 + 1}/{total_chunks} 句段完成")
        
        final = {"model": self.model_type, "lang": self.lang, "high_load_words": [w['word'] for w in all_analysis if w['load_level'] == 'high'], "word_analysis": all_analysis}
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f: json.dump(final, f, ensure_ascii=False, indent=2)
            print(f"[文件處理] 分析完成，結果已存至: {output_path}")
        return final

if __name__ == "__main__":
    p = CognitiveLoadPipeline(model_type='bert', lang='zh')
    print(json.dumps(p.run("大部分日子都是早晨六點起床，晚上二十四時左右睡覺。"), ensure_ascii=False, indent=2))

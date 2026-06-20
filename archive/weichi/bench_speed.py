"""
測量 pipeline 各階段耗時，找出速度瓶頸。
"""
import sys, io, time
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from cognitive_load_pipeline import CognitiveLoadPipeline

# 三段長度不同的文本（模擬實際使用場景）
SHORT_TEXT = "The ubiquitous phenomenon completely bewildered the researchers during the observation."

MEDIUM_TEXT = """The ubiquitous phenomenon completely bewildered the researchers during the observation.
The complex financial derivative products led to an unprecedented global economic crisis.
If you don't mind, we should investigate the mysterious laboratory immediately.
Quantum entanglement describes a non-local correlation between two particles.
The paradigm shift in cognitive psychology occurred during the late twentieth century.""".strip()

LONG_TEXT = (MEDIUM_TEXT + "\n") * 5  # 25 句

p = CognitiveLoadPipeline(model_type='gpt2', lang='en')

# Warmup
print("\n[Warmup]...")
_ = p.run(SHORT_TEXT)

def bench(label, text, n_runs=3):
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        result = p.run(text)
        times.append(time.time() - t0)
    avg = sum(times) / len(times)
    n_words = len(result['word_analysis'])
    print(f"  {label:<14} | 詞數={n_words:>4} | 平均={avg*1000:>7.1f}ms | per_word={avg*1000/n_words:>5.2f}ms")

print(f"\n[單句 run()] (device={p.calculator.device})")
bench("Short  (1 sent)",  SHORT_TEXT)
bench("Medium (5 sents)", MEDIUM_TEXT)
bench("Long   (25 sents)", LONG_TEXT)

# 拆解 run() 各階段耗時 (用 Medium 文本)
print(f"\n[各階段耗時拆解 — Medium 文本]")
import re
text = MEDIUM_TEXT

# 1. spaCy
t0 = time.time()
doc = p.nlp(text)
words = [t.text for t in doc]
pos_tags = [t.pos_ for t in doc]
spacy_ms = (time.time() - t0) * 1000

# 2. GPT-2
t0 = time.time()
metrics = p.calculator.compute(words)
gpt_ms = (time.time() - t0) * 1000

# 3. 完整 run（含 spaCy + GPT-2 + scoring）
t0 = time.time()
_ = p.run(text)
total_ms = (time.time() - t0) * 1000

scoring_ms = total_ms - spacy_ms - gpt_ms

print(f"  spaCy 解析   : {spacy_ms:>6.1f} ms ({spacy_ms/total_ms*100:>4.1f}%)")
print(f"  GPT-2 推理   : {gpt_ms:>6.1f} ms ({gpt_ms/total_ms*100:>4.1f}%)")
print(f"  Scoring+其他 : {scoring_ms:>6.1f} ms ({scoring_ms/total_ms*100:>4.1f}%)")
print(f"  TOTAL        : {total_ms:>6.1f} ms")

# spaCy components
print(f"\n[spaCy 已啟用 components]")
for name in p.nlp.pipe_names:
    print(f"  - {name}")

# process_file 速度測試（模擬實際 PDF 場景：多句長文件）
import tempfile, os
print(f"\n[process_file vs 逐句呼叫 run()] (模擬 25 句長文件)")

# 寫入臨時 .txt 檔案測 process_file
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
    f.write(LONG_TEXT)
    tmp_path = f.name

# v8: process_file (chunked)
t0 = time.time()
res = p.process_file(tmp_path)
pf_ms = (time.time() - t0) * 1000
n_words = len(res['word_analysis'])
print(f"  process_file (v8 chunked) : {pf_ms:>7.1f} ms  | 詞數={n_words} | per_word={pf_ms/n_words:.2f}ms")

# 對照組：逐句呼叫 run() (模擬 v7 行為)
sentences = LONG_TEXT.split("\n")
sentences = [s.strip() for s in sentences if s.strip()]
t0 = time.time()
mem = [0.0, 0.0]
total_words = 0
for s in sentences:
    r = p.run(s, cognitive_memory=mem)
    total_words += len(r['word_analysis'])
    if r['word_analysis']:
        mem = [r['word_analysis'][-1]['load_score'], mem[0]]
per_sent_ms = (time.time() - t0) * 1000
print(f"  逐句呼叫 run() (v7 模擬)  : {per_sent_ms:>7.1f} ms  | 詞數={total_words} | per_word={per_sent_ms/total_words:.2f}ms")
print(f"  → 加速比: {per_sent_ms/pf_ms:.1f}x")

os.unlink(tmp_path)

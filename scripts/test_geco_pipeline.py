import sys
import os
import re
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.cognition import CognitiveLoadPipeline

input_file = os.path.join(PROJECT_ROOT, "archive/data/geco/geco_pp01_trial5_clean.csv")
df = pd.read_csv(input_file)
sentence_words = df['WORD'].tolist()
full_sentence = " ".join(sentence_words)

pipeline = CognitiveLoadPipeline(model_type='bert', lang='en')
cog_result = pipeline.run(full_sentence)
word_analysis = cog_result.get("word_analysis", [])

def clean_word(w):
    return re.sub(r'[^a-zA-Z0-9]', '', w.lower())

def align_words(geco_words, pipeline_words):
    aligned = []
    p_idx = 0
    for gw in geco_words:
        cgw = clean_word(gw)
        if not cgw:
            aligned.append(None)
            continue
            
        found = False
        # Look ahead for exact match first
        for k in range(p_idx, min(p_idx + 15, len(pipeline_words))):
            cpw = clean_word(pipeline_words[k]["word"])
            if cgw == cpw:
                aligned.append(pipeline_words[k])
                p_idx = k + 1
                found = True
                break
                
        if not found:
            # Look ahead for prefix/prefix match (min length 2)
            for k in range(p_idx, min(p_idx + 15, len(pipeline_words))):
                cpw = clean_word(pipeline_words[k]["word"])
                if len(cpw) >= 2 and (cgw.startswith(cpw) or cpw.startswith(cgw)):
                    aligned.append(pipeline_words[k])
                    p_idx = k + 1
                    found = True
                    break
                    
        if not found:
            # Search globally in a window around p_idx for exact match
            for k in range(max(0, p_idx - 10), min(len(pipeline_words), p_idx + 30)):
                cpw = clean_word(pipeline_words[k]["word"])
                if cgw == cpw:
                    aligned.append(pipeline_words[k])
                    p_idx = k + 1
                    found = True
                    break
                    
        if not found:
            # Search globally in a window around p_idx for prefix match (min length 2)
            for k in range(max(0, p_idx - 10), min(len(pipeline_words), p_idx + 30)):
                cpw = clean_word(pipeline_words[k]["word"])
                if len(cpw) >= 2 and (cgw.startswith(cpw) or cpw.startswith(cgw)):
                    aligned.append(pipeline_words[k])
                    p_idx = k + 1
                    found = True
                    break
                    
        if not found:
            aligned.append(None)
            
    return aligned

aligned = align_words(sentence_words, word_analysis)

print(f"GECO rows: {len(df)}")
print(f"Aligned output rows: {len(aligned)}")
print(f"Unmatched count: {aligned.count(None)}")

for idx in range(min(30, len(df))):
    matched_word = aligned[idx]["word"] if aligned[idx] else "NONE"
    matched_score = aligned[idx]["load_score"] if aligned[idx] else 0.0
    print(f"GECO: {df.iloc[idx]['WORD'].strip()} -> Matched: {matched_word} (score={matched_score})")

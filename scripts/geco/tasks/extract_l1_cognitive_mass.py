import os
if os.name != 'nt' and os.environ.get("HF_HOME"):
    _hf_home = os.environ["HF_HOME"]
    if ":" in _hf_home or _hf_home.startswith("D:") or _hf_home.startswith("C:"):
        del os.environ["HF_HOME"]
import pandas as pd
import torch
import math
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

# Path configuration
input_file = "data/geco/geco_l1_pp01_trial5_clean.csv"
output_file = "data/geco/geco_l1_pp01_cognitive_mass.csv"

print("⏳ Loading L1 (Dutch) dataset...")
df = pd.read_csv(input_file)

sentence_words = df['WORD'].tolist()
full_sentence = " ".join([str(w) for w in sentence_words])

# Use Multilingual BERT for Dutch support
MODEL_NAME = "bert-base-multilingual-cased"
print(f"📦 Loading Multilingual Model: {MODEL_NAME}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, output_attentions=True).to(device)
model.eval()

print("🧠 Extracting Multilingual Neuro-Symbolic features...")

inputs = tokenizer(full_sentence, return_tensors="pt").to(device)
input_ids = inputs["input_ids"][0]
tokens = tokenizer.convert_ids_to_tokens(input_ids)

results = []

with torch.no_grad():
    outputs = model(**inputs)
    # Average across all heads of the last layer
    attention_matrix = outputs.attentions[-1][0].mean(dim=0) 

    # Align GECO words to BERT tokens
    word_token_indices = []
    for idx, row in df.iterrows():
        target_word = str(row['WORD']).strip()
        
        token_index = -1
        for i, token in enumerate(tokens):
            # Multilingual BERT uses '##' for subwords, but handles casing differently
            clean_token = token.replace("##", "")
            if clean_token.lower() == target_word.lower():
                token_index = i
                break
        word_token_indices.append(token_index)
                
    # Build Word-to-Word Attention Matrix
    num_words = len(df)
    word_attention_matrix = np.zeros((num_words, num_words))
    
    for i in range(num_words):
        ti = word_token_indices[i]
        for j in range(num_words):
            tj = word_token_indices[j]
            if ti != -1 and tj != -1:
                word_attention_matrix[i, j] = attention_matrix[ti, tj].item()

    # Calculate Surprisal and Attention Centrality
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Computing CM"):
        token_index = word_token_indices[i]
        if token_index != -1 and 0 < token_index < len(input_ids) - 1:
            gold_id = input_ids[token_index].item()
            
            # Masked Surprisal
            masked_ids = input_ids.clone()
            masked_ids[token_index] = tokenizer.mask_token_id
            
            mask_outputs = model(masked_ids.unsqueeze(0))
            logits = mask_outputs.logits[0, token_index]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            word_prob = probs[gold_id].item()
            surprisal = -math.log2(word_prob) if word_prob > 0 else 15.0
            
            # Attention Centrality
            attention_score = word_attention_matrix[:, i].sum()
            
            cognitive_mass = surprisal * attention_score
            
            results.append({
                "WORD_ID": row["WORD_ID"],
                "WORD": row["WORD"],
                "true_x": row["true_x"],
                "true_y": row["true_y"],
                "surprisal_score": round(surprisal, 4),
                "attention_score": round(attention_score, 4),
                "cognitive_mass": round(cognitive_mass, 4)
            })
        else:
             # Fallback for tokens not found (e.g. very rare words or punctuation issues)
             results.append({
                "WORD_ID": row["WORD_ID"],
                "WORD": row["WORD"],
                "true_x": row["true_x"],
                "true_y": row["true_y"],
                "surprisal_score": 10.0,
                "attention_score": 0.5,
                "cognitive_mass": 5.0
            })

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv(output_file, index=False)

# Save Attention Matrix
attention_output_file = output_file.replace(".csv", "_attention.npy")
np.save(attention_output_file, word_attention_matrix)

print(f"✅ Multilingual CM extraction complete: {output_file}")
print(f"✅ Attention Matrix saved: {attention_output_file}")
print("\nTop 3 high-mass Dutch words:")
print(df_results.sort_values(by="cognitive_mass", ascending=False).head(3))

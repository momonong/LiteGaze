"""
train_ridge_geco.py
用 GECO 語料訓練 Ridge Regression 預測 mean TRT，儲存模型供 v6 pipeline 使用。

用法：
  python train_ridge_geco.py [--train_n 100] [--val_n 20]
"""
import sys
import io
import os
import json
import re
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cognitive_load_pipeline import CognitiveLoadPipeline

GECO_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GECO_data")
MODEL_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ridge_model.json")

FEATURES = ["surprisal", "entropy", "aoa_score", "word_length", "zipf_score", "pos_score"]

# ─── 資料載入 ─────────────────────────────────────────────

def load_data():
    mat = pd.read_csv(os.path.join(GECO_DIR, "EnglishMaterial(ALL).csv"))
    mat = mat.dropna(subset=["SENTENCE_ID"]).copy()
    split = mat["SENTENCE_ID"].str.split("-", expand=True)
    mat["PART"]     = pd.to_numeric(split.iloc[:, 0], errors="coerce").fillna(0).astype(int)
    mat["SENT_NUM"] = pd.to_numeric(split.iloc[:, 1], errors="coerce").fillna(0).astype(int)
    mat = mat.sort_values("CHRON_ID").reset_index(drop=True)

    print("[讀取] MonolingualReadingData.csv…")
    eye = pd.read_csv(os.path.join(GECO_DIR, "MonolingualReadingData.csv"), low_memory=False)
    eye = eye[eye["LANGUAGE"] == "English"].copy()
    eye["TRT"] = pd.to_numeric(eye["WORD_TOTAL_READING_TIME"], errors="coerce")
    trt = eye.groupby("WORD_ID")["TRT"].agg(mean_trt="mean", n_readers="count").reset_index()

    return mat.merge(trt, on="WORD_ID", how="left")


def get_sentence(merged, sent_idx):
    """取第 sent_idx 句（1-based）材料。"""
    sent_order = (
        merged[["SENTENCE_ID", "PART", "SENT_NUM"]]
        .drop_duplicates()
        .sort_values(["PART", "SENT_NUM"])
        .reset_index(drop=True)
    )
    if sent_idx < 1 or sent_idx > len(sent_order):
        return pd.DataFrame()
    sid = sent_order.iloc[sent_idx - 1]["SENTENCE_ID"]
    return merged[merged["SENTENCE_ID"] == sid].copy()


_PUNCT = re.compile(r"""^[',.\!?;:\)\]"]""")
_OPEN  = re.compile(r"""^[\(\["]$""")

def reconstruct_text(chunk):
    words, parts = chunk["WORD"].tolist(), []
    for i, w in enumerate(words):
        if i == 0 or _PUNCT.match(w) or (i > 0 and _OPEN.match(words[i - 1])):
            parts.append(w)
        else:
            parts.append(" " + w)
    return "".join(parts)


# ─── 特徵抽取 ─────────────────────────────────────────────

def _normalize_key(w: str) -> str:
    return w.lower().strip("'.,!?;:()")


def extract_features(pipeline, chunk, text) -> list[dict]:
    """跑 pipeline，比對 GECO content words，回傳 feature rows。"""
    result = pipeline.run(text)

    # word → 分析結果（同一詞多次出現只保第一筆，後續以出現順序標記）
    pred_list = [w for w in result["word_analysis"]
                 if w["surprisal"] > 0 or w["aoa_score"] > 0]
    # 建立 (normalized_key, idx_in_sentence) → analysis 的 dict
    pred_by_key: dict[str, list] = {}
    for wa in pred_list:
        k = _normalize_key(wa["word"])
        pred_by_key.setdefault(k, []).append(wa)
    # 使用計數器讓重複詞能逐一匹配
    pred_counters: dict[str, int] = {}

    content = chunk[chunk["CONTENT_WORD"] == 1].dropna(subset=["mean_trt"])
    rows = []
    for _, row in content.iterrows():
        k = _normalize_key(row["WORD"])
        candidates = pred_by_key.get(k, [])
        if not candidates:
            continue
        used = pred_counters.get(k, 0)
        wa = candidates[min(used, len(candidates) - 1)]
        pred_counters[k] = used + 1

        rows.append({
            "word":       row["WORD"],
            "mean_trt":   row["mean_trt"],
            "surprisal":  wa["surprisal"],
            "entropy":    wa["entropy"],
            "aoa_score":  wa.get("aoa_score", 0.5),
            "zipf_score": wa["zipf_score"],
            "word_length": wa["word_length"],
            "pos_score":  wa["pos_score"],
        })
    return rows


# ─── 訓練與儲存 ───────────────────────────────────────────

def collect(pipeline, merged, start, n, label):
    rows = []
    for s_idx in range(start, start + n):
        chunk = get_sentence(merged, s_idx)
        if chunk.empty:
            continue
        text = reconstruct_text(chunk)
        if len(text.split()) < 3:
            continue
        rows.extend(extract_features(pipeline, chunk, text))
        if (s_idx - start + 1) % 10 == 0:
            print(f"  [{label}] {s_idx - start + 1}/{n} 句，累積 {len(rows)} 樣本")
    return rows


def main(train_n=100, val_n=20):
    print(f"=== GECO Ridge Regression 訓練 (train={train_n}, val={val_n}) ===\n")
    merged   = load_data()
    pipeline = CognitiveLoadPipeline(model_type="gpt2", lang="en")

    print(f"\n[收集訓練資料] 句子 1~{train_n}")
    train_rows = collect(pipeline, merged, 1,           train_n, "Train")
    print(f"\n[收集驗證資料] 句子 {train_n+1}~{train_n+val_n}")
    val_rows   = collect(pipeline, merged, train_n + 1, val_n,   "Val  ")

    train_df = pd.DataFrame(train_rows).dropna()
    val_df   = pd.DataFrame(val_rows).dropna()
    print(f"\n訓練樣本: {len(train_df)}  |  驗證樣本: {len(val_df)}")
    if len(train_df) < 50:
        print("[錯誤] 訓練樣本不足，請增加 --train_n")
        return

    X_tr = train_df[FEATURES].values;  y_tr = train_df["mean_trt"].values
    X_va = val_df[FEATURES].values;    y_va = val_df["mean_trt"].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    # 搜尋最佳 alpha
    best_r2, best_alpha, best_ridge = -np.inf, 10.0, None
    for alpha in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]:
        m = Ridge(alpha=alpha).fit(X_tr_s, y_tr)
        r2 = r2_score(y_va, m.predict(X_va_s))
        print(f"  alpha={alpha:<6}  val R²={r2:.4f}")
        if r2 > best_r2:
            best_r2, best_alpha, best_ridge = r2, alpha, m

    r2_tr = r2_score(y_tr, best_ridge.predict(X_tr_s))
    print(f"\n最佳 alpha={best_alpha}  →  Train R²={r2_tr:.4f}  |  Val R²={best_r2:.4f}")
    print("特徵係數（標準化後）：")
    for f, c in zip(FEATURES, best_ridge.coef_):
        print(f"  {f:<14} {c:+.4f}")

    model_data = {
        "features":     FEATURES,
        "coef":         best_ridge.coef_.tolist(),
        "intercept":    float(best_ridge.intercept_),
        "scaler_mean":  scaler.mean_.tolist(),
        "scaler_std":   scaler.scale_.tolist(),
        "alpha":        best_alpha,
        "r2_train":     float(r2_tr),
        "r2_val":       float(best_r2),
        "train_n":      train_n,
        "val_n":        val_n,
        "n_samples":    len(train_df),
    }
    with open(MODEL_OUT, "w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=2, ensure_ascii=False)
    print(f"\n[完成] 模型已存至 {MODEL_OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_n", type=int, default=100)
    parser.add_argument("--val_n",   type=int, default=20)
    args = parser.parse_args()
    main(args.train_n, args.val_n)

"""
GECO Corpus Benchmark for LexiGaze Cognitive Load Pipeline

用法：
  python benchmark_geco.py [--start 1] [--n 5] [--pct 70]

  --start  起始句子編號（從 1 開始，預設 1）
  --n      要測試的句子數（預設 5）
  --pct    TRT 百分位門檻，高於此視為 GT 高負荷（預設 70）
"""

import sys
import io
import os
import argparse
import re

import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cognitive_load_pipeline import CognitiveLoadPipeline

GECO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GECO_data")
MATERIAL_CSV = os.path.join(GECO_DIR, "EnglishMaterial(ALL).csv")
EYE_CSV = os.path.join(GECO_DIR, "MonolingualReadingData.csv")


# ─── 資料載入 ─────────────────────────────────────────────

def load_material() -> pd.DataFrame:
    """讀入 EnglishMaterial，解析 PART / SENT_NUM 供排序用。"""
    mat = pd.read_csv(MATERIAL_CSV)
    mat = mat.dropna(subset=["SENTENCE_ID"]).copy()
    split = mat["SENTENCE_ID"].str.split("-", expand=True)
    mat["PART"] = pd.to_numeric(split.iloc[:, 0], errors="coerce").fillna(0).astype(int)
    mat["SENT_NUM"] = pd.to_numeric(split.iloc[:, 1], errors="coerce").fillna(0).astype(int)
    return mat.sort_values("CHRON_ID").reset_index(drop=True)


def load_mean_trt() -> pd.DataFrame:
    """讀入眼動資料，對 L1 English 讀者聚合每詞平均 TRT。"""
    print("[讀取] MonolingualReadingData.csv…")
    eye = pd.read_csv(EYE_CSV, low_memory=False)

    # 只保留英文母語者
    eye = eye[eye["LANGUAGE"] == "English"].copy()

    # TRT 欄位：'.' 代表 NaN（詞被略過）
    eye["TRT"] = pd.to_numeric(eye["WORD_TOTAL_READING_TIME"], errors="coerce")

    trt_agg = (
        eye.groupby("WORD_ID")["TRT"]
        .agg(mean_trt="mean", n_readers="count")
        .reset_index()
    )
    return trt_agg


def get_chunk(mat: pd.DataFrame, start: int, n: int) -> pd.DataFrame:
    """取出從第 start 句（1-based）開始的 n 個句子。"""
    sent_order = (
        mat[["SENTENCE_ID", "PART", "SENT_NUM"]]
        .drop_duplicates()
        .sort_values(["PART", "SENT_NUM"])
        .reset_index(drop=True)
    )
    if start < 1 or start > len(sent_order):
        raise ValueError(f"start={start} 超出範圍（共 {len(sent_order)} 句）")
    target = sent_order.iloc[start - 1 : start - 1 + n]["SENTENCE_ID"].tolist()
    return mat[mat["SENTENCE_ID"].isin(target)].copy()


# ─── 文字重建 ─────────────────────────────────────────────

_PUNCT = re.compile(r"""^[',.\!?;:\)\]"]""")
_OPEN_PUNCT = re.compile(r"""^[\(\["]$""")

def reconstruct_text(chunk: pd.DataFrame) -> str:
    """將 GECO 的 WORD 欄位拼回自然語言文字。"""
    words = chunk["WORD"].tolist()
    parts = []
    for i, w in enumerate(words):
        if i == 0 or _PUNCT.match(w) or _OPEN_PUNCT.match(words[i - 1]):
            parts.append(w)
        else:
            parts.append(" " + w)
    return "".join(parts)


# ─── GT 定義 ──────────────────────────────────────────────

def define_gt(chunk: pd.DataFrame, trt_agg: pd.DataFrame, pct: int) -> set:
    """
    GT = content words with mean_trt > pct-th percentile（在本 chunk 的 content words 中）
    """
    merged = chunk.merge(trt_agg, on="WORD_ID", how="left")
    content = merged[merged["CONTENT_WORD"] == 1].copy()
    valid = content.dropna(subset=["mean_trt"])
    if valid.empty:
        return set()
    thresh = np.percentile(valid["mean_trt"], pct)
    gt = set(valid[valid["mean_trt"] >= thresh]["WORD"].str.lower())
    return gt, thresh, valid


# ─── 評估 ─────────────────────────────────────────────────

def compute_metrics(pred: set, gt: set) -> dict:
    hits = pred & gt
    fp = pred - gt
    fn = gt - pred
    prec = len(hits) / len(pred) if pred else 0
    rec = len(hits) / len(gt) if gt else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return dict(precision=prec, recall=rec, f1=f1,
                hits=sorted(hits), fp=sorted(fp), fn=sorted(fn))


def print_word_table(chunk: pd.DataFrame, trt_agg: pd.DataFrame,
                     gt: set, pred: set, pct: int):
    merged = chunk.merge(trt_agg, on="WORD_ID", how="left")
    content = merged[merged["CONTENT_WORD"] == 1].copy()
    valid = content.dropna(subset=["mean_trt"])
    thresh = np.percentile(valid["mean_trt"], pct) if not valid.empty else 0

    print(f"\n{'詞':<20} {'mean TRT':>10} {'GT':>5} {'Pred':>6} {'狀態'}")
    print("-" * 55)
    for _, row in content.iterrows():
        word = row["WORD"]
        word_l = word.lower()
        trt_val = row.get("mean_trt", float("nan"))
        in_gt = word_l in gt
        in_pred = word_l in pred
        trt_str = f"{trt_val:>10.1f}" if not np.isnan(trt_val) else f"{'N/A':>10}"
        gt_str = "✓" if in_gt else ""
        pred_str = "✓" if in_pred else ""
        if in_gt and in_pred:
            status = "命中"
        elif in_gt and not in_pred:
            status = "漏失"
        elif not in_gt and in_pred:
            status = "誤報"
        else:
            status = ""
        if in_gt or in_pred:
            print(f"{word:<20}{trt_str} {gt_str:>5} {pred_str:>6}  {status}")
    print(f"\n  TRT 門檻（{pct}th pct） = {thresh:.1f} ms")


# ─── 主流程 ───────────────────────────────────────────────

def run_geco_benchmark(start: int = 1, n: int = 5, pct: int = 70):
    mat = load_material()
    trt_agg = load_mean_trt()

    chunk = get_chunk(mat, start, n)
    text = reconstruct_text(chunk)
    gt_result = define_gt(chunk, trt_agg, pct)
    gt_set, thresh, valid_content = gt_result

    print(f"\n{'='*65}")
    print(f"[GECO Benchmark]  句子 {start}～{start+n-1}  |  TRT 門檻 {pct}th pct")
    print(f"{'='*65}")
    print(f"[文字片段] ({len(chunk)} 詞, {n} 句)")
    print(text[:400] + ("…" if len(text) > 400 else ""))
    print(f"\n[GT 高負荷詞] ({len(gt_set)} 詞, TRT > {thresh:.0f} ms): {sorted(gt_set)}")

    pipeline = CognitiveLoadPipeline(model_type="gpt2", lang="en")
    result = pipeline.run(text)
    predicted = {
        w.lower().strip() for w in result["high_load_words"]
        if w.strip() and all(ord(c) < 128 for c in w.strip())
    }
    print(f"[預測高負荷詞] ({len(predicted)} 詞): {sorted(predicted)}")

    print_word_table(chunk, trt_agg, gt_set, predicted, pct)

    m = compute_metrics(predicted, gt_set)
    print(f"\n{'─'*45}")
    print(f"  Precision : {m['precision']:.2%}  ({len(m['hits'])} / {len(m['hits'])+len(m['fp'])})")
    print(f"  Recall    : {m['recall']:.2%}  ({len(m['hits'])} / {len(m['hits'])+len(m['fn'])})")
    print(f"  F1 Score  : {m['f1']:.2%}")
    if m["fn"]:
        print(f"  漏失      : {m['fn']}")
    if m["fp"]:
        print(f"  誤報      : {m['fp']}")
    return m


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GECO Benchmark for LexiGaze")
    parser.add_argument("--start", type=int, default=1, help="起始句子編號（1-based）")
    parser.add_argument("--n", type=int, default=5, help="測試句子數")
    parser.add_argument("--pct", type=int, default=70, help="TRT 百分位門檻（預設 70）")
    args = parser.parse_args()
    run_geco_benchmark(start=args.start, n=args.n, pct=args.pct)

"""
train_xgb_geco.py  —  Opt3: 以 XGBoost 取代 Ridge 作為 pipeline 評分後端
特徵：surprisal, renyi_entropy, aoa_score, word_length, zipf_score, pos_score, dep_load（POS-gated）
目標：log(mean_TRT)（比 raw TRT 更接近常態分布）
輸出：xgb_model.json（pipeline 自動優先讀取）

Usage:
    python train_xgb_geco.py [--train_n 120] [--val_n 30]
"""
import sys, os, argparse, warnings
warnings.filterwarnings("ignore")
# stdout redirect handled by validate_geco import below

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_geco import (
    load_geco, select_sentences, run_pipeline, merge_eye, sig_stars
)

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xgb_model.json")

XGB_FEATS = ["surprisal", "renyi_entropy", "aoa_score",
             "word_length", "zipf_score", "pos_score", "dependency_load"]

def build_dataset(eye_merged: pd.DataFrame) -> pd.DataFrame:
    """Add log_trt and ensure all XGB feature columns exist."""
    df = eye_merged.copy()
    # dependency_load is already POS-gated inside the updated pipeline
    for col in XGB_FEATS:
        if col not in df.columns:
            df[col] = 0.0
    df["log_trt"] = np.log(df["mean_trt"].clip(lower=1))
    return df.dropna(subset=XGB_FEATS + ["log_trt"])


def train_xgb(train_df: pd.DataFrame, val_df: pd.DataFrame):
    X_tr = train_df[XGB_FEATS].values.astype(float)
    y_tr = train_df["log_trt"].values
    X_va = val_df[XGB_FEATS].values.astype(float)
    y_va = val_df["log_trt"].values

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=XGB_FEATS)
    dval   = xgb.DMatrix(X_va, label=y_va, feature_names=XGB_FEATS)

    params = {
        "objective":        "reg:squarederror",
        "eval_metric":      "rmse",
        "max_depth":        4,
        "eta":              0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "seed":             42,
        "verbosity":        0,
    }

    evals_result = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=30,
        evals_result=evals_result,
        verbose_eval=False,
    )

    preds_tr = booster.predict(dtrain)
    preds_va = booster.predict(dval)
    r2_tr = r2_score(y_tr, preds_tr)
    r2_va = r2_score(y_va, preds_va)

    rho_va, p_rho = stats.spearmanr(
        np.exp(preds_va), val_df["mean_trt"].values
    )

    print(f"\n  Train R² = {r2_tr:.4f}  |  Val R² = {r2_va:.4f}")
    print(f"  Val Spearman ρ (pred_TRT vs mean_TRT) = {rho_va:.3f}  {sig_stars(p_rho)}")
    print(f"  Best round: {booster.best_iteration}")

    # Feature importance
    scores = booster.get_fscore()
    print("\n  Feature importance (gain):")
    for feat in XGB_FEATS:
        print(f"    {feat:<20} {scores.get(feat, 0):>6.0f}")

    return booster, r2_va, rho_va


def main(train_n: int, val_n: int):
    print(f"=== XGBoost Training (train={train_n}, val={val_n}) ===\n")

    mat, eye = load_geco()
    all_sent_ids = select_sentences(mat, train_n + val_n)
    train_ids = all_sent_ids[:train_n]
    val_ids   = all_sent_ids[train_n:]

    print(f"[1/3] Pipeline on {train_n} train sentences...")
    train_scores = run_pipeline(mat, train_ids)
    # add renyi_entropy from word_analysis if available
    print(f"[2/3] Pipeline on {val_n} val sentences...")
    val_scores = run_pipeline(mat, val_ids)

    from validate_geco import merge_eye
    train_content, _ = merge_eye(train_scores, eye)
    val_content,   _ = merge_eye(val_scores,   eye)

    train_df = build_dataset(train_content)
    val_df   = build_dataset(val_content)
    print(f"  Train content words: {len(train_df)}  |  Val: {len(val_df)}")

    if len(train_df) < 50:
        print("[錯誤] 訓練樣本不足，請增加 --train_n")
        return

    print("\n[3/3] Training XGBoost...")
    booster, r2_va, rho_va = train_xgb(train_df, val_df)

    booster.save_model(OUT_PATH)
    print(f"\n[完成] 模型儲存至 {OUT_PATH}")
    print(f"  Val R² = {r2_va:.4f}  |  Val ρ(TRT) = {rho_va:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_n", type=int, default=120)
    parser.add_argument("--val_n",   type=int, default=30)
    args = parser.parse_args()
    main(args.train_n, args.val_n)

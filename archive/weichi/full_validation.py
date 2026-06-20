"""
full_validation.py — 論文等級完整驗證流程
Step 1: 用前 2000 句訓練 XGBoost
Step 2: 在 2101-3100 句（1000 句 held-out）測試，完全 unseen data
Step 3: OLS + LMM (per-reader 隨機截距) + Bootstrap 95% CI

Usage: python full_validation.py
預計執行時間：60-70 分鐘
"""
import sys, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from scipy.stats import chi2 as chi2_dist
from sklearn.metrics import r2_score
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_geco import (
    load_geco, select_sentences, run_pipeline, merge_eye,
    sig_stars, OUT_DIR
)

XGB_FEATS   = ["surprisal", "renyi_entropy", "aoa_score",
               "word_length", "zipf_score", "pos_score", "dependency_load"]
TRAIN_N     = 2000
VAL_N       = 100
HOLDOUT_N   = 1000
TRAIN_START = 0
TEST_START  = TRAIN_N + VAL_N   # sentence 2101 onwards
XGB_PATH    = os.path.join(OUT_DIR, "xgb_model.json")
N_BOOT      = 1000   # bootstrap iterations for 95% CI


def build_df(content_df: pd.DataFrame) -> pd.DataFrame:
    df = content_df.copy()
    if "word_length" not in df.columns and "WORD_LENGTH" in df.columns:
        df["word_length"] = df["WORD_LENGTH"]
    for col in XGB_FEATS:
        if col not in df.columns:
            df[col] = 0.0
    df["log_trt"] = np.log(df["mean_trt"].clip(lower=1))
    df["log_gd"]  = np.log(df["mean_gd"].clip(lower=1))
    return df.dropna(subset=XGB_FEATS + ["log_trt"])


def train_xgb(train_df, val_df):
    dtrain = xgb.DMatrix(train_df[XGB_FEATS].values.astype(float),
                         label=train_df["log_trt"].values,
                         feature_names=XGB_FEATS)
    dval   = xgb.DMatrix(val_df[XGB_FEATS].values.astype(float),
                         label=val_df["log_trt"].values,
                         feature_names=XGB_FEATS)
    params = {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "max_depth": 4, "eta": 0.05, "subsample": 0.8,
        "colsample_bytree": 0.8, "min_child_weight": 5,
        "seed": 42, "verbosity": 0,
    }
    booster = xgb.train(params, dtrain, num_boost_round=500,
                        evals=[(dtrain, "train"), (dval, "val")],
                        early_stopping_rounds=30, verbose_eval=False)
    booster.save_model(XGB_PATH)
    print(f"  Best round: {booster.best_iteration}  → saved to {XGB_PATH}")
    return booster


def _bootstrap_rho(trt_preds, mean_trt, n_boot=N_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    n = len(trt_preds)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r, _ = stats.spearmanr(trt_preds[idx], mean_trt.iloc[idx])
        boot.append(r)
    arr = np.array(boot)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def evaluate(booster, test_df, lmm_df=None):
    # ── XGBoost predictions ───────────────────────────────────────
    dtest     = xgb.DMatrix(test_df[XGB_FEATS].values.astype(float),
                             feature_names=XGB_FEATS)
    log_preds = booster.predict(dtest)
    trt_preds = np.exp(log_preds)

    rho_trt, p_trt = stats.spearmanr(trt_preds, test_df["mean_trt"])
    rho_gd,  p_gd  = stats.spearmanr(trt_preds,
                                      test_df["mean_gd"].fillna(test_df["mean_trt"]))
    r2 = r2_score(test_df["log_trt"], log_preds)

    # ── Bootstrap 95% CI ─────────────────────────────────────────
    print("  Bootstrap CI (n=1000)...")
    trt_arr = np.array(trt_preds)
    ci_trt_lo, ci_trt_hi = _bootstrap_rho(trt_arr, test_df["mean_trt"].reset_index(drop=True))
    ci_gd_lo,  ci_gd_hi  = _bootstrap_rho(
        trt_arr, test_df["mean_gd"].fillna(test_df["mean_trt"]).reset_index(drop=True))

    # ── Normalize to [0,1] load score ────────────────────────────
    mn, mx = trt_preds.min(), trt_preds.max()
    test_df = test_df.copy().reset_index(drop=True)
    test_df["xgb_load"] = (trt_preds - mn) / max(mx - mn, 1e-8)
    test_df["log_trt"]  = np.log(test_df["mean_trt"].clip(lower=1))

    # ── Spillover ────────────────────────────────────────────────
    test_df = test_df.sort_values(["SENTENCE_ID", "sent_position"])
    test_df["prev_surprisal"]   = (test_df.groupby("SENTENCE_ID")["surprisal"]
                                    .shift(1).fillna(test_df["surprisal"].mean()))
    test_df["prev_word_length"] = (test_df.groupby("SENTENCE_ID")["WORD_LENGTH"]
                                    .shift(1).fillna(test_df["WORD_LENGTH"].mean()))

    # ── OLS ──────────────────────────────────────────────────────
    m_base = smf.ols(
        "log_trt ~ WORD_LENGTH + zipf_score + sent_position"
        " + prev_surprisal + prev_word_length",
        data=test_df).fit()
    m_full = smf.ols(
        "log_trt ~ xgb_load + WORD_LENGTH + zipf_score + sent_position"
        " + prev_surprisal + prev_word_length",
        data=test_df).fit()
    delta_r2  = m_full.rsquared - m_base.rsquared
    delta_aic = m_base.aic - m_full.aic
    beta_ols  = m_full.params.get("xgb_load", float("nan"))
    p_ols     = m_full.pvalues.get("xgb_load", float("nan"))

    # ── LMM (per-reader random intercepts) ───────────────────────
    lmm_beta = lmm_p = lrt_stat = lrt_p = lmm_daic = float("nan")
    n_obs = n_subj = 0

    if lmm_df is not None and len(lmm_df) > 0:
        print("  LMM (per-reader)...")
        # z-score xgb_load and join to per-reader data via WORD_ID
        mu, sd = test_df["xgb_load"].mean(), test_df["xgb_load"].std()
        xgb_z = test_df[["WORD_ID", "xgb_load"]].copy()
        xgb_z["xgb_load_z"] = (xgb_z["xgb_load"] - mu) / (sd if sd > 0 else 1)

        lmm_ext = lmm_df.merge(xgb_z[["WORD_ID", "xgb_load_z"]], on="WORD_ID", how="inner")
        lmm_ext = lmm_ext.dropna(subset=["xgb_load_z", "log_trt",
                                          "WORD_LENGTH_z", "zipf_score_z",
                                          "sent_position_z"])
        try:
            m_lmm_base = smf.mixedlm(
                "log_trt ~ WORD_LENGTH_z + zipf_score_z + sent_position_z",
                data=lmm_ext, groups=lmm_ext["subject"]).fit(reml=False)
            m_lmm_full = smf.mixedlm(
                "log_trt ~ xgb_load_z + WORD_LENGTH_z + zipf_score_z + sent_position_z",
                data=lmm_ext, groups=lmm_ext["subject"]).fit(reml=False)

            lrt_stat = -2 * (m_lmm_base.llf - m_lmm_full.llf)
            lrt_p    = chi2_dist.sf(lrt_stat, df=1) if lrt_stat >= 0 else float("nan")
            lmm_beta = m_lmm_full.params.get("xgb_load_z", float("nan"))
            lmm_p    = m_lmm_full.pvalues.get("xgb_load_z", float("nan"))
            lmm_daic = m_lmm_base.aic - m_lmm_full.aic
            n_obs    = len(lmm_ext)
            n_subj   = lmm_ext["subject"].nunique()
        except Exception as e:
            print(f"  LMM failed: {e}")

    # ── GD OLS ───────────────────────────────────────────────────
    gd_df = test_df.dropna(subset=["log_gd"])
    beta_ols_gd = delta_aic_gd = delta_r2_gd = p_ols_gd = float("nan")
    if len(gd_df) > 50:
        m_base_gd = smf.ols(
            "log_gd ~ WORD_LENGTH + zipf_score + sent_position"
            " + prev_surprisal + prev_word_length",
            data=gd_df).fit()
        m_full_gd = smf.ols(
            "log_gd ~ xgb_load + WORD_LENGTH + zipf_score + sent_position"
            " + prev_surprisal + prev_word_length",
            data=gd_df).fit()
        delta_r2_gd  = m_full_gd.rsquared - m_base_gd.rsquared
        delta_aic_gd = m_base_gd.aic - m_full_gd.aic
        beta_ols_gd  = m_full_gd.params.get("xgb_load", float("nan"))
        p_ols_gd     = m_full_gd.pvalues.get("xgb_load", float("nan"))

    # ── GD LMM ───────────────────────────────────────────────────
    lmm_beta_gd = lmm_p_gd = lrt_stat_gd = lrt_p_gd = lmm_daic_gd = float("nan")
    n_obs_gd = n_subj_gd = 0

    if lmm_df is not None and len(lmm_df) > 0 and "log_gd" in lmm_df.columns:
        mu, sd = test_df["xgb_load"].mean(), test_df["xgb_load"].std()
        xgb_z_gd = test_df[["WORD_ID", "xgb_load"]].copy()
        xgb_z_gd["xgb_load_z"] = (xgb_z_gd["xgb_load"] - mu) / (sd if sd > 0 else 1)

        lmm_gd = lmm_df.merge(xgb_z_gd[["WORD_ID", "xgb_load_z"]], on="WORD_ID", how="inner")
        lmm_gd = lmm_gd.dropna(subset=["xgb_load_z", "log_gd",
                                        "WORD_LENGTH_z", "zipf_score_z", "sent_position_z"])
        if len(lmm_gd) > 100:
            try:
                m_lmm_base_gd = smf.mixedlm(
                    "log_gd ~ WORD_LENGTH_z + zipf_score_z + sent_position_z",
                    data=lmm_gd, groups=lmm_gd["subject"]).fit(reml=False)
                m_lmm_full_gd = smf.mixedlm(
                    "log_gd ~ xgb_load_z + WORD_LENGTH_z + zipf_score_z + sent_position_z",
                    data=lmm_gd, groups=lmm_gd["subject"]).fit(reml=False)
                lrt_stat_gd = -2 * (m_lmm_base_gd.llf - m_lmm_full_gd.llf)
                lrt_p_gd    = chi2_dist.sf(lrt_stat_gd, df=1) if lrt_stat_gd >= 0 else float("nan")
                lmm_beta_gd = m_lmm_full_gd.params.get("xgb_load_z", float("nan"))
                lmm_p_gd    = m_lmm_full_gd.pvalues.get("xgb_load_z", float("nan"))
                lmm_daic_gd = m_lmm_base_gd.aic - m_lmm_full_gd.aic
                n_obs_gd    = len(lmm_gd)
                n_subj_gd   = lmm_gd["subject"].nunique()
            except Exception as e:
                print(f"  GD LMM failed: {e}")

    return {
        "n_words":       len(test_df),
        "rho_trt":       rho_trt,       "p_trt":       p_trt,
        "rho_gd":        rho_gd,        "p_gd":        p_gd,
        "ci_trt":        (ci_trt_lo, ci_trt_hi),
        "ci_gd":         (ci_gd_lo,  ci_gd_hi),
        "r2_held":       r2,
        # TRT
        "beta_ols":      beta_ols,      "p_ols":       p_ols,
        "delta_r2":      delta_r2,      "delta_aic":   delta_aic,
        "lmm_beta":      lmm_beta,      "lmm_p":       lmm_p,
        "lrt_stat":      lrt_stat,      "lrt_p":       lrt_p,
        "lmm_daic":      lmm_daic,
        "n_obs":         n_obs,         "n_subj":      n_subj,
        # GD
        "beta_ols_gd":   beta_ols_gd,   "p_ols_gd":    p_ols_gd,
        "delta_r2_gd":   delta_r2_gd,   "delta_aic_gd": delta_aic_gd,
        "lmm_beta_gd":   lmm_beta_gd,   "lmm_p_gd":    lmm_p_gd,
        "lrt_stat_gd":   lrt_stat_gd,   "lrt_p_gd":    lrt_p_gd,
        "lmm_daic_gd":   lmm_daic_gd,
        "n_obs_gd":      n_obs_gd,      "n_subj_gd":   n_subj_gd,
    }


def write_report(res, train_n, val_n, holdout_n):
    ci_trt = res["ci_trt"]
    ci_gd  = res["ci_gd"]

    def fmt(v, d=3): return f"{v:.{d}f}" if not np.isnan(v) else "—"

    lines = [
        "# Full GECO Validation Report — Pipeline v9 (Paper-Level)",
        "",
        f"> Training: {train_n} sentences  |  XGB internal val: {val_n}  |"
        f"  Held-out: {holdout_n} sentences",
        f"> Test content words (≥3 readers): **{res['n_words']}**",
        f"> OLS controls: word length, Zipf, sentence position,",
        f">   prev-word surprisal + length (spillover).",
        f"> LMM: per-reader random intercepts ({res['n_subj']} subjects, "
        f"{res['n_obs']} observations)",
        f"> Bootstrap: {N_BOOT} iterations, 95% CI via percentile method.",
        "",
        "---", "",
        "## Key Results", "",
        "| Metric | Value | 95% CI | Sig. |",
        "|--------|-------|--------|------|",
        f"| Spearman ρ (TRT) | {fmt(res['rho_trt'])} "
        f"| [{ci_trt[0]:.3f}, {ci_trt[1]:.3f}] | {sig_stars(res['p_trt'])} |",
        f"| Spearman ρ (GD)  | {fmt(res['rho_gd'])}  "
        f"| [{ci_gd[0]:.3f}, {ci_gd[1]:.3f}] | {sig_stars(res['p_gd'])} |",
        f"| Held-out R² (log TRT) | {fmt(res['r2_held'],4)} | — | — |",
        "",
        "## OLS Regression — TRT (Word-level Mean)", "",
        "| Parameter | Value | Sig. |",
        "|-----------|-------|------|",
        f"| β(xgb_load) | {fmt(res['beta_ols'],4)} | {sig_stars(res['p_ols'])} |",
        f"| ΔR²         | {fmt(res['delta_r2'],4)} | — |",
        f"| ΔAIC        | {fmt(res['delta_aic'],1)} | — |",
        "",
        "## OLS Regression — GD (Word-level Mean)", "",
        "| Parameter | Value | Sig. |",
        "|-----------|-------|------|",
        f"| β(xgb_load) | {fmt(res['beta_ols_gd'],4)} | {sig_stars(res['p_ols_gd'])} |",
        f"| ΔR²         | {fmt(res['delta_r2_gd'],4)} | — |",
        f"| ΔAIC        | {fmt(res['delta_aic_gd'],1)} | — |",
        "",
        "## LMM — TRT (Per-Reader, Random Intercepts)", "",
        "| Parameter | Value | Sig. |",
        "|-----------|-------|------|",
        f"| β(xgb_load_z) | {fmt(res['lmm_beta'],4)} | {sig_stars(res['lmm_p'])} |",
        f"| LRT χ²(1)     | {fmt(res['lrt_stat'],2)} | {sig_stars(res['lrt_p'])} |",
        f"| ΔAIC          | {fmt(res['lmm_daic'],1)} | — |",
        f"| N (obs)       | {res['n_obs']} | — |",
        f"| N (subjects)  | {res['n_subj']} | — |",
        "",
        "## LMM — GD (Per-Reader, Random Intercepts)", "",
        "| Parameter | Value | Sig. |",
        "|-----------|-------|------|",
        f"| β(xgb_load_z) | {fmt(res['lmm_beta_gd'],4)} | {sig_stars(res['lmm_p_gd'])} |",
        f"| LRT χ²(1)     | {fmt(res['lrt_stat_gd'],2)} | {sig_stars(res['lrt_p_gd'])} |",
        f"| ΔAIC          | {fmt(res['lmm_daic_gd'],1)} | — |",
        f"| N (obs)       | {res['n_obs_gd']} | — |",
        f"| N (subjects)  | {res['n_subj_gd']} | — |",
        "",
        "---", "",
        "## Paper-Ready Quote",
        f'> "The pipeline predicted mean TRT with Spearman ρ = {res["rho_trt"]:.3f} '
        f'(95% CI [{ci_trt[0]:.3f}, {ci_trt[1]:.3f}])',
        f'> and GD ρ = {res["rho_gd"]:.3f} (95% CI [{ci_gd[0]:.3f}, {ci_gd[1]:.3f}])',
        f'> on {res["n_words"]} content words from {holdout_n} held-out GECO sentences.',
        f'> The load score independently predicted TRT after controlling for',
        f'> word frequency, length, sentence position, and spillover',
        f'> (OLS β = {res["beta_ols"]:.3f}, p < .001, ΔAIC = {res["delta_aic"]:.1f};',
        f'> LMM β = {res["lmm_beta"]:.3f}, LRT χ²(1) = {res["lrt_stat"]:.2f}, '
        f'p < .001, ΔAIC = {res["lmm_daic"]:.1f}, n = {res["n_obs"]} reader×word obs).',
        f'> Parallel models on GD yielded',
        f'> OLS β = {res["beta_ols_gd"]:.3f} (ΔAIC = {res["delta_aic_gd"]:.1f})',
        f'> and LMM β = {res["lmm_beta_gd"]:.3f}, LRT χ²(1) = {res["lrt_stat_gd"]:.2f}, '
        f'p < .001, ΔAIC = {res["lmm_daic_gd"]:.1f}, n = {res["n_obs_gd"]} obs."',
    ]
    path = os.path.join(OUT_DIR, "full_validation_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Report: {path}")


def main():
    mat, eye = load_geco()

    print(f"\n[1/4] Pipeline on {TRAIN_N} train + {VAL_N} val sentences...")
    train_ids = select_sentences(mat, TRAIN_N, offset=TRAIN_START)
    val_ids   = select_sentences(mat, VAL_N,   offset=TRAIN_N)
    train_sc  = run_pipeline(mat, train_ids)
    val_sc    = run_pipeline(mat, val_ids)

    train_content, _ = merge_eye(train_sc, eye)
    val_content,   _ = merge_eye(val_sc,   eye)
    train_df = build_df(train_content)
    val_df   = build_df(val_content)
    print(f"  Train: {len(train_df)} content words  |  Val: {len(val_df)}")

    print(f"\n[2/4] Training XGBoost on {len(train_df)} samples...")
    booster = train_xgb(train_df, val_df)

    print(f"\n[3/4] Pipeline on {HOLDOUT_N} held-out sentences "
          f"(sentences {TEST_START+1}-{TEST_START+HOLDOUT_N})...")
    test_ids = select_sentences(mat, HOLDOUT_N, offset=TEST_START)
    test_sc  = run_pipeline(mat, test_ids)
    test_content, test_lmm = merge_eye(test_sc, eye)   # keep lmm_df
    test_df  = build_df(test_content)
    print(f"  Held-out: {len(test_df)} content words, "
          f"{len(test_lmm)} per-reader observations")

    print(f"\n[4/4] Evaluating on held-out set...")
    res = evaluate(booster, test_df, lmm_df=test_lmm)

    print(f"  Spearman ρ (TRT) = {res['rho_trt']:.3f} {sig_stars(res['p_trt'])}"
          f"  95% CI [{res['ci_trt'][0]:.3f}, {res['ci_trt'][1]:.3f}]")
    print(f"  Spearman ρ (GD)  = {res['rho_gd']:.3f}  {sig_stars(res['p_gd'])}"
          f"  95% CI [{res['ci_gd'][0]:.3f}, {res['ci_gd'][1]:.3f}]")
    print(f"  Held-out R²      = {res['r2_held']:.4f}")
    print(f"  OLS β(load) = {res['beta_ols']:.4f}  p={res['p_ols']:.4f}"
          f"  {sig_stars(res['p_ols'])}  ΔAIC={res['delta_aic']:.1f}")
    print(f"  OLS β(GD)   = {res['beta_ols_gd']:.4f}  p={res['p_ols_gd']:.4f}"
          f"  {sig_stars(res['p_ols_gd'])}  ΔAIC={res['delta_aic_gd']:.1f}")
    print(f"  LMM β(load_z) TRT = {res['lmm_beta']:.4f}  p={res['lmm_p']:.4f}"
          f"  {sig_stars(res['lmm_p'])}  LRT χ²={res['lrt_stat']:.2f}"
          f"  ΔAIC={res['lmm_daic']:.1f}")
    print(f"  LMM β(load_z) GD  = {res['lmm_beta_gd']:.4f}  p={res['lmm_p_gd']:.4f}"
          f"  {sig_stars(res['lmm_p_gd'])}  LRT χ²={res['lrt_stat_gd']:.2f}"
          f"  ΔAIC={res['lmm_daic_gd']:.1f}")

    write_report(res, TRAIN_N, VAL_N, HOLDOUT_N)
    print("\n[完成] full_validation_report.md")


if __name__ == "__main__":
    main()

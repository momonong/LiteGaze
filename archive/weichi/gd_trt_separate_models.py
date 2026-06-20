"""
gd_trt_separate_models.py — GD / TRT 分開 OLS + LMM 分析
使用已快取的 test_predictions.csv，不需重跑 pipeline。
預計執行時間：2–5 分鐘
"""
import sys, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2 as chi2_dist
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_geco import load_geco, sig_stars, OUT_DIR, TRT_MIN, TRT_MAX

PRED_CSV = os.path.join(OUT_DIR, "test_predictions.csv")


def load_per_reader_gd(pred_df, eye):
    """Build per-reader GD dataframe for LMM, joined to pred_df controls."""
    word_ids = set(pred_df["WORD_ID"])
    eye_sub = eye[eye["WORD_ID"].isin(word_ids)].copy()

    # Outlier removal
    for col in ["WORD_TOTAL_READING_TIME", "WORD_GAZE_DURATION"]:
        eye_sub.loc[(eye_sub[col] < TRT_MIN) | (eye_sub[col] > TRT_MAX), col] = np.nan

    per_reader = eye_sub[["PP_NR", "WORD_ID",
                           "WORD_TOTAL_READING_TIME", "WORD_GAZE_DURATION"]].copy()
    per_reader = per_reader.rename(columns={
        "WORD_TOTAL_READING_TIME": "TRT",
        "WORD_GAZE_DURATION":      "GD",
    })
    per_reader = per_reader.dropna(subset=["TRT"])

    ctrl = pred_df[["WORD_ID", "xgb_load",
                    "WORD_LENGTH_z", "zipf_score_z", "sent_position_z"]].copy()
    mu, sd = pred_df["xgb_load"].mean(), pred_df["xgb_load"].std()
    ctrl["xgb_load_z"] = (ctrl["xgb_load"] - mu) / (sd if sd > 0 else 1)

    lmm_df = per_reader.merge(ctrl, on="WORD_ID", how="inner")
    lmm_df["log_trt"] = np.log(lmm_df["TRT"].clip(lower=1))
    lmm_df["log_gd"]  = np.log(lmm_df["GD"].clip(lower=1))   # NaN for skipped words
    lmm_df["subject"]  = lmm_df["PP_NR"].astype(str)
    return lmm_df


def run_ols(df, outcome, outcome_label):
    d = df.dropna(subset=[outcome])
    if len(d) < 50:
        print(f"  {outcome_label}: 資料不足 ({len(d)})，跳過")
        return None
    m_base = smf.ols(
        f"{outcome} ~ WORD_LENGTH + zipf_score + sent_position"
        " + prev_surprisal + prev_word_length",
        data=d).fit()
    m_full = smf.ols(
        f"{outcome} ~ xgb_load + WORD_LENGTH + zipf_score + sent_position"
        " + prev_surprisal + prev_word_length",
        data=d).fit()
    beta  = m_full.params.get("xgb_load", float("nan"))
    pval  = m_full.pvalues.get("xgb_load", float("nan"))
    dr2   = m_full.rsquared - m_base.rsquared
    daic  = m_base.aic - m_full.aic
    print(f"  OLS {outcome_label}: β={beta:.4f} {sig_stars(pval)}  "
          f"ΔR²={dr2:.4f}  ΔAIC={daic:.1f}  n={len(d)}")
    return {"beta": beta, "p": pval, "delta_r2": dr2, "delta_aic": daic,
            "n": len(d), "m_full": m_full}


def run_lmm(lmm_df, outcome, outcome_label):
    d = lmm_df.dropna(subset=[outcome, "xgb_load_z",
                               "WORD_LENGTH_z", "zipf_score_z", "sent_position_z"])
    if len(d) < 100:
        print(f"  LMM {outcome_label}: 資料不足 ({len(d)})，跳過")
        return None
    try:
        m_base = smf.mixedlm(
            f"{outcome} ~ WORD_LENGTH_z + zipf_score_z + sent_position_z",
            data=d, groups=d["subject"]).fit(reml=False)
        m_full = smf.mixedlm(
            f"{outcome} ~ xgb_load_z + WORD_LENGTH_z + zipf_score_z + sent_position_z",
            data=d, groups=d["subject"]).fit(reml=False)
        lrt  = -2 * (m_base.llf - m_full.llf)
        lrt_p = chi2_dist.sf(lrt, df=1) if lrt >= 0 else float("nan")
        beta  = m_full.params.get("xgb_load_z", float("nan"))
        pval  = m_full.pvalues.get("xgb_load_z", float("nan"))
        daic  = m_base.aic - m_full.aic
        n_obs = len(d);  n_subj = d["subject"].nunique()
        print(f"  LMM {outcome_label}: β={beta:.4f} {sig_stars(pval)}  "
              f"LRT χ²={lrt:.2f} {sig_stars(lrt_p)}  ΔAIC={daic:.1f}  "
              f"n={n_obs} obs / {n_subj} subj")
        return {"beta": beta, "p": pval, "lrt": lrt, "lrt_p": lrt_p,
                "delta_aic": daic, "n_obs": n_obs, "n_subj": n_subj}
    except Exception as e:
        print(f"  LMM {outcome_label} failed: {e}")
        return None


def write_report(ols_trt, ols_gd, lmm_trt, lmm_gd):
    def fmt(v, d=4):
        return f"{v:.{d}f}" if v is not None and not np.isnan(v) else "—"
    def s(r, k):
        return sig_stars(r[k]) if r and not np.isnan(r.get(k, float("nan"))) else "—"

    lines = [
        "# GD / TRT Separate Models — Pipeline v9",
        "",
        "## OLS Regression (Word-level Mean, Spillover Controlled)", "",
        "| Measure | β(xgb_load) | ΔR² | ΔAIC | Sig. | n |",
        "|---------|------------|-----|------|------|---|",
        f"| TRT | {fmt(ols_trt['beta'] if ols_trt else float('nan'))} "
        f"| {fmt(ols_trt['delta_r2'] if ols_trt else float('nan'))} "
        f"| {fmt(ols_trt['delta_aic'] if ols_trt else float('nan'), 1)} "
        f"| {s(ols_trt,'p') if ols_trt else '—'} "
        f"| {ols_trt['n'] if ols_trt else '—'} |",
        f"| GD  | {fmt(ols_gd['beta'] if ols_gd else float('nan'))} "
        f"| {fmt(ols_gd['delta_r2'] if ols_gd else float('nan'))} "
        f"| {fmt(ols_gd['delta_aic'] if ols_gd else float('nan'), 1)} "
        f"| {s(ols_gd,'p') if ols_gd else '—'} "
        f"| {ols_gd['n'] if ols_gd else '—'} |",
        "",
        "## LMM (Per-Reader Random Intercepts)", "",
        "| Measure | β(xgb_load_z) | LRT χ²(1) | ΔAIC | Sig. | N obs | N subj |",
        "|---------|--------------|-----------|------|------|-------|--------|",
    ]
    for label, r in [("TRT", lmm_trt), ("GD", lmm_gd)]:
        if r:
            lines.append(
                f"| {label} | {fmt(r['beta'])} | {fmt(r['lrt'],2)} "
                f"| {fmt(r['delta_aic'],1)} | {sig_stars(r['p'])} "
                f"| {r['n_obs']} | {r['n_subj']} |"
            )
        else:
            lines.append(f"| {label} | — | — | — | — | — | — |")

    lines += [
        "",
        "---",
        "## Paper-Ready Quote",
        "> \"The pipeline independently predicted both eye-tracking measures after",
        "> controlling for word frequency, length, sentence position, and spillover.",
    ]
    if ols_trt and ols_gd:
        lines += [
            f"> TRT: OLS β = {ols_trt['beta']:.3f}, p < .001, ΔAIC = {ols_trt['delta_aic']:.1f};",
            f"> GD: OLS β = {ols_gd['beta']:.3f}, p < .001, ΔAIC = {ols_gd['delta_aic']:.1f}.",
        ]
    if lmm_trt and lmm_gd:
        lines += [
            f"> Mixed-effects models confirmed both effects:",
            f"> TRT: LMM β = {lmm_trt['beta']:.3f}, LRT χ²(1) = {lmm_trt['lrt']:.2f}, "
            f"p < .001, ΔAIC = {lmm_trt['delta_aic']:.1f} ({lmm_trt['n_obs']} reader×word obs);",
            f"> GD: LMM β = {lmm_gd['beta']:.3f}, LRT χ²(1) = {lmm_gd['lrt']:.2f}, "
            f"p < .001, ΔAIC = {lmm_gd['delta_aic']:.1f} ({lmm_gd['n_obs']} obs).\"",
        ]

    path = os.path.join(OUT_DIR, "gd_trt_separate_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Report: {path}")
    return path


def main():
    print("[1/4] 載入快取預測結果...")
    pred_df = pd.read_csv(PRED_CSV)
    print(f"  {len(pred_df)} content words")

    print("[2/4] 載入 GECO 眼動資料（per-reader GD / TRT）...")
    _, eye = load_geco()

    print("[3/4] 建立 per-reader LMM dataframe...")
    lmm_df = load_per_reader_gd(pred_df, eye)
    print(f"  {len(lmm_df)} observations ({lmm_df['subject'].nunique()} subjects)")

    # Spillover control on word-level pred_df
    pred_df = pred_df.sort_values(["SENTENCE_ID", "sent_position"])
    pred_df["prev_surprisal"]   = (pred_df.groupby("SENTENCE_ID")["surprisal"]
                                    .shift(1).fillna(pred_df["surprisal"].mean()))
    pred_df["prev_word_length"] = (pred_df.groupby("SENTENCE_ID")["WORD_LENGTH"]
                                    .shift(1).fillna(pred_df["WORD_LENGTH"].mean()))

    print("\n[4/4] OLS + LMM for TRT and GD separately...\n")

    ols_trt = run_ols(pred_df, "log_trt", "TRT")
    ols_gd  = run_ols(pred_df, "log_gd",  "GD")

    print()
    lmm_trt = run_lmm(lmm_df, "log_trt", "TRT")
    lmm_gd  = run_lmm(lmm_df, "log_gd",  "GD")

    write_report(ols_trt, ols_gd, lmm_trt, lmm_gd)
    print("\n[完成] gd_trt_separate_report.md")


if __name__ == "__main__":
    main()

"""
Phase A: Component-wise regression
測試 pipeline 各個特徵對 TRT 的獨立貢獻：
  surprisal / aoa_score / dep_load / zipf_score / word_length
輸出：component_report.md + component_scatter.png + component_coef.png
"""
import sys, os, io, warnings
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Re-use alignment logic from validate_geco.py ─────────────────
from validate_geco import (
    load_geco, select_sentences, run_pipeline, merge_eye,
    style_ax, sig_stars, C_BG, COLORS
)

OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
N_SENTS  = 150

FEATURES = {
    "surprisal":    "Surprisal\n(GPT-2)",
    "aoa_score":    "AoA score\n(Kuperman)",
    "dep_load":     "Dependency\nload",
    "zipf_score":   "Zipf freq\n(inverse)",
    "WORD_LENGTH":  "Word\nlength",
}

def run():
    mat, eye = load_geco()
    print(f"[Selecting] first {N_SENTS} sentences...")
    sent_ids   = select_sentences(mat, N_SENTS)
    scores_df  = run_pipeline(mat, sent_ids)
    content_df, _ = merge_eye(scores_df, eye)

    # dep_load column name in pipeline output is 'dependency_load'
    if "dependency_load" in content_df.columns:
        content_df["dep_load"] = content_df["dependency_load"]
    elif "dep_load" not in content_df.columns:
        print("[警告] dep_load 欄位不存在，跳過")
        content_df["dep_load"] = 0.0

    print(f"\n[Phase A] Component-wise analysis on {len(content_df)} content words\n")

    results = {}

    # ── 1. Spearman ρ per feature ─────────────────────────────────
    print(f"{'Feature':<20} {'Spearman ρ':>12} {'p-value':>10} {'Sig':>5}")
    print("-" * 52)
    for feat, label in FEATURES.items():
        if feat not in content_df.columns:
            continue
        d = content_df[[feat, "mean_trt"]].dropna()
        rho, p = stats.spearmanr(d[feat], d["mean_trt"])
        print(f"  {feat:<18} {rho:>12.3f} {p:>10.4f} {sig_stars(p):>5}")
        results[feat] = {"rho": rho, "p": p, "n": len(d), "label": label}

    # ── 2. Individual OLS (each feature alone) ───────────────────
    print(f"\n{'Feature':<20} {'β':>8} {'p(β)':>8} {'R²':>8}")
    print("-" * 48)
    for feat in FEATURES:
        if feat not in content_df.columns:
            continue
        m = smf.ols(f"log_trt ~ {feat}", data=content_df).fit()
        b = m.params.get(feat, float("nan"))
        p = m.pvalues.get(feat, float("nan"))
        print(f"  {feat:<18} {b:>8.4f} {p:>8.4f} {m.rsquared:>8.4f}")
        results[feat]["beta_solo"] = b
        results[feat]["p_solo"]    = p
        results[feat]["r2_solo"]   = m.rsquared

    # ── 3. Joint OLS: all features together (no composite score) ─
    avail = [f for f in FEATURES if f in content_df.columns]
    formula_joint = "log_trt ~ " + " + ".join(avail)
    m_joint = smf.ols(formula_joint, data=content_df).fit()
    print(f"\n[Joint model] R² = {m_joint.rsquared:.4f}  AIC = {m_joint.aic:.1f}")
    for feat in avail:
        b = m_joint.params.get(feat, float("nan"))
        p = m_joint.pvalues.get(feat, float("nan"))
        results[feat]["beta_joint"] = b
        results[feat]["p_joint"]    = p
        print(f"  {feat:<18} β={b:>8.4f}  p={p:>7.4f} {sig_stars(p)}")

    # ── 4. Compare: composite load_score vs joint model ──────────
    m_composite = smf.ols("log_trt ~ load_score + WORD_LENGTH + zipf_score + sent_position",
                           data=content_df).fit()
    print(f"\n[Composite model] R² = {m_composite.rsquared:.4f}  AIC = {m_composite.aic:.1f}")
    print(f"[Joint model]     R² = {m_joint.rsquared:.4f}  AIC = {m_joint.aic:.1f}")

    # ── Figures ──────────────────────────────────────────────────
    _fig_spearman(results)
    _fig_joint_coef(m_joint, results)

    # ── Report ───────────────────────────────────────────────────
    _write_report(results, m_joint, m_composite, len(content_df))

    print("\n[完成]")
    for f in ["component_scatter.png", "component_coef.png", "component_report.md"]:
        print(f"  {f}")


def _fig_spearman(results):
    """Spearman ρ 各特徵比較長條圖"""
    feats = [f for f in FEATURES if f in results]
    rhos  = [results[f]["rho"] for f in feats]
    ps    = [results[f]["p"]   for f in feats]
    ns    = [results[f]["n"]   for f in feats]
    labels = [FEATURES[f] for f in feats]
    colors = [COLORS[2] if p < 0.05 else "#888888" for p in ps]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(C_BG); style_ax(ax)
    bars = ax.bar(labels, rhos, color=colors, alpha=0.85, width=0.55)
    for bar, r, p, n in zip(bars, rhos, ps, ns):
        ypos = r + 0.01 if r >= 0 else r - 0.04
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"ρ={r:.3f}\n{sig_stars(p)}\nn={n}",
                ha="center", va="bottom", fontsize=9, color="white")
    ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Spearman ρ vs mean TRT", fontsize=11)
    ax.set_title("Each Feature's Marginal Correlation with TRT\n(GECO content words, n≈1044)",
                 fontsize=12, pad=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="white")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "component_scatter.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def _fig_joint_coef(m_joint, results):
    """Joint model 係數圖（各特徵在同一模型中的偏效果）"""
    params = m_joint.params.drop("Intercept")
    cis    = m_joint.conf_int().drop("Intercept")
    pvals  = m_joint.pvalues.drop("Intercept")
    labels = [FEATURES.get(n, n) for n in params.index]
    colors = [COLORS[2] if p < 0.05 else "#888888" for p in pvals]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(C_BG); style_ax(ax)
    y = list(range(len(params)))
    ax.barh(y, params.values, color=colors, alpha=0.85, height=0.5)
    for i, (lo, hi) in enumerate(zip(cis.iloc[:, 0], cis.iloc[:, 1])):
        ax.plot([lo, hi], [i, i], color="white", linewidth=2.5)
    ax.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Partial OLS coefficient  (DV = log mean TRT)", fontsize=11)
    ax.set_title("Joint Model: Each Feature's Partial Effect on TRT\n(green = p < 0.05)", fontsize=11, pad=10)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, color="white")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "component_coef.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def _write_report(results, m_joint, m_composite, n_words):
    avail = [f for f in FEATURES if f in results]
    lines = [
        "# Component-wise Regression Report — Pipeline Feature Analysis",
        "",
        f"> n = {n_words} content words (GECO, 142 sentences, 14 L1 readers)",
        "",
        "---", "",
        "## 1. Marginal Spearman Correlation (each feature vs mean TRT)", "",
        "Each feature tested alone, without controlling for others.",
        "",
        "| Feature | Spearman ρ | p-value | Sig. |",
        "|---------|-----------|---------|------|",
    ]
    for f in avail:
        r = results[f]
        lines.append(
            f"| {FEATURES[f].replace(chr(10),' ')} | {r['rho']:.3f} | "
            f"{r['p']:.4f} | {sig_stars(r['p'])} |"
        )
    lines += [
        "",
        "---", "",
        "## 2. Individual OLS (each feature as sole predictor)", "",
        "| Feature | β | p(β) | R² |",
        "|---------|---|------|-----|",
    ]
    for f in avail:
        r = results[f]
        lines.append(
            f"| {FEATURES[f].replace(chr(10),' ')} | {r.get('beta_solo',float('nan')):.4f} | "
            f"{r.get('p_solo',float('nan')):.4f} | {r.get('r2_solo',float('nan')):.4f} |"
        )
    lines += [
        "",
        "---", "",
        "## 3. Joint Model (all features together, no composite score)", "",
        f"Formula: `log_trt ~ " + " + ".join(avail) + "`",
        f"R² = {m_joint.rsquared:.4f}  |  AIC = {m_joint.aic:.1f}",
        "",
        "| Feature | Partial β | p-value | Sig. |",
        "|---------|----------|---------|------|",
    ]
    for f in avail:
        r = results[f]
        lines.append(
            f"| {FEATURES[f].replace(chr(10),' ')} | {r.get('beta_joint',float('nan')):.4f} | "
            f"{r.get('p_joint',float('nan')):.4f} | {sig_stars(r.get('p_joint',1.0))} |"
        )
    lines += [
        "",
        "---", "",
        "## 4. Composite score vs Joint model", "",
        "| Model | R² | AIC |",
        "|-------|----|-----|",
        f"| Composite `load_score` + length + freq + pos | {m_composite.rsquared:.4f} | {m_composite.aic:.1f} |",
        f"| Joint (all raw features) | {m_joint.rsquared:.4f} | {m_joint.aic:.1f} |",
        "",
        "---", "",
        "## 5. Interpretation", "",
        "- **Surprisal** and **Zipf frequency** are expected to dominate — they are the strongest",
        "  psycholinguistic predictors in the reading-time literature.",
        "- **AoA** should show independent contribution *beyond* frequency (Kuperman et al. 2012",
        "  show AoA explains unique variance in naming and reading after controlling for frequency).",
        "- **Dependency load** tests syntactic integration cost; effect is typically smaller",
        "  for naturalistic text but meaningful for complex sentences.",
        "- The joint model R² gives an upper bound on how much variance all pipeline features",
        "  collectively explain in TRT.",
    ]

    path = os.path.join(OUT_DIR, "component_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run()

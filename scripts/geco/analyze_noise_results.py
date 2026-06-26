"""Summarize noise_tolerance_results.csv → plain text under docs/NeurIPS/experiments/."""
import os
from datetime import datetime

import pandas as pd


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    csv_path = os.path.join(project_root, "data", "geco", "benchmark", "noise_tolerance_results.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    out_dir = os.path.join(project_root, "output", "NeurIPS", "experiments")
    os.makedirs(out_dir, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(out_dir, f"{today}_noise_stress_test_summary.txt")

    n_pairs = df.groupby(["Lang", "Subject"]).ngroups
    n_trials = len(df["Trial"].unique())
    drift_levels = sorted(df["Drift_Y"].unique())

    lines = []
    lines.append("LexiGaze — Noise stress test (aggregated)")
    lines.append(f"Generated: {today}")
    lines.append(f"Source CSV: data/geco/benchmark/noise_tolerance_results.csv")
    lines.append("")
    lines.append(
        "METRIC: Trajectory recovery = fraction of fixations where the decoded word lies on the "
        "same typographic line as the ground-truth word (lines inferred from layout true_y gaps > 22px)."
    )
    lines.append(
        "This definition applies to STOCK-T and spatial baselines (fair comparison; no drift oracle for baseline)."
    )
    lines.append("")
    lines.append(f"Participants (Lang x Subject pairs): {n_pairs}")
    lines.append(f"Unique trial IDs in table: {n_trials}")
    lines.append(f"Drift levels (px): {drift_levels}")
    lines.append("")

    # Backward compatible: accept either old *_Rec columns or the new dual-metric columns.
    if "STOCK-T_Edge_LineRec" in df.columns:
        agg_cols = ["STOCK-T_Edge_LineRec", "STOCK-T_Surprisal_LineRec", "Baseline_LineRec"]
        has_y = "BaselineY_LineRec" in df.columns
        y_col = "BaselineY_LineRec"
    else:
        agg_cols = ["STOCK-T_Edge_Rec", "STOCK-T_Surprisal_Rec", "Baseline_Rec"]
        has_y = "BaselineY_Rec" in df.columns
        y_col = "BaselineY_Rec"
    if has_y:
        agg_cols.append("BaselineY_Rec")

    df_agg = df.groupby("Drift_Y", as_index=False)[agg_cols].mean()

    lines.append("Mean line-recovery (%) by drift_y")
    header = "drift_px\tSTOCK-T_Edge\tSTOCK-T_Surprisal\tBaseline_spatial2D"
    if has_y:
        header += "\tBaseline_Y_only"
    lines.append(header)
    for _, row in df_agg.iterrows():
        if "STOCK-T_Edge_LineRec" in df.columns:
            line = (
                f"{row['Drift_Y']:.0f}\t{row['STOCK-T_Edge_LineRec']:.2f}\t"
                f"{row['STOCK-T_Surprisal_LineRec']:.2f}\t{row['Baseline_LineRec']:.2f}"
            )
        else:
            line = (
                f"{row['Drift_Y']:.0f}\t{row['STOCK-T_Edge_Rec']:.2f}\t"
                f"{row['STOCK-T_Surprisal_Rec']:.2f}\t{row['Baseline_Rec']:.2f}"
            )
        if has_y:
            line += f"\t{row[y_col]:.2f}"
        lines.append(line)
    lines.append("")

    df_lang = df.groupby(["Drift_Y", "Lang"], as_index=False)[agg_cols].mean()

    lines.append("Mean line-recovery (%) by drift_y and Lang")
    header = "drift_px\tLang\tSTOCK-T_Edge\tSTOCK-T_Surprisal\tBaseline_spatial2D"
    if has_y:
        header += "\tBaseline_Y_only"
    lines.append(header)
    for _, row in df_lang.iterrows():
        line = (
            f"{row['Drift_Y']:.0f}\t{row['Lang']}\t{row['STOCK-T_Edge_Rec']:.2f}\t"
            f"{row['STOCK-T_Surprisal_Rec']:.2f}\t{row['Baseline_Rec']:.2f}"
        )
        if has_y:
            line += f"\t{row['BaselineY_Rec']:.2f}"
        lines.append(line)
    lines.append("")

    per_subject = df.groupby(["Lang", "Subject"]).size()
    lines.append(f"Rows per (Lang, Subject): min={per_subject.min()}, max={per_subject.max()}, mean={per_subject.mean():.1f}")

    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(text)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

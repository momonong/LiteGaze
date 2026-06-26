"""Summarize full_corpus_results.csv → plain text under docs/NeurIPS/experiments/."""
import os
from datetime import datetime

import pandas as pd


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    csv_path = os.path.join(project_root, "data", "geco", "benchmark", "full_corpus_results.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    out_dir = os.path.join(project_root, "output", "NeurIPS", "experiments")
    os.makedirs(out_dir, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(out_dir, f"{today}_full_population_benchmark_summary.txt")

    metrics = [c for c in df.columns if "_Acc" in c or "_Rec" in c or "_Top3" in c]
    df_agg = df.groupby("Lang")[metrics].agg(["mean", "std"]).reset_index()

    n_pairs = df.groupby(["Lang", "Subject"]).ngroups
    l1_n = df[df["Lang"] == "L1"]["Subject"].nunique()
    l2_n = df[df["Lang"] == "L2"]["Subject"].nunique()
    total_trials = len(df)

    lines = []
    lines.append("LexiGaze — Full GECO corpus benchmark (+45px vertical drift, axis-aligned Gaussian noise)")
    lines.append(f"Generated: {today}")
    lines.append(f"Source CSV: data/geco/benchmark/full_corpus_results.csv")
    lines.append("")
    lines.append(
        "METRIC: Trajectory recovery = line-level consistency (predicted word same typographic line as target)."
    )
    lines.append("")
    lines.append(f"Total trial rows: {total_trials}")
    lines.append(f"Participants L1: {l1_n}, L2: {l2_n}, combined (Lang x Subject): {n_pairs}")
    lines.append("")

    variants = [
        ("STOCK-T (Edge/Uniform)", "STOCK-T_Edge"),
        ("STOCK-T (Surprisal)", "STOCK-T_Surprisal"),
        ("w/o POM (Rule-based)", "w/o_POM"),
        ("w/o EM (Kalman)", "w/o_EM"),
        ("w/o Temp (Nearest)", "w/o_Temp"),
    ]

    lines.append("Mean ± std (%) by model (columns: L1_mean, L1_std, L2_mean, L2_std, overall_mean, overall_std)")
    for label, key in variants:
        for suf, mname in [("_Acc", "Word_Acc"), ("_Rec", "Trajectory_LineRec")]:
            col = key + suf
            l1m = df_agg.loc[df_agg["Lang"] == "L1", (col, "mean")].values[0]
            l1s = df_agg.loc[df_agg["Lang"] == "L1", (col, "std")].values[0]
            l2m = df_agg.loc[df_agg["Lang"] == "L2", (col, "mean")].values[0]
            l2s = df_agg.loc[df_agg["Lang"] == "L2", (col, "std")].values[0]
            om = df[col].mean()
            os_ = df[col].std()
            lines.append(
                f"{label} | {mname}: L1 {l1m:.2f}±{l1s:.2f}, L2 {l2m:.2f}±{l2s:.2f}, all {om:.2f}±{os_:.2f}"
            )

    lines.append("")
    trial_counts = df.groupby(["Lang", "Subject"]).size()
    lines.append(
        f"Trials per (Lang, Subject): min={trial_counts.min()}, max={trial_counts.max()}, mean={trial_counts.mean():.1f}"
    )

    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(text)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

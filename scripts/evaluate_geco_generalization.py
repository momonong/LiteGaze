"""Leakage-resistant GECO cross-subject and cross-trial evaluation.

The preregistered protocol lives in
``docs/GECO_GENERALIZATION_PROTOCOL_2026-08-03.md``. This script intentionally
uses only cached, label-free text features for its primary model and requires no
GPU or language-model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

if __package__:
    from .experiment_manifest import (
        capture_source_snapshot,
        write_experiment_manifest,
    )
else:
    from experiment_manifest import capture_source_snapshot, write_experiment_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/geco/population"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
PROTOCOL_PATH = PROJECT_ROOT / "docs/GECO_GENERALIZATION_PROTOCOL_2026-08-03.md"

PROTOCOL_VERSION = 1
RANDOM_SEED = 20260803
N_SUBJECT_FOLDS = 5
N_TRIAL_FOLDS = 5
RIDGE_ALPHA = 1.0
BOOTSTRAP_SAMPLES = 10_000
SIGN_FLIP_SAMPLES = 100_000
EXPECTED_SUBJECTS = 37
EXPECTED_PARTICIPANT_TRIALS = 5_892

PRIMARY_FEATURES = (
    "log_surprisal",
    "log_attention",
    "word_length",
    "is_l2",
)
DOUBLE_HOLDOUT_MODELS = (
    "ridge_text_only",
    "surprisal_only",
    "attention_only",
    "word_length_only",
    "cognitive_mass_provenance_risk",
)

LAYOUT_COLUMNS = (
    "WORD_ID_WITHIN_TRIAL",
    "WORD",
    "surprisal_score",
    "attention_score",
    "cognitive_mass",
)
FIXATION_COLUMNS = ("WORD_ID_WITHIN_TRIAL", "reading_time")


def _stable_digest(value: str, seed: int = RANDOM_SEED) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def assign_balanced_folds(
    values: Iterable[str],
    n_folds: int,
    *,
    seed: int = RANDOM_SEED,
    strata: Mapping[str, str] | None = None,
) -> dict[str, int]:
    """Assign deterministic, outcome-independent, approximately balanced folds."""
    unique_values = sorted(set(values))
    grouped: dict[str, list[str]] = {}
    for value in unique_values:
        stratum = strata[value] if strata is not None else "all"
        grouped.setdefault(stratum, []).append(value)

    assignments: dict[str, int] = {}
    for stratum in sorted(grouped):
        ordered = sorted(
            grouped[stratum],
            key=lambda value: (_stable_digest(f"{stratum}|{value}", seed), value),
        )
        for index, value in enumerate(ordered):
            assignments[value] = index % n_folds
    return assignments


def fit_ridge(
    features: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float = RIDGE_ALPHA,
) -> dict[str, np.ndarray]:
    """Fit a standardized Ridge model with an unpenalized intercept."""
    if features.ndim != 2 or target.ndim != 1 or len(features) != len(target):
        raise ValueError("features must be 2D and aligned with a 1D target")
    if not len(target):
        raise ValueError("cannot fit Ridge on an empty training set")

    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    standardized = (features - mean) / scale
    design = np.column_stack([np.ones(len(standardized)), standardized])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ target,
    )
    return {"mean": mean, "scale": scale, "coefficients": coefficients}


def predict_ridge(model: Mapping[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    standardized = (features - model["mean"]) / model["scale"]
    design = np.column_stack([np.ones(len(standardized)), standardized])
    return design @ model["coefficients"]


def _safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    valid = np.isfinite(left_array) & np.isfinite(right_array)
    if valid.sum() < 3:
        return math.nan
    if np.ptp(left_array[valid]) == 0 or np.ptp(right_array[valid]) == 0:
        return math.nan
    return float(spearmanr(left_array[valid], right_array[valid]).statistic)


def _roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores_array)
    labels_array = labels_array[valid]
    scores_array = scores_array[valid]
    positives = labels_array == 1
    n_positive = int(positives.sum())
    n_negative = int((~positives).sum())
    if n_positive == 0 or n_negative == 0:
        return math.nan
    ranks = rankdata(scores_array, method="average")
    return float(
        (ranks[positives].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def add_population_priors(frame: pd.DataFrame) -> pd.DataFrame:
    """Add same-item population priors after subtracting each participant."""
    result = frame.copy()
    keys = ["lang", "trial", "word_id"]
    groups = result.groupby(keys, sort=False, observed=True)

    fixated = result["fixated"].astype(float)
    group_size = groups["fixated"].transform("size").astype(float)
    group_fixated = groups["fixated"].transform("sum").astype(float)
    other_count = group_size - 1.0
    result["population_fixation_rate"] = np.where(
        other_count > 0,
        (group_fixated - fixated) / other_count,
        np.nan,
    )

    own_log_time = result["log_reading_time"].fillna(0.0)
    group_log_sum = groups["log_reading_time"].transform("sum").fillna(0.0)
    group_duration_count = groups["log_reading_time"].transform("count").astype(float)
    other_duration_count = group_duration_count - fixated
    result["population_log_duration"] = np.where(
        other_duration_count > 0,
        (group_log_sum - own_log_time) / other_duration_count,
        np.nan,
    )
    return result


def cross_fitted_double_holdout(
    frame: pd.DataFrame,
    *,
    alpha: float = RIDGE_ALPHA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict every positive-duration row without its subject or trial fold."""
    positive = frame.loc[frame["fixated"]].copy().reset_index(drop=True)
    features = positive.loc[:, PRIMARY_FEATURES].to_numpy(dtype=float)
    target = positive["log_reading_time"].to_numpy(dtype=float)
    predictions = np.full(len(positive), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []

    for subject_fold in range(N_SUBJECT_FOLDS):
        for trial_fold in range(N_TRIAL_FOLDS):
            train_mask = (positive["subject_fold"].to_numpy() != subject_fold) & (
                positive["trial_fold"].to_numpy() != trial_fold
            )
            test_mask = (positive["subject_fold"].to_numpy() == subject_fold) & (
                positive["trial_fold"].to_numpy() == trial_fold
            )
            if not train_mask.any() or not test_mask.any():
                raise ValueError(
                    f"empty double-holdout split subject_fold={subject_fold}, "
                    f"trial_fold={trial_fold}"
                )

            model = fit_ridge(features[train_mask], target[train_mask], alpha=alpha)
            predictions[test_mask] = predict_ridge(model, features[test_mask])
            test_target = target[test_mask]
            test_prediction = predictions[test_mask]
            fold_rows.append(
                {
                    "subject_fold": subject_fold,
                    "trial_fold": trial_fold,
                    "train_rows": int(train_mask.sum()),
                    "test_rows": int(test_mask.sum()),
                    "ridge_spearman_rho": _safe_spearman(test_prediction, test_target),
                    "ridge_mae_log": float(
                        np.mean(np.abs(test_prediction - test_target))
                    ),
                }
            )

    if not np.isfinite(predictions).all():
        raise RuntimeError("not every duration row received a cross-fitted prediction")

    positive["ridge_text_only"] = predictions
    positive["surprisal_only"] = positive["surprisal_score"].to_numpy(dtype=float)
    positive["attention_only"] = positive["attention_score"].to_numpy(dtype=float)
    positive["word_length_only"] = positive["word_length"].to_numpy(dtype=float)
    positive["cognitive_mass_provenance_risk"] = positive["cognitive_mass"].to_numpy(
        dtype=float
    )
    return positive, pd.DataFrame(fold_rows)


def _file_sha256(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.digest()


def load_population(data_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load paired layout/fixation files and calculate a stable tree digest."""
    layout_paths = sorted(data_root.glob("*/*/trial_*/layout.csv"))
    if not layout_paths:
        raise FileNotFoundError(f"no population layouts found under {data_root}")

    frames: list[pd.DataFrame] = []
    tree_digest = hashlib.sha256()
    total_bytes = 0
    used_files = 0

    for index, layout_path in enumerate(layout_paths, start=1):
        fixation_path = layout_path.with_name("fixations.csv")
        if not fixation_path.is_file():
            raise FileNotFoundError(f"missing paired fixation file: {fixation_path}")

        relative_parts = layout_path.relative_to(data_root).parts
        lang, subject, trial = relative_parts[:3]
        for path in (layout_path, fixation_path):
            relative = path.relative_to(data_root).as_posix()
            size = path.stat().st_size
            tree_digest.update(relative.encode("utf-8"))
            tree_digest.update(b"\0")
            tree_digest.update(str(size).encode("ascii"))
            tree_digest.update(b"\0")
            tree_digest.update(_file_sha256(path))
            total_bytes += size
            used_files += 1

        layout = pd.read_csv(layout_path, usecols=list(LAYOUT_COLUMNS))
        fixations = pd.read_csv(
            fixation_path,
            usecols=list(FIXATION_COLUMNS),
            na_values=["."],
            keep_default_na=True,
        )
        if layout["WORD_ID_WITHIN_TRIAL"].duplicated().any():
            raise ValueError(f"duplicate layout word IDs: {layout_path}")
        if fixations["WORD_ID_WITHIN_TRIAL"].duplicated().any():
            raise ValueError(f"duplicate fixation word IDs: {fixation_path}")

        merged = layout.merge(
            fixations,
            on="WORD_ID_WITHIN_TRIAL",
            how="left",
            validate="one_to_one",
        )
        reading_time = pd.to_numeric(merged["reading_time"], errors="coerce")
        surprisal = pd.to_numeric(merged["surprisal_score"], errors="coerce")
        attention = pd.to_numeric(merged["attention_score"], errors="coerce")
        cognitive_mass = pd.to_numeric(merged["cognitive_mass"], errors="coerce")
        if not (
            np.isfinite(surprisal).all()
            and np.isfinite(attention).all()
            and np.isfinite(cognitive_mass).all()
        ):
            raise ValueError(f"non-finite required text feature: {layout_path}")

        fixated = np.isfinite(reading_time) & (reading_time > 0)
        frame = pd.DataFrame(
            {
                "lang": lang,
                "subject": subject,
                "subject_key": f"{lang}|{subject}",
                "trial": trial,
                "word_id": merged["WORD_ID_WITHIN_TRIAL"].astype(str),
                "word_length": merged["WORD"].astype(str).str.strip().str.len(),
                "surprisal_score": surprisal.astype(float),
                "attention_score": attention.astype(float),
                "cognitive_mass": cognitive_mass.astype(float),
                "reading_time": reading_time.astype(float),
                "fixated": fixated.astype(bool),
            }
        )
        frame["log_reading_time"] = np.where(
            frame["fixated"], np.log1p(frame["reading_time"]), np.nan
        )
        frame["log_surprisal"] = np.log1p(np.clip(frame["surprisal_score"], 0.0, None))
        frame["log_attention"] = np.log1p(np.clip(frame["attention_score"], 0.0, None))
        frame["is_l2"] = (frame["lang"] == "L2").astype(float)
        frames.append(frame)

        if index % 500 == 0 or index == len(layout_paths):
            print(f"Loaded {index:,}/{len(layout_paths):,} participant-trials")

    population = pd.concat(frames, ignore_index=True)
    subject_keys = sorted(population["subject_key"].unique())
    trials = sorted(population["trial"].unique())
    subject_strata = {
        subject_key: subject_key.split("|", 1)[0] for subject_key in subject_keys
    }
    subject_folds = assign_balanced_folds(
        subject_keys,
        N_SUBJECT_FOLDS,
        strata=subject_strata,
    )
    trial_folds = assign_balanced_folds(trials, N_TRIAL_FOLDS)
    population["subject_fold"] = population["subject_key"].map(subject_folds)
    population["trial_fold"] = population["trial"].map(trial_folds)

    fingerprint = {
        "path": data_root.relative_to(PROJECT_ROOT).as_posix(),
        "patterns": ["*/*/trial_*/layout.csv", "*/*/trial_*/fixations.csv"],
        "sha256_tree": tree_digest.hexdigest(),
        "file_count": used_files,
        "size_bytes": total_bytes,
        "participant_trial_count": len(layout_paths),
        "row_count": len(population),
        "fixated_row_count": int(population["fixated"].sum()),
        "subject_count": len(subject_keys),
        "subjects_by_language": {
            lang: int(
                population.loc[population["lang"] == lang, "subject_key"].nunique()
            )
            for lang in sorted(population["lang"].unique())
        },
        "unique_trial_count": len(trials),
        "subject_fold_assignments": subject_folds,
        "trial_fold_assignments": trial_folds,
    }
    return population, fingerprint


def _duration_subject_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (lang, subject_key), group in predictions.groupby(
        ["lang", "subject_key"], sort=True, observed=True
    ):
        target = group["log_reading_time"].to_numpy(dtype=float)
        for model_name in DOUBLE_HOLDOUT_MODELS:
            values = group[model_name].to_numpy(dtype=float)
            rows.append(
                {
                    "protocol": "new_reader_new_trial_double_holdout",
                    "outcome": "conditional_log_reading_time",
                    "model": model_name,
                    "lang": lang,
                    "subject": subject_key,
                    "n": len(group),
                    "spearman_rho": _safe_spearman(values, target),
                    "mae_log": (
                        float(np.mean(np.abs(values - target)))
                        if model_name == "ridge_text_only"
                        else math.nan
                    ),
                    "roc_auc": math.nan,
                    "brier": math.nan,
                }
            )
    return pd.DataFrame(rows)


def _known_passage_subject_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (lang, subject_key), group in frame.groupby(
        ["lang", "subject_key"], sort=True, observed=True
    ):
        fixation_scores = group["population_fixation_rate"].to_numpy(dtype=float)
        fixation_labels = group["fixated"].astype(int).to_numpy()
        fixation_valid = np.isfinite(fixation_scores)
        rows.append(
            {
                "protocol": "new_reader_known_passage_loso",
                "outcome": "fixation_occurrence",
                "model": "population_fixation_rate",
                "lang": lang,
                "subject": subject_key,
                "n": int(fixation_valid.sum()),
                "spearman_rho": math.nan,
                "mae_log": math.nan,
                "roc_auc": _roc_auc(fixation_labels, fixation_scores),
                "brier": float(
                    np.mean(
                        (
                            fixation_scores[fixation_valid]
                            - fixation_labels[fixation_valid]
                        )
                        ** 2
                    )
                ),
            }
        )

        duration_valid = (
            group["fixated"]
            & np.isfinite(group["population_log_duration"])
            & np.isfinite(group["log_reading_time"])
        )
        duration_group = group.loc[duration_valid]
        duration_prediction = duration_group["population_log_duration"].to_numpy(
            dtype=float
        )
        duration_target = duration_group["log_reading_time"].to_numpy(dtype=float)
        rows.append(
            {
                "protocol": "new_reader_known_passage_loso",
                "outcome": "conditional_log_reading_time",
                "model": "population_duration_prior",
                "lang": lang,
                "subject": subject_key,
                "n": len(duration_group),
                "spearman_rho": _safe_spearman(duration_prediction, duration_target),
                "mae_log": float(
                    np.mean(np.abs(duration_prediction - duration_target))
                ),
                "roc_auc": math.nan,
                "brier": math.nan,
            }
        )
    return pd.DataFrame(rows)


def _stratified_bootstrap(
    values: pd.DataFrame,
    *,
    value_column: str,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    clean = values.loc[np.isfinite(values[value_column]), ["lang", value_column]]
    observed = float(clean[value_column].mean())
    rng = np.random.default_rng(seed)
    boot = np.zeros(samples, dtype=float)
    groups = [
        group[value_column].to_numpy(dtype=float)
        for _, group in clean.groupby("lang", sort=True, observed=True)
    ]
    total_count = sum(len(group) for group in groups)
    for group in groups:
        indices = rng.integers(0, len(group), size=(samples, len(group)))
        boot += group[indices].sum(axis=1) / total_count
    lower, upper = np.quantile(boot, [0.025, 0.975])
    return observed, float(lower), float(upper)


def _sign_flip_pvalue(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    observed = abs(float(array.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    processed = 0
    chunk_size = 10_000
    while processed < samples:
        chunk = min(chunk_size, samples - processed)
        signs = rng.choice((-1.0, 1.0), size=(chunk, len(array)))
        null_values = np.abs((signs * array).mean(axis=1))
        extreme += int(np.count_nonzero(null_values >= observed))
        processed += chunk
    return float((extreme + 1) / (samples + 1))


def summarize_subject_metrics(
    metrics: pd.DataFrame,
    *,
    bootstrap_samples: int,
    sign_flip_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    duration = metrics.loc[
        (metrics["protocol"] == "new_reader_new_trial_double_holdout")
        & (metrics["outcome"] == "conditional_log_reading_time")
    ]
    for model_index, model_name in enumerate(DOUBLE_HOLDOUT_MODELS):
        model_rows = duration.loc[duration["model"] == model_name]
        mean_rho, lower, upper = _stratified_bootstrap(
            model_rows,
            value_column="spearman_rho",
            samples=bootstrap_samples,
            seed=RANDOM_SEED + model_index,
        )
        summaries.append(
            {
                "protocol": "new_reader_new_trial_double_holdout",
                "outcome": "conditional_log_reading_time",
                "model": model_name,
                "subject_count": int(model_rows["subject"].nunique()),
                "macro_mean_spearman_rho": mean_rho,
                "bootstrap_ci_95": [lower, upper],
                "mean_mae_log": (
                    float(model_rows["mae_log"].mean())
                    if np.isfinite(model_rows["mae_log"]).any()
                    else None
                ),
                "primary_eligible": model_name != "cognitive_mass_provenance_risk",
            }
        )

    ridge = duration.loc[duration["model"] == "ridge_text_only"].sort_values("subject")
    surprisal = duration.loc[duration["model"] == "surprisal_only"].sort_values(
        "subject"
    )
    if ridge["subject"].tolist() != surprisal["subject"].tolist():
        raise RuntimeError("paired primary comparator subjects do not align")
    primary_tests = {
        "ridge_vs_zero_two_sided_p": _sign_flip_pvalue(
            ridge["spearman_rho"],
            samples=sign_flip_samples,
            seed=RANDOM_SEED + 100,
        ),
        "ridge_vs_surprisal_paired_two_sided_p": _sign_flip_pvalue(
            ridge["spearman_rho"].to_numpy() - surprisal["spearman_rho"].to_numpy(),
            samples=sign_flip_samples,
            seed=RANDOM_SEED + 101,
        ),
        "ridge_minus_surprisal_macro_rho": float(
            (
                ridge["spearman_rho"].to_numpy() - surprisal["spearman_rho"].to_numpy()
            ).mean()
        ),
    }

    known = metrics.loc[metrics["protocol"] == "new_reader_known_passage_loso"]
    for model_name, outcome, value_column in (
        ("population_duration_prior", "conditional_log_reading_time", "spearman_rho"),
        ("population_fixation_rate", "fixation_occurrence", "roc_auc"),
    ):
        model_rows = known.loc[
            (known["model"] == model_name) & (known["outcome"] == outcome)
        ]
        mean_value, lower, upper = _stratified_bootstrap(
            model_rows,
            value_column=value_column,
            samples=bootstrap_samples,
            seed=RANDOM_SEED + len(summaries),
        )
        summaries.append(
            {
                "protocol": "new_reader_known_passage_loso",
                "outcome": outcome,
                "model": model_name,
                "subject_count": int(model_rows["subject"].nunique()),
                f"macro_mean_{value_column}": mean_value,
                "bootstrap_ci_95": [lower, upper],
                "mean_mae_log": (
                    float(model_rows["mae_log"].mean())
                    if np.isfinite(model_rows["mae_log"]).any()
                    else None
                ),
                "mean_brier": (
                    float(model_rows["brier"].mean())
                    if np.isfinite(model_rows["brier"]).any()
                    else None
                ),
                "primary_eligible": False,
            }
        )
    return summaries, primary_tests


def _write_report(
    path: Path,
    *,
    fingerprint: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    primary_tests: Mapping[str, Any],
    duration_seconds: float,
) -> None:
    double_holdout = {
        row["model"]: row
        for row in summaries
        if row["protocol"] == "new_reader_new_trial_double_holdout"
    }
    known = {
        row["model"]: row
        for row in summaries
        if row["protocol"] == "new_reader_known_passage_loso"
    }

    rows = []
    for model_name in DOUBLE_HOLDOUT_MODELS:
        row = double_holdout[model_name]
        lower, upper = row["bootstrap_ci_95"]
        note = (
            "exploratory; upstream target provenance risk"
            if model_name == "cognitive_mass_provenance_risk"
            else "label-free input"
        )
        rows.append(
            f"| `{model_name}` | {row['macro_mean_spearman_rho']:.4f} "
            f"| [{lower:.4f}, {upper:.4f}] | {note} |"
        )

    ridge = double_holdout["ridge_text_only"]
    prior_duration = known["population_duration_prior"]
    fixation_prior = known["population_fixation_rate"]
    report = f"""# GECO Cross-Subject Generalization Results

Protocol: `docs/GECO_GENERALIZATION_PROTOCOL_2026-08-03.md` (v{PROTOCOL_VERSION})

Run completed: {datetime.now(UTC).isoformat()}

Runtime: {duration_seconds:.2f} seconds, CPU-only

## Dataset

- {fingerprint["subject_count"]} participants ({fingerprint["subjects_by_language"]})
- {fingerprint["participant_trial_count"]:,} participant-trials
- {fingerprint["row_count"]:,} word observations; {fingerprint["fixated_row_count"]:,} positive-duration observations
- {fingerprint["file_count"]:,} source files / {fingerprint["size_bytes"]:,} bytes
- Dataset tree SHA-256: `{fingerprint["sha256_tree"]}`

## Primary result: unseen reader and unseen trial

Each prediction was generated by a model trained without the test participant fold and without the test trial fold. The sole primary endpoint was macro per-participant Spearman correlation.

| Model | Macro Spearman rho | Stratified participant bootstrap 95% CI | Status |
| --- | ---: | ---: | --- |
{os.linesep.join(rows)}

Primary Ridge result: **rho = {ridge["macro_mean_spearman_rho"]:.4f}**, 95% CI **[{ridge["bootstrap_ci_95"][0]:.4f}, {ridge["bootstrap_ci_95"][1]:.4f}]**. Two-sided participant sign-flip p-value against zero: **{primary_tests["ridge_vs_zero_two_sided_p"]:.6f}**.

Prespecified Ridge minus surprisal difference: **{primary_tests["ridge_minus_surprisal_macro_rho"]:.4f}**; paired two-sided sign-flip p-value: **{primary_tests["ridge_vs_surprisal_paired_two_sided_p"]:.6f}**.

## Secondary result: new reader on known passages

- Other-reader duration prior macro Spearman: **{prior_duration["macro_mean_spearman_rho"]:.4f}**, 95% CI **[{prior_duration["bootstrap_ci_95"][0]:.4f}, {prior_duration["bootstrap_ci_95"][1]:.4f}]**; mean log-time MAE **{prior_duration["mean_mae_log"]:.4f}**.
- Other-reader fixation-rate prior macro ROC AUC: **{fixation_prior["macro_mean_roc_auc"]:.4f}**, 95% CI **[{fixation_prior["bootstrap_ci_95"][0]:.4f}, {fixation_prior["bootstrap_ci_95"][1]:.4f}]**; mean Brier score **{fixation_prior["mean_brier"]:.4f}**.

## Interpretation guardrails

- No question-answer benchmark or test-answer tuning was used.
- `cognitive_mass` is not eligible for the primary conclusion because its upstream XGBoost/Ridge provenance includes GECO reading-time supervision.
- The earlier single-trial fusion correlation uses target-derived dwell time and remains a descriptive calibration result, not evidence of predictive generalization.
- The primary study uses cached text features only and does not initialize CUDA.
- Results apply to these GECO populations and should not be generalized to new corpora without a corpus-level zero-shot evaluation.
"""
    path.write_text(report, encoding="utf-8")


def _write_plot(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    rows = [
        row
        for row in summaries
        if row["protocol"] == "new_reader_new_trial_double_holdout"
    ]
    labels = [row["model"].replace("_", "\n") for row in rows]
    values = np.array([row["macro_mean_spearman_rho"] for row in rows])
    lower = np.array([row["bootstrap_ci_95"][0] for row in rows])
    upper = np.array([row["bootstrap_ci_95"][1] for row in rows])
    colors = [
        "#d95f02" if "provenance_risk" in row["model"] else "#1b9e77" for row in rows
    ]

    fig, axis = plt.subplots(figsize=(11, 6))
    positions = np.arange(len(rows))
    axis.bar(positions, values, color=colors, alpha=0.85)
    axis.errorbar(
        positions,
        values,
        yerr=np.vstack([values - lower, upper - values]),
        fmt="none",
        ecolor="black",
        capsize=5,
        linewidth=1.2,
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Macro participant Spearman rho")
    axis.set_title("GECO new-reader + new-trial double holdout")
    axis.text(
        0.99,
        0.02,
        "Orange = provenance-risk diagnostic (not primary eligible)",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--sign-flip-samples", type=int, default=SIGN_FLIP_SAMPLES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    source_snapshot = capture_source_snapshot(PROJECT_ROOT)
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if os.environ.get("LEXIGAZE_DEVICE", "auto").strip().lower() != "cpu":
        raise RuntimeError(
            "preregistered run requires LEXIGAZE_DEVICE=cpu; no GPU is needed"
        )

    population, fingerprint = load_population(data_root)
    if fingerprint["subject_count"] != EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"expected {EXPECTED_SUBJECTS} subjects, found {fingerprint['subject_count']}"
        )
    if fingerprint["participant_trial_count"] != EXPECTED_PARTICIPANT_TRIALS:
        raise RuntimeError(
            "expected "
            f"{EXPECTED_PARTICIPANT_TRIALS} participant-trials, found "
            f"{fingerprint['participant_trial_count']}"
        )

    predictions, fold_metrics = cross_fitted_double_holdout(population)
    population_with_priors = add_population_priors(population)
    subject_metrics = pd.concat(
        [
            _duration_subject_metrics(predictions),
            _known_passage_subject_metrics(population_with_priors),
        ],
        ignore_index=True,
    )
    summaries, primary_tests = summarize_subject_metrics(
        subject_metrics,
        bootstrap_samples=args.bootstrap_samples,
        sign_flip_samples=args.sign_flip_samples,
    )

    fingerprint_path = output_dir / "geco_population_fingerprint.json"
    fold_path = output_dir / "geco_generalization_fold_metrics.csv"
    subject_path = output_dir / "geco_generalization_subject_metrics.csv"
    summary_path = output_dir / "geco_generalization_summary.json"
    report_path = output_dir / "geco_generalization_report.md"
    plot_path = output_dir / "geco_generalization.png"
    manifest_path = output_dir / "geco_generalization_manifest.json"

    fingerprint_path.write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fold_metrics.to_csv(fold_path, index=False)
    subject_metrics.to_csv(subject_path, index=False)
    duration_seconds = time.perf_counter() - started
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "duration_seconds": duration_seconds,
        "configuration": {
            "seed": RANDOM_SEED,
            "subject_folds": N_SUBJECT_FOLDS,
            "trial_folds": N_TRIAL_FOLDS,
            "ridge_alpha": RIDGE_ALPHA,
            "bootstrap_samples": args.bootstrap_samples,
            "sign_flip_samples": args.sign_flip_samples,
            "primary_features": list(PRIMARY_FEATURES),
        },
        "dataset": fingerprint,
        "model_summaries": summaries,
        "primary_tests": primary_tests,
        "leakage_controls": {
            "question_answer_dataset_used": False,
            "test_fold_used_for_tuning": False,
            "scaler_fit_on_training_only": True,
            "cognitive_mass_primary_eligible": False,
            "same_event_gaze_used_as_predictor": False,
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _write_report(
        report_path,
        fingerprint=fingerprint,
        summaries=summaries,
        primary_tests=primary_tests,
        duration_seconds=duration_seconds,
    )
    _write_plot(plot_path, summaries)

    artifacts = [
        fingerprint_path,
        fold_path,
        subject_path,
        summary_path,
        report_path,
        plot_path,
    ]
    write_experiment_manifest(
        manifest_path,
        "geco_cross_subject_generalization_v1",
        root=PROJECT_ROOT,
        datasets=[fingerprint_path],
        artifacts=artifacts,
        config=summary["configuration"],
        metrics={
            "subject_count": fingerprint["subject_count"],
            "participant_trial_count": fingerprint["participant_trial_count"],
            "row_count": fingerprint["row_count"],
            "primary_model": "ridge_text_only",
            "primary_summary": next(
                row for row in summaries if row["model"] == "ridge_text_only"
            ),
            "primary_tests": primary_tests,
        },
        seed=RANDOM_SEED,
        duration_seconds=duration_seconds,
        source_snapshot=source_snapshot,
    )

    print(f"Completed GECO generalization study in {duration_seconds:.2f}s")
    print(f"Report: {report_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Leakage-resistant evaluation for the candidate text/gaze fusion path.

Promotion uses an independent difficulty target, never reading time reconstructed
from the same gaze signal.  All candidate models share deterministic grouped
folds, group-balanced training weights, and a shuffled-target sentinel.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .generalization import (
    cross_fit_grouped_ridge,
    grouped_spearman_table,
    paired_bootstrap_mean_difference,
)

REQUIRED_COLUMNS = (
    "participant_id",
    "session_id",
    "device_id",
    "article_id",
    "word_id",
    "difficulty_target",
    "gaze_score",
    "gaze_confidence",
    "text_score",
)

FUSION_FEATURE_SETS = {
    "text_only": ("text_score",),
    "gaze_only": ("gaze_score", "gaze_confidence", "gaze_weighted"),
    "combined": (
        "gaze_score",
        "gaze_confidence",
        "gaze_weighted",
        "text_score",
        "gaze_text_interaction",
    ),
}


@dataclass(frozen=True)
class FusionValidationConfig:
    """Frozen numeric settings for the v1 fusion decision gate."""

    n_folds: int = 5
    alpha: float = 1.0
    seed: int = 20260806
    bootstrap_samples: int = 10_000
    minimum_groups: int = 10
    minimum_positive_folds: int = 4


def prepare_fusion_evaluation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate independent word outcomes and add frozen interaction features."""
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"fusion evaluation is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("fusion evaluation frame is empty")

    prepared = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    identity_columns = (
        "participant_id",
        "session_id",
        "device_id",
        "article_id",
        "word_id",
    )
    for column in identity_columns:
        prepared[column] = prepared[column].astype(str).str.strip()
        if prepared[column].eq("").any():
            raise ValueError(f"{column} contains an empty identifier")
    if prepared.duplicated(list(identity_columns)).any():
        raise ValueError(
            "fusion input must contain one aggregated row per capture/article/word"
        )

    numeric_columns = (
        "difficulty_target",
        "gaze_score",
        "gaze_confidence",
        "text_score",
    )
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    numeric = prepared.loc[:, list(numeric_columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("fusion scores and targets must be finite")
    if np.any((numeric < 0.0) | (numeric > 1.0)):
        raise ValueError("fusion scores and independent target must be within [0, 1]")

    prepared["capture_group_id"] = (
        prepared["participant_id"]
        + "|"
        + prepared["session_id"]
        + "|"
        + prepared["device_id"]
    )
    prepared["gaze_weighted"] = (
        prepared["gaze_score"] * prepared["gaze_confidence"]
    )
    prepared["gaze_text_interaction"] = (
        prepared["gaze_score"] * prepared["text_score"]
    )
    return prepared


def validate_fusion_dataset_contract(metadata: Mapping[str, Any]) -> None:
    """Reject circular, tuned, QA-derived, or non-independent evaluation data."""
    required_values = {
        "dataset_role": "independent_real_capture",
        "question_answer_dataset_used": False,
        "difficulty_target_derived_from_gaze": False,
        "difficulty_target_derived_from_text_model": False,
        "fusion_parameters_frozen_before_outcomes": True,
    }
    mismatches = {
        key: {"expected": expected, "observed": metadata.get(key)}
        for key, expected in required_values.items()
        if metadata.get(key) is not expected
        if not isinstance(expected, str) or metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"fusion dataset contract failed: {mismatches}")
    target_source = str(metadata.get("difficulty_target_source", "")).strip()
    if not target_source:
        raise ValueError("difficulty_target_source must be recorded")


def evaluate_fusion_candidate(
    frame: pd.DataFrame,
    *,
    dataset_metadata: Mapping[str, Any],
    config: FusionValidationConfig | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate capture- and article-held-out models under one frozen gate."""
    validate_fusion_dataset_contract(dataset_metadata)
    active = config or FusionValidationConfig()
    if active.bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if active.minimum_groups < active.n_folds:
        raise ValueError("minimum_groups must be at least n_folds")

    prepared = prepare_fusion_evaluation_frame(frame)
    axes = {
        "capture_group": "capture_group_id",
        "article": "article_id",
    }
    summaries: dict[str, Any] = {}
    prediction_tables: list[pd.DataFrame] = []
    for axis_index, (axis_name, group_column) in enumerate(axes.items()):
        summary, predictions = _evaluate_holdout_axis(
            prepared,
            axis_name=axis_name,
            group_column=group_column,
            config=active,
            seed=active.seed + axis_index * 1_000,
        )
        summaries[axis_name] = summary
        predictions.insert(0, "holdout_axis", axis_name)
        prediction_tables.append(predictions)

    both_pass = all(summary["gate"]["passed"] for summary in summaries.values())
    result = {
        "schema_version": 1,
        "protocol_id": "production-text-fusion-v1",
        "configuration": {
            "outer_folds": active.n_folds,
            "ridge_alpha": active.alpha,
            "seed": active.seed,
            "bootstrap_samples": active.bootstrap_samples,
            "minimum_groups": active.minimum_groups,
            "minimum_positive_folds": active.minimum_positive_folds,
            "feature_sets": {
                name: list(features)
                for name, features in FUSION_FEATURE_SETS.items()
            },
        },
        "dataset": dict(dataset_metadata),
        "row_count": len(prepared),
        "capture_group_count": prepared["capture_group_id"].nunique(),
        "article_count": prepared["article_id"].nunique(),
        "holdouts": summaries,
        "promotion": {
            "status": "promote_candidate" if both_pass else "retain_gaze_only",
            "passed": both_pass,
            "both_holdout_axes_required": True,
            "production_model_changed": False,
        },
        "leakage_controls": {
            "question_answer_dataset_used": False,
            "difficulty_target_derived_from_gaze": False,
            "difficulty_target_derived_from_text_model": False,
            "same_folds_for_all_models": True,
            "complete_group_holdout": True,
            "group_balanced_training_weights": True,
            "target_shuffle_sentinel_included": True,
        },
        "compute": {"device": "cpu", "gpu_used": False},
    }
    return result, pd.concat(prediction_tables, ignore_index=True)


def _evaluate_holdout_axis(
    frame: pd.DataFrame,
    *,
    axis_name: str,
    group_column: str,
    config: FusionValidationConfig,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    group_count = frame[group_column].nunique()
    if group_count < config.minimum_groups:
        raise ValueError(
            f"{axis_name} holdout requires at least {config.minimum_groups} groups; "
            f"observed {group_count}"
        )
    group_sizes = frame.groupby(group_column)[group_column].transform("size")
    evaluation = frame.copy()
    evaluation["group_balanced_weight"] = 1.0 / group_sizes.to_numpy(dtype=float)

    predictions, diagnostics = cross_fit_grouped_ridge(
        evaluation,
        group_column=group_column,
        target_column="difficulty_target",
        feature_sets=FUSION_FEATURE_SETS,
        n_folds=config.n_folds,
        alpha=config.alpha,
        seed=seed,
        shuffled_target_model="combined",
        sample_weight_column="group_balanced_weight",
    )
    prediction_columns = (
        "prediction_text_only",
        "prediction_gaze_only",
        "prediction_combined",
        "prediction_target_shuffle_sentinel",
    )
    group_metrics = grouped_spearman_table(
        predictions,
        target_column="difficulty_target",
        prediction_columns=prediction_columns,
        grouping_columns=("outer_fold", group_column),
    )
    pivot = group_metrics.pivot_table(
        index=["outer_fold", group_column],
        columns="model",
        values="spearman_rho",
        aggfunc="first",
    )
    required_models = {
        "text_only",
        "gaze_only",
        "combined",
        "target_shuffle_sentinel",
    }
    if not required_models.issubset(pivot.columns):
        raise RuntimeError(f"{axis_name} metrics are missing one or more models")
    finite_primary = pivot[["gaze_only", "combined"]].dropna()
    if len(finite_primary) < config.minimum_groups:
        raise ValueError(
            f"{axis_name} holdout has only {len(finite_primary)} non-constant groups"
        )

    comparisons = {
        "combined_minus_gaze_only": _paired_comparison(
            pivot,
            first="combined",
            second="gaze_only",
            samples=config.bootstrap_samples,
            seed=seed + 100,
        ),
        "combined_minus_text_only": _paired_comparison(
            pivot,
            first="combined",
            second="text_only",
            samples=config.bootstrap_samples,
            seed=seed + 200,
        ),
    }
    sentinel = _sentinel_summary(
        pivot["target_shuffle_sentinel"].to_numpy(dtype=float),
        samples=config.bootstrap_samples,
        seed=seed + 300,
    )
    primary = comparisons["combined_minus_gaze_only"]
    gate = {
        "combined_minus_gaze_ci_positive": primary["ci_95_low"] > 0,
        "minimum_positive_folds_met": (
            primary["positive_outer_folds"] >= config.minimum_positive_folds
        ),
        "shuffle_sentinel_not_strictly_positive": sentinel["ci_95_low"] <= 0,
    }
    gate["passed"] = all(gate.values())

    models: dict[str, Any] = {}
    target = predictions["difficulty_target"].to_numpy(dtype=float)
    for prediction_column in prediction_columns:
        model_name = prediction_column.removeprefix("prediction_")
        values = group_metrics.loc[
            group_metrics["model"].eq(model_name), "spearman_rho"
        ].to_numpy(dtype=float)
        predicted = predictions[prediction_column].to_numpy(dtype=float)
        models[model_name] = {
            "macro_group_spearman": float(np.nanmean(values)),
            "valid_group_count": int(np.isfinite(values).sum()),
            "mean_absolute_error": float(np.mean(np.abs(predicted - target))),
        }

    keep_columns = [
        *REQUIRED_COLUMNS,
        "capture_group_id",
        "outer_fold",
        *prediction_columns,
    ]
    return (
        {
            "group_column": group_column,
            "group_count": group_count,
            "models": models,
            "comparisons": comparisons,
            "target_shuffle_sentinel": sentinel,
            "gate": gate,
            "cross_fit_diagnostics": diagnostics,
        },
        predictions.loc[:, keep_columns].copy(),
    )


def _paired_comparison(
    pivot: pd.DataFrame,
    *,
    first: str,
    second: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    paired = pivot[[first, second]].dropna()
    bootstrap = paired_bootstrap_mean_difference(
        paired[first].to_numpy(dtype=float),
        paired[second].to_numpy(dtype=float),
        samples=samples,
        seed=seed,
    )
    per_fold = (paired[first] - paired[second]).groupby("outer_fold").mean()
    return {
        **bootstrap,
        "positive_outer_folds": int((per_fold > 0).sum()),
        "outer_fold_count": len(per_fold),
        "fold_mean_differences": {
            str(int(fold)): float(value) for fold, value in per_fold.items()
        },
    }


def _sentinel_summary(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError("target shuffle sentinel has no finite groups")
    return paired_bootstrap_mean_difference(
        finite,
        np.zeros(len(finite), dtype=np.float64),
        samples=samples,
        seed=seed,
    )

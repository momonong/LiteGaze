"""Nested, group-held-out evaluation for motion-diverse gaze calibration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .calibration_regression import (
    MOTION_PROMOTION_MIN_ABSOLUTE_PX,
    MOTION_PROMOTION_MIN_RELATIVE,
    fit_best_stage,
    fit_standardized_ridge,
    motion_challenger_decision,
    motion_conditioned_features,
    polynomial_design,
    standardized_design,
)

BASELINE_MODEL = "gaze_polynomial"
CHALLENGER_MODEL = "motion_conditioned_ridge_v1"
VALIDATION_SCHEME = "nested_leave_one_motion_block_out"


def _error_statistics(errors: np.ndarray) -> dict[str, float]:
    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("pixel errors must be a non-empty finite vector")
    return {
        "mean_px": float(np.mean(values)),
        "median_px": float(np.median(values)),
        "p95_px": float(np.percentile(values, 95)),
    }


def _pixel_errors(
    predictions: np.ndarray,
    targets: np.ndarray,
    viewports: np.ndarray,
) -> np.ndarray:
    return np.linalg.norm((predictions - targets) * (viewports * 0.5), axis=1)


def evaluate_motion_candidates(
    gaze_pitch_yaw: np.ndarray,
    head_pitch_yaw: np.ndarray,
    face_geometry: np.ndarray,
    targets: np.ndarray,
    viewport_list: Sequence[Sequence[float]],
    motion_blocks: Sequence[str],
) -> dict[str, Any]:
    """Compare M0/M1 with an outer motion-block holdout and inner CV.

    The complete outer block is invisible to hyperparameter selection. The
    remaining blocks are passed to the existing group-held-out fitters, so
    degree/regularization selection happens only inside the outer training
    partition.
    """

    gaze = np.asarray(gaze_pitch_yaw, dtype=np.float64)
    head = np.asarray(head_pitch_yaw, dtype=np.float64)
    face = np.asarray(face_geometry, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    viewports = np.asarray(viewport_list, dtype=np.float64)
    if gaze.ndim != 2 or gaze.shape[1] != 2:
        raise ValueError("gaze_pitch_yaw must have shape (N, 2)")
    if head.shape != gaze.shape:
        raise ValueError("head_pitch_yaw must have shape (N, 2)")
    if face.shape != (len(gaze), 3):
        raise ValueError("face_geometry must have shape (N, 3)")
    if target_array.shape != gaze.shape:
        raise ValueError("targets must have shape (N, 2)")
    if viewports.shape != gaze.shape:
        raise ValueError("viewport_list must have shape (N, 2)")
    if len(motion_blocks) != len(gaze):
        raise ValueError("motion_blocks must align with samples")
    if not (
        np.isfinite(gaze).all()
        and np.isfinite(head).all()
        and np.isfinite(face).all()
        and np.isfinite(target_array).all()
        and np.isfinite(viewports).all()
    ):
        raise ValueError("experiment arrays must contain only finite values")
    if (viewports <= 0).any():
        raise ValueError("viewport dimensions must be positive")

    normalized_groups = [str(group).strip() for group in motion_blocks]
    if any(not group for group in normalized_groups):
        raise ValueError("motion_blocks must not contain blank values")
    group_names = sorted(set(normalized_groups))
    if len(group_names) < 3:
        raise ValueError("nested validation requires at least three motion blocks")
    group_array = np.asarray(normalized_groups)
    motion_features = motion_conditioned_features(gaze, head, face)

    baseline_errors: list[float] = []
    challenger_errors: list[float] = []
    folds: list[dict[str, Any]] = []
    for outer_group in group_names:
        train_indices = np.flatnonzero(group_array != outer_group)
        validation_indices = np.flatnonzero(group_array == outer_group)
        inner_groups = group_array[train_indices].tolist()
        unique_targets = len(np.unique(target_array[train_indices], axis=0))

        (
            baseline_weights,
            baseline_degree,
            baseline_alpha,
            baseline_inner_error,
        ) = fit_best_stage(
            gaze[train_indices],
            target_array[train_indices],
            viewports[train_indices],
            unique_targets,
            is_stage_1=True,
            validation_groups=inner_groups,
        )
        baseline_predictions = (
            polynomial_design(
                gaze[validation_indices],
                degree=baseline_degree,
                is_stage_1=True,
            )
            @ baseline_weights
        )

        (
            challenger_weights,
            feature_mean,
            feature_scale,
            challenger_alpha,
            challenger_inner_error,
        ) = fit_standardized_ridge(
            motion_features[train_indices],
            target_array[train_indices],
            viewports[train_indices],
            validation_groups=inner_groups,
        )
        challenger_predictions = (
            standardized_design(
                motion_features[validation_indices],
                feature_mean,
                feature_scale,
            )
            @ challenger_weights
        )

        fold_baseline_errors = _pixel_errors(
            baseline_predictions,
            target_array[validation_indices],
            viewports[validation_indices],
        )
        fold_challenger_errors = _pixel_errors(
            challenger_predictions,
            target_array[validation_indices],
            viewports[validation_indices],
        )
        baseline_errors.extend(fold_baseline_errors.astype(float).tolist())
        challenger_errors.extend(fold_challenger_errors.astype(float).tolist())
        folds.append(
            {
                "outer_motion_block": outer_group,
                "train_motion_block_count": len(set(inner_groups)),
                "train_samples": int(len(train_indices)),
                "validation_samples": int(len(validation_indices)),
                "gaze_polynomial": {
                    "degree": int(baseline_degree),
                    "alpha": float(baseline_alpha),
                    "inner_validation_mean_px": float(baseline_inner_error),
                    **_error_statistics(fold_baseline_errors),
                },
                "motion_conditioned_ridge_v1": {
                    "alpha": float(challenger_alpha),
                    "inner_validation_mean_px": float(challenger_inner_error),
                    **_error_statistics(fold_challenger_errors),
                },
            }
        )

    baseline_array = np.asarray(baseline_errors, dtype=np.float64)
    challenger_array = np.asarray(challenger_errors, dtype=np.float64)
    baseline_macro = float(
        np.mean([fold[BASELINE_MODEL]["mean_px"] for fold in folds])
    )
    challenger_macro = float(
        np.mean([fold[CHALLENGER_MODEL]["mean_px"] for fold in folds])
    )
    promote, required, observed = motion_challenger_decision(
        baseline_macro,
        challenger_macro,
    )
    relative = observed / baseline_macro if baseline_macro > 0 else 0.0
    selected_model = CHALLENGER_MODEL if promote else BASELINE_MODEL

    return {
        "schema_version": 1,
        "validation_scheme": VALIDATION_SCHEME,
        "sample_count": int(len(gaze)),
        "outer_motion_block_count": len(group_names),
        "candidates": {
            BASELINE_MODEL: {
                "macro_mean_px": baseline_macro,
                **_error_statistics(baseline_array),
            },
            CHALLENGER_MODEL: {
                "macro_mean_px": challenger_macro,
                **_error_statistics(challenger_array),
            },
        },
        "promotion_gate": {
            "min_absolute_improvement_px": MOTION_PROMOTION_MIN_ABSOLUTE_PX,
            "min_relative_improvement": MOTION_PROMOTION_MIN_RELATIVE,
            "required_improvement_px": required,
            "observed_improvement_px": observed,
            "observed_relative_improvement": relative,
            "passed": promote,
            "selected_model": selected_model,
        },
        "folds": folds,
    }

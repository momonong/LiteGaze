"""Small ridge calibrators with leakage-resistant validation splits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite, sqrt
from typing import Any

import numpy as np

MOTION_FEATURE_NAMES = (
    "gaze_yaw",
    "gaze_pitch",
    "head_yaw",
    "head_pitch",
    "face_center_x",
    "face_center_y",
    "face_scale",
    "gaze_yaw_x_head_yaw",
    "gaze_pitch_x_head_pitch",
    "gaze_yaw_x_face_scale",
    "gaze_pitch_x_face_scale",
)


def motion_challenger_decision(
    baseline_error: float,
    challenger_error: float,
    *,
    min_absolute_improvement: float = 5.0,
    min_relative_improvement: float = 0.05,
) -> tuple[bool, float, float]:
    """Apply the frozen absolute-and-relative promotion rule."""

    values = (
        baseline_error,
        challenger_error,
        min_absolute_improvement,
        min_relative_improvement,
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("promotion inputs must be finite")
    if baseline_error < 0 or challenger_error < 0:
        raise ValueError("validation errors must be non-negative")
    if min_absolute_improvement < 0 or not 0 <= min_relative_improvement < 1:
        raise ValueError("promotion margins are invalid")
    required = max(
        min_absolute_improvement,
        baseline_error * min_relative_improvement,
    )
    observed = baseline_error - challenger_error
    return observed >= required, required, observed


def face_geometry_from_bbox(face_bbox: Mapping[str, Any]) -> tuple[float, float, float]:
    """Convert a normalized face box to center-x, center-y, and scale."""

    try:
        x = float(face_bbox["x_norm"])
        y = float(face_bbox["y_norm"])
        width = float(face_bbox["w_norm"])
        height = float(face_bbox["h_norm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("face_bbox lacks normalized geometry") from exc
    values = (x, y, width, height)
    if not all(isfinite(value) for value in values) or width < 0 or height < 0:
        raise ValueError("face_bbox normalized geometry must be finite and non-negative")
    return x + width * 0.5, y + height * 0.5, sqrt(width * height)


def _design_matrix(
    inputs: np.ndarray,
    *,
    degree: int,
    is_stage_1: bool,
) -> np.ndarray:
    if is_stage_1:
        value_1 = inputs[:, 1]  # gaze yaw -> screen x
        value_2 = inputs[:, 0]  # gaze pitch -> screen y
    else:
        value_1 = inputs[:, 0]
        value_2 = inputs[:, 1]

    if degree == 1:
        return np.column_stack([value_1, value_2, np.ones(len(inputs))])
    if degree == 2:
        return np.column_stack(
            [
                value_1,
                value_2,
                value_1 * value_1,
                value_2 * value_2,
                value_1 * value_2,
                np.ones(len(inputs)),
            ]
        )
    raise ValueError("degree must be 1 or 2")


def _ridge_weights(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    identity = np.eye(X.shape[1])
    identity[-1, -1] = 0.0
    return np.linalg.solve(X.T @ X + alpha * identity, X.T @ Y)


def motion_conditioned_features(
    gaze_pitch_yaw: np.ndarray,
    head_pitch_yaw: np.ndarray,
    face_geometry: np.ndarray,
) -> np.ndarray:
    """Build the frozen v1 pose/distance-aware calibration feature schema."""

    gaze = np.asarray(gaze_pitch_yaw, dtype=np.float64)
    head = np.asarray(head_pitch_yaw, dtype=np.float64)
    face = np.asarray(face_geometry, dtype=np.float64)
    if gaze.ndim != 2 or gaze.shape[1] != 2:
        raise ValueError("gaze_pitch_yaw must have shape (N, 2)")
    if head.shape != gaze.shape:
        raise ValueError("head_pitch_yaw must have shape (N, 2)")
    if face.shape != (len(gaze), 3):
        raise ValueError("face_geometry must have shape (N, 3)")
    if not (np.isfinite(gaze).all() and np.isfinite(head).all() and np.isfinite(face).all()):
        raise ValueError("motion-conditioned inputs must be finite")

    gaze_pitch, gaze_yaw = gaze[:, 0], gaze[:, 1]
    head_pitch, head_yaw = head[:, 0], head[:, 1]
    face_center_x, face_center_y, face_scale = face[:, 0], face[:, 1], face[:, 2]
    return np.column_stack(
        [
            gaze_yaw,
            gaze_pitch,
            head_yaw,
            head_pitch,
            face_center_x,
            face_center_y,
            face_scale,
            gaze_yaw * head_yaw,
            gaze_pitch * head_pitch,
            gaze_yaw * face_scale,
            gaze_pitch * face_scale,
        ]
    )


def standardized_design(
    features: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    feature_mean = np.asarray(feature_mean, dtype=np.float64)
    feature_scale = np.asarray(feature_scale, dtype=np.float64)
    if feature_mean.shape != (features.shape[1],):
        raise ValueError("feature_mean does not match feature width")
    if feature_scale.shape != (features.shape[1],) or (feature_scale <= 0).any():
        raise ValueError("feature_scale must be positive and match feature width")
    standardized = (features - feature_mean) / feature_scale
    return np.column_stack([standardized, np.ones(len(features))])


def _validation_folds(
    sample_count: int,
    validation_groups: Sequence[str] | None,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    all_indices = np.arange(sample_count)
    if validation_groups is None:
        return tuple(
            (np.delete(all_indices, index), np.array([index]))
            for index in range(sample_count)
        )

    if len(validation_groups) != sample_count:
        raise ValueError("validation_groups must align with inputs")
    normalized_groups = [str(group).strip() for group in validation_groups]
    if any(not group for group in normalized_groups):
        raise ValueError("validation_groups must not contain blank values")
    group_array = np.asarray(normalized_groups)
    group_names = sorted(set(normalized_groups))
    if len(group_names) < 2:
        raise ValueError("at least two validation groups are required")
    return tuple(
        (
            all_indices[group_array != group_name],
            all_indices[group_array == group_name],
        )
        for group_name in group_names
    )


def fit_standardized_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    viewport_list: Sequence[Sequence[float]],
    *,
    validation_groups: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Fit a standardized ridge model without leaking fold statistics."""

    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    viewports = np.asarray(viewport_list, dtype=np.float64)
    if features.ndim != 2 or len(features) < 2:
        raise ValueError("features must have shape (N, F) with N >= 2")
    if targets.shape != (len(features), 2):
        raise ValueError("targets must have shape (N, 2)")
    if viewports.shape != (len(features), 2):
        raise ValueError("viewport_list must have shape (N, 2)")
    if not (
        np.isfinite(features).all()
        and np.isfinite(targets).all()
        and np.isfinite(viewports).all()
    ):
        raise ValueError("ridge arrays must contain only finite values")
    if (viewports <= 0).any():
        raise ValueError("viewport dimensions must be positive")

    folds = _validation_folds(len(features), validation_groups)
    best_alpha = 0.1
    best_validation_error = float("inf")
    for alpha in (1e-3, 1e-2, 0.1, 1.0, 10.0):
        fold_errors: list[float] = []
        for train_indices, validation_indices in folds:
            train_features = features[train_indices]
            feature_mean = train_features.mean(axis=0)
            feature_scale = train_features.std(axis=0)
            feature_scale[feature_scale < 1e-8] = 1.0
            train_design = standardized_design(
                train_features,
                feature_mean,
                feature_scale,
            )
            validation_design = standardized_design(
                features[validation_indices],
                feature_mean,
                feature_scale,
            )
            try:
                weights = _ridge_weights(
                    train_design,
                    targets[train_indices],
                    alpha,
                )
            except np.linalg.LinAlgError:
                continue
            predictions = validation_design @ weights
            pixel_deltas = (
                predictions - targets[validation_indices]
            ) * (viewports[validation_indices] * 0.5)
            fold_errors.extend(
                np.linalg.norm(pixel_deltas, axis=1).astype(float).tolist()
            )
        if fold_errors:
            mean_error = float(np.mean(fold_errors))
            if mean_error < best_validation_error:
                best_validation_error = mean_error
                best_alpha = alpha

    if not np.isfinite(best_validation_error):
        raise ValueError("no valid standardized-ridge fold could be fitted")
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    final_design = standardized_design(features, feature_mean, feature_scale)
    final_weights = _ridge_weights(final_design, targets, best_alpha)
    return (
        final_weights,
        feature_mean,
        feature_scale,
        best_alpha,
        best_validation_error,
    )


def fit_best_stage(
    inputs: np.ndarray,
    targets: np.ndarray,
    viewport_list: Sequence[Sequence[float]],
    unique_targets: int,
    *,
    is_stage_1: bool = True,
    validation_groups: Sequence[str] | None = None,
) -> tuple[np.ndarray, int, float, float]:
    """Select ridge hyperparameters using sample- or group-held-out error.

    When ``validation_groups`` is provided, every sample from a group is held
    out together. Motion-diverse calibration passes ``motion_block_id`` here,
    preventing adjacent frames from the same posture from leaking into train
    and validation simultaneously.
    """

    inputs = np.asarray(inputs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    viewports = np.asarray(viewport_list, dtype=np.float64)
    if inputs.ndim != 2 or inputs.shape[1] != 2:
        raise ValueError("inputs must have shape (N, 2)")
    if targets.shape != inputs.shape:
        raise ValueError("targets must have shape (N, 2)")
    if viewports.shape != inputs.shape:
        raise ValueError("viewport_list must have shape (N, 2)")
    if len(inputs) < 2:
        raise ValueError("at least two calibration samples are required")
    if not (
        np.isfinite(inputs).all()
        and np.isfinite(targets).all()
        and np.isfinite(viewports).all()
    ):
        raise ValueError("calibration arrays must contain only finite values")
    if (viewports <= 0).any():
        raise ValueError("viewport dimensions must be positive")

    folds = _validation_folds(len(inputs), validation_groups)
    candidate_degrees = [1]
    if unique_targets > 5 and len(inputs) >= 6:
        candidate_degrees.append(2)
    candidate_alphas = (1e-4, 1e-3, 1e-2, 0.1)

    best_degree = 1
    best_alpha = 1e-3
    best_validation_error = float("inf")
    for degree in candidate_degrees:
        design = _design_matrix(inputs, degree=degree, is_stage_1=is_stage_1)
        for alpha in candidate_alphas:
            fold_errors: list[float] = []
            for train_indices, validation_indices in folds:
                try:
                    weights = _ridge_weights(
                        design[train_indices],
                        targets[train_indices],
                        alpha,
                    )
                except np.linalg.LinAlgError:
                    continue
                predictions = design[validation_indices] @ weights
                pixel_deltas = (
                    predictions - targets[validation_indices]
                ) * (viewports[validation_indices] * 0.5)
                fold_errors.extend(
                    np.linalg.norm(pixel_deltas, axis=1).astype(float).tolist()
                )

            if fold_errors:
                mean_error = float(np.mean(fold_errors))
                if mean_error < best_validation_error:
                    best_validation_error = mean_error
                    best_degree = degree
                    best_alpha = alpha

    if not np.isfinite(best_validation_error):
        raise ValueError("no valid calibration validation fold could be fitted")
    final_design = _design_matrix(
        inputs,
        degree=best_degree,
        is_stage_1=is_stage_1,
    )
    final_weights = _ridge_weights(final_design, targets, best_alpha)
    return final_weights, best_degree, best_alpha, best_validation_error

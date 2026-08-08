"""Aggregate-only metrics for Columbia cross-domain gaze evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from scripts.gaze_diversity.metrics import (
    angles_to_unit_vectors,
    angular_errors_degrees,
    paired_participant_bootstrap,
    summarize_errors,
    unit_vectors_to_angles,
)


def fuse_angle_predictions(predictions: np.ndarray) -> np.ndarray:
    """Fuse (member,row,pitch-yaw) predictions through normalized vectors."""
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 2 or values.shape[0] < 1:
        raise ValueError("predictions must have shape (members, rows, 2)")
    vectors = np.stack([angles_to_unit_vectors(member) for member in values], axis=0)
    fused = vectors.mean(axis=0)
    norms = np.linalg.norm(fused, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("prediction ensemble produced a zero vector")
    return unit_vectors_to_angles(fused / norms)


def summarize_model(
    prediction: np.ndarray,
    targets: np.ndarray,
    subjects: np.ndarray,
    *,
    head_poses: np.ndarray,
    vertical_gazes: np.ndarray,
    horizontal_gazes: np.ndarray,
    zero_subject_means: dict[str, float],
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Summarize one model without persisting row-level predictions."""
    errors = angular_errors_degrees(prediction, targets)
    subject_means = _group_means(errors, subjects)
    ordered_subjects = sorted(subject_means)
    differences = np.asarray(
        [
            subject_means[subject] - zero_subject_means[subject]
            for subject in ordered_subjects
        ]
    )
    return {
        "micro": summarize_errors(errors),
        "macro_subject_mean_degrees": float(np.mean(list(subject_means.values()))),
        "worst_subject_mean_degrees": float(np.max(list(subject_means.values()))),
        "subject_count": len(subject_means),
        "subjects_beating_zero_gaze": int(np.sum(differences < 0.0)),
        "model_minus_zero_subject_bootstrap": paired_participant_bootstrap(
            differences,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
        "by_head_pose_degrees": _factor_summary(errors, head_poses),
        "by_vertical_gaze_degrees": _factor_summary(errors, vertical_gazes),
        "by_horizontal_gaze_degrees": _factor_summary(errors, horizontal_gazes),
    }


def zero_gaze_summary(
    targets: np.ndarray,
    subjects: np.ndarray,
    *,
    head_poses: np.ndarray,
    vertical_gazes: np.ndarray,
    horizontal_gazes: np.ndarray,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Return the frozen symmetric-grid zero-gaze baseline and subject means."""
    prediction = np.zeros_like(targets, dtype=np.float64)
    errors = angular_errors_degrees(prediction, targets)
    subject_means = _group_means(errors, subjects)
    summary = {
        "micro": summarize_errors(errors),
        "macro_subject_mean_degrees": float(np.mean(list(subject_means.values()))),
        "worst_subject_mean_degrees": float(np.max(list(subject_means.values()))),
        "subject_count": len(subject_means),
        "by_head_pose_degrees": _factor_summary(errors, head_poses),
        "by_vertical_gaze_degrees": _factor_summary(errors, vertical_gazes),
        "by_horizontal_gaze_degrees": _factor_summary(errors, horizontal_gazes),
    }
    return summary, subject_means


def _group_means(errors: np.ndarray, groups: Iterable[object]) -> dict[str, float]:
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    labels = np.asarray(list(groups))
    if len(values) != len(labels) or not np.isfinite(values).all():
        raise ValueError("grouped errors are invalid")
    result: dict[str, float] = {}
    for label in sorted(set(labels.tolist()), key=str):
        selected = values[labels == label]
        if selected.size == 0:
            raise ValueError("empty aggregate group")
        result[str(label)] = float(np.mean(selected))
    return result


def _factor_summary(
    errors: np.ndarray, factors: np.ndarray
) -> dict[str, dict[str, float | int]]:
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    labels = np.asarray(factors)
    if len(values) != len(labels):
        raise ValueError("factor and error row counts differ")
    return {
        str(label): summarize_errors(values[labels == label])
        for label in sorted(set(labels.tolist()))
    }

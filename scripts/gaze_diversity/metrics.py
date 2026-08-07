"""Numerically stable metrics and frozen baselines for gaze-diversity v1."""

from __future__ import annotations

from typing import Any

import numpy as np


def angles_to_unit_vectors(angles: np.ndarray) -> np.ndarray:
    """Convert [pitch, yaw] radians to MPIIGaze-normalized 3D directions."""
    values = np.asarray(angles, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("angles must have shape (n, 2)")
    if not np.isfinite(values).all():
        raise ValueError("angles contain non-finite values")
    pitch = values[:, 0]
    yaw = values[:, 1]
    cos_pitch = np.cos(pitch)
    vectors = np.column_stack(
        (
            -cos_pitch * np.sin(yaw),
            -np.sin(pitch),
            -cos_pitch * np.cos(yaw),
        )
    )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def unit_vectors_to_angles(vectors: np.ndarray) -> np.ndarray:
    """Convert normalized 3D directions to [pitch, yaw] radians."""
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("vectors must have shape (n, 3)")
    if not np.isfinite(values).all():
        raise ValueError("vectors contain non-finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("vectors contain a zero norm")
    normalized = values / norms
    pitch = np.arcsin(np.clip(-normalized[:, 1], -1.0, 1.0))
    yaw = np.arctan2(-normalized[:, 0], -normalized[:, 2])
    return np.column_stack((pitch, yaw))


def angular_errors_degrees(
    prediction_angles: np.ndarray, target_angles: np.ndarray
) -> np.ndarray:
    """Return one 3D angular error in degrees per row."""
    prediction = angles_to_unit_vectors(prediction_angles)
    target = angles_to_unit_vectors(target_angles)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target row counts differ")
    cosine = np.sum(prediction * target, axis=1)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def summarize_errors(errors: np.ndarray) -> dict[str, float | int]:
    """Summarize a finite non-empty error vector."""
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("errors must be finite and non-empty")
    return {
        "rows": int(values.size),
        "mean_degrees": float(np.mean(values)),
        "median_degrees": float(np.median(values)),
        "p90_degrees": float(np.percentile(values, 90)),
    }


def constant_train_mean_prediction(
    training_targets: np.ndarray, row_count: int
) -> np.ndarray:
    """Predict the normalized mean outer-training gaze direction."""
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    training_vectors = angles_to_unit_vectors(training_targets)
    mean_vector = training_vectors.mean(axis=0)
    norm = np.linalg.norm(mean_vector)
    if norm <= 1e-12:
        raise ValueError("training mean gaze vector has zero norm")
    mean_angle = unit_vectors_to_angles(mean_vector / norm)[0]
    return np.repeat(mean_angle.reshape(1, 2), row_count, axis=0)


def fit_pose_only_ridge(
    training_poses: np.ndarray,
    training_targets: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Fit the frozen linear pose baseline with an unpenalized intercept."""
    poses = np.asarray(training_poses, dtype=np.float64)
    targets = np.asarray(training_targets, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 2:
        raise ValueError("training poses must have shape (n, 2)")
    if targets.shape != poses.shape:
        raise ValueError("training targets must match pose shape")
    if alpha < 0 or not np.isfinite(alpha):
        raise ValueError("ridge alpha must be finite and non-negative")
    design = np.column_stack((poses, np.ones(len(poses))))
    penalty = np.diag((alpha, alpha, 0.0))
    return np.linalg.solve(design.T @ design + penalty, design.T @ targets)


def predict_pose_only_ridge(poses: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Apply the frozen pose-only ridge coefficients."""
    values = np.asarray(poses, dtype=np.float64)
    weights = np.asarray(coefficients, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or weights.shape != (3, 2):
        raise ValueError("pose ridge dimensions are invalid")
    design = np.column_stack((values, np.ones(len(values))))
    prediction = design @ weights
    if not np.isfinite(prediction).all():
        raise ValueError("pose ridge prediction is non-finite")
    return prediction


def summarize_days(
    errors: np.ndarray, days: np.ndarray, *, minimum_rows: int
) -> dict[str, dict[str, float | int]]:
    """Report temporal slices without using them for model selection."""
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    labels = np.asarray(days).reshape(-1)
    if len(values) != len(labels) or minimum_rows <= 0:
        raise ValueError("day summary inputs are invalid")
    result: dict[str, dict[str, float | int]] = {}
    for day in sorted(set(str(value) for value in labels)):
        selected = values[labels.astype(str) == day]
        if len(selected) >= minimum_rows:
            result[day] = summarize_errors(selected)
    return result


def paired_participant_bootstrap(
    differences: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap the macro participant difference with a frozen seed."""
    values = np.asarray(differences, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all() or resamples <= 0:
        raise ValueError("bootstrap inputs are invalid")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(resamples, values.size))
    sampled_means = values[draws].mean(axis=1)
    return {
        "participants": int(values.size),
        "resamples": int(resamples),
        "seed": int(seed),
        "mean_difference_degrees": float(values.mean()),
        "ci95_lower_degrees": float(np.percentile(sampled_means, 2.5)),
        "ci95_upper_degrees": float(np.percentile(sampled_means, 97.5)),
    }


def aggregate_experiment_results(
    *,
    subjects: list[str] | tuple[str, ...],
    expected_candidate_seeds: list[int] | tuple[int, ...],
    expected_sentinel_seed: int,
    candidate_jobs: list[dict[str, Any]],
    sentinel_jobs: list[dict[str, Any]],
    baseline_by_subject: dict[str, dict[str, Any]],
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Aggregate frozen fold/seed outputs without selecting a winning seed."""
    ordered_subjects = tuple(subjects)
    expected_subjects = set(ordered_subjects)
    candidate_seeds = tuple(int(seed) for seed in expected_candidate_seeds)
    if not ordered_subjects or len(expected_subjects) != len(ordered_subjects):
        raise ValueError("subjects must be non-empty and unique")
    if not candidate_seeds or len(set(candidate_seeds)) != len(candidate_seeds):
        raise ValueError("candidate seeds must be non-empty and unique")
    if set(baseline_by_subject) != expected_subjects:
        raise ValueError("baselines must cover exactly the expected subjects")
    expected_candidate_pairs = {
        (subject, seed) for subject in ordered_subjects for seed in candidate_seeds
    }
    observed_candidate_pairs = [
        (str(job["test_subject"]), int(job["seed"])) for job in candidate_jobs
    ]
    if (
        len(observed_candidate_pairs) != len(set(observed_candidate_pairs))
        or set(observed_candidate_pairs) != expected_candidate_pairs
    ):
        raise ValueError("candidate results do not match the frozen seed schedule")
    observed_sentinel_pairs = [
        (str(job["test_subject"]), int(job["seed"])) for job in sentinel_jobs
    ]
    expected_sentinel_pairs = {
        (subject, int(expected_sentinel_seed)) for subject in ordered_subjects
    }
    if (
        len(observed_sentinel_pairs) != len(set(observed_sentinel_pairs))
        or set(observed_sentinel_pairs) != expected_sentinel_pairs
    ):
        raise ValueError("sentinel results do not match the frozen seed schedule")

    candidate_by_subject: dict[str, list[float]] = {
        subject: [] for subject in ordered_subjects
    }
    candidate_by_seed: dict[int, list[float]] = {}
    sentinel_by_subject: dict[str, list[float]] = {
        subject: [] for subject in ordered_subjects
    }
    for job in candidate_jobs:
        subject = str(job["test_subject"])
        seed = int(job["seed"])
        value = float(job["test"]["mean_degrees"])
        candidate_by_subject[subject].append(value)
        candidate_by_seed.setdefault(seed, []).append(value)
    for job in sentinel_jobs:
        subject = str(job["test_subject"])
        sentinel_by_subject[subject].append(float(job["test"]["mean_degrees"]))

    candidate_subject_means = {
        subject: float(np.mean(values))
        for subject, values in candidate_by_subject.items()
    }
    candidate_seed_macros = {
        str(seed): float(np.mean(values))
        for seed, values in sorted(candidate_by_seed.items())
    }
    pose_subject_means = {
        subject: float(baseline_by_subject[subject]["pose_only"]["mean_degrees"])
        for subject in ordered_subjects
    }
    constant_subject_means = {
        subject: float(
            baseline_by_subject[subject]["constant_train_mean"]["mean_degrees"]
        )
        for subject in ordered_subjects
    }
    sentinel_subject_means = {
        subject: float(values[0]) for subject, values in sentinel_by_subject.items()
    }
    differences = np.array(
        [
            candidate_subject_means[subject] - pose_subject_means[subject]
            for subject in ordered_subjects
        ],
        dtype=np.float64,
    )
    bootstrap = paired_participant_bootstrap(
        differences,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    return {
        "candidate": {
            "macro_mean_degrees": float(
                np.mean(list(candidate_subject_means.values()))
            ),
            "worst_subject_mean_degrees": float(
                np.max(list(candidate_subject_means.values()))
            ),
            "per_subject_mean_degrees": candidate_subject_means,
            "per_seed_macro_mean_degrees": candidate_seed_macros,
            "seed_macro_standard_deviation_degrees": float(
                np.std(list(candidate_seed_macros.values()), ddof=0)
            ),
        },
        "pose_only": {
            "macro_mean_degrees": float(np.mean(list(pose_subject_means.values()))),
            "per_subject_mean_degrees": pose_subject_means,
        },
        "constant_train_mean": {
            "macro_mean_degrees": float(np.mean(list(constant_subject_means.values()))),
            "per_subject_mean_degrees": constant_subject_means,
        },
        "shuffled_label_sentinel": {
            "macro_mean_degrees": float(np.mean(list(sentinel_subject_means.values()))),
            "per_subject_mean_degrees": sentinel_subject_means,
        },
        "candidate_minus_pose_only": bootstrap,
        "subjects_candidate_beats_pose_only": int(np.sum(differences < 0)),
    }

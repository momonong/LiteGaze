"""Leakage-resistant utilities for grouped text-model generalization studies.

The module intentionally depends only on NumPy, pandas, and SciPy.  Language
model inference is kept outside the fitting code so cached, label-free features
can be audited independently from reading-time outcomes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(frozen=True)
class StandardizedRidge:
    """Ridge parameters with training-fold feature statistics."""

    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    alpha: float


def stable_group_folds(
    values: Iterable[str],
    n_folds: int,
    *,
    seed: int,
) -> dict[str, int]:
    """Assign complete groups to deterministic, approximately balanced folds."""
    unique = sorted({str(value) for value in values})
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    if len(unique) < n_folds:
        raise ValueError("number of unique groups must be at least n_folds")

    def digest(value: str) -> str:
        return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()

    ordered = sorted(unique, key=lambda value: (digest(value), value))
    return {value: index % n_folds for index, value in enumerate(ordered)}


def fit_standardized_ridge(
    features: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float,
) -> StandardizedRidge:
    """Fit Ridge with an unpenalized intercept and training-only scaling."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("features must be 2D and aligned with a 1D target")
    if not len(y):
        raise ValueError("cannot fit Ridge on an empty training fold")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Ridge inputs must be finite")

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    standardized = (x - mean) / scale
    design = np.column_stack([np.ones(len(standardized)), standardized])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ y,
    )
    return StandardizedRidge(
        mean=mean,
        scale=scale,
        coefficients=coefficients,
        alpha=float(alpha),
    )


def fit_weighted_standardized_ridge(
    features: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
    *,
    alpha: float,
) -> StandardizedRidge:
    """Fit Ridge with weighted scaling and an unpenalized intercept.

    Weights are normalized to sum to the number of rows so ``alpha`` keeps the
    same interpretation as :func:`fit_standardized_ridge`.  This is useful for
    corpus- or group-balanced training without duplicating observations.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("features must be 2D and aligned with a 1D target")
    if weights.ndim != 1 or len(weights) != len(y):
        raise ValueError("sample_weight must be 1D and aligned with target")
    if not len(y):
        raise ValueError("cannot fit Ridge on an empty training fold")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Ridge inputs must be finite")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("sample weights must be finite and strictly positive")

    weights = weights * (len(weights) / weights.sum())
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale = np.where(scale > 0, scale, 1.0)
    standardized = (x - mean) / scale
    design = np.column_stack([np.ones(len(standardized)), standardized])
    root_weight = np.sqrt(weights)
    weighted_design = design * root_weight[:, np.newaxis]
    weighted_target = y * root_weight
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_target,
    )
    return StandardizedRidge(
        mean=mean,
        scale=scale,
        coefficients=coefficients,
        alpha=float(alpha),
    )


def predict_standardized_ridge(
    model: StandardizedRidge,
    features: np.ndarray,
) -> np.ndarray:
    """Predict with feature statistics learned from the corresponding train fold."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(model.mean):
        raise ValueError("prediction feature shape does not match the Ridge model")
    if not np.isfinite(x).all():
        raise ValueError("prediction features must be finite")
    standardized = (x - model.mean) / model.scale
    return model.coefficients[0] + standardized @ model.coefficients[1:]


def safe_spearman(first: Sequence[float], second: Sequence[float]) -> float:
    """Return Spearman rho, or NaN for an invalid/constant comparison."""
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    x = x[valid]
    y = y[valid]
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def cross_fit_grouped_ridge(
    frame: pd.DataFrame,
    *,
    group_column: str,
    target_column: str,
    feature_sets: Mapping[str, Sequence[str]],
    n_folds: int,
    alpha: float,
    seed: int,
    shuffled_target_model: str | None = None,
    sample_weight_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Generate predictions with complete groups held out from every fitted model."""
    required = {group_column, target_column}
    if sample_weight_column is not None:
        required.add(sample_weight_column)
    required.update(feature for features in feature_sets.values() for feature in features)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"benchmark frame is missing columns: {sorted(missing)}")
    if not feature_sets:
        raise ValueError("at least one feature set is required")
    if shuffled_target_model is not None and shuffled_target_model not in feature_sets:
        raise ValueError("shuffled_target_model must name one of the feature sets")

    result = frame.copy().reset_index(drop=True)
    if result[target_column].isna().any():
        raise ValueError("target column contains missing values")
    if sample_weight_column is not None:
        weights = result[sample_weight_column].to_numpy(dtype=np.float64)
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("sample weights must be finite and strictly positive")
    group_values = result[group_column].astype(str)
    assignments = stable_group_folds(group_values, n_folds, seed=seed)
    result["outer_fold"] = group_values.map(assignments).astype(int)
    diagnostics: list[dict[str, object]] = []

    for model_name in feature_sets:
        result[f"prediction_{model_name}"] = np.nan
    if shuffled_target_model is not None:
        result["prediction_target_shuffle_sentinel"] = np.nan

    for fold in range(n_folds):
        test_mask = result["outer_fold"].eq(fold).to_numpy()
        train_mask = ~test_mask
        train_groups = set(group_values[train_mask])
        test_groups = set(group_values[test_mask])
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise RuntimeError(f"group leakage detected in fold {fold}: {sorted(overlap)}")

        y_train = result.loc[train_mask, target_column].to_numpy(dtype=np.float64)
        train_weight = None
        if sample_weight_column is not None:
            train_weight = result.loc[
                train_mask, sample_weight_column
            ].to_numpy(dtype=np.float64)
        for model_name, columns in feature_sets.items():
            x_train = result.loc[train_mask, list(columns)].to_numpy(dtype=np.float64)
            x_test = result.loc[test_mask, list(columns)].to_numpy(dtype=np.float64)
            if train_weight is None:
                model = fit_standardized_ridge(x_train, y_train, alpha=alpha)
            else:
                model = fit_weighted_standardized_ridge(
                    x_train,
                    y_train,
                    train_weight,
                    alpha=alpha,
                )
            result.loc[test_mask, f"prediction_{model_name}"] = (
                predict_standardized_ridge(model, x_test)
            )

        if shuffled_target_model is not None:
            columns = list(feature_sets[shuffled_target_model])
            x_train = result.loc[train_mask, columns].to_numpy(dtype=np.float64)
            x_test = result.loc[test_mask, columns].to_numpy(dtype=np.float64)
            rng = np.random.default_rng(seed + 10_000 + fold)
            shuffled_target = rng.permutation(y_train)
            if train_weight is None:
                sentinel = fit_standardized_ridge(
                    x_train, shuffled_target, alpha=alpha
                )
            else:
                sentinel = fit_weighted_standardized_ridge(
                    x_train,
                    shuffled_target,
                    train_weight,
                    alpha=alpha,
                )
            result.loc[test_mask, "prediction_target_shuffle_sentinel"] = (
                predict_standardized_ridge(sentinel, x_test)
            )

        diagnostics.append(
            {
                "fold": fold,
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "train_groups": len(train_groups),
                "test_groups": len(test_groups),
                "group_overlap": 0,
                "train_weight_sum": (
                    float(train_weight.sum()) if train_weight is not None else None
                ),
            }
        )

    prediction_columns = [column for column in result if column.startswith("prediction_")]
    if result[prediction_columns].isna().any().any():
        raise RuntimeError("cross-fitting left missing predictions")
    return result, {
        "fold_assignments": assignments,
        "folds": diagnostics,
        "scaler_fit_on_training_only": True,
        "group_overlap": 0,
    }


def grouped_spearman_table(
    predictions: pd.DataFrame,
    *,
    target_column: str,
    prediction_columns: Sequence[str],
    grouping_columns: Sequence[str],
) -> pd.DataFrame:
    """Compute one correlation row per group and prediction column."""
    rows: list[dict[str, object]] = []
    group_key: str | list[str]
    group_key = grouping_columns[0] if len(grouping_columns) == 1 else list(grouping_columns)
    for group_values, group in predictions.groupby(group_key, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        identity = dict(zip(grouping_columns, group_values, strict=True))
        for prediction_column in prediction_columns:
            rows.append(
                {
                    **identity,
                    "model": prediction_column.removeprefix("prediction_"),
                    "spearman_rho": safe_spearman(
                        group[prediction_column],
                        group[target_column],
                    ),
                    "n_rows": len(group),
                }
            )
    return pd.DataFrame(rows)


def paired_bootstrap_mean_difference(
    first: Sequence[float],
    second: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap a paired mean difference across independent groups."""
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    differences = x[valid] - y[valid]
    if not len(differences):
        raise ValueError("paired bootstrap has no finite pairs")
    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    return {
        "mean_difference": float(differences.mean()),
        "ci_95_low": float(np.quantile(bootstrap, 0.025)),
        "ci_95_high": float(np.quantile(bootstrap, 0.975)),
        "n_pairs": len(differences),
    }

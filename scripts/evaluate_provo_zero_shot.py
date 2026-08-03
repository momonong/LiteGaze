"""Frozen GECO-to-PROVO cross-corpus zero-shot evaluation.

The protocol is locked in ``docs/PROVO_ZERO_SHOT_PROTOCOL_2026-08-03.md``.
This module uses lexical features only, imports no Torch code, and requires an
explicit CPU device policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordfreq import zipf_frequency

if __package__:
    from .download_provo import verify_provo_file
    from .evaluate_geco_generalization import (
        _roc_auc,
        _safe_spearman,
        _sign_flip_pvalue,
        aggregate_fixation_durations,
        fit_ridge,
        predict_ridge,
    )
    from .experiment_manifest import (
        capture_source_snapshot,
        write_experiment_manifest,
    )
else:
    from download_provo import verify_provo_file
    from evaluate_geco_generalization import (
        _roc_auc,
        _safe_spearman,
        _sign_flip_pvalue,
        aggregate_fixation_durations,
        fit_ridge,
        predict_ridge,
    )
    from experiment_manifest import (
        capture_source_snapshot,
        write_experiment_manifest,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GECO_ROOT = PROJECT_ROOT / "data/geco/population/L1"
DEFAULT_PROVO_PATH = (
    PROJECT_ROOT / "data/provo/raw/Provo_Corpus-Eyetracking_Data.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
PROTOCOL_PATH = (
    PROJECT_ROOT / "docs/PROVO_ZERO_SHOT_PROTOCOL_V1_1_2026-08-03.md"
)

PROTOCOL_VERSION = "1.1"
RANDOM_SEED = 20260804
RIDGE_ALPHA = 1.0
LOGISTIC_ALPHA = 1.0
LOGISTIC_MAX_ITERATIONS = 100
LOGISTIC_TOLERANCE = 1e-8
BOOTSTRAP_SAMPLES = 10_000
SIGN_FLIP_SAMPLES = 100_000
EXPECTED_GECO_SUBJECTS = 18
EXPECTED_PROVO_SUBJECTS = 84
EXPECTED_PROVO_TEXTS = 55

FEATURE_COLUMNS = (
    "log_char_length",
    "zipf_frequency",
    "relative_position",
    "starts_upper",
    "ends_punctuation",
)
DURATION_MODELS = (
    "geco_lexical_ridge",
    "word_length_only",
    "lexical_rarity_only",
)
GECO_LAYOUT_COLUMNS = ("WORD_ID_WITHIN_TRIAL", "WORD")
GECO_FIXATION_COLUMNS = ("WORD_ID_WITHIN_TRIAL", "reading_time")
PROVO_COLUMNS = (
    "Participant_ID",
    "Word_Unique_ID",
    "Text_ID",
    "Word_Number",
    "Word",
    "Word_Cleaned",
    "Word_Length",
    "IA_ID",
    "IA_LABEL",
    "TRIAL_INDEX",
    "IA_DWELL_TIME",
    "IA_SKIP",
)


def _hash_file(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.digest()


def _json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                default=_json_default,
            )
            handle.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _strip_edge_punctuation(value: str) -> str:
    start = 0
    end = len(value)
    while start < end and unicodedata.category(value[start]).startswith("P"):
        start += 1
    while end > start and unicodedata.category(value[end - 1]).startswith("P"):
        end -= 1
    return value[start:end]


@cache
def word_features(word: str) -> tuple[float, float, float, float, float, str]:
    """Return frozen lexical features and the normalized lookup token."""
    normalized = unicodedata.normalize("NFKC", str(word)).strip()
    if not normalized:
        raise ValueError("display word is empty after NFKC normalization")

    char_length = sum(
        unicodedata.category(character)[0] in {"L", "N"}
        for character in normalized
    )
    lookup_token = _strip_edge_punctuation(normalized).casefold()
    frequency = (
        float(zipf_frequency(lookup_token, "en")) if lookup_token else 0.0
    )
    frequency = float(np.clip(frequency, 0.0, 8.0))

    first_cased = next(
        (
            character
            for character in normalized
            if character.lower() != character.upper()
        ),
        "",
    )
    starts_upper = float(bool(first_cased) and first_cased.isupper())
    ends_punctuation = float(
        unicodedata.category(normalized[-1]).startswith("P")
    )
    return (
        float(np.log1p(char_length)),
        frequency,
        float(char_length),
        starts_upper,
        ends_punctuation,
        lookup_token,
    )


def add_frozen_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the same outcome-independent feature function to either corpus."""
    required = {"subject", "trial", "word_id", "word", "word_position"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"feature frame is missing columns: {sorted(missing)}")

    result = frame.copy()
    if result[["subject", "trial", "word_id"]].duplicated().any():
        raise ValueError("duplicate subject/trial/word item before feature extraction")
    positions = pd.to_numeric(result["word_position"], errors="coerce")
    if not np.isfinite(positions).all():
        raise ValueError("word positions must all be finite")
    result["word_position"] = positions.astype(float)

    groups = result.groupby(["subject", "trial"], sort=False, observed=True)
    duplicate_positions = groups["word_position"].transform(
        lambda values: values.duplicated(keep=False)
    )
    if bool(duplicate_positions.any()):
        raise ValueError("word positions must be unique within each subject/trial")
    rank = groups["word_position"].rank(method="dense", ascending=True) - 1.0
    item_count = groups["word_id"].transform("size").astype(float)
    result["relative_position"] = rank / np.maximum(item_count - 1.0, 1.0)

    feature_rows = [word_features(value) for value in result["word"].astype(str)]
    lexical = pd.DataFrame(
        feature_rows,
        columns=(
            "log_char_length",
            "zipf_frequency",
            "char_length",
            "starts_upper",
            "ends_punctuation",
            "lookup_token",
        ),
        index=result.index,
    )
    for column in lexical.columns:
        result[column] = lexical[column]

    numeric_features = result.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(numeric_features).all():
        raise ValueError("frozen lexical features must all be finite")
    return result


def load_geco_l1(data_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load all GECO L1 word observations and fingerprint every source file."""
    root = data_root.resolve()
    layout_paths = sorted(root.glob("*/trial_*/layout.csv"))
    if not layout_paths:
        raise FileNotFoundError(f"no GECO L1 layouts found under {root}")

    frames: list[pd.DataFrame] = []
    tree_digest = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    for index, layout_path in enumerate(layout_paths, start=1):
        fixation_path = layout_path.with_name("fixations.csv")
        if not fixation_path.is_file():
            raise FileNotFoundError(f"missing paired GECO fixation file: {fixation_path}")

        for path in (layout_path, fixation_path):
            relative = path.relative_to(root).as_posix()
            size = path.stat().st_size
            tree_digest.update(relative.encode("utf-8"))
            tree_digest.update(b"\0")
            tree_digest.update(str(size).encode("ascii"))
            tree_digest.update(b"\0")
            tree_digest.update(_hash_file(path))
            total_bytes += size
            file_count += 1

        relative_parts = layout_path.relative_to(root).parts
        subject, trial = relative_parts[:2]
        layout = pd.read_csv(
            layout_path,
            usecols=list(GECO_LAYOUT_COLUMNS),
            keep_default_na=False,
        )
        fixations = pd.read_csv(
            fixation_path,
            usecols=list(GECO_FIXATION_COLUMNS),
            na_values=["."],
            keep_default_na=True,
        )
        if layout["WORD_ID_WITHIN_TRIAL"].duplicated().any():
            raise ValueError(f"duplicate GECO layout word IDs: {layout_path}")
        aggregated = aggregate_fixation_durations(fixations)
        merged = layout.merge(
            aggregated,
            on="WORD_ID_WITHIN_TRIAL",
            how="left",
            validate="one_to_one",
        )
        position = pd.to_numeric(
            merged["WORD_ID_WITHIN_TRIAL"], errors="coerce"
        )
        if not np.isfinite(position).all():
            raise ValueError(f"non-finite GECO word position: {layout_path}")
        reading_time = pd.to_numeric(merged["reading_time"], errors="coerce")
        fixated = np.isfinite(reading_time) & (reading_time > 0)
        frames.append(
            pd.DataFrame(
                {
                    "corpus": "GECO_L1",
                    "subject": str(subject),
                    "trial": str(trial),
                    "word_id": merged["WORD_ID_WITHIN_TRIAL"].astype(str),
                    "word_position": position.astype(float),
                    "word": merged["WORD"].astype(str),
                    "reading_time": reading_time.astype(float),
                    "fixated": fixated.astype(bool),
                }
            )
        )
        if index % 500 == 0 or index == len(layout_paths):
            print(f"Loaded GECO L1 {index:,}/{len(layout_paths):,} participant-trials")

    population = add_frozen_features(pd.concat(frames, ignore_index=True))
    population["log_reading_time"] = np.where(
        population["fixated"], np.log1p(population["reading_time"]), np.nan
    )
    subject_count = int(population["subject"].nunique())
    if subject_count != EXPECTED_GECO_SUBJECTS:
        raise RuntimeError(
            f"expected {EXPECTED_GECO_SUBJECTS} GECO L1 subjects, found {subject_count}"
        )
    fingerprint = {
        "corpus": "GECO_L1",
        "root": root.as_posix(),
        "subject_count": subject_count,
        "trial_count": int(population["trial"].nunique()),
        "participant_trial_count": int(
            population[["subject", "trial"]].drop_duplicates().shape[0]
        ),
        "row_count": len(population),
        "fixated_row_count": int(population["fixated"].sum()),
        "file_count": file_count,
        "size_bytes": total_bytes,
        "sha256_tree": tree_digest.hexdigest(),
    }
    return population, fingerprint


def validate_provo_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate the preregistered PROVO schema without outcome-based filtering."""
    missing = set(PROVO_COLUMNS).difference(raw.columns)
    if missing:
        raise ValueError(f"PROVO data is missing columns: {sorted(missing)}")
    values = raw.loc[:, PROVO_COLUMNS].copy()

    for column in ("Participant_ID", "IA_LABEL"):
        values[column] = values[column].astype(str).str.strip()
        if bool((values[column] == "").any()):
            raise ValueError(f"PROVO {column} contains empty values")

    text_id = pd.to_numeric(values["Text_ID"], errors="coerce")
    interest_area_id = pd.to_numeric(values["IA_ID"], errors="coerce")
    trial_index = pd.to_numeric(values["TRIAL_INDEX"], errors="coerce")
    if not (
        np.isfinite(text_id).all()
        and np.isfinite(interest_area_id).all()
        and np.isfinite(trial_index).all()
    ):
        raise ValueError(
            "PROVO text IDs, interest-area IDs, and trial indexes must be finite"
        )

    raw_skip = values["IA_SKIP"].replace("", np.nan)
    skip = pd.to_numeric(raw_skip, errors="coerce")
    invalid_skip = raw_skip.notna() & skip.isna()
    if bool(invalid_skip.any()) or bool(skip.isna().any()):
        raise ValueError("PROVO IA_SKIP contains missing or non-numeric values")
    if not bool(skip.isin([0, 1]).all()):
        raise ValueError("PROVO IA_SKIP must contain only zero or one")

    raw_dwell = values["IA_DWELL_TIME"].replace({"": np.nan, ".": np.nan})
    dwell = pd.to_numeric(raw_dwell, errors="coerce")
    invalid_dwell = raw_dwell.notna() & dwell.isna()
    if bool(invalid_dwell.any()):
        raise ValueError("PROVO IA_DWELL_TIME contains unexpected non-numeric values")
    positive_dwell = np.isfinite(dwell) & (dwell > 0)

    frame = pd.DataFrame(
        {
            "corpus": "PROVO",
            "subject": values["Participant_ID"],
            "trial": text_id.astype(int).astype(str),
            "word_id": interest_area_id.astype(int).astype(str),
            "word_position": interest_area_id.astype(float),
            "word": values["IA_LABEL"],
            "trial_index": trial_index.astype(int),
            "annotation_word_id": values["Word_Unique_ID"].astype(str),
            "annotation_word_number": values["Word_Number"].astype(str),
            "annotation_word": values["Word"].astype(str),
            "word_cleaned": values["Word_Cleaned"],
            "reported_word_length": values["Word_Length"].astype(str),
            "reading_time": dwell.astype(float),
            "fixated": positive_dwell.astype(bool),
            "first_pass_skipped": (skip == 1).astype(bool),
        }
    )
    key = ["subject", "trial", "word_id"]
    if frame[key].duplicated().any():
        raise ValueError("duplicate PROVO participant/text/word item rows")

    consistency = frame.groupby(
        ["trial", "word_id"], sort=False, observed=True
    ).agg(word_values=("word", "nunique"), positions=("word_position", "nunique"))
    if bool((consistency > 1).any().any()):
        raise ValueError("PROVO item text or position differs across participants")
    return add_frozen_features(frame)


def load_provo(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_fingerprint = verify_provo_file(path)
    raw = pd.read_csv(
        path,
        usecols=list(PROVO_COLUMNS),
        keep_default_na=False,
        low_memory=False,
    )
    population = validate_provo_frame(raw)
    population["log_reading_time"] = np.where(
        population["fixated"], np.log1p(population["reading_time"]), np.nan
    )
    subject_count = int(population["subject"].nunique())
    text_count = int(population["trial"].nunique())
    if subject_count != EXPECTED_PROVO_SUBJECTS:
        raise RuntimeError(
            f"expected {EXPECTED_PROVO_SUBJECTS} PROVO subjects, found {subject_count}"
        )
    if text_count != EXPECTED_PROVO_TEXTS:
        raise RuntimeError(
            f"expected {EXPECTED_PROVO_TEXTS} PROVO texts, found {text_count}"
        )
    fingerprint = {
        **source_fingerprint,
        "corpus": "PROVO",
        "subject_count": subject_count,
        "text_count": text_count,
        "participant_text_count": int(
            population[["subject", "trial"]].drop_duplicates().shape[0]
        ),
        "unique_word_item_count": int(
            population[["trial", "word_id"]].drop_duplicates().shape[0]
        ),
        "row_count": len(population),
        "fixated_row_count": int(population["fixated"].sum()),
        "skipped_row_count": int((~population["fixated"]).sum()),
        "first_pass_skipped_row_count": int(
            population["first_pass_skipped"].sum()
        ),
    }
    return population, fingerprint


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float = LOGISTIC_ALPHA,
    max_iterations: int = LOGISTIC_MAX_ITERATIONS,
    tolerance: float = LOGISTIC_TOLERANCE,
) -> dict[str, Any]:
    """Fit deterministic standardized L2 logistic regression with Newton steps."""
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("features must be 2D and aligned with one-dimensional labels")
    if not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("logistic labels must be a non-empty binary vector")
    if len(np.unique(labels)) != 2:
        raise ValueError("logistic training requires both classes")

    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    standardized = (features - mean) / scale
    design = np.column_stack([np.ones(len(standardized)), standardized])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    prevalence = float(np.clip(labels.mean(), 1e-8, 1 - 1e-8))
    coefficients = np.zeros(design.shape[1], dtype=float)
    coefficients[0] = math.log(prevalence / (1.0 - prevalence))

    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        linear = np.clip(design @ coefficients, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-linear))
        weights = np.clip(probabilities * (1.0 - probabilities), 1e-12, None)
        gradient = design.T @ (probabilities - labels) + penalty @ coefficients
        hessian = design.T @ (design * weights[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if np.linalg.norm(step, ord=np.inf) <= tolerance * (
            1.0 + np.linalg.norm(coefficients, ord=np.inf)
        ):
            converged = True
            break
    if not converged:
        raise RuntimeError(
            f"fixed logistic solver did not converge in {max_iterations} iterations"
        )
    return {
        "mean": mean,
        "scale": scale,
        "coefficients": coefficients,
        "iterations": iterations,
        "converged": converged,
    }


def predict_logistic(model: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    standardized = (features - model["mean"]) / model["scale"]
    design = np.column_stack([np.ones(len(standardized)), standardized])
    linear = np.clip(design @ model["coefficients"], -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-linear))


def fit_geco_models(frame: pd.DataFrame) -> dict[str, Any]:
    features = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    duration_rows = frame["fixated"].to_numpy(dtype=bool)
    duration_model = fit_ridge(
        features[duration_rows],
        frame.loc[duration_rows, "log_reading_time"].to_numpy(dtype=float),
        alpha=RIDGE_ALPHA,
    )
    fixation_model = fit_logistic(
        features,
        frame["fixated"].astype(int).to_numpy(),
        alpha=LOGISTIC_ALPHA,
    )
    return {
        "duration": duration_model,
        "fixation": fixation_model,
        "training_rows": len(frame),
        "duration_training_rows": int(duration_rows.sum()),
    }


def score_provo(frame: pd.DataFrame, models: Mapping[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    features = result.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    result["geco_lexical_ridge"] = predict_ridge(models["duration"], features)
    result["word_length_only"] = result["char_length"].to_numpy(dtype=float)
    result["lexical_rarity_only"] = -result["zipf_frequency"].to_numpy(dtype=float)
    result["geco_fixation_logistic"] = predict_logistic(models["fixation"], features)
    prediction_columns = [*DURATION_MODELS, "geco_fixation_logistic"]
    if not np.isfinite(result.loc[:, prediction_columns].to_numpy(dtype=float)).all():
        raise RuntimeError("every PROVO row must receive finite frozen predictions")
    return result


def participant_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject, group in scored.groupby("subject", sort=True, observed=True):
        duration = group.loc[group["fixated"]]
        target = duration["log_reading_time"].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "subject": subject,
            "duration_n": len(duration),
            "fixation_n": len(group),
        }
        for model_name in DURATION_MODELS:
            prediction = duration[model_name].to_numpy(dtype=float)
            row[f"{model_name}_spearman_rho"] = _safe_spearman(prediction, target)
        ridge_prediction = duration["geco_lexical_ridge"].to_numpy(dtype=float)
        row["geco_lexical_ridge_mae_log"] = float(
            np.mean(np.abs(ridge_prediction - target))
        )
        labels = group["fixated"].astype(int).to_numpy()
        fixation_scores = group["geco_fixation_logistic"].to_numpy(dtype=float)
        row["geco_fixation_logistic_roc_auc"] = _roc_auc(labels, fixation_scores)
        row["geco_fixation_logistic_brier"] = float(
            np.mean((fixation_scores - labels) ** 2)
        )
        rows.append(row)
    metrics = pd.DataFrame(rows)
    primary_column = "geco_lexical_ridge_spearman_rho"
    if len(metrics) != EXPECTED_PROVO_SUBJECTS or not np.isfinite(
        metrics[primary_column]
    ).all():
        raise RuntimeError("primary duration metric must be finite for all PROVO subjects")
    return metrics


def _bootstrap_mean(
    values: Sequence[float], *, samples: int, seed: int
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        raise ValueError("cannot bootstrap an empty metric")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    bootstrap = array[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return float(array.mean()), float(lower), float(upper)


def summarize_metrics(
    metrics: pd.DataFrame,
    *,
    bootstrap_samples: int,
    sign_flip_samples: int,
) -> dict[str, Any]:
    duration_summaries: list[dict[str, Any]] = []
    for index, model_name in enumerate(DURATION_MODELS):
        column = f"{model_name}_spearman_rho"
        mean, lower, upper = _bootstrap_mean(
            metrics[column],
            samples=bootstrap_samples,
            seed=RANDOM_SEED + index,
        )
        duration_summaries.append(
            {
                "model": model_name,
                "participant_count": int(np.isfinite(metrics[column]).sum()),
                "macro_mean_spearman_rho": mean,
                "bootstrap_ci_95": [lower, upper],
                "mean_mae_log": (
                    float(metrics["geco_lexical_ridge_mae_log"].mean())
                    if model_name == "geco_lexical_ridge"
                    else None
                ),
            }
        )

    ridge = metrics["geco_lexical_ridge_spearman_rho"].to_numpy(dtype=float)
    word_length = metrics["word_length_only_spearman_rho"].to_numpy(dtype=float)
    difference = ridge - word_length
    difference_mean, difference_lower, difference_upper = _bootstrap_mean(
        difference,
        samples=bootstrap_samples,
        seed=RANDOM_SEED + 20,
    )
    ridge_summary = duration_summaries[0]
    paired_p = _sign_flip_pvalue(
        difference,
        samples=sign_flip_samples,
        seed=RANDOM_SEED + 101,
    )
    against_zero_p = _sign_flip_pvalue(
        ridge,
        samples=sign_flip_samples,
        seed=RANDOM_SEED + 100,
    )
    if difference_mean > 0 and paired_p < 0.05:
        decision = "incremental_cross_corpus_evidence"
    elif ridge_summary["bootstrap_ci_95"][0] > 0:
        decision = "basic_lexical_transfer_only"
    else:
        decision = "no_confirmed_transfer"

    auc_mean, auc_lower, auc_upper = _bootstrap_mean(
        metrics["geco_fixation_logistic_roc_auc"],
        samples=bootstrap_samples,
        seed=RANDOM_SEED + 30,
    )
    brier_mean, brier_lower, brier_upper = _bootstrap_mean(
        metrics["geco_fixation_logistic_brier"],
        samples=bootstrap_samples,
        seed=RANDOM_SEED + 31,
    )
    return {
        "duration_models": duration_summaries,
        "primary_tests": {
            "ridge_vs_zero_two_sided_p": against_zero_p,
            "ridge_minus_word_length_macro_rho": difference_mean,
            "ridge_minus_word_length_bootstrap_ci_95": [
                difference_lower,
                difference_upper,
            ],
            "ridge_vs_word_length_paired_two_sided_p": paired_p,
        },
        "secondary_fixation": {
            "model": "geco_fixation_logistic",
            "participant_count": int(
                np.isfinite(metrics["geco_fixation_logistic_roc_auc"]).sum()
            ),
            "macro_mean_roc_auc": auc_mean,
            "roc_auc_bootstrap_ci_95": [auc_lower, auc_upper],
            "macro_mean_brier": brier_mean,
            "brier_bootstrap_ci_95": [brier_lower, brier_upper],
        },
        "decision": decision,
    }


def _serializable_models(models: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "training_corpus": "GECO_L1",
        "training_rows": models["training_rows"],
        "duration_training_rows": models["duration_training_rows"],
        "duration_ridge": {
            "alpha": RIDGE_ALPHA,
            "mean": models["duration"]["mean"].tolist(),
            "scale": models["duration"]["scale"].tolist(),
            "coefficients": models["duration"]["coefficients"].tolist(),
        },
        "fixation_logistic": {
            "alpha": LOGISTIC_ALPHA,
            "max_iterations": LOGISTIC_MAX_ITERATIONS,
            "tolerance": LOGISTIC_TOLERANCE,
            "iterations": models["fixation"]["iterations"],
            "converged": models["fixation"]["converged"],
            "mean": models["fixation"]["mean"].tolist(),
            "scale": models["fixation"]["scale"].tolist(),
            "coefficients": models["fixation"]["coefficients"].tolist(),
        },
    }


def _write_report(
    path: Path,
    *,
    fingerprints: Mapping[str, Any],
    summaries: Mapping[str, Any],
    duration_seconds: float,
) -> None:
    table_rows = []
    for row in summaries["duration_models"]:
        lower, upper = row["bootstrap_ci_95"]
        table_rows.append(
            f"| `{row['model']}` | {row['macro_mean_spearman_rho']:.4f} "
            f"| [{lower:.4f}, {upper:.4f}] |"
        )
    primary = summaries["duration_models"][0]
    tests = summaries["primary_tests"]
    fixation = summaries["secondary_fixation"]
    report = f"""# GECO-to-PROVO Zero-Shot Results

Protocol: `docs/PROVO_ZERO_SHOT_PROTOCOL_V1_1_2026-08-03.md` (v{PROTOCOL_VERSION})

Run completed: {datetime.now(UTC).isoformat()}

Runtime: {duration_seconds:.2f} seconds, CPU-only; Torch/CUDA not imported

## Data identity

- GECO L1 training: {fingerprints['geco']['subject_count']} participants,
  {fingerprints['geco']['participant_trial_count']:,} participant-trials,
  {fingerprints['geco']['row_count']:,} word rows
- GECO tree SHA-256: `{fingerprints['geco']['sha256_tree']}`
- PROVO test: {fingerprints['provo']['subject_count']} participants,
  {fingerprints['provo']['text_count']} texts,
  {fingerprints['provo']['row_count']:,} word rows
- PROVO source SHA-256: `{fingerprints['provo']['sha256']}`

## Frozen primary evaluation

The model was fitted on GECO L1 only. No PROVO observation was used for fitting,
feature scaling, calibration, filtering, or model selection.

| Model | Macro participant Spearman rho | Participant bootstrap 95% CI |
| --- | ---: | ---: |
{chr(10).join(table_rows)}

Primary Ridge result: **rho = {primary['macro_mean_spearman_rho']:.4f}**,
95% CI **[{primary['bootstrap_ci_95'][0]:.4f},
{primary['bootstrap_ci_95'][1]:.4f}]**; participant sign-flip p-value against
zero **{tests['ridge_vs_zero_two_sided_p']:.6f}**.

Ridge minus word length: **{tests['ridge_minus_word_length_macro_rho']:.4f}**,
95% bootstrap CI **[{tests['ridge_minus_word_length_bootstrap_ci_95'][0]:.4f},
{tests['ridge_minus_word_length_bootstrap_ci_95'][1]:.4f}]**; paired sign-flip
p-value **{tests['ridge_vs_word_length_paired_two_sided_p']:.6f}**.

Frozen decision: **`{summaries['decision']}`**.

## Secondary fixation occurrence

The GECO-trained logistic model reached macro participant ROC AUC
**{fixation['macro_mean_roc_auc']:.4f}**, 95% CI
**[{fixation['roc_auc_bootstrap_ci_95'][0]:.4f},
{fixation['roc_auc_bootstrap_ci_95'][1]:.4f}]**, and macro Brier score
**{fixation['macro_mean_brier']:.4f}**.

## Interpretation guardrails

- PROVO is a completely independent corpus and was never used to tune this model.
- No question-answer data, PROVO cloze norm, LSA value, or predictability field was
  used.
- GECO surprisal, attention, and `cognitive_mass` were excluded because an identical
  frozen feature generator was not available for both corpora or provenance was
  target-sensitive.
- Spearman is the primary endpoint because absolute reading-time calibration can
  change across eye trackers, preprocessing pipelines, and participant populations.
- PROVO v1 is now a frozen test result and must not become a development set.
"""
    path.write_text(report, encoding="utf-8", newline="\n")


def _write_plot(path: Path, summaries: Mapping[str, Any]) -> None:
    rows = summaries["duration_models"]
    labels = [row["model"].replace("_", "\n") for row in rows]
    values = np.asarray([row["macro_mean_spearman_rho"] for row in rows])
    lower = np.asarray([row["bootstrap_ci_95"][0] for row in rows])
    upper = np.asarray([row["bootstrap_ci_95"][1] for row in rows])
    positions = np.arange(len(rows))

    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.bar(positions, values, color=["#1b9e77", "#7570b3", "#d95f02"])
    axis.errorbar(
        positions,
        values,
        yerr=np.vstack([values - lower, upper - values]),
        fmt="none",
        ecolor="black",
        capsize=5,
        linewidth=1.2,
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Macro participant Spearman rho")
    axis.set_title("Frozen GECO L1 to PROVO zero-shot transfer")
    axis.text(
        0.99,
        0.02,
        f"Decision: {summaries['decision'].replace('_', ' ')}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geco-root", type=Path, default=DEFAULT_GECO_ROOT)
    parser.add_argument("--provo-path", type=Path, default=DEFAULT_PROVO_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--sign-flip-samples", type=int, default=SIGN_FLIP_SAMPLES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.environ.get("LEXIGAZE_DEVICE", "auto").strip().lower() != "cpu":
        raise RuntimeError("frozen PROVO run requires LEXIGAZE_DEVICE=cpu")
    if "torch" in sys.modules:
        raise RuntimeError("PROVO evaluator must not import Torch/CUDA")
    if args.bootstrap_samples <= 0 or args.sign_flip_samples <= 0:
        raise ValueError("resampling counts must be positive")

    started = time.perf_counter()
    source_snapshot = capture_source_snapshot(
        PROJECT_ROOT,
        extra_files=[PROTOCOL_PATH, Path(__file__).with_name("download_provo.py")],
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    geco, geco_fingerprint = load_geco_l1(args.geco_root)
    provo, provo_fingerprint = load_provo(args.provo_path)
    models = fit_geco_models(geco)
    scored = score_provo(provo, models)
    subject_metrics = participant_metrics(scored)
    summaries = summarize_metrics(
        subject_metrics,
        bootstrap_samples=args.bootstrap_samples,
        sign_flip_samples=args.sign_flip_samples,
    )
    if "torch" in sys.modules:
        raise RuntimeError("Torch was imported during the CPU-only PROVO run")

    duration_seconds = time.perf_counter() - started
    fingerprint_path = output_dir / "provo_zero_shot_fingerprint.json"
    subject_path = output_dir / "provo_zero_shot_subject_metrics.csv"
    model_path = output_dir / "provo_zero_shot_models.json"
    summary_path = output_dir / "provo_zero_shot_summary.json"
    report_path = output_dir / "provo_zero_shot_report.md"
    plot_path = output_dir / "provo_zero_shot.png"
    manifest_path = output_dir / "provo_zero_shot_manifest.json"

    fingerprints = {"geco": geco_fingerprint, "provo": provo_fingerprint}
    _write_json(fingerprint_path, fingerprints)
    subject_metrics.to_csv(subject_path, index=False, lineterminator="\n")
    _write_json(model_path, _serializable_models(models))
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "duration_seconds": duration_seconds,
        "configuration": {
            "seed": RANDOM_SEED,
            "ridge_alpha": RIDGE_ALPHA,
            "logistic_alpha": LOGISTIC_ALPHA,
            "logistic_max_iterations": LOGISTIC_MAX_ITERATIONS,
            "logistic_tolerance": LOGISTIC_TOLERANCE,
            "bootstrap_samples": args.bootstrap_samples,
            "sign_flip_samples": args.sign_flip_samples,
            "feature_columns": list(FEATURE_COLUMNS),
            "protocol_path": PROTOCOL_PATH.relative_to(PROJECT_ROOT).as_posix(),
        },
        "datasets": fingerprints,
        "results": summaries,
        "leakage_controls": {
            "question_answer_dataset_used": False,
            "provo_outcomes_used_for_training": False,
            "provo_features_used_for_scaling": False,
            "provo_corpus_offset_fitted": False,
            "provo_cloze_or_lsa_used": False,
            "geco_cognitive_mass_used": False,
            "torch_imported": False,
            "device": "cpu",
        },
    }
    _write_json(summary_path, summary)
    _write_report(
        report_path,
        fingerprints=fingerprints,
        summaries=summaries,
        duration_seconds=duration_seconds,
    )
    _write_plot(plot_path, summaries)

    artifacts = [
        fingerprint_path,
        subject_path,
        model_path,
        summary_path,
        report_path,
        plot_path,
    ]
    write_experiment_manifest(
        manifest_path,
        "geco_to_provo_zero_shot_v1_1",
        root=PROJECT_ROOT,
        datasets=[args.provo_path, fingerprint_path],
        artifacts=artifacts,
        config=summary["configuration"],
        metrics={
            "decision": summaries["decision"],
            "primary_model": summaries["duration_models"][0],
            "primary_tests": summaries["primary_tests"],
            "secondary_fixation": summaries["secondary_fixation"],
        },
        seed=RANDOM_SEED,
        duration_seconds=duration_seconds,
        source_snapshot=source_snapshot,
        packages=("numpy", "pandas", "scipy", "matplotlib", "wordfreq"),
    )

    print(f"Completed frozen PROVO zero-shot study in {duration_seconds:.2f}s")
    print(f"Decision: {summaries['decision']}")
    print(f"Report: {report_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

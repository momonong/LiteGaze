"""Runtime-safe loading and scoring for provenance-complete text artifacts.

The module deliberately depends only on NumPy and the standard library.  It
does not import the language-model runtime, touch CUDA, or perform request-local
normalization.  Feature extraction remains a separate, auditable step.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_TYPE = "standardized_ridge_text_difficulty"


class ArtifactValidationError(ValueError):
    """Raised when a candidate artifact violates its frozen contract."""


@dataclass(frozen=True)
class TextDifficultyPrediction:
    """Candidate text scores plus an explicit distribution guard."""

    raw_prediction: np.ndarray
    score: np.ndarray
    in_distribution: np.ndarray
    out_of_bounds_feature_count: np.ndarray


@dataclass(frozen=True)
class TextDifficultyArtifact:
    """Validated standardized-Ridge parameters and fixed score calibration."""

    artifact_id: str
    language: str
    features: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    alpha: float
    score_lower: float
    score_upper: float
    feature_lower: np.ndarray
    feature_upper: np.ndarray
    payload: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> TextDifficultyArtifact:
        """Validate and materialize one JSON-compatible artifact mapping."""
        if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ArtifactValidationError("unsupported text artifact schema")
        if payload.get("artifact_type") != ARTIFACT_TYPE:
            raise ArtifactValidationError("unexpected text artifact type")

        artifact_id = str(payload.get("artifact_id", "")).strip()
        if not artifact_id:
            raise ArtifactValidationError("artifact_id must be non-empty")
        language = str(payload.get("language", "")).strip().lower()
        if language != "en":
            raise ArtifactValidationError("v1 text artifact must target English")

        raw_features = payload.get("feature_schema")
        if not isinstance(raw_features, list) or not raw_features:
            raise ArtifactValidationError("feature_schema must be a non-empty list")
        features = tuple(str(feature) for feature in raw_features)
        if len(features) != len(set(features)):
            raise ArtifactValidationError("feature_schema contains duplicates")
        if "causal_surprisal" not in features:
            raise ArtifactValidationError("causal_surprisal is required")
        if any("entropy" in feature for feature in features):
            raise ArtifactValidationError("entropy is not eligible for artifact v1")

        feature_policy = _mapping(payload, "feature_policy")
        if feature_policy.get("context_direction") != "left_only":
            raise ArtifactValidationError("text artifact must use left-only context")
        if feature_policy.get("language_model_frozen") is not True:
            raise ArtifactValidationError("language model must remain frozen")
        if feature_policy.get("language_model_fine_tuned") is not False:
            raise ArtifactValidationError("fine-tuned language models are not allowed")

        model = _mapping(payload, "model")
        if model.get("family") != "standardized_ridge":
            raise ArtifactValidationError("artifact model must be standardized Ridge")
        mean = _finite_vector(model.get("feature_mean"), "model.feature_mean")
        scale = _finite_vector(model.get("feature_scale"), "model.feature_scale")
        coefficients = _finite_vector(
            model.get("coefficients"), "model.coefficients"
        )
        if not (len(mean) == len(scale) == len(coefficients) == len(features)):
            raise ArtifactValidationError("model vectors do not match feature_schema")
        if np.any(scale <= 0):
            raise ArtifactValidationError("model feature scales must be positive")
        intercept = _finite_float(model.get("intercept"), "model.intercept")
        alpha = _finite_float(model.get("alpha"), "model.alpha")
        if alpha < 0:
            raise ArtifactValidationError("model alpha must be non-negative")

        calibration = _mapping(payload, "score_calibration")
        if calibration.get("method") != "fixed_training_prediction_quantile_clip":
            raise ArtifactValidationError("unexpected score calibration method")
        if calibration.get("request_local_normalization") is not False:
            raise ArtifactValidationError("request-local normalization is forbidden")
        score_lower = _finite_float(
            calibration.get("raw_lower"), "score_calibration.raw_lower"
        )
        score_upper = _finite_float(
            calibration.get("raw_upper"), "score_calibration.raw_upper"
        )
        if score_upper <= score_lower:
            raise ArtifactValidationError("score calibration bounds are invalid")

        distribution = _mapping(payload, "training_distribution")
        feature_lower = _finite_vector(
            distribution.get("feature_lower"),
            "training_distribution.feature_lower",
        )
        feature_upper = _finite_vector(
            distribution.get("feature_upper"),
            "training_distribution.feature_upper",
        )
        if len(feature_lower) != len(features) or len(feature_upper) != len(features):
            raise ArtifactValidationError(
                "training distribution bounds do not match feature_schema"
            )
        if np.any(feature_upper < feature_lower):
            raise ArtifactValidationError("training distribution bounds are invalid")

        training = _mapping(payload, "training")
        corpora = {str(name) for name in training.get("corpora", [])}
        excluded = {str(name) for name in training.get("excluded_corpora", [])}
        if corpora != {"PROVO", "GECO_L2_English"}:
            raise ArtifactValidationError("artifact has an unexpected training corpus")
        if "OneStop_Eye_Movements" in corpora:
            raise ArtifactValidationError("OneStop must never enter artifact training")
        if "OneStop_Eye_Movements" not in excluded:
            raise ArtifactValidationError("OneStop exclusion must be explicit")

        leakage = _mapping(payload, "leakage_controls")
        false_controls = (
            "question_answer_dataset_used",
            "onestop_used_for_training_selection_or_thresholding",
            "language_model_fine_tuned",
            "request_local_normalization_used",
        )
        if any(leakage.get(name) is not False for name in false_controls):
            raise ArtifactValidationError("artifact leakage controls are incomplete")

        promotion = _mapping(payload, "promotion")
        if promotion.get("status") != "candidate":
            raise ArtifactValidationError("v1 artifact must remain a candidate")
        if promotion.get("independent_fusion_evaluation_required") is not True:
            raise ArtifactValidationError("independent fusion evaluation is required")

        return cls(
            artifact_id=artifact_id,
            language=language,
            features=features,
            mean=mean,
            scale=scale,
            coefficients=coefficients,
            intercept=intercept,
            alpha=alpha,
            score_lower=score_lower,
            score_upper=score_upper,
            feature_lower=feature_lower,
            feature_upper=feature_upper,
            payload=payload,
        )

    def predict_matrix(self, values: np.ndarray) -> TextDifficultyPrediction:
        """Score a matrix using only frozen training-time transformations."""
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.features):
            raise ValueError("feature matrix shape does not match artifact schema")
        if not np.isfinite(matrix).all():
            raise ValueError("feature matrix must contain only finite values")

        standardized = (matrix - self.mean) / self.scale
        raw = self.intercept + standardized @ self.coefficients
        score = np.clip(
            (raw - self.score_lower) / (self.score_upper - self.score_lower),
            0.0,
            1.0,
        )
        out_of_bounds = (matrix < self.feature_lower) | (matrix > self.feature_upper)
        count = out_of_bounds.sum(axis=1).astype(np.int64)
        return TextDifficultyPrediction(
            raw_prediction=raw,
            score=score,
            in_distribution=count == 0,
            out_of_bounds_feature_count=count,
        )

    def predict_records(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> TextDifficultyPrediction:
        """Score feature mappings in the exact artifact column order."""
        matrix = np.asarray(
            [
                [_finite_float(record.get(feature), feature) for feature in self.features]
                for record in records
            ],
            dtype=np.float64,
        )
        if not len(records):
            matrix = np.empty((0, len(self.features)), dtype=np.float64)
        return self.predict_matrix(matrix)


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for an artifact or manifest input."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_text_difficulty_artifact(
    path: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> TextDifficultyArtifact:
    """Load an artifact and optionally verify its external SHA-256 manifest."""
    artifact_path = Path(path)
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"cannot read text artifact: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ArtifactValidationError("text artifact root must be an object")

    artifact = TextDifficultyArtifact.from_mapping(payload)
    if manifest_path is None:
        return artifact

    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"cannot read artifact manifest: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ArtifactValidationError("artifact manifest root must be an object")
    if manifest.get("schema_version") != 1:
        raise ArtifactValidationError("unsupported artifact manifest schema")
    if manifest.get("artifact_id") != artifact.artifact_id:
        raise ArtifactValidationError("artifact manifest ID mismatch")
    artifact_entry = _mapping(manifest, "artifact")
    if artifact_entry.get("file") != artifact_path.name:
        raise ArtifactValidationError("artifact manifest filename mismatch")
    expected_hash = str(artifact_entry.get("sha256", ""))
    if not expected_hash or sha256_file(artifact_path) != expected_hash:
        raise ArtifactValidationError("artifact SHA-256 mismatch")
    return artifact


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ArtifactValidationError(f"{key} must be an object")
    return value


def _finite_vector(value: Any, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"{name} must be numeric") from exc
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ArtifactValidationError(f"{name} must be a finite vector")
    return vector


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ArtifactValidationError(f"{name} must be finite")
    return result

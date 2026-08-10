"""Deterministic, CPU-only audit of webcam gaze measurement resolution.

This module intentionally uses only the Python standard library. It evaluates
explicit fixed-target observations and never treats natural-reading word
assignments as gaze ground truth.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .uncertainty_contract import (
    canonical_json_bytes as _uncertainty_canonical_json_bytes,
    canonical_sha256 as _uncertainty_canonical_sha256,
    normalize_uncertainty_observation,
    verified_definition,
)


SCHEMA_VERSION = 1
ANALYSIS_ID = "webcam-gaze-measurement-ceiling-v1"
CORRECTION_ID = "start_trained_median_translation"
DEFAULT_TARGET_OVERLAP_TOLERANCE_SIGNED = 0.2
FROZEN_PROTOCOL_TARGET_SEPARATION_VIEWPORT_FRACTION = 0.1
MAX_CROSS_PHASE_CAMERA_ASPECT_RATIO_DIFFERENCE = 0.02
SIGNED_COORDINATE_MIN = -1.0
SIGNED_COORDINATE_MAX = 1.0
COORDINATE_ABS_TOLERANCE = 1e-12
REPEATABILITY_PROXY_COVERAGE_LEVELS = (0.2, 0.4, 0.6, 0.8, 1.0)
UNCERTAINTY_V2_COVERAGE_LEVELS = (1.0, 0.8, 0.6, 0.4, 0.2)
FIXED_TARGET_CLUSTER_COUNT = 5
FIXED_TARGET_REPEATS_PER_PHASE = 3


class MeasurementCeilingError(ValueError):
    """Raised when input artifacts cannot support the bounded audit."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MeasurementCeilingError(f"cannot read {label}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise MeasurementCeilingError(f"{label} must contain a JSON object")
    return decoded


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise MeasurementCeilingError(f"cannot read {label}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MeasurementCeilingError(
                f"{label} line {line_number} is not valid JSON"
            ) from exc
        if not isinstance(decoded, dict):
            raise MeasurementCeilingError(
                f"{label} line {line_number} must contain an object"
            )
        records.append(decoded)
    if not records:
        raise MeasurementCeilingError(f"{label} has no records")
    return records


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise MeasurementCeilingError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementCeilingError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise MeasurementCeilingError(f"{field} must be finite")
    return number


def _positive(value: Any, *, field: str) -> float:
    number = _finite(value, field=field)
    if number <= 0:
        raise MeasurementCeilingError(f"{field} must be positive")
    return number


def _signed_normalized(value: Any, *, field: str) -> float:
    number = _finite(value, field=field)
    if not SIGNED_COORDINATE_MIN <= number <= SIGNED_COORDINATE_MAX:
        raise MeasurementCeilingError(f"{field} must be within [-1, 1]")
    return number


def _signed_distance_tolerance(value: Any) -> float:
    number = _positive(value, field="target_overlap_tolerance")
    maximum = math.hypot(
        SIGNED_COORDINATE_MAX - SIGNED_COORDINATE_MIN,
        SIGNED_COORDINATE_MAX - SIGNED_COORDINATE_MIN,
    )
    if number > maximum:
        raise MeasurementCeilingError(
            "target_overlap_tolerance exceeds the signed-coordinate diagonal"
        )
    return number


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise MeasurementCeilingError("cannot calculate a quantile without values")
    if not 0 <= fraction <= 1:
        raise MeasurementCeilingError("quantile fraction must be between zero and one")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    """Match the participant collection's empirical percentile contract."""

    if not values:
        raise MeasurementCeilingError("cannot calculate a percentile without values")
    if not 0 < fraction <= 1:
        raise MeasurementCeilingError(
            "nearest-rank fraction must be greater than zero and at most one"
        )
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def _value_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise MeasurementCeilingError("metric summary has no values")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": _nearest_rank(values, 0.90),
        "p95": _nearest_rank(values, 0.95),
    }


def _axis_summary(values: Sequence[float]) -> dict[str, float | int]:
    absolute = [abs(value) for value in values]
    return {
        "count": len(values),
        "signed_mean": statistics.fmean(values),
        "signed_median": statistics.median(values),
        "absolute_mean": statistics.fmean(absolute),
        "absolute_median": statistics.median(absolute),
        "absolute_p90": _nearest_rank(absolute, 0.90),
        "absolute_p95": _nearest_rank(absolute, 0.95),
    }


def _validation_records(
    validation: Mapping[str, Any],
    *,
    phase: str,
) -> tuple[list[dict[str, Any]], int]:
    raw_samples = validation.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise MeasurementCeilingError(f"{phase} validation has no samples")
    successful: list[dict[str, Any]] = []
    for index, sample in enumerate(raw_samples):
        if not isinstance(sample, Mapping):
            raise MeasurementCeilingError(
                f"{phase} validation sample {index} must be an object"
            )
        if sample.get("prediction_success") is not True:
            continue
        target_id = str(sample.get("target_id") or "").strip()
        if not target_id:
            raise MeasurementCeilingError(
                f"{phase} validation sample {index} lacks target_id"
            )
        target_x = _finite(
            sample.get("target_x_px"), field=f"{phase}.target_x_px"
        )
        target_y = _finite(
            sample.get("target_y_px"), field=f"{phase}.target_y_px"
        )
        predicted_x = _finite(
            sample.get("predicted_x_px"), field=f"{phase}.predicted_x_px"
        )
        predicted_y = _finite(
            sample.get("predicted_y_px"), field=f"{phase}.predicted_y_px"
        )
        delta_x = predicted_x - target_x
        delta_y = predicted_y - target_y
        successful.append(
            {
                "target_id": target_id,
                "target_x_px": target_x,
                "target_y_px": target_y,
                "predicted_x_px": predicted_x,
                "predicted_y_px": predicted_y,
                "signed_error_x_px": delta_x,
                "signed_error_y_px": delta_y,
                "spatial_error_px": math.hypot(delta_x, delta_y),
            }
        )
    if not successful:
        raise MeasurementCeilingError(
            f"{phase} validation has no successful predictions"
        )
    return successful, len(raw_samples)


def _target_coordinates(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[float, float]]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for record in records:
        grouped[str(record["target_id"])].append(
            (float(record["target_x_px"]), float(record["target_y_px"]))
        )
    coordinates: dict[str, tuple[float, float]] = {}
    for target_id in sorted(grouped):
        values = grouped[target_id]
        x_values = [item[0] for item in values]
        y_values = [item[1] for item in values]
        if max(x_values) - min(x_values) > 1e-6 or max(y_values) - min(y_values) > 1e-6:
            raise MeasurementCeilingError(
                f"target {target_id} has inconsistent coordinates"
            )
        coordinates[target_id] = (x_values[0], y_values[0])
    target_ids = sorted(coordinates)
    for position, target_id in enumerate(target_ids):
        for other_id in target_ids[position + 1 :]:
            if math.dist(coordinates[target_id], coordinates[other_id]) <= 1e-9:
                raise MeasurementCeilingError(
                    f"targets {target_id} and {other_id} share one coordinate"
                )
    return coordinates


def _nearest_target(
    predicted_x: float,
    predicted_y: float,
    targets: Mapping[str, tuple[float, float]],
) -> str:
    return min(
        sorted(targets),
        key=lambda target_id: (
            (predicted_x - targets[target_id][0]) ** 2
            + (predicted_y - targets[target_id][1]) ** 2,
            target_id,
        ),
    )


def _phase_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    attempted_count: int,
) -> dict[str, Any]:
    targets = _target_coordinates(records)
    by_target: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    confusion: dict[str, dict[str, int]] = {
        target_id: {predicted_id: 0 for predicted_id in sorted(targets)}
        for target_id in sorted(targets)
    }
    correct = 0
    for record in records:
        target_id = str(record["target_id"])
        by_target[target_id].append(record)
        predicted_id = _nearest_target(
            float(record["predicted_x_px"]),
            float(record["predicted_y_px"]),
            targets,
        )
        confusion[target_id][predicted_id] += 1
        correct += int(predicted_id == target_id)

    target_metrics: dict[str, dict[str, Any]] = {}
    macro_mean_errors: list[float] = []
    macro_median_errors: list[float] = []
    target_bias_x: list[float] = []
    target_bias_y: list[float] = []
    target_bias_magnitudes: list[float] = []
    for target_id in sorted(by_target):
        target_records = by_target[target_id]
        errors = [float(item["spatial_error_px"]) for item in target_records]
        x_errors = [float(item["signed_error_x_px"]) for item in target_records]
        y_errors = [float(item["signed_error_y_px"]) for item in target_records]
        mean_error = statistics.fmean(errors)
        median_error = statistics.median(errors)
        mean_bias_x = statistics.fmean(x_errors)
        mean_bias_y = statistics.fmean(y_errors)
        macro_mean_errors.append(mean_error)
        macro_median_errors.append(median_error)
        target_bias_x.append(mean_bias_x)
        target_bias_y.append(mean_bias_y)
        bias_magnitude = math.hypot(mean_bias_x, mean_bias_y)
        target_bias_magnitudes.append(bias_magnitude)
        target_metrics[target_id] = {
            "sample_count": len(target_records),
            "target_x_px": targets[target_id][0],
            "target_y_px": targets[target_id][1],
            "predicted_centroid_x_px": statistics.fmean(
                float(item["predicted_x_px"]) for item in target_records
            ),
            "predicted_centroid_y_px": statistics.fmean(
                float(item["predicted_y_px"]) for item in target_records
            ),
            "signed_bias_x_px": mean_bias_x,
            "signed_bias_y_px": mean_bias_y,
            "bias_magnitude_px": bias_magnitude,
            "mean_spatial_error_px": mean_error,
            "median_spatial_error_px": median_error,
            "p90_spatial_error_px": _nearest_rank(errors, 0.90),
        }

    errors = [float(record["spatial_error_px"]) for record in records]
    x_errors = [float(record["signed_error_x_px"]) for record in records]
    y_errors = [float(record["signed_error_y_px"]) for record in records]
    return {
        "attempted_sample_count": attempted_count,
        "successful_sample_count": len(records),
        "prediction_success_fraction": len(records) / attempted_count,
        "target_count": len(targets),
        "spatial_error_px": _value_summary(errors),
        "x_error_px": _axis_summary(x_errors),
        "y_error_px": _axis_summary(y_errors),
        "target_macro": {
            "mean_spatial_error_px": statistics.fmean(macro_mean_errors),
            "median_spatial_error_px": statistics.median(macro_median_errors),
            "signed_bias_x_px": statistics.fmean(target_bias_x),
            "signed_bias_y_px": statistics.fmean(target_bias_y),
            "mean_bias_magnitude_px": statistics.fmean(target_bias_magnitudes),
            "median_bias_magnitude_px": statistics.median(target_bias_magnitudes),
        },
        "coarse_region": {
            "definition": "nearest explicit evaluation target in pixel space",
            "correct_count": correct,
            "sample_count": len(records),
            "accuracy": correct / len(records),
            "confusion": confusion,
        },
        "targets": target_metrics,
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and math.isclose(
            values[order[end]],
            values[order[position]],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            end += 1
        average_rank = (position + 1 + end) / 2.0
        for ordered_index in order[position:end]:
            ranks[ordered_index] = average_rank
        position = end
    return ranks


def _spearman(values_x: Sequence[float], values_y: Sequence[float]) -> float | None:
    if len(values_x) != len(values_y) or len(values_x) < 2:
        return None
    ranks_x = _average_ranks(values_x)
    ranks_y = _average_ranks(values_y)
    mean_x = statistics.fmean(ranks_x)
    mean_y = statistics.fmean(ranks_y)
    centered_x = [value - mean_x for value in ranks_x]
    centered_y = [value - mean_y for value in ranks_y]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0.0:
        return None
    return sum(
        value_x * value_y
        for value_x, value_y in zip(centered_x, centered_y, strict=True)
    ) / denominator


def _temporal_repeatability_proxy(
    start_records: Sequence[Mapping[str, Any]],
    end_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Rank end risk with a score computed only from repeated start predictions.

    This is a target-cluster repeatability diagnostic, not predictive model
    uncertainty and not a deployable per-sample abstention score.
    """

    start_by_target: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    end_by_target: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in start_records:
        start_by_target[str(record["target_id"])].append(record)
    for record in end_records:
        end_by_target[str(record["target_id"])].append(record)
    if set(start_by_target) != set(end_by_target):
        return {
            "status": "not_evaluable",
            "claim_boundary": "proxy_not_predictive_uncertainty",
            "reason": "start and end validation target sets differ",
        }
    if len(start_by_target) != 5:
        return {
            "status": "not_evaluable",
            "claim_boundary": "proxy_not_predictive_uncertainty",
            "reason": "fixed coverage grid requires exactly five target clusters",
            "observed_target_cluster_count": len(start_by_target),
        }

    targets: dict[str, dict[str, Any]] = {}
    for target_id in sorted(start_by_target):
        start_target = start_by_target[target_id]
        end_target = end_by_target[target_id]
        if len(start_target) < 2 or not end_target:
            return {
                "status": "not_evaluable",
                "claim_boundary": "proxy_not_predictive_uncertainty",
                "reason": (
                    "every target needs at least two successful start predictions "
                    "and one successful end prediction"
                ),
            }
        centroid_x = statistics.fmean(
            float(record["predicted_x_px"]) for record in start_target
        )
        centroid_y = statistics.fmean(
            float(record["predicted_y_px"]) for record in start_target
        )
        squared_distances = [
            (float(record["predicted_x_px"]) - centroid_x) ** 2
            + (float(record["predicted_y_px"]) - centroid_y) ** 2
            for record in start_target
        ]
        end_errors = [float(record["spatial_error_px"]) for record in end_target]
        targets[target_id] = {
            "start_successful_repeat_count": len(start_target),
            "start_repeatability_rms_px": math.sqrt(
                statistics.fmean(squared_distances)
            ),
            "end_successful_sample_count": len(end_target),
            "end_mean_spatial_error_px": statistics.fmean(end_errors),
            "end_median_spatial_error_px": statistics.median(end_errors),
        }

    ordered_target_ids = sorted(
        targets,
        key=lambda target_id: (
            targets[target_id]["start_repeatability_rms_px"],
            target_id,
        ),
    )
    curve: list[dict[str, Any]] = []
    target_count = len(ordered_target_ids)
    for requested_coverage in REPEATABILITY_PROXY_COVERAGE_LEVELS:
        retained_count = max(1, math.ceil(target_count * requested_coverage))
        retained = ordered_target_ids[:retained_count]
        curve.append(
            {
                "requested_coverage": requested_coverage,
                "achieved_coverage": retained_count / target_count,
                "retained_target_cluster_count": retained_count,
                "retained_target_ids": retained,
                "end_target_macro_mean_spatial_error_px": statistics.fmean(
                    targets[target_id]["end_mean_spatial_error_px"]
                    for target_id in retained
                ),
                "end_target_macro_median_spatial_error_px": statistics.median(
                    targets[target_id]["end_median_spatial_error_px"]
                    for target_id in retained
                ),
            }
        )
    proxy_values = [
        targets[target_id]["start_repeatability_rms_px"]
        for target_id in sorted(targets)
    ]
    end_risks = [
        targets[target_id]["end_mean_spatial_error_px"]
        for target_id in sorted(targets)
    ]
    spearman = _spearman(proxy_values, end_risks)
    lowest_risk = float(curve[0]["end_target_macro_mean_spatial_error_px"])
    full_risk = float(curve[-1]["end_target_macro_mean_spatial_error_px"])
    improves = lowest_risk < full_risk
    return {
        "status": "evaluable_descriptive_proxy",
        "claim_boundary": "proxy_not_predictive_uncertainty",
        "score_fit_phase": "start_validation_predictions_only",
        "risk_evaluation_phase": "end_validation_target_errors_only",
        "selection_unit": "evaluation_target_id_cluster",
        "score_definition": (
            "root_mean_squared_radial_distance_from_start_prediction_centroid_px"
        ),
        "score_uses_target_error_or_end_data": False,
        "fixed_requested_coverages": list(REPEATABILITY_PROXY_COVERAGE_LEVELS),
        "threshold_selection_authorized": False,
        "quality_band_change_authorized": False,
        "per_sample_abstention_authorized": False,
        "target_cluster_count": target_count,
        "ordered_target_ids_low_to_high_proxy": ordered_target_ids,
        "targets": targets,
        "coverage_risk_curve": curve,
        "association": {
            "metric": "spearman_start_proxy_vs_end_target_mean_error",
            "value": spearman,
            "expected_direction_if_proxy_were_useful": "positive",
        },
        "negative_result": {
            "lowest_coverage_end_mean_error_px": lowest_risk,
            "full_coverage_end_mean_error_px": full_risk,
            "lowest_coverage_minus_full_coverage_px": lowest_risk - full_risk,
            "lowest_coverage_improves_over_full": improves,
            "conclusion": (
                "available_start_repeatability_proxy_reduces_end_risk"
                if improves and spearman is not None and spearman > 0.0
                else "available_start_repeatability_proxy_does_not_rank_end_risk"
            ),
        },
    }


def _lower_hex_sha256(value: Any, *, field: str) -> str:
    digest = str(value or "").strip()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise MeasurementCeilingError(f"{field} must be a lowercase SHA-256")
    return digest


def _uncertainty_model_binding(model: Mapping[str, Any]) -> dict[str, Any]:
    """Verify that runtime scores are bound to the frozen training-only v2."""

    try:
        definition, definition_sha256 = verified_definition()
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "failed_integrity",
            "reason": f"frozen uncertainty definition could not be verified: {exc}",
        }
    try:
        frozen_grid = [
            float(value)
            for value in dict(definition.get("coverage_risk") or {}).get(
                "coverage_grid", []
            )
        ]
    except (TypeError, ValueError) as exc:
        return {
            "status": "failed_integrity",
            "reason": f"frozen uncertainty coverage grid is invalid: {exc}",
            "frozen_definition_sha256": definition_sha256,
        }
    if frozen_grid != list(UNCERTAINTY_V2_COVERAGE_LEVELS):
        return {
            "status": "failed_integrity",
            "reason": "frozen uncertainty coverage grid changed",
            "frozen_definition_sha256": definition_sha256,
        }
    bundle = model.get("uncertainty_v2")
    if not isinstance(bundle, Mapping):
        return {
            "status": "unavailable",
            "reason": "model artifact has no uncertainty_v2 bundle",
            "frozen_definition_sha256": definition_sha256,
            "coverage_grid": frozen_grid,
        }
    state = bundle.get("final_score_state")
    oof = bundle.get("oof_evidence")
    abstention = bundle.get("abstention_policy")
    state_without_hash: dict[str, Any] = dict(state) if isinstance(state, Mapping) else {}
    stored_state_sha256 = state_without_hash.pop("state_sha256", None)
    try:
        calculated_state_sha256 = _uncertainty_canonical_sha256(
            state_without_hash
        )
    except (OverflowError, TypeError, ValueError) as exc:
        return {
            "status": "failed_integrity",
            "reason": (
                "model uncertainty v2 final score state is not canonical JSON: "
                f"{exc}"
            ),
            "frozen_definition_sha256": definition_sha256,
            "model_definition_sha256": bundle.get("definition_sha256"),
            "coverage_grid": frozen_grid,
        }
    checks = {
        "bundle_schema_version_is_v2": bundle.get("schema_version") == 2,
        "bundle_status_is_score_only": bundle.get("status")
        == "scored_no_threshold",
        "bundle_definition_matches_frozen": bundle.get("definition_sha256")
        == definition_sha256,
        "bundle_threshold_is_unselected": bundle.get("threshold") is None,
        "bundle_abstention_is_unselected": isinstance(abstention, Mapping)
        and abstention.get("status") == "not_selected"
        and abstention.get("threshold") is None
        and abstention.get("quality_band") is None,
        "final_score_state_present": isinstance(state, Mapping),
        "final_score_state_status_is_score_only": isinstance(state, Mapping)
        and state.get("status") == "scored_no_threshold",
        "final_score_state_definition_matches_frozen": isinstance(state, Mapping)
        and state.get("definition_sha256") == definition_sha256,
        "final_score_state_uses_all_training_blocks": isinstance(state, Mapping)
        and state.get("fit_scope") == "all_training_motion_blocks",
        "final_score_state_threshold_is_unselected": isinstance(state, Mapping)
        and state.get("threshold") is None
        and state.get("abstention_status") == "not_selected",
        "final_score_state_hash_is_valid": isinstance(stored_state_sha256, str)
        and stored_state_sha256
        == calculated_state_sha256,
        "oof_evidence_present": isinstance(oof, Mapping),
        "oof_definition_matches_frozen": isinstance(oof, Mapping)
        and oof.get("definition_sha256") == definition_sha256,
        "oof_coverage_grid_matches_frozen": isinstance(oof, Mapping)
        and oof.get("coverage_grid") == frozen_grid,
        "oof_threshold_is_unselected": isinstance(oof, Mapping)
        and oof.get("threshold_selected") is False
        and oof.get("threshold") is None,
        "fresh_matched_contract_capture_required": isinstance(oof, Mapping)
        and oof.get("fresh_matched_contract_capture_required") is True,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "status": "passed" if not failed else "failed_integrity",
        "reason": (
            "model uncertainty v2 is bound to the frozen definition"
            if not failed
            else "model uncertainty v2 binding failed: " + ", ".join(failed)
        ),
        "frozen_definition_sha256": definition_sha256,
        "model_definition_sha256": bundle.get("definition_sha256"),
        "final_score_state_sha256": stored_state_sha256,
        "coverage_grid": frozen_grid,
        "checks": checks,
    }


def _receipt_phase_rows(
    validation: Mapping[str, Any],
    *,
    phase: str,
    collection: Mapping[str, Any],
    model_artifact_file_sha256: str,
    viewport_width: float,
    viewport_height: float,
) -> dict[str, Any]:
    """Verify one server receipt summary and recover only bound score/error rows."""

    try:
        if validation.get("prediction_receipt_status") != "verified" or (
            validation.get("prediction_receipts_verified") is not True
        ):
            return {
                "status": "not_evaluable_receipts_unavailable",
                "integrity_status": "not_applicable",
                "reason": f"{phase} validation has no verified prediction receipts",
            }
        samples = validation.get("samples")
        observations = validation.get("uncertainty_observations")
        summary = validation.get("uncertainty_summary")
        bundle = validation.get("prediction_receipt_bundle")
        capture_contract = validation.get("capture_contract")
        if not isinstance(samples, list) or not isinstance(observations, list):
            raise MeasurementCeilingError(
                f"{phase} receipt samples and uncertainty observations must be arrays"
            )
        expected_count = FIXED_TARGET_CLUSTER_COUNT * FIXED_TARGET_REPEATS_PER_PHASE
        if len(samples) != expected_count or len(observations) != expected_count:
            raise MeasurementCeilingError(
                f"{phase} receipt evidence must contain exactly {expected_count} rows"
            )
        if not isinstance(summary, Mapping) or not isinstance(bundle, Mapping):
            raise MeasurementCeilingError(
                f"{phase} receipt bundle or uncertainty summary is unavailable"
            )
        if not isinstance(capture_contract, Mapping):
            raise MeasurementCeilingError(
                f"{phase} verified receipt capture contract is unavailable"
            )

        model_sha256 = _lower_hex_sha256(
            validation.get("model_artifact_sha256"),
            field=f"{phase}.model_artifact_sha256",
        )
        if model_sha256 != model_artifact_file_sha256:
            raise MeasurementCeilingError(
                f"{phase} receipt model artifact SHA-256 does not match input model"
            )
        if collection.get("model_artifact_sha256") != model_sha256:
            raise MeasurementCeilingError(
                f"{phase} receipt model artifact does not match the frozen assessment"
            )
        assessment_viewport = validation.get("assessment_viewport")
        expected_viewport = {
            "width_px": int(viewport_width),
            "height_px": int(viewport_height),
        }
        if assessment_viewport != expected_viewport or (
            collection.get("assessment_viewport") != expected_viewport
        ):
            raise MeasurementCeilingError(
                f"{phase} receipt viewport does not match the frozen assessment"
            )

        measurement_sha256 = _lower_hex_sha256(
            validation.get("gaze_measurement_contract_sha256"),
            field=f"{phase}.gaze_measurement_contract_sha256",
        )
        measurement_snapshot = collection.get("gaze_measurement_contract")
        if not isinstance(measurement_snapshot, Mapping) or not isinstance(
            measurement_snapshot.get("contract"), Mapping
        ):
            raise MeasurementCeilingError(
                f"{phase} frozen gaze measurement contract is unavailable"
            )
        measurement_contract = measurement_snapshot["contract"]
        if (
            measurement_snapshot.get("sha256") != measurement_sha256
            or _uncertainty_canonical_sha256(measurement_contract)
            != measurement_sha256
            or measurement_snapshot.get("contract_id")
            != measurement_contract.get("contract_id")
            or measurement_snapshot.get("contract_version")
            != measurement_contract.get("contract_version")
        ):
            raise MeasurementCeilingError(
                f"{phase} gaze measurement contract hash mismatch"
            )
        target_independence_contract = measurement_contract.get(
            "target_independence"
        )
        raw_frozen_targets = (
            target_independence_contract.get("selected_validation_targets")
            if isinstance(target_independence_contract, Mapping)
            else None
        )
        if not isinstance(raw_frozen_targets, list) or len(raw_frozen_targets) != (
            FIXED_TARGET_CLUSTER_COUNT
        ):
            raise MeasurementCeilingError(
                f"{phase} frozen five-target validation contract is unavailable"
            )
        frozen_targets: list[dict[str, Any]] = []
        for target_index, raw_target in enumerate(raw_frozen_targets):
            if not isinstance(raw_target, Mapping):
                raise MeasurementCeilingError(
                    f"{phase} frozen target {target_index} is invalid"
                )
            target_id = str(raw_target.get("target_id") or "").strip()
            fraction_x = _finite(
                raw_target.get("target_x_viewport_fraction"),
                field=f"{phase}.frozen_target_x_viewport_fraction",
            )
            fraction_y = _finite(
                raw_target.get("target_y_viewport_fraction"),
                field=f"{phase}.frozen_target_y_viewport_fraction",
            )
            target_x_norm = _signed_normalized(
                raw_target.get("target_x_norm"),
                field=f"{phase}.frozen_target_x_norm",
            )
            target_y_norm = _signed_normalized(
                raw_target.get("target_y_norm"),
                field=f"{phase}.frozen_target_y_norm",
            )
            if (
                not target_id
                or not 0.0 <= fraction_x <= 1.0
                or not 0.0 <= fraction_y <= 1.0
                or not math.isclose(
                    target_x_norm, 2.0 * fraction_x - 1.0, abs_tol=1e-9
                )
                or not math.isclose(
                    target_y_norm, 2.0 * fraction_y - 1.0, abs_tol=1e-9
                )
            ):
                raise MeasurementCeilingError(
                    f"{phase} frozen target {target_index} coordinates are invalid"
                )
            frozen_targets.append(
                {
                    "target_id": target_id,
                    "target_x_norm": target_x_norm,
                    "target_y_norm": target_y_norm,
                    "target_x_px": float(
                        math.floor(fraction_x * viewport_width + 0.5)
                    ),
                    "target_y_px": float(
                        math.floor(fraction_y * viewport_height + 0.5)
                    ),
                }
            )
        if len({target["target_id"] for target in frozen_targets}) != (
            FIXED_TARGET_CLUSTER_COUNT
        ):
            raise MeasurementCeilingError(
                f"{phase} frozen validation target IDs are not unique"
            )
        if validation.get("samples_sha256") != _uncertainty_canonical_sha256(
            samples
        ):
            raise MeasurementCeilingError(f"{phase} validation sample hash mismatch")

        record_sha256s = bundle.get("receipt_record_sha256s")
        if (
            type(bundle.get("schema_version")) is not int
            or bundle.get("schema_version") != 1
            or bundle.get("status") != "verified"
            or bundle.get("phase") != phase
            or type(bundle.get("count")) is not int
            or bundle.get("count") != expected_count
            or not isinstance(record_sha256s, list)
            or len(record_sha256s) != expected_count
            or len(set(record_sha256s)) != expected_count
        ):
            raise MeasurementCeilingError(f"{phase} prediction receipt bundle is invalid")
        normalized_record_sha256s = [
            _lower_hex_sha256(value, field=f"{phase}.receipt_record_sha256")
            for value in record_sha256s
        ]
        bundle_core = {
            "schema_version": 1,
            "status": "verified",
            "phase": phase,
            "count": expected_count,
            "receipt_record_sha256s": normalized_record_sha256s,
        }
        if bundle.get("bundle_sha256") != _uncertainty_canonical_sha256(
            bundle_core
        ):
            raise MeasurementCeilingError(f"{phase} prediction receipt bundle hash mismatch")

        observation_hashes = [
            _uncertainty_canonical_sha256(observation)
            for observation in observations
        ]
        expected_summary = {
            "schema_version": 1,
            "status": "verified",
            "count": expected_count,
            "scored_count": sum(
                isinstance(observation, Mapping)
                and isinstance(observation.get("uncertainty"), Mapping)
                and observation["uncertainty"].get("status")
                == "scored_no_threshold"
                for observation in observations
            ),
            "unavailable_count": sum(
                not (
                    isinstance(observation, Mapping)
                    and isinstance(observation.get("uncertainty"), Mapping)
                    and observation["uncertainty"].get("status")
                    == "scored_no_threshold"
                )
                for observation in observations
            ),
            "observation_sha256s": observation_hashes,
            "observations_sha256": _uncertainty_canonical_sha256(observations),
        }
        if _uncertainty_canonical_json_bytes(summary) != (
            _uncertainty_canonical_json_bytes(expected_summary)
        ):
            raise MeasurementCeilingError(
                f"{phase} uncertainty observation/list hash summary mismatch"
            )

        payload = {
            "samples": samples,
            "capture_contract": capture_contract,
            "prediction_receipt_bundle": bundle,
            "uncertainty_observations": observations,
            "uncertainty_summary": summary,
            "prediction_receipt_status": "verified",
            "prediction_receipts_verified": True,
            "model_artifact_sha256": model_sha256,
            "gaze_measurement_contract_sha256": measurement_sha256,
            "assessment_viewport": expected_viewport,
        }
        if validation.get("validation_payload_sha256") != (
            _uncertainty_canonical_sha256(payload)
        ):
            raise MeasurementCeilingError(f"{phase} validation payload hash mismatch")

        rows: list[dict[str, Any]] = []
        target_order = [str(target["target_id"]) for target in frozen_targets]
        successful_unavailable_count = 0
        no_face_count = 0
        exact_observation_fields = {
            "schema_version",
            "receipt_record_sha256",
            "phase",
            "receipt_ordinal",
            "target_id",
            "target_repeat_index",
            "prediction_success",
            "uncertainty",
        }
        for ordinal, (sample, observation) in enumerate(
            zip(samples, observations, strict=True)
        ):
            if not isinstance(sample, Mapping) or not isinstance(observation, Mapping):
                raise MeasurementCeilingError(
                    f"{phase} receipt row {ordinal} must contain objects"
                )
            if set(observation) != exact_observation_fields:
                raise MeasurementCeilingError(
                    f"{phase} uncertainty observation {ordinal} fields changed"
                )
            target_id = str(sample.get("target_id") or "").strip()
            prediction_success = sample.get("prediction_success")
            if not target_id or type(prediction_success) is not bool:
                raise MeasurementCeilingError(
                    f"{phase} validation sample {ordinal} outcome is invalid"
                )
            frozen_target = frozen_targets[
                ordinal // FIXED_TARGET_REPEATS_PER_PHASE
            ]
            sample_target_x_norm = _signed_normalized(
                sample.get("target_x_norm"), field=f"{phase}.target_x_norm"
            )
            sample_target_y_norm = _signed_normalized(
                sample.get("target_y_norm"), field=f"{phase}.target_y_norm"
            )
            if (
                type(observation.get("schema_version")) is not int
                or observation.get("schema_version") != 1
                or observation.get("receipt_record_sha256")
                != normalized_record_sha256s[ordinal]
                or observation.get("phase") != phase
                or type(observation.get("receipt_ordinal")) is not int
                or observation.get("receipt_ordinal") != ordinal
                or observation.get("target_id") != target_id
                or type(observation.get("target_repeat_index")) is not int
                or observation.get("target_repeat_index")
                != ordinal % FIXED_TARGET_REPEATS_PER_PHASE
                or observation.get("prediction_success") is not prediction_success
                or target_id != frozen_target["target_id"]
            ):
                raise MeasurementCeilingError(
                    f"{phase} uncertainty observation {ordinal} receipt binding mismatch"
                )
            raw_uncertainty = observation.get("uncertainty")
            try:
                normalized_uncertainty = normalize_uncertainty_observation(
                    raw_uncertainty,
                    viewport=(viewport_width, viewport_height),
                )
            except (OSError, TypeError, ValueError) as exc:
                raise MeasurementCeilingError(
                    f"{phase} uncertainty observation {ordinal} is invalid: {exc}"
                ) from exc
            if _uncertainty_canonical_json_bytes(raw_uncertainty) != (
                _uncertainty_canonical_json_bytes(normalized_uncertainty)
            ):
                raise MeasurementCeilingError(
                    f"{phase} uncertainty observation {ordinal} is not canonical"
                )

            target_x = _finite(
                sample.get("target_x_px"), field=f"{phase}.target_x_px"
            )
            target_y = _finite(
                sample.get("target_y_px"), field=f"{phase}.target_y_px"
            )
            if (
                not math.isclose(
                    target_x,
                    float(frozen_target["target_x_px"]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    target_y,
                    float(frozen_target["target_y_px"]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    sample_target_x_norm,
                    float(frozen_target["target_x_norm"]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    sample_target_y_norm,
                    float(frozen_target["target_y_norm"]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise MeasurementCeilingError(
                    f"{phase} validation sample {ordinal} target is not frozen"
                )
            score: float | None = None
            spatial_error: float | None = None
            if prediction_success:
                predicted_x = _finite(
                    sample.get("predicted_x_px"),
                    field=f"{phase}.predicted_x_px",
                )
                predicted_y = _finite(
                    sample.get("predicted_y_px"),
                    field=f"{phase}.predicted_y_px",
                )
                spatial_error = math.hypot(
                    predicted_x - target_x,
                    predicted_y - target_y,
                )
                stored_error = _finite(
                    sample.get("spatial_error_px"),
                    field=f"{phase}.spatial_error_px",
                )
                if not math.isclose(
                    stored_error, spatial_error, rel_tol=0.0, abs_tol=1e-9
                ):
                    raise MeasurementCeilingError(
                        f"{phase} validation sample {ordinal} error mismatch"
                    )
                if normalized_uncertainty.get("status") != "scored_no_threshold":
                    successful_unavailable_count += 1
                else:
                    score = float(normalized_uncertainty["score"])
            else:
                no_face_count += 1
                if normalized_uncertainty.get("status") != (
                    "unavailable_sensor_failure"
                ):
                    raise MeasurementCeilingError(
                        f"{phase} no-face observation {ordinal} is not sensor-unavailable"
                    )
                if any(
                    sample.get(field) is not None
                    for field in (
                        "predicted_x_px",
                        "predicted_y_px",
                        "spatial_error_px",
                    )
                ):
                    raise MeasurementCeilingError(
                        f"{phase} failed prediction {ordinal} contains coordinates"
                    )
            rows.append(
                {
                    "sample_id": (
                        f"{phase}:{ordinal:02d}:"
                        f"{normalized_record_sha256s[ordinal]}"
                    ),
                    "phase": phase,
                    "receipt_ordinal": ordinal,
                    "receipt_record_sha256": normalized_record_sha256s[ordinal],
                    "target_id": target_id,
                    "prediction_success": prediction_success,
                    "uncertainty_score": score,
                    "spatial_error_px": spatial_error,
                }
            )
        for start in range(0, expected_count, FIXED_TARGET_REPEATS_PER_PHASE):
            chunk = rows[start : start + FIXED_TARGET_REPEATS_PER_PHASE]
            if len({row["target_id"] for row in chunk}) != 1:
                raise MeasurementCeilingError(
                    f"{phase} target repeats are not contiguous receipt clusters"
                )
        return {
            "status": (
                "not_evaluable_successful_uncertainty_unavailable"
                if successful_unavailable_count
                else "verified_scored"
            ),
            "integrity_status": "passed",
            "reason": (
                f"{successful_unavailable_count} successful predictions lack a "
                "scored frozen-v2 uncertainty observation"
                if successful_unavailable_count
                else "all successful predictions have receipt-bound frozen-v2 scores"
            ),
            "receipt_integrity": {
                "status": "passed",
                "prediction_receipt_bundle_sha256": bundle["bundle_sha256"],
                "uncertainty_observations_sha256": summary[
                    "observations_sha256"
                ],
                "validation_payload_sha256": validation[
                    "validation_payload_sha256"
                ],
                "model_artifact_sha256": model_sha256,
                "gaze_measurement_contract_sha256": measurement_sha256,
            },
            "attempted_count": expected_count,
            "successful_count": expected_count - no_face_count,
            "no_face_count": no_face_count,
            "successful_uncertainty_unavailable_count": (
                successful_unavailable_count
            ),
            "target_order": target_order,
            "rows": rows,
        }
    except (
        MeasurementCeilingError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        return {
            "status": "failed_integrity",
            "integrity_status": "failed",
            "reason": str(exc),
        }


def _fixed_uncertainty_coverage_scope(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    target_order: Sequence[str],
) -> dict[str, Any]:
    """Build frozen score-ranked risk while treating five targets as clusters."""

    normalized_targets = [str(value) for value in target_order]
    if (
        len(normalized_targets) != FIXED_TARGET_CLUSTER_COUNT
        or len(set(normalized_targets)) != FIXED_TARGET_CLUSTER_COUNT
    ):
        raise MeasurementCeilingError(
            "uncertainty coverage-risk requires exactly five target clusters"
        )
    attempted_rows = [dict(row) for row in rows]
    successful = [
        row
        for row in attempted_rows
        if row.get("prediction_success") is True
    ]
    if not successful:
        raise MeasurementCeilingError(
            "uncertainty coverage-risk has no successful fixed-target predictions"
        )
    for row in successful:
        if row.get("uncertainty_score") is None or row.get("spatial_error_px") is None:
            raise MeasurementCeilingError(
                "successful fixed-target prediction lacks score or held-out error"
            )
    ordered = sorted(
        successful,
        key=lambda row: (
            float(row["uncertainty_score"]),
            str(row["sample_id"]),
        ),
    )
    ordered_ids = [str(row["sample_id"]) for row in ordered]
    curve: list[dict[str, Any]] = []
    for coverage in UNCERTAINTY_V2_COVERAGE_LEVELS:
        retained_count = min(
            len(ordered),
            max(1, int(round(coverage * len(ordered)))),
        )
        retained_ids = set(ordered_ids[:retained_count])
        retained = [
            row for row in successful if str(row["sample_id"]) in retained_ids
        ]
        retained_errors = [float(row["spatial_error_px"]) for row in retained]
        per_target: dict[str, dict[str, Any]] = {}
        zero_coverage_targets: list[str] = []
        for target_id in normalized_targets:
            target_attempts = [
                row for row in attempted_rows if row["target_id"] == target_id
            ]
            target_successes = [
                row for row in target_attempts if row["prediction_success"] is True
            ]
            target_retained = [
                row
                for row in target_successes
                if str(row["sample_id"]) in retained_ids
            ]
            target_errors = [
                float(row["spatial_error_px"]) for row in target_retained
            ]
            if not target_retained:
                zero_coverage_targets.append(target_id)
            per_target[target_id] = {
                "attempted_count": len(target_attempts),
                "successful_count": len(target_successes),
                "retained_count": len(target_retained),
                "capture_success_fraction": (
                    len(target_successes) / len(target_attempts)
                    if target_attempts
                    else None
                ),
                "score_coverage_within_successful": (
                    len(target_retained) / len(target_successes)
                    if target_successes
                    else None
                ),
                "end_to_end_retained_fraction": (
                    len(target_retained) / len(target_attempts)
                    if target_attempts
                    else None
                ),
                "spatial_error_px": (
                    {
                        "mean": statistics.fmean(target_errors),
                        "median": statistics.median(target_errors),
                        "p90": _nearest_rank(target_errors, 0.90),
                    }
                    if target_errors
                    else {"mean": None, "median": None, "p90": None}
                ),
            }
        nonempty_target_means = [
            float(target["spatial_error_px"]["mean"])
            for target in per_target.values()
            if target["spatial_error_px"]["mean"] is not None
        ]
        all_cluster_macro = (
            statistics.fmean(nonempty_target_means)
            if not zero_coverage_targets
            else None
        )
        curve.append(
            {
                "requested_score_coverage": coverage,
                "retained_successful_count": retained_count,
                "successful_prediction_count": len(successful),
                "attempted_capture_count": len(attempted_rows),
                "achieved_score_coverage_within_successful": (
                    retained_count / len(successful)
                ),
                "achieved_end_to_end_attempt_coverage": (
                    retained_count / len(attempted_rows)
                ),
                "retained_sample_ids": ordered_ids[:retained_count],
                "overall_retained_row_risk_px": {
                    "availability": "descriptive_rows_not_iid",
                    "mean": statistics.fmean(retained_errors),
                    "median": statistics.median(retained_errors),
                    "p90": _nearest_rank(retained_errors, 0.90),
                },
                "target_clusters_with_zero_coverage": zero_coverage_targets,
                "target_cluster_macro_all_clusters": {
                    "availability": (
                        "all_five_clusters_have_nonzero_coverage"
                        if not zero_coverage_targets
                        else "unavailable_due_to_zero_coverage_clusters"
                    ),
                    "mean_spatial_error_px": all_cluster_macro,
                },
                "target_cluster_macro_nonempty_clusters": {
                    "availability": "descriptive_nonempty_clusters_only",
                    "nonempty_target_cluster_count": len(nonempty_target_means),
                    "mean_spatial_error_px": statistics.fmean(
                        nonempty_target_means
                    ),
                },
                "worst_target_cluster_mean_spatial_error_px": (
                    max(nonempty_target_means)
                    if not zero_coverage_targets
                    else None
                ),
                "worst_nonempty_target_cluster_mean_spatial_error_px": max(
                    nonempty_target_means
                ),
                "per_target_cluster": per_target,
            }
        )
    # Build the hypothetical flags from the already frozen curve. Keeping these
    # flags beside raw rows preserves both views without selecting a threshold.
    curve_retained = {
        f"{point['requested_score_coverage']:.1f}": set(
            point["retained_sample_ids"]
        )
        for point in curve
    }
    output_rows = [
        {
            **row,
            "would_abstain_at_fixed_coverage": {
                key: (
                    None
                    if row.get("prediction_success") is not True
                    else str(row["sample_id"]) not in retained_ids
                )
                for key, retained_ids in curve_retained.items()
            },
        }
        for row in attempted_rows
    ]
    return {
        "status": "evaluable_descriptive_heldout",
        "scope": scope,
        "claim_boundary": (
            "receipt_verified_training_only_score_ranked_heldout_fixed_target_"
            "development_evidence"
        ),
        "score_source": (
            "server_runtime_training_only_uncertainty_observation; no target, "
            "residual, or held-out error enters score or ordering"
        ),
        "risk_source": "receipt_verified_heldout_fixed_target_spatial_error_px",
        "fixed_requested_coverages": list(UNCERTAINTY_V2_COVERAGE_LEVELS),
        "threshold_selected": False,
        "threshold": None,
        "quality_band_change_authorized": False,
        "abstention_policy_authorized": False,
        "attempted_capture_count": len(attempted_rows),
        "successful_prediction_count": len(successful),
        "no_face_count": len(attempted_rows) - len(successful),
        "capture_success_fraction": len(successful) / len(attempted_rows),
        "independent_target_cluster_count": FIXED_TARGET_CLUSTER_COUNT,
        "sample_rows_are_independent_units": False,
        "inferential_claim_authorized": False,
        "limitation": (
            "only five predeclared target clusters are observed; repeated rows and "
            "start/end phases are not treated as 15 or 30 independent units"
        ),
        "ordered_sample_ids_low_to_high_training_only_score": ordered_ids,
        "rows": output_rows,
        "coverage_risk_curve": curve,
    }


def _heldout_uncertainty_coverage_risk(
    model: Mapping[str, Any],
    validations: Mapping[str, Any],
    collection: Mapping[str, Any],
    *,
    model_artifact_file_sha256: str,
    viewport_width: float,
    viewport_height: float,
) -> dict[str, Any]:
    model_binding = _uncertainty_model_binding(model)
    if model_binding["status"] == "unavailable":
        return {
            "status": "not_evaluable",
            "integrity_status": "not_applicable",
            "reason": model_binding["reason"],
            "model_binding": model_binding,
            "threshold_selected": False,
            "quality_band_change_authorized": False,
        }
    if model_binding["status"] != "passed":
        return {
            "status": "not_evaluable_integrity_failure",
            "integrity_status": "failed",
            "reason": model_binding["reason"],
            "model_binding": model_binding,
            "threshold_selected": False,
            "quality_band_change_authorized": False,
        }

    phases: dict[str, dict[str, Any]] = {}
    for phase in ("start", "end"):
        validation = validations.get(phase)
        if not isinstance(validation, Mapping):
            phases[phase] = {
                "status": "not_evaluable_receipts_unavailable",
                "integrity_status": "not_applicable",
                "reason": f"{phase} validation is unavailable",
            }
            continue
        verified = _receipt_phase_rows(
            validation,
            phase=phase,
            collection=collection,
            model_artifact_file_sha256=model_artifact_file_sha256,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
        )
        if verified.get("status") == "verified_scored":
            coverage = _fixed_uncertainty_coverage_scope(
                verified["rows"],
                scope=phase,
                target_order=verified["target_order"],
            )
            coverage["receipt_integrity"] = verified["receipt_integrity"]
            phases[phase] = coverage
        else:
            phases[phase] = verified

    if any(phase.get("integrity_status") == "failed" for phase in phases.values()):
        return {
            "status": "not_evaluable_integrity_failure",
            "integrity_status": "failed",
            "reason": "at least one fixed-target receipt phase failed integrity",
            "model_binding": model_binding,
            "phases": phases,
            "threshold_selected": False,
            "quality_band_change_authorized": False,
        }
    if any(
        phase.get("status") != "evaluable_descriptive_heldout"
        for phase in phases.values()
    ):
        return {
            "status": "not_evaluable",
            "integrity_status": "passed",
            "reason": (
                "both receipt-verified phases require scored uncertainty for every "
                "successful prediction"
            ),
            "model_binding": model_binding,
            "phases": phases,
            "threshold_selected": False,
            "quality_band_change_authorized": False,
        }
    if phases["start"]["independent_target_cluster_count"] != (
        phases["end"]["independent_target_cluster_count"]
    ):
        return {
            "status": "not_evaluable_integrity_failure",
            "integrity_status": "failed",
            "reason": "start/end uncertainty target cluster counts differ",
            "model_binding": model_binding,
            "phases": phases,
            "threshold_selected": False,
            "quality_band_change_authorized": False,
        }
    start_target_order = [
        target_id
        for target_id in phases["start"]["coverage_risk_curve"][0][
            "per_target_cluster"
        ]
    ]
    end_target_order = [
        target_id
        for target_id in phases["end"]["coverage_risk_curve"][0][
            "per_target_cluster"
        ]
    ]
    if start_target_order != end_target_order:
        return {
            "status": "not_evaluable_integrity_failure",
            "integrity_status": "failed",
            "reason": "start/end uncertainty target IDs or order differ",
            "model_binding": model_binding,
            "phases": phases,
            "threshold_selected": False,
            "quality_band_change_authorized": False,
        }
    combined_rows = [
        dict(row)
        for phase in ("start", "end")
        for row in phases[phase]["rows"]
    ]
    combined = _fixed_uncertainty_coverage_scope(
        combined_rows,
        scope="combined_start_end_repeated_measurements",
        target_order=start_target_order,
    )
    return {
        "status": "evaluable_descriptive_heldout",
        "integrity_status": "passed",
        "definition_sha256": model_binding["frozen_definition_sha256"],
        "model_binding": model_binding,
        "fixed_requested_coverages": list(UNCERTAINTY_V2_COVERAGE_LEVELS),
        "threshold_selected": False,
        "threshold": None,
        "quality_band_change_authorized": False,
        "abstention_policy_authorized": False,
        "phases": phases,
        "combined": combined,
        "decision_boundary": (
            "descriptive fresh fixed-target evidence only; no threshold, quality "
            "band, production, line, word, or population claim"
        ),
    }


def _future_uncertainty_v2_requirements(
    model: Mapping[str, Any],
    calibration_records: Sequence[Mapping[str, Any]],
    validations: Mapping[str, Any],
    heldout_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    calibration_fields = sorted(
        {str(field) for record in calibration_records for field in record}
    )
    model_oof_fields: list[str] = []
    uncertainty_bundle = model.get("uncertainty_v2")
    if isinstance(uncertainty_bundle, Mapping):
        for field in (
            "definition_sha256",
            "final_score_state",
            "grid_validation",
            "oof_evidence",
        ):
            if field in uncertainty_bundle:
                model_oof_fields.append(f"uncertainty_v2.{field}")
    validation_uncertainty_fields = sorted(
        {
            f"{phase}.{field}"
            for phase in ("start", "end")
            for field in (
                "prediction_receipt_bundle",
                "uncertainty_observations",
                "uncertainty_summary",
                "validation_payload_sha256",
            )
            if isinstance(validations.get(phase), Mapping)
            and field in validations[phase]
        }
    )
    calibration_sensor_fields = sorted(
        set(calibration_fields)
        & {
            "raw_gaze_pitch_yaw",
            "gaze_pitch_yaw",
            "gaze_embedding",
            "sensor_features",
        }
    )
    constructable = heldout_evaluation.get("status") == (
        "evaluable_descriptive_heldout"
    )
    return {
        "status": (
            "fulfilled_for_receipt_verified_descriptive_fixed_target_evaluation"
            if constructable
            else "required_before_predictive_uncertainty_claim"
        ),
        "current_evidence_inventory": {
            "model_oof_or_uncertainty_fields_present": model_oof_fields,
            "validation_uncertainty_fields_present": validation_uncertainty_fields,
            "calibration_reconstructable_sensor_fields_present": (
                calibration_sensor_fields
            ),
            "predictive_uncertainty_curve_constructable_from_current_artifacts": (
                constructable
            ),
            "heldout_evaluation_status": heldout_evaluation.get("status"),
            "reason": heldout_evaluation.get(
                "reason",
                (
                    "receipt-verified frozen-v2 runtime scores and held-out target "
                    "errors are available"
                    if constructable
                    else "required frozen-v2 receipt evidence is unavailable"
                ),
            ),
        },
        "protocol_requirement": {
            "new_version": "v2",
            "freeze_before_new_untouched_capture": True,
            "v1_result_may_not_select_definition_or_threshold": True,
        },
        "required_per_oof_sample_fields": [
            "sample_id",
            "outer_fold_id",
            "outer_holdout_group_id",
            "target_id",
            "oof_predicted_x_px",
            "oof_predicted_y_px",
            "oof_residual_x_px",
            "oof_residual_y_px",
            "oof_spatial_error_px",
            "training_only_ood_score",
            "training_only_leverage_score",
            "training_only_prediction_covariance_px",
        ],
        "required_definition_binding": [
            "uncertainty_definition_id",
            "uncertainty_definition_version",
            "uncertainty_definition_sha256",
            "training_partition_only_fit_proof",
            "coverage_grid",
            "frozen_abstention_thresholds_or_explicit_none",
        ],
        "required_evaluation": {
            "untouched_capture_required": True,
            "score_must_be_computed_without_holdout_target_error": True,
            "coverage_risk_unit": "predeclared_independent_cluster",
            "report_raw_and_abstained_predictions": True,
            "participant_capture_session_and_device_confirmation_axes_required": True,
        },
        "required_fixed_target_receipt_fields": [
            "prediction_receipt_bundle",
            "uncertainty_observations",
            "uncertainty_summary",
            "validation_payload_sha256",
            "model_artifact_sha256",
            "gaze_measurement_contract_sha256",
            "assessment_viewport",
        ],
    }


def _target_independence(
    calibration_records: Sequence[Mapping[str, Any]],
    evaluation_targets: Mapping[str, tuple[float, float]],
    *,
    viewport_width: float,
    viewport_height: float,
    tolerance: float,
) -> dict[str, Any]:
    calibration_points: set[tuple[float, float]] = set()
    for index, record in enumerate(calibration_records):
        if record.get("ok") is not True:
            continue
        calibration_points.add(
            (
                _signed_normalized(
                    record.get("target_x_norm"),
                    field=f"calibration[{index}].target_x_norm",
                ),
                _signed_normalized(
                    record.get("target_y_norm"),
                    field=f"calibration[{index}].target_y_norm",
                ),
            )
        )
    if not calibration_points:
        raise MeasurementCeilingError("calibration manifest has no usable targets")

    evaluation_norm: dict[str, tuple[float, float]] = {}
    for target_id, target in evaluation_targets.items():
        if not 0.0 <= target[0] <= viewport_width:
            raise MeasurementCeilingError(
                f"evaluation target {target_id} x is outside the viewport"
            )
        if not 0.0 <= target[1] <= viewport_height:
            raise MeasurementCeilingError(
                f"evaluation target {target_id} y is outside the viewport"
            )
        evaluation_norm[target_id] = (
            target[0] / viewport_width * 2.0 - 1.0,
            target[1] / viewport_height * 2.0 - 1.0,
        )
    overlaps: list[str] = []
    target_distances: dict[str, dict[str, float]] = {}
    minimum_distance = float("inf")
    for target_id, evaluation_point in evaluation_norm.items():
        distance = min(
            math.hypot(
                evaluation_point[0] - calibration_point[0],
                evaluation_point[1] - calibration_point[1],
            )
            for calibration_point in calibration_points
        )
        minimum_distance = min(minimum_distance, distance)
        target_distances[target_id] = {
            "signed_normalized_euclidean": distance,
            "viewport_fraction_euclidean": distance / 2.0,
        }
        if distance < tolerance and not math.isclose(
            distance,
            tolerance,
            rel_tol=0.0,
            abs_tol=COORDINATE_ABS_TOLERANCE,
        ):
            overlaps.append(target_id)
    frozen_threshold_match = math.isclose(
        tolerance,
        DEFAULT_TARGET_OVERLAP_TOLERANCE_SIGNED,
        rel_tol=0.0,
        abs_tol=COORDINATE_ABS_TOLERANCE,
    )
    return {
        "status": (
            "passed" if not overlaps and frozen_threshold_match else "failed"
        ),
        "definition": (
            "evaluation targets must be at least the configured Euclidean separation "
            "from every calibration target"
        ),
        "coordinate_range": "signed normalized screen coordinates [-1, 1]",
        "coordinate_transform": "signed = 2 * viewport_fraction - 1",
        "distance_metric": "two-dimensional Euclidean distance",
        "independence_rule": "distance >= tolerance",
        "overlap_rule": "distance < tolerance",
        "frozen_protocol_threshold_match": frozen_threshold_match,
        "threshold_source": (
            "frozen_protocol_default"
            if frozen_threshold_match
            else "explicit_non_protocol_override"
        ),
        "normalized_coordinate_tolerance": tolerance,
        "signed_normalized_tolerance": tolerance,
        "viewport_fraction_tolerance_equivalent": tolerance / 2.0,
        "frozen_protocol_viewport_fraction_separation": (
            FROZEN_PROTOCOL_TARGET_SEPARATION_VIEWPORT_FRACTION
        ),
        "calibration_target_count": len(calibration_points),
        "evaluation_target_count": len(evaluation_norm),
        "overlap_count": len(overlaps),
        "overlapping_evaluation_target_ids": sorted(overlaps),
        "minimum_normalized_target_distance": minimum_distance,
        "minimum_signed_normalized_euclidean_distance": minimum_distance,
        "minimum_viewport_fraction_euclidean_distance": minimum_distance / 2.0,
        "evaluation_target_minimum_distances": {
            target_id: target_distances[target_id]
            for target_id in sorted(target_distances)
        },
    }


def _drift_vectors(
    start_metrics: Mapping[str, Any],
    end_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    start_targets = start_metrics["targets"]
    end_targets = end_metrics["targets"]
    if set(start_targets) != set(end_targets):
        return {
            "status": "not_evaluable",
            "reason": "start and end validation target sets differ",
        }
    vectors: dict[str, dict[str, float]] = {}
    magnitudes: list[float] = []
    for target_id in sorted(start_targets):
        start = start_targets[target_id]
        end = end_targets[target_id]
        delta_x = (
            float(end["predicted_centroid_x_px"])
            - float(start["predicted_centroid_x_px"])
        )
        delta_y = (
            float(end["predicted_centroid_y_px"])
            - float(start["predicted_centroid_y_px"])
        )
        magnitude = math.hypot(delta_x, delta_y)
        magnitudes.append(magnitude)
        vectors[target_id] = {
            "predicted_centroid_delta_x_px": delta_x,
            "predicted_centroid_delta_y_px": delta_y,
            "predicted_centroid_drift_magnitude_px": magnitude,
            "signed_bias_delta_x_px": (
                float(end["signed_bias_x_px"])
                - float(start["signed_bias_x_px"])
            ),
            "signed_bias_delta_y_px": (
                float(end["signed_bias_y_px"])
                - float(start["signed_bias_y_px"])
            ),
            "median_spatial_error_change_px": (
                float(end["median_spatial_error_px"])
                - float(start["median_spatial_error_px"])
            ),
        }
    return {
        "status": "evaluable",
        "target_count": len(vectors),
        "target_macro_mean_drift_magnitude_px": statistics.fmean(magnitudes),
        "target_macro_median_drift_magnitude_px": statistics.median(magnitudes),
        "targets": vectors,
    }


def _corrected_records(
    records: Sequence[Mapping[str, Any]],
    *,
    correction_x: float,
    correction_y: float,
) -> list[dict[str, Any]]:
    corrected: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        predicted_x = float(record["predicted_x_px"]) + correction_x
        predicted_y = float(record["predicted_y_px"]) + correction_y
        delta_x = predicted_x - float(record["target_x_px"])
        delta_y = predicted_y - float(record["target_y_px"])
        item.update(
            {
                "predicted_x_px": predicted_x,
                "predicted_y_px": predicted_y,
                "signed_error_x_px": delta_x,
                "signed_error_y_px": delta_y,
                "spatial_error_px": math.hypot(delta_x, delta_y),
            }
        )
        corrected.append(item)
    return corrected


def _cluster_bootstrap(
    raw_records: Sequence[Mapping[str, Any]],
    corrected_records: Sequence[Mapping[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if resamples <= 0:
        raise MeasurementCeilingError("bootstrap resamples must be positive")
    raw_target_sequence = [str(record["target_id"]) for record in raw_records]
    corrected_target_sequence = [
        str(record["target_id"]) for record in corrected_records
    ]
    if raw_target_sequence != corrected_target_sequence:
        raise MeasurementCeilingError(
            "raw and corrected records must retain paired target order"
        )
    raw_by_target: dict[str, list[float]] = defaultdict(list)
    corrected_by_target: dict[str, list[float]] = defaultdict(list)
    for record in raw_records:
        raw_by_target[str(record["target_id"])].append(
            float(record["spatial_error_px"])
        )
    for record in corrected_records:
        corrected_by_target[str(record["target_id"])].append(
            float(record["spatial_error_px"])
        )
    target_ids = sorted(raw_by_target)
    if set(target_ids) != set(corrected_by_target):
        raise MeasurementCeilingError("raw and corrected target clusters differ")
    if len(target_ids) < 2:
        raise MeasurementCeilingError(
            "cluster bootstrap requires at least two evaluation targets"
        )
    if any(
        len(raw_by_target[target_id]) != len(corrected_by_target[target_id])
        for target_id in target_ids
    ):
        raise MeasurementCeilingError(
            "raw and corrected target cluster sizes must match"
        )
    deltas: list[float] = []
    for resample_index in range(resamples):
        selected: list[str] = []
        for draw_index in range(len(target_ids)):
            digest = hashlib.sha256(
                f"{seed}:{resample_index}:{draw_index}".encode("ascii")
            ).digest()
            selected.append(
                target_ids[int.from_bytes(digest, "big") % len(target_ids)]
            )
        raw_values = [value for target in selected for value in raw_by_target[target]]
        corrected_values = [
            value for target in selected for value in corrected_by_target[target]
        ]
        deltas.append(
            statistics.median(corrected_values) - statistics.median(raw_values)
        )
    return {
        "cluster_unit": "evaluation_target_id",
        "paired_raw_and_corrected": True,
        "target_cluster_count": len(target_ids),
        "cluster_draws_per_resample": len(target_ids),
        "resamples": resamples,
        "seed": seed,
        "deterministic_sampler": "sha256(seed:resample_index:draw_index) modulo target_count",
        "delta_definition": "corrected minus raw end-validation median error",
        "interval_quantile_method": "linear interpolation at (n - 1) * p",
        "ci95_lower_px": _quantile(deltas, 0.025),
        "ci95_upper_px": _quantile(deltas, 0.975),
        "fraction_improved": sum(delta < 0 for delta in deltas) / resamples,
        "fraction_unchanged": sum(delta == 0 for delta in deltas) / resamples,
        "inferential_claim_authorized": False,
        "limitation": (
            "resamples only the observed target clusters; the interval is descriptive "
            "and does not establish population-level correction benefit"
        ),
    }


def _temporal_correction(
    start_records: Sequence[Mapping[str, Any]],
    end_records: Sequence[Mapping[str, Any]],
    *,
    end_attempted_count: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    correction_x = statistics.median(
        float(record["target_x_px"]) - float(record["predicted_x_px"])
        for record in start_records
    )
    correction_y = statistics.median(
        float(record["target_y_px"]) - float(record["predicted_y_px"])
        for record in start_records
    )
    corrected = _corrected_records(
        end_records,
        correction_x=correction_x,
        correction_y=correction_y,
    )
    raw_metrics = _phase_metrics(end_records, attempted_count=end_attempted_count)
    corrected_metrics = _phase_metrics(
        corrected,
        attempted_count=end_attempted_count,
    )
    raw_median = float(raw_metrics["spatial_error_px"]["median"])
    corrected_median = float(corrected_metrics["spatial_error_px"]["median"])
    raw_p90 = float(raw_metrics["spatial_error_px"]["p90"])
    corrected_p90 = float(corrected_metrics["spatial_error_px"]["p90"])
    bootstrap = _cluster_bootstrap(
        end_records,
        corrected,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    return {
        "correction_id": CORRECTION_ID,
        "fit_scope": "start validation successful samples only",
        "evaluation_scope": "temporally later end validation only",
        "selected_using_end_validation": False,
        "translation_x_px": correction_x,
        "translation_y_px": correction_y,
        "raw_end": {
            "median_spatial_error_px": raw_median,
            "p90_spatial_error_px": raw_p90,
        },
        "corrected_end": {
            "median_spatial_error_px": corrected_median,
            "p90_spatial_error_px": corrected_p90,
        },
        "corrected_minus_raw": {
            "median_spatial_error_px": corrected_median - raw_median,
            "p90_spatial_error_px": corrected_p90 - raw_p90,
        },
        "bootstrap": bootstrap,
        "decision": "development_only_no_model_or_quality_band_promotion",
    }


def _manifest_provenance(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    usable = [record for record in records if record.get("ok") is True]

    def distinct(field: str) -> list[str]:
        return sorted(
            {
                str(record.get(field)).strip()
                for record in usable
                if str(record.get(field) or "").strip()
            }
        )

    return {
        "row_count": len(records),
        "usable_row_count": len(usable),
        "capture_run_count": len(distinct("capture_run_id")),
        "capture_sources": distinct("capture_source"),
        "collection_protocols": distinct("collection_protocol"),
        "motion_blocks": distinct("motion_block_id"),
    }


def _capture_provenance(
    session_metadata: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare every manifest row to authoritative server session metadata."""

    def value(item: Mapping[str, Any], field: str) -> str:
        return str(item.get(field) or "").strip()

    def distinct(field: str) -> list[str]:
        return sorted(
            {value(record, field) for record in records if value(record, field)}
        )

    session_capture_run_id = value(session_metadata, "capture_run_id")
    session_capture_source = value(session_metadata, "capture_source")
    session_source_session_id = value(session_metadata, "source_session_id")
    manifest_capture_run_ids = distinct("capture_run_id")
    manifest_capture_sources = distinct("capture_source")
    manifest_source_session_ids = distinct("source_session_id")
    checks = {
        "manifest_capture_run_id_matches_session_metadata": (
            bool(session_capture_run_id)
            and all(
                value(record, "capture_run_id") == session_capture_run_id
                for record in records
            )
        ),
        "manifest_capture_source_matches_session_metadata": (
            bool(session_capture_source)
            and all(
                value(record, "capture_source") == session_capture_source
                for record in records
            )
        ),
        "manifest_source_session_id_matches_session_metadata": (
            all(
                value(record, "source_session_id") == session_source_session_id
                for record in records
            )
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "authority": "server-side calibration session metadata",
        "checks": checks,
        "session_metadata": {
            "capture_run_id_present": bool(session_capture_run_id),
            "capture_run_id_sha256": hashlib.sha256(
                session_capture_run_id.encode("utf-8")
            ).hexdigest(),
            "capture_source": session_capture_source or None,
            "source_session_id_present": bool(session_source_session_id),
            "source_session_id_sha256": hashlib.sha256(
                session_source_session_id.encode("utf-8")
            ).hexdigest(),
        },
        "manifest": {
            "row_scope": "all manifest records, including non-usable rows",
            "row_count": len(records),
            "capture_run_id_count": len(manifest_capture_run_ids),
            "capture_run_ids_match_one_authoritative_value": checks[
                "manifest_capture_run_id_matches_session_metadata"
            ],
            "capture_sources": manifest_capture_sources,
            "source_session_id_count": len(manifest_source_session_ids),
        },
    }


def _calibration_viewport_provenance(
    records: Sequence[Mapping[str, Any]],
    *,
    evaluation_width: float,
    evaluation_height: float,
) -> dict[str, Any]:
    usable = [record for record in records if record.get("ok") is True]
    viewports: set[tuple[float, float]] = set()
    rows_with_viewport = 0
    for index, record in enumerate(usable):
        if record.get("viewport_width") is None or record.get("viewport_height") is None:
            continue
        rows_with_viewport += 1
        viewports.add(
            (
                _positive(
                    record.get("viewport_width"),
                    field=f"calibration[{index}].viewport_width",
                ),
                _positive(
                    record.get("viewport_height"),
                    field=f"calibration[{index}].viewport_height",
                ),
            )
        )
    expected = (evaluation_width, evaluation_height)
    complete = rows_with_viewport == len(usable)
    consistent = complete and viewports == {expected}
    return {
        "status": "passed" if consistent else "failed",
        "usable_row_count": len(usable),
        "rows_with_viewport_count": rows_with_viewport,
        "distinct_calibration_viewport_count": len(viewports),
        "matches_evaluation_viewport": consistent,
        "evaluation_viewport": {
            "width_px": evaluation_width,
            "height_px": evaluation_height,
        },
    }


def _estimated_frame_rate_band(
    value: Any,
) -> tuple[str | None, float | None, float | None]:
    """Return the participant UI's coarse FPS band as [lower, upper)."""

    band = str(value or "").strip()
    ranges: dict[str, tuple[float | None, float | None]] = {
        "under_15": (None, 15.0),
        "15_23": (15.0, 24.0),
        "24_30": (24.0, 31.0),
        "over_30": (31.0, None),
    }
    if band not in ranges:
        return (band or None, None, None)
    lower, upper = ranges[band]
    return band, lower, upper


def _cross_phase_camera_geometry(
    records: Sequence[Mapping[str, Any]],
    *,
    participant_device: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit calibration camera metadata against the participant system check.

    Aspect ratio is the only numeric camera-geometry integrity gate here.
    Absolute resolution and frame rate are retained as diagnostics because the
    runtime contract permits those values to change while preserving geometry.
    """

    usable = [record for record in records if record.get("ok") is True]
    calibration_resolutions: set[tuple[float, float]] = set()
    calibration_frame_rates: set[float] = set()
    rows_with_camera_geometry = 0
    rows_with_frame_rate = 0
    for index, record in enumerate(usable):
        width = record.get("camera_width")
        height = record.get("camera_height")
        if width is not None and height is not None:
            rows_with_camera_geometry += 1
            calibration_resolutions.add(
                (
                    _positive(width, field=f"calibration[{index}].camera_width"),
                    _positive(height, field=f"calibration[{index}].camera_height"),
                )
            )
        frame_rate = record.get("camera_frame_rate")
        if frame_rate is not None:
            rows_with_frame_rate += 1
            calibration_frame_rates.add(
                _positive(
                    frame_rate,
                    field=f"calibration[{index}].camera_frame_rate",
                )
            )

    participant_width_raw = participant_device.get("camera_width")
    participant_height_raw = participant_device.get("camera_height")
    participant_width = (
        _positive(participant_width_raw, field="participant.camera_width")
        if participant_width_raw is not None
        else None
    )
    participant_height = (
        _positive(participant_height_raw, field="participant.camera_height")
        if participant_height_raw is not None
        else None
    )
    participant_geometry_present = (
        participant_width is not None and participant_height is not None
    )
    participant_aspect_ratio = (
        participant_width / participant_height
        if participant_geometry_present
        else None
    )

    calibration_geometry_complete = (
        bool(usable) and rows_with_camera_geometry == len(usable)
    )
    resolution_rows = [
        {
            "width_px": width,
            "height_px": height,
            "aspect_ratio": width / height,
        }
        for width, height in sorted(calibration_resolutions)
    ]
    aspect_differences = (
        [
            abs(float(item["aspect_ratio"]) - participant_aspect_ratio)
            for item in resolution_rows
        ]
        if participant_aspect_ratio is not None
        else []
    )
    maximum_aspect_difference = max(aspect_differences, default=None)
    aspect_ratio_matches = (
        calibration_geometry_complete
        and participant_geometry_present
        and maximum_aspect_difference is not None
        and maximum_aspect_difference
        <= MAX_CROSS_PHASE_CAMERA_ASPECT_RATIO_DIFFERENCE
        + COORDINATE_ABS_TOLERANCE
    )

    exact_resolution_matches = (
        calibration_geometry_complete
        and participant_geometry_present
        and calibration_resolutions == {(participant_width, participant_height)}
    )
    participant_band, band_lower, band_upper = _estimated_frame_rate_band(
        participant_device.get("estimated_camera_fps_band")
    )
    frame_rate_comparable = (
        rows_with_frame_rate == len(usable)
        and bool(calibration_frame_rates)
        and participant_band is not None
        and (band_lower is not None or band_upper is not None)
    )
    frame_rate_matches_band: bool | None = None
    if frame_rate_comparable:
        frame_rate_matches_band = all(
            (band_lower is None or rate >= band_lower)
            and (band_upper is None or rate < band_upper)
            for rate in calibration_frame_rates
        )

    warnings: list[str] = []
    if calibration_geometry_complete and participant_geometry_present:
        if not exact_resolution_matches:
            warnings.append("absolute_camera_resolution_changed_diagnostic_only")
    if rows_with_frame_rate != len(usable):
        warnings.append("calibration_frame_rate_missing_on_some_usable_rows")
    if not calibration_frame_rates:
        warnings.append("calibration_frame_rate_unavailable")
    if participant_band is None or (band_lower is None and band_upper is None):
        warnings.append("participant_estimated_frame_rate_band_unavailable")
    elif frame_rate_matches_band is False:
        warnings.append(
            "calibration_frame_rate_outside_participant_estimated_band_diagnostic_only"
        )

    hard_failures: list[str] = []
    if not calibration_geometry_complete:
        hard_failures.append("calibration_camera_geometry_incomplete")
    if not participant_geometry_present:
        hard_failures.append("participant_system_check_camera_geometry_missing")
    if (
        calibration_geometry_complete
        and participant_geometry_present
        and not aspect_ratio_matches
    ):
        hard_failures.append("cross_phase_camera_aspect_ratio_mismatch")

    return {
        "status": "passed" if not hard_failures else "failed",
        "hard_integrity_rule": (
            "every usable calibration row and the participant system check must "
            "provide camera geometry, and maximum absolute aspect-ratio difference "
            f"must be <= {MAX_CROSS_PHASE_CAMERA_ASPECT_RATIO_DIFFERENCE}"
        ),
        "maximum_allowed_absolute_aspect_ratio_difference": (
            MAX_CROSS_PHASE_CAMERA_ASPECT_RATIO_DIFFERENCE
        ),
        "absolute_resolution_policy": "diagnostic_warning_only",
        "frame_rate_policy": "diagnostic_warning_only",
        "calibration_manifest": {
            "usable_row_count": len(usable),
            "rows_with_camera_geometry_count": rows_with_camera_geometry,
            "rows_with_frame_rate_count": rows_with_frame_rate,
            "distinct_resolutions": resolution_rows,
            "distinct_frame_rates_fps": sorted(calibration_frame_rates),
        },
        "participant_system_check": {
            "camera_width_px": participant_width,
            "camera_height_px": participant_height,
            "aspect_ratio": participant_aspect_ratio,
            "estimated_frame_rate_band": participant_band,
            "estimated_frame_rate_band_minimum_inclusive_fps": band_lower,
            "estimated_frame_rate_band_maximum_exclusive_fps": band_upper,
        },
        "checks": {
            "calibration_camera_geometry_complete": calibration_geometry_complete,
            "participant_camera_geometry_present": participant_geometry_present,
            "aspect_ratio_matches_within_tolerance": aspect_ratio_matches,
            "exact_absolute_resolution_matches_diagnostic": exact_resolution_matches,
            "frame_rate_matches_participant_estimated_band_diagnostic": (
                frame_rate_matches_band
            ),
        },
        "maximum_observed_absolute_aspect_ratio_difference": (
            maximum_aspect_difference
        ),
        "hard_failures": hard_failures,
        "warnings": warnings,
    }


def _model_metrics(model: Mapping[str, Any]) -> dict[str, Any]:
    stages = model.get("stages")
    if not isinstance(stages, list) or not stages or not isinstance(stages[-1], Mapping):
        raise MeasurementCeilingError("model artifact has no calibration stage")
    stage = stages[-1]
    comparison = model.get("candidate_comparison")
    selected_outer: float | None = None
    stage_name = str(stage.get("calibrator_type") or "gaze_polynomial").strip()
    selected_name = stage_name
    comparison_selected: str | None = None
    comparison_selected_known: bool | None = None
    if isinstance(comparison, Mapping):
        comparison_selected = str(comparison.get("selected") or "").strip()
        selected_name = comparison_selected or stage_name
        outer_keys = {
            "gaze_polynomial": "baseline_gaze_only_px",
            "motion_conditioned_ridge_v1": "motion_conditioned_px",
        }
        comparison_selected_known = selected_name in outer_keys
        if comparison_selected_known:
            key = outer_keys[selected_name]
            if comparison.get(key) is not None:
                selected_outer = _finite(
                    comparison.get(key),
                    field=f"model.{key}",
                )
    top_level = (
        _finite(model.get("validation_px_error"), field="model.validation_px_error")
        if model.get("validation_px_error") is not None
        else None
    )
    stage_validation = (
        _finite(stage.get("validation_px_error"), field="stage.validation_px_error")
        if stage.get("validation_px_error") is not None
        else None
    )
    top_hyperparameter = (
        _finite(
            model.get("hyperparameter_cv_px_error"),
            field="model.hyperparameter_cv_px_error",
        )
        if model.get("hyperparameter_cv_px_error") is not None
        else None
    )
    stage_hyperparameter = (
        _finite(
            stage.get("hyperparameter_cv_px_error"),
            field="stage.hyperparameter_cv_px_error",
        )
        if stage.get("hyperparameter_cv_px_error") is not None
        else None
    )
    hyperparameter_value = (
        stage_hyperparameter
        if stage_hyperparameter is not None
        else top_hyperparameter
    )
    expected = selected_outer if selected_outer is not None else stage_validation
    top_matches_expected = (
        top_level is not None
        and expected is not None
        and math.isclose(top_level, expected, rel_tol=0.0, abs_tol=1e-9)
    )
    stage_matches_expected = (
        stage_validation is not None
        and expected is not None
        and math.isclose(stage_validation, expected, rel_tol=0.0, abs_tol=1e-9)
    )
    selection_matches_stage = (
        None
        if not isinstance(comparison, Mapping)
        else bool(comparison_selected) and selected_name == stage_name
    )
    outer_metric_present = (
        None if not isinstance(comparison, Mapping) else selected_outer is not None
    )
    hyperparameter_fields_match = (
        None
        if top_hyperparameter is None or stage_hyperparameter is None
        else math.isclose(
            top_hyperparameter,
            stage_hyperparameter,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    )
    validation_schemes = [
        str(value).strip()
        for value in (
            model.get("validation_scheme"),
            stage.get("validation_scheme"),
            comparison.get("validation_scheme")
            if isinstance(comparison, Mapping)
            else None,
        )
        if str(value or "").strip()
    ]
    validation_scheme_consistent = len(set(validation_schemes)) <= 1
    checks: dict[str, bool | None] = {
        "top_level_matches_selected_outer_or_stage": top_matches_expected,
        "selected_stage_matches_selected_outer": stage_matches_expected,
        "comparison_selected_model_is_known": comparison_selected_known,
        "comparison_selection_matches_stage": selection_matches_stage,
        "selected_outer_metric_is_present": outer_metric_present,
        "top_and_stage_hyperparameter_cv_match_when_both_present": (
            hyperparameter_fields_match
        ),
        "validation_scheme_fields_are_consistent": validation_scheme_consistent,
    }
    consistent = all(value is not False for value in checks.values())
    return {
        "selected_calibrator": selected_name,
        "validation_scheme": model.get("validation_scheme"),
        "top_level_validation_px_error": top_level,
        "selected_stage_validation_px_error": stage_validation,
        "selected_nested_outer_macro_px_error": selected_outer,
        "hyperparameter_cv_px_error": hyperparameter_value,
        "top_level_hyperparameter_cv_px_error": top_hyperparameter,
        "selected_stage_hyperparameter_cv_px_error": stage_hyperparameter,
        "validation_metric_consistency": {
            "status": "passed" if consistent else "failed",
            "expected_top_level_px_error": expected,
            "checks": checks,
            "note": (
                "top-level and selected-stage validation must identify the selected "
                "held-out score; hyperparameter CV remains a separate metric"
            ),
        },
    }


def _binding_checks(
    participant: Mapping[str, Any],
    session_metadata_path: Path,
    session_metadata: Mapping[str, Any],
    manifest_path: Path,
    model: Mapping[str, Any],
) -> dict[str, Any]:
    linked = participant.get("linked_data")
    if not isinstance(linked, Mapping):
        raise MeasurementCeilingError("participant session lacks linked_data")
    gaze_session_id = str(linked.get("gaze_session_id") or "").strip()
    model_name = str(linked.get("model_name") or "").strip()
    session_id = str(session_metadata.get("session_id") or "").strip()
    participant_study_session_id = str(
        participant.get("study_session_id") or ""
    ).strip()
    calibration_study_session_id = str(
        session_metadata.get("study_session_id") or ""
    ).strip()
    checks = {
        "session_metadata_is_sibling_of_manifest": (
            session_metadata_path.parent == manifest_path.parent
        ),
        "session_metadata_matches_linked_gaze_session": (
            bool(gaze_session_id) and session_id == gaze_session_id
        ),
        "calibration_session_matches_participant_study_session": (
            bool(participant_study_session_id)
            and calibration_study_session_id == participant_study_session_id
        ),
        "manifest_parent_matches_linked_gaze_session": (
            bool(gaze_session_id) and manifest_path.parent.name == gaze_session_id
        ),
        "model_data_session_matches_linked_gaze_session": (
            bool(gaze_session_id)
            and str(model.get("data_session_id") or "").strip() == gaze_session_id
        ),
        "model_name_matches_linked_model": (
            bool(model_name) and str(model.get("name") or "").strip() == model_name
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "linked_gaze_session_id_sha256": hashlib.sha256(
            gaze_session_id.encode("utf-8")
        ).hexdigest(),
        "linked_model_name_sha256": hashlib.sha256(
            model_name.encode("utf-8")
        ).hexdigest(),
    }


def build_measurement_ceiling_result(
    *,
    participant_session_path: str | Path,
    calibration_session_metadata_path: str | Path,
    calibration_manifest_path: str | Path,
    model_artifact_path: str | Path,
    line_gap_px: float,
    median_word_width_px: float,
    participant_session_label: str = "participant-session",
    calibration_manifest_label: str = "calibration-manifest",
    model_artifact_label: str = "model-artifact",
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20260810,
    target_overlap_tolerance: float = DEFAULT_TARGET_OVERLAP_TOLERANCE_SIGNED,
) -> dict[str, Any]:
    """Build a deterministic aggregate result from explicit target evidence."""

    participant_path = Path(participant_session_path).resolve()
    session_metadata_path = Path(calibration_session_metadata_path).resolve()
    manifest_path = Path(calibration_manifest_path).resolve()
    model_path = Path(model_artifact_path).resolve()
    line_gap = _positive(line_gap_px, field="line_gap_px")
    word_width = _positive(median_word_width_px, field="median_word_width_px")
    tolerance = _signed_distance_tolerance(target_overlap_tolerance)
    participant = _load_object(participant_path, label="participant session")
    session_metadata = _load_object(
        session_metadata_path,
        label="calibration session metadata",
    )
    calibration_records = _load_jsonl(manifest_path, label="calibration manifest")
    model = _load_object(model_path, label="model artifact")
    model_file_sha256 = _sha256(model_path)

    general_collection = participant.get("general_collection")
    if not isinstance(general_collection, Mapping):
        raise MeasurementCeilingError(
            "participant session lacks general_collection data"
        )
    validations = general_collection.get("validations")
    if not isinstance(validations, Mapping):
        raise MeasurementCeilingError("participant session lacks validations")
    start_validation = validations.get("start")
    end_validation = validations.get("end")
    if not isinstance(start_validation, Mapping) or not isinstance(
        end_validation, Mapping
    ):
        raise MeasurementCeilingError("start and end validation are required")

    quality = participant.get("quality")
    system_check = quality.get("general_system_check") if isinstance(quality, Mapping) else None
    device = system_check.get("device") if isinstance(system_check, Mapping) else None
    if not isinstance(device, Mapping):
        raise MeasurementCeilingError("participant session lacks viewport metadata")
    viewport_width = _positive(device.get("viewport_width"), field="viewport_width")
    viewport_height = _positive(device.get("viewport_height"), field="viewport_height")

    start_records, start_attempted = _validation_records(
        start_validation,
        phase="start",
    )
    end_records, end_attempted = _validation_records(end_validation, phase="end")
    start_targets = _target_coordinates(start_records)
    end_targets = _target_coordinates(end_records)
    if start_targets != end_targets:
        raise MeasurementCeilingError(
            "start and end evaluation target coordinates must match"
        )
    start_metrics = _phase_metrics(start_records, attempted_count=start_attempted)
    end_metrics = _phase_metrics(end_records, attempted_count=end_attempted)
    target_independence = _target_independence(
        calibration_records,
        start_targets,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        tolerance=tolerance,
    )
    correction = _temporal_correction(
        start_records,
        end_records,
        end_attempted_count=end_attempted,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    repeatability_proxy = _temporal_repeatability_proxy(
        start_records,
        end_records,
    )
    heldout_uncertainty = _heldout_uncertainty_coverage_risk(
        model,
        validations,
        general_collection,
        model_artifact_file_sha256=model_file_sha256,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
    )
    uncertainty_v2_requirements = _future_uncertainty_v2_requirements(
        model,
        calibration_records,
        validations,
        heldout_uncertainty,
    )
    binding = _binding_checks(
        participant,
        session_metadata_path,
        session_metadata,
        manifest_path,
        model,
    )
    capture_provenance = _capture_provenance(
        session_metadata,
        calibration_records,
    )
    viewport_provenance = _calibration_viewport_provenance(
        calibration_records,
        evaluation_width=viewport_width,
        evaluation_height=viewport_height,
    )
    camera_geometry = _cross_phase_camera_geometry(
        calibration_records,
        participant_device=device,
    )
    model_metrics = _model_metrics(model)

    def normalized_resolution(metrics: Mapping[str, Any]) -> dict[str, float]:
        median_error = float(metrics["spatial_error_px"]["median"])
        p90_error = float(metrics["spatial_error_px"]["p90"])
        return {
            "median_error_in_line_gaps": median_error / line_gap,
            "p90_error_in_line_gaps": p90_error / line_gap,
            "median_error_in_median_word_widths": median_error / word_width,
            "p90_error_in_median_word_widths": p90_error / word_width,
        }

    integrity_gate_passed = (
        binding["status"] == "passed"
        and capture_provenance["status"] == "passed"
        and viewport_provenance["status"] == "passed"
        and camera_geometry["status"] == "passed"
        and target_independence["status"] == "passed"
        and model_metrics["validation_metric_consistency"]["status"] == "passed"
        and heldout_uncertainty.get("integrity_status") != "failed"
    )
    result_status = "completed" if integrity_gate_passed else "failed_integrity_gate"
    linked = participant.get("linked_data", {})
    not_evaluable = {
        "natural_reading_line_accuracy": {
            "status": "not_evaluable",
            "reason": "no independent line-level gaze ground truth was recorded",
        },
        "natural_reading_word_accuracy": {
            "status": "not_evaluable",
            "reason": "no independent word-level gaze ground truth was recorded",
        },
    }
    if heldout_uncertainty.get("status") != "evaluable_descriptive_heldout":
        not_evaluable["per_sample_uncertainty_calibration"] = {
            "status": "not_evaluable",
            "reason": str(
                heldout_uncertainty.get(
                    "reason",
                    "receipt-verified frozen-v2 uncertainty evidence is unavailable",
                )
            ),
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": ANALYSIS_ID,
        "status": result_status,
        "evidence_class": "exploratory_self_development_only",
        "analysis_contract": {
            "cpu_only": True,
            "standard_library_only": True,
            "images_or_videos_opened": False,
            "torch_or_gpu_loaded": False,
            "natural_reading_nearest_word_index_used_as_ground_truth": False,
            "correction_fit_data": "start validation only",
            "correction_evaluation_data": "end validation only",
            "model_or_threshold_selection_authorized": False,
            "repeatability_proxy_threshold_selection_authorized": False,
            "repeatability_proxy_quality_band_change_authorized": False,
            "uncertainty_v2_threshold_selection_authorized": False,
            "uncertainty_v2_quality_band_change_authorized": False,
            "uncertainty_v2_risk_uses_heldout_target_error_only": True,
            "uncertainty_v2_score_uses_heldout_target_error": False,
            "fixed_target_sample_rows_treated_as_independent": False,
            "descriptive_percentile_method": "nearest rank at ceil(n * p)",
        },
        "inputs": {
            "participant_session": {
                "label": participant_session_label,
                "sha256": _sha256(participant_path),
            },
            "calibration_session_metadata": {
                "label": "linked calibration session metadata",
                "sha256": _sha256(session_metadata_path),
            },
            "calibration_manifest": {
                "label": calibration_manifest_label,
                "sha256": _sha256(manifest_path),
            },
            "model_artifact": {
                "label": model_artifact_label,
                "sha256": model_file_sha256,
            },
        },
        "provenance": {
            "participant": {
                "schema_version": participant.get("schema_version"),
                "protocol_id": participant.get("protocol_id"),
                "protocol_version": participant.get("protocol_version"),
                "mode": participant.get("mode"),
                "state": participant.get("state"),
                "dataset_role": participant.get("dataset_role"),
                "linked_gaze_present": bool(linked.get("gaze_session_id"))
                if isinstance(linked, Mapping)
                else False,
                "linked_model_present": bool(linked.get("model_name"))
                if isinstance(linked, Mapping)
                else False,
            },
            "calibration_manifest": _manifest_provenance(calibration_records),
            "capture_contract": capture_provenance,
            "calibration_viewport_contract": viewport_provenance,
            "cross_phase_camera_geometry": camera_geometry,
            "bindings": binding,
            "model": model_metrics,
            "uncertainty_v2_integrity": {
                "status": heldout_uncertainty.get("integrity_status"),
                "evaluation_status": heldout_uncertainty.get("status"),
                "reason": heldout_uncertainty.get("reason"),
            },
        },
        "viewport": {
            "width_px": viewport_width,
            "height_px": viewport_height,
        },
        "target_independence": target_independence,
        "raw_validation": {
            "start": start_metrics,
            "end": end_metrics,
        },
        "layout_normalized_resolution": {
            "status": "descriptive_resolution_only_not_reading_accuracy",
            "line_gap_px": line_gap,
            "median_word_width_px": word_width,
            "start": normalized_resolution(start_metrics),
            "end": normalized_resolution(end_metrics),
        },
        "drift": _drift_vectors(start_metrics, end_metrics),
        "temporal_correction": correction,
        "temporal_repeatability_proxy": repeatability_proxy,
        "heldout_uncertainty_coverage_risk": heldout_uncertainty,
        "future_uncertainty_v2_data_requirements": uncertainty_v2_requirements,
        "not_evaluable": not_evaluable,
        "decision": {
            "quality_band_changed": False,
            "production_model_changed": False,
            "eligible_claim": "coarse fixed-target development evidence only",
            "line_or_word_accuracy_claimed": False,
            "integrity_gate_passed": integrity_gate_passed,
        },
    }
    return result


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def render_measurement_ceiling_markdown(
    result: Mapping[str, Any],
    *,
    result_reference: str = "results/2026-08-10-webcam-gaze-measurement-ceiling-v1.json",
) -> str:
    """Render the aggregate result without adding wall-clock state."""

    start = result["raw_validation"]["start"]
    end = result["raw_validation"]["end"]
    independence = result["target_independence"]
    correction = result["temporal_correction"]
    bootstrap = correction["bootstrap"]
    repeatability_proxy = result["temporal_repeatability_proxy"]
    heldout_uncertainty = result["heldout_uncertainty_coverage_risk"]
    uncertainty_v2 = result["future_uncertainty_v2_data_requirements"]
    model = result["provenance"]["model"]
    capture = result["provenance"]["capture_contract"]
    camera_geometry = result["provenance"]["cross_phase_camera_geometry"]
    drift = result["drift"]

    proxy_lines = [
        "## Start-only repeatability proxy (descriptive only)",
        "",
        f"Claim boundary: `{repeatability_proxy['claim_boundary']}`. The score is "
        "computed only from repeated start-validation predictions; target risk is "
        "computed only from end-validation target error. The analysis unit is a "
        "whole target cluster, not an individual frame or reading sample.",
        "",
        "The coverage grid is frozen at `20/40/60/80/100%`; it is not searched, "
        "and this result cannot select an abstention threshold, change a quality "
        "band, or authorize per-sample abstention.",
        "",
    ]
    if repeatability_proxy["status"] == "evaluable_descriptive_proxy":
        proxy_lines.extend(
            [
                "| Target | Start repeats | Start RMS repeatability px | End samples | End mean error px | End median error px |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for target_id, target in repeatability_proxy["targets"].items():
            proxy_lines.append(
                f"| {target_id} | {target['start_successful_repeat_count']} | "
                f"{_fmt(target['start_repeatability_rms_px'])} | "
                f"{target['end_successful_sample_count']} | "
                f"{_fmt(target['end_mean_spatial_error_px'])} | "
                f"{_fmt(target['end_median_spatial_error_px'])} |"
            )
        proxy_lines.extend(
            [
                "",
                "| Requested coverage | Achieved coverage | Retained targets | End target-macro mean error px | End target-macro median error px |",
                "| ---: | ---: | --- | ---: | ---: |",
            ]
        )
        for point in repeatability_proxy["coverage_risk_curve"]:
            proxy_lines.append(
                f"| {_fmt(point['requested_coverage'] * 100, 0)}% | "
                f"{_fmt(point['achieved_coverage'] * 100, 0)}% | "
                f"{', '.join(point['retained_target_ids'])} | "
                f"{_fmt(point['end_target_macro_mean_spatial_error_px'])} | "
                f"{_fmt(point['end_target_macro_median_spatial_error_px'])} |"
            )
        association = repeatability_proxy["association"]
        negative = repeatability_proxy["negative_result"]
        proxy_lines.extend(
            [
                "",
                f"Target-level Spearman association (`{association['metric']}`): "
                f"`{_fmt(association['value'], 3)}`; a useful low-to-high risk proxy "
                "would have a positive association.",
                "",
                "At 20% coverage, end target-macro mean error was "
                f"`{_fmt(negative['lowest_coverage_end_mean_error_px'])} px`, versus "
                f"`{_fmt(negative['full_coverage_end_mean_error_px'])} px` at full "
                "coverage (difference "
                f"`{_fmt(negative['lowest_coverage_minus_full_coverage_px'])} px`). "
                f"Recorded conclusion: `{negative['conclusion']}`.",
                "",
                "This is a preserved negative descriptive result, not predictive "
                "uncertainty calibration.",
                "",
            ]
        )
    else:
        proxy_lines.extend(
            [
                f"Status: `not_evaluable`; {repeatability_proxy['reason']}.",
                "",
            ]
        )

    inventory = uncertainty_v2["current_evidence_inventory"]

    def inline_fields(values: Sequence[str]) -> str:
        return ", ".join(f"`{value}`" for value in values) or "`none`"

    heldout_uncertainty_lines = [
        "## Receipt-verified held-out uncertainty coverage-risk",
        "",
    ]
    if heldout_uncertainty["status"] == "evaluable_descriptive_heldout":
        heldout_uncertainty_lines.extend(
            [
                "Status: `evaluable_descriptive_heldout`. Runtime scores are bound "
                "to frozen training-only uncertainty definition "
                f"`{heldout_uncertainty['definition_sha256']}`. Held-out target "
                "coordinates enter only the risk calculation, never the score or "
                "ordering.",
                "",
                "The grid is frozen at `100/80/60/40/20%`. No threshold was "
                "selected, no quality band changes, and the hypothetically "
                "abstained rows remain in the machine-readable result.",
                "",
                "| Scope | Attempts | Successful | No face | Capture success | Target clusters |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for scope_name, scope_result in (
            ("Start", heldout_uncertainty["phases"]["start"]),
            ("End", heldout_uncertainty["phases"]["end"]),
            ("Combined", heldout_uncertainty["combined"]),
        ):
            heldout_uncertainty_lines.append(
                f"| {scope_name} | {scope_result['attempted_capture_count']} | "
                f"{scope_result['successful_prediction_count']} | "
                f"{scope_result['no_face_count']} | "
                f"{_fmt(scope_result['capture_success_fraction'] * 100)}% | "
                f"{scope_result['independent_target_cluster_count']} |"
            )
        heldout_uncertainty_lines.append("")
        for scope_name, scope_result in (
            ("Start", heldout_uncertainty["phases"]["start"]),
            ("End", heldout_uncertainty["phases"]["end"]),
            ("Combined", heldout_uncertainty["combined"]),
        ):
            heldout_uncertainty_lines.extend(
                [
                    f"### {scope_name}",
                    "",
                    "| Requested score coverage | Achieved among successful | End-to-end attempt coverage | Retained | Target-macro mean px | Worst target mean px | Zero-coverage targets |",
                    "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
                ]
            )
            for point in scope_result["coverage_risk_curve"]:
                heldout_uncertainty_lines.append(
                    f"| {_fmt(point['requested_score_coverage'] * 100, 0)}% | "
                    f"{_fmt(point['achieved_score_coverage_within_successful'] * 100, 0)}% | "
                    f"{_fmt(point['achieved_end_to_end_attempt_coverage'] * 100, 0)}% | "
                    f"{point['retained_successful_count']} | "
                    f"{_fmt(point['target_cluster_macro_all_clusters']['mean_spatial_error_px'])} | "
                    f"{_fmt(point['worst_target_cluster_mean_spatial_error_px'])} | "
                    f"{', '.join(point['target_clusters_with_zero_coverage']) or 'none'} |"
                )
            heldout_uncertainty_lines.append("")
        heldout_uncertainty_lines.extend(
            [
                "Only five target clusters are observed. The 15 phase rows and 30 "
                "combined rows are repeated measurements, not independent units; "
                "zero-coverage clusters make the all-cluster macro and worst-cluster "
                "metrics unavailable rather than silently dropping those targets.",
                "",
            ]
        )
    else:
        heldout_uncertainty_lines.extend(
            [
                f"Status: `{heldout_uncertainty['status']}`; "
                f"{heldout_uncertainty.get('reason', 'required evidence unavailable')}.",
                "",
                "No fixed coverage-risk curve, threshold, abstention policy, or "
                "quality-band change is authorized. The start-only repeatability "
                "proxy remains explicitly `proxy_not_predictive_uncertainty`.",
                "",
            ]
        )

    curve_constructable = inventory[
        "predictive_uncertainty_curve_constructable_from_current_artifacts"
    ]
    uncertainty_v2_lines = [
        "## Predictive uncertainty v2 evidence requirements",
        "",
        f"Status: `{uncertainty_v2['status']}`. A receipt-verified descriptive "
        "fixed-target coverage-risk curve "
        + ("is reconstructable." if curve_constructable else "is not reconstructable."),
        "",
        "Current evidence inventory:",
        "",
        "- Model OOF/uncertainty fields: "
        f"{inline_fields(inventory['model_oof_or_uncertainty_fields_present'])}",
        "- Validation uncertainty fields: "
        f"{inline_fields(inventory['validation_uncertainty_fields_present'])}",
        "- Reconstructable calibration sensor fields: "
        f"{inline_fields(inventory['calibration_reconstructable_sensor_fields_present'])}",
        f"- Reason: {inventory['reason']}.",
        "",
        "A frozen v2 must record one row per outer-fold held-out sample with: "
        f"{inline_fields(uncertainty_v2['required_per_oof_sample_fields'])}.",
        "",
        "The uncertainty definition must be bound before evaluation with: "
        f"{inline_fields(uncertainty_v2['required_definition_binding'])}.",
        "",
        "The score, OOD/leverage model, and covariance must be fit using training "
        "partitions only. Evaluation requires a new untouched capture, preserves "
        "raw and abstained predictions, and cannot use holdout target error to "
        "construct the uncertainty score. V1 may not choose a definition or "
        "threshold from this descriptive result.",
        "",
    ]
    lines = [
        "# Webcam Gaze Measurement Ceiling v1 - Existing-Data Audit",
        "",
        f"Status: `{result['status']}`; exploratory self-development evidence only. "
        "This audit does not promote a model, threshold, gaze quality band, or "
        "participant claim.",
        "",
        f"Machine-readable result: [`{result_reference}`]({result_reference})",
        "",
        "## Integrity and provenance",
        "",
        "| Check | Result |",
        "| --- | --- |",
        f"| Artifact bindings | {result['provenance']['bindings']['status']} |",
        f"| Server session/manifest capture contract | {capture['status']} |",
        "| Calibration/evaluation viewport contract | "
        f"{result['provenance']['calibration_viewport_contract']['status']} |",
        "| Cross-phase camera aspect-ratio integrity | "
        f"{camera_geometry['status']} |",
        f"| Calibration/evaluation target independence | {independence['status']} |",
        f"| Model validation metric consistency | {model['validation_metric_consistency']['status']} |",
        "| Frozen-v2 receipt uncertainty integrity | "
        f"{result['provenance']['uncertainty_v2_integrity']['status']} |",
        f"| Calibration targets | {independence['calibration_target_count']} |",
        f"| Evaluation targets | {independence['evaluation_target_count']} |",
        f"| Below-tolerance overlaps | {independence['overlap_count']} |",
        "| Target-separation tolerance | "
        f"{_fmt(independence['signed_normalized_tolerance'])} signed = "
        f"{_fmt(independence['viewport_fraction_tolerance_equivalent'])} "
        "viewport fraction |",
        "| Observed minimum target distance | "
        f"{_fmt(independence['minimum_signed_normalized_euclidean_distance'], 6)} "
        "signed = "
        f"{_fmt(independence['minimum_viewport_fraction_euclidean_distance'], 6)} "
        "viewport fraction |",
        "| Images, videos, Torch, or GPU opened | no |",
        "| Natural-reading nearest-word index used as truth | no |",
        "",
        "Input SHA-256 values:",
        "",
    ]
    for name in (
        "participant_session",
        "calibration_session_metadata",
        "calibration_manifest",
        "model_artifact",
    ):
        source = result["inputs"][name]
        lines.append(f"- `{name}`: `{source['sha256']}`")
    lines.extend(
        [
            "",
            "Target-distance coordinates use `signed = 2 * viewport_fraction - 1`; "
            f"the frozen `{_fmt(independence['signed_normalized_tolerance'])}` "
            "signed Euclidean threshold therefore equals "
            f"`{_fmt(independence['viewport_fraction_tolerance_equivalent'])}` "
            "in `[0, 1]` viewport-fraction coordinates. Distances equal to the "
            "threshold are independent; only smaller distances overlap.",
            "",
            "Server-side calibration session metadata is authoritative for capture "
            "provenance.",
            "",
            f"- Session capture source: `{capture['session_metadata']['capture_source']}`",
            "- Manifest capture sources: "
            f"`{', '.join(capture['manifest']['capture_sources']) or 'missing'}`",
            "",
            "Cross-phase camera geometry uses the calibration manifest and the "
            "participant system-check record. Aspect ratio is a hard integrity "
            "boundary; absolute resolution and frame rate are diagnostic warnings "
            "only.",
            "",
            "- Calibration camera resolutions: "
            f"`{json.dumps(camera_geometry['calibration_manifest']['distinct_resolutions'], sort_keys=True)}`",
            "- Calibration actual frame rates (fps): "
            f"`{json.dumps(camera_geometry['calibration_manifest']['distinct_frame_rates_fps'])}`",
            "- Participant system-check camera: "
            f"`{_fmt(camera_geometry['participant_system_check']['camera_width_px'], 0)}x"
            f"{_fmt(camera_geometry['participant_system_check']['camera_height_px'], 0)}`; "
            "estimated FPS band "
            f"`{camera_geometry['participant_system_check']['estimated_frame_rate_band'] or 'unavailable'}`",
            "- Maximum absolute aspect-ratio difference: "
            f"`{_fmt(camera_geometry['maximum_observed_absolute_aspect_ratio_difference'], 6)}` "
            "(hard maximum "
            f"`{_fmt(camera_geometry['maximum_allowed_absolute_aspect_ratio_difference'], 2)}`)",
            "- Diagnostic warnings: "
            f"`{', '.join(camera_geometry['warnings']) or 'none'}`",
        ]
    )
    if capture["status"] != "passed":
        lines.extend(
            [
                "",
                "**Hard provenance failure:** at least one manifest capture field does "
                "not match the server-created calibration session. The numeric audit "
                "is retained for diagnosis but is ineligible for promotion.",
            ]
        )
    if camera_geometry["status"] != "passed":
        lines.extend(
            [
                "",
                "**Hard cross-phase camera-geometry failure:** "
                f"`{', '.join(camera_geometry['hard_failures'])}`. The numeric audit "
                "is retained for diagnosis but cannot support a matched-capture "
                "measurement claim.",
            ]
        )
    if not independence["frozen_protocol_threshold_match"]:
        lines.extend(
            [
                "",
                "**Target-threshold contract failure:** the configured threshold "
                "does not match the frozen protocol default. Results are development "
                "diagnostics only.",
            ]
        )
    if independence["overlap_count"]:
        overlapping_targets = ", ".join(
            independence["overlapping_evaluation_target_ids"]
        )
        lines.extend(
            [
                "",
                "**Target-independence failure:** calibration and evaluation share "
                f"the following below-tolerance target region(s): "
                f"`{overlapping_targets}`. The frozen threshold is "
                f"`{_fmt(independence['signed_normalized_tolerance'])}` in signed "
                "`[-1, 1]` Euclidean coordinates, equal to "
                f"`{_fmt(independence['viewport_fraction_tolerance_equivalent'])}` "
                "in `[0, 1]` viewport-fraction coordinates. Metrics remain "
                "descriptive and cannot establish target-held-out accuracy.",
            ]
        )
    lines.extend(
        [
            "",
            "## Raw fixed-target result",
            "",
            "| Phase | Median px | P90 px | Target-macro mean px | Target-macro bias px | Median absolute X px | Median absolute Y px | Coarse nearest-target accuracy |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            (
                "| Start | "
                f"{_fmt(start['spatial_error_px']['median'])} | "
                f"{_fmt(start['spatial_error_px']['p90'])} | "
                f"{_fmt(start['target_macro']['mean_spatial_error_px'])} | "
                f"{_fmt(start['target_macro']['mean_bias_magnitude_px'])} | "
                f"{_fmt(start['x_error_px']['absolute_median'])} | "
                f"{_fmt(start['y_error_px']['absolute_median'])} | "
                f"{_fmt(start['coarse_region']['accuracy'] * 100)}% |"
            ),
            (
                "| End | "
                f"{_fmt(end['spatial_error_px']['median'])} | "
                f"{_fmt(end['spatial_error_px']['p90'])} | "
                f"{_fmt(end['target_macro']['mean_spatial_error_px'])} | "
                f"{_fmt(end['target_macro']['mean_bias_magnitude_px'])} | "
                f"{_fmt(end['x_error_px']['absolute_median'])} | "
                f"{_fmt(end['y_error_px']['absolute_median'])} | "
                f"{_fmt(end['coarse_region']['accuracy'] * 100)}% |"
            ),
            "",
            "The five targets are widely separated. High nearest-target accuracy is "
            "coarse-region evidence and does not imply line- or word-level resolution.",
            "Target-macro bias is the equal-weight mean magnitude of each target's "
            "prediction-centroid bias vector.",
            "P90 uses the participant collection's nearest-rank `ceil(n * p)` "
            "definition.",
            "",
            "Axis errors preserve direction as well as absolute magnitude:",
            "",
            "| Phase | Axis | Signed mean px | Signed median px | Absolute median px | Absolute P90 px |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
            (
                "| Start | X | "
                f"{_fmt(start['x_error_px']['signed_mean'])} | "
                f"{_fmt(start['x_error_px']['signed_median'])} | "
                f"{_fmt(start['x_error_px']['absolute_median'])} | "
                f"{_fmt(start['x_error_px']['absolute_p90'])} |"
            ),
            (
                "| Start | Y | "
                f"{_fmt(start['y_error_px']['signed_mean'])} | "
                f"{_fmt(start['y_error_px']['signed_median'])} | "
                f"{_fmt(start['y_error_px']['absolute_median'])} | "
                f"{_fmt(start['y_error_px']['absolute_p90'])} |"
            ),
            (
                "| End | X | "
                f"{_fmt(end['x_error_px']['signed_mean'])} | "
                f"{_fmt(end['x_error_px']['signed_median'])} | "
                f"{_fmt(end['x_error_px']['absolute_median'])} | "
                f"{_fmt(end['x_error_px']['absolute_p90'])} |"
            ),
            (
                "| End | Y | "
                f"{_fmt(end['y_error_px']['signed_mean'])} | "
                f"{_fmt(end['y_error_px']['signed_median'])} | "
                f"{_fmt(end['y_error_px']['absolute_median'])} | "
                f"{_fmt(end['y_error_px']['absolute_p90'])} |"
            ),
            "",
            "Coarse nearest-target confusion matrices (rows are actual targets; "
            "columns are predicted targets):",
            "",
            "Start:",
            "",
            "```json",
            json.dumps(
                start["coarse_region"]["confusion"],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "End:",
            "",
            "```json",
            json.dumps(
                end["coarse_region"]["confusion"],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Layout-relative resolution",
            "",
            f"Configured line gap: `{_fmt(result['layout_normalized_resolution']['line_gap_px'])} px`; "
            f"median word width: `{_fmt(result['layout_normalized_resolution']['median_word_width_px'])} px`.",
            "",
            "| Phase | Median in line gaps | P90 in line gaps | Median in word widths | P90 in word widths |",
            "| --- | ---: | ---: | ---: | ---: |",
            (
                "| Start | "
                f"{_fmt(result['layout_normalized_resolution']['start']['median_error_in_line_gaps'])} | "
                f"{_fmt(result['layout_normalized_resolution']['start']['p90_error_in_line_gaps'])} | "
                f"{_fmt(result['layout_normalized_resolution']['start']['median_error_in_median_word_widths'])} | "
                f"{_fmt(result['layout_normalized_resolution']['start']['p90_error_in_median_word_widths'])} |"
            ),
            (
                "| End | "
                f"{_fmt(result['layout_normalized_resolution']['end']['median_error_in_line_gaps'])} | "
                f"{_fmt(result['layout_normalized_resolution']['end']['p90_error_in_line_gaps'])} | "
                f"{_fmt(result['layout_normalized_resolution']['end']['median_error_in_median_word_widths'])} | "
                f"{_fmt(result['layout_normalized_resolution']['end']['p90_error_in_median_word_widths'])} |"
            ),
            "",
            "These ratios describe measurement resolution only; the natural-reading "
            "trace has no independent line or word truth.",
            "",
            "## Target-wise drift",
            "",
            "| Target | Centroid drift X px | Centroid drift Y px | Drift magnitude px | Median error change px |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    if drift["status"] == "evaluable":
        for target_id, target in drift["targets"].items():
            lines.append(
                f"| {target_id} | "
                f"{_fmt(target['predicted_centroid_delta_x_px'])} | "
                f"{_fmt(target['predicted_centroid_delta_y_px'])} | "
                f"{_fmt(target['predicted_centroid_drift_magnitude_px'])} | "
                f"{_fmt(target['median_spatial_error_change_px'])} |"
            )
    lines.extend(
        [
            "",
            "A single start-minus-end median can conceal target-specific reversals; "
            "the vectors above remain the primary drift description.",
            "",
            "## Start-trained temporal correction",
            "",
            f"Frozen correction: `{correction['correction_id']}`. Translation was fit "
            "only on start validation and applied once to end validation.",
            "",
            "| Metric | Raw end | Corrected end | Corrected - raw |",
            "| --- | ---: | ---: | ---: |",
            (
                "| Median spatial error px | "
                f"{_fmt(correction['raw_end']['median_spatial_error_px'])} | "
                f"{_fmt(correction['corrected_end']['median_spatial_error_px'])} | "
                f"{_fmt(correction['corrected_minus_raw']['median_spatial_error_px'])} |"
            ),
            (
                "| P90 spatial error px | "
                f"{_fmt(correction['raw_end']['p90_spatial_error_px'])} | "
                f"{_fmt(correction['corrected_end']['p90_spatial_error_px'])} | "
                f"{_fmt(correction['corrected_minus_raw']['p90_spatial_error_px'])} |"
            ),
            "",
            (
                f"Target-cluster bootstrap ({bootstrap['resamples']} resamples, seed "
                f"`{bootstrap['seed']}`) gives a corrected-minus-raw median-error "
                f"95% interval of `[{_fmt(bootstrap['ci95_lower_px'])}, "
                f"{_fmt(bootstrap['ci95_upper_px'])}] px`; "
                f"`{_fmt(bootstrap['fraction_improved'] * 100)}%` of resamples improve."
            ),
            (
                f"The paired resampling unit is `{bootstrap['cluster_unit']}` "
                f"({bootstrap['target_cluster_count']} observed clusters; "
                f"{bootstrap['cluster_draws_per_resample']} draws per resample). "
                f"Sampler: `{bootstrap['deterministic_sampler']}`."
            ),
            "The bootstrap interval is descriptive only and does not establish a "
            "population-level correction benefit.",
            "",
            "This result cannot relabel the session or select a production correction.",
            "",
            *proxy_lines,
            *heldout_uncertainty_lines,
            *uncertainty_v2_lines,
            "## Model metric contract",
            "",
            "| Field | Value |",
            "| --- | ---: |",
            f"| Selected calibrator | {model['selected_calibrator']} |",
            f"| Selected nested outer macro px | {_fmt(model['selected_nested_outer_macro_px_error'])} |",
            f"| Selected stage validation px | {_fmt(model['selected_stage_validation_px_error'])} |",
            f"| Top-level validation px | {_fmt(model['top_level_validation_px_error'])} |",
            f"| Hyperparameter CV px | {_fmt(model['hyperparameter_cv_px_error'])} |",
            f"| Stage hyperparameter CV px | {_fmt(model['selected_stage_hyperparameter_cv_px_error'])} |",
            f"| Top-level hyperparameter CV px | {_fmt(model['top_level_hyperparameter_cv_px_error'])} |",
            f"| Metric consistency | {model['validation_metric_consistency']['status']} |",
            "",
            "A failed consistency check records the historical M0 artifact bug; it does "
            "not rewrite the artifact or substitute the inner CV score for held-out evidence.",
            "",
            "## Not evaluable",
            "",
            *(
                [
                    "- Predictive uncertainty calibration: **not evaluable**; "
                    + str(
                        result["not_evaluable"][
                            "per_sample_uncertainty_calibration"
                        ]["reason"]
                    )
                    + "."
                ]
                if "per_sample_uncertainty_calibration" in result["not_evaluable"]
                else []
            ),
            "- Natural-reading line accuracy: **not evaluable**; no independent line-level "
            "ground truth exists.",
            "- Natural-reading word accuracy: **not evaluable**; no independent word-level "
            "ground truth exists.",
            "",
            "## Decision",
            "",
            "Preserve the negative and mixed findings. The current data support at most "
            "coarse fixed-target development evidence. Any failed integrity check is a "
            "hard stop. No quality band, production model, or line/word claim is promoted.",
            "",
        ]
    )
    return "\n".join(lines)


def deterministic_json(result: Mapping[str, Any]) -> str:
    """Return the canonical human-readable JSON representation."""

    return json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

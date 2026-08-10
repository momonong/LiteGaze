"""Leakage-safe descriptive analysis for the frozen 193-row gaze run.

The acquisition artifact is verified by :mod:`measurement_schedule` before
any metric is calculated.  This module consumes only frozen target geometry
and sensor-derived predictions.  It has no text, cursor, cognitive-profile,
reading-outcome, or threshold-selection input.

The output is deliberately descriptive.  It cannot authorize a production
quality band, an abstention threshold, natural-reading accuracy, or a
population-level claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from .measurement_schedule import (
    MeasurementScheduleError,
    build_run_manifest,
    canonical_sha256,
    load_frozen_protocol,
    verify_capture_artifact,
)


ANALYSIS_SCHEMA_VERSION = 1
ANALYSIS_TYPE = "webcam_gaze_measurement_ceiling_descriptive_analysis_v1"
PERCENTILE_ALGORITHM = "linear_r7_v1"
BOOTSTRAP_ALGORITHM = "sha256_counter_target_cluster_v1"
UNCERTAINTY_COVERAGES = (1.0, 0.8, 0.6, 0.4, 0.2)
DEFAULT_BOOTSTRAP_RESAMPLES = 20_000
DEFAULT_BOOTSTRAP_SEED = 20_260_810
ANALYSIS_DEFINITION_RELATIVE_PATH = (
    "docs/experiments/protocols/"
    "2026-08-10-webcam-gaze-measurement-analysis-v1.json"
)
ANALYSIS_DEFINITION_PATH = (
    Path(__file__).resolve().parents[2] / ANALYSIS_DEFINITION_RELATIVE_PATH
)
EXPECTED_ANALYSIS_DEFINITION_SHA256 = (
    "d3118fb8a1cb4eff437ea45e2b9b4619ce78e856d2bfbf84a4acef80f278755a"
)


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise MeasurementScheduleError(f"{field} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementScheduleError(f"{field} must be finite") from exc
    if not math.isfinite(number):
        raise MeasurementScheduleError(f"{field} must be finite")
    return number


def load_frozen_analysis_definition(
    path: str | Path | None = None,
) -> tuple[dict[str, Any], str]:
    definition_path = (
        Path(path).resolve() if path is not None else ANALYSIS_DEFINITION_PATH
    )
    try:
        payload = json.loads(definition_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementScheduleError(
            "frozen measurement analysis definition is unavailable or invalid"
        ) from exc
    if not isinstance(payload, dict):
        raise MeasurementScheduleError(
            "frozen measurement analysis definition must be an object"
        )
    digest = canonical_sha256(payload)
    if digest != EXPECTED_ANALYSIS_DEFINITION_SHA256:
        raise MeasurementScheduleError(
            "frozen measurement analysis definition canonical SHA-256 mismatch"
        )
    if payload.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise MeasurementScheduleError("analysis definition schema changed")
    if payload.get("definition_id") != (
        "webcam-gaze-measurement-ceiling-analysis-v1"
    ):
        raise MeasurementScheduleError("analysis definition id changed")
    if payload.get("status") != "frozen_before_new_capture":
        raise MeasurementScheduleError("analysis definition status changed")
    geometry = dict(payload.get("geometry_metrics") or {})
    uncertainty = dict(payload.get("uncertainty") or {})
    bootstrap = dict(payload.get("bootstrap") or {})
    decoder = dict(payload.get("target_region_decoder") or {})
    if geometry.get("percentile_algorithm") != PERCENTILE_ALGORITHM:
        raise MeasurementScheduleError("analysis percentile algorithm changed")
    if decoder.get("algorithm") != (
        "nearest_axis_on_frozen_4x4_target_grid_v1"
    ):
        raise MeasurementScheduleError("analysis target decoder changed")
    if uncertainty.get("fixed_conditional_coverage_grid") != list(
        UNCERTAINTY_COVERAGES
    ):
        raise MeasurementScheduleError("analysis uncertainty coverage grid changed")
    if bootstrap != {
        "algorithm": BOOTSTRAP_ALGORITHM,
        "cluster_unit": "target_id_within_capture_block",
        "resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
        "seed": DEFAULT_BOOTSTRAP_SEED,
        "interval_quantiles": [0.025, 0.975],
        "independent_population_interval_claimed": False,
    }:
        raise MeasurementScheduleError("analysis bootstrap contract changed")
    return payload, digest


def _mean(values: Sequence[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _percentile(values: Sequence[float], probability: float) -> float | None:
    """R-7 linear percentile, matching NumPy's default linear convention."""

    if not values:
        return None
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _median_absolute_deviation(values: Sequence[float]) -> float | None:
    median = _percentile(values, 0.5)
    if median is None:
        return None
    return _percentile([abs(value - median) for value in values], 0.5)


def _error_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row["prediction_success"] is True]
    errors = [float(row["spatial_error_px"]) for row in successes]
    dx = [float(row["signed_error_x_px"]) for row in successes]
    dy = [float(row["signed_error_y_px"]) for row in successes]
    attempted = len(rows)

    cluster_errors: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in successes:
        cluster_errors[(str(row["block_id"]), str(row["target_id"]))].append(
            float(row["spatial_error_px"])
        )
    cluster_means = [statistics.fmean(values) for values in cluster_errors.values()]
    return {
        "attempted_count": attempted,
        "successful_count": len(successes),
        "prediction_success_fraction": (
            len(successes) / attempted if attempted else None
        ),
        "covered_target_cluster_count": len(cluster_means),
        "target_macro_mean_spatial_error_px": _mean(cluster_means),
        "mean_spatial_error_px": _mean(errors),
        "median_spatial_error_px": _percentile(errors, 0.5),
        "p90_spatial_error_px": _percentile(errors, 0.9),
        "signed_mean_error_x_px": _mean(dx),
        "signed_mean_error_y_px": _mean(dy),
        "median_absolute_error_x_px": _percentile(
            [abs(value) for value in dx], 0.5
        ),
        "median_absolute_error_y_px": _percentile(
            [abs(value) for value in dy], 0.5
        ),
    }


def _sha256_index(seed: int, replicate: int, draw: int, population: int) -> int:
    material = f"{seed}:{replicate}:{draw}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % population


def _target_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    clusters: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["prediction_success"] is True:
            clusters[(str(row["block_id"]), str(row["target_id"]))].append(
                float(row["spatial_error_px"])
            )
    ordered = sorted(clusters)
    cluster_means = [statistics.fmean(clusters[key]) for key in ordered]
    if not cluster_means:
        return {
            "status": "not_evaluable_no_successful_target_clusters",
            "unit": "target_id_within_capture_block",
            "algorithm": BOOTSTRAP_ALGORITHM,
            "resamples": resamples,
            "seed": seed,
            "cluster_count": 0,
            "point_estimate_px": None,
            "ci95_low_px": None,
            "ci95_high_px": None,
        }
    replicates: list[float] = []
    for replicate in range(resamples):
        sampled = [
            cluster_means[
                _sha256_index(seed, replicate, draw, len(cluster_means))
            ]
            for draw in range(len(cluster_means))
        ]
        replicates.append(statistics.fmean(sampled))
    return {
        "status": "descriptive_cluster_interval",
        "unit": "target_id_within_capture_block",
        "algorithm": BOOTSTRAP_ALGORITHM,
        "resamples": resamples,
        "seed": seed,
        "cluster_count": len(cluster_means),
        "point_estimate_px": statistics.fmean(cluster_means),
        "ci95_low_px": _percentile(replicates, 0.025),
        "ci95_high_px": _percentile(replicates, 0.975),
        "independent_population_interval_claimed": False,
    }


def _nearest_axis_index(value: float, reference: Sequence[float]) -> int:
    return min(
        range(len(reference)),
        key=lambda index: (abs(value - reference[index]), index),
    )


def _region_metrics(
    rows: Sequence[Mapping[str, Any]],
    evaluation_points: Sequence[Sequence[float]],
) -> dict[str, Any]:
    xs = sorted({float(point[0]) for point in evaluation_points})
    ys = sorted({float(point[1]) for point in evaluation_points})
    if len(xs) != 4 or len(ys) != 4:
        raise MeasurementScheduleError("evaluation grid must be exactly 4x4")
    point_to_region = {
        (float(x), float(y)): y_index * 4 + x_index
        for y_index, y in enumerate(ys)
        for x_index, x in enumerate(xs)
    }
    confusion = [[0 for _ in range(16)] for _ in range(16)]
    correct = 0
    classified = 0
    row_predictions: dict[int, int] = {}
    actual_regions: dict[int, int] = {}
    for row in rows:
        if row["prediction_success"] is not True:
            continue
        viewport_width = float(row["viewport_width"])
        viewport_height = float(row["viewport_height"])
        predicted_fraction_x = float(row["predicted_x_px"]) / viewport_width
        predicted_fraction_y = float(row["predicted_y_px"]) / viewport_height
        predicted = (
            _nearest_axis_index(predicted_fraction_y, ys) * 4
            + _nearest_axis_index(predicted_fraction_x, xs)
        )
        target_fraction = (
            float(row["target_x_px"]) / viewport_width,
            float(row["target_y_px"]) / viewport_height,
        )
        actual = min(
            point_to_region,
            key=lambda point: (
                math.dist(point, target_fraction),
                point_to_region[point],
            ),
        )
        actual_index = point_to_region[actual]
        confusion[actual_index][predicted] += 1
        classified += 1
        correct += int(actual_index == predicted)
        sequence_index = int(row["sequence_index"])
        row_predictions[sequence_index] = predicted
        actual_regions[sequence_index] = actual_index

    # Frozen negative control: cyclically rotate target labels by one region.
    permuted_correct = sum(
        int(row_predictions[index] == (actual_regions[index] + 1) % 16)
        for index in sorted(row_predictions)
    )
    observed_accuracy = correct / classified if classified else None
    permuted_accuracy = permuted_correct / classified if classified else None
    return {
        "decoder": "nearest_axis_on_frozen_4x4_target_grid_v1",
        "classified_successful_count": classified,
        "accuracy": observed_accuracy,
        "confusion_matrix": confusion,
        "target_label_cyclic_permutation_negative_control": {
            "permutation": "actual_region_plus_1_mod_16",
            "accuracy": permuted_accuracy,
            "status": (
                "passed_observed_exceeds_permuted"
                if observed_accuracy is not None
                and permuted_accuracy is not None
                and observed_accuracy > permuted_accuracy
                else "negative_result_not_demonstrated"
            ),
            "model_or_threshold_selection_allowed": False,
        },
    }


def _neutral_drift(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_block_target: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if (
            row["prediction_success"] is True
            and row["block_id"] in {"neutral_start", "neutral_end"}
        ):
            by_block_target[(str(row["block_id"]), str(row["target_id"]))].append(
                (float(row["predicted_x_px"]), float(row["predicted_y_px"]))
            )
    targets = sorted(
        {
            target_id
            for block_id, target_id in by_block_target
            if block_id in {"neutral_start", "neutral_end"}
        }
    )
    per_target: list[dict[str, Any]] = []
    for target_id in targets:
        start = by_block_target.get(("neutral_start", target_id), [])
        end = by_block_target.get(("neutral_end", target_id), [])
        if not start or not end:
            per_target.append(
                {
                    "target_id": target_id,
                    "status": "not_evaluable_missing_successful_phase",
                    "start_success_count": len(start),
                    "end_success_count": len(end),
                }
            )
            continue
        start_x = statistics.fmean(value[0] for value in start)
        start_y = statistics.fmean(value[1] for value in start)
        end_x = statistics.fmean(value[0] for value in end)
        end_y = statistics.fmean(value[1] for value in end)
        drift_x = end_x - start_x
        drift_y = end_y - start_y
        per_target.append(
            {
                "target_id": target_id,
                "status": "available",
                "start_success_count": len(start),
                "end_success_count": len(end),
                "start_prediction_centroid_px": [start_x, start_y],
                "end_prediction_centroid_px": [end_x, end_y],
                "drift_vector_px": [drift_x, drift_y],
                "drift_magnitude_px": math.hypot(drift_x, drift_y),
            }
        )
    magnitudes = [
        float(item["drift_magnitude_px"])
        for item in per_target
        if item["status"] == "available"
    ]
    return {
        "status": "available" if magnitudes else "not_evaluable",
        "available_target_count": len(magnitudes),
        "expected_target_count": 16,
        "target_macro_mean_drift_magnitude_px": _mean(magnitudes),
        "median_drift_magnitude_px": _percentile(magnitudes, 0.5),
        "p90_drift_magnitude_px": _percentile(magnitudes, 0.9),
        "per_target": per_target,
    }


def _timing_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["inference_latency_ms"]) for row in rows]
    intervals: list[float] = []
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["block_id"]), int(row["repeat_index"]))].append(row)
    for group_rows in groups.values():
        ordered = sorted(group_rows, key=lambda row: int(row["sequence_index"]))
        captures = [float(row["frame_capture_monotonic_ms"]) for row in ordered]
        intervals.extend(
            later - earlier for earlier, later in zip(captures, captures[1:])
        )
    median_interval = _percentile(intervals, 0.5)
    return {
        "inference_latency_ms": {
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
            "median_absolute_deviation": _median_absolute_deviation(latencies),
        },
        "within_block_inference_start_interval_ms": {
            "count": len(intervals),
            "p50": median_interval,
            "p95": _percentile(intervals, 0.95),
            "median_absolute_deviation": _median_absolute_deviation(intervals),
        },
        "effective_within_block_inference_start_rate_hz": (
            1000.0 / median_interval
            if median_interval is not None and median_interval > 0
            else None
        ),
        "block_transition_intervals_excluded": True,
        "camera_exposure_timestamp_available": False,
        "camera_capture_jitter": {
            "status": "not_evaluable_without_trusted_exposure_timestamp",
            "v1_sample_field_limitation": (
                "frame_capture_monotonic_ms is constrained by the frozen v1 "
                "artifact equation to equal the inference-start timestamp"
            ),
        },
    }


def _uncertainty_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row["prediction_success"] is True]
    scored = [row for row in successes if row.get("sensor_uncertainty_score") is not None]
    if not successes:
        return {
            "status": "not_evaluable_no_successful_predictions",
            "threshold_selected": False,
            "coverage_risk": [],
            "deciles": [],
        }
    if len(scored) != len(successes):
        return {
            "status": "not_evaluable_incomplete_successful_prediction_scores",
            "successful_count": len(successes),
            "scored_count": len(scored),
            "threshold_selected": False,
            "coverage_risk": [],
            "deciles": [],
        }
    ordered = sorted(
        scored,
        key=lambda row: (
            float(row["sensor_uncertainty_score"]),
            int(row["sequence_index"]),
        ),
    )
    coverage_risk: list[dict[str, Any]] = []
    for coverage in UNCERTAINTY_COVERAGES:
        keep_count = max(1, int(math.ceil(coverage * len(ordered))))
        retained = ordered[:keep_count]
        retained_indices = {int(row["sequence_index"]) for row in retained}
        for row in rows:
            row.setdefault("uncertainty_hypothetical_retained", {})[
                f"{coverage:.1f}"
            ] = (
                int(row["sequence_index"]) in retained_indices
                if row["prediction_success"] is True
                else False
            )
        summary = _error_summary(retained)
        coverage_risk.append(
            {
                "requested_coverage": coverage,
                "retained_count": keep_count,
                "realized_conditional_coverage": keep_count / len(ordered),
                "observed_fixed_coverage_cutoff_score": float(
                    retained[-1]["sensor_uncertainty_score"]
                ),
                "mean_spatial_error_px": summary["mean_spatial_error_px"],
                "target_macro_mean_spatial_error_px": summary[
                    "target_macro_mean_spatial_error_px"
                ],
                "hypothetical_only": True,
                "deployable_threshold_selected": False,
            }
        )
    deciles: list[dict[str, Any]] = []
    for decile in range(10):
        start = math.floor(decile * len(ordered) / 10)
        end = math.floor((decile + 1) * len(ordered) / 10)
        bucket = ordered[start:end]
        if not bucket:
            continue
        deciles.append(
            {
                "decile": decile + 1,
                "count": len(bucket),
                "score_min": float(bucket[0]["sensor_uncertainty_score"]),
                "score_max": float(bucket[-1]["sensor_uncertainty_score"]),
                "mean_spatial_error_px": _mean(
                    [float(row["spatial_error_px"]) for row in bucket]
                ),
            }
        )
    return {
        "status": "scored_no_threshold_descriptive",
        "successful_count": len(successes),
        "scored_count": len(scored),
        "ranking_direction": "lower_score_retained_first",
        "coverage_grid": list(UNCERTAINTY_COVERAGES),
        "coverage_risk": coverage_risk,
        "deciles": deciles,
        "threshold_selected": False,
        "abstention_policy_changed": False,
    }


def _derived_rows(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        success = sample["prediction_success"] is True
        row = {
            "sequence_index": int(sample["sequence_index"]),
            "block_id": str(sample["block_id"]),
            "block_role": str(sample["block_role"]),
            "repeat_index": int(sample["repeat_index"]),
            "target_id": str(sample["target_id"]),
            "target_x_px": float(sample["target_x_px"]),
            "target_y_px": float(sample["target_y_px"]),
            "viewport_width": float(sample["viewport_width"]),
            "viewport_height": float(sample["viewport_height"]),
            "prediction_success": success,
            "predicted_x_px": (
                float(sample["predicted_x_px"])
                if sample.get("predicted_x_px") is not None
                else None
            ),
            "predicted_y_px": (
                float(sample["predicted_y_px"])
                if sample.get("predicted_y_px") is not None
                else None
            ),
            "inference_latency_ms": float(sample["inference_latency_ms"]),
            "frame_capture_monotonic_ms": float(
                sample["frame_capture_monotonic_ms"]
            ),
            "model_id": str(sample["model_id"]),
            "model_sha256": str(sample["model_sha256"]),
        }
        if success:
            dx = float(sample["predicted_x_px"]) - float(sample["target_x_px"])
            dy = float(sample["predicted_y_px"]) - float(sample["target_y_px"])
            row.update(
                {
                    "signed_error_x_px": dx,
                    "signed_error_y_px": dy,
                    "spatial_error_px": math.hypot(dx, dy),
                }
            )
        else:
            row.update(
                {
                    "signed_error_x_px": None,
                    "signed_error_y_px": None,
                    "spatial_error_px": None,
                }
            )
        if sample.get("sensor_uncertainty_score") is not None:
            row["sensor_uncertainty_score"] = _finite(
                sample["sensor_uncertainty_score"],
                field=f"sample {row['sequence_index']} sensor_uncertainty_score",
            )
        else:
            row["sensor_uncertainty_score"] = None
        rows.append(row)
    return rows


def analyze_measurement_run(
    artifact: Mapping[str, Any],
    *,
    protocol_path: str | Path | None = None,
    analysis_definition_path: str | Path | None = None,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Verify and descriptively analyze one frozen acquisition artifact."""

    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or bootstrap_resamples <= 0
    ):
        raise MeasurementScheduleError(
            "bootstrap_resamples must be a positive integer"
        )
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise MeasurementScheduleError("bootstrap_seed must be an integer")
    verified = verify_capture_artifact(artifact, protocol_path=protocol_path)
    protocol, protocol_sha256 = load_frozen_protocol(protocol_path)
    if protocol_sha256 != verified["protocol_sha256"]:
        raise MeasurementScheduleError(
            "analysis protocol digest differs from verified capture artifact"
        )
    analysis_definition, analysis_definition_sha256 = (
        load_frozen_analysis_definition(analysis_definition_path)
    )
    source_protocol = dict(analysis_definition.get("source_protocol") or {})
    if source_protocol.get("protocol_id") != protocol.get("protocol_id") or (
        source_protocol.get("canonical_sha256") != protocol_sha256
    ):
        raise MeasurementScheduleError(
            "analysis definition is not bound to the verified capture protocol"
        )
    samples = artifact.get("samples")
    if not isinstance(samples, list):
        raise MeasurementScheduleError("verified artifact samples are unavailable")
    rows = _derived_rows(samples)
    evaluation_rows = [row for row in rows if row["block_role"] == "evaluation"]
    calibration_rows = [row for row in rows if row["block_role"] == "calibration"]

    uncertainty = _uncertainty_metrics(evaluation_rows)
    evaluation_summary = _error_summary(evaluation_rows)
    calibration_summary = _error_summary(calibration_rows)
    blocks = {
        block_id: _error_summary(
            [row for row in evaluation_rows if row["block_id"] == block_id]
        )
        for block_id in sorted({str(row["block_id"]) for row in evaluation_rows})
    }

    layout = dict(dict(protocol.get("metrics") or {}).get("layout_reference") or {})
    line_gap = _finite(layout.get("line_gap_px"), field="layout line_gap_px")
    word_width = _finite(
        layout.get("median_word_width_px"), field="layout median_word_width_px"
    )
    median_error = evaluation_summary["median_spatial_error_px"]
    p90_error = evaluation_summary["p90_spatial_error_px"]
    median_abs_x = evaluation_summary["median_absolute_error_x_px"]
    median_abs_y = evaluation_summary["median_absolute_error_y_px"]

    evaluation_points = dict(dict(protocol.get("targets") or {}).get("evaluation") or {}).get(
        "points"
    )
    if not isinstance(evaluation_points, list):
        raise MeasurementScheduleError("frozen evaluation target points unavailable")

    baseline_rows: list[dict[str, Any]] = []
    for row in evaluation_rows:
        center_x = float(row["viewport_width"]) * 0.5
        center_y = float(row["viewport_height"]) * 0.5
        dx = center_x - float(row["target_x_px"])
        dy = center_y - float(row["target_y_px"])
        baseline_rows.append(
            {
                **deepcopy(row),
                "prediction_success": True,
                "predicted_x_px": center_x,
                "predicted_y_px": center_y,
                "signed_error_x_px": dx,
                "signed_error_y_px": dy,
                "spatial_error_px": math.hypot(dx, dy),
            }
        )

    model_bindings = {
        role: {
            "model_id": role_rows[0]["model_id"],
            "model_sha256": role_rows[0]["model_sha256"],
        }
        for role, role_rows in (
            ("calibration", calibration_rows),
            ("evaluation", evaluation_rows),
        )
        if role_rows
    }
    result: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_type": ANALYSIS_TYPE,
        "status": (
            "synthetic_diagnostic_only"
            if verified["evidence_class"] == "dry_run_synthetic"
            else "descriptive_metrics_only_pending_runner_provenance"
        ),
        "evidence_class": verified["evidence_class"],
        "capture_run_id": verified["capture_run_id"],
        "protocol_sha256": verified["protocol_sha256"],
        "run_manifest_sha256": verified["run_manifest_sha256"],
        "capture_artifact_sha256": verified["artifact_sha256"],
        "analysis_definition_sha256": analysis_definition_sha256,
        "analysis_contract": {
            "percentile_algorithm": PERCENTILE_ALGORITHM,
            "bootstrap_algorithm": BOOTSTRAP_ALGORITHM,
            "bootstrap_resamples": int(bootstrap_resamples),
            "bootstrap_seed": bootstrap_seed,
            "uncertainty_coverages": list(UNCERTAINTY_COVERAGES),
            "matches_frozen_default_configuration": (
                bootstrap_resamples == DEFAULT_BOOTSTRAP_RESAMPLES
                and bootstrap_seed == DEFAULT_BOOTSTRAP_SEED
            ),
            "geometry_inputs_only": True,
            "text_cursor_cognitive_or_reading_inputs_accepted": False,
        },
        "claim_boundary": {
            "measurement_claim_authorized": False,
            "quality_band_change_authorized": False,
            "threshold_selected": False,
            "natural_reading_word_or_line_accuracy_claimed": False,
            "population_or_device_generalization_claimed": False,
            "candidate_model_selected_from_evaluation_targets": False,
            "runner_capture_and_training_provenance_verified_here": False,
            "analysis_integrity_scope": (
                "verified frozen acquisition artifact schedule, fields, hashes, "
                "and geometry only"
            ),
        },
        "model_bindings": model_bindings,
        "coverage": {
            "all_attempts": verified["sample_count"],
            "all_successes": verified["successful_sample_count"],
            "calibration_frozen_base_on_fit_targets": calibration_summary,
            "evaluation": evaluation_summary,
        },
        "evaluation": {
            "selected_personal_model": evaluation_summary,
            "viewport_center_baseline_b0": _error_summary(baseline_rows),
            "candidate_comparison_boundary": (
                "B0 has full coverage; selected-model error is conditional on "
                "successful sensor predictions and coverage is reported separately"
            ),
            "by_block": blocks,
            "target_region_4x4": _region_metrics(
                evaluation_rows, evaluation_points
            ),
            "neutral_start_to_end_drift": _neutral_drift(evaluation_rows),
            "target_cluster_bootstrap": _target_cluster_bootstrap(
                evaluation_rows,
                resamples=int(bootstrap_resamples),
                seed=bootstrap_seed,
            ),
        },
        "layout_normalized_resolution": {
            "line_gap_px": line_gap,
            "median_word_width_px": word_width,
            "median_spatial_error_in_line_gaps": (
                median_error / line_gap if median_error is not None else None
            ),
            "p90_spatial_error_in_line_gaps": (
                p90_error / line_gap if p90_error is not None else None
            ),
            "median_spatial_error_in_word_widths": (
                median_error / word_width if median_error is not None else None
            ),
            "p90_spatial_error_in_word_widths": (
                p90_error / word_width if p90_error is not None else None
            ),
            "median_absolute_x_error_in_word_widths": (
                median_abs_x / word_width if median_abs_x is not None else None
            ),
            "median_absolute_y_error_in_line_gaps": (
                median_abs_y / line_gap if median_abs_y is not None else None
            ),
            "natural_reading_reference_available": False,
        },
        "timing": _timing_metrics(rows),
        "uncertainty": uncertainty,
        "negative_controls": {
            "target_label_permutation": _region_metrics(
                evaluation_rows, evaluation_points
            )["target_label_cyclic_permutation_negative_control"],
            "text_prior_mutation": "not_applicable_no_text_input_surface",
            "cursor_reference": "not_applicable_cursor_not_accepted",
            "evaluation_target_fitting": (
                "requires_runner_training_provenance_verification"
            ),
        },
        "derived_rows": rows,
        "decision": {
            "finest_supported_resolution_band": None,
            "reason": (
                "the frozen v1 run reports geometry ratios and fixed coverage-risk "
                "descriptively; no product threshold or natural-reading reference "
                "is authorized"
            ),
            "next_step": (
                "preserve this result, then confirm any proposed threshold on a "
                "new participant/device capture frozen before inspection"
            ),
        },
    }
    result["analysis_sha256"] = canonical_sha256(result)
    return result


def _validated_live_runner_evidence(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Recheck the frame-free bundle returned by an authenticated live runner.

    This is deliberately private.  A persisted JSON file plus a recomputed
    checksum is not an authority: the public entrypoint below first asks the
    authenticated runner/store to re-read its ledgers, model, purge proof, and
    sealed artifacts in the same process.
    """

    if not isinstance(evidence, Mapping):
        raise MeasurementScheduleError("live runner evidence must be an object")
    bundle = deepcopy(dict(evidence))
    expected_fields = {
        "schema_version",
        "evidence_type",
        "capture_run_id",
        "verification",
        "status",
        "capture_artifact",
        "attempt_sidecar",
        "training_provenance",
        "model_sha256",
        "calibration_image_purge",
        "spool_absence_verified",
        "raw_frames_included",
        "run_token_included",
        "measurement_claim_authorized",
        "physical_capture_claim_authorized",
        "evidence_sha256",
    }
    if set(bundle) != expected_fields:
        raise MeasurementScheduleError("live runner evidence fields changed")
    if (
        bundle.get("schema_version") != 1
        or bundle.get("evidence_type")
        != "webcam_gaze_measurement_ceiling_verified_analysis_evidence_v1"
    ):
        raise MeasurementScheduleError("live runner evidence identity changed")
    stored_evidence_sha = str(bundle.get("evidence_sha256") or "")
    evidence_core = deepcopy(bundle)
    evidence_core.pop("evidence_sha256", None)
    if stored_evidence_sha != canonical_sha256(evidence_core):
        raise MeasurementScheduleError("live runner evidence SHA-256 mismatch")
    for field in (
        "spool_absence_verified",
    ):
        if bundle.get(field) is not True:
            raise MeasurementScheduleError(f"live runner {field} is not verified")
    for field in (
        "raw_frames_included",
        "run_token_included",
        "measurement_claim_authorized",
        "physical_capture_claim_authorized",
    ):
        if bundle.get(field) is not False:
            raise MeasurementScheduleError(f"live runner {field} boundary changed")

    artifact = bundle.get("capture_artifact")
    if not isinstance(artifact, Mapping):
        raise MeasurementScheduleError("live runner capture artifact is unavailable")
    verified = verify_capture_artifact(artifact)
    run_id = str(bundle.get("capture_run_id") or "")
    if run_id != verified["capture_run_id"]:
        raise MeasurementScheduleError("live runner capture run changed")

    verification = bundle.get("verification")
    status = bundle.get("status")
    if not isinstance(verification, Mapping) or not isinstance(status, Mapping):
        raise MeasurementScheduleError("live runner verification state is unavailable")
    if (
        verification.get("phase") != "artifact_verified"
        or verification.get("acquisition_artifact_verified") is not True
        or verification.get("capture_contract_binding_verified") is not True
        or verification.get("artifact_sha256") != verified["artifact_sha256"]
        or verification.get("protocol_sha256") != verified["protocol_sha256"]
    ):
        raise MeasurementScheduleError("live runner artifact verification changed")
    if (
        status.get("capture_run_id") != run_id
        or status.get("phase") != "artifact_verified"
        or status.get("acquisition_artifact_verified") is not True
        or status.get("capture_contract_binding_verified") is not True
        or status.get("protocol_sha256") != verified["protocol_sha256"]
        or status.get("manifest_sha256") != verified["run_manifest_sha256"]
        or status.get("measurement_claim_authorized") is not False
        or status.get("physical_capture_claim_authorized") is not False
    ):
        raise MeasurementScheduleError("live runner public status changed")
    progress = status.get("progress")
    if not isinstance(progress, Mapping) or (
        progress.get("next_sequence_index") != 193
        or progress.get("calibration_count") != 65
        or progress.get("evaluation_count") != 128
    ):
        raise MeasurementScheduleError("live runner progress is not exactly 193 rows")

    capture_metadata = status.get("capture_artifact")
    if not isinstance(capture_metadata, Mapping) or (
        capture_metadata.get("artifact_sha256") != verified["artifact_sha256"]
        or capture_metadata.get("sample_count") != 193
    ):
        raise MeasurementScheduleError("live runner sealed artifact binding changed")

    sidecar = bundle.get("attempt_sidecar")
    if not isinstance(sidecar, Mapping):
        raise MeasurementScheduleError("live runner attempt sidecar is unavailable")
    sidecar_object = deepcopy(dict(sidecar))
    sidecar_sha = str(sidecar_object.pop("sidecar_sha256", ""))
    if sidecar_sha != canonical_sha256(sidecar_object):
        raise MeasurementScheduleError("live runner attempt sidecar SHA-256 mismatch")
    entries = sidecar.get("entries")
    if (
        sidecar.get("schema_version") != 1
        or sidecar.get("sidecar_type")
        != "webcam_gaze_measurement_ceiling_attempt_sidecar_v1"
        or sidecar.get("capture_run_id") != run_id
        or sidecar.get("protocol_sha256") != verified["protocol_sha256"]
        or sidecar.get("manifest_sha256") != verified["run_manifest_sha256"]
        or sidecar.get("capture_artifact_sha256") != verified["artifact_sha256"]
        or not isinstance(entries, list)
        or len(entries) != 193
        or sidecar.get("entries_sha256") != canonical_sha256(entries)
        or sidecar.get("measurement_claim_authorized") is not False
        or sidecar.get("physical_capture_claim_authorized") is not False
        or capture_metadata.get("attempt_sidecar_sha256") != sidecar_sha
        or capture_metadata.get("attempt_sidecar_entries_sha256")
        != sidecar.get("entries_sha256")
    ):
        raise MeasurementScheduleError("live runner attempt sidecar binding changed")

    samples = artifact.get("samples")
    if not isinstance(samples, list) or len(samples) != len(entries):
        raise MeasurementScheduleError("live runner sample/sidecar count changed")
    no_face_count = 0
    for index, (entry, sample) in enumerate(zip(entries, samples, strict=True)):
        if not isinstance(entry, Mapping) or not isinstance(sample, Mapping):
            raise MeasurementScheduleError("live runner sidecar row is invalid")
        expected_role = "calibration" if index < 65 else "evaluation"
        success = sample.get("prediction_success")
        if type(success) is not bool:  # noqa: E721 - reject bool/int coercion
            raise MeasurementScheduleError("live runner prediction outcome is invalid")
        failure_code = None if success else "no_face_detected"
        if (
            entry.get("sequence_index") != index
            or entry.get("ledger_role") != expected_role
            or entry.get("prediction_success") is not success
            or entry.get("failure_code") != failure_code
            or entry.get("sample_sha256") != canonical_sha256(sample)
        ):
            raise MeasurementScheduleError("live runner sidecar sample binding changed")
        for field in (
            "ledger_record_sha256",
            "capture_contract_evidence_sha256",
            "server_timing_evidence_sha256",
            "frame_sha256",
        ):
            value = str(entry.get(field) or "")
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise MeasurementScheduleError(f"live runner {field} is malformed")
        timing = entry.get("server_timing_evidence")
        if not isinstance(timing, Mapping) or (
            canonical_sha256(timing) != entry.get("server_timing_evidence_sha256")
        ):
            raise MeasurementScheduleError("live runner timing evidence changed")
        if not success:
            no_face_count += 1

    provenance = bundle.get("training_provenance")
    if not isinstance(provenance, Mapping):
        raise MeasurementScheduleError("live runner training provenance is unavailable")
    calibration_rows = [
        row for row in build_run_manifest(run_id)["rows"]
        if row["block_role"] == "calibration"
    ]
    evaluation_rows = [
        row for row in build_run_manifest(run_id)["rows"]
        if row["block_role"] == "evaluation"
    ]
    calibration_targets = sorted({str(row["target_id"]) for row in calibration_rows})
    evaluation_targets = sorted({str(row["target_id"]) for row in evaluation_rows})
    if (
        provenance.get("schema_version") != 1
        or provenance.get("provenance_type")
        != "webcam_gaze_measurement_ceiling_training_v1"
        or provenance.get("capture_run_id") != run_id
        or provenance.get("protocol_sha256") != verified["protocol_sha256"]
        or provenance.get("manifest_sha256") != verified["run_manifest_sha256"]
        or provenance.get("calibration_ordinals") != list(range(65))
        or provenance.get("training_role") != "calibration_only"
        or provenance.get("train_samples") != 65
        or provenance.get("allow_cuda") is not False
        or provenance.get("training_device_required") != "cpu"
        or provenance.get("evaluation_labels_used") is not False
        or provenance.get("evaluation_rows_used") != 0
        or provenance.get("evaluation_targets_excluded") is not True
        or provenance.get("calibration_evaluation_target_intersection_count") != 0
        or provenance.get("text_cursor_cognitive_inputs_used") is not False
        or provenance.get("measurement_claim_authorized") is not False
        or provenance.get("calibration_schedule_rows_sha256")
        != canonical_sha256(calibration_rows)
        or provenance.get("evaluation_schedule_rows_sha256")
        != canonical_sha256(evaluation_rows)
        or provenance.get("calibration_target_ids_sha256")
        != canonical_sha256(calibration_targets)
        or provenance.get("evaluation_target_ids_sha256")
        != canonical_sha256(evaluation_targets)
    ):
        raise MeasurementScheduleError("live runner training provenance changed")
    provenance_sha = canonical_sha256(provenance)
    runner_status = status.get("runner")
    trained = runner_status.get("trained_artifact") if isinstance(runner_status, Mapping) else None
    model_binding = status.get("model_binding")
    if (
        not isinstance(trained, Mapping)
        or not isinstance(model_binding, Mapping)
        or trained.get("training_provenance_sha256") != provenance_sha
        or trained.get("model_sha256") != bundle.get("model_sha256")
        or model_binding.get("model_sha256") != bundle.get("model_sha256")
        or trained.get("calibration_ledger_sha256")
        != provenance.get("calibration_ledger_sha256")
    ):
        raise MeasurementScheduleError("live runner trained model binding changed")

    purge = bundle.get("calibration_image_purge")
    status_purge = runner_status.get("calibration_image_purge") if isinstance(runner_status, Mapping) else None
    if (
        not isinstance(purge, Mapping)
        or purge != status_purge
        or purge.get("status") != "verified"
        or purge.get("postcondition_verified") is not True
    ):
        raise MeasurementScheduleError("live runner calibration purge changed")
    return {
        "evidence_sha256": stored_evidence_sha,
        "attempt_sidecar_sha256": sidecar_sha,
        "attempt_sidecar_entries_sha256": sidecar.get("entries_sha256"),
        "training_provenance_sha256": provenance_sha,
        "model_sha256": bundle.get("model_sha256"),
        "no_face_attempt_count": no_face_count,
        "capture_contract_proof_count": runner_status.get(
            "capture_contract_proof_count"
        ) if isinstance(runner_status, Mapping) else None,
    }


def _analyze_reverified_live_evidence(
    evidence: Mapping[str, Any],
    *,
    protocol_path: str | Path | None = None,
    analysis_definition_path: str | Path | None = None,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Analyze evidence already re-read by the canonical authenticated runner.

    This is intentionally private.  The public authority-bearing entry point
    lives on ``MeasurementRunner`` so an arbitrary duck-typed object cannot
    self-assert that a rehashed bundle came from the live store.
    """

    verified_live = _validated_live_runner_evidence(evidence)
    result = analyze_measurement_run(
        evidence["capture_artifact"],
        protocol_path=protocol_path,
        analysis_definition_path=analysis_definition_path,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    result["status"] = "integrity_verified_descriptive_live_runner"
    result["live_runner_provenance"] = {
        **verified_live,
        "verification_mode": "authenticated_live_runner_store_revalidation",
        "persisted_bundle_self_attestation_accepted": False,
    }
    result["claim_boundary"][
        "runner_capture_and_training_provenance_verified_here"
    ] = True
    result["claim_boundary"]["analysis_integrity_scope"] = (
        "authenticated live runner revalidation of the frozen artifact, attempt "
        "sidecar, capture contract, training-only provenance, model binding, and "
        "calibration-image purge; no accuracy or population claim is authorized"
    )
    result["negative_controls"]["evaluation_target_fitting"] = (
        "verified_excluded_by_authenticated_live_runner_training_provenance"
    )
    result["analysis_sha256"] = canonical_sha256(
        {key: value for key, value in result.items() if key != "analysis_sha256"}
    )
    return result


def _markdown_number(value: object, digits: int = 2) -> str:
    if value is None:
        return "not evaluable"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def render_measurement_run_markdown(result: Mapping[str, Any]) -> str:
    """Render a compact report without embedding raw sensor rows."""

    selected = dict(dict(result.get("evaluation") or {}).get("selected_personal_model") or {})
    baseline = dict(dict(result.get("evaluation") or {}).get("viewport_center_baseline_b0") or {})
    bootstrap = dict(dict(result.get("evaluation") or {}).get("target_cluster_bootstrap") or {})
    region = dict(dict(result.get("evaluation") or {}).get("target_region_4x4") or {})
    drift = dict(dict(result.get("evaluation") or {}).get("neutral_start_to_end_drift") or {})
    layout = dict(result.get("layout_normalized_resolution") or {})
    timing = dict(result.get("timing") or {})
    latency = dict(timing.get("inference_latency_ms") or {})
    uncertainty = dict(result.get("uncertainty") or {})
    blocks = dict(dict(result.get("evaluation") or {}).get("by_block") or {})
    lines = [
        "# Webcam gaze measurement-ceiling 193-row analysis",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Evidence class: `{result.get('evidence_class')}`",
        f"- Frozen protocol SHA-256: `{result.get('protocol_sha256')}`",
        (
            "- Frozen analysis-definition SHA-256: "
            f"`{result.get('analysis_definition_sha256')}`"
        ),
        f"- Capture-artifact SHA-256: `{result.get('capture_artifact_sha256')}`",
        f"- Analysis SHA-256: `{result.get('analysis_sha256')}`",
        "",
        "This is engineering/self-development geometry evidence only. It does "
        "not authorize a quality band, an abstention threshold, natural-reading "
        "word or line accuracy, or participant/device generalization.",
        "",
        "## Evaluation geometry",
        "",
        (
            "| Candidate | Attempts | Success | Success fraction | "
            "Target-macro mean px | Median px | P90 px |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            "| Selected personal model | "
            f"{selected.get('attempted_count', 0)} | "
            f"{selected.get('successful_count', 0)} | "
            f"{_markdown_number(selected.get('prediction_success_fraction'), 4)} | "
            f"{_markdown_number(selected.get('target_macro_mean_spatial_error_px'))} | "
            f"{_markdown_number(selected.get('median_spatial_error_px'))} | "
            f"{_markdown_number(selected.get('p90_spatial_error_px'))} |"
        ),
        (
            "| Viewport-center B0 | "
            f"{baseline.get('attempted_count', 0)} | "
            f"{baseline.get('successful_count', 0)} | "
            f"{_markdown_number(baseline.get('prediction_success_fraction'), 4)} | "
            f"{_markdown_number(baseline.get('target_macro_mean_spatial_error_px'))} | "
            f"{_markdown_number(baseline.get('median_spatial_error_px'))} | "
            f"{_markdown_number(baseline.get('p90_spatial_error_px'))} |"
        ),
        "",
        (
            "B0 has full coverage; selected-model error is conditional on "
            "successful sensor predictions. Coverage must be read with error."
        ),
        "",
        "## Block diagnostics",
        "",
        "| Block | Success fraction | Target-macro mean px | Median px | P90 px |",
        "|---|---:|---:|---:|---:|",
    ]
    for block_id, summary_value in sorted(blocks.items()):
        summary = dict(summary_value or {})
        lines.append(
            f"| {block_id} | "
            f"{_markdown_number(summary.get('prediction_success_fraction'), 4)} | "
            f"{_markdown_number(summary.get('target_macro_mean_spatial_error_px'))} | "
            f"{_markdown_number(summary.get('median_spatial_error_px'))} | "
            f"{_markdown_number(summary.get('p90_spatial_error_px'))} |"
        )
    lines.extend(
        [
            "",
            "## Resolution and drift",
            "",
            (
                "- Frozen 4x4 target-region accuracy: "
                f"`{_markdown_number(region.get('accuracy'), 4)}`."
            ),
            (
                "- Neutral start-to-end target-macro mean drift magnitude: "
                f"`{_markdown_number(drift.get('target_macro_mean_drift_magnitude_px'))} px` "
                f"across `{drift.get('available_target_count', 0)}/16` targets."
            ),
            (
                "- Median / P90 spatial error in fixed-layout line gaps: "
                f"`{_markdown_number(layout.get('median_spatial_error_in_line_gaps'))}` / "
                f"`{_markdown_number(layout.get('p90_spatial_error_in_line_gaps'))}`."
            ),
            (
                "- Median / P90 spatial error in median word widths: "
                f"`{_markdown_number(layout.get('median_spatial_error_in_word_widths'))}` / "
                f"`{_markdown_number(layout.get('p90_spatial_error_in_word_widths'))}`."
            ),
            "- These layout ratios are descriptive; no natural-reading ground truth is available.",
            "",
            "## Timing and uncertainty",
            "",
            (
                "- CPU inference latency P50 / P95: "
                f"`{_markdown_number(latency.get('p50'))}` / "
                f"`{_markdown_number(latency.get('p95'))} ms`."
            ),
            (
                "- Camera exposure/capture jitter: `not evaluable`; frozen v1 "
                "stores an inference-start proxy in the frame timestamp field."
            ),
            f"- Uncertainty status: `{uncertainty.get('status')}`; threshold selected: `no`.",
            "",
            "| Requested conditional coverage | Realized | Mean error px | Target-macro mean px |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in uncertainty.get("coverage_risk") or []:
        item = dict(row)
        lines.append(
            f"| {_markdown_number(item.get('requested_coverage'), 2)} | "
            f"{_markdown_number(item.get('realized_conditional_coverage'), 4)} | "
            f"{_markdown_number(item.get('mean_spatial_error_px'))} | "
            f"{_markdown_number(item.get('target_macro_mean_spatial_error_px'))} |"
        )
    lines.extend(
        [
            "",
            "## Descriptive interval and controls",
            "",
            (
                "- Target-cluster bootstrap mean and 95% interval: "
                f"`{_markdown_number(bootstrap.get('point_estimate_px'))}` "
                f"[`{_markdown_number(bootstrap.get('ci95_low_px'))}`, "
                f"`{_markdown_number(bootstrap.get('ci95_high_px'))}`] px; "
                f"`{bootstrap.get('cluster_count', 0)}` clusters, "
                f"`{bootstrap.get('resamples', 0)}` resamples."
            ),
            (
                "- This is not an independent-population confidence interval; "
                "all clusters come from one self-development capture."
            ),
            (
                "- Target-label permutation control: "
                "`"
                + str(
                    dict(
                        region.get(
                            "target_label_cyclic_permutation_negative_control"
                        )
                        or {}
                    ).get("status")
                )
                + "`."
            ),
            "- Evaluation-target fitting remains pending runner training-provenance verification.",
            "",
            "## Decision",
            "",
            "No finest production resolution band or abstention threshold is selected. "
            "Preserve positive and negative results, then freeze any proposed threshold "
            "before a new participant/device confirmation capture.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "ANALYSIS_TYPE",
    "BOOTSTRAP_ALGORITHM",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "EXPECTED_ANALYSIS_DEFINITION_SHA256",
    "PERCENTILE_ALGORITHM",
    "UNCERTAINTY_COVERAGES",
    "analyze_measurement_run",
    "load_frozen_analysis_definition",
    "render_measurement_run_markdown",
]

"""Versioned webcam capture contracts shared by calibration and inference.

The contract describes the pixels that actually reached the gaze preprocessor,
not merely the constraints requested from ``getUserMedia``.  Legacy datasets
and models may omit it; callers must treat that case as unchecked rather than
silently claiming compatibility.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any


CAPTURE_CONTRACT_SCHEMA_VERSION = 1
CAPTURE_RESIZE_POLICY = "fit_width_preserve_aspect"
CAPTURE_MIME_TYPE = "image/jpeg"
CAPTURE_ASPECT_TOLERANCE = 0.02
FIT_TARGET_CONTRACT_SCHEMA_VERSION = 1
SIGNED_SCREEN_COORDINATE_SYSTEM = "signed_normalized_screen_coordinates_[-1,1]"
PARTICIPANT_GAZE_MEASUREMENT_CONTRACT_PATH = Path(__file__).with_name(
    "participant_gaze_measurement_contract_v1.json"
)
PARTICIPANT_CALIBRATION_LABEL_AUTHORITY = (
    "server_frozen_participant_motion_calibration_v1"
)
FROZEN_PARTICIPANT_CALIBRATION_TARGETS = (
    (0.08, 0.10),
    (0.50, 0.10),
    (0.92, 0.10),
    (0.08, 0.50),
    (0.50, 0.50),
    (0.92, 0.50),
    (0.08, 0.90),
    (0.50, 0.90),
    (0.92, 0.90),
    (0.29, 0.30),
    (0.71, 0.30),
    (0.29, 0.70),
    (0.71, 0.70),
)
FROZEN_PARTICIPANT_MOTION_BLOCKS = {
    "neutral": {"posture_condition": "neutral", "distance_condition": "nominal"},
    "left": {"posture_condition": "left", "distance_condition": "nominal"},
    "right": {"posture_condition": "right", "distance_condition": "nominal"},
    "near": {"posture_condition": "neutral", "distance_condition": "near"},
    "far": {"posture_condition": "neutral", "distance_condition": "far"},
}
PARTICIPANT_CALIBRATION_SERVER_FIELDS = (
    "phase",
    "point_index",
    "repeat_index",
    "target_x_norm",
    "target_y_norm",
    "collect_mode",
    "collection_protocol",
    "motion_block_id",
    "posture_condition",
    "distance_condition",
    "lighting_condition",
    "capture_burst_id",
    "calibration_label_authority",
    "target_pixel_role",
)


def load_participant_gaze_measurement_contract(
    path: Path | None = None,
) -> dict[str, Any]:
    """Load the additive gaze contract without changing frozen study hashes."""

    payload = json.loads(
        (path or PARTICIPANT_GAZE_MEASUREMENT_CONTRACT_PATH).read_text(
            encoding="utf-8"
        )
    )
    required = {
        "schema_version",
        "contract_id",
        "contract_version",
        "participant_protocol_compatibility",
        "participant_calibration",
        "capture_contract",
        "target_independence",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"participant gaze measurement contract missing: {missing}")
    _validate_participant_calibration_specification(payload)
    return payload


def _validate_participant_calibration_specification(
    measurement_contract: Mapping[str, object],
) -> None:
    specification = measurement_contract.get("participant_calibration")
    if not isinstance(specification, Mapping):
        raise ValueError("participant calibration contract must be an object")
    if (
        specification.get("label_authority")
        != PARTICIPANT_CALIBRATION_LABEL_AUTHORITY
    ):
        raise ValueError("participant calibration label authority is invalid")
    if specification.get("collection_protocol") != "motion-diverse-v1":
        raise ValueError("participant calibration collection protocol is invalid")
    if specification.get("collect_mode") != "motion_robust":
        raise ValueError("participant calibration collect mode is invalid")
    if specification.get("phase") != "calibration":
        raise ValueError("participant calibration phase is invalid")
    if specification.get("lighting_condition") != "ambient":
        raise ValueError("participant calibration lighting condition is invalid")
    if (
        specification.get("target_coordinate_system")
        != SIGNED_SCREEN_COORDINATE_SYSTEM
    ):
        raise ValueError("participant calibration target coordinate system is invalid")
    if specification.get("repeats_per_target_per_block") != 1:
        raise ValueError("participant calibration repeat count is invalid")
    if specification.get("target_pixel_role") != "client_reported_diagnostic_only":
        raise ValueError("participant calibration target pixel role is invalid")
    if specification.get("capture_burst_id_policy") != (
        "{session_id}:{motion_block_id}:r{repeat_index}"
    ):
        raise ValueError("participant calibration capture burst policy is invalid")
    if specification.get("server_overwritten_fields") != list(
        PARTICIPANT_CALIBRATION_SERVER_FIELDS
    ):
        raise ValueError("participant calibration authoritative fields are invalid")

    raw_targets = specification.get("frozen_targets")
    if not isinstance(raw_targets, list) or len(raw_targets) != len(
        FROZEN_PARTICIPANT_CALIBRATION_TARGETS
    ):
        raise ValueError("participant calibration targets are invalid")
    normalized_targets: list[tuple[int, float, float]] = []
    for index, raw in enumerate(raw_targets):
        if not isinstance(raw, Mapping):
            raise ValueError("participant calibration target is invalid")
        point_index = _strict_nonnegative_integer(
            raw.get("point_index"),
            field=f"participant_calibration.frozen_targets[{index}].point_index",
        )
        x_fraction = _finite_number(
            raw.get("target_x_viewport_fraction"),
            field=(
                f"participant_calibration.frozen_targets[{index}]"
                ".target_x_viewport_fraction"
            ),
        )
        y_fraction = _finite_number(
            raw.get("target_y_viewport_fraction"),
            field=(
                f"participant_calibration.frozen_targets[{index}]"
                ".target_y_viewport_fraction"
            ),
        )
        normalized_targets.append((point_index, x_fraction, y_fraction))
    expected_targets = [
        (index, x_fraction, y_fraction)
        for index, (x_fraction, y_fraction) in enumerate(
            FROZEN_PARTICIPANT_CALIBRATION_TARGETS
        )
    ]
    if normalized_targets != expected_targets:
        raise ValueError("participant calibration targets changed")

    raw_blocks = specification.get("motion_blocks")
    if not isinstance(raw_blocks, list):
        raise ValueError("participant calibration motion blocks are invalid")
    normalized_blocks: dict[str, dict[str, str]] = {}
    for raw in raw_blocks:
        if not isinstance(raw, Mapping):
            raise ValueError("participant calibration motion block is invalid")
        block_id = str(raw.get("motion_block_id") or "")
        if not block_id or block_id in normalized_blocks:
            raise ValueError("participant calibration motion block ID is invalid")
        normalized_blocks[block_id] = {
            "posture_condition": str(raw.get("posture_condition") or ""),
            "distance_condition": str(raw.get("distance_condition") or ""),
        }
    if normalized_blocks != FROZEN_PARTICIPANT_MOTION_BLOCKS:
        raise ValueError("participant calibration motion blocks changed")


def _strict_nonnegative_integer(value: object, *, field: str) -> int:
    number = _finite_number(value, field=field)
    if not number.is_integer() or number < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(number)


def authoritative_participant_calibration_labels(
    payload: Mapping[str, object],
    *,
    session_id: str,
) -> dict[str, object]:
    """Return frozen labels for one participant-linked calibration sample."""

    point_index = _strict_nonnegative_integer(
        payload.get("point_index"),
        field="participant calibration point_index",
    )
    if point_index >= len(FROZEN_PARTICIPANT_CALIBRATION_TARGETS):
        raise ValueError("participant calibration point_index is outside frozen design")
    repeat_index = _strict_nonnegative_integer(
        payload.get("repeat_index"),
        field="participant calibration repeat_index",
    )
    if repeat_index != 0:
        raise ValueError("participant calibration repeat_index must be zero")
    block_id = str(payload.get("motion_block_id") or "").strip()
    block = FROZEN_PARTICIPANT_MOTION_BLOCKS.get(block_id)
    if block is None:
        raise ValueError("participant calibration motion_block_id is outside frozen design")
    target_x_fraction, target_y_fraction = FROZEN_PARTICIPANT_CALIBRATION_TARGETS[
        point_index
    ]
    return {
        "phase": "calibration",
        "point_index": point_index,
        "repeat_index": repeat_index,
        "target_x_norm": target_x_fraction * 2.0 - 1.0,
        "target_y_norm": target_y_fraction * 2.0 - 1.0,
        "collect_mode": "motion_robust",
        "collection_protocol": "motion-diverse-v1",
        "motion_block_id": block_id,
        "posture_condition": block["posture_condition"],
        "distance_condition": block["distance_condition"],
        "lighting_condition": "ambient",
        "capture_burst_id": f"{session_id}:{block_id}:r{repeat_index}",
        "calibration_label_authority": PARTICIPANT_CALIBRATION_LABEL_AUTHORITY,
        "target_pixel_role": "client_reported_diagnostic_only",
    }


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"capture contract field {field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"capture contract field {field} must be numeric") from exc
    if not isfinite(number):
        raise ValueError(f"capture contract field {field} must be finite")
    return number


def _positive_int(value: object, *, field: str) -> int:
    number = _finite_number(value, field=field)
    if not number.is_integer() or not 1 <= number <= 16384:
        raise ValueError(
            f"capture contract field {field} must be an integer between 1 and 16384"
        )
    return int(number)


def normalize_capture_contract(payload: Mapping[str, object]) -> dict[str, Any]:
    """Validate and normalize a browser-supplied capture contract."""

    if not isinstance(payload, Mapping):
        raise ValueError("capture contract must be an object")
    schema_version = _positive_int(
        payload.get("schema_version"), field="schema_version"
    )
    if schema_version != CAPTURE_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported capture contract schema version")

    normalized: dict[str, Any] = {
        "schema_version": schema_version,
        "intent_width_px": _positive_int(
            payload.get("intent_width_px"), field="intent_width_px"
        ),
        "intent_height_px": _positive_int(
            payload.get("intent_height_px"), field="intent_height_px"
        ),
        "source_width_px": _positive_int(
            payload.get("source_width_px"), field="source_width_px"
        ),
        "source_height_px": _positive_int(
            payload.get("source_height_px"), field="source_height_px"
        ),
        "transport_width_px": _positive_int(
            payload.get("transport_width_px"), field="transport_width_px"
        ),
        "transport_height_px": _positive_int(
            payload.get("transport_height_px"), field="transport_height_px"
        ),
    }
    for field in ("intent_frame_rate_hz", "source_frame_rate_hz"):
        number = _finite_number(payload.get(field, 0.0), field=field)
        if not 0 <= number <= 240:
            raise ValueError(f"capture contract field {field} is out of range")
        normalized[field] = number

    resize_policy = str(payload.get("resize_policy") or "").strip()
    if resize_policy != CAPTURE_RESIZE_POLICY:
        raise ValueError("unsupported capture resize policy")
    normalized["resize_policy"] = resize_policy

    mime_type = str(payload.get("mime_type") or "").strip().lower()
    if mime_type != CAPTURE_MIME_TYPE:
        raise ValueError("unsupported capture MIME type")
    normalized["mime_type"] = mime_type

    jpeg_quality = _finite_number(payload.get("jpeg_quality"), field="jpeg_quality")
    if not 0.1 <= jpeg_quality <= 1.0:
        raise ValueError("capture contract JPEG quality is out of range")
    normalized["jpeg_quality"] = jpeg_quality

    mirror_applied = payload.get("mirror_applied")
    if not isinstance(mirror_applied, bool):
        raise ValueError("capture contract mirror_applied must be boolean")
    normalized["mirror_applied"] = mirror_applied

    facing_mode = str(payload.get("facing_mode") or "unknown").strip().lower()
    if not facing_mode or len(facing_mode) > 32:
        raise ValueError("capture contract facing mode is invalid")
    normalized["facing_mode"] = facing_mode

    expected_transport_height = round(
        normalized["transport_width_px"]
        * normalized["source_height_px"]
        / normalized["source_width_px"]
    )
    if abs(normalized["transport_height_px"] - expected_transport_height) > 1:
        raise ValueError(
            "capture contract transport dimensions do not preserve source aspect ratio"
        )
    return normalized


def validate_transport_frame(
    capture_contract: Mapping[str, object],
    *,
    frame_width_px: int,
    frame_height_px: int,
) -> dict[str, Any]:
    """Verify that decoded pixels match the declared transport dimensions."""

    normalized = normalize_capture_contract(capture_contract)
    actual = (
        _positive_int(frame_width_px, field="decoded_frame_width_px"),
        _positive_int(frame_height_px, field="decoded_frame_height_px"),
    )
    declared = (
        normalized["transport_width_px"],
        normalized["transport_height_px"],
    )
    if actual != declared:
        raise ValueError(
            "decoded frame dimensions do not match the capture contract "
            f"({actual[0]}x{actual[1]} != {declared[0]}x{declared[1]})"
        )
    return normalized


def compare_capture_contracts(
    calibration: Mapping[str, object] | None,
    observed: Mapping[str, object] | None,
    *,
    aspect_tolerance: float = CAPTURE_ASPECT_TOLERANCE,
) -> dict[str, Any]:
    """Compare geometry-affecting fields without rejecting benign FPS changes."""

    if calibration is None or observed is None:
        return {
            "status": "unavailable",
            "compatible": None,
            "reasons": ["capture_contract_unavailable"],
            "warnings": [],
        }
    reference = normalize_capture_contract(calibration)
    candidate = normalize_capture_contract(observed)
    reasons: list[str] = []
    warnings: list[str] = []

    for field, reason in (
        ("resize_policy", "resize_policy_mismatch"),
        ("mirror_applied", "mirror_policy_mismatch"),
        ("facing_mode", "facing_mode_mismatch"),
    ):
        if reference[field] != candidate[field]:
            reasons.append(reason)

    reference_source_aspect = (
        reference["source_width_px"] / reference["source_height_px"]
    )
    candidate_source_aspect = (
        candidate["source_width_px"] / candidate["source_height_px"]
    )
    if abs(reference_source_aspect - candidate_source_aspect) > aspect_tolerance:
        reasons.append("source_aspect_ratio_mismatch")

    reference_transport_aspect = (
        reference["transport_width_px"] / reference["transport_height_px"]
    )
    candidate_transport_aspect = (
        candidate["transport_width_px"] / candidate["transport_height_px"]
    )
    if abs(reference_transport_aspect - candidate_transport_aspect) > aspect_tolerance:
        reasons.append("transport_aspect_ratio_mismatch")

    if (
        reference["source_width_px"],
        reference["source_height_px"],
    ) != (
        candidate["source_width_px"],
        candidate["source_height_px"],
    ):
        warnings.append("source_resolution_changed")
    if (
        reference["transport_width_px"],
        reference["transport_height_px"],
    ) != (
        candidate["transport_width_px"],
        candidate["transport_height_px"],
    ):
        warnings.append("transport_resolution_changed")
    if (
        reference["intent_width_px"],
        reference["intent_height_px"],
    ) != (
        candidate["intent_width_px"],
        candidate["intent_height_px"],
    ):
        warnings.append("capture_intent_resolution_changed")
    reference_fps = float(reference["source_frame_rate_hz"])
    candidate_fps = float(candidate["source_frame_rate_hz"])
    if reference_fps and candidate_fps and abs(reference_fps - candidate_fps) > 5.0:
        warnings.append("source_frame_rate_changed")
    if abs(float(reference["jpeg_quality"]) - float(candidate["jpeg_quality"])) > 0.05:
        warnings.append("jpeg_quality_changed")

    return {
        "status": "compatible" if not reasons else "mismatch",
        "compatible": not reasons,
        "reasons": reasons,
        "warnings": warnings,
    }


def representative_capture_contract(
    records: Sequence[Mapping[str, object]],
) -> dict[str, Any] | None:
    """Return one checked contract for a calibration manifest.

    Models created from legacy manifests remain usable and simply carry no
    contract.  If a new manifest starts recording contracts, every usable row
    must provide a geometry-compatible one.
    """

    supplied = [record.get("capture_contract") for record in records]
    if all(value is None for value in supplied):
        return None
    if any(not isinstance(value, Mapping) for value in supplied):
        raise ValueError("calibration capture contract is incomplete or malformed")
    contracts = [
        normalize_capture_contract(value)
        for value in supplied
        if isinstance(value, Mapping)
    ]
    reference = contracts[0]
    for contract in contracts[1:]:
        comparison = compare_capture_contracts(reference, contract)
        if comparison["compatible"] is not True:
            raise ValueError(
                "calibration capture contract changed within the session: "
                + ", ".join(comparison["reasons"])
            )
    result = dict(reference)
    result["source_frame_rate_hz"] = float(
        statistics.median(float(item["source_frame_rate_hz"]) for item in contracts)
    )
    return result


def _normalized_fit_target(
    raw: object,
    *,
    index: int,
) -> tuple[float, float]:
    if isinstance(raw, Mapping):
        raw_x = raw.get("target_x_norm")
        raw_y = raw.get("target_y_norm")
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        if len(raw) != 2:
            raise ValueError(f"fit target {index} must contain exactly two coordinates")
        raw_x, raw_y = raw
    else:
        raise ValueError(f"fit target {index} must be an object or coordinate pair")
    target_x = _finite_number(raw_x, field=f"fit_targets[{index}].target_x_norm")
    target_y = _finite_number(raw_y, field=f"fit_targets[{index}].target_y_norm")
    if not -1.0 <= target_x <= 1.0 or not -1.0 <= target_y <= 1.0:
        raise ValueError(f"fit target {index} is outside signed screen coordinates")
    return target_x, target_y


def normalize_fit_target_contract(payload: Mapping[str, object]) -> dict[str, Any]:
    """Validate calibration-target provenance stored beside a fitted model."""

    if not isinstance(payload, Mapping):
        raise ValueError("fit target contract must be an object")
    schema_version = _positive_int(
        payload.get("schema_version"),
        field="fit_target_contract.schema_version",
    )
    if schema_version != FIT_TARGET_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported fit target contract schema version")
    coordinate_system = str(payload.get("coordinate_system") or "")
    if coordinate_system != SIGNED_SCREEN_COORDINATE_SYSTEM:
        raise ValueError("fit target contract coordinate system is invalid")
    status = str(payload.get("status") or "")
    if status not in {"available", "unavailable"}:
        raise ValueError("fit target contract status is invalid")
    complete = payload.get("complete")
    if not isinstance(complete, bool) or complete != (status == "available"):
        raise ValueError("fit target contract completeness is inconsistent")
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list):
        raise ValueError("fit target contract targets must be an array")
    unique_targets = sorted(
        {
            _normalized_fit_target(raw, index=index)
            for index, raw in enumerate(raw_targets)
        }
    )
    if status == "available" and not unique_targets:
        raise ValueError("available fit target contract must contain targets")
    target_count_number = _finite_number(
        payload.get("target_count"),
        field="fit_target_contract.target_count",
    )
    if (
        not target_count_number.is_integer()
        or not 0 <= target_count_number <= 4096
    ):
        raise ValueError("fit target contract target count is invalid")
    target_count = int(target_count_number)
    if target_count != len(unique_targets):
        raise ValueError("fit target contract target count is inconsistent")
    raw_reasons = payload.get("reasons", [])
    if not isinstance(raw_reasons, list) or any(
        not isinstance(reason, str) or not reason for reason in raw_reasons
    ):
        raise ValueError("fit target contract reasons are invalid")
    if status == "unavailable" and not raw_reasons:
        raise ValueError("unavailable fit target contract must explain why")
    return {
        "schema_version": schema_version,
        "status": status,
        "complete": complete,
        "coordinate_system": coordinate_system,
        "coordinate_range": [-1.0, 1.0],
        "source": "successful_samples_used_by_personalized_calibration_stages",
        "target_count": len(unique_targets),
        "targets": [
            {"target_x_norm": target_x, "target_y_norm": target_y}
            for target_x, target_y in unique_targets
        ],
        "reasons": list(raw_reasons),
    }


def build_fit_target_contract(
    successful_targets: Sequence[object],
    *,
    inherited_contract: Mapping[str, object] | None = None,
    inherited_targets_required: bool = False,
) -> dict[str, Any]:
    """Describe every known screen target used by personalized fit stages."""

    target_pairs = {
        _normalized_fit_target(raw, index=index)
        for index, raw in enumerate(successful_targets)
    }
    reasons: list[str] = []
    complete = True
    if inherited_targets_required:
        if inherited_contract is None:
            complete = False
            reasons.append("inherited_fit_target_contract_unavailable")
        else:
            try:
                inherited = normalize_fit_target_contract(inherited_contract)
            except ValueError:
                complete = False
                reasons.append("inherited_fit_target_contract_invalid")
            else:
                target_pairs.update(
                    (target["target_x_norm"], target["target_y_norm"])
                    for target in inherited["targets"]
                )
                if inherited["status"] != "available":
                    complete = False
                    reasons.append("inherited_fit_target_contract_unavailable")
    if not target_pairs:
        complete = False
        reasons.append("no_successful_fit_targets")
    status = "available" if complete else "unavailable"
    payload = {
        "schema_version": FIT_TARGET_CONTRACT_SCHEMA_VERSION,
        "status": status,
        "complete": complete,
        "coordinate_system": SIGNED_SCREEN_COORDINATE_SYSTEM,
        "coordinate_range": [-1.0, 1.0],
        "source": "successful_samples_used_by_personalized_calibration_stages",
        "target_count": len(target_pairs),
        "targets": [
            {"target_x_norm": target_x, "target_y_norm": target_y}
            for target_x, target_y in sorted(target_pairs)
        ],
        "reasons": sorted(set(reasons)),
    }
    return normalize_fit_target_contract(payload)

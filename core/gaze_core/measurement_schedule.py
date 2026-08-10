"""Frozen, CPU-only acquisition schedule for webcam gaze ceiling v1.

This module deliberately owns only acquisition identity and integrity.  It does
not collect webcam frames, train a model, calculate measurement results, or
interact with participant-study state.  A physical capture artifact is eligible
for downstream analysis only after its schedule and sample contract pass here;
passing this verifier is not itself a gaze-quality or accuracy result.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
RUN_MANIFEST_TYPE = "webcam_gaze_measurement_ceiling_acquisition_run_v1"
CAPTURE_ARTIFACT_TYPE = "webcam_gaze_measurement_ceiling_capture_artifact_v1"
SCHEDULE_DEFINITION_ID = "webcam-gaze-measurement-ceiling-schedule-v1"
SCHEDULE_DEFINITION_VERSION = "2026-08-10.v1"
EXPECTED_PROTOCOL_ID = "webcam-gaze-measurement-ceiling-v1"
EXPECTED_PROTOCOL_CANONICAL_SHA256 = (
    "be4dfb0956ce3594546336fe7a54da6ba878f2d6fcd457d36cbaf0159132fced"
)
EXPECTED_CALIBRATION_SAMPLE_COUNT = 65
EXPECTED_EVALUATION_SAMPLE_COUNT = 128
EXPECTED_SAMPLE_COUNT = 193
EXPECTED_BLOCK_IDS = (
    "calibration_neutral",
    "calibration_left",
    "calibration_right",
    "calibration_near",
    "calibration_far",
    "neutral_start",
    "evaluation_left",
    "evaluation_right",
    "evaluation_near",
    "evaluation_far",
    "neutral_end",
)
PROTOCOL_RELATIVE_PATH = (
    "docs/experiments/protocols/"
    "2026-08-10-webcam-gaze-measurement-ceiling-v1.json"
)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_PATH = ROOT / PROTOCOL_RELATIVE_PATH
CAPTURE_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
LOWER_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SHUFFLE_ALGORITHM = "sha256_rank_v1"
SEED_SOURCE = "sha256(capture_run_id + block_id + repeat_index)"
TARGET_COORDINATE_TRANSFORM = "signed = 2 * viewport_fraction - 1"

FORBIDDEN_SAMPLE_INPUT_KEYS = frozenset(
    {
        "base64",
        "blob",
        "canvas_data_url",
        "cognitive_profile",
        "cursor_position",
        "data_url",
        "frame",
        "frame_base64",
        "frame_bytes",
        "image",
        "image_base64",
        "image_bytes",
        "image_data",
        "jpeg",
        "png",
        "question_or_answer_correctness",
        "raw_frame",
        "raw_image",
        "raw_media",
        "raw_video",
        "reading_outcome",
        "text_difficulty",
        "video",
        "webcam_frame",
        "word_identity",
    }
)

SCHEDULE_ROW_FIELDS = (
    "capture_run_id",
    "block_id",
    "block_index",
    "block_role",
    "posture",
    "distance",
    "repeat_index",
    "block_sequence_index",
    "sequence_index",
    "target_id",
    "target_index",
    "target_x_viewport_fraction",
    "target_y_viewport_fraction",
    "target_x_norm",
    "target_y_norm",
    "target_order_seed_sha256",
)


class MeasurementScheduleError(ValueError):
    """Raised when a frozen schedule or acquisition artifact fails closed."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return portable strict JSON bytes used by every acquisition hash."""

    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise MeasurementScheduleError(
            "acquisition payload is not strict canonical JSON"
        ) from exc
    return text.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def deterministic_json(value: Any) -> str:
    """Render a human-readable deterministic file without changing its hash rules."""

    try:
        return (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise MeasurementScheduleError(
            "acquisition payload is not strict JSON"
        ) from exc


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise MeasurementScheduleError(f"{field} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementScheduleError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise MeasurementScheduleError(f"{field} must be finite")
    return number


def _strict_nonnegative_integer(value: Any, *, field: str) -> int:
    number = _finite_number(value, field=field)
    if not number.is_integer() or number < 0:
        raise MeasurementScheduleError(f"{field} must be a non-negative integer")
    return int(number)


def _positive_number(value: Any, *, field: str) -> float:
    number = _finite_number(value, field=field)
    if number <= 0:
        raise MeasurementScheduleError(f"{field} must be positive")
    return number


def _exact_vector(
    value: Any,
    *,
    field: str,
    length: int,
    minimum: float | None = None,
    maximum: float | None = None,
) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MeasurementScheduleError(f"{field} must be a length-{length} array")
    if len(value) != length:
        raise MeasurementScheduleError(f"{field} must be a length-{length} array")
    normalized = [
        _finite_number(item, field=f"{field}[{index}]")
        for index, item in enumerate(value)
    ]
    if minimum is not None and any(item < minimum for item in normalized):
        raise MeasurementScheduleError(f"{field} contains a value below {minimum}")
    if maximum is not None and any(item > maximum for item in normalized):
        raise MeasurementScheduleError(f"{field} contains a value above {maximum}")
    return normalized


def _normalized_capture_run_id(value: Any) -> str:
    capture_run_id = str(value or "")
    if not CAPTURE_RUN_ID_PATTERN.fullmatch(capture_run_id):
        raise MeasurementScheduleError(
            "capture_run_id must match [A-Za-z0-9][A-Za-z0-9._:-]{0,127}"
        )
    return capture_run_id


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _assert_no_forbidden_sample_inputs(
    value: Any,
    *,
    location: str,
) -> None:
    """Reject raw media and non-sensor priors at any nesting depth."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = _normalized_key(key)
            if normalized in FORBIDDEN_SAMPLE_INPUT_KEYS:
                raise MeasurementScheduleError(
                    f"{location} contains forbidden sensor input {key!r}"
                )
            _assert_no_forbidden_sample_inputs(
                nested,
                location=f"{location}.{key}",
            )
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, nested in enumerate(value):
            _assert_no_forbidden_sample_inputs(
                nested,
                location=f"{location}[{index}]",
            )
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raise MeasurementScheduleError(
            f"{location} contains forbidden raw binary media"
        )


def _validate_optional_sensor_fields(
    sample: Mapping[str, Any],
    *,
    index: int,
    optional_fields: Sequence[str],
) -> None:
    scalar_fields = set(optional_fields) - {"prediction_covariance_px"}
    for field in sorted(scalar_fields):
        if field in sample:
            _finite_number(sample[field], field=f"samples[{index}].{field}")
    if "prediction_covariance_px" not in sample:
        return
    covariance = sample["prediction_covariance_px"]
    if (
        isinstance(covariance, (str, bytes))
        or not isinstance(covariance, Sequence)
        or len(covariance) != 2
    ):
        raise MeasurementScheduleError(
            f"samples[{index}].prediction_covariance_px must be a 2x2 matrix"
        )
    rows = [
        _exact_vector(
            row,
            field=f"samples[{index}].prediction_covariance_px[{row_index}]",
            length=2,
        )
        for row_index, row in enumerate(covariance)
    ]
    if not math.isclose(rows[0][1], rows[1][0], rel_tol=1e-9, abs_tol=1e-6):
        raise MeasurementScheduleError(
            f"samples[{index}].prediction_covariance_px must be symmetric"
        )
    if rows[0][0] < 0 or rows[1][1] < 0:
        raise MeasurementScheduleError(
            f"samples[{index}].prediction_covariance_px must be positive semidefinite"
        )
    determinant = rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0]
    if determinant < -1e-6:
        raise MeasurementScheduleError(
            f"samples[{index}].prediction_covariance_px must be positive semidefinite"
        )


def _target_points(protocol: Mapping[str, Any], *, role: str) -> list[list[float]]:
    raw = dict(protocol.get("targets") or {}).get(role, {}).get("points")
    expected = 13 if role == "calibration" else 16
    if not isinstance(raw, list) or len(raw) != expected:
        raise MeasurementScheduleError(
            f"frozen protocol must contain exactly {expected} {role} targets"
        )
    points: list[list[float]] = []
    for index, point in enumerate(raw):
        normalized = _exact_vector(
            point,
            field=f"targets.{role}.points[{index}]",
            length=2,
            minimum=0.0,
            maximum=1.0,
        )
        points.append(normalized)
    if len({tuple(point) for point in points}) != expected:
        raise MeasurementScheduleError(f"frozen {role} targets must be unique")
    return points


def _validate_frozen_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise MeasurementScheduleError("frozen protocol schema_version changed")
    if protocol.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise MeasurementScheduleError("frozen protocol_id changed")
    if protocol.get("status") != "frozen_before_new_capture":
        raise MeasurementScheduleError("frozen protocol status changed")
    if protocol.get("branch") != "experiment/webcam-gaze-measurement-ceiling-v1":
        raise MeasurementScheduleError("frozen protocol branch changed")

    target_contract = dict(protocol.get("targets") or {})
    if target_contract.get("coordinate_system") != "normalized viewport fractions":
        raise MeasurementScheduleError("frozen target coordinate system changed")
    calibration_points = _target_points(protocol, role="calibration")
    evaluation_points = _target_points(protocol, role="evaluation")
    evaluation_contract = dict(target_contract.get("evaluation") or {})
    minimum_distance = _positive_number(
        evaluation_contract.get("minimum_normalized_distance_from_any_calibration_point"),
        field="minimum evaluation target distance",
    )
    if min(
        math.dist(calibration, evaluation)
        for calibration in calibration_points
        for evaluation in evaluation_points
    ) + 1e-12 < minimum_distance:
        raise MeasurementScheduleError("frozen evaluation targets overlap calibration")
    target_order = dict(target_contract.get("target_order") or {})
    if target_order.get("policy") != (
        "deterministic seeded shuffle independently within every block and repeat"
    ):
        raise MeasurementScheduleError("frozen target-order policy changed")
    if target_order.get("seed_source") != SEED_SOURCE:
        raise MeasurementScheduleError("frozen target-order seed source changed")

    blocks = protocol.get("blocks")
    if not isinstance(blocks, list) or [
        str(dict(block).get("id") or "")
        for block in blocks
        if isinstance(block, Mapping)
    ] != list(EXPECTED_BLOCK_IDS):
        raise MeasurementScheduleError("frozen block order changed")
    calibration_count = 0
    evaluation_count = 0
    for index, raw in enumerate(blocks):
        if not isinstance(raw, Mapping):
            raise MeasurementScheduleError(f"blocks[{index}] must be an object")
        role = str(raw.get("role") or "")
        if role not in {"calibration", "evaluation"}:
            raise MeasurementScheduleError(f"blocks[{index}].role is invalid")
        repeats = _strict_nonnegative_integer(
            raw.get("repeats"), field=f"blocks[{index}].repeats"
        )
        if repeats < 1:
            raise MeasurementScheduleError(f"blocks[{index}].repeats must be positive")
        count = repeats * (13 if role == "calibration" else 16)
        if role == "calibration":
            calibration_count += count
        else:
            evaluation_count += count
    if calibration_count != EXPECTED_CALIBRATION_SAMPLE_COUNT:
        raise MeasurementScheduleError("frozen calibration sample count changed")
    if evaluation_count != EXPECTED_EVALUATION_SAMPLE_COUNT:
        raise MeasurementScheduleError("frozen evaluation sample count changed")

    sample_contract = dict(protocol.get("sample_contract") or {})
    required_fields = sample_contract.get("required_fields")
    if (
        not isinstance(required_fields, list)
        or any(not isinstance(field, str) or not field for field in required_fields)
        or len(required_fields) != len(set(required_fields))
    ):
        raise MeasurementScheduleError("frozen required sample fields are invalid")
    for field in (
        "capture_run_id",
        "block_id",
        "block_role",
        "target_id",
        "target_x_norm",
        "target_y_norm",
        "repeat_index",
        "sequence_index",
        "prediction_success",
        "viewport_width",
        "viewport_height",
    ):
        if field not in required_fields:
            raise MeasurementScheduleError(
                f"frozen required sample fields omit {field}"
            )
    compute = dict(protocol.get("compute") or {})
    if (
        compute.get("analysis_device") != "cpu"
        or compute.get("gpu_allowed") is not False
        or compute.get("network_allowed") is not False
    ):
        raise MeasurementScheduleError("frozen CPU/offline compute boundary changed")


def load_frozen_protocol(
    path: str | Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Load and hash the exact immutable v1 acquisition protocol."""

    protocol_path = Path(path or DEFAULT_PROTOCOL_PATH)
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MeasurementScheduleError(
            f"unable to load frozen protocol: {protocol_path}"
        ) from exc
    if not isinstance(protocol, Mapping):
        raise MeasurementScheduleError("frozen protocol must be a JSON object")
    normalized = dict(protocol)
    protocol_sha256 = canonical_sha256(normalized)
    if protocol_sha256 != EXPECTED_PROTOCOL_CANONICAL_SHA256:
        raise MeasurementScheduleError(
            "frozen protocol canonical SHA-256 mismatch; use a new protocol version"
        )
    _validate_frozen_protocol(normalized)
    return normalized, protocol_sha256


def _target_definitions(
    protocol: Mapping[str, Any], *, role: str
) -> list[dict[str, Any]]:
    return [
        {
            "target_id": f"{role}_{index:02d}",
            "target_index": index,
            "target_x_viewport_fraction": point[0],
            "target_y_viewport_fraction": point[1],
            "target_x_norm": point[0] * 2.0 - 1.0,
            "target_y_norm": point[1] * 2.0 - 1.0,
        }
        for index, point in enumerate(_target_points(protocol, role=role))
    ]


def _target_order_seed(capture_run_id: str, block_id: str, repeat_index: int) -> str:
    material = f"{capture_run_id}{block_id}{repeat_index}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _shuffled_targets(
    targets: Sequence[Mapping[str, Any]], *, seed_sha256: str
) -> list[dict[str, Any]]:
    seed_bytes = bytes.fromhex(seed_sha256)
    return sorted(
        (dict(target) for target in targets),
        key=lambda target: (
            hashlib.sha256(
                seed_bytes + b":" + str(target["target_id"]).encode("utf-8")
            ).hexdigest(),
            str(target["target_id"]),
        ),
    )


def _manifest_without_hash(manifest: Mapping[str, Any]) -> dict[str, Any]:
    core = deepcopy(dict(manifest))
    core.pop("manifest_sha256", None)
    return core


def build_run_manifest(
    capture_run_id: str,
    *,
    protocol_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the exact server-authoritative 11-block, 193-row run schedule."""

    normalized_run_id = _normalized_capture_run_id(capture_run_id)
    protocol, protocol_sha256 = load_frozen_protocol(protocol_path)
    required_fields = list(dict(protocol["sample_contract"])["required_fields"])
    rows: list[dict[str, Any]] = []
    block_summaries: list[dict[str, Any]] = []
    role_counts: Counter[str] = Counter()
    sequence_index = 0
    for block_index, raw_block in enumerate(protocol["blocks"]):
        block = dict(raw_block)
        block_id = str(block["id"])
        role = str(block["role"])
        targets = _target_definitions(protocol, role=role)
        block_start = sequence_index
        repeat_seeds: list[str] = []
        for repeat_index in range(int(block["repeats"])):
            seed_sha256 = _target_order_seed(
                normalized_run_id, block_id, repeat_index
            )
            repeat_seeds.append(seed_sha256)
            ordered_targets = _shuffled_targets(targets, seed_sha256=seed_sha256)
            for block_sequence_index, target in enumerate(ordered_targets):
                rows.append(
                    {
                        "capture_run_id": normalized_run_id,
                        "block_id": block_id,
                        "block_index": block_index,
                        "block_role": role,
                        "posture": str(block["posture"]),
                        "distance": str(block["distance"]),
                        "repeat_index": repeat_index,
                        "block_sequence_index": block_sequence_index,
                        "sequence_index": sequence_index,
                        **target,
                        "target_order_seed_sha256": seed_sha256,
                    }
                )
                sequence_index += 1
                role_counts[role] += 1
        block_summaries.append(
            {
                "block_id": block_id,
                "block_index": block_index,
                "block_role": role,
                "posture": str(block["posture"]),
                "distance": str(block["distance"]),
                "repeat_count": int(block["repeats"]),
                "target_count_per_repeat": len(targets),
                "sample_count": sequence_index - block_start,
                "sequence_start_inclusive": block_start,
                "sequence_end_exclusive": sequence_index,
                "repeat_seed_sha256": repeat_seeds,
            }
        )

    if sequence_index != EXPECTED_SAMPLE_COUNT:
        raise MeasurementScheduleError("generated schedule does not contain 193 rows")
    if role_counts != Counter(
        calibration=EXPECTED_CALIBRATION_SAMPLE_COUNT,
        evaluation=EXPECTED_EVALUATION_SAMPLE_COUNT,
    ):
        raise MeasurementScheduleError("generated role counts changed")

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": RUN_MANIFEST_TYPE,
        "status": "scheduled_uncollected",
        "measurement_claim_authorized": False,
        "capture_run_id": normalized_run_id,
        "protocol": {
            "protocol_id": EXPECTED_PROTOCOL_ID,
            "schema_version": int(protocol["schema_version"]),
            "canonical_sha256": protocol_sha256,
            "relative_path": PROTOCOL_RELATIVE_PATH,
        },
        "schedule_contract": {
            "definition_id": SCHEDULE_DEFINITION_ID,
            "definition_version": SCHEDULE_DEFINITION_VERSION,
            "authority": "frozen_protocol_core_generator",
            "browser_may_override_schedule_fields": False,
            "hash_semantics": "unkeyed_integrity_not_server_authentication",
            "server_authoritative_fields": list(SCHEDULE_ROW_FIELDS),
            "server_authoritative_fields_sha256": canonical_sha256(
                list(SCHEDULE_ROW_FIELDS)
            ),
            "required_sample_fields": required_fields,
            "required_sample_fields_sha256": canonical_sha256(required_fields),
            "optional_sensor_uncertainty_fields": list(
                dict(protocol["sample_contract"])[
                    "optional_sensor_uncertainty_fields"
                ]
            ),
            "optional_sensor_uncertainty_fields_sha256": canonical_sha256(
                list(
                    dict(protocol["sample_contract"])[
                        "optional_sensor_uncertainty_fields"
                    ]
                )
            ),
            "target_order_policy": dict(protocol["targets"])["target_order"][
                "policy"
            ],
            "seed_source": SEED_SOURCE,
            "shuffle_algorithm": SHUFFLE_ALGORITHM,
            "target_coordinate_system": (
                "signed_normalized_screen_coordinates_[-1,1]"
            ),
            "target_coordinate_transform": TARGET_COORDINATE_TRANSFORM,
        },
        "expected_counts": {
            "block_count": len(block_summaries),
            "calibration_sample_count": role_counts["calibration"],
            "evaluation_sample_count": role_counts["evaluation"],
            "total_sample_count": len(rows),
        },
        "blocks": block_summaries,
        "rows": rows,
        "rows_sha256": canonical_sha256(rows),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def verify_run_manifest(
    manifest: Mapping[str, Any],
    *,
    protocol_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless a run manifest is the exact regenerated schedule."""

    if not isinstance(manifest, Mapping):
        raise MeasurementScheduleError("run manifest must be an object")
    actual = deepcopy(dict(manifest))
    missing_top = sorted(
        {
            "schema_version",
            "manifest_type",
            "capture_run_id",
            "protocol",
            "schedule_contract",
            "expected_counts",
            "blocks",
            "rows",
            "rows_sha256",
            "manifest_sha256",
        }
        - set(actual)
    )
    if missing_top:
        raise MeasurementScheduleError(f"run manifest missing fields: {missing_top}")
    stored_manifest_sha = str(actual.get("manifest_sha256") or "")
    if not LOWER_SHA256_PATTERN.fullmatch(stored_manifest_sha):
        raise MeasurementScheduleError("run manifest SHA-256 is malformed")
    if stored_manifest_sha != canonical_sha256(_manifest_without_hash(actual)):
        raise MeasurementScheduleError("run manifest SHA-256 mismatch")

    rows = actual.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_SAMPLE_COUNT:
        raise MeasurementScheduleError("run manifest must contain exactly 193 rows")
    if actual.get("rows_sha256") != canonical_sha256(rows):
        raise MeasurementScheduleError("run manifest rows SHA-256 mismatch")
    sequence_indices: list[int] = []
    identities: list[tuple[Any, ...]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise MeasurementScheduleError(f"run manifest row {index} must be an object")
        missing = sorted(set(SCHEDULE_ROW_FIELDS) - set(row))
        if missing:
            raise MeasurementScheduleError(
                f"run manifest row {index} missing fields: {missing}"
            )
        sequence_indices.append(
            _strict_nonnegative_integer(
                row.get("sequence_index"), field=f"rows[{index}].sequence_index"
            )
        )
        identities.append(
            (
                row.get("block_id"),
                row.get("repeat_index"),
                row.get("target_id"),
            )
        )
    if sequence_indices != list(range(EXPECTED_SAMPLE_COUNT)):
        raise MeasurementScheduleError("run manifest row order is not exact")
    if len(set(identities)) != EXPECTED_SAMPLE_COUNT:
        raise MeasurementScheduleError("run manifest contains a duplicate schedule row")

    expected = build_run_manifest(
        _normalized_capture_run_id(actual.get("capture_run_id")),
        protocol_path=protocol_path,
    )
    for index, (observed, frozen) in enumerate(zip(rows, expected["rows"], strict=True)):
        if canonical_json_bytes(observed) != canonical_json_bytes(frozen):
            raise MeasurementScheduleError(
                f"run manifest row {index} differs from frozen deterministic schedule"
            )
    if canonical_json_bytes(actual) != canonical_json_bytes(expected):
        raise MeasurementScheduleError(
            "run manifest metadata differs from frozen deterministic schedule"
        )
    return {
        "status": "passed",
        "capture_run_id": expected["capture_run_id"],
        "protocol_sha256": expected["protocol"]["canonical_sha256"],
        "manifest_sha256": expected["manifest_sha256"],
        **expected["expected_counts"],
        "measurement_claim_authorized": False,
    }


def _artifact_without_hash(artifact: Mapping[str, Any]) -> dict[str, Any]:
    core = deepcopy(dict(artifact))
    core.pop("artifact_sha256", None)
    return core


def _rounded_target_pixel(fraction: float, extent: float) -> float:
    return float(math.floor(fraction * extent + 0.5))


def _validate_sample(
    sample: Mapping[str, Any],
    schedule_row: Mapping[str, Any],
    *,
    index: int,
    required_fields: Sequence[str],
    optional_fields: Sequence[str],
) -> dict[str, Any]:
    missing = sorted(set(required_fields) - set(sample))
    if missing:
        raise MeasurementScheduleError(
            f"capture sample {index} missing required fields: {missing}"
        )
    _assert_no_forbidden_sample_inputs(
        sample,
        location=f"capture sample {index}",
    )
    allowed_fields = set(required_fields) | set(optional_fields) | set(
        SCHEDULE_ROW_FIELDS
    )
    unexpected = sorted(set(sample) - allowed_fields)
    if unexpected:
        raise MeasurementScheduleError(
            f"capture sample {index} contains fields outside the frozen contract: "
            f"{unexpected}"
        )
    _validate_optional_sensor_fields(
        sample,
        index=index,
        optional_fields=optional_fields,
    )
    for field in (
        "capture_run_id",
        "block_id",
        "block_role",
        "target_id",
    ):
        if sample.get(field) != schedule_row.get(field):
            raise MeasurementScheduleError(
                f"capture sample {index} field {field} differs from schedule"
            )
    for field in ("repeat_index", "sequence_index"):
        observed_integer = _strict_nonnegative_integer(
            sample.get(field), field=f"samples[{index}].{field}"
        )
        if observed_integer != schedule_row.get(field):
            raise MeasurementScheduleError(
                f"capture sample {index} field {field} differs from schedule"
            )
    for field in ("target_x_norm", "target_y_norm"):
        observed = _finite_number(sample.get(field), field=f"samples[{index}].{field}")
        expected = float(schedule_row[field])
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise MeasurementScheduleError(
                f"capture sample {index} field {field} differs from schedule"
            )

    capture_source = str(sample.get("capture_source") or "").strip()
    if not capture_source or capture_source != sample.get("capture_source"):
        raise MeasurementScheduleError(
            f"samples[{index}].capture_source must be non-blank and normalized"
        )
    viewport_width = _positive_number(
        sample.get("viewport_width"), field=f"samples[{index}].viewport_width"
    )
    viewport_height = _positive_number(
        sample.get("viewport_height"), field=f"samples[{index}].viewport_height"
    )
    target_x_px = _finite_number(
        sample.get("target_x_px"), field=f"samples[{index}].target_x_px"
    )
    target_y_px = _finite_number(
        sample.get("target_y_px"), field=f"samples[{index}].target_y_px"
    )
    expected_x_px = _rounded_target_pixel(
        float(schedule_row["target_x_viewport_fraction"]), viewport_width
    )
    expected_y_px = _rounded_target_pixel(
        float(schedule_row["target_y_viewport_fraction"]), viewport_height
    )
    if target_x_px != expected_x_px or target_y_px != expected_y_px:
        raise MeasurementScheduleError(
            f"capture sample {index} target pixels do not match frozen viewport target"
        )

    captured_ms = _finite_number(
        sample.get("frame_capture_monotonic_ms"),
        field=f"samples[{index}].frame_capture_monotonic_ms",
    )
    completed_ms = _finite_number(
        sample.get("inference_completed_monotonic_ms"),
        field=f"samples[{index}].inference_completed_monotonic_ms",
    )
    latency_ms = _finite_number(
        sample.get("inference_latency_ms"),
        field=f"samples[{index}].inference_latency_ms",
    )
    if captured_ms < 0 or completed_ms < captured_ms or latency_ms < 0:
        raise MeasurementScheduleError(
            f"capture sample {index} has invalid monotonic timing"
        )
    if not math.isclose(
        completed_ms - captured_ms,
        latency_ms,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise MeasurementScheduleError(
            f"capture sample {index} inference latency does not match timestamps"
        )

    model_id = str(sample.get("model_id") or "").strip()
    if not model_id or model_id != sample.get("model_id"):
        raise MeasurementScheduleError(
            f"samples[{index}].model_id must be non-blank and normalized"
        )
    model_sha256 = str(sample.get("model_sha256") or "")
    if not LOWER_SHA256_PATTERN.fullmatch(model_sha256):
        raise MeasurementScheduleError(
            f"samples[{index}].model_sha256 must be lowercase SHA-256"
        )
    prediction_success = sample.get("prediction_success")
    if not isinstance(prediction_success, bool):
        raise MeasurementScheduleError(
            f"samples[{index}].prediction_success must be boolean"
        )
    sensor_fields = (
        ("raw_gaze_pitch_yaw", 2, None, None),
        ("head_pose_pitch_yaw", 2, None, None),
        ("normalized_face_bbox", 4, -0.25, 1.25),
    )
    if prediction_success:
        for field, length, minimum, maximum in sensor_fields:
            _exact_vector(
                sample.get(field),
                field=f"samples[{index}].{field}",
                length=length,
                minimum=minimum,
                maximum=maximum,
            )
        _finite_number(
            sample.get("predicted_x_px"),
            field=f"samples[{index}].predicted_x_px",
        )
        _finite_number(
            sample.get("predicted_y_px"),
            field=f"samples[{index}].predicted_y_px",
        )
    else:
        for field, length, minimum, maximum in sensor_fields:
            value = sample.get(field)
            if value is not None:
                _exact_vector(
                    value,
                    field=f"samples[{index}].{field}",
                    length=length,
                    minimum=minimum,
                    maximum=maximum,
                )
        for field in ("predicted_x_px", "predicted_y_px"):
            value = sample.get(field)
            if value is not None:
                _finite_number(value, field=f"samples[{index}].{field}")

    camera_width = _positive_number(
        sample.get("camera_width"), field=f"samples[{index}].camera_width"
    )
    camera_height = _positive_number(
        sample.get("camera_height"), field=f"samples[{index}].camera_height"
    )
    _positive_number(
        sample.get("camera_frame_rate"),
        field=f"samples[{index}].camera_frame_rate",
    )
    dpr = _positive_number(
        sample.get("device_pixel_ratio"),
        field=f"samples[{index}].device_pixel_ratio",
    )
    aspect_ratio = camera_width / camera_height
    if not 1.3 <= aspect_ratio <= 1.9:
        raise MeasurementScheduleError(
            f"capture sample {index} camera aspect ratio is outside frozen range"
        )
    return {
        "camera_aspect_ratio": aspect_ratio,
        "frame_capture_monotonic_ms": captured_ms,
        "inference_completed_monotonic_ms": completed_ms,
        "viewport_width": viewport_width,
        "viewport_height": viewport_height,
        "device_pixel_ratio": dpr,
        "capture_source": capture_source,
        "block_role": str(sample["block_role"]),
        "model_binding": (model_id, model_sha256),
    }


def build_capture_artifact(
    run_manifest: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    *,
    evidence_class: str,
    protocol_path: str | Path | None = None,
) -> dict[str, Any]:
    """Bind scheduled rows to captured rows without fabricating a quality claim."""

    manifest_summary = verify_run_manifest(
        run_manifest, protocol_path=protocol_path
    )
    if evidence_class not in {"dry_run_synthetic", "physical_self_development"}:
        raise MeasurementScheduleError("capture artifact evidence_class is invalid")
    normalized_samples = [dict(sample) for sample in samples]
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": CAPTURE_ARTIFACT_TYPE,
        "evidence_class": evidence_class,
        "measurement_evidence_status": (
            "synthetic_not_evidence"
            if evidence_class == "dry_run_synthetic"
            else "contract_complete_pending_quality_analysis"
        ),
        "measurement_claim_authorized": False,
        "capture_run_id": manifest_summary["capture_run_id"],
        "protocol_sha256": manifest_summary["protocol_sha256"],
        "run_manifest_sha256": manifest_summary["manifest_sha256"],
        "run_manifest": deepcopy(dict(run_manifest)),
        "samples": normalized_samples,
        "samples_sha256": canonical_sha256(normalized_samples),
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    verify_capture_artifact(artifact, protocol_path=protocol_path)
    return artifact


def verify_capture_artifact(
    artifact: Mapping[str, Any],
    *,
    protocol_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify exact schedule order, required fields, hashes, and geometry."""

    if not isinstance(artifact, Mapping):
        raise MeasurementScheduleError("capture artifact must be an object")
    actual = deepcopy(dict(artifact))
    required_top = {
        "schema_version",
        "artifact_type",
        "evidence_class",
        "measurement_evidence_status",
        "measurement_claim_authorized",
        "capture_run_id",
        "protocol_sha256",
        "run_manifest_sha256",
        "run_manifest",
        "samples",
        "samples_sha256",
        "artifact_sha256",
    }
    missing_top = sorted(required_top - set(actual))
    if missing_top:
        raise MeasurementScheduleError(
            f"capture artifact missing fields: {missing_top}"
        )
    stored_artifact_sha = str(actual.get("artifact_sha256") or "")
    if not LOWER_SHA256_PATTERN.fullmatch(stored_artifact_sha):
        raise MeasurementScheduleError("capture artifact SHA-256 is malformed")
    if stored_artifact_sha != canonical_sha256(_artifact_without_hash(actual)):
        raise MeasurementScheduleError("capture artifact SHA-256 mismatch")
    if actual.get("schema_version") != SCHEMA_VERSION:
        raise MeasurementScheduleError("capture artifact schema_version changed")
    if actual.get("artifact_type") != CAPTURE_ARTIFACT_TYPE:
        raise MeasurementScheduleError("capture artifact type changed")
    evidence_class = actual.get("evidence_class")
    expected_status = {
        "dry_run_synthetic": "synthetic_not_evidence",
        "physical_self_development": "contract_complete_pending_quality_analysis",
    }.get(evidence_class)
    if expected_status is None or actual.get("measurement_evidence_status") != expected_status:
        raise MeasurementScheduleError("capture artifact evidence boundary changed")
    if actual.get("measurement_claim_authorized") is not False:
        raise MeasurementScheduleError(
            "capture contract verification cannot authorize a measurement claim"
        )

    manifest = actual.get("run_manifest")
    if not isinstance(manifest, Mapping):
        raise MeasurementScheduleError("capture artifact run_manifest must be an object")
    manifest_summary = verify_run_manifest(manifest, protocol_path=protocol_path)
    for field, expected in (
        ("capture_run_id", manifest_summary["capture_run_id"]),
        ("protocol_sha256", manifest_summary["protocol_sha256"]),
        ("run_manifest_sha256", manifest_summary["manifest_sha256"]),
    ):
        if actual.get(field) != expected:
            raise MeasurementScheduleError(
                f"capture artifact {field} does not match run manifest"
            )

    samples = actual.get("samples")
    if not isinstance(samples, list) or len(samples) != EXPECTED_SAMPLE_COUNT:
        raise MeasurementScheduleError(
            "capture artifact must contain exactly 193 samples"
        )
    if actual.get("samples_sha256") != canonical_sha256(samples):
        raise MeasurementScheduleError("capture artifact samples SHA-256 mismatch")
    required_fields = list(manifest["schedule_contract"]["required_sample_fields"])
    optional_fields = list(
        manifest["schedule_contract"]["optional_sensor_uncertainty_fields"]
    )
    sample_diagnostics: list[dict[str, float]] = []
    for index, (sample, schedule_row) in enumerate(
        zip(samples, manifest["rows"], strict=True)
    ):
        if not isinstance(sample, Mapping):
            raise MeasurementScheduleError(f"capture sample {index} must be an object")
        sample_diagnostics.append(
            _validate_sample(
                sample,
                schedule_row,
                index=index,
                required_fields=required_fields,
                optional_fields=optional_fields,
            )
        )
    sample_identities = [
        (
            sample.get("capture_run_id"),
            sample.get("block_id"),
            sample.get("repeat_index"),
            sample.get("target_id"),
        )
        for sample in samples
    ]
    if len(set(sample_identities)) != EXPECTED_SAMPLE_COUNT:
        raise MeasurementScheduleError("capture artifact contains duplicate samples")
    aspect_ratios = [item["camera_aspect_ratio"] for item in sample_diagnostics]
    if max(aspect_ratios) - min(aspect_ratios) > 0.02 + 1e-12:
        raise MeasurementScheduleError(
            "capture artifact camera aspect ratio changed by more than 0.02"
        )
    capture_times = [
        item["frame_capture_monotonic_ms"] for item in sample_diagnostics
    ]
    if any(
        later < earlier
        for earlier, later in zip(capture_times, capture_times[1:])
    ):
        raise MeasurementScheduleError(
            "capture artifact monotonic frame order differs from schedule"
        )
    viewports = {
        (item["viewport_width"], item["viewport_height"])
        for item in sample_diagnostics
    }
    if len(viewports) != 1:
        raise MeasurementScheduleError("capture artifact viewport changed within run")
    dprs = {item["device_pixel_ratio"] for item in sample_diagnostics}
    if len(dprs) != 1:
        raise MeasurementScheduleError(
            "capture artifact device pixel ratio changed within run"
        )
    capture_sources = {item["capture_source"] for item in sample_diagnostics}
    if len(capture_sources) != 1:
        raise MeasurementScheduleError("capture source changed within run")
    for role in ("calibration", "evaluation"):
        role_models = {
            item["model_binding"]
            for item in sample_diagnostics
            if item["block_role"] == role
        }
        if len(role_models) != 1:
            raise MeasurementScheduleError(
                f"capture artifact {role} model binding changed within run"
            )
    successful_count = sum(sample.get("prediction_success") is True for sample in samples)
    return {
        "status": "passed",
        "evidence_class": evidence_class,
        "measurement_claim_authorized": False,
        "capture_run_id": manifest_summary["capture_run_id"],
        "protocol_sha256": manifest_summary["protocol_sha256"],
        "run_manifest_sha256": manifest_summary["manifest_sha256"],
        "artifact_sha256": stored_artifact_sha,
        "sample_count": len(samples),
        "successful_sample_count": successful_count,
        "calibration_sample_count": EXPECTED_CALIBRATION_SAMPLE_COUNT,
        "evaluation_sample_count": EXPECTED_EVALUATION_SAMPLE_COUNT,
    }


__all__ = [
    "CAPTURE_ARTIFACT_TYPE",
    "DEFAULT_PROTOCOL_PATH",
    "EXPECTED_CALIBRATION_SAMPLE_COUNT",
    "EXPECTED_EVALUATION_SAMPLE_COUNT",
    "EXPECTED_PROTOCOL_CANONICAL_SHA256",
    "EXPECTED_SAMPLE_COUNT",
    "MeasurementScheduleError",
    "RUN_MANIFEST_TYPE",
    "build_capture_artifact",
    "build_run_manifest",
    "canonical_json_bytes",
    "canonical_sha256",
    "deterministic_json",
    "load_frozen_protocol",
    "verify_capture_artifact",
    "verify_run_manifest",
]

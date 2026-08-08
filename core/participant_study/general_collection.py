"""Frozen design helpers for the generalizable participant collection.

The module intentionally contains no model inference.  It validates the study
contract, assigns fixed forms before outcomes exist, and normalizes only the
derived observations that the collection store is allowed to persist.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

GENERAL_PROTOCOL_PATH = Path(__file__).with_name("general_collection_v1.json")
GENERAL_BANK_PATH = Path(__file__).with_name("general_collection_bank_v1.json")
WORD_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)
DIRECT_IDENTIFIER_KEYS = {
    "name",
    "email",
    "phone",
    "address",
    "exact_birth_date",
    "exact_age",
}


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def load_general_protocol(path: Path | None = None) -> dict[str, Any]:
    protocol = json.loads((path or GENERAL_PROTOCOL_PATH).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "protocol_id",
        "protocol_version",
        "status",
        "sessions",
        "profile_schema",
        "system_check",
        "reading_layout",
        "gaze_collection",
        "gaze_quality",
        "outcomes",
        "assignment",
        "data_contract",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"general collection protocol missing fields: {missing}")
    return protocol


def load_general_bank(path: Path | None = None) -> dict[str, Any]:
    bank = json.loads((path or GENERAL_BANK_PATH).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "bank_id",
        "bank_version",
        "status",
        "forms",
        "practice",
        "passages",
    }
    missing = sorted(required - set(bank))
    if missing:
        raise ValueError(f"general collection bank missing fields: {missing}")
    return bank


def _surface_count(text: str, surface: str) -> int:
    pattern = re.compile(
        rf"(?<![\w'-]){re.escape(surface)}(?![\w'-])",
        re.IGNORECASE | re.UNICODE,
    )
    return len(pattern.findall(text))


def validate_general_design(
    protocol: Mapping[str, object] | None = None,
    bank: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Fail closed when the frozen protocol and bank disagree."""

    active_protocol = dict(protocol or load_general_protocol())
    active_bank = dict(bank or load_general_bank())
    errors: list[str] = []
    passages = active_bank.get("passages")
    forms = active_bank.get("forms")
    if not isinstance(passages, list) or not isinstance(forms, dict):
        raise ValueError("general collection bank passages/forms are malformed")

    by_id: dict[str, dict[str, Any]] = {}
    families: list[str] = []
    total_probes = 0
    for raw_passage in passages:
        if not isinstance(raw_passage, dict):
            errors.append("passage_not_object")
            continue
        passage = dict(raw_passage)
        passage_id = str(passage.get("passage_id") or "")
        family_id = str(passage.get("family_id") or "")
        text = str(passage.get("text") or "")
        if not passage_id or passage_id in by_id:
            errors.append(f"duplicate_or_missing_passage_id:{passage_id}")
            continue
        by_id[passage_id] = passage
        families.append(family_id)
        words = WORD_PATTERN.findall(text)
        if not 90 <= len(words) <= 160:
            errors.append(f"passage_word_count_out_of_range:{passage_id}:{len(words)}")
        probes = passage.get("probes")
        if not isinstance(probes, list) or len(probes) != 8:
            errors.append(f"passage_probe_count_invalid:{passage_id}")
            continue
        probe_ids: set[str] = set()
        surfaces: set[str] = set()
        for raw_probe in probes:
            if not isinstance(raw_probe, dict):
                errors.append(f"probe_not_object:{passage_id}")
                continue
            probe_id = str(raw_probe.get("probe_id") or "")
            surface = str(raw_probe.get("surface") or "").strip()
            if not probe_id or probe_id in probe_ids:
                errors.append(f"duplicate_or_missing_probe_id:{passage_id}:{probe_id}")
            if not surface or surface.casefold() in surfaces:
                errors.append(f"duplicate_or_missing_probe_surface:{passage_id}:{surface}")
            if _surface_count(text, surface) != 1:
                errors.append(f"probe_surface_not_unique:{passage_id}:{probe_id}")
            probe_ids.add(probe_id)
            surfaces.add(surface.casefold())
        total_probes += len(probes)

    duplicate_families = sorted(
        family_id for family_id, count in Counter(families).items() if count > 1
    )
    if duplicate_families:
        errors.append(f"duplicate_passage_families:{duplicate_families}")

    expected_passages_per_session = int(
        dict(active_protocol.get("sessions") or {}).get("passages_per_session", 0)
    )
    form_members: list[str] = []
    form_balance: dict[str, dict[str, int]] = {}
    for form_id in ("A", "B"):
        member_ids = forms.get(form_id)
        if not isinstance(member_ids, list):
            errors.append(f"missing_form:{form_id}")
            continue
        if len(member_ids) != expected_passages_per_session:
            errors.append(f"form_size_invalid:{form_id}:{len(member_ids)}")
        if len(member_ids) != len(set(member_ids)):
            errors.append(f"form_duplicate_passage:{form_id}")
        unknown = sorted(set(member_ids) - set(by_id))
        if unknown:
            errors.append(f"form_unknown_passage:{form_id}:{unknown}")
        bands = Counter(
            str(by_id[item].get("difficulty_band"))
            for item in member_ids
            if item in by_id
        )
        form_balance[form_id] = dict(sorted(bands.items()))
        if bands != Counter({"foundation": 2, "standard": 2, "advanced": 2}):
            errors.append(f"form_difficulty_imbalance:{form_id}:{dict(bands)}")
        form_members.extend(member_ids)
    if Counter(form_members) != Counter({passage_id: 1 for passage_id in by_id}):
        errors.append("forms_do_not_partition_bank_exactly_once")

    practice = active_bank.get("practice")
    if not isinstance(practice, dict) or not str(practice.get("text") or ""):
        errors.append("practice_passage_missing")
    else:
        practice_probes = practice.get("probes")
        if not isinstance(practice_probes, list) or len(practice_probes) != 3:
            errors.append("practice_probe_count_invalid")
        else:
            for probe in practice_probes:
                if _surface_count(str(practice["text"]), str(probe.get("surface") or "")) != 1:
                    errors.append(f"practice_probe_surface_not_unique:{probe.get('probe_id')}")

    expected_probes = (
        expected_passages_per_session
        * int(dict(active_protocol.get("sessions") or {}).get("word_probes_per_passage", 0))
        * 2
    )
    if total_probes != expected_probes:
        errors.append(f"total_probe_count_invalid:{total_probes}:{expected_probes}")
    if errors:
        raise ValueError("general collection design invalid: " + "; ".join(errors))
    return {
        "ok": True,
        "protocol_id": active_protocol["protocol_id"],
        "protocol_version": active_protocol["protocol_version"],
        "protocol_sha256": canonical_sha256(active_protocol),
        "bank_id": active_bank["bank_id"],
        "bank_version": active_bank["bank_version"],
        "bank_sha256": canonical_sha256(active_bank),
        "passage_count": len(by_id),
        "passage_family_count": len(set(families)),
        "probe_count": total_probes,
        "form_balance": form_balance,
    }


def williams_order(size: int, row: int) -> list[int]:
    """Return one row of an even-order Williams balanced Latin square."""

    if size < 2 or size % 2:
        raise ValueError("Williams order requires an even size of at least two")
    if not 0 <= row < size:
        raise ValueError(f"Williams row must be between 0 and {size - 1}")
    first = [0]
    for step in range(1, size):
        first.append((step + 1) // 2 if step % 2 else size - step // 2)
    return [(value + row) % size for value in first]


def assignment_for_cell(
    schedule_cell: int,
    *,
    bank: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    active_bank = dict(bank or load_general_bank())
    if not 0 <= int(schedule_cell) < 12:
        raise ValueError("schedule cell must be between 0 and 11")
    cell = int(schedule_cell)
    sequence = "A_then_B" if cell < 6 else "B_then_A"
    order_cell = cell % 6
    form_sequence = ["A", "B"] if sequence == "A_then_B" else ["B", "A"]
    visits: list[dict[str, Any]] = []
    for visit_index, form_id in enumerate(form_sequence, start=1):
        form_passages = list(dict(active_bank["forms"])[form_id])
        order = williams_order(len(form_passages), order_cell)
        visits.append(
            {
                "visit_index": visit_index,
                "form_id": form_id,
                "passage_order": [form_passages[index] for index in order],
            }
        )
    return {
        "schedule_cell": cell,
        "sequence": sequence,
        "order_cell": order_cell,
        "visits": visits,
    }


def passage_by_id(
    passage_id: str,
    *,
    bank: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    active_bank = dict(bank or load_general_bank())
    for passage in active_bank["passages"]:
        if passage.get("passage_id") == passage_id:
            return deepcopy(passage)
    raise ValueError("unknown general collection passage")


def public_passage(passage_id: str) -> dict[str, Any]:
    passage = passage_by_id(passage_id)
    return {
        "passage_id": passage["passage_id"],
        "family_id": passage["family_id"],
        "difficulty_band": passage["difficulty_band"],
        "genre": passage["genre"],
        "domain": passage["domain"],
        "text": passage["text"],
        "word_count": len(WORD_PATTERN.findall(passage["text"])),
    }


def public_practice(*, bank: Mapping[str, object] | None = None) -> dict[str, Any]:
    active_bank = dict(bank or load_general_bank())
    practice = dict(active_bank["practice"])
    return {
        "passage_id": practice["passage_id"],
        "text": practice["text"],
        "word_count": len(WORD_PATTERN.findall(practice["text"])),
        "probes": [
            {"probe_id": item["probe_id"], "surface": item["surface"]}
            for item in practice["probes"]
        ],
        "excluded_from_analysis": True,
    }


def probe_order(
    passage_id: str,
    participant_id: str,
    visit_index: int,
) -> list[dict[str, str]]:
    passage = passage_by_id(passage_id)
    probes = [dict(item) for item in passage["probes"]]
    probes.sort(
        key=lambda item: hashlib.sha256(
            f"{participant_id}:{visit_index}:{passage_id}:{item['probe_id']}".encode()
        ).digest()
    )
    return probes


def validate_profile(
    profile: Mapping[str, object],
    *,
    protocol: Mapping[str, object] | None = None,
) -> dict[str, str]:
    active_protocol = dict(protocol or load_general_protocol())
    if not isinstance(profile, Mapping):
        raise ValueError("participant profile must be an object")
    supplied = {str(key) for key in profile}
    prohibited = supplied & DIRECT_IDENTIFIER_KEYS
    if prohibited:
        raise ValueError(f"direct participant identifiers are prohibited: {sorted(prohibited)}")
    schema = dict(dict(active_protocol["profile_schema"])["required"])
    unknown = supplied - set(schema)
    missing = set(schema) - supplied
    if unknown:
        raise ValueError(f"participant profile contains unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"participant profile is missing fields: {sorted(missing)}")
    normalized: dict[str, str] = {}
    for field, allowed in schema.items():
        value = str(profile.get(field) or "")
        if value not in allowed:
            raise ValueError(f"participant profile field {field} has an invalid value")
        normalized[field] = value
    return normalized


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def validate_system_profile(
    payload: Mapping[str, object],
    *,
    protocol: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    active_protocol = dict(protocol or load_general_protocol())
    spec = dict(active_protocol["system_check"])
    required_checks = set(spec["required"])
    checks = payload.get("checks")
    device = payload.get("device")
    if not isinstance(checks, Mapping) or not isinstance(device, Mapping):
        raise ValueError("system checks and coarse device profile must be objects")
    failed = sorted(name for name in required_checks if checks.get(name) is not True)
    if failed:
        raise ValueError(f"required system checks failed: {failed}")
    allowed_device_fields = set(spec["coarse_device_fields"])
    unknown = set(device) - allowed_device_fields
    missing = allowed_device_fields - set(device)
    if unknown or missing:
        raise ValueError(
            f"coarse device fields mismatch; missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    normalized_device = {key: device[key] for key in sorted(allowed_device_fields)}
    for field in (
        "device_class",
        "browser_family",
        "device_pixel_ratio_bucket",
        "estimated_camera_fps_band",
    ):
        value = str(normalized_device[field])
        if not value or len(value) > 32:
            raise ValueError(f"coarse device field {field} is invalid")
        normalized_device[field] = value
    for field in ("viewport_width", "viewport_height", "camera_width", "camera_height"):
        value = int(_finite_number(normalized_device[field], field=field))
        if not 1 <= value <= 16384:
            raise ValueError(f"coarse device field {field} is out of range")
        normalized_device[field] = value
    minimum_viewport = dict(spec["minimum_viewport"])
    minimum_camera = dict(spec["minimum_camera"])
    if (
        normalized_device["viewport_width"] < int(minimum_viewport["width_px"])
        or normalized_device["viewport_height"] < int(minimum_viewport["height_px"])
    ):
        raise ValueError("viewport is below the frozen minimum")
    if (
        normalized_device["camera_width"] < int(minimum_camera["width_px"])
        or normalized_device["camera_height"] < int(minimum_camera["height_px"])
    ):
        raise ValueError("camera resolution is below the frozen minimum")
    return {
        "checks": {name: True for name in sorted(required_checks)},
        "device": normalized_device,
    }


def summarize_validation_samples(
    samples: Sequence[Mapping[str, object]],
    *,
    expected_points: int = 5,
    expected_samples_per_point: int = 3,
) -> dict[str, Any]:
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise ValueError("validation samples must be an array")
    if len(samples) != expected_points * expected_samples_per_point:
        raise ValueError("validation sample count does not match the frozen design")
    grouped_errors: dict[str, list[tuple[float, float, float]]] = {}
    success_count = 0
    normalized_samples: list[dict[str, Any]] = []
    for raw in samples:
        if not isinstance(raw, Mapping):
            raise ValueError("validation sample must be an object")
        if {"image", "image_data", "frame", "video", "audio"} & set(raw):
            raise ValueError("raw media fields are prohibited in validation samples")
        target_id = str(raw.get("target_id") or "")
        if not target_id or len(target_id) > 24:
            raise ValueError("validation target ID is invalid")
        target_x = _finite_number(raw.get("target_x_px"), field="target_x_px")
        target_y = _finite_number(raw.get("target_y_px"), field="target_y_px")
        if raw.get("prediction_success") is True:
            predicted_x = _finite_number(raw.get("predicted_x_px"), field="predicted_x_px")
            predicted_y = _finite_number(raw.get("predicted_y_px"), field="predicted_y_px")
            error = math.hypot(predicted_x - target_x, predicted_y - target_y)
            grouped_errors.setdefault(target_id, []).append((predicted_x, predicted_y, error))
            success_count += 1
            normalized_samples.append(
                {
                    "target_id": target_id,
                    "target_x_px": target_x,
                    "target_y_px": target_y,
                    "prediction_success": True,
                    "predicted_x_px": predicted_x,
                    "predicted_y_px": predicted_y,
                    "spatial_error_px": error,
                }
            )
        else:
            grouped_errors.setdefault(target_id, [])
            normalized_samples.append(
                {
                    "target_id": target_id,
                    "target_x_px": target_x,
                    "target_y_px": target_y,
                    "prediction_success": False,
                    "predicted_x_px": None,
                    "predicted_y_px": None,
                    "spatial_error_px": None,
                }
            )
    if len(grouped_errors) != expected_points:
        raise ValueError("validation target coverage does not match the frozen design")
    if any(len(values) > expected_samples_per_point for values in grouped_errors.values()):
        raise ValueError("too many validation samples were assigned to a target")
    errors = [item[2] for values in grouped_errors.values() for item in values]
    precision_residuals: list[float] = []
    for values in grouped_errors.values():
        if not values:
            continue
        mean_x = statistics.fmean(item[0] for item in values)
        mean_y = statistics.fmean(item[1] for item in values)
        precision_residuals.extend(
            math.hypot(item[0] - mean_x, item[1] - mean_y) for item in values
        )
    sorted_errors = sorted(errors)
    p90_index = max(0, math.ceil(len(sorted_errors) * 0.9) - 1) if sorted_errors else 0
    return {
        "sample_count": len(samples),
        "successful_sample_count": success_count,
        "target_count": len(grouped_errors),
        "targets_with_prediction": sum(bool(values) for values in grouped_errors.values()),
        "prediction_success_fraction": success_count / len(samples),
        "median_spatial_error_px": statistics.median(errors) if errors else None,
        "p90_spatial_error_px": sorted_errors[p90_index] if sorted_errors else None,
        "precision_rms_px": (
            math.sqrt(statistics.fmean(value * value for value in precision_residuals))
            if precision_residuals
            else None
        ),
        "samples": normalized_samples,
    }


def classify_gaze_quality(metrics: Mapping[str, object]) -> str:
    protocol = load_general_protocol()
    bands = dict(dict(protocol["gaze_quality"])["rehearsal_quality_bands"])
    median_error = _finite_number(
        metrics.get("median_spatial_error_px"), field="median_spatial_error_px"
    )
    success = _finite_number(
        metrics.get("prediction_success_fraction"),
        field="prediction_success_fraction",
    )
    sampling_hz = _finite_number(
        metrics.get("effective_sampling_hz"), field="effective_sampling_hz"
    )
    p90_raw = metrics.get("p90_spatial_error_px")
    p90_error = _finite_number(p90_raw, field="p90_spatial_error_px")
    word = dict(bands["word_level_candidate"])
    if (
        median_error <= float(word["maximum_median_error_px"])
        and p90_error <= float(word["maximum_p90_error_px"])
        and success >= float(word["minimum_success_fraction"])
        and sampling_hz >= float(word["minimum_effective_sampling_hz"])
    ):
        return "word_level_candidate"
    passage = dict(bands["passage_level_only"])
    if (
        median_error <= float(passage["maximum_median_error_px"])
        and success >= float(passage["minimum_success_fraction"])
        and sampling_hz >= float(passage["minimum_effective_sampling_hz"])
    ):
        return "passage_level_only"
    return "behavioral_only"


def normalize_telemetry_batch(
    payload: Mapping[str, object],
    *,
    maximum_word_index: int,
) -> dict[str, Any]:
    forbidden = {"image", "image_data", "frame", "video", "audio"} & set(payload)
    if forbidden:
        raise ValueError(f"raw media fields are prohibited: {sorted(forbidden)}")
    batch_id = str(payload.get("batch_id") or "")
    passage_id = str(payload.get("passage_id") or "")
    if not re.fullmatch(r"B-[A-Za-z0-9_-]{8,64}", batch_id):
        raise ValueError("telemetry batch ID is invalid")
    if not passage_id or len(passage_id) > 128:
        raise ValueError("telemetry passage ID is invalid")
    viewport = payload.get("viewport")
    samples = payload.get("samples")
    if not isinstance(viewport, Mapping) or not isinstance(samples, list):
        raise ValueError("telemetry viewport and samples are required")
    if not 1 <= len(samples) <= 64:
        raise ValueError("telemetry batch must contain between 1 and 64 samples")
    width = int(_finite_number(viewport.get("width_px"), field="viewport.width_px"))
    height = int(_finite_number(viewport.get("height_px"), field="viewport.height_px"))
    if not 1 <= width <= 16384 or not 1 <= height <= 16384:
        raise ValueError("telemetry viewport is out of range")
    normalized_samples: list[dict[str, Any]] = []
    previous_elapsed = -1.0
    for raw in samples:
        if not isinstance(raw, Mapping):
            raise ValueError("telemetry sample must be an object")
        if {"image", "image_data", "frame", "video", "audio"} & set(raw):
            raise ValueError("raw media fields are prohibited in telemetry samples")
        elapsed = _finite_number(raw.get("monotonic_elapsed_ms"), field="monotonic_elapsed_ms")
        if elapsed < previous_elapsed or not 0 <= elapsed <= 600_000:
            raise ValueError("telemetry elapsed time is out of order or range")
        previous_elapsed = elapsed
        success = raw.get("prediction_success") is True
        sample: dict[str, Any] = {
            "monotonic_elapsed_ms": elapsed,
            "prediction_success": success,
            "coarse_failure_code": None,
        }
        if success:
            norm = raw.get("screen_xy_norm")
            pixels = raw.get("screen_xy_px")
            gaze = raw.get("gaze_pitch_yaw")
            pose = raw.get("head_pose_pitch_yaw")
            bbox = raw.get("normalized_face_bbox")
            vectors = {
                "screen_xy_norm": (norm, 2, -1.25, 1.25),
                "screen_xy_px": (pixels, 2, -width * 0.25, width * 1.25),
                "gaze_pitch_yaw": (gaze, 2, -4.0, 4.0),
                "head_pose_pitch_yaw": (pose, 2, -4.0, 4.0),
                "normalized_face_bbox": (bbox, 4, -0.25, 1.25),
            }
            for field, (values, size, minimum, maximum) in vectors.items():
                if not isinstance(values, list) or len(values) != size:
                    raise ValueError(f"telemetry field {field} is malformed")
                normalized_values = [
                    _finite_number(value, field=field) for value in values
                ]
                if any(value < minimum or value > maximum for value in normalized_values):
                    raise ValueError(f"telemetry field {field} is out of range")
                sample[field] = normalized_values
            word_index = raw.get("nearest_word_index")
            if word_index is None:
                sample["nearest_word_index"] = None
            else:
                value = int(_finite_number(word_index, field="nearest_word_index"))
                if not 0 <= value < maximum_word_index:
                    raise ValueError("nearest word index is out of range")
                sample["nearest_word_index"] = value
        else:
            failure = str(raw.get("coarse_failure_code") or "prediction_failed")
            allowed_failures = {
                "prediction_failed",
                "no_face",
                "timeout",
                "camera_unavailable",
                "network_error",
            }
            if failure not in allowed_failures:
                raise ValueError("telemetry failure code is invalid")
            sample["coarse_failure_code"] = failure
        normalized_samples.append(sample)
    return {
        "batch_id": batch_id,
        "passage_id": passage_id,
        "viewport": {"width_px": width, "height_px": height},
        "samples": normalized_samples,
    }


def validate_round_payload(
    payload: Mapping[str, object],
    *,
    passage_id: str,
    participant_id: str,
    visit_index: int,
) -> dict[str, Any]:
    forbidden = {
        "image",
        "image_data",
        "frame",
        "video",
        "audio",
        *DIRECT_IDENTIFIER_KEYS,
    } & set(payload)
    if forbidden:
        raise ValueError(f"prohibited round fields were supplied: {sorted(forbidden)}")
    passage = passage_by_id(passage_id)
    elapsed = int(_finite_number(payload.get("reading_elapsed_ms"), field="reading_elapsed_ms"))
    if not 20_000 <= elapsed <= 480_000:
        raise ValueError("reading duration is outside the frozen range")
    if payload.get("scroll_occurred") is not False:
        raise ValueError("scrolling during reading invalidates the standardized round")
    zoom = _finite_number(payload.get("zoom_ratio"), field="zoom_ratio")
    if not 0.99 <= zoom <= 1.01:
        raise ValueError("browser zoom differs from the frozen layout")
    responses = payload.get("word_reviews")
    if not isinstance(responses, Mapping):
        raise ValueError("word reviews must be an object")
    expected_probes = probe_order(passage_id, participant_id, visit_index)
    expected_ids = {item["probe_id"] for item in expected_probes}
    if set(responses) != expected_ids:
        raise ValueError("word review IDs do not match the frozen probe set")
    allowed_levels = {"no_review", "unsure", "review_needed"}
    word_reviews: list[dict[str, str]] = []
    for probe in expected_probes:
        label = str(responses[probe["probe_id"]])
        if label not in allowed_levels:
            raise ValueError("word review label is invalid")
        word_reviews.append(
            {
                "probe_id": probe["probe_id"],
                "surface": probe["surface"],
                "stratum": probe["stratum"],
                "label": label,
            }
        )
    report = payload.get("passage_self_report")
    if not isinstance(report, Mapping):
        raise ValueError("passage self report must be an object")
    understanding = int(_finite_number(report.get("understanding"), field="understanding"))
    mental_effort = int(_finite_number(report.get("mental_effort"), field="mental_effort"))
    if understanding not in range(1, 6) or mental_effort not in range(1, 6):
        raise ValueError("passage self-report ratings must be between 1 and 5")
    if not isinstance(report.get("read_complete"), bool) or not isinstance(
        report.get("interrupted"), bool
    ):
        raise ValueError("passage completion and interruption must be booleans")
    layout = payload.get("word_layout")
    word_count = len(WORD_PATTERN.findall(passage["text"]))
    if not isinstance(layout, list) or len(layout) != word_count:
        raise ValueError("word layout snapshot does not match the passage word count")
    normalized_layout: list[dict[str, float | int]] = []
    for expected_index, raw in enumerate(layout):
        if not isinstance(raw, Mapping) or raw.get("word_index") != expected_index:
            raise ValueError("word layout indices must be complete and ordered")
        item: dict[str, float | int] = {"word_index": expected_index}
        for field in ("left_px", "top_px", "right_px", "bottom_px"):
            item[field] = _finite_number(raw.get(field), field=f"word_layout.{field}")
        if item["right_px"] <= item["left_px"] or item["bottom_px"] <= item["top_px"]:
            raise ValueError("word layout contains a non-positive rectangle")
        normalized_layout.append(item)
    return {
        "passage_id": passage_id,
        "passage_family_id": passage["family_id"],
        "difficulty_band": passage["difficulty_band"],
        "reading_elapsed_ms": elapsed,
        "scroll_occurred": False,
        "zoom_ratio": zoom,
        "word_reviews": word_reviews,
        "passage_self_report": {
            "understanding": understanding,
            "mental_effort": mental_effort,
            "read_complete": report["read_complete"],
            "interrupted": report["interrupted"],
        },
        "word_layout": normalized_layout,
        "word_layout_sha256": canonical_sha256(normalized_layout),
        "probe_order_sha256": canonical_sha256(
            [item["probe_id"] for item in expected_probes]
        ),
    }

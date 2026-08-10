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

from core.gaze_core.capture_contract import (
    SIGNED_SCREEN_COORDINATE_SYSTEM,
    load_participant_gaze_measurement_contract,
    normalize_fit_target_contract,
)

GENERAL_PROTOCOL_PATH = Path(__file__).with_name("general_collection_v1.json")
GENERAL_BANK_PATH = Path(__file__).with_name("general_collection_bank_v1.json")
WORD_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)
FROZEN_HELDOUT_GRID_4X4 = tuple(
    (x, y)
    for y in (0.20, 0.40, 0.60, 0.80)
    for x in (0.18, 0.39, 0.61, 0.82)
)
FROZEN_VALIDATION_TARGETS = (
    ("heldout_top_left", 0.18, 0.20),
    ("heldout_top_right", 0.82, 0.20),
    ("heldout_center_upper_left", 0.39, 0.40),
    ("heldout_bottom_left", 0.18, 0.80),
    ("heldout_bottom_right", 0.82, 0.80),
)
FROZEN_CALIBRATION_REFERENCE_TARGETS = (
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


def validation_target_definitions(
    measurement_contract: Mapping[str, object] | None = None,
) -> list[dict[str, Any]]:
    """Return the five frozen held-out targets after integrity checks."""

    active_contract = dict(
        measurement_contract or load_participant_gaze_measurement_contract()
    )
    specification = dict(active_contract.get("target_independence") or {})
    if specification.get("fit_coordinate_system") != SIGNED_SCREEN_COORDINATE_SYSTEM:
        raise ValueError("validation fit coordinate system is invalid")
    tolerance_signed = _finite_number(
        specification.get("overlap_threshold_signed"),
        field="overlap_threshold_signed",
    )
    if not math.isclose(tolerance_signed, 0.2, abs_tol=1e-12):
        raise ValueError("validation target overlap threshold is not frozen at 0.2")
    if specification.get("overlap_boundary_is_failure") is not False:
        raise ValueError("validation target distance boundary must pass at 0.2")
    if (
        specification.get("validation_sample_target_authority")
        != "server_frozen_target_id_mapping"
    ):
        raise ValueError("validation sample target authority is invalid")
    if specification.get("validation_sample_required_target_fields") != [
        "target_id",
        "target_x_norm",
        "target_y_norm",
        "target_x_px",
        "target_y_px",
    ]:
        raise ValueError("validation sample target fields are invalid")

    def coordinate_pairs(raw: object, *, field: str) -> tuple[tuple[float, float], ...]:
        if not isinstance(raw, list):
            raise ValueError(f"{field} must be an array")
        pairs: list[tuple[float, float]] = []
        for index, pair in enumerate(raw):
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or isinstance(pair[0], bool)
                or isinstance(pair[1], bool)
            ):
                raise ValueError(f"{field}[{index}] is invalid")
            x = _finite_number(pair[0], field=f"{field}[{index}].x")
            y = _finite_number(pair[1], field=f"{field}[{index}].y")
            pairs.append((x, y))
        return tuple(pairs)

    grid = coordinate_pairs(
        specification.get("frozen_heldout_grid_4x4"),
        field="frozen_heldout_grid_4x4",
    )
    if grid != FROZEN_HELDOUT_GRID_4X4:
        raise ValueError("held-out 4x4 validation grid changed")
    calibration_reference = coordinate_pairs(
        specification.get("selection_reference_calibration_targets"),
        field="selection_reference_calibration_targets",
    )
    if calibration_reference != FROZEN_CALIBRATION_REFERENCE_TARGETS:
        raise ValueError("validation calibration reference targets changed")

    raw_selected = specification.get("selected_validation_targets")
    if not isinstance(raw_selected, list):
        raise ValueError("selected validation targets must be an array")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_selected):
        if not isinstance(raw, Mapping):
            raise ValueError(f"selected validation target {index} is invalid")
        target_id = str(raw.get("target_id") or "")
        x_fraction = _finite_number(
            raw.get("target_x_viewport_fraction"),
            field=f"selected_validation_targets[{index}].target_x_viewport_fraction",
        )
        y_fraction = _finite_number(
            raw.get("target_y_viewport_fraction"),
            field=f"selected_validation_targets[{index}].target_y_viewport_fraction",
        )
        target_x_norm = _finite_number(
            raw.get("target_x_norm"),
            field=f"selected_validation_targets[{index}].target_x_norm",
        )
        target_y_norm = _finite_number(
            raw.get("target_y_norm"),
            field=f"selected_validation_targets[{index}].target_y_norm",
        )
        if not math.isclose(
            target_x_norm,
            x_fraction * 2.0 - 1.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"selected validation target {target_id} x coordinates differ"
            )
        if not math.isclose(
            target_y_norm,
            y_fraction * 2.0 - 1.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"selected validation target {target_id} y coordinates differ"
            )
        normalized.append(
            {
                "target_id": target_id,
                "target_x_viewport_fraction": x_fraction,
                "target_y_viewport_fraction": y_fraction,
                "target_x_norm": target_x_norm,
                "target_y_norm": target_y_norm,
            }
        )

    frozen_selection = tuple(
        (
            target["target_id"],
            target["target_x_viewport_fraction"],
            target["target_y_viewport_fraction"],
        )
        for target in normalized
    )
    if frozen_selection != FROZEN_VALIDATION_TARGETS:
        raise ValueError("the five selected held-out validation targets changed")
    calibration_signed = [
        (x_fraction * 2.0 - 1.0, y_fraction * 2.0 - 1.0)
        for x_fraction, y_fraction in calibration_reference
    ]
    minimum_distance = min(
        math.hypot(
            target["target_x_norm"] - calibration_x,
            target["target_y_norm"] - calibration_y,
        )
        for target in normalized
        for calibration_x, calibration_y in calibration_signed
    )
    if minimum_distance < tolerance_signed - 1e-12:
        raise ValueError("selected validation targets overlap calibration references")
    return normalized


def validate_general_design(
    protocol: Mapping[str, object] | None = None,
    bank: Mapping[str, object] | None = None,
    measurement_contract: Mapping[str, object] | None = None,
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
    validation_targets: list[dict[str, Any]] = []
    active_measurement_contract: dict[str, Any] = {}
    try:
        active_measurement_contract = dict(
            measurement_contract
            if measurement_contract is not None
            else load_participant_gaze_measurement_contract()
        )
        validation_targets = validation_target_definitions(
            active_measurement_contract
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"validation_target_contract_invalid:{exc}")
    compatibility = dict(
        active_measurement_contract.get("participant_protocol_compatibility") or {}
    )
    compatible_protocol = (
        compatibility.get("general_collection_protocol_id")
        == active_protocol.get("protocol_id")
        and compatibility.get("general_collection_protocol_version")
        == active_protocol.get("protocol_version")
    )
    if not compatible_protocol:
        errors.append("gaze_measurement_contract_protocol_compatibility_invalid")
    if compatibility.get("changes_general_collection_protocol_digest") is not False:
        errors.append("gaze_measurement_contract_must_be_digest_additive")
    capture_specification = dict(
        active_measurement_contract.get("capture_contract") or {}
    )
    must_match = capture_specification.get(
        "must_match_between_calibration_validation_and_reading"
    )
    expected_must_match = [
        "source_aspect_ratio_within_0.02",
        "transport_aspect_ratio_within_0.02",
        "resize_policy",
        "mirror_policy",
        "facing_mode",
    ]
    if must_match != expected_must_match:
        errors.append("capture_contract_must_match_fields_invalid")
    if capture_specification.get("exact_source_resolution_must_match") is not False:
        errors.append("capture_contract_must_not_require_exact_source_resolution")
    expected_validation_points = int(
        dict(active_protocol.get("gaze_quality") or {}).get(
            "independent_validation_points",
            0,
        )
    )
    if validation_targets and len(validation_targets) != expected_validation_points:
        errors.append("additive_validation_target_count_differs_from_general_v1")
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
        "validation_target_count": len(validation_targets),
        "gaze_measurement_contract_sha256": canonical_sha256(
            active_measurement_contract
        ),
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
    viewport_width_px: object,
    viewport_height_px: object,
    measurement_contract: Mapping[str, object] | None = None,
    expected_samples_per_point: int = 3,
    prediction_receipt_status: str = "unavailable",
) -> dict[str, Any]:
    targets = validation_target_definitions(measurement_contract)
    targets_by_id = {target["target_id"]: target for target in targets}
    viewport_width = _finite_number(viewport_width_px, field="viewport_width_px")
    viewport_height = _finite_number(viewport_height_px, field="viewport_height_px")
    if not 1 <= viewport_width <= 16384 or not 1 <= viewport_height <= 16384:
        raise ValueError("validation viewport dimensions are out of range")
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise ValueError("validation samples must be an array")
    if len(samples) != len(targets) * expected_samples_per_point:
        raise ValueError("validation sample count does not match the frozen design")
    if prediction_receipt_status not in {"verified", "unavailable"}:
        raise ValueError("prediction receipt status is invalid")
    grouped_errors: dict[str, list[tuple[float, float, float]]] = {}
    sample_counts: Counter[str] = Counter()
    success_count = 0
    normalized_samples: list[dict[str, Any]] = []
    for raw in samples:
        if not isinstance(raw, Mapping):
            raise ValueError("validation sample must be an object")
        if {"image", "image_data", "frame", "video", "audio"} & set(raw):
            raise ValueError("raw media fields are prohibited in validation samples")
        target_id = str(raw.get("target_id") or "")
        target = targets_by_id.get(target_id)
        if target is None:
            raise ValueError("validation target ID is not in the frozen held-out set")
        reported_x_norm = _finite_number(
            raw.get("target_x_norm"),
            field="target_x_norm",
        )
        reported_y_norm = _finite_number(
            raw.get("target_y_norm"),
            field="target_y_norm",
        )
        if not math.isclose(
            reported_x_norm,
            float(target["target_x_norm"]),
            abs_tol=1e-9,
        ) or not math.isclose(
            reported_y_norm,
            float(target["target_y_norm"]),
            abs_tol=1e-9,
        ):
            raise ValueError(
                "validation signed target coordinates do not match target ID"
            )
        target_x = float(
            math.floor(
                float(target["target_x_viewport_fraction"]) * viewport_width + 0.5
            )
        )
        target_y = float(
            math.floor(
                float(target["target_y_viewport_fraction"]) * viewport_height + 0.5
            )
        )
        reported_x_px = _finite_number(
            raw.get("target_x_px"),
            field="target_x_px",
        )
        reported_y_px = _finite_number(
            raw.get("target_y_px"),
            field="target_y_px",
        )
        if not math.isclose(reported_x_px, target_x, abs_tol=1e-9) or not math.isclose(
            reported_y_px,
            target_y,
            abs_tol=1e-9,
        ):
            raise ValueError("validation pixel target coordinates do not match target ID")
        sample_counts[target_id] += 1
        if raw.get("prediction_success") is True:
            predicted_x = _finite_number(
                raw.get("predicted_x_px"),
                field="predicted_x_px",
            )
            predicted_y = _finite_number(
                raw.get("predicted_y_px"),
                field="predicted_y_px",
            )
            error = math.hypot(predicted_x - target_x, predicted_y - target_y)
            grouped_errors.setdefault(target_id, []).append(
                (predicted_x, predicted_y, error)
            )
            success_count += 1
            normalized_samples.append(
                {
                    "target_id": target_id,
                    "target_x_px": target_x,
                    "target_y_px": target_y,
                    "target_x_norm": float(target["target_x_norm"]),
                    "target_y_norm": float(target["target_y_norm"]),
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
                    "target_x_norm": float(target["target_x_norm"]),
                    "target_y_norm": float(target["target_y_norm"]),
                    "prediction_success": False,
                    "predicted_x_px": None,
                    "predicted_y_px": None,
                    "spatial_error_px": None,
                }
            )
    if set(grouped_errors) != set(targets_by_id):
        raise ValueError("validation target coverage does not match the frozen design")
    if any(
        sample_counts[target_id] != expected_samples_per_point
        for target_id in targets_by_id
    ):
        raise ValueError("validation samples per target do not match the frozen design")
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
        "prediction_receipt_status": prediction_receipt_status,
        "prediction_receipts_verified": prediction_receipt_status == "verified",
        "sample_count": len(samples),
        "successful_sample_count": success_count,
        "target_count": len(grouped_errors),
        "target_coordinate_system": SIGNED_SCREEN_COORDINATE_SYSTEM,
        "validation_targets": deepcopy(targets),
        "viewport": {
            "width_px": int(viewport_width),
            "height_px": int(viewport_height),
        },
        "targets_with_prediction": sum(
            bool(values) for values in grouped_errors.values()
        ),
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


def unavailable_validation_summary(
    *,
    viewport_width_px: object,
    viewport_height_px: object,
    measurement_contract: Mapping[str, object] | None = None,
    reason: str = "prediction_receipts_unavailable",
) -> dict[str, Any]:
    """Return a fail-closed summary without trusting legacy client predictions."""

    targets = validation_target_definitions(measurement_contract)
    viewport_width = _finite_number(viewport_width_px, field="viewport_width_px")
    viewport_height = _finite_number(viewport_height_px, field="viewport_height_px")
    if not 1 <= viewport_width <= 16384 or not 1 <= viewport_height <= 16384:
        raise ValueError("validation viewport dimensions are out of range")
    return {
        "prediction_receipt_status": "unavailable",
        "prediction_receipts_verified": False,
        "prediction_receipt_reasons": [str(reason)],
        "sample_count": 0,
        "expected_sample_count": len(targets) * 3,
        "successful_sample_count": 0,
        "target_count": len(targets),
        "target_coordinate_system": SIGNED_SCREEN_COORDINATE_SYSTEM,
        "validation_targets": deepcopy(targets),
        "viewport": {
            "width_px": int(viewport_width),
            "height_px": int(viewport_height),
        },
        "targets_with_prediction": 0,
        "prediction_success_fraction": 0.0,
        "median_spatial_error_px": None,
        "p90_spatial_error_px": None,
        "precision_rms_px": None,
        "samples": [],
    }


def evaluate_validation_target_independence(
    validation_summary: Mapping[str, object],
    fit_target_contract: Mapping[str, object] | None,
    *,
    measurement_contract: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Compare server-authoritative validation targets with actual fit targets."""

    active_contract = dict(
        measurement_contract or load_participant_gaze_measurement_contract()
    )
    specification = dict(active_contract.get("target_independence") or {})
    tolerance_signed = _finite_number(
        specification.get("overlap_threshold_signed"),
        field="overlap_threshold_signed",
    )
    base = {
        "coordinate_system": SIGNED_SCREEN_COORDINATE_SYSTEM,
        "overlap_threshold_signed": tolerance_signed,
        "overlap_threshold_viewport_fraction": tolerance_signed / 2.0,
        "boundary_at_threshold_is_independent": True,
    }
    if fit_target_contract is None:
        return {
            **base,
            "status": "unavailable",
            "independent": None,
            "minimum_signed_target_distance": None,
            "overlapping_validation_target_ids": [],
            "reasons": ["fit_target_contract_unavailable"],
        }
    try:
        normalized_fit = normalize_fit_target_contract(fit_target_contract)
    except (TypeError, ValueError):
        return {
            **base,
            "status": "failed",
            "independent": False,
            "minimum_signed_target_distance": None,
            "overlapping_validation_target_ids": [],
            "reasons": ["fit_target_contract_invalid"],
        }
    if normalized_fit["status"] != "available":
        return {
            **base,
            "status": "unavailable",
            "independent": None,
            "minimum_signed_target_distance": None,
            "overlapping_validation_target_ids": [],
            "fit_target_count": normalized_fit["target_count"],
            "reasons": [
                "fit_target_contract_unavailable",
                *normalized_fit["reasons"],
            ],
        }
    raw_validation_targets = validation_summary.get("validation_targets")
    if (
        validation_summary.get("target_coordinate_system")
        != SIGNED_SCREEN_COORDINATE_SYSTEM
        or not isinstance(raw_validation_targets, list)
    ):
        return {
            **base,
            "status": "failed",
            "independent": False,
            "minimum_signed_target_distance": None,
            "overlapping_validation_target_ids": [],
            "fit_target_count": normalized_fit["target_count"],
            "reasons": ["validation_target_contract_invalid"],
        }
    validation_targets: list[tuple[str, float, float]] = []
    try:
        for index, raw in enumerate(raw_validation_targets):
            if not isinstance(raw, Mapping):
                raise ValueError("validation target is not an object")
            target_id = str(raw.get("target_id") or "")
            if not target_id:
                raise ValueError("validation target ID is missing")
            target_x = _finite_number(
                raw.get("target_x_norm"),
                field=f"validation_targets[{index}].target_x_norm",
            )
            target_y = _finite_number(
                raw.get("target_y_norm"),
                field=f"validation_targets[{index}].target_y_norm",
            )
            validation_targets.append((target_id, target_x, target_y))
    except (TypeError, ValueError):
        return {
            **base,
            "status": "failed",
            "independent": False,
            "minimum_signed_target_distance": None,
            "overlapping_validation_target_ids": [],
            "fit_target_count": normalized_fit["target_count"],
            "reasons": ["validation_target_contract_invalid"],
        }
    expected_targets = validation_target_definitions(active_contract)
    expected_by_id = {
        target["target_id"]: (
            float(target["target_x_norm"]),
            float(target["target_y_norm"]),
        )
        for target in expected_targets
    }
    observed_by_id = {
        target_id: (target_x, target_y)
        for target_id, target_x, target_y in validation_targets
    }
    if observed_by_id != expected_by_id or len(observed_by_id) != len(
        validation_targets
    ):
        return {
            **base,
            "status": "failed",
            "independent": False,
            "minimum_signed_target_distance": None,
            "overlapping_validation_target_ids": [],
            "fit_target_count": normalized_fit["target_count"],
            "reasons": ["validation_target_contract_invalid"],
        }

    fit_targets = [
        (float(target["target_x_norm"]), float(target["target_y_norm"]))
        for target in normalized_fit["targets"]
    ]
    distances: dict[str, float] = {}
    for target_id, (target_x, target_y) in expected_by_id.items():
        distances[target_id] = min(
            math.hypot(target_x - fit_x, target_y - fit_y)
            for fit_x, fit_y in fit_targets
        )
    overlaps = sorted(
        target_id
        for target_id, distance in distances.items()
        if distance < tolerance_signed - 1e-12
    )
    minimum_distance = min(distances.values())
    return {
        **base,
        "status": "passed" if not overlaps else "failed",
        "independent": not overlaps,
        "minimum_signed_target_distance": minimum_distance,
        "minimum_viewport_fraction_target_distance": minimum_distance / 2.0,
        "fit_target_count": len(fit_targets),
        "validation_target_count": len(validation_targets),
        "overlapping_validation_target_ids": overlaps,
        "reasons": [] if not overlaps else ["validation_targets_overlap_fit_targets"],
    }


def classify_provisional_geometry_quality(
    validation_summary: Mapping[str, object],
    *,
    capture_contract_check: Mapping[str, object] | None = None,
    target_independence_check: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Describe pre-reading geometry using only independent target samples.

    This intentionally omits reading sampling rate, text layout, cognitive
    features, and behavioral outcomes. It reuses the frozen rehearsal spatial
    and prediction-success thresholds only when target independence is proven.
    """

    protocol = load_general_protocol()
    bands = dict(dict(protocol["gaze_quality"])["rehearsal_quality_bands"])
    success = _finite_number(
        validation_summary.get("prediction_success_fraction"),
        field="prediction_success_fraction",
    )
    median_raw = validation_summary.get("median_spatial_error_px")
    p90_raw = validation_summary.get("p90_spatial_error_px")
    median_error = (
        _finite_number(median_raw, field="median_spatial_error_px")
        if median_raw is not None
        else None
    )
    p90_error = (
        _finite_number(p90_raw, field="p90_spatial_error_px")
        if p90_raw is not None
        else None
    )
    receipt_status = str(
        validation_summary.get("prediction_receipt_status") or "unavailable"
    )
    receipts_verified = (
        receipt_status == "verified"
        and validation_summary.get("prediction_receipts_verified") is True
    )

    spatial_band = "behavioral_only"
    reasons: list[str] = []
    word = dict(bands["word_level_candidate"])
    passage = dict(bands["passage_level_only"])
    if (
        median_error is not None
        and p90_error is not None
        and median_error <= float(word["maximum_median_error_px"])
        and p90_error <= float(word["maximum_p90_error_px"])
        and success >= float(word["minimum_success_fraction"])
    ):
        spatial_band = "word_level_candidate"
    elif (
        median_error is not None
        and median_error <= float(passage["maximum_median_error_px"])
        and success >= float(passage["minimum_success_fraction"])
    ):
        spatial_band = "passage_level_only"
    else:
        reasons.append("spatial_or_prediction_success_threshold_not_met")
    if not receipts_verified:
        reasons.append("prediction_receipts_unavailable")

    contract_status = "unavailable"
    contract_compatible: bool | None = None
    if isinstance(capture_contract_check, Mapping):
        contract_status = str(capture_contract_check.get("status") or "unavailable")
        compatible_value = capture_contract_check.get("compatible")
        contract_compatible = (
            compatible_value if isinstance(compatible_value, bool) else None
        )
        if contract_compatible is False:
            reasons.append("capture_contract_mismatch")
        elif contract_compatible is None:
            reasons.append("capture_contract_unavailable")
    else:
        reasons.append("capture_contract_unavailable")

    independence_status = "unavailable"
    target_independent: bool | None = None
    if isinstance(target_independence_check, Mapping):
        independence_status = str(
            target_independence_check.get("status") or "unavailable"
        )
        independent_value = target_independence_check.get("independent")
        target_independent = (
            independent_value if isinstance(independent_value, bool) else None
        )
        if independence_status != "passed" or target_independent is not True:
            if target_independent is False:
                reasons.append("validation_target_independence_failed")
            else:
                reasons.append("validation_target_independence_unavailable")
    else:
        reasons.append("validation_target_independence_unavailable")

    # Geometry-dependent output is fail-closed. Behavioral labels remain usable.
    recommended_mode = (
        spatial_band
        if receipts_verified
        and contract_compatible is True
        and independence_status == "passed"
        and target_independent is True
        else "behavioral_only"
    )
    recommendation = {
        "word_level_candidate": "continue_with_provisional_word_candidate",
        "passage_level_only": "continue_with_passage_level_only",
        "behavioral_only": "recalibration_recommended_before_gaze_use",
    }[recommended_mode]
    return {
        "status": "provisional_sensor_geometry_only",
        "prediction_receipt_status": receipt_status,
        "prediction_receipts_verified": receipts_verified,
        "spatial_band": spatial_band,
        "recommended_gaze_mode": recommended_mode,
        "recommendation": recommendation,
        "median_spatial_error_px": median_error,
        "p90_spatial_error_px": p90_error,
        "prediction_success_fraction": success,
        "capture_contract_status": contract_status,
        "capture_contract_compatible": contract_compatible,
        "target_independence_status": independence_status,
        "validation_targets_independent": target_independent,
        "effective_sampling_hz_evaluated": False,
        "final_quality_pending": True,
        "threshold_status": "rehearsal_descriptive_not_promotion_thresholds",
        "reasons": reasons,
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
                "capture_contract_mismatch",
                "viewport_contract_mismatch",
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

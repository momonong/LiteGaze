"""Pure-stdlib contract for persisted runtime uncertainty observations.

This module owns only frozen-definition verification and label-free response
validation. NumPy fitting and scoring remain in :mod:`uncertainty`, so receipt,
export, and audit code can validate the same evidence without importing the
modeling runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DEFINITION_PATH = Path(__file__).with_name("uncertainty_v2_definition.json")
UNCERTAINTY_SCHEMA_VERSION = 2
OUTPUT_STATUS = "scored_no_threshold"
ABSTENTION_STATUS = "not_selected"
RUNTIME_OBSERVATION_SCHEMA_VERSION = 1
UNAVAILABLE_STATUSES = frozenset(
    {
        "unavailable_capture_failure",
        "unavailable_definition_error",
        "unavailable_definition_mismatch",
        "unavailable_frozen_baseline",
        "unavailable_invalid_observation",
        "unavailable_invalid_state",
        "unavailable_legacy_stage_state",
        "unavailable_missing_score_state",
        "unavailable_not_scored",
        "unavailable_policy_mismatch",
        "unavailable_prediction_failure",
        "unavailable_receipt_missing",
        "unavailable_sensor_failure",
        "unavailable_viewport_missing",
    }
)

__all__ = [
    "ABSTENTION_STATUS",
    "DEFINITION_PATH",
    "OUTPUT_STATUS",
    "RUNTIME_OBSERVATION_SCHEMA_VERSION",
    "UNCERTAINTY_SCHEMA_VERSION",
    "UNAVAILABLE_STATUSES",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_frozen_definition",
    "normalize_uncertainty_observation",
    "unavailable_uncertainty",
    "verified_definition",
]


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value for a stable definition/state hash."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _verify_definition_document(
    document: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    if document.get("schema_version") != UNCERTAINTY_SCHEMA_VERSION:
        raise ValueError("uncertainty definition schema_version must be 2")
    definition = document.get("definition")
    if not isinstance(definition, dict):
        raise ValueError("uncertainty definition document lacks definition")
    observed = canonical_sha256(definition)
    if document.get("definition_sha256") != observed:
        raise ValueError("uncertainty definition hash mismatch")
    if definition.get("output_status") != OUTPUT_STATUS:
        raise ValueError("uncertainty definition must remain scored_no_threshold")
    runtime_policy = definition.get("runtime_policy")
    if not isinstance(runtime_policy, dict) or runtime_policy.get("threshold") is not None:
        raise ValueError("uncertainty definition must not select a threshold")
    return definition, observed


def load_frozen_definition(path: Path | None = None) -> dict[str, Any]:
    """Load and verify the canonical score definition frozen before capture."""

    source = DEFINITION_PATH if path is None else Path(path)
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("uncertainty definition document must be a mapping")
    _verify_definition_document(document)
    return document


def verified_definition(
    document: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Return the verified definition and its canonical SHA-256 digest."""

    verified = load_frozen_definition() if document is None else dict(document)
    return _verify_definition_document(verified)


def _is_boolean(value: Any) -> bool:
    value_type = type(value)
    return isinstance(value, bool) or (
        value_type.__module__.partition(".")[0] == "numpy"
        and value_type.__name__ in {"bool", "bool_"}
    )


def _finite_number(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if _is_boolean(value):
        raise ValueError(f"{field} must not be boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field} must be <= {maximum}")
    return number


def _fixed_length_values(value: Any, *, length: int, field: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ValueError(f"{field} must contain exactly {length} values")
    try:
        observed_length = len(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain exactly {length} values") from exc
    if observed_length != length:
        raise ValueError(f"{field} must contain exactly {length} values")
    try:
        return [value[index] for index in range(length)]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"{field} must contain exactly {length} values") from exc


def _normalized_covariance(value: Any, *, field: str) -> list[list[float]]:
    rows = _fixed_length_values(value, length=2, field=field)
    first = _fixed_length_values(rows[0], length=2, field=field)
    second = _fixed_length_values(rows[1], length=2, field=field)
    a = _finite_number(first[0], field=field)
    b = _finite_number(first[1], field=field)
    c = _finite_number(second[0], field=field)
    d = _finite_number(second[1], field=field)

    tolerance = 1e-9
    if not math.isclose(b, c, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{field} must be symmetric within atol={tolerance}")
    off_diagonal = b * 0.5 + c * 0.5
    midpoint = a * 0.5 + d * 0.5
    half_difference = a * 0.5 - d * 0.5
    minimum_eigenvalue = midpoint - math.hypot(half_difference, off_diagonal)
    if minimum_eigenvalue < -tolerance:
        raise ValueError(f"{field} must be PSD within eigmin tolerance={tolerance}")
    return [[a, off_diagonal], [off_diagonal, d]]


def _viewport_covariance(
    covariance_norm: Sequence[Sequence[float]],
    *,
    viewport_width: float,
    viewport_height: float,
) -> list[list[float]]:
    scale_x = viewport_width * 0.5
    scale_y = viewport_height * 0.5
    expected = [
        [
            covariance_norm[0][0] * scale_x * scale_x,
            covariance_norm[0][1] * scale_x * scale_y,
        ],
        [
            covariance_norm[1][0] * scale_x * scale_y,
            covariance_norm[1][1] * scale_y * scale_y,
        ],
    ]
    if any(not math.isfinite(value) for row in expected for value in row):
        raise ValueError("pixel covariance viewport transform must be finite")
    return expected


def _matrices_close(
    observed: Sequence[Sequence[float]],
    expected: Sequence[Sequence[float]],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> bool:
    return all(
        math.isclose(
            observed[row][column],
            expected[row][column],
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        )
        for row in range(2)
        for column in range(2)
    )


def unavailable_uncertainty(status: str, reason: str) -> dict[str, Any]:
    """Create the stable, label-free unavailable observation shape."""

    normalized_status = str(status).strip()
    if normalized_status not in UNAVAILABLE_STATUSES or re.fullmatch(
        r"unavailable_[a-z0-9_]{1,64}", normalized_status
    ) is None:
        raise ValueError("unavailable uncertainty status is not allowlisted")
    reason_text = str(reason)
    reason_text = "".join(
        " " if ord(character) < 32 or 127 <= ord(character) <= 159 else character
        for character in reason_text
    )
    reason_text = re.sub(
        r"(?i)\b[a-z]:[\\/][^\s]+",
        "[redacted_path]",
        reason_text,
    )
    normalized_reason = " ".join(reason_text.split())[:240].strip()
    if not normalized_reason:
        raise ValueError("unavailable uncertainty reason must not be blank")
    return {
        "schema_version": RUNTIME_OBSERVATION_SCHEMA_VERSION,
        "status": normalized_status,
        "reason": normalized_reason,
    }


def normalize_uncertainty_observation(
    observation: Mapping[str, Any],
    *,
    viewport: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Validate and canonicalize the stable runtime uncertainty response.

    The function accepts no label, target, residual, text, or outcome argument.
    It is suitable for inference, receipt persistence, export, and measurement
    audit validation without importing NumPy.
    """

    if not isinstance(observation, Mapping):
        raise ValueError("uncertainty observation must be a mapping")
    schema_version = observation.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("uncertainty observation schema_version must be 1")
    status = str(observation.get("status", "")).strip()
    if status.startswith("unavailable_"):
        return unavailable_uncertainty(status, str(observation.get("reason", "")))
    if status != OUTPUT_STATUS:
        raise ValueError("uncertainty status is not recognized")

    _, frozen_definition_sha = verified_definition()
    definition_sha = str(observation.get("definition_sha256", "")).strip()
    if len(definition_sha) != 64 or any(
        character not in "0123456789abcdef" for character in definition_sha
    ):
        raise ValueError("uncertainty definition_sha256 must be 64 lowercase hex chars")
    if definition_sha != frozen_definition_sha:
        raise ValueError("uncertainty definition_sha256 does not match frozen v2")
    score = _finite_number(
        observation.get("score"),
        field="uncertainty score",
        minimum=0.0,
        maximum=1.0,
    )

    raw_components = observation.get("components")
    if not isinstance(raw_components, Mapping):
        raise ValueError("uncertainty components must be a mapping")
    components: dict[str, dict[str, float]] = {}
    for name in ("ood", "leverage", "disagreement"):
        component = raw_components.get(name)
        if not isinstance(component, Mapping):
            raise ValueError(f"uncertainty component {name} is missing")
        value = _finite_number(
            component.get("value"),
            field=f"uncertainty component {name}",
            minimum=0.0,
        )
        percentile = _finite_number(
            component.get("percentile"),
            field=f"uncertainty component {name} percentile",
            minimum=0.0,
            maximum=1.0,
        )
        components[name] = {"value": value, "percentile": percentile}

    expected_score = max(component["percentile"] for component in components.values())
    if not math.isclose(score, expected_score, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("uncertainty score must equal the maximum component percentile")

    abstention = observation.get("abstention")
    if (
        not isinstance(abstention, Mapping)
        or abstention.get("status") != ABSTENTION_STATUS
        or abstention.get("threshold") is not None
    ):
        raise ValueError("uncertainty abstention must remain not_selected/null")

    if viewport is None:
        raise ValueError("scored uncertainty requires viewport (width, height)")
    viewport_values = _fixed_length_values(viewport, length=2, field="viewport")
    viewport_width = _finite_number(
        viewport_values[0], field="viewport width", minimum=sys.float_info.min
    )
    viewport_height = _finite_number(
        viewport_values[1], field="viewport height", minimum=sys.float_info.min
    )
    covariance_norm = _normalized_covariance(
        observation.get("jackknife_disagreement_covariance_norm"),
        field="jackknife_disagreement_covariance_norm",
    )
    covariance_px = _normalized_covariance(
        observation.get("jackknife_disagreement_covariance_px"),
        field="jackknife_disagreement_covariance_px",
    )

    mean_normalized_variance = (
        covariance_norm[0][0] * 0.5 + covariance_norm[1][1] * 0.5
    )
    expected_disagreement = math.sqrt(max(mean_normalized_variance, 0.0))
    if not math.isclose(
        components["disagreement"]["value"],
        expected_disagreement,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "disagreement value must equal sqrt(trace(covariance_norm) / 2)"
        )

    expected_covariance_px = _viewport_covariance(
        covariance_norm,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
    )
    if not _matrices_close(
        covariance_px,
        expected_covariance_px,
        relative_tolerance=1e-9,
        absolute_tolerance=1e-6,
    ):
        raise ValueError("pixel covariance does not match viewport transform")

    return {
        "schema_version": RUNTIME_OBSERVATION_SCHEMA_VERSION,
        "status": OUTPUT_STATUS,
        "definition_sha256": definition_sha,
        "score": score,
        "components": components,
        "jackknife_disagreement_covariance_norm": covariance_norm,
        "jackknife_disagreement_covariance_px": covariance_px,
        "abstention": {"status": ABSTENTION_STATUS, "threshold": None},
    }

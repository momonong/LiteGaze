"""Server-side validation of the structural browser capture gate.

Client evidence can prove only that the shipped browser state machine reported
the expected target geometry, focus, visibility, and dwell.  It is not visual
attestation and is deliberately discarded before sensor inference and ledger
construction.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


GATE_SCHEMA_VERSION = 1
GATE_IMPLEMENTATION_ID = "browser-visible-focus-dwell-render-v1"
MINIMUM_DWELL_MS = 900.0
MINIMUM_STABLE_RENDER_FRAMES = 2
TARGET_TOLERANCE_PX = 3.0
VIEWPORT_TOLERANCE_PX = 0.5
DPR_TOLERANCE = 0.001

CLIENT_GATE_FIELDS = frozenset(
    {
        "schema_version",
        "implementation_id",
        "sequence_index",
        "visibility_state",
        "document_focused",
        "viewport_width",
        "viewport_height",
        "device_pixel_ratio",
        "rendered_target_center_x_px",
        "rendered_target_center_y_px",
        "minimum_dwell_ms",
        "observed_stable_dwell_ms",
        "stable_render_frame_count",
        "structural_browser_gate_only",
        "visual_attestation_claimed",
    }
)


class MeasurementBrowserGateError(ValueError):
    """Raised when structural browser evidence fails closed."""


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise MeasurementBrowserGateError(f"{field} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementBrowserGateError(f"{field} must be finite") from exc
    if not math.isfinite(number):
        raise MeasurementBrowserGateError(f"{field} must be finite")
    return number


def _positive(value: object, *, field: str) -> float:
    number = _finite(value, field=field)
    if number <= 0:
        raise MeasurementBrowserGateError(f"{field} must be positive")
    return number


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    number = _finite(value, field=field)
    if not number.is_integer() or number < minimum:
        raise MeasurementBrowserGateError(f"{field} must be an integer >= {minimum}")
    return int(number)


def _runtime_viewport(value: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise MeasurementBrowserGateError("runtime viewport is unavailable")
    required = {"width", "height", "device_pixel_ratio"}
    if set(value) != required:
        raise MeasurementBrowserGateError("runtime viewport fields are invalid")
    return {
        "width": _positive(value["width"], field="runtime viewport width"),
        "height": _positive(value["height"], field="runtime viewport height"),
        "device_pixel_ratio": _positive(
            value["device_pixel_ratio"],
            field="runtime device pixel ratio",
        ),
    }


def validate_measurement_browser_gate(
    client_gate: Mapping[str, Any],
    *,
    schedule_row: Mapping[str, Any],
    runtime_viewport: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize client evidence against server-owned labels."""

    if not isinstance(client_gate, Mapping):
        raise MeasurementBrowserGateError("client_gate must be an object")
    if set(client_gate) != CLIENT_GATE_FIELDS:
        raise MeasurementBrowserGateError("client_gate fields are invalid")
    if not isinstance(schedule_row, Mapping):
        raise MeasurementBrowserGateError("server schedule row is unavailable")
    viewport = _runtime_viewport(runtime_viewport)

    if client_gate.get("schema_version") != GATE_SCHEMA_VERSION:
        raise MeasurementBrowserGateError("client_gate schema changed")
    if client_gate.get("implementation_id") != GATE_IMPLEMENTATION_ID:
        raise MeasurementBrowserGateError("client_gate implementation changed")
    sequence_index = _integer(
        client_gate.get("sequence_index"),
        field="client sequence index",
    )
    server_sequence = _integer(
        schedule_row.get("sequence_index"),
        field="server sequence index",
    )
    if sequence_index != server_sequence:
        raise MeasurementBrowserGateError("client_gate belongs to another schedule row")
    if client_gate.get("visibility_state") != "visible":
        raise MeasurementBrowserGateError("document was not visible at capture")
    if client_gate.get("document_focused") is not True:
        raise MeasurementBrowserGateError("document was not focused at capture")
    if client_gate.get("structural_browser_gate_only") is not True:
        raise MeasurementBrowserGateError("structural gate boundary changed")
    if client_gate.get("visual_attestation_claimed") is not False:
        raise MeasurementBrowserGateError("visual attestation must remain false")

    observed_viewport = {
        "width": _positive(
            client_gate.get("viewport_width"),
            field="client viewport width",
        ),
        "height": _positive(
            client_gate.get("viewport_height"),
            field="client viewport height",
        ),
        "device_pixel_ratio": _positive(
            client_gate.get("device_pixel_ratio"),
            field="client device pixel ratio",
        ),
    }
    if (
        abs(observed_viewport["width"] - viewport["width"])
        > VIEWPORT_TOLERANCE_PX
        or abs(observed_viewport["height"] - viewport["height"])
        > VIEWPORT_TOLERANCE_PX
        or abs(
            observed_viewport["device_pixel_ratio"]
            - viewport["device_pixel_ratio"]
        )
        > DPR_TOLERANCE
    ):
        raise MeasurementBrowserGateError("viewport changed after run binding")

    minimum_dwell = _finite(
        client_gate.get("minimum_dwell_ms"),
        field="minimum dwell",
    )
    observed_dwell = _finite(
        client_gate.get("observed_stable_dwell_ms"),
        field="observed stable dwell",
    )
    stable_frames = _integer(
        client_gate.get("stable_render_frame_count"),
        field="stable render frame count",
    )
    if minimum_dwell != MINIMUM_DWELL_MS:
        raise MeasurementBrowserGateError("minimum dwell contract changed")
    if observed_dwell < MINIMUM_DWELL_MS:
        raise MeasurementBrowserGateError("target dwell was too short")
    if stable_frames < MINIMUM_STABLE_RENDER_FRAMES:
        raise MeasurementBrowserGateError("target was not stable for two render frames")

    target_x_fraction = _finite(
        schedule_row.get("target_x_viewport_fraction"),
        field="server target x fraction",
    )
    target_y_fraction = _finite(
        schedule_row.get("target_y_viewport_fraction"),
        field="server target y fraction",
    )
    if not 0 <= target_x_fraction <= 1 or not 0 <= target_y_fraction <= 1:
        raise MeasurementBrowserGateError("server target fraction is invalid")
    rendered_x = _finite(
        client_gate.get("rendered_target_center_x_px"),
        field="rendered target x",
    )
    rendered_y = _finite(
        client_gate.get("rendered_target_center_y_px"),
        field="rendered target y",
    )
    expected_x = target_x_fraction * viewport["width"]
    expected_y = target_y_fraction * viewport["height"]
    if (
        abs(rendered_x - expected_x) > TARGET_TOLERANCE_PX
        or abs(rendered_y - expected_y) > TARGET_TOLERANCE_PX
    ):
        raise MeasurementBrowserGateError("rendered target differs from server schedule")

    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "implementation_id": GATE_IMPLEMENTATION_ID,
        "status": "structural_gate_passed",
        "sequence_index": sequence_index,
        "minimum_dwell_ms": MINIMUM_DWELL_MS,
        "observed_stable_dwell_ms": observed_dwell,
        "stable_render_frame_count": stable_frames,
        "target_center_error_px": [
            rendered_x - expected_x,
            rendered_y - expected_y,
        ],
        "structural_browser_gate_only": True,
        "visual_attestation_claimed": False,
        "sensor_model_input": False,
        "ledger_persistence_authorized": False,
    }

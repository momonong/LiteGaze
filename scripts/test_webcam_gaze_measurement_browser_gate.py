"""Pure-standard-library tests for server-owned browser-gate validation."""

from __future__ import annotations

import unittest

from core.gaze_core.measurement_browser_gate import (
    MeasurementBrowserGateError,
    validate_measurement_browser_gate,
)


ROW = {
    "sequence_index": 7,
    "target_x_viewport_fraction": 0.71,
    "target_y_viewport_fraction": 0.3,
}
VIEWPORT = {"width": 1440, "height": 900, "device_pixel_ratio": 1.25}


def _gate(**updates: object) -> dict:
    gate = {
        "schema_version": 1,
        "implementation_id": "browser-visible-focus-dwell-render-v1",
        "sequence_index": 7,
        "visibility_state": "visible",
        "document_focused": True,
        "viewport_width": 1440,
        "viewport_height": 900,
        "device_pixel_ratio": 1.25,
        "rendered_target_center_x_px": 1022.4,
        "rendered_target_center_y_px": 270.0,
        "minimum_dwell_ms": 900,
        "observed_stable_dwell_ms": 932.5,
        "stable_render_frame_count": 4,
        "structural_browser_gate_only": True,
        "visual_attestation_claimed": False,
    }
    gate.update(updates)
    return gate


class MeasurementBrowserGateTests(unittest.TestCase):
    def test_expected_structural_evidence_passes_without_model_or_ledger_authority(self) -> None:
        result = validate_measurement_browser_gate(
            _gate(),
            schedule_row=ROW,
            runtime_viewport=VIEWPORT,
        )
        self.assertEqual(result["status"], "structural_gate_passed")
        self.assertFalse(result["visual_attestation_claimed"])
        self.assertFalse(result["sensor_model_input"])
        self.assertFalse(result["ledger_persistence_authorized"])

    def test_visibility_focus_and_attestation_fail_closed(self) -> None:
        for update in (
            {"visibility_state": "hidden"},
            {"document_focused": False},
            {"structural_browser_gate_only": False},
            {"visual_attestation_claimed": True},
        ):
            with self.subTest(update=update):
                with self.assertRaises(MeasurementBrowserGateError):
                    validate_measurement_browser_gate(
                        _gate(**update),
                        schedule_row=ROW,
                        runtime_viewport=VIEWPORT,
                    )

    def test_viewport_sequence_and_rendered_target_are_server_bound(self) -> None:
        for update in (
            {"sequence_index": 8},
            {"viewport_width": 1439},
            {"viewport_height": 899},
            {"device_pixel_ratio": 1.5},
            {"rendered_target_center_x_px": 1000},
            {"rendered_target_center_y_px": 280},
        ):
            with self.subTest(update=update):
                with self.assertRaises(MeasurementBrowserGateError):
                    validate_measurement_browser_gate(
                        _gate(**update),
                        schedule_row=ROW,
                        runtime_viewport=VIEWPORT,
                    )

    def test_dwell_and_two_frame_minimum_are_exact(self) -> None:
        for update in (
            {"minimum_dwell_ms": 899},
            {"minimum_dwell_ms": 901},
            {"observed_stable_dwell_ms": 899.9},
            {"stable_render_frame_count": 1},
        ):
            with self.subTest(update=update):
                with self.assertRaises(MeasurementBrowserGateError):
                    validate_measurement_browser_gate(
                        _gate(**update),
                        schedule_row=ROW,
                        runtime_viewport=VIEWPORT,
                    )

    def test_extra_raw_label_or_prior_fields_are_rejected(self) -> None:
        for field, value in (
            ("target_x_norm", 0.4),
            ("cursor_position", [100, 200]),
            ("cognitive_profile", {"theta": 1.0}),
            ("image_data", "raw"),
            ("model_name", "chosen"),
        ):
            with self.subTest(field=field):
                with self.assertRaises(MeasurementBrowserGateError):
                    validate_measurement_browser_gate(
                        _gate(**{field: value}),
                        schedule_row=ROW,
                        runtime_viewport=VIEWPORT,
                    )


if __name__ == "__main__":
    unittest.main()

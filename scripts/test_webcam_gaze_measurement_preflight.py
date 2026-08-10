"""Pure-standard-library tests for target-free camera readiness."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from core.gaze_core.measurement_preflight import (
    MeasurementPreflightAuthenticationError,
    MeasurementPreflightRegistry,
    MeasurementPreflightStateError,
    MeasurementPreflightValidationError,
)


def _capture_contract(*, width: int = 1280, frame_rate: float = 30.0) -> dict:
    return {
        "schema_version": 1,
        "intent_width_px": 1280,
        "intent_height_px": 720,
        "intent_frame_rate_hz": 30,
        "source_width_px": width,
        "source_height_px": 720,
        "source_frame_rate_hz": frame_rate,
        "transport_width_px": 640,
        "transport_height_px": round(640 * 720 / width),
        "resize_policy": "fit_width_preserve_aspect",
        "mime_type": "image/jpeg",
        "jpeg_quality": 0.8,
        "mirror_applied": False,
        "facing_mode": "user",
    }


VIEWPORT = {"width": 1440, "height": 900, "device_pixel_ratio": 1.25}
BASE_BUNDLE = {
    "model_id": "unigaze_b16_joint@" + "1" * 40,
    "model_name": "unigaze_b16_joint",
    "model_sha256": "a" * 64,
    "bundle_sha256": "a" * 64,
    "checkpoint_sha256": "b" * 64,
}


class Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


class MeasurementPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="lexigaze-preflight-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.clock = Clock()
        self.inference_payloads: list[dict] = []
        self.responses: list[tuple[dict, int]] = []

        def infer(_root: Path, payload: dict) -> tuple[dict, int]:
            self.inference_payloads.append(dict(payload))
            if self.responses:
                return self.responses.pop(0)
            return {
                "ok": True,
                "model_name": "before",
                "screen_xy_norm": [0.0, 0.0],
            }, 200

        def validate(image_data: str, _contract: dict) -> str:
            return hashlib.sha256(image_data.encode("utf-8")).hexdigest()

        self.registry = MeasurementPreflightRegistry(
            self.root,
            base_inference_bundle=BASE_BUNDLE,
            infer=infer,
            frame_validator=validate,
            monotonic=self.clock,
        )

    def _start(self) -> dict:
        return self.registry.start(
            capture_contract=_capture_contract(),
            viewport=VIEWPORT,
        )

    def test_three_distinct_successes_are_single_use_and_cpu_fixed(self) -> None:
        started = self._start()
        token = started.pop("preflight_token")
        self.assertNotIn(token, repr(self.registry._entries))
        results = [
            self.registry.submit_frame(
                token,
                image_data=f"frame-{index}",
                capture_contract=_capture_contract(),
            )
            for index in range(3)
        ]
        self.assertEqual([row["consecutive_successes"] for row in results], [1, 2, 3])
        self.assertTrue(results[-1]["ready"])
        for payload in self.inference_payloads:
            self.assertEqual(
                set(payload),
                {
                    "image_data",
                    "capture_contract",
                    "model_name",
                    "viewport_width",
                    "viewport_height",
                    "allow_cuda",
                },
            )
            self.assertEqual(payload["model_name"], "before")
            self.assertIs(payload["allow_cuda"], False)
            self.assertFalse(any("target" in key for key in payload))
        proof = self.registry.consume(
            token,
            capture_contract=_capture_contract(),
            viewport=VIEWPORT,
        )
        self.assertEqual(proof["consecutive_successes"], 3)
        self.assertEqual(proof["distinct_frame_count"], 3)
        self.assertTrue(proof["target_free"])
        self.assertFalse(proof["persistent_storage_used"])
        self.assertEqual(
            proof["base_inference_bundle"],
            BASE_BUNDLE,
        )
        receipt = dict(proof)
        stored_receipt_sha = receipt.pop("receipt_sha256")
        from core.gaze_core.measurement_schedule import canonical_sha256

        self.assertEqual(stored_receipt_sha, canonical_sha256(receipt))
        with self.assertRaises(MeasurementPreflightAuthenticationError):
            self.registry.consume(
                token,
                capture_contract=_capture_contract(),
                viewport=VIEWPORT,
            )

    def test_exact_last_frame_replay_is_idempotent_and_not_reinferred(self) -> None:
        token = self._start()["preflight_token"]
        first = self.registry.submit_frame(
            token,
            image_data="same-frame",
            capture_contract=_capture_contract(),
        )
        replay = self.registry.submit_frame(
            token,
            image_data="same-frame",
            capture_contract=_capture_contract(),
        )
        self.assertEqual(first["consecutive_successes"], 1)
        self.assertEqual(replay["consecutive_successes"], 1)
        self.assertTrue(replay["idempotent"])
        self.assertEqual(len(self.inference_payloads), 1)

    def test_nonconsecutive_replay_breaks_success_streak(self) -> None:
        token = self._start()["preflight_token"]
        for frame in ("frame-a", "frame-b"):
            self.registry.submit_frame(
                token,
                image_data=frame,
                capture_contract=_capture_contract(),
            )
        replay = self.registry.submit_frame(
            token,
            image_data="frame-a",
            capture_contract=_capture_contract(),
        )
        after = self.registry.submit_frame(
            token,
            image_data="frame-c",
            capture_contract=_capture_contract(),
        )
        self.assertEqual(replay["classification"], "nonconsecutive_frame_replay")
        self.assertEqual(replay["consecutive_successes"], 0)
        self.assertEqual(after["consecutive_successes"], 1)
        self.assertFalse(after["ready"])

    def test_success_response_must_be_http_200_from_fixed_before_model(self) -> None:
        token = self._start()["preflight_token"]
        self.responses.extend(
            [
                ({"ok": True, "model_name": "before"}, 201),
                ({"ok": True, "model_name": "personalized"}, 200),
            ]
        )
        wrong_status = self.registry.submit_frame(
            token,
            image_data="wrong-status",
            capture_contract=_capture_contract(),
        )
        wrong_model = self.registry.submit_frame(
            token,
            image_data="wrong-model",
            capture_contract=_capture_contract(),
        )
        self.assertEqual(
            wrong_status["classification"],
            "preflight_inference_contract_error",
        )
        self.assertEqual(
            wrong_model["classification"],
            "preflight_inference_contract_error",
        )
        self.assertEqual(wrong_model["consecutive_successes"], 0)

    def test_sensor_failure_and_hard_error_break_consecutive_sequence(self) -> None:
        token = self._start()["preflight_token"]
        self.registry.submit_frame(
            token, image_data="pass-1", capture_contract=_capture_contract()
        )
        self.responses.extend(
            [
                (
                    {
                        "ok": False,
                        "failure_stage": "attributable_sensor_failure",
                        "failure_code": "no_face_detected",
                    },
                    400,
                ),
                (
                    {
                        "ok": False,
                        "failure_stage": "inference_hard_error",
                        "error": "runtime unavailable",
                    },
                    500,
                ),
            ]
        )
        no_face = self.registry.submit_frame(
            token, image_data="no-face", capture_contract=_capture_contract()
        )
        hard = self.registry.submit_frame(
            token, image_data="hard", capture_contract=_capture_contract()
        )
        self.assertEqual(no_face["classification"], "attributable_sensor_failure")
        self.assertEqual(no_face["consecutive_successes"], 0)
        self.assertTrue(no_face["retryable"])
        self.assertEqual(hard["classification"], "inference_hard_error")
        self.assertEqual(hard["consecutive_successes"], 0)

    def test_capture_contract_change_resets_and_fails_closed(self) -> None:
        token = self._start()["preflight_token"]
        self.registry.submit_frame(
            token, image_data="pass", capture_contract=_capture_contract()
        )
        with self.assertRaises(MeasurementPreflightValidationError):
            self.registry.submit_frame(
                token,
                image_data="changed",
                capture_contract={**_capture_contract(), "mirror_applied": True},
            )
        with self.assertRaises(MeasurementPreflightStateError):
            self.registry.consume(
                token,
                capture_contract=_capture_contract(),
                viewport=VIEWPORT,
            )

    def test_resolution_and_frame_rate_change_are_warning_only_when_geometry_matches(self) -> None:
        token = self._start()["preflight_token"]
        changed = _capture_contract(frame_rate=20.0)
        first = self.registry.submit_frame(
            token,
            image_data="warning-frame",
            capture_contract=changed,
        )
        self.assertTrue(first["ok"])
        self.assertIn("source_frame_rate_changed", first["capture_contract_warnings"])

    def test_consume_rejects_viewport_change(self) -> None:
        token = self._start()["preflight_token"]
        for index in range(3):
            self.registry.submit_frame(
                token,
                image_data=f"frame-{index}",
                capture_contract=_capture_contract(),
            )
        with self.assertRaises(MeasurementPreflightValidationError):
            self.registry.consume(
                token,
                capture_contract=_capture_contract(),
                viewport={**VIEWPORT, "width": 1439},
            )

    def test_expiry_and_capacity_are_bounded(self) -> None:
        bounded = MeasurementPreflightRegistry(
            self.root,
            base_inference_bundle=BASE_BUNDLE,
            infer=lambda _root, _payload: (
                {"ok": True, "model_name": "before"},
                200,
            ),
            frame_validator=lambda image, _contract: hashlib.sha256(
                image.encode("utf-8")
            ).hexdigest(),
            monotonic=self.clock,
            ttl_seconds=10,
            max_active=1,
        )
        first = bounded.start(
            capture_contract=_capture_contract(), viewport=VIEWPORT
        )
        with self.assertRaises(MeasurementPreflightStateError):
            bounded.start(capture_contract=_capture_contract(), viewport=VIEWPORT)
        self.clock.value += 11
        second = bounded.start(
            capture_contract=_capture_contract(), viewport=VIEWPORT
        )
        self.assertNotEqual(first["preflight_id"], second["preflight_id"])
        with self.assertRaises(MeasurementPreflightAuthenticationError):
            bounded.submit_frame(
                first["preflight_token"],
                image_data="expired",
                capture_contract=_capture_contract(),
            )

    def test_registry_persists_only_hashes_and_never_raw_media(self) -> None:
        token = self._start()["preflight_token"]
        raw_marker = "RAW-MEDIA-MUST-NOT-PERSIST"
        self.registry.submit_frame(
            token,
            image_data=raw_marker,
            capture_contract=_capture_contract(),
        )
        self.assertNotIn(raw_marker, repr(self.registry._entries))


if __name__ == "__main__":
    unittest.main()

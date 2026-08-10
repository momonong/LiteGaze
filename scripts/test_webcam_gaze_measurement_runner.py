"""CPU-only focused tests for the dedicated measurement runner adapter."""

from __future__ import annotations

import base64
from contextlib import nullcontext
import hashlib
import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from core.gaze_core.capture_contract import normalize_capture_contract
from core.gaze_core.measurement_runner import (
    CAPTURE_SOURCE,
    MeasurementRunner,
    MeasurementRunnerError,
)
from core.gaze_core.measurement_run_store import MeasurementRunIntegrityError
from core.gaze_core.measurement_schedule import canonical_sha256
from core.gaze_core.training import (
    MEASUREMENT_TRAINING_BINDING_TYPE,
    _canonical_sha256 as training_canonical_sha256,
    _read_measurement_training_image,
    _validated_measurement_training_binding,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _contract(
    *,
    source_width: int = 1280,
    source_height: int = 720,
    fps: float = 30.0,
    transport_width: int = 640,
    transport_height: int | None = None,
    mirror: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "intent_width_px": 1280,
        "intent_height_px": 720,
        "intent_frame_rate_hz": 30.0,
        "source_width_px": source_width,
        "source_height_px": source_height,
        "source_frame_rate_hz": fps,
        "transport_width_px": transport_width,
        "transport_height_px": (
            transport_height
            if transport_height is not None
            else round(transport_width * source_height / source_width)
        ),
        "resize_policy": "fit_width_preserve_aspect",
        "mime_type": "image/jpeg",
        "jpeg_quality": 0.8,
        "mirror_applied": mirror,
        "facing_mode": "user",
    }


def _bundle() -> dict[str, str]:
    return {
        "model_id": "unigaze_b16_joint@" + "1" * 40,
        "model_name": "unigaze_b16_joint",
        "model_sha256": "a" * 64,
        "bundle_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
    }


def _preflight(
    contract: dict[str, Any], viewport: dict[str, float], bundle: dict[str, str]
) -> dict[str, Any]:
    normalized = normalize_capture_contract(contract)
    proof: dict[str, Any] = {
        "schema_version": 1,
        "implementation_id": "target-free-camera-readiness-v1",
        "preflight_id": "PF-" + "9" * 24,
        "capture_contract_sha256": canonical_sha256(normalized),
        "consumed_capture_contract_sha256": canonical_sha256(normalized),
        "capture_contract_warnings": [],
        "viewport_sha256": canonical_sha256(viewport),
        "consecutive_successes": 3,
        "distinct_frame_count": 3,
        "distinct_frame_set_sha256": "c" * 64,
        "target_free": True,
        "persistent_storage_used": False,
        "fixed_model_name": "before",
        "base_inference_bundle": dict(bundle),
        "allow_cuda": False,
        "measurement_claim_authorized": False,
        "receipt_type": "target_free_camera_readiness_receipt_v1",
    }
    proof["receipt_sha256"] = canonical_sha256(proof)
    return proof


class _Harness:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.bundle = _bundle()
        self.base_calls = 0
        self.predict_calls = 0
        self.save_calls = 0
        self.train_calls = 0
        self.train_mode = "success"
        self.last_training_payload: dict[str, Any] | None = None
        self.reset_calls = 0
        self.create_session_calls = 0
        self.purge_fails = False
        self.face_detected = True
        self.save_failures_remaining = 0
        self.predict_mode = "success"
        self.decoded_width = 640
        self.decoded_height = 360
        self.raw = b"server-decoded-frame"
        self.clock = 1000.0

    def base_bundle(self, code_root: Path) -> dict[str, str]:
        self.base_calls += 1
        if code_root != REPOSITORY_ROOT:
            raise AssertionError("code_root drifted into data_root")
        return dict(self.bundle)

    @staticmethod
    def verify_bundle(bundle: dict[str, Any], code_root: Path) -> dict[str, Any]:
        return {"status": "passed", "bundle_sha256": bundle["bundle_sha256"]}

    def reset_cache(self) -> None:
        self.reset_calls += 1

    def create_session(self, root: Path, **kwargs: Any) -> dict[str, Any]:
        self.create_session_calls += 1
        run_id = str(kwargs["capture_run_id"])
        session_id = "session_" + run_id.replace("-", "_")
        session_dir = root / "data" / "sessions" / session_id
        for name in ("raw", "crop", "normalized_face"):
            (session_dir / name).mkdir(parents=True, exist_ok=True)
        metadata = {
            "session_id": session_id,
            "participant_id": f"wgmc_{run_id}",
            "capture_run_id": run_id,
            "capture_source": CAPTURE_SOURCE,
        }
        (session_dir / "session.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        return {"ok": True, "session_id": session_id, "capture_run_id": run_id}

    def decode(self, _image_data: str) -> tuple[bytes, int, int]:
        return self.raw, self.decoded_width, self.decoded_height

    def predict(self, _root: Path, payload: dict[str, Any]) -> tuple[dict, int]:
        self.predict_calls += 1
        if payload.get("allow_cuda") is not False:
            raise AssertionError("runner enabled CUDA")
        forbidden = {"target", "text", "cursor", "cognitive_profile"} & set(payload)
        if forbidden:
            raise AssertionError(f"forbidden inference fields: {forbidden}")
        if self.predict_mode == "hard":
            return {"ok": False, "failure_stage": "model_error", "error": "boom"}, 500
        if self.predict_mode == "no_face":
            return {
                "ok": False,
                "failure_code": "no_face_detected",
                "failure_stage": "attributable_sensor_failure",
                "error": "no face",
                "model_name": payload["model_name"],
            }, 400
        response = {
            "ok": True,
            "gaze_pitch_yaw": [0.01, -0.02],
            "screen_xy_px": [400.0, 300.0],
            "head_pose_pitch_yaw": [0.02, -0.01],
            "face_bbox": {"x_norm": 0.2, "y_norm": 0.1, "w_norm": 0.5, "h_norm": 0.7},
            "uncertainty": {"status": "scored_no_threshold", "score": 0.2},
            "model_name": payload["model_name"],
        }
        if self.predict_mode == "wrong_model":
            response["model_name"] = "wrong-model"
        elif self.predict_mode == "missing_model":
            response.pop("model_name")
        elif self.predict_mode == "malformed_no_face":
            response = {
                "ok": False,
                "failure_code": "no_face_detected",
                "failure_stage": "attributable_sensor_failure",
                "model_name": payload["model_name"],
            }
            return response, 500
        return response, 200

    def save(self, root: Path, payload: dict[str, Any]) -> tuple[dict, int]:
        self.save_calls += 1
        if self.save_failures_remaining:
            self.save_failures_remaining -= 1
            raise RuntimeError("injected save crash")
        session_dir = root / "data" / "sessions" / str(payload["session_id"])
        manifest_path = session_dir / "manifest.jsonl"
        existing = manifest_path.read_text(encoding="utf-8").splitlines() if manifest_path.exists() else []
        index = len(existing)
        raw_relative = f"raw/sample_{index:03d}.jpg"
        raw_bytes = base64.b64decode(str(payload["image_data"]).split(",", 1)[-1])
        (session_dir / raw_relative).write_bytes(raw_bytes)
        record = {
            "sample_index": index,
            "phase": payload["phase"],
            "point_index": payload["point_index"],
            "repeat_index": payload["repeat_index"],
            "target_x_norm": payload["target_x_norm"],
            "target_y_norm": payload["target_y_norm"],
            "collection_protocol": payload["collection_protocol"],
            "motion_block_id": payload["motion_block_id"],
            "posture_condition": payload["posture_condition"],
            "distance_condition": payload["distance_condition"],
            "capture_run_id": payload["capture_run_id"],
            "capture_source": payload["capture_source"],
            "calibration_label_authority": payload["calibration_label_authority"],
            "capture_contract": payload["capture_contract"],
            "raw_path": raw_relative,
            "face_detected": self.face_detected,
        }
        if self.face_detected:
            crop = f"crop/sample_{index:03d}.jpg"
            normalized = f"normalized_face/sample_{index:03d}.jpg"
            (session_dir / crop).write_bytes(b"crop")
            (session_dir / normalized).write_bytes(b"normalized")
            record["crop_path"] = crop
            record["normalized_face_path"] = normalized
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"ok": True, "sample_index": index, "face_detected": self.face_detected}, 200

    def purge(self, root: Path, session_id: str) -> dict[str, Any]:
        if self.purge_fails:
            raise PermissionError("injected purge denial")
        session_dir = root / "data" / "sessions" / session_id
        removed: list[str] = []
        for name in ("raw", "crop", "normalized_face"):
            target = session_dir / name
            if target.is_dir():
                shutil.rmtree(target)
                removed.append(name)
        return {"ok": True, "removed_directories": removed}

    def monotonic(self) -> float:
        self.clock += 5.0
        return self.clock

    @staticmethod
    def model_path(root: Path, model_name: str) -> Path:
        parent = root / "examples" / "models"
        parent.mkdir(parents=True, exist_ok=True)
        return parent / f"{model_name}.json"

    def train(self, root: Path, payload: dict[str, Any]) -> tuple[dict, int]:
        self.train_calls += 1
        self.last_training_payload = json.loads(json.dumps(payload))
        binding = payload["measurement_training_binding"]
        model_name = str(payload["output_model_name"])
        if self.train_mode == "partial_invalid_crash":
            self.model_path(root, model_name).write_text(
                "{partial-invalid-json", encoding="utf-8"
            )
            raise RuntimeError("injected trainer crash after partial model write")
        artifact = {
            "name": model_name,
            "data_session_id": payload["data_session_id"],
            "train_samples": 65,
            "training_device": "cpu",
            "uncertainty_v2": {"status": "scored_no_threshold"},
            "measurement_training_input_binding": {
                "binding_sha256": binding["binding_sha256"],
                "rows_sha256": binding["rows_sha256"],
                "row_count": 65,
                "capture_run_id": binding["capture_run_id"],
                "base_inference_bundle": binding["base_inference_bundle"],
            },
        }
        self.model_path(root, model_name).write_text(
            json.dumps(artifact), encoding="utf-8"
        )
        response = {
            "ok": True,
            "model_name": model_name,
            "train_samples": 65,
            "training_device": "cpu",
            "uncertainty_v2": {"status": "scored_no_threshold"},
            "consumed_training_rows_sha256": binding["rows_sha256"],
            "measurement_training_binding_sha256": binding["binding_sha256"],
            "base_inference_bundle": binding["base_inference_bundle"],
        }
        if self.train_mode == "wrong_base_echo":
            response["base_inference_bundle"] = {
                **binding["base_inference_bundle"],
                "checkpoint_sha256": "f" * 64,
            }
        return response, 200

    def runner(self) -> MeasurementRunner:
        return MeasurementRunner(
            self.root,
            code_root=REPOSITORY_ROOT,
            create_session_fn=self.create_session,
            save_sample_fn=self.save,
            train_fn=self.train,
            predict_fn=self.predict,
            purge_fn=self.purge,
            model_path_fn=self.model_path,
            base_bundle_fn=self.base_bundle,
            verify_base_bundle_fn=self.verify_bundle,
            reset_inference_cache_fn=self.reset_cache,
            decode_frame_fn=self.decode,
            monotonic_ms_fn=self.monotonic,
        )


class MeasurementRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.harness = _Harness(self.root)
        self.runner = self.harness.runner()
        self.contract = _contract()
        self.viewport = {"width": 1600.0, "height": 900.0, "device_pixel_ratio": 1.0}
        self.request_id = "WGMCREQ-" + "1" * 32
        self.token = "wgmr_client_" + "2" * 64

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _create(self, *, contract: dict[str, Any] | None = None) -> dict[str, Any]:
        selected = contract or self.contract
        return self.runner.create_run(
            create_request_id=self.request_id,
            run_token=self.token,
            capture_contract=selected,
            viewport_width=self.viewport["width"],
            viewport_height=self.viewport["height"],
            device_pixel_ratio=self.viewport["device_pixel_ratio"],
            readiness_preflight=_preflight(selected, self.viewport, self.harness.bundle),
        )

    def _submit(self, created: dict[str, Any], challenge: dict[str, Any], *, contract: dict[str, Any] | None = None, image_data: str | None = "valid") -> dict[str, Any]:
        if image_data == "valid":
            image_data = "data:image/jpeg;base64," + base64.b64encode(
                self.harness.raw
            ).decode("ascii")
        return self.runner.submit_frame(
            created["capture_run_id"],
            self.token,
            challenge["challenge_token"],
            image_data=image_data,
            observed_capture_contract=contract or self.contract,
            observed_viewport_width=self.viewport["width"],
            observed_viewport_height=self.viewport["height"],
            observed_device_pixel_ratio=self.viewport["device_pixel_ratio"],
        )

    def _collect_calibration(
        self,
        created: dict[str, Any],
        *,
        no_face_first: bool = False,
    ) -> None:
        for ordinal in range(65):
            self.harness.face_detected = not (no_face_first and ordinal == 0)
            challenge = self.runner.issue_next_challenge(
                created["capture_run_id"], self.token
            )
            result = self._submit(created, challenge)
            self.assertTrue(result["consumed"])

    def test_create_is_restart_idempotent_exact_and_separate_root(self) -> None:
        created = self._create()
        self.assertTrue(created["created_new"])
        self.assertNotIn("run_token", created)
        self.assertEqual(self.harness.base_calls, 1)
        restarted = self.harness.runner()
        replay = restarted.create_run(
            create_request_id=self.request_id,
            run_token=self.token,
            capture_contract=self.contract,
            viewport_width=1600,
            viewport_height=900,
            device_pixel_ratio=1,
            readiness_preflight=None,
        )
        self.assertEqual(replay["capture_run_id"], created["capture_run_id"])
        self.assertTrue(replay["idempotent"])
        self.assertNotIn("run_token", replay)
        self.assertEqual(self.harness.reset_calls, 2)
        self.assertEqual(self.harness.base_calls, 2)
        lookup = restarted.lookup_create_request(
            create_request_id=self.request_id, run_token=self.token
        )
        self.assertTrue(lookup["exists"])
        self.assertNotIn("run_token", lookup)
        changed = _contract(fps=15.0)
        with self.assertRaises(MeasurementRunnerError):
            restarted.create_run(
                create_request_id=self.request_id,
                run_token=self.token,
                capture_contract=changed,
                viewport_width=1600,
                viewport_height=900,
                device_pixel_ratio=1,
                readiness_preflight=None,
            )
        self.assertTrue((self.root / "data" / "measurement_ceiling_runs").is_dir())
        self.assertNotEqual(self.root.resolve(), REPOSITORY_ROOT)

    def test_concurrent_same_authority_creates_one_external_session(self) -> None:
        barrier = threading.Barrier(2)
        results: list[dict[str, Any]] = []
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                runner = self.harness.runner()
                barrier.wait()
                results.append(
                    runner.create_run(
                        create_request_id=self.request_id,
                        run_token=self.token,
                        capture_contract=self.contract,
                        viewport_width=1600,
                        viewport_height=900,
                        device_pixel_ratio=1,
                        readiness_preflight=_preflight(
                            self.contract, self.viewport, self.harness.bundle
                        ),
                    )
                )
            except BaseException as exc:  # noqa: BLE001 - test captures threads
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        self.assertFalse(errors)
        self.assertEqual(len(results), 2)
        self.assertEqual(
            {item["capture_run_id"] for item in results},
            {results[0]["capture_run_id"]},
        )
        self.assertEqual(self.harness.create_session_calls, 1)

    def test_predict_success_then_sample_preprocessor_no_face_consumes_once(self) -> None:
        self.harness.face_detected = False
        created = self._create()
        challenge = self.runner.issue_next_challenge(created["capture_run_id"], self.token)
        result = self._submit(created, challenge)
        self.assertTrue(result["consumed"])
        self.assertEqual(result["classification"], "attributable_sensor_failure")
        self.assertEqual(self.harness.predict_calls, 1)
        status = self.runner.get_status(created["capture_run_id"], self.token)
        self.assertEqual(status["progress"]["calibration_count"], 1)
        self.assertEqual(status["runner"]["calibration_usable_manifest_count"], 0)
        session_id = status["runner"]["runtime_binding"]["calibration_session_id"]
        manifest = self.root / "data" / "sessions" / session_id / "manifest.jsonl"
        self.assertEqual(manifest.read_bytes(), b"")
        retry = self._submit(created, challenge, image_data=None)
        self.assertEqual(retry["classification"], "exact_retry")
        self.assertEqual(self.harness.predict_calls, 1)

    def test_save_crash_resumes_from_server_spool_without_reinference(self) -> None:
        self.harness.save_failures_remaining = 1
        created = self._create()
        challenge = self.runner.issue_next_challenge(created["capture_run_id"], self.token)
        first = self._submit(created, challenge)
        self.assertFalse(first["consumed"])
        self.assertTrue(first["server_spool_retry_available"])
        self.assertEqual(self.harness.predict_calls, 1)
        restarted = self.harness.runner()
        self.runner = restarted
        second = self._submit(created, challenge, image_data=None)
        self.assertTrue(second["consumed"])
        self.assertEqual(self.harness.predict_calls, 1)
        spool_dir = self.root / "data" / "measurement_ceiling_runs" / created["capture_run_id"] / "spool"
        self.assertFalse(spool_dir.is_dir() and any(spool_dir.iterdir()))

    def test_spool_ciphertext_tamper_is_terminal_and_abort_required(self) -> None:
        created = self._create()
        run_id = created["capture_run_id"]
        challenge = self.runner.issue_next_challenge(run_id, self.token)
        self.harness.save_failures_remaining = 1
        first = self._submit(created, challenge)
        self.assertTrue(first["server_spool_retry_available"])
        spool_dir = self.root / "data" / "measurement_ceiling_runs" / run_id / "spool"
        spool_path = next(spool_dir.glob("*.aesgcm"))
        ciphertext = bytearray(spool_path.read_bytes())
        ciphertext[-1] ^= 1
        spool_path.write_bytes(ciphertext)
        failed = self.runner.inspect_challenge(
            run_id, self.token, challenge["challenge_token"]
        )
        self.assertTrue(failed["abort_required"])
        self.assertFalse(failed["retryable"])
        self.assertEqual(failed["phase"], "failed_integrity")
        self.assertFalse(spool_path.exists())
        cleaned = self.runner.abort_and_cleanup(
            run_id, self.token, reason="authenticated failed-run cleanup"
        )
        self.assertTrue(cleaned["cleanup_verified"])

    def test_crash_after_predict_before_observation_never_replays_inference(self) -> None:
        created = self._create()
        run_id = created["capture_run_id"]
        challenge = self.runner.issue_next_challenge(run_id, self.token)
        original = self.runner.store.record_attempt_observation

        def crash_before_seal(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("injected crash before observation seal")

        self.runner.store.record_attempt_observation = crash_before_seal  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, "observation seal"):
            self._submit(created, challenge)
        self.runner.store.record_attempt_observation = original  # type: ignore[method-assign]
        self.assertEqual(self.harness.predict_calls, 1)
        pending = self.runner.get_status(run_id, self.token)
        self.assertTrue(pending["runner"]["inference_in_progress"])

        self.runner = self.harness.runner()
        failed = self._submit(created, challenge, image_data=None)
        self.assertTrue(failed["abort_required"])
        self.assertFalse(failed["retryable"])
        self.assertEqual(failed["phase"], "failed_integrity")
        self.assertEqual(self.harness.predict_calls, 1)
        status = self.runner.get_status(run_id, self.token)
        self.assertEqual(status["phase"], "failed_integrity")
        self.assertFalse(status["runner"]["inference_in_progress"])

    def test_no_face_reclassification_survives_post_truncate_crash(self) -> None:
        created = self._create()
        challenge = self.runner.issue_next_challenge(
            created["capture_run_id"], self.token
        )
        self.harness.face_detected = False
        original = self.runner._discard_unusable_manifest_row
        crashed = False

        def truncate_then_crash(*args: Any, **kwargs: Any) -> None:
            nonlocal crashed
            original(*args, **kwargs)
            if not crashed:
                crashed = True
                raise RuntimeError("injected crash after manifest truncation")

        self.runner._discard_unusable_manifest_row = truncate_then_crash  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, "manifest truncation"):
            self._submit(created, challenge)
        self.runner._discard_unusable_manifest_row = original  # type: ignore[method-assign]
        before_predict = self.harness.predict_calls
        before_save = self.harness.save_calls
        self.harness.face_detected = True
        recovered = self._submit(created, challenge, image_data=None)
        self.assertEqual(recovered["classification"], "attributable_sensor_failure")
        self.assertTrue(recovered["consumed"])
        self.assertEqual(self.harness.predict_calls, before_predict)
        self.assertEqual(self.harness.save_calls, before_save)
        status = self.runner.get_status(created["capture_run_id"], self.token)
        self.assertEqual(status["runner"]["calibration_usable_manifest_count"], 0)

    def test_model_echo_and_no_face_response_contract_fail_closed(self) -> None:
        created = self._create()
        challenge = self.runner.issue_next_challenge(
            created["capture_run_id"], self.token
        )
        for mode in ("wrong_model", "missing_model", "malformed_no_face"):
            self.harness.predict_mode = mode
            failed = self._submit(created, challenge)
            self.assertFalse(failed["consumed"])
            self.assertTrue(failed["new_frame_retry_allowed"])
            status = self.runner.get_status(created["capture_run_id"], self.token)
            self.assertEqual(status["progress"]["next_sequence_index"], 0)
            self.assertEqual(status["runner"]["capture_contract_proof_count"], 0)
        self.harness.predict_mode = "no_face"
        accepted = self._submit(created, challenge)
        self.assertTrue(accepted["consumed"])
        self.assertEqual(accepted["classification"], "attributable_sensor_failure")

    def test_contract_drift_warning_and_hard_mismatches(self) -> None:
        created = self._create()
        challenge = self.runner.issue_next_challenge(created["capture_run_id"], self.token)
        observed = _contract(
            source_width=640,
            source_height=360,
            fps=15.0,
            transport_width=320,
        )
        self.harness.decoded_width = 320
        self.harness.decoded_height = 180
        result = self._submit(created, challenge, contract=observed)
        self.assertTrue(result["consumed"])
        state_path = self.root / "data" / "measurement_ceiling_runs" / created["capture_run_id"] / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        warnings = state["runner"]["capture_contract_proofs"][0][
            "capture_contract_evidence"
        ]["contract_comparison"]["warnings"]
        self.assertIn("source_resolution_changed", warnings)
        self.assertIn("source_frame_rate_changed", warnings)

        second = self.runner.issue_next_challenge(created["capture_run_id"], self.token)
        incompatible = _contract(
            source_width=640,
            source_height=480,
            transport_width=320,
            transport_height=240,
        )
        failed = self._submit(created, second, contract=incompatible)
        self.assertFalse(failed["consumed"])
        self.assertEqual(self.harness.predict_calls, 1)

        mirrored = _contract(mirror=True)
        self.harness.decoded_width = 640
        self.harness.decoded_height = 360
        failed = self._submit(created, second, contract=mirrored)
        self.assertFalse(failed["consumed"])
        self.assertEqual(self.harness.predict_calls, 1)

        self.harness.decoded_width = 320
        self.harness.decoded_height = 180
        failed = self._submit(created, second, contract=self.contract)
        self.assertFalse(failed["consumed"])
        self.assertEqual(self.harness.predict_calls, 1)
        self.assertEqual(self.harness.base_calls, 2)

    def test_hard_inference_does_not_consume_and_allows_new_frame(self) -> None:
        self.harness.predict_mode = "hard"
        created = self._create()
        challenge = self.runner.issue_next_challenge(
            created["capture_run_id"], self.token
        )
        first = self._submit(created, challenge)
        self.assertFalse(first["consumed"])
        self.assertTrue(first["new_frame_retry_allowed"])
        status = self.runner.get_status(created["capture_run_id"], self.token)
        self.assertEqual(status["progress"]["next_sequence_index"], 0)
        self.harness.predict_mode = "success"
        second = self._submit(created, challenge)
        self.assertTrue(second["consumed"])
        self.assertEqual(self.harness.predict_calls, 2)

    def test_abort_purge_failure_is_retryable_and_orphan_spool_is_removed(self) -> None:
        created = self._create()
        run_id = created["capture_run_id"]
        challenge = self.runner.issue_next_challenge(run_id, self.token)
        run_dir = self.root / "data" / "measurement_ceiling_runs" / run_id
        orphan = run_dir / "spool" / f"{challenge['challenge_id']}.aesgcm"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_bytes(b"ciphertext-before-state-crash")
        self.harness.purge_fails = True
        first = self.runner.abort_and_cleanup(
            run_id, self.token, reason="test cleanup"
        )
        self.assertFalse(first["cleanup_verified"])
        self.assertFalse(first["terminal"])
        self.assertFalse(orphan.exists())
        self.assertNotEqual(
            self.runner.get_status(run_id, self.token)["phase"], "aborted"
        )
        self.harness.purge_fails = False
        second = self.runner.abort_and_cleanup(
            run_id, self.token, reason="test cleanup"
        )
        self.assertTrue(second["cleanup_verified"])
        self.assertTrue(second["terminal"])
        self.assertEqual(second["phase"], "aborted")

    def test_exact_65_training_binds_consumed_bytes_and_purges(self) -> None:
        created = self._create()
        self._collect_calibration(created)
        result = self.runner.finalize_calibration(
            created["capture_run_id"], self.token
        )
        self.assertEqual(result["phase"], "model_bound")
        self.assertEqual(self.harness.train_calls, 1)
        self.assertEqual(self.harness.base_calls, 4)
        payload = self.harness.last_training_payload
        self.assertIsNotNone(payload)
        self.assertIs(payload["allow_cuda"], False)
        binding = payload["measurement_training_binding"]
        self.assertEqual(len(binding["rows"]), 65)
        model_path = self.harness.model_path(
            self.root, result["model_binding"]["model_id"]
        )
        artifact = json.loads(model_path.read_text(encoding="utf-8"))
        self.assertEqual(
            artifact["measurement_training_input_binding"]["rows_sha256"],
            binding["rows_sha256"],
        )
        provenance = artifact["measurement_ceiling_provenance"]
        self.assertEqual(provenance["evaluation_rows_used"], 0)
        self.assertEqual(
            provenance["calibration_evaluation_target_intersection_count"], 0
        )
        session_id = result["runner"]["runtime_binding"]["calibration_session_id"]
        session_dir = self.root / "data" / "sessions" / session_id
        self.assertTrue(
            all(not (session_dir / name).exists() for name in ("raw", "crop", "normalized_face"))
        )

    def test_one_no_face_is_terminal_negative_without_training(self) -> None:
        created = self._create()
        self._collect_calibration(created, no_face_first=True)
        result = self.runner.finalize_calibration(
            created["capture_run_id"], self.token
        )
        self.assertEqual(result["classification"], "negative_calibration_result")
        self.assertEqual(result["calibration_attempt_count"], 65)
        self.assertEqual(result["usable_calibration_count"], 64)
        self.assertEqual(self.harness.train_calls, 0)
        self.assertEqual(result["phase"], "failed_integrity")
        self.assertTrue(result["cleanup_verified"])

    def test_training_image_tamper_fails_before_fit_and_cleans(self) -> None:
        created = self._create()
        self._collect_calibration(created)
        status = self.runner.get_status(created["capture_run_id"], self.token)
        session_id = status["runner"]["runtime_binding"]["calibration_session_id"]
        target = self.root / "data" / "sessions" / session_id / "normalized_face" / "sample_000.jpg"
        target.write_bytes(b"tampered-normalized-face")
        result = self.runner.finalize_calibration(
            created["capture_run_id"], self.token
        )
        self.assertEqual(result["phase"], "failed_integrity")
        self.assertEqual(result["classification"], "calibration_finalization_failed")
        self.assertEqual(self.harness.train_calls, 0)
        self.assertTrue(result["cleanup_verified"])

    def test_partial_invalid_training_artifact_is_deleted_from_reserved_path(self) -> None:
        created = self._create()
        self._collect_calibration(created)
        self.harness.train_mode = "partial_invalid_crash"
        result = self.runner.finalize_calibration(
            created["capture_run_id"], self.token
        )
        self.assertEqual(result["phase"], "failed_integrity")
        self.assertTrue(result["cleanup_verified"])
        self.assertEqual(result["cleanup_errors"], [])
        model_path = self.harness.model_path(
            self.root,
            self.runner._personal_model_name(created["capture_run_id"]),
        )
        self.assertFalse(model_path.exists())
        intent = result["runner"]["training_artifact_intent"]
        self.assertEqual(intent["status"], "cleanup_verified")

    def test_training_base_echo_mismatch_fails_and_cleans(self) -> None:
        created = self._create()
        self._collect_calibration(created)
        self.harness.train_mode = "wrong_base_echo"
        result = self.runner.finalize_calibration(
            created["capture_run_id"], self.token
        )
        self.assertEqual(result["phase"], "failed_integrity")
        self.assertTrue(result["cleanup_verified"])
        self.assertFalse(
            self.harness.model_path(
                self.root,
                self.runner._personal_model_name(created["capture_run_id"]),
            ).exists()
        )

    def test_evaluation_rehash_fails_closed_before_predict(self) -> None:
        created = self._create()
        self._collect_calibration(created)
        bound = self.runner.finalize_calibration(
            created["capture_run_id"], self.token
        )
        model_path = self.harness.model_path(
            self.root, bound["model_binding"]["model_id"]
        )
        challenge = self.runner.issue_next_challenge(
            created["capture_run_id"], self.token
        )
        artifact = json.loads(model_path.read_text(encoding="utf-8"))
        artifact["tampered_but_valid_json"] = True
        model_path.write_text(json.dumps(artifact), encoding="utf-8")
        before = self.harness.predict_calls
        with self.assertRaises(MeasurementRunIntegrityError):
            self._submit(created, challenge)
        self.assertEqual(self.harness.predict_calls, before)
        failed = self.runner.get_status(created["capture_run_id"], self.token)
        self.assertEqual(failed["phase"], "failed_integrity")
        self.assertFalse(model_path.exists())
        self.assertTrue(failed["failure"]["cleanup"]["cleanup_verified"])

    def test_analysis_evidence_rehashes_model_and_rechecks_purge(self) -> None:
        session_id = "session_analysis_evidence"
        session_dir = self.root / "data" / "sessions" / session_id
        session_dir.mkdir(parents=True)
        model_path = self.harness.model_path(self.root, "wgmc_analysis")
        model = {
            "measurement_ceiling_provenance": {
                "capture_run_id": "WGMC-analysis",
                "training_role": "calibration_only",
            }
        }
        model_bytes = json.dumps(model, sort_keys=True).encode("utf-8")
        model_path.write_bytes(model_bytes)
        status = {
            "capture_run_id": "WGMC-analysis",
            "phase": "artifact_verified",
            "acquisition_artifact_verified": True,
            "capture_contract_binding_verified": True,
            "runner": {
                "runtime_binding": {"calibration_session_id": session_id},
                "trained_artifact": {
                    "artifact_relative_path": model_path.relative_to(
                        self.root
                    ).as_posix(),
                    "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
                },
                "calibration_image_purge": {
                    "status": "verified",
                    "calibration_session_id": session_id,
                },
            },
        }

        class _EvidenceStore:
            @staticmethod
            def adapter_operation_lock(_run_id: str):
                return nullcontext()

            @staticmethod
            def verify_sealed_artifact(_run_id: str, _token: str):
                return {"status": "passed"}

            @staticmethod
            def get_status(_run_id: str, _token: str):
                return status

            @staticmethod
            def read_sealed_artifact(_run_id: str, _token: str):
                return {"samples": []}

            @staticmethod
            def read_sealed_attempt_sidecar(_run_id: str, _token: str):
                return {"attempts": []}

        evidence_runner = MeasurementRunner(
            self.root,
            code_root=REPOSITORY_ROOT,
            store=_EvidenceStore(),  # type: ignore[arg-type]
            model_path_fn=self.harness.model_path,
        )
        evidence = evidence_runner.read_verified_analysis_evidence(
            "WGMC-analysis", self.token
        )
        self.assertTrue(evidence["calibration_image_absence_reverified"])
        (session_dir / "raw").mkdir()
        with self.assertRaisesRegex(
            MeasurementRunIntegrityError, "purge no longer holds"
        ):
            evidence_runner.read_verified_analysis_evidence(
                "WGMC-analysis", self.token
            )
        (session_dir / "raw").rmdir()
        model_path.write_bytes(model_bytes + b" ")
        with self.assertRaisesRegex(
            MeasurementRunIntegrityError, "model SHA-256 changed"
        ):
            evidence_runner.read_verified_analysis_evidence(
                "WGMC-analysis", self.token
            )

    def test_authenticated_analysis_entrypoint_is_canonical_runner_method(self) -> None:
        import core.gaze_core.measurement_run_analysis as analysis_module

        self.assertFalse(
            hasattr(analysis_module, "analyze_authenticated_measurement_run")
        )
        self.assertIn(
            "read_verified_analysis_evidence",
            MeasurementRunner.analyze_verified_run.__code__.co_names,
        )

    def test_training_binding_legacy_path_and_same_byte_tamper_guard(self) -> None:
        records = [
            {
                "normalized_face_path": f"normalized_face/{index:03d}.jpg",
                "raw_path": f"raw/{index:03d}.jpg",
            }
            for index in range(65)
        ]
        self.assertIsNone(
            _validated_measurement_training_binding(
                None, dataset_id="legacy", records=records
            )
        )
        rows = [
            {
                "sequence_index": index,
                "manifest_sample_index": index,
                "manifest_record_sha256": training_canonical_sha256(record),
                "frame_sha256": "a" * 64,
                "normalized_face_path": record["normalized_face_path"],
                "normalized_face_sha256": "b" * 64,
            }
            for index, record in enumerate(records)
        ]
        binding = {
            "schema_version": 1,
            "binding_type": MEASUREMENT_TRAINING_BINDING_TYPE,
            "data_session_id": "dedicated",
            "capture_run_id": "WGMC-test",
            "manifest_sha256": "c" * 64,
            "base_inference_bundle": dict(_bundle()),
            "rows": rows,
            "rows_sha256": training_canonical_sha256(rows),
        }
        binding["binding_sha256"] = training_canonical_sha256(binding)
        self.assertEqual(
            _validated_measurement_training_binding(
                binding, dataset_id="dedicated", records=records
            ),
            binding,
        )

        import cv2
        import numpy as np

        session = self.root / "byte_guard"
        (session / "raw").mkdir(parents=True)
        (session / "normalized_face").mkdir()
        raw = b"raw-frame"
        ok, encoded = cv2.imencode(
            ".jpg", np.zeros((8, 8, 3), dtype=np.uint8)
        )
        self.assertTrue(ok)
        normalized = encoded.tobytes()
        (session / "raw/000.jpg").write_bytes(raw)
        (session / "normalized_face/000.jpg").write_bytes(normalized)
        record = records[0]
        expected = dict(rows[0])
        expected["frame_sha256"] = hashlib.sha256(raw).hexdigest()
        expected["normalized_face_sha256"] = hashlib.sha256(normalized).hexdigest()
        image, consumed = _read_measurement_training_image(
            session, record, expected, manifest_index=0
        )
        self.assertEqual(image.shape[:2], (8, 8))
        self.assertEqual(consumed["normalized_face_sha256"], expected["normalized_face_sha256"])
        (session / "normalized_face/000.jpg").write_bytes(normalized + b"mutation")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            _read_measurement_training_image(
                session, record, expected, manifest_index=0
            )


if __name__ == "__main__":
    unittest.main()

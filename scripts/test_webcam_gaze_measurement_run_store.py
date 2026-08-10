"""Focused contracts for the persistent 193-sample self-development runner."""

from __future__ import annotations

import json
import math
import multiprocessing
import tempfile
import threading
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from core.gaze_core.measurement_run_store import (
    CALIBRATION_LEDGER_FILENAME,
    CAPTURE_ARTIFACT_FILENAME,
    CREATE_REGISTRY_FILENAME,
    PHASE_ABORTED,
    PHASE_ARTIFACT_VERIFIED,
    PHASE_CALIBRATION_SEALED,
    PHASE_CAPTURE_SEALED,
    PHASE_FAILED_INTEGRITY,
    PHASE_MODEL_BOUND,
    STATE_FILENAME,
    STORE_RELATIVE_PATH,
    MeasurementRunAuthenticationError,
    MeasurementRunChallengeError,
    MeasurementRunIntegrityError,
    MeasurementRunStateError,
    MeasurementRunStore,
    MeasurementRunValidationError,
)
from core.gaze_core.measurement_schedule import (
    EXPECTED_CALIBRATION_SAMPLE_COUNT,
    EXPECTED_EVALUATION_SAMPLE_COUNT,
    EXPECTED_SAMPLE_COUNT,
    canonical_json_bytes,
    canonical_sha256,
    verify_capture_artifact,
    verify_run_manifest,
)


CALIBRATION_MODEL_ID = "frozen-base-encoder"
CALIBRATION_MODEL_SHA256 = "a" * 64
EVALUATION_MODEL_ID = "self-development-personal-calibrator"
EVALUATION_MODEL_SHA256 = "b" * 64


def _observation(
    row: dict,
    *,
    model_id: str | None = None,
    model_sha256: str | None = None,
) -> dict:
    sequence_index = int(row["sequence_index"])
    viewport_width = 1600
    viewport_height = 900
    target_x_px = float(
        math.floor(row["target_x_viewport_fraction"] * viewport_width + 0.5)
    )
    target_y_px = float(
        math.floor(row["target_y_viewport_fraction"] * viewport_height + 0.5)
    )
    calibration = row["block_role"] == "calibration"
    captured_ms = float(sequence_index * 100)
    latency_ms = 12.5
    return {
        "capture_source": "direct-webcam-self-development",
        "target_x_px": target_x_px,
        "target_y_px": target_y_px,
        "frame_capture_monotonic_ms": captured_ms,
        "inference_completed_monotonic_ms": captured_ms + latency_ms,
        "inference_latency_ms": latency_ms,
        "model_id": model_id
        or (CALIBRATION_MODEL_ID if calibration else EVALUATION_MODEL_ID),
        "model_sha256": model_sha256
        or (
            CALIBRATION_MODEL_SHA256
            if calibration
            else EVALUATION_MODEL_SHA256
        ),
        "prediction_success": True,
        "raw_gaze_pitch_yaw": [0.01, -0.02],
        "predicted_x_px": target_x_px + 8.0,
        "predicted_y_px": target_y_px - 5.0,
        "head_pose_pitch_yaw": [0.02, -0.01],
        "normalized_face_bbox": [0.2, 0.1, 0.8, 0.9],
        "camera_width": 1280,
        "camera_height": 720,
        "camera_frame_rate": 30.0,
        "viewport_width": viewport_width,
        "viewport_height": viewport_height,
        "device_pixel_ratio": 1.0,
    }


def _run_directory(root: Path, capture_run_id: str) -> Path:
    return root / STORE_RELATIVE_PATH / capture_run_id


def _rehash_state(state: dict) -> None:
    core = deepcopy(state)
    core.pop("state_sha256", None)
    state["state_sha256"] = canonical_sha256(core)


def _multiprocess_consume_worker(
    root: str,
    capture_run_id: str,
    run_token: str,
    challenge_token: str,
    observation: dict,
    queue: object,
) -> None:
    try:
        receipt = MeasurementRunStore(Path(root)).consume_challenge(
            capture_run_id,
            run_token,
            challenge_token,
            observation,
        )
        queue.put(("ok", receipt))
    except Exception as exc:  # pragma: no cover - child-process evidence
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


class WebcamGazeMeasurementRunStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = MeasurementRunStore(self.root)
        self.create_request_id = "WGMCREQ-" + "c" * 32
        self.client_run_token = "wgmr_client_" + "5" * 64
        created = self.store.create_run(
            create_request_id=self.create_request_id,
            run_token=self.client_run_token,
        )
        self.capture_run_id = created["capture_run_id"]
        self.run_token = self.client_run_token
        self.run_dir = _run_directory(self.root, self.capture_run_id)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _issue_and_consume(self) -> tuple[dict, dict, dict]:
        challenge = self.store.issue_next_challenge(
            self.capture_run_id, self.run_token
        )
        observation = _observation(challenge["schedule_row"])
        receipt = self.store.consume_challenge(
            self.capture_run_id,
            self.run_token,
            challenge["challenge_token"],
            observation,
        )
        return challenge, observation, receipt

    def _collect_calibration(self) -> None:
        for _ in range(EXPECTED_CALIBRATION_SAMPLE_COUNT):
            self._issue_and_consume()

    def test_create_is_dedicated_193_rows_and_hashes_plaintext_tokens(self) -> None:
        manifest = json.loads(
            (self.run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        summary = verify_run_manifest(manifest)
        self.assertEqual(summary["total_sample_count"], EXPECTED_SAMPLE_COUNT)
        self.assertEqual(summary["calibration_sample_count"], 65)
        self.assertEqual(summary["evaluation_sample_count"], 128)
        persisted = "\n".join(
            path.read_text(encoding="utf-8", errors="strict")
            for path in self.run_dir.iterdir()
            if path.is_file() and path.name != ".lock"
        )
        self.assertNotIn(self.run_token, persisted)
        self.assertNotIn("participant", str(self.run_dir).lower())
        state = json.loads((self.run_dir / STATE_FILENAME).read_text(encoding="utf-8"))
        self.assertRegex(state["run_token_sha256"], r"^[0-9a-f]{64}$")
        self.assertFalse(state["measurement_claim_authorized"])
        self.assertFalse(state["physical_capture_claim_authorized"])
        self.assertFalse(state["acquisition_artifact_verified"])
        self.assertFalse(state["capture_contract_binding_verified"])
        self.assertEqual(state["progress"]["next_sequence_index"], 0)

    def test_client_create_authority_is_restart_idempotent_and_strict(self) -> None:
        replay = self.store.create_run(
            create_request_id=self.create_request_id,
            run_token=self.run_token,
        )
        self.assertEqual(replay["capture_run_id"], self.capture_run_id)
        self.assertTrue(replay["idempotent"])
        self.assertFalse(replay["created_new"])

        registry_path = self.root / STORE_RELATIVE_PATH / CREATE_REGISTRY_FILENAME
        registry_path.unlink()
        restarted = MeasurementRunStore(self.root)
        recovered = restarted.lookup_create_request(
            create_request_id=self.create_request_id,
            run_token=self.run_token,
        )
        self.assertTrue(recovered["exists"])
        self.assertEqual(recovered["capture_run_id"], self.capture_run_id)
        self.assertTrue(registry_path.is_file())
        with self.assertRaises(MeasurementRunAuthenticationError):
            restarted.create_run(
                create_request_id=self.create_request_id,
                run_token="wgmr_client_" + "6" * 64,
            )
        for bad_request in ("a" * 32, "WGMCREQ-" + "A" * 32, True):
            with self.assertRaises(MeasurementRunAuthenticationError):
                restarted.lookup_create_request(
                    create_request_id=bad_request,
                    run_token=self.run_token,
                )
        for bad_token in ("a" * 64, "wgmr_client_" + "A" * 64, False):
            with self.assertRaises(MeasurementRunAuthenticationError):
                restarted.lookup_create_request(
                    create_request_id=self.create_request_id,
                    run_token=bad_token,
                )

        durable = registry_path.read_text(encoding="utf-8") + (
            self.run_dir / STATE_FILENAME
        ).read_text(encoding="utf-8")
        self.assertNotIn(self.create_request_id, durable)
        self.assertNotIn(self.run_token, durable)

    def test_challenge_rotation_schedule_authority_and_no_raw_media(self) -> None:
        first = self.store.issue_next_challenge(
            self.capture_run_id, self.run_token
        )
        forged = _observation(first["schedule_row"])
        forged["target_id"] = "forged"
        with self.assertRaisesRegex(
            MeasurementRunValidationError, "server schedule field target_id"
        ):
            self.store.consume_challenge(
                self.capture_run_id,
                self.run_token,
                first["challenge_token"],
                forged,
            )
        raw = _observation(first["schedule_row"])
        raw["image_data"] = "data:image/jpeg;base64,AAAA"
        with self.assertRaisesRegex(MeasurementRunValidationError, "raw-media"):
            self.store.consume_challenge(
                self.capture_run_id,
                self.run_token,
                first["challenge_token"],
                raw,
            )

        replacement = self.store.rotate_unconsumed_challenge(
            self.capture_run_id, self.run_token
        )
        self.assertEqual(replacement["ordinal"], first["ordinal"])
        self.assertNotEqual(replacement["challenge_token"], first["challenge_token"])
        with self.assertRaises(MeasurementRunChallengeError):
            self.store.consume_challenge(
                self.capture_run_id,
                self.run_token,
                first["challenge_token"],
                _observation(first["schedule_row"]),
            )
        receipt = self.store.consume_challenge(
            self.capture_run_id,
            self.run_token,
            replacement["challenge_token"],
            _observation(replacement["schedule_row"]),
        )
        self.assertEqual(receipt["ordinal"], 0)
        self.assertFalse(receipt["physical_capture_claim_authorized"])
        ledger_text = (self.run_dir / CALIBRATION_LEDGER_FILENAME).read_text(
            encoding="utf-8"
        )
        self.assertNotIn(replacement["challenge_token"], ledger_text)
        self.assertNotIn("image_data", ledger_text)

    def test_restart_and_exact_retry_are_idempotent(self) -> None:
        challenge, observation, receipt = self._issue_and_consume()
        restarted = MeasurementRunStore(self.root)
        retried = restarted.consume_challenge(
            self.capture_run_id,
            self.run_token,
            challenge["challenge_token"],
            observation,
        )
        self.assertTrue(retried["idempotent"])
        self.assertEqual(retried["record_sha256"], receipt["record_sha256"])
        self.assertEqual(
            restarted.get_status(self.capture_run_id, self.run_token)["progress"][
                "calibration_count"
            ],
            1,
        )
        lines = (self.run_dir / CALIBRATION_LEDGER_FILENAME).read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertEqual(len(lines), 1)

    def test_conflicting_consumed_challenge_replay_is_terminal(self) -> None:
        challenge, observation, _ = self._issue_and_consume()
        changed = deepcopy(observation)
        changed["predicted_x_px"] += 1.0
        with self.assertRaisesRegex(MeasurementRunIntegrityError, "replay differs"):
            self.store.consume_challenge(
                self.capture_run_id,
                self.run_token,
                challenge["challenge_token"],
                changed,
            )
        state = json.loads((self.run_dir / STATE_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(state["phase"], PHASE_FAILED_INTEGRITY)
        self.assertEqual(
            state["failure"]["code"], "conflicting_consumed_challenge_replay"
        )

    def test_auth_and_unknown_challenge_fail_without_mutating_state(self) -> None:
        challenge = self.store.issue_next_challenge(
            self.capture_run_id, self.run_token
        )
        before = (self.run_dir / STATE_FILENAME).read_bytes()
        with self.assertRaises(MeasurementRunAuthenticationError):
            self.store.get_status(self.capture_run_id, "wrong-token")
        self.assertEqual((self.run_dir / STATE_FILENAME).read_bytes(), before)
        with self.assertRaises(MeasurementRunChallengeError):
            self.store.consume_challenge(
                self.capture_run_id,
                self.run_token,
                "wgmc_ch_unknown",
                _observation(challenge["schedule_row"]),
            )
        self.assertEqual((self.run_dir / STATE_FILENAME).read_bytes(), before)

    def test_restart_reconciles_one_committed_ledger_ahead_of_state(self) -> None:
        challenge = self.store.issue_next_challenge(
            self.capture_run_id, self.run_token
        )
        original_write = self.store._write_state

        def crash_after_ledger(*_args, **_kwargs):
            raise OSError("simulated process loss after ledger replace")

        self.store._write_state = crash_after_ledger  # type: ignore[method-assign]
        try:
            with self.assertRaisesRegex(OSError, "simulated process loss"):
                self.store.consume_challenge(
                    self.capture_run_id,
                    self.run_token,
                    challenge["challenge_token"],
                    _observation(challenge["schedule_row"]),
                )
        finally:
            self.store._write_state = original_write  # type: ignore[method-assign]

        restarted = MeasurementRunStore(self.root)
        status = restarted.get_status(self.capture_run_id, self.run_token)
        self.assertEqual(status["progress"]["calibration_count"], 1)
        self.assertEqual(status["progress"]["next_sequence_index"], 1)
        self.assertFalse(status["challenge_outstanding"])
        replay = restarted.consume_challenge(
            self.capture_run_id,
            self.run_token,
            challenge["challenge_token"],
            _observation(challenge["schedule_row"]),
        )
        self.assertTrue(replay["idempotent"])

    def test_semantically_rehashed_ledger_tamper_fails_closed(self) -> None:
        self._issue_and_consume()
        ledger_path = self.run_dir / CALIBRATION_LEDGER_FILENAME
        record = json.loads(ledger_path.read_text(encoding="utf-8").strip())
        record["sample"]["block_id"] = "forged-block"
        record["sample_sha256"] = canonical_sha256(record["sample"])
        core = deepcopy(record)
        core.pop("record_sha256", None)
        record["record_sha256"] = canonical_sha256(core)
        ledger_path.write_bytes(canonical_json_bytes(record) + b"\n")
        with self.assertRaises(MeasurementRunIntegrityError):
            self.store.get_status(self.capture_run_id, self.run_token)
        state = json.loads((self.run_dir / STATE_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(state["phase"], PHASE_FAILED_INTEGRITY)
        self.assertEqual(state["failure"]["code"], "persisted_run_integrity_failed")

    def test_model_gate_and_full_193_sample_lifecycle(self) -> None:
        self._collect_calibration()
        status = self.store.get_status(self.capture_run_id, self.run_token)
        self.assertEqual(status["phase"], PHASE_CALIBRATION_SEALED)
        self.assertEqual(
            status["progress"]["calibration_count"],
            EXPECTED_CALIBRATION_SAMPLE_COUNT,
        )
        sealed_sha = status["ledgers"]["calibration"]["sealed_sha256"]
        with self.assertRaisesRegex(
            MeasurementRunIntegrityError, "does not reference"
        ):
            self.store.bind_model(
                self.capture_run_id,
                self.run_token,
                model_id=EVALUATION_MODEL_ID,
                model_sha256=EVALUATION_MODEL_SHA256,
                calibration_ledger_sha256="c" * 64,
            )
        bound = self.store.bind_model(
            self.capture_run_id,
            self.run_token,
            model_id=EVALUATION_MODEL_ID,
            model_sha256=EVALUATION_MODEL_SHA256,
            calibration_ledger_sha256=sealed_sha,
        )
        self.assertEqual(bound["phase"], PHASE_MODEL_BOUND)

        first_evaluation = self.store.issue_next_challenge(
            self.capture_run_id, self.run_token
        )
        with self.assertRaisesRegex(
            MeasurementRunValidationError, "differs from bound model"
        ):
            self.store.consume_challenge(
                self.capture_run_id,
                self.run_token,
                first_evaluation["challenge_token"],
                _observation(
                    first_evaluation["schedule_row"],
                    model_id="wrong-model",
                    model_sha256="d" * 64,
                ),
            )
        self.store.consume_challenge(
            self.capture_run_id,
            self.run_token,
            first_evaluation["challenge_token"],
            _observation(first_evaluation["schedule_row"]),
        )
        challenge = first_evaluation
        observation = _observation(first_evaluation["schedule_row"])
        for _ in range(EXPECTED_EVALUATION_SAMPLE_COUNT - 1):
            challenge, observation, _ = self._issue_and_consume()

        sealed = self.store.get_status(self.capture_run_id, self.run_token)
        self.assertEqual(sealed["phase"], PHASE_CAPTURE_SEALED)
        self.assertEqual(
            sealed["progress"]["next_sequence_index"], EXPECTED_SAMPLE_COUNT
        )
        self.assertFalse(sealed["acquisition_artifact_verified"])
        self.assertFalse(sealed["capture_contract_binding_verified"])
        self.assertFalse(sealed["physical_capture_claim_authorized"])
        artifact = self.store.read_sealed_artifact(
            self.capture_run_id, self.run_token
        )
        artifact_summary = verify_capture_artifact(artifact)
        self.assertEqual(artifact_summary["sample_count"], EXPECTED_SAMPLE_COUNT)
        self.assertEqual(
            artifact_summary["evidence_class"], "physical_self_development"
        )
        self.assertFalse(artifact_summary["measurement_claim_authorized"])
        verified = self.store.verify_sealed_artifact(
            self.capture_run_id, self.run_token
        )
        self.assertEqual(verified["phase"], PHASE_ARTIFACT_VERIFIED)
        self.assertTrue(verified["acquisition_artifact_verified"])
        self.assertFalse(verified["capture_contract_binding_verified"])
        self.assertFalse(verified["physical_capture_claim_authorized"])
        artifact_text = (self.run_dir / CAPTURE_ARTIFACT_FILENAME).read_text(
            encoding="utf-8"
        )
        self.assertNotIn(self.run_token, artifact_text)

        final_retry = self.store.consume_challenge(
            self.capture_run_id,
            self.run_token,
            challenge["challenge_token"],
            observation,
        )
        self.assertTrue(final_retry["idempotent"])
        self.assertEqual(final_retry["phase"], PHASE_ARTIFACT_VERIFIED)

    def test_same_challenge_concurrent_retry_commits_once(self) -> None:
        challenge = self.store.issue_next_challenge(
            self.capture_run_id, self.run_token
        )
        observation = _observation(challenge["schedule_row"])
        results: list[dict] = []
        errors: list[BaseException] = []

        def consume() -> None:
            try:
                results.append(
                    self.store.consume_challenge(
                        self.capture_run_id,
                        self.run_token,
                        challenge["challenge_token"],
                        observation,
                    )
                )
            except BaseException as exc:  # pragma: no cover - assertion captures it
                errors.append(exc)

        threads = [threading.Thread(target=consume) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertFalse(errors)
        self.assertEqual(len(results), 2)
        self.assertEqual({result["idempotent"] for result in results}, {False, True})
        self.assertEqual(
            len(
                (self.run_dir / CALIBRATION_LEDGER_FILENAME)
                .read_text(encoding="utf-8")
                .splitlines()
            ),
            1,
        )

    def test_final_artifact_state_write_crash_reconciles_from_ledger(self) -> None:
        self._collect_calibration()
        status = self.store.get_status(self.capture_run_id, self.run_token)
        self.store.bind_model(
            self.capture_run_id,
            self.run_token,
            model_id=EVALUATION_MODEL_ID,
            model_sha256=EVALUATION_MODEL_SHA256,
            calibration_ledger_sha256=status["ledgers"]["calibration"][
                "sealed_sha256"
            ],
        )
        for _ in range(EXPECTED_EVALUATION_SAMPLE_COUNT - 1):
            self._issue_and_consume()
        final = self.store.issue_next_challenge(
            self.capture_run_id, self.run_token
        )
        observation = _observation(final["schedule_row"])
        with patch.object(
            self.store,
            "_write_state",
            side_effect=OSError("injected final state write crash"),
        ):
            with self.assertRaises(OSError):
                self.store.consume_challenge(
                    self.capture_run_id,
                    self.run_token,
                    final["challenge_token"],
                    observation,
                )
        self.assertTrue((self.run_dir / CAPTURE_ARTIFACT_FILENAME).is_file())
        restarted = MeasurementRunStore(self.root)
        recovered = restarted.get_status(self.capture_run_id, self.run_token)
        self.assertEqual(recovered["phase"], PHASE_CAPTURE_SEALED)
        self.assertEqual(
            recovered["progress"]["next_sequence_index"], EXPECTED_SAMPLE_COUNT
        )

    def test_abort_is_terminal_and_does_not_delete_evidence(self) -> None:
        self._issue_and_consume()
        aborted = self.store.abort_run(
            self.capture_run_id, self.run_token, reason="operator stopped rehearsal"
        )
        self.assertEqual(aborted["phase"], PHASE_ABORTED)
        self.assertTrue((self.run_dir / CALIBRATION_LEDGER_FILENAME).exists())
        with self.assertRaisesRegex(MeasurementRunStateError, "terminal"):
            self.store.issue_next_challenge(self.capture_run_id, self.run_token)

    def test_store_source_is_cpu_only_and_separate_from_participant_state(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "core"
            / "gaze_core"
            / "measurement_run_store.py"
        ).read_text(encoding="utf-8")
        for forbidden in (
            "import cv2",
            "import numpy",
            "import requests",
            "import socket",
            "import torch",
            "participant_study",
        ):
            self.assertNotIn(forbidden, source)
        self.assertNotIn("purge_session_images(", source)


class WebcamGazeMeasurementRunStoreMultiprocessTests(unittest.TestCase):
    """Process-lock contract kept out of the subprocess-denying offline worker."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = MeasurementRunStore(self.root)
        self.run_token = "wgmr_client_" + "6" * 64
        created = self.store.create_run(
            create_request_id="WGMCREQ-" + "d" * 32,
            run_token=self.run_token,
        )
        self.capture_run_id = created["capture_run_id"]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_multiprocess_spawn_exact_replay_commits_once(self) -> None:
        challenge = self.store.issue_next_challenge(
            self.capture_run_id, self.run_token
        )
        observation = _observation(challenge["schedule_row"])
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        processes = [
            context.Process(
                target=_multiprocess_consume_worker,
                args=(
                    str(self.root),
                    self.capture_run_id,
                    self.run_token,
                    challenge["challenge_token"],
                    observation,
                    queue,
                ),
            )
            for _ in range(2)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=20)
            self.assertEqual(process.exitcode, 0)
        results = [queue.get(timeout=5) for _ in processes]
        self.assertTrue(all(kind == "ok" for kind, _ in results), results)
        self.assertEqual(
            self.store.get_status(self.capture_run_id, self.run_token)["progress"][
                "next_sequence_index"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()

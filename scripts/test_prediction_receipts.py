"""Adversarial tests for server-issued single-use validation receipts."""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The focused receipt lane is CPU-only and uses NumPy for the shared uncertainty
# validator. Avoid the participant facade, which initializes unrelated OpenCV
# calibration helpers and Torch inference dependencies.
import core as core_package

_PARTICIPANT_PREFIX = "core.participant_study"
_saved_modules = {
    name: module
    for name, module in sys.modules.items()
    if name == _PARTICIPANT_PREFIX or name.startswith(_PARTICIPANT_PREFIX + ".")
}
_had_parent_attribute = hasattr(core_package, "participant_study")
_saved_parent_attribute = getattr(core_package, "participant_study", None)
_installed_namespace = _PARTICIPANT_PREFIX not in sys.modules
if _installed_namespace:
    package = types.ModuleType(_PARTICIPANT_PREFIX)
    package.__path__ = [str(ROOT / "core" / "participant_study")]
    sys.modules[_PARTICIPANT_PREFIX] = package

try:
    _general_collection = importlib.import_module(
        "core.participant_study.general_collection"
    )
    _protocol = importlib.import_module("core.participant_study.protocol")
    _store_module = importlib.import_module("core.participant_study.store")
finally:
    if _installed_namespace:
        for module_name in list(sys.modules):
            if module_name == _PARTICIPANT_PREFIX or module_name.startswith(
                _PARTICIPANT_PREFIX + "."
            ):
                del sys.modules[module_name]
        sys.modules.update(_saved_modules)
        if _had_parent_attribute:
            core_package.participant_study = _saved_parent_attribute
        elif hasattr(core_package, "participant_study"):
            delattr(core_package, "participant_study")

ParticipantStudyStore = _store_module.ParticipantStudyStore
canonical_sha256 = _general_collection.canonical_sha256
load_general_protocol = _general_collection.load_general_protocol
validation_target_definitions = _general_collection.validation_target_definitions
load_protocol = _protocol.load_protocol

TEST_MODEL_ARTIFACT_SHA256 = "a" * 64
UNCERTAINTY_DEFINITION_SHA256 = (
    "75a24c436e9a89024462268812ecc9be149a1958b3911e5cd71c3974b235a180"
)


def _scored_uncertainty() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "scored_no_threshold",
        "definition_sha256": UNCERTAINTY_DEFINITION_SHA256,
        "score": 0.75,
        "components": {
            "ood": {"value": 0.1, "percentile": 0.25},
            "leverage": {"value": 0.2, "percentile": 0.5},
            "disagreement": {"value": 0.0, "percentile": 0.75},
        },
        "jackknife_disagreement_covariance_norm": [[0.0, 0.0], [0.0, 0.0]],
        "jackknife_disagreement_covariance_px": [[0.0, 0.0], [0.0, 0.0]],
        "abstention": {"status": "not_selected", "threshold": None},
    }


def _rehearsal_settings() -> dict[str, object]:
    return {
        "LEXIGAZE_STUDY_MODE": "rehearsal",
        "LEXIGAZE_STUDY_REHEARSAL_MODE": "1",
        "LEXIGAZE_REHEARSAL_ACKNOWLEDGED_DEVELOPMENT_ONLY": "1",
        "LEXIGAZE_REHEARSAL_INVITES_ONLY": "1",
        "LEXIGAZE_REQUEST_BODY_LOGGING_DISABLED": "1",
        "LEXIGAZE_STORAGE_ENCRYPTED": "1",
        "LEXIGAZE_DATA_LOCATION": "encrypted-test-volume",
        "LEXIGAZE_PUBLIC_BASE_URL": "http://127.0.0.1:8080",
        "LEXIGAZE_DATA_RETENTION_DAYS": "30",
        "LEXIGAZE_RAW_FRAME_RETENTION_HOURS": "1",
    }


def _consent_payload(invite_code: str) -> dict[str, object]:
    protocol = load_protocol()
    return {
        "mode": "rehearsal",
        "invite_code": invite_code,
        "adult_confirmed": True,
        "private_space_confirmed": True,
        "consent_statements": {
            item["id"]: True for item in protocol["required_consent_statements"]
        },
        "comprehension_answers": {
            item["id"]: item["correct"]
            for item in protocol["comprehension_checks"]
        },
        "optional_scopes": {
            item["id"]: False for item in protocol["optional_scopes"]
        },
    }


def _profile() -> dict[str, str]:
    protocol = load_general_protocol()
    return {
        field: values[0]
        for field, values in protocol["profile_schema"]["required"].items()
    }


def _system_profile() -> dict[str, object]:
    protocol = load_general_protocol()
    return {
        "checks": {name: True for name in protocol["system_check"]["required"]},
        "device": {
            "device_class": "desktop",
            "browser_family": "chromium",
            "viewport_width": 1280,
            "viewport_height": 800,
            "device_pixel_ratio_bucket": "1_2",
            "camera_width": 640,
            "camera_height": 480,
            "estimated_camera_fps_band": "20_30",
        },
    }


def _assessment_viewport() -> dict[str, int]:
    return {"width_px": 1280, "height_px": 800}


def _capture_contract() -> dict[str, object]:
    return {
        "schema_version": 1,
        "intent_width_px": 1280,
        "intent_height_px": 720,
        "intent_frame_rate_hz": 30.0,
        "source_width_px": 1280,
        "source_height_px": 720,
        "source_frame_rate_hz": 30.0,
        "transport_width_px": 640,
        "transport_height_px": 360,
        "resize_policy": "fit_width_preserve_aspect",
        "mime_type": "image/jpeg",
        "jpeg_quality": 0.8,
        "mirror_applied": False,
        "facing_mode": "user",
    }


def _validation_samples(offset: float = 10.0) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for target in validation_target_definitions():
        target_x = float(
            int(float(target["target_x_viewport_fraction"]) * 1280 + 0.5)
        )
        target_y = float(
            int(float(target["target_y_viewport_fraction"]) * 800 + 0.5)
        )
        for repeat in range(3):
            samples.append(
                {
                    "target_id": target["target_id"],
                    "target_x_px": target_x,
                    "target_y_px": target_y,
                    "target_x_norm": target["target_x_norm"],
                    "target_y_norm": target["target_y_norm"],
                    "prediction_success": True,
                    "predicted_x_px": target_x + offset + repeat,
                    "predicted_y_px": target_y + offset - repeat,
                }
            )
    return samples


def _issue_prediction_receipts(
    store: object,
    session_id: str,
    access_token: str,
    *,
    phase: str,
    samples: list[dict[str, object]] | None = None,
    uncertainty: object = "scored",
) -> list[str]:
    issued: list[str] = []
    public = store.get_session(session_id, access_token)
    model_name = str(public["linked_data"]["model_name"])
    for sample in samples or _validation_samples():
        challenge = store.prepare_general_prediction_receipt(
            session_id,
            access_token,
            phase=phase,
            target_id=str(sample["target_id"]),
            model_name=model_name,
            model_artifact_sha256=TEST_MODEL_ARTIFACT_SHA256,
            viewport=_assessment_viewport(),
        )
        predicted_x = float(sample.get("predicted_x_px", 0.0))
        predicted_y = float(sample.get("predicted_y_px", 0.0))
        prediction_response = {
            "ok": True,
            "screen_xy_px": [predicted_x, predicted_y],
            "screen_xy_norm": [
                predicted_x / 1280.0 * 2.0 - 1.0,
                predicted_y / 800.0 * 2.0 - 1.0,
            ],
            "capture_contract_check": {
                "status": "compatible",
                "compatible": True,
                "reasons": [],
                "warnings": [],
            },
        }
        if uncertainty == "scored":
            prediction_response["uncertainty"] = _scored_uncertainty()
        elif uncertainty is not None:
            prediction_response["uncertainty"] = uncertainty
        receipt = store.issue_general_prediction_receipt(
            session_id,
            access_token,
            challenge=challenge,
            model_artifact_sha256_after=TEST_MODEL_ARTIFACT_SHA256,
            capture_contract=_capture_contract(),
            prediction_response=prediction_response,
            prediction_status=200,
        )
        issued.append(receipt["token"])
    return issued


def _validation_payload(summary: dict[str, object]) -> dict[str, object]:
    return {
        "samples": summary["samples"],
        "capture_contract": summary["capture_contract"],
        "prediction_receipt_bundle": summary["prediction_receipt_bundle"],
        "uncertainty_observations": summary["uncertainty_observations"],
        "uncertainty_summary": summary["uncertainty_summary"],
        "prediction_receipt_status": summary["prediction_receipt_status"],
        "prediction_receipts_verified": summary["prediction_receipts_verified"],
        "model_artifact_sha256": summary["model_artifact_sha256"],
        "gaze_measurement_contract_sha256": summary[
            "gaze_measurement_contract_sha256"
        ],
        "assessment_viewport": summary["assessment_viewport"],
    }


class PredictionReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="lexigaze-receipts-")
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.store = ParticipantStudyStore(
            self.root,
            settings=_rehearsal_settings(),
        )
        self.session_id, self.token = self._new_collection("receipt-test-model")

    def _new_collection(self, model_name: str) -> tuple[str, str]:
        pair = self.store.create_collection_invite_pairs(1)[0]
        enrolled = self.store.enroll(
            _consent_payload(pair["visits"][0]["invite_code"])
        )
        session_id = enrolled["study_session_id"]
        token = enrolled["access_token"]
        self.store.record_general_profile(session_id, token, _profile())
        self.store.record_general_system_check(session_id, token, _system_profile())
        self.store.start_calibration(session_id, token, "GAZE-RECEIPT")
        self.store.complete_calibration(
            session_id,
            token,
            {
                "passed": True,
                "test_fixture": True,
                "model_artifact_sha256": TEST_MODEL_ARTIFACT_SHA256,
                "capture_contract": _capture_contract(),
            },
            model_name=model_name,
            model_artifact_sha256=TEST_MODEL_ARTIFACT_SHA256,
        )
        self.store.start_general_collection(
            session_id,
            token,
            assessment_viewport=_assessment_viewport(),
            model_artifact_sha256=TEST_MODEL_ARTIFACT_SHA256,
        )
        return session_id, token

    def _issue(self, *, session_id: str | None = None, token: str | None = None):
        return _issue_prediction_receipts(
            self.store,
            session_id or self.session_id,
            token or self.token,
            phase="start",
        )

    def _record(self, receipts: list[str], *, artifact: str = TEST_MODEL_ARTIFACT_SHA256):
        return self.store.record_general_validation(
            self.session_id,
            self.token,
            phase="start",
            prediction_receipts=receipts,
            model_artifact_sha256=artifact,
        )

    def test_registry_is_private_and_all_hash_layers_recompute(self) -> None:
        receipts = self._issue()
        public_before = self.store.get_session(self.session_id, self.token)
        self.assertNotIn("prediction_receipts", public_before["general_collection"])
        session_path = next(self.root.rglob(f"{self.session_id}/session.json"))
        raw_before = session_path.read_text(encoding="utf-8")
        self.assertTrue(all(token not in raw_before for token in receipts))

        public = self._record(receipts)
        self.assertTrue(all(token not in json.dumps(public) for token in receipts))
        validation_public = public["general_collection"]["validations"]["start"]
        self.assertEqual(validation_public["prediction_receipt_status"], "verified")
        self.assertTrue(validation_public["prediction_receipts_verified"])
        bundle = validation_public["prediction_receipt_bundle"]
        self.assertEqual(bundle["schema_version"], 1)
        self.assertEqual(bundle["status"], "verified")
        self.assertEqual(bundle["count"], 15)
        bundle_core = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
        self.assertEqual(bundle["bundle_sha256"], canonical_sha256(bundle_core))
        uncertainty_summary = validation_public["uncertainty_summary"]
        self.assertEqual(uncertainty_summary["status"], "verified")
        self.assertEqual(uncertainty_summary["count"], 15)
        self.assertEqual(uncertainty_summary["scored_count"], 15)
        self.assertEqual(uncertainty_summary["unavailable_count"], 0)

        stored = json.loads(session_path.read_text(encoding="utf-8"))
        self.assertNotIn("prediction_receipts", public["general_collection"])
        validation = stored["general_collection"]["validations"]["start"]
        registry = stored["general_collection"]["prediction_receipts"]
        self.assertEqual(registry["schema_version"], 1)
        self.assertEqual(len(registry["records"]), 15)
        self.assertTrue(all(token not in json.dumps(stored) for token in receipts))
        ordered_records = sorted(
            registry["records"].values(),
            key=lambda item: item["issued"]["receipt_ordinal"],
        )
        for ordinal, record in enumerate(ordered_records):
            issued = record["issued"]
            self.assertEqual(
                record["issued_record_sha256"],
                canonical_sha256(issued),
            )
            self.assertEqual(issued["study_session_id"], self.session_id)
            self.assertEqual(issued["assessment_id"], stored["linked_data"]["assessment_id"])
            self.assertEqual(issued["model_name"], "receipt-test-model")
            self.assertEqual(
                issued["model_artifact_sha256"], TEST_MODEL_ARTIFACT_SHA256
            )
            self.assertEqual(issued["capture_session_id"], "GAZE-RECEIPT")
            self.assertEqual(issued["phase"], "start")
            self.assertEqual(issued["receipt_ordinal"], ordinal)
            self.assertEqual(issued["viewport"], _assessment_viewport())
            self.assertEqual(
                issued["measurement_contract_sha256"],
                stored["general_collection"]["gaze_measurement_contract"]["sha256"],
            )
            self.assertTrue(issued["capture_contract_check"]["compatible"])
            self.assertEqual(
                issued["prediction"]["uncertainty_schema_version"],
                1,
            )
            self.assertEqual(
                issued["prediction"]["uncertainty"]["status"],
                "scored_no_threshold",
            )
            self.assertEqual(record["consumed_validation_phase"], "start")
            self.assertTrue(record["consumed_at_utc"])
        observations = validation["uncertainty_observations"]
        self.assertEqual(len(observations), 15)
        self.assertEqual(
            [item["receipt_ordinal"] for item in observations],
            list(range(15)),
        )
        self.assertEqual({item["phase"] for item in observations}, {"start"})
        self.assertTrue(
            all("token" not in json.dumps(observation) for observation in observations)
        )
        self.assertTrue(
            all(
                not ({"screen_xy_px", "screen_xy_norm", "error"} & set(observation))
                for observation in observations
            )
        )
        self.assertEqual(
            validation["uncertainty_summary"]["observation_sha256s"],
            [canonical_sha256(observation) for observation in observations],
        )
        self.assertEqual(
            validation["uncertainty_summary"]["observations_sha256"],
            canonical_sha256(observations),
        )
        self.assertEqual(
            validation["validation_payload_sha256"],
            canonical_sha256(_validation_payload(validation)),
        )

    def test_exact_bundle_retry_is_idempotent_but_reorder_and_forgery_fail(self) -> None:
        receipts = self._issue()
        first = self._record(receipts)
        with self.store._lock:
            _, stored = self.store._read(self.session_id)
            collection = stored["general_collection"]
            summary = collection["validations"]["start"]
            self.assertTrue(
                self.store._is_idempotent_prediction_receipt_replay_locked(
                    stored,
                    collection,
                    phase="start",
                    receipt_tokens=receipts,
                    model_artifact_sha256=TEST_MODEL_ARTIFACT_SHA256,
                    existing_summary=summary,
                )
            )
        second = self._record(receipts)
        self.assertEqual(first, second)

        reordered = list(receipts)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        self.assertFalse(
            self.store._is_idempotent_prediction_receipt_replay_locked(
                stored,
                collection,
                phase="start",
                receipt_tokens=reordered,
                model_artifact_sha256=TEST_MODEL_ARTIFACT_SHA256,
                existing_summary=summary,
            )
        )
        with self.assertRaisesRegex(Exception, "validation is not expected"):
            self._record(reordered)
        forged = list(receipts)
        forged[-1] = forged[-1][:-1] + ("0" if forged[-1][-1] != "0" else "1")
        self.assertFalse(
            self.store._is_idempotent_prediction_receipt_replay_locked(
                stored,
                collection,
                phase="start",
                receipt_tokens=forged,
                model_artifact_sha256=TEST_MODEL_ARTIFACT_SHA256,
                existing_summary=summary,
            )
        )
        with self.assertRaisesRegex(Exception, "validation is not expected"):
            self._record(forged)

    def test_target_phase_model_viewport_and_model_artifact_drift_fail_closed(self) -> None:
        targets = validation_target_definitions()
        common = {
            "session_id": self.session_id,
            "access_token": self.token,
            "phase": "start",
            "target_id": targets[0]["target_id"],
            "model_name": "receipt-test-model",
            "model_artifact_sha256": TEST_MODEL_ARTIFACT_SHA256,
            "viewport": _assessment_viewport(),
        }
        for field, value, message in (
            ("target_id", targets[1]["target_id"], "server-frozen sequence"),
            ("phase", "end", "not expected"),
            ("model_name", "forged-model", "not linked"),
            ("viewport", {"width_px": 1279, "height_px": 800}, "viewport"),
        ):
            forged = {**common, field: value}
            with self.subTest(field=field), self.assertRaisesRegex(Exception, message):
                self.store.prepare_general_prediction_receipt(**forged)
        with self.assertRaisesRegex(Exception, "invalid study session credential"):
            self.store.prepare_general_prediction_receipt(
                **{**common, "access_token": "forged-access-token"}
            )

        first = self.store.prepare_general_prediction_receipt(**common)
        successful_response = {
            "ok": True,
            "screen_xy_px": [240.0, 170.0],
            "screen_xy_norm": [-0.625, -0.575],
            "capture_contract_check": {
                "status": "compatible",
                "compatible": True,
            },
        }
        with self.assertRaisesRegex(Exception, "changed during prediction"):
            self.store.issue_general_prediction_receipt(
                self.session_id,
                self.token,
                challenge=first,
                model_artifact_sha256_after="b" * 64,
                capture_contract=_capture_contract(),
                prediction_response=successful_response,
                prediction_status=200,
            )
        issued = self.store.issue_general_prediction_receipt(
            self.session_id,
            self.token,
            challenge=first,
            model_artifact_sha256_after=TEST_MODEL_ARTIFACT_SHA256,
            capture_contract=_capture_contract(),
            prediction_response=successful_response,
            prediction_status=200,
        )
        self.assertRegex(issued["token"], r"^PR-[A-F0-9]{48}$")
        with self.assertRaisesRegex(Exception, "artifact changed"):
            self.store.prepare_general_prediction_receipt(
                **{**common, "model_artifact_sha256": "b" * 64}
            )
        tampered_challenge = dict(
            self.store.prepare_general_prediction_receipt(**common)
        )
        tampered_challenge["receipt_ordinal"] = 14
        with self.assertRaisesRegex(Exception, "challenge hash mismatch"):
            self.store.issue_general_prediction_receipt(
                self.session_id,
                self.token,
                challenge=tampered_challenge,
                model_artifact_sha256_after=TEST_MODEL_ARTIFACT_SHA256,
                capture_contract=_capture_contract(),
                prediction_response={
                    "ok": True,
                    "screen_xy_px": [240.0, 170.0],
                    "screen_xy_norm": [-0.625, -0.575],
                    "capture_contract_check": {
                        "status": "compatible",
                        "compatible": True,
                    },
                },
                prediction_status=200,
            )

    def test_cross_session_and_validation_time_artifact_tamper_fail_closed(self) -> None:
        receipts = self._issue()
        other_session, other_token = self._new_collection("receipt-test-model")
        _issue_prediction_receipts(
            self.store,
            other_session,
            other_token,
            phase="start",
            samples=_validation_samples()[:1],
        )
        with self.assertRaisesRegex(Exception, "does not belong"):
            self.store.record_general_validation(
                other_session,
                other_token,
                phase="start",
                prediction_receipts=receipts,
                model_artifact_sha256=TEST_MODEL_ARTIFACT_SHA256,
            )
        with self.assertRaisesRegex(Exception, "artifact changed"):
            self._record(receipts, artifact="b" * 64)

    def test_only_explicit_no_face_failure_can_receive_a_failure_receipt(self) -> None:
        target_id = validation_target_definitions()[0]["target_id"]
        challenge = self.store.prepare_general_prediction_receipt(
            self.session_id,
            self.token,
            phase="start",
            target_id=target_id,
            model_name="receipt-test-model",
            model_artifact_sha256=TEST_MODEL_ARTIFACT_SHA256,
            viewport=_assessment_viewport(),
        )
        no_face = self.store.issue_general_prediction_receipt(
            self.session_id,
            self.token,
            challenge=challenge,
            model_artifact_sha256_after=TEST_MODEL_ARTIFACT_SHA256,
            capture_contract=_capture_contract(),
            prediction_response={
                "ok": False,
                "failure_stage": "attributable_sensor_failure",
                "failure_code": "no_face_detected",
                "error": "no face detected in frame",
                "capture_contract_check": {
                    "status": "compatible",
                    "compatible": True,
                },
            },
            prediction_status=400,
        )
        self.assertRegex(no_face["token"], r"^PR-[A-F0-9]{48}$")

        next_challenge = self.store.prepare_general_prediction_receipt(
            self.session_id,
            self.token,
            phase="start",
            target_id=target_id,
            model_name="receipt-test-model",
            model_artifact_sha256=TEST_MODEL_ARTIFACT_SHA256,
            viewport=_assessment_viewport(),
        )
        with self.assertRaisesRegex(Exception, "explicit no-face"):
            self.store.issue_general_prediction_receipt(
                self.session_id,
                self.token,
                challenge=next_challenge,
                model_artifact_sha256_after=TEST_MODEL_ARTIFACT_SHA256,
                capture_contract=_capture_contract(),
                prediction_response={
                    "ok": False,
                    "failure_stage": "inference_hard_error",
                    "capture_contract_check": {
                        "status": "compatible",
                        "compatible": True,
                    },
                },
                prediction_status=500,
            )
        for failure_code, status, screen_xy_px in (
            ("blink_detected", 400, None),
            ("no_face_detected", 200, None),
            ("no_face_detected", 400, [240.0, 170.0]),
        ):
            with self.subTest(
                failure_code=failure_code,
                status=status,
                coords=screen_xy_px,
            ), self.assertRaisesRegex(Exception, "explicit no-face"):
                self.store.issue_general_prediction_receipt(
                    self.session_id,
                    self.token,
                    challenge=next_challenge,
                    model_artifact_sha256_after=TEST_MODEL_ARTIFACT_SHA256,
                    capture_contract=_capture_contract(),
                    prediction_response={
                        "ok": False,
                        "failure_stage": "attributable_sensor_failure",
                        "failure_code": failure_code,
                        "screen_xy_px": screen_xy_px,
                        "capture_contract_check": {
                            "status": "compatible",
                            "compatible": True,
                        },
                    },
                    prediction_status=status,
                )
        with self.assertRaisesRegex(Exception, "coordinates disagree"):
            self.store.issue_general_prediction_receipt(
                self.session_id,
                self.token,
                challenge=next_challenge,
                model_artifact_sha256_after=TEST_MODEL_ARTIFACT_SHA256,
                capture_contract=_capture_contract(),
                prediction_response={
                    "ok": True,
                    "screen_xy_px": [240.0, 170.0],
                    "screen_xy_norm": [0.0, 0.0],
                    "capture_contract_check": {
                        "status": "compatible",
                        "compatible": True,
                    },
                },
                prediction_status=200,
            )
        with self.assertRaisesRegex(Exception, "capture-hard"):
            self.store.issue_general_prediction_receipt(
                self.session_id,
                self.token,
                challenge=next_challenge,
                model_artifact_sha256_after=TEST_MODEL_ARTIFACT_SHA256,
                capture_contract=_capture_contract(),
                prediction_response={
                    "ok": False,
                    "failure_stage": "capture_hard_error",
                    "capture_contract_check": {
                        "status": "mismatch",
                        "compatible": False,
                    },
                },
                prediction_status=409,
            )
        remaining = _issue_prediction_receipts(
            self.store,
            self.session_id,
            self.token,
            phase="start",
            samples=_validation_samples()[1:],
        )
        public = self._record([no_face["token"], *remaining])
        summary = public["general_collection"]["validations"]["start"]
        self.assertEqual(summary["uncertainty_summary"]["count"], 15)
        self.assertEqual(summary["uncertainty_summary"]["scored_count"], 14)
        self.assertEqual(summary["uncertainty_summary"]["unavailable_count"], 1)
        self.assertEqual(
            summary["uncertainty_observations"][0]["uncertainty"]["status"],
            "unavailable_sensor_failure",
        )
        self.assertEqual(
            summary["uncertainty_observations"][0]["uncertainty"][
                "schema_version"
            ],
            1,
        )
        session_path = next(self.root.rglob(f"{self.session_id}/session.json"))
        stored_summary = json.loads(session_path.read_text(encoding="utf-8"))[
            "general_collection"
        ]["validations"]["start"]
        self.assertEqual(
            stored_summary["validation_payload_sha256"],
            canonical_sha256(_validation_payload(stored_summary)),
        )

    def test_malformed_and_legacy_missing_uncertainty_are_explicitly_unavailable(
        self,
    ) -> None:
        malformed = _issue_prediction_receipts(
            self.store,
            self.session_id,
            self.token,
            phase="start",
            uncertainty={
                "schema_version": 1,
                "status": "scored_no_threshold",
                "definition_sha256": "0" * 64,
            },
        )
        malformed_public = self._record(malformed)
        malformed_summary = malformed_public["general_collection"]["validations"][
            "start"
        ]
        self.assertEqual(malformed_summary["uncertainty_summary"]["scored_count"], 0)
        self.assertEqual(
            {
                item["uncertainty"]["status"]
                for item in malformed_summary["uncertainty_observations"]
            },
            {"unavailable_invalid_observation"},
        )
        self.assertTrue(
            all(
                item["prediction_success"] is True
                for item in malformed_summary["uncertainty_observations"]
            )
        )
        baseline_id, baseline_token = self._new_collection("uncertainty-baseline-model")
        baseline_receipts = _issue_prediction_receipts(
            self.store,
            baseline_id,
            baseline_token,
            phase="start",
        )
        baseline_public = self.store.record_general_validation(
            baseline_id,
            baseline_token,
            phase="start",
            prediction_receipts=baseline_receipts,
            model_artifact_sha256=TEST_MODEL_ARTIFACT_SHA256,
        )
        self.assertEqual(
            malformed_public["general_collection"]["provisional_geometry_quality"][
                "recommended_gaze_mode"
            ],
            baseline_public["general_collection"]["provisional_geometry_quality"][
                "recommended_gaze_mode"
            ],
        )

        contradictory_id, contradictory_token = self._new_collection(
            "contradictory-uncertainty-model"
        )
        contradictory_receipts = _issue_prediction_receipts(
            self.store,
            contradictory_id,
            contradictory_token,
            phase="start",
            uncertainty={
                "schema_version": 1,
                "status": "unavailable_capture_failure",
                "reason": "capture failed",
            },
        )
        contradictory_public = self.store.record_general_validation(
            contradictory_id,
            contradictory_token,
            phase="start",
            prediction_receipts=contradictory_receipts,
            model_artifact_sha256=TEST_MODEL_ARTIFACT_SHA256,
        )
        self.assertEqual(
            {
                item["uncertainty"]["status"]
                for item in contradictory_public["general_collection"]["validations"][
                    "start"
                ]["uncertainty_observations"]
            },
            {"unavailable_invalid_observation"},
        )

        legacy_session_id, legacy_token = self._new_collection("legacy-receipt-model")
        legacy_receipts = _issue_prediction_receipts(
            self.store,
            legacy_session_id,
            legacy_token,
            phase="start",
        )
        session_path = next(self.root.rglob(f"{legacy_session_id}/session.json"))
        stored = json.loads(session_path.read_text(encoding="utf-8"))
        records = stored["general_collection"]["prediction_receipts"]["records"]
        for record in records.values():
            prediction = record["issued"]["prediction"]
            prediction.pop("uncertainty_schema_version")
            prediction.pop("uncertainty")
            record["issued_record_sha256"] = canonical_sha256(record["issued"])
        session_path.write_text(
            json.dumps(stored, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        legacy_public = self.store.record_general_validation(
            legacy_session_id,
            legacy_token,
            phase="start",
            prediction_receipts=legacy_receipts,
            model_artifact_sha256=TEST_MODEL_ARTIFACT_SHA256,
        )
        legacy_summary = legacy_public["general_collection"]["validations"]["start"]
        self.assertEqual(legacy_summary["uncertainty_summary"]["count"], 15)
        self.assertEqual(legacy_summary["uncertainty_summary"]["scored_count"], 0)
        self.assertEqual(
            {
                item["uncertainty"]["status"]
                for item in legacy_summary["uncertainty_observations"]
            },
            {"unavailable_receipt_missing"},
        )
        self.assertTrue(
            all(
                item["prediction_success"] is True
                for item in legacy_summary["uncertainty_observations"]
            )
        )

    def test_legacy_client_coordinates_are_ignored_and_explicitly_unavailable(self) -> None:
        public = self.store.record_general_validation(
            self.session_id,
            self.token,
            phase="start",
            samples=_validation_samples(offset=-999.0),
            capture_contract=_capture_contract(),
        )
        collection = public["general_collection"]
        summary = collection["validations"]["start"]
        self.assertEqual(summary["prediction_receipt_status"], "unavailable")
        self.assertFalse(summary["prediction_receipts_verified"])
        self.assertEqual(summary["sample_count"], 0)
        self.assertEqual(summary["prediction_receipt_bundle"]["count"], 0)
        self.assertEqual(summary["uncertainty_observations"], [])
        self.assertEqual(summary["uncertainty_summary"]["status"], "unavailable")
        self.assertEqual(summary["uncertainty_summary"]["count"], 0)
        self.assertFalse(collection["gaze_integrity"]["eligible"])
        self.assertEqual(
            collection["provisional_geometry_quality"]["recommended_gaze_mode"],
            "behavioral_only",
        )

    def test_receipt_record_mutation_is_detected_before_consumption(self) -> None:
        receipts = self._issue()
        session_path = next(self.root.rglob(f"{self.session_id}/session.json"))
        stored = json.loads(session_path.read_text(encoding="utf-8"))
        first_record = next(
            iter(stored["general_collection"]["prediction_receipts"]["records"].values())
        )
        first_record["issued"]["prediction"]["screen_xy_px"][0] += 999.0
        session_path.write_text(
            json.dumps(stored, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(Exception, "record hash mismatch"):
            self._record(receipts)

    def test_uncertainty_record_and_summary_tamper_fail_hash_or_idempotency(self) -> None:
        receipts = self._issue()
        session_path = next(self.root.rglob(f"{self.session_id}/session.json"))
        stored = json.loads(session_path.read_text(encoding="utf-8"))
        first_record = next(
            iter(stored["general_collection"]["prediction_receipts"]["records"].values())
        )
        first_record["issued"]["prediction"]["uncertainty"]["score"] = 0.123
        session_path.write_text(
            json.dumps(stored, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(Exception, "record hash mismatch"):
            self._record(receipts)

        semantic_id, semantic_token = self._new_collection("semantic-tamper-model")
        semantic_receipts = _issue_prediction_receipts(
            self.store,
            semantic_id,
            semantic_token,
            phase="start",
        )
        semantic_path = next(self.root.rglob(f"{semantic_id}/session.json"))
        semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
        semantic_record = next(
            iter(
                semantic["general_collection"]["prediction_receipts"][
                    "records"
                ].values()
            )
        )
        semantic_record["issued"]["prediction"]["uncertainty"]["score"] = 0.123
        semantic_record["issued_record_sha256"] = canonical_sha256(
            semantic_record["issued"]
        )
        semantic_path.write_text(
            json.dumps(semantic, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(Exception, "uncertainty evidence is invalid"):
            self.store.record_general_validation(
                semantic_id,
                semantic_token,
                phase="start",
                prediction_receipts=semantic_receipts,
                model_artifact_sha256=TEST_MODEL_ARTIFACT_SHA256,
            )

        clean_session_id, clean_token = self._new_collection("summary-tamper-model")
        clean_receipts = _issue_prediction_receipts(
            self.store,
            clean_session_id,
            clean_token,
            phase="start",
        )
        self.store.record_general_validation(
            clean_session_id,
            clean_token,
            phase="start",
            prediction_receipts=clean_receipts,
            model_artifact_sha256=TEST_MODEL_ARTIFACT_SHA256,
        )
        clean_path = next(self.root.rglob(f"{clean_session_id}/session.json"))
        clean = json.loads(clean_path.read_text(encoding="utf-8"))
        clean["general_collection"]["validations"]["start"]["uncertainty_summary"][
            "count"
        ] = 14
        clean_path.write_text(
            json.dumps(clean, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(Exception, "validation is not expected"):
            self.store.record_general_validation(
                clean_session_id,
                clean_token,
                phase="start",
                prediction_receipts=clean_receipts,
                model_artifact_sha256=TEST_MODEL_ARTIFACT_SHA256,
            )


if __name__ == "__main__":
    unittest.main()

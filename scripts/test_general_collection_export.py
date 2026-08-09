"""Focused stdlib tests for gaze provenance in the private exporter."""

from __future__ import annotations

import copy
import csv
import importlib
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from core.gaze_core.capture_contract import (
    build_fit_target_contract,
    compare_capture_contracts,
    load_participant_gaze_measurement_contract,
)

# System Python in the lightweight lane lacks OpenCV, which the package facade
# imports for unrelated calibration helpers. Temporarily install a namespace
# only while loading the pure exporter dependencies, then restore *all* module
# and parent-package state so a shared unittest worker can import the real
# participant-study facade afterward.
import core as core_package

_PARTICIPANT_PREFIX = "core.participant_study"
_saved_participant_modules = {
    name: module
    for name, module in sys.modules.items()
    if name == _PARTICIPANT_PREFIX or name.startswith(_PARTICIPANT_PREFIX + ".")
}
_had_core_participant_attribute = hasattr(core_package, "participant_study")
_saved_core_participant_attribute = getattr(core_package, "participant_study", None)
_installed_temporary_namespace = _PARTICIPANT_PREFIX not in sys.modules
if _installed_temporary_namespace:
    participant_study_package = types.ModuleType(_PARTICIPANT_PREFIX)
    participant_study_package.__path__ = [
        str(ROOT / "core" / "participant_study")
    ]
    sys.modules[_PARTICIPANT_PREFIX] = participant_study_package

try:
    _general_collection = importlib.import_module(
        "core.participant_study.general_collection"
    )
    _participant_protocol = importlib.import_module("core.participant_study.protocol")
    _exporter = importlib.import_module("scripts.export_general_collection_dataset")
finally:
    if _installed_temporary_namespace:
        for module_name in list(sys.modules):
            if module_name == _PARTICIPANT_PREFIX or module_name.startswith(
                _PARTICIPANT_PREFIX + "."
            ):
                del sys.modules[module_name]
        sys.modules.update(_saved_participant_modules)
        if _had_core_participant_attribute:
            core_package.participant_study = _saved_core_participant_attribute
        elif hasattr(core_package, "participant_study"):
            delattr(core_package, "participant_study")

canonical_sha256 = _general_collection.canonical_sha256
evaluate_validation_target_independence = (
    _general_collection.evaluate_validation_target_independence
)
load_general_bank = _general_collection.load_general_bank
load_general_protocol = _general_collection.load_general_protocol
summarize_validation_samples = _general_collection.summarize_validation_samples
validation_target_definitions = _general_collection.validation_target_definitions
load_protocol = _participant_protocol.load_protocol
export_bundle = _exporter.export_bundle


VIEWPORT = {"width_px": 1280, "height_px": 800}


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
        "jpeg_quality": 0.86,
        "mirror_applied": False,
        "facing_mode": "user",
    }


def _fit_target_contract(measurement_contract: dict[str, object]) -> dict[str, object]:
    participant_calibration = dict(
        measurement_contract["participant_calibration"]  # type: ignore[arg-type]
    )
    targets = [
        {
            "target_x_norm": float(target["target_x_viewport_fraction"]) * 2.0 - 1.0,
            "target_y_norm": float(target["target_y_viewport_fraction"]) * 2.0 - 1.0,
        }
        for target in participant_calibration["frozen_targets"]  # type: ignore[index]
    ]
    return build_fit_target_contract(targets)


def _validation_summary(
    measurement_contract: dict[str, object],
    *,
    offset_px: float,
    calibration_capture: dict[str, object],
    fit_target_contract: dict[str, object],
) -> dict[str, object]:
    samples: list[dict[str, object]] = []
    for target in validation_target_definitions(measurement_contract):
        target_x_px = float(
            math.floor(
                float(target["target_x_viewport_fraction"])
                * VIEWPORT["width_px"]
                + 0.5
            )
        )
        target_y_px = float(
            math.floor(
                float(target["target_y_viewport_fraction"])
                * VIEWPORT["height_px"]
                + 0.5
            )
        )
        for _ in range(3):
            samples.append(
                {
                    "target_id": target["target_id"],
                    "target_x_norm": target["target_x_norm"],
                    "target_y_norm": target["target_y_norm"],
                    "target_x_px": target_x_px,
                    "target_y_px": target_y_px,
                    "prediction_success": True,
                    "predicted_x_px": target_x_px + offset_px,
                    "predicted_y_px": target_y_px,
                }
            )
    summary = summarize_validation_samples(
        samples,
        viewport_width_px=VIEWPORT["width_px"],
        viewport_height_px=VIEWPORT["height_px"],
        measurement_contract=measurement_contract,
    )
    contract_sha256 = canonical_sha256(measurement_contract)
    provenance = {
        "contract_id": measurement_contract["contract_id"],
        "contract_version": measurement_contract["contract_version"],
        "sha256": contract_sha256,
    }
    observed_capture = _capture_contract()
    summary["capture_contract"] = observed_capture
    summary["assessment_viewport"] = dict(VIEWPORT)
    summary["gaze_measurement_contract"] = provenance
    summary["gaze_measurement_contract_sha256"] = contract_sha256
    summary["samples_sha256"] = canonical_sha256(summary["samples"])
    summary["validation_payload_sha256"] = canonical_sha256(
        {
            "samples": summary["samples"],
            "capture_contract": observed_capture,
            "gaze_measurement_contract_sha256": contract_sha256,
            "assessment_viewport": VIEWPORT,
        }
    )
    summary["capture_contract_check"] = compare_capture_contracts(
        calibration_capture,
        observed_capture,
    )
    summary["target_independence_check"] = evaluate_validation_target_independence(
        summary,
        fit_target_contract,
        measurement_contract=measurement_contract,
    )
    return summary


def _session(
    root: Path,
    *,
    session_id: str,
    participant_id: str,
    pair_id: str,
    visit_index: int,
    measurement_contract: dict[str, object] | None = None,
) -> Path:
    protocol = load_general_protocol()
    bank = load_general_bank()
    contract = copy.deepcopy(
        measurement_contract or load_participant_gaze_measurement_contract()
    )
    contract_sha256 = canonical_sha256(contract)
    calibration_capture = _capture_contract()
    fit_target_contract = _fit_target_contract(contract)
    start = _validation_summary(
        contract,
        offset_px=6.0,
        calibration_capture=calibration_capture,
        fit_target_contract=fit_target_contract,
    )
    end = _validation_summary(
        contract,
        offset_px=9.0,
        calibration_capture=calibration_capture,
        fit_target_contract=fit_target_contract,
    )
    face_scale = (0.6 - 0.2) * (0.6 - 0.2)
    expected_quality = {
        "median_spatial_error_px": 9.0,
        "p90_spatial_error_px": 9.0,
        "precision_rms_px": 0.0,
        "prediction_success_fraction": 1.0,
        "effective_sampling_hz": 10.0,
        "head_pose_range": [0.0, 0.0],
        "face_scale_range": 0.0,
        "drift_change_px": 3.0,
        "capture_contract_eligible": True,
        "target_independence_eligible": True,
        "gaze_integrity_eligible": True,
        "telemetry_segments_contiguous": True,
    }
    session = {
        "schema_version": 1,
        "study_session_id": session_id,
        "participant_id": participant_id,
        "state": "completed",
        "created_at_utc": "2026-08-10T00:00:00+00:00",
        "events": [
            {
                "event": "general_collection_completed",
                "at_utc": "2026-08-10T01:00:00+00:00",
            }
        ],
        "collection_assignment": {
            "protocol_sha256": canonical_sha256(protocol),
            "bank_sha256": canonical_sha256(bank),
            "pair_id": pair_id,
            "visit_index": visit_index,
            "schedule_cell": 0,
            "sequence": "AB",
            "form_id": "form_a" if visit_index == 1 else "form_b",
            "order_cell": 0,
        },
        "linked_data": {"gaze_session_id": f"GAZE-{session_id}"},
        "data_governance": {
            "storage_security": "unencrypted_self_development",
            "retention_policy": "manual_until_researcher_deletes",
            "self_only": True,
        },
        "general_collection": {
            "assessment_id": f"GC-{session_id}",
            "profile": {
                "english_l1": "yes",
                "english_age_of_acquisition_band": "0_5",
                "weekly_english_reading_band": "5_plus_hours",
                "vision_correction": "none",
                "education_band": "graduate",
            },
            "gaze_measurement_contract": {
                "contract_id": contract["contract_id"],
                "contract_version": contract["contract_version"],
                "sha256": contract_sha256,
                "contract": contract,
            },
            "assessment_viewport": dict(VIEWPORT),
            "validations": {"start": start, "end": end},
            "gaze_integrity": {"eligible": True, "reasons": []},
            "rounds": [{"reading_elapsed_ms": 1000.0}],
            "telemetry_stats": {
                "batch_count": 1,
                "attempt_count": 10,
                "successful_count": 10,
                "head_pose_min": [0.0, 0.0],
                "head_pose_max": [0.0, 0.0],
                "face_scale_min": face_scale,
                "face_scale_max": face_scale,
            },
            "gaze_quality_metrics": dict(expected_quality),
            "gaze_quality_band": "word_level_candidate",
        },
        "quality": {
            "general_system_check": {
                "device": {
                    "device_class": "desktop",
                    "browser_family": "chromium",
                    "viewport_width": VIEWPORT["width_px"],
                    "viewport_height": VIEWPORT["height_px"],
                    "camera_width": 1280,
                    "camera_height": 720,
                    "estimated_camera_fps_band": "20_30",
                }
            },
            "calibration": {
                "capture_contract": calibration_capture,
                "fit_target_contract": fit_target_contract,
            },
            "general_collection": {
                **expected_quality,
                "geometry_contract_eligible": True,
                "gaze_quality_band": "word_level_candidate",
            },
        },
    }
    session_dir = (
        root
        / "data"
        / "participant_studies"
        / load_protocol()["protocol_id"]
        / "rehearsals"
        / session_id
    )
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "session.json").write_text(
        json.dumps(session, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    telemetry_payload = {
        "batch_id": "B-12345678",
        "passage_id": "fixture-passage",
        "viewport": dict(VIEWPORT),
        "samples": [
            {
                "monotonic_elapsed_ms": float(sample_index * 100),
                "prediction_success": True,
                "coarse_failure_code": None,
                "screen_xy_norm": [0.0, 0.0],
                "screen_xy_px": [640.0, 400.0],
                "gaze_pitch_yaw": [0.0, 0.0],
                "head_pose_pitch_yaw": [0.0, 0.0],
                "normalized_face_bbox": [0.2, 0.2, 0.6, 0.6],
                "nearest_word_index": None,
            }
            for sample_index in range(10)
        ],
    }
    telemetry = {
        "schema_version": 1,
        "participant_id": participant_id,
        "study_session_id": session_id,
        "visit_index": visit_index,
        "capture_session_id": f"GAZE-{session_id}",
        "payload_sha256": canonical_sha256(telemetry_payload),
        **telemetry_payload,
    }
    telemetry_path = (
        session_dir / "collection" / "telemetry" / "fixture-passage" / "B-12345678.json"
    )
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_path.write_text(
        json.dumps(telemetry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    round_path = session_dir / "collection" / "rounds" / "R01.json"
    round_path.parent.mkdir(parents=True, exist_ok=True)
    round_path.write_text(
        json.dumps(
            {
                "round_number": 1,
                "passage_id": "fixture-passage",
                "passage_family_id": "fixture-family",
                "difficulty_band": "standard",
                "reading_elapsed_ms": 1000,
                "passage_self_report": {
                    "understanding": 4,
                    "mental_effort": 2,
                    "read_complete": True,
                    "interrupted": False,
                },
                "word_reviews": [
                    {
                        "probe_id": "fixture-probe",
                        "surface": "fixture",
                        "stratum": "known",
                        "label": "known",
                    }
                ],
                "word_layout": [],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return session_dir / "session.json"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class GeneralCollectionExportProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="lexigaze-export-")
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def test_dependency_bootstrap_does_not_leak_fake_participant_package(self) -> None:
        participant_module = sys.modules.get(_PARTICIPANT_PREFIX)
        self.assertTrue(
            participant_module is None
            or bool(getattr(participant_module, "__file__", None)),
            "temporary participant-study namespace leaked into the unittest worker",
        )

    def test_valid_session_exports_bound_gaze_with_provenance(self) -> None:
        session_path = _session(
            self.root,
            session_id="ST-VALID",
            participant_id="P-VALID",
            pair_id="PAIR-VALID",
            visit_index=1,
        )
        output = self.root / "export"
        manifest = export_bundle(self.root, output)
        contract_sha256 = json.loads(session_path.read_text(encoding="utf-8"))[
            "general_collection"
        ]["gaze_measurement_contract"]["sha256"]

        session_row = _rows(output / "sessions.csv")[0]
        self.assertEqual(session_row["gaze_export_status"], "eligible")
        self.assertEqual(
            session_row["gaze_measurement_contract_sha256"],
            contract_sha256,
        )
        self.assertEqual(session_row["pair_gaze_comparison_status"], "single_visit_only")
        self.assertEqual(len(_rows(output / "validation_samples.csv")), 30)
        self.assertEqual(len(_rows(output / "gaze_telemetry.csv")), 10)
        self.assertEqual(len(_rows(output / "gaze_excluded_sessions.csv")), 0)
        self.assertEqual(
            manifest["gaze_provenance"]["session_gaze_eligible_count"],
            1,
        )
        self.assertFalse(
            manifest["gaze_provenance"]["prediction_values_tamper_resistant"]
        )

    def test_legacy_gaze_is_excluded_while_behavior_is_retained(self) -> None:
        session_path = _session(
            self.root,
            session_id="ST-LEGACY",
            participant_id="P-LEGACY",
            pair_id="PAIR-LEGACY",
            visit_index=1,
        )
        session = json.loads(session_path.read_text(encoding="utf-8"))
        del session["general_collection"]["gaze_measurement_contract"]
        session_path.write_text(
            json.dumps(session, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output = self.root / "export"
        manifest = export_bundle(self.root, output)
        session_row = _rows(output / "sessions.csv")[0]
        excluded = _rows(output / "gaze_excluded_sessions.csv")[0]
        self.assertEqual(session_row["gaze_export_status"], "excluded")
        self.assertEqual(session_row["gaze_quality_band"], "unavailable")
        self.assertEqual(session_row["median_spatial_error_px"], "")
        self.assertIn("measurement_contract_snapshot_unavailable", excluded["exclusion_reasons"])
        self.assertEqual(len(_rows(output / "passages.csv")), 1)
        self.assertEqual(len(_rows(output / "word_reviews.csv")), 1)
        self.assertEqual(len(_rows(output / "validation_samples.csv")), 0)
        self.assertEqual(len(_rows(output / "gaze_telemetry.csv")), 0)
        self.assertEqual(
            manifest["gaze_provenance"]["session_gaze_excluded_count"],
            1,
        )

    def test_tampered_validation_payload_hash_excludes_only_gaze(self) -> None:
        session_path = _session(
            self.root,
            session_id="ST-TAMPER",
            participant_id="P-TAMPER",
            pair_id="PAIR-TAMPER",
            visit_index=1,
        )
        session = json.loads(session_path.read_text(encoding="utf-8"))
        session["general_collection"]["validations"]["start"][
            "validation_payload_sha256"
        ] = "0" * 64
        session_path.write_text(
            json.dumps(session, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output = self.root / "export"
        export_bundle(self.root, output)
        excluded = _rows(output / "gaze_excluded_sessions.csv")[0]
        self.assertIn(
            "start_validation_payload_hash_mismatch",
            excluded["exclusion_reasons"],
        )
        self.assertEqual(len(_rows(output / "passages.csv")), 1)
        self.assertEqual(len(_rows(output / "validation_samples.csv")), 0)

    def test_completed_session_without_telemetry_excludes_gaze(self) -> None:
        session_path = _session(
            self.root,
            session_id="ST-NO-TELEMETRY",
            participant_id="P-NO-TELEMETRY",
            pair_id="PAIR-NO-TELEMETRY",
            visit_index=1,
        )
        telemetry_paths = list(
            (session_path.parent / "collection" / "telemetry").rglob("*.json")
        )
        self.assertEqual(len(telemetry_paths), 1)
        telemetry_paths[0].unlink()

        output = self.root / "export"
        export_bundle(self.root, output)
        session_row = _rows(output / "sessions.csv")[0]
        excluded = _rows(output / "gaze_excluded_sessions.csv")[0]
        self.assertEqual(session_row["gaze_export_status"], "excluded")
        self.assertIn(
            "completed_gaze_session_has_no_telemetry",
            excluded["exclusion_reasons"],
        )
        self.assertEqual(len(_rows(output / "gaze_telemetry.csv")), 0)
        self.assertEqual(len(_rows(output / "validation_samples.csv")), 0)
        self.assertEqual(len(_rows(output / "passages.csv")), 1)

    def test_tampered_final_quality_metrics_exclude_gaze(self) -> None:
        session_path = _session(
            self.root,
            session_id="ST-QUALITY-TAMPER",
            participant_id="P-QUALITY-TAMPER",
            pair_id="PAIR-QUALITY-TAMPER",
            visit_index=1,
        )
        session = json.loads(session_path.read_text(encoding="utf-8"))
        final_quality = session["quality"]["general_collection"]
        final_quality["median_spatial_error_px"] = 999999.0
        final_quality["effective_sampling_hz"] = 999999.0
        session_path.write_text(
            json.dumps(session, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output = self.root / "export"
        export_bundle(self.root, output)
        session_row = _rows(output / "sessions.csv")[0]
        excluded = _rows(output / "gaze_excluded_sessions.csv")[0]
        self.assertEqual(session_row["gaze_export_status"], "excluded")
        self.assertEqual(session_row["median_spatial_error_px"], "")
        self.assertEqual(session_row["effective_sampling_hz"], "")
        self.assertIn(
            "stored_final_median_spatial_error_px_mismatch",
            excluded["exclusion_reasons"],
        )
        self.assertIn(
            "stored_final_effective_sampling_hz_mismatch",
            excluded["exclusion_reasons"],
        )
        self.assertEqual(len(_rows(output / "gaze_telemetry.csv")), 0)
        self.assertEqual(len(_rows(output / "validation_samples.csv")), 0)

    def test_pair_contract_mismatch_keeps_sessions_but_forbids_pair_comparison(self) -> None:
        first_contract = load_participant_gaze_measurement_contract()
        second_contract = copy.deepcopy(first_contract)
        second_contract["export_fixture_revision"] = "different-frozen-contract"
        _session(
            self.root,
            session_id="ST-PAIR1",
            participant_id="P-PAIR",
            pair_id="PAIR-2",
            visit_index=1,
            measurement_contract=first_contract,
        )
        _session(
            self.root,
            session_id="ST-PAIR2",
            participant_id="P-PAIR",
            pair_id="PAIR-2",
            visit_index=2,
            measurement_contract=second_contract,
        )

        output = self.root / "export"
        manifest = export_bundle(self.root, output)
        sessions = _rows(output / "sessions.csv")
        self.assertEqual({row["gaze_export_status"] for row in sessions}, {"eligible"})
        self.assertEqual(
            {row["pair_gaze_comparison_status"] for row in sessions},
            {"pair_measurement_contract_mismatch"},
        )
        self.assertEqual({row["pair_gaze_comparable"] for row in sessions}, {"False"})
        self.assertEqual(len(_rows(output / "validation_samples.csv")), 60)
        self.assertEqual(len(_rows(output / "gaze_telemetry.csv")), 20)
        self.assertEqual(
            manifest["gaze_provenance"]["pair_comparable_count"],
            0,
        )
        self.assertEqual(
            manifest["gaze_provenance"]["pair_comparison_status_counts"],
            {"pair_measurement_contract_mismatch": 1},
        )

    def test_ineligible_sibling_does_not_exclude_valid_visit_gaze(self) -> None:
        _session(
            self.root,
            session_id="ST-SIBLING1",
            participant_id="P-SIBLING",
            pair_id="PAIR-SIBLING",
            visit_index=1,
        )
        invalid_path = _session(
            self.root,
            session_id="ST-SIBLING2",
            participant_id="P-SIBLING",
            pair_id="PAIR-SIBLING",
            visit_index=2,
        )
        invalid = json.loads(invalid_path.read_text(encoding="utf-8"))
        del invalid["general_collection"]["gaze_measurement_contract"]
        invalid_path.write_text(
            json.dumps(invalid, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output = self.root / "export"
        manifest = export_bundle(self.root, output)
        sessions = {
            row["study_session_id"]: row for row in _rows(output / "sessions.csv")
        }
        self.assertEqual(sessions["ST-SIBLING1"]["gaze_export_status"], "eligible")
        self.assertEqual(sessions["ST-SIBLING2"]["gaze_export_status"], "excluded")
        self.assertEqual(
            {
                row["pair_gaze_comparison_status"]
                for row in sessions.values()
            },
            {"paired_visit_gaze_ineligible"},
        )
        self.assertEqual(len(_rows(output / "validation_samples.csv")), 30)
        self.assertEqual(len(_rows(output / "gaze_telemetry.csv")), 10)
        self.assertEqual(
            manifest["gaze_provenance"]["session_gaze_eligible_count"],
            1,
        )


if __name__ == "__main__":
    unittest.main()

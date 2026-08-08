"""CPU-only contracts for the frozen Columbia cross-domain experiment."""

from __future__ import annotations

import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from scripts.audit_columbia_cross_domain_result import _nested_close
from scripts.columbia_gaze.data import (
    EyeCorners,
    affine_eye_crop,
    candidate_eye_pair,
    load_eye_corner_annotations,
    parse_filename,
)
from scripts.columbia_gaze.metrics import (
    fuse_angle_predictions,
    summarize_model,
    zero_gaze_summary,
)
from scripts.columbia_gaze.prepare import _failure_code
from scripts.run_columbia_cross_domain_gaze import (
    _effectiveness_decision,
    _validate_protocols,
    _write_prediction_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
BASE_PROTOCOL = (
    ROOT
    / "docs"
    / "experiments"
    / "protocols"
    / "2026-08-08-columbia-cross-domain-gaze-v1.json"
)
V2_PROTOCOL = (
    ROOT
    / "docs"
    / "experiments"
    / "protocols"
    / "2026-08-08-columbia-cross-domain-gaze-v2.json"
)


class ColumbiaDataContractTests(unittest.TestCase):
    def test_filename_contract_preserves_frozen_signs(self) -> None:
        path = Path("dataset") / "0001" / "0001_2m_-30P_10V_15H.jpg"
        sample = parse_filename(path)

        self.assertEqual(sample.subject, "0001")
        np.testing.assert_allclose(
            sample.target_pitch_yaw,
            np.radians([10.0, 15.0]),
        )
        np.testing.assert_allclose(
            sample.head_pitch_yaw,
            np.radians([0.0, -30.0]),
        )

    def test_filename_contract_rejects_unknown_grid_or_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "unexpected horizontal"):
            parse_filename(Path("0001") / "0001_2m_0P_0V_20H.jpg")
        with self.assertRaisesRegex(ValueError, "directory mismatch"):
            parse_filename(Path("0002") / "0001_2m_0P_0V_0H.jpg")

    def test_eye_annotation_schema_and_coordinates_are_exact(self) -> None:
        header = (
            "IMAGE,RIGHT_EYE_IN_X,RIGHT_EYE_IN_Y,RIGHT_EYE_OUT_X,"
            "RIGHT_EYE_OUT_Y,LEFT_EYE_IN_X,LEFT_EYE_IN_Y,LEFT_EYE_OUT_X,"
            "LEFT_EYE_OUT_Y\n"
        )
        with tempfile.TemporaryDirectory(prefix="lexigaze-columbia-csv-") as name:
            path = Path(name) / "corners.csv"
            path.write_text(
                header + "sample,10,20,30,40,50,60,70,80\n",
                encoding="utf-8",
            )
            loaded = load_eye_corner_annotations(path)
            self.assertEqual(
                loaded["sample"],
                EyeCorners(
                    right_in=(10.0, 20.0),
                    right_out=(30.0, 40.0),
                    left_in=(50.0, 60.0),
                    left_out=(70.0, 80.0),
                ),
            )
            path.write_text(
                header + "sample,5184,20,30,40,50,60,70,80\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "out-of-frame"):
                load_eye_corner_annotations(path)

    def test_affine_eye_crop_maps_frozen_corner_targets(self) -> None:
        gray = np.tile(np.arange(120, dtype=np.uint8), (80, 1))
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        with (
            mock.patch("scripts.columbia_gaze.data.EXPECTED_WIDTH", 120),
            mock.patch("scripts.columbia_gaze.data.EXPECTED_HEIGHT", 80),
        ):
            crop = affine_eye_crop(image, (60.0, 40.0), (30.0, 40.0))

        self.assertEqual(crop.shape, (36, 60))
        self.assertAlmostEqual(float(crop[18, 15]), 30.0, delta=1.0)
        self.assertAlmostEqual(float(crop[18, 45]), 60.0, delta=1.0)

    def test_candidate_pair_flips_only_anatomical_right_eye(self) -> None:
        left = np.arange(12, dtype=np.uint8).reshape(3, 4)
        right = (np.arange(12, dtype=np.uint8) + 20).reshape(3, 4)
        corners = EyeCorners((1, 1), (2, 1), (3, 1), (4, 1))
        with mock.patch(
            "scripts.columbia_gaze.data.affine_eye_crop",
            side_effect=[left, right],
        ):
            result_left, result_right = candidate_eye_pair(
                np.zeros((1, 1, 3), dtype=np.uint8),
                corners,
            )

        np.testing.assert_array_equal(result_left, left)
        np.testing.assert_array_equal(result_right, right[:, ::-1])
        self.assertTrue(result_right.flags.c_contiguous)


class ColumbiaMetricContractTests(unittest.TestCase):
    def test_vector_fusion_preserves_identical_angle_predictions(self) -> None:
        prediction = np.asarray([[0.1, -0.2], [-0.3, 0.4]])
        members = np.stack((prediction, prediction, prediction))

        np.testing.assert_allclose(
            fuse_angle_predictions(members),
            prediction,
            atol=1e-12,
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            fuse_angle_predictions(np.zeros((2, 2)))

    def test_aggregate_metrics_and_effectiveness_use_subject_pairs(self) -> None:
        targets = np.asarray(
            [[0.1, 0.0], [0.2, 0.1], [0.1, -0.1], [0.2, 0.0]],
            dtype=np.float64,
        )
        subjects = np.asarray([0, 0, 1, 1])
        factors = np.asarray([0, 1, 0, 1])
        zero, zero_subjects = zero_gaze_summary(
            targets,
            subjects,
            head_poses=factors,
            vertical_gazes=factors,
            horizontal_gazes=factors,
        )
        model = summarize_model(
            targets,
            targets,
            subjects,
            head_poses=factors,
            vertical_gazes=factors,
            horizontal_gazes=factors,
            zero_subject_means=zero_subjects,
            bootstrap_resamples=1000,
            bootstrap_seed=20260813,
        )
        decision = _effectiveness_decision(model, zero, minimum_subjects=2)

        self.assertLess(model["macro_subject_mean_degrees"], 1e-6)
        self.assertEqual(model["subjects_beating_zero_gaze"], 2)
        self.assertTrue(decision["passed"])

    def test_nested_result_comparison_is_numeric_and_schema_strict(self) -> None:
        self.assertTrue(_nested_close({"x": [1.0]}, {"x": [1.0 + 1e-13]}))
        self.assertFalse(_nested_close({"x": 1}, {"y": 1}))
        self.assertFalse(_nested_close({"x": True}, {"x": 1}))


class ColumbiaExecutionContractTests(unittest.TestCase):
    def test_v2_protocol_inherits_the_committed_v1_contract(self) -> None:
        base_bytes = BASE_PROTOCOL.read_bytes()
        base = json.loads(base_bytes.decode("utf-8"))
        protocol = json.loads(V2_PROTOCOL.read_text(encoding="utf-8"))

        _validate_protocols(protocol, base, base_bytes)
        self.assertEqual(
            protocol["base_protocol"]["sha256"],
            hashlib.sha256(base_bytes).hexdigest(),
        )

    def test_prediction_evidence_round_trip_is_pickle_free(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-columbia-evidence-") as name:
            path = Path(name) / "evidence.npz"
            _write_prediction_evidence(
                path,
                integrity={"protocol_sha256": "abc"},
                candidate_prediction=np.zeros((3, 2)),
                production_prediction=np.zeros((2, 2)),
                targets=np.zeros((3, 2)),
                subject_indices=np.asarray([0, 0, 1]),
                head_poses=np.asarray([0, 15, -15]),
                vertical_gazes=np.asarray([0, 10, -10]),
                horizontal_gazes=np.asarray([0, 5, -5]),
                fallback_mask=np.asarray([False, False, True]),
                production_success=np.asarray([True, False, True]),
            )
            with np.load(path, allow_pickle=False) as evidence:
                self.assertEqual(
                    json.loads(str(evidence["metadata_json"].item())),
                    {"protocol_sha256": "abc"},
                )
                self.assertEqual(evidence["candidate_prediction"].shape, (3, 2))
                self.assertEqual(evidence["production_prediction"].shape, (2, 2))

    def test_failure_codes_are_aggregate_and_stable(self) -> None:
        self.assertEqual(
            _failure_code(ValueError("no face detected")), "no_face_detected"
        )
        self.assertEqual(_failure_code(ValueError("bad shape")), "value_error")
        self.assertEqual(_failure_code(RuntimeError("boom")), "RuntimeError")

    def test_cpu_stage_modules_have_no_top_level_torch_import(self) -> None:
        paths = (
            ROOT / "scripts" / "columbia_gaze" / "data.py",
            ROOT / "scripts" / "columbia_gaze" / "metrics.py",
            ROOT / "scripts" / "columbia_gaze" / "prepare.py",
            ROOT / "scripts" / "columbia_gaze" / "train.py",
            ROOT / "scripts" / "run_columbia_cross_domain_gaze.py",
            ROOT / "scripts" / "audit_columbia_cross_domain_result.py",
        )
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports = []
            for node in tree.body:
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            self.assertNotIn("torch", imports, path.name)


if __name__ == "__main__":
    unittest.main()

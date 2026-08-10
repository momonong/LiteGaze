"""CPU-only contracts for leakage-resistant gaze uncertainty v2 phase 1."""

from __future__ import annotations

import ast
import inspect
import json
import math
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from core.gaze_core import uncertainty as compatibility_uncertainty
from core.gaze_core import uncertainty_contract
from core.gaze_core.motion_experiment import build_uncertainty_v2_bundle
from core.gaze_core.stage_pipeline import apply_stage_chain
from core.gaze_core.uncertainty import (
    build_fixed_coverage_risk,
    canonical_sha256,
    load_frozen_definition,
    normalize_uncertainty_observation,
    score_samples,
    unavailable_uncertainty,
    uncertainty_status_for_model,
    validate_complete_motion_grid,
)


ROOT = Path(__file__).resolve().parents[1]
BLOCKS = {
    "neutral": (0.04, 0.00, 0.50, 0.50, 0.58),
    "left": (0.04, -0.30, 0.47, 0.50, 0.58),
    "right": (0.04, 0.30, 0.53, 0.50, 0.58),
    "near": (0.10, 0.00, 0.50, 0.53, 0.78),
    "far": (-0.08, 0.00, 0.50, 0.47, 0.40),
}
TARGETS = np.asarray(
    [
        (-0.84, -0.80),
        (0.00, -0.80),
        (0.84, -0.80),
        (-0.84, 0.00),
        (0.00, 0.00),
        (0.84, 0.00),
        (-0.84, 0.80),
        (0.00, 0.80),
        (0.84, 0.80),
        (-0.42, -0.40),
        (0.42, -0.40),
        (-0.42, 0.40),
        (0.42, 0.40),
    ],
    dtype=np.float64,
)
IDENTITY_STAGE = {
    "stage": 1,
    "calibrator_type": "gaze_polynomial",
    "W": [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
    "poly_degree": 1,
    "alpha": 0.001,
}

FROZEN_DEFINITION_SHA256 = (
    "75a24c436e9a89024462268812ecc9be149a1958b3911e5cd71c3974b235a180"
)


def _golden_runtime_observation() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "scored_no_threshold",
        "definition_sha256": FROZEN_DEFINITION_SHA256,
        "score": 0.6,
        "components": {
            "ood": {"value": 1.25, "percentile": 0.2},
            "leverage": {"value": 0.5, "percentile": 0.4},
            "disagreement": {"value": 0.1, "percentile": 0.6},
        },
        "jackknife_disagreement_covariance_norm": [[0.02, 0.0], [0.0, 0.0]],
        "jackknife_disagreement_covariance_px": [[200.0, 0.0], [0.0, 0.0]],
        "abstention": {"status": "not_selected", "threshold": None},
    }


def _synthetic_inputs() -> tuple[object, ...]:
    gaze_rows: list[tuple[float, float]] = []
    head_rows: list[tuple[float, float]] = []
    face_rows: list[tuple[float, float, float]] = []
    target_rows: list[tuple[float, float]] = []
    motion_blocks: list[str] = []
    sample_ids: list[str] = []
    target_ids: list[str] = []
    for block_id, (head_pitch, head_yaw, face_x, face_y, scale) in BLOCKS.items():
        for target_index, (target_x, target_y) in enumerate(TARGETS):
            gaze_yaw = (
                target_x - 0.90 * head_yaw - 0.15 * face_x
            ) / (1.10 + 0.50 * scale)
            gaze_pitch = (
                target_y - 0.70 * head_pitch - 0.12 * face_y
            ) / (1.05 + 0.45 * scale)
            gaze_rows.append((gaze_pitch, gaze_yaw))
            head_rows.append((head_pitch, head_yaw))
            face_rows.append((face_x, face_y, scale))
            target_rows.append((target_x, target_y))
            motion_blocks.append(block_id)
            sample_ids.append(f"sample-{block_id}-{target_index}")
            target_ids.append(str(target_index))
    viewports = np.tile(np.asarray([[1920.0, 1080.0]]), (len(sample_ids), 1))
    return (
        np.asarray(gaze_rows),
        np.asarray(head_rows),
        np.asarray(face_rows),
        np.asarray(target_rows),
        viewports,
        motion_blocks,
        sample_ids,
        target_ids,
    )


class FrozenDefinitionTests(unittest.TestCase):
    def test_definition_and_protocol_hashes_are_canonical(self) -> None:
        definition_document = load_frozen_definition()
        self.assertEqual(
            definition_document["definition_sha256"],
            "75a24c436e9a89024462268812ecc9be149a1958b3911e5cd71c3974b235a180",
        )
        self.assertEqual(
            definition_document["definition_sha256"],
            canonical_sha256(definition_document["definition"]),
        )
        self.assertIsNone(
            definition_document["definition"]["runtime_policy"]["threshold"]
        )
        self.assertEqual(
            definition_document["definition"]["output_status"],
            "scored_no_threshold",
        )

        protocol_path = ROOT / (
            "docs/experiments/protocols/"
            "2026-08-10-training-only-gaze-uncertainty-v2-phase1.json"
        )
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        self.assertEqual(
            protocol["canonical_sha256"],
            canonical_sha256(protocol["protocol"]),
        )
        self.assertEqual(
            protocol["canonical_sha256"],
            "a6f622277291ac6484c69606da7197ebfde625f4e1a3e60686ca7241380b42c3",
        )
        self.assertEqual(
            protocol["protocol"]["definition_sha256"],
            definition_document["definition_sha256"],
        )
        self.assertEqual(
            protocol["protocol"]["data_boundary"][
                "effective_independent_motion_clusters"
            ],
            5,
        )
        self.assertTrue(
            protocol["protocol"]["claim_boundary"][
                "fresh_matched_contract_capture_required_before_threshold_selection"
            ]
        )

    def test_any_frozen_definition_mutation_changes_hash(self) -> None:
        document = load_frozen_definition()
        changed = deepcopy(document["definition"])
        changed["coverage_risk"]["coverage_grid"] = [1.0, 0.5]
        self.assertNotEqual(
            document["definition_sha256"],
            canonical_sha256(changed),
        )


class PureStdlibRuntimeContractTests(unittest.TestCase):
    def test_contract_module_imports_only_the_standard_library(self) -> None:
        source_path = ROOT / "core/gaze_core/uncertainty_contract.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.partition(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.partition(".")[0])
        self.assertLessEqual(
            imported_roots,
            {
                "__future__",
                "collections",
                "hashlib",
                "json",
                "math",
                "pathlib",
                "re",
                "sys",
                "typing",
            },
        )

    def test_scored_golden_bytes_match_backward_reexport(self) -> None:
        raw = _golden_runtime_observation()
        direct = uncertainty_contract.normalize_uncertainty_observation(
            raw,
            viewport=(200.0, 100.0),
        )
        compatible = compatibility_uncertainty.normalize_uncertainty_observation(
            raw,
            viewport=(200.0, 100.0),
        )
        expected = (
            b'{"abstention":{"status":"not_selected","threshold":null},'
            b'"components":{"disagreement":{"percentile":0.6,"value":0.1},'
            b'"leverage":{"percentile":0.4,"value":0.5},'
            b'"ood":{"percentile":0.2,"value":1.25}},'
            b'"definition_sha256":"75a24c436e9a89024462268812ecc9be149a1958b3911e5cd'
            b'71c3974b235a180",'
            b'"jackknife_disagreement_covariance_norm":[[0.02,0.0],[0.0,0.0]],'
            b'"jackknife_disagreement_covariance_px":[[200.0,0.0],[0.0,0.0]],'
            b'"schema_version":1,"score":0.6,"status":"scored_no_threshold"}'
        )
        self.assertEqual(direct, compatible)
        self.assertEqual(uncertainty_contract.canonical_json_bytes(direct), expected)
        self.assertEqual(
            compatibility_uncertainty.canonical_json_bytes(compatible),
            expected,
        )
        self.assertIs(
            compatibility_uncertainty.normalize_uncertainty_observation,
            uncertainty_contract.normalize_uncertainty_observation,
        )

    def test_unavailable_golden_bytes_match_backward_reexport(self) -> None:
        direct = uncertainty_contract.unavailable_uncertainty(
            "unavailable_invalid_state",
            "bad\nC:\\private\\runtime\\model.json\x7f next",
        )
        compatible = compatibility_uncertainty.unavailable_uncertainty(
            "unavailable_invalid_state",
            "bad\nC:\\private\\runtime\\model.json\x7f next",
        )
        expected = (
            b'{"reason":"bad [redacted_path] next","schema_version":1,'
            b'"status":"unavailable_invalid_state"}'
        )
        self.assertEqual(direct, compatible)
        self.assertEqual(uncertainty_contract.canonical_json_bytes(direct), expected)

    def test_psd_tolerance_and_numpy_boolean_edges_fail_closed(self) -> None:
        within_tolerance = _golden_runtime_observation()
        within_tolerance["components"]["disagreement"]["value"] = math.sqrt(
            (0.02 - 5e-10) / 2.0
        )
        within_tolerance["jackknife_disagreement_covariance_norm"] = [
            [-5e-10, 0.0],
            [0.0, 0.02],
        ]
        within_tolerance["jackknife_disagreement_covariance_px"] = [
            [-5e-10, 0.0],
            [0.0, 0.02],
        ]
        uncertainty_contract.normalize_uncertainty_observation(
            within_tolerance,
            viewport=(2.0, 2.0),
        )

        below_tolerance = deepcopy(within_tolerance)
        below_tolerance["jackknife_disagreement_covariance_norm"][0][0] = -2e-9
        below_tolerance["jackknife_disagreement_covariance_px"][0][0] = -2e-9
        with self.assertRaisesRegex(ValueError, "PSD"):
            uncertainty_contract.normalize_uncertainty_observation(
                below_tolerance,
                viewport=(2.0, 2.0),
            )

        boolean_component = _golden_runtime_observation()
        boolean_component["components"]["ood"]["value"] = np.bool_(True)
        with self.assertRaisesRegex(ValueError, "must not be boolean"):
            uncertainty_contract.normalize_uncertainty_observation(
                boolean_component,
                viewport=(200.0, 100.0),
            )


class StagePipelineTests(unittest.TestCase):
    def test_polynomial_cascade_uses_production_coordinate_order(self) -> None:
        gaze = np.asarray([[0.2, -0.4], [-0.3, 0.5]])
        second_stage = {
            "stage": 2,
            "calibrator_type": "gaze_polynomial",
            "W": [[0.5, 0.0], [0.0, 0.5], [0.1, -0.1]],
            "poly_degree": 1,
            "alpha": 0.001,
        }
        observed = apply_stage_chain(
            gaze,
            [IDENTITY_STAGE, second_stage],
            clamp=True,
        )
        first = np.column_stack([gaze[:, 1], gaze[:, 0]])
        expected = first * 0.5 + np.asarray([0.1, -0.1])
        np.testing.assert_allclose(observed, expected)

    def test_degree_two_stage_matches_exact_production_design(self) -> None:
        gaze = np.asarray([[0.2, -0.4], [-0.3, 0.5]])
        weights = np.asarray(
            [
                [0.7, -0.1],
                [0.2, 0.8],
                [0.05, 0.02],
                [-0.03, 0.04],
                [0.06, -0.07],
                [0.1, -0.2],
            ]
        )
        stage = {
            "stage": 1,
            "calibrator_type": "gaze_polynomial",
            "W": weights.tolist(),
            "poly_degree": 2,
            "alpha": 0.001,
        }
        observed = apply_stage_chain(gaze, [stage], clamp=True)
        yaw = gaze[:, 1]
        pitch = gaze[:, 0]
        design = np.column_stack(
            [
                yaw,
                pitch,
                yaw * yaw,
                pitch * pitch,
                yaw * pitch,
                np.ones(len(gaze)),
            ]
        )
        np.testing.assert_allclose(observed, np.clip(design @ weights, -1.0, 1.0))

    def test_motion_stage_uses_raw_sensor_observation(self) -> None:
        gaze = np.asarray([[0.2, -0.4], [-0.3, 0.5]])
        head = np.zeros((2, 2))
        face = np.asarray([[0.5, 0.5, 0.6], [0.5, 0.5, 0.6]])
        weights = np.zeros((12, 2))
        weights[0, 0] = 1.0
        weights[1, 1] = 1.0
        stage = {
            "stage": 1,
            "calibrator_type": "motion_conditioned_ridge_v1",
            "feature_mean": [0.0] * 11,
            "feature_scale": [1.0] * 11,
            "W": weights.tolist(),
            "alpha": 0.1,
        }
        observed = apply_stage_chain(
            gaze,
            [stage],
            head_pitch_yaw=head,
            face_geometry=face,
        )
        np.testing.assert_allclose(observed, np.column_stack([gaze[:, 1], gaze[:, 0]]))


class GridContractTests(unittest.TestCase):
    def test_exact_13_by_5_grid_is_required(self) -> None:
        *_, blocks, sample_ids, target_ids = _synthetic_inputs()
        result = validate_complete_motion_grid(sample_ids, target_ids, blocks)
        self.assertEqual(result["sample_count"], 65)
        self.assertEqual(result["pair_count"], 65)

        with self.assertRaisesRegex(ValueError, "exactly 65"):
            validate_complete_motion_grid(
                sample_ids[:-1], target_ids[:-1], blocks[:-1]
            )

        duplicated_targets = list(target_ids)
        duplicated_targets[-1] = duplicated_targets[-2]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_complete_motion_grid(
                sample_ids,
                duplicated_targets,
                blocks,
            )


class OofEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = _synthetic_inputs()
        cls.bundle = build_uncertainty_v2_bundle(
            *cls.inputs,
            [IDENTITY_STAGE],
        )

    def test_outer_partitions_are_disjoint_and_every_row_is_retained(self) -> None:
        evidence = self.bundle["oof_evidence"]
        self.assertEqual(len(evidence["rows"]), 65)
        self.assertEqual(len(evidence["folds"]), 5)
        for fold in evidence["folds"]:
            proof = fold["training_partition_proof"]
            self.assertEqual(proof["sample_id_intersection_count"], 0)
            self.assertEqual(proof["train_sample_count"], 52)
            self.assertEqual(proof["holdout_sample_count"], 13)
            self.assertEqual(
                fold["score_state_sha256"],
                fold["score_state"]["state_sha256"],
            )
            self.assertEqual(len(fold["selected_stages"]), 1)
            state = fold["score_state"]
            self.assertEqual(
                state["jackknife_member_holdout_motion_block_ids"],
                sorted(proof["train_motion_block_ids"]),
            )
            self.assertEqual(len(state["jackknife_member_proofs"]), 4)
            for member in state["jackknife_member_proofs"]:
                self.assertEqual(member["fit_sample_count"], 39)
                self.assertEqual(member["held_out_sample_count"], 13)
        for row in evidence["rows"]:
            self.assertEqual(row["training_partition_proof"]["sample_id_intersection_count"], 0)
            self.assertEqual(row["threshold"], None)
            self.assertEqual(row["abstention_status"], "not_selected")
            self.assertEqual(
                set(row["would_abstain_at_fixed_coverage"]),
                {"1.0", "0.8", "0.6", "0.4", "0.2"},
            )

    def test_score_is_label_free_and_threshold_remains_unselected(self) -> None:
        self.assertNotIn("target", inspect.signature(score_samples).parameters)
        self.assertNotIn("residual", inspect.signature(score_samples).parameters)
        self.assertEqual(self.bundle["status"], "scored_no_threshold")
        self.assertIsNone(self.bundle["threshold"])
        self.assertIsNone(self.bundle["abstention_policy"]["threshold"])
        self.assertEqual(
            self.bundle["oof_evidence"]["coverage_grid"],
            [1.0, 0.8, 0.6, 0.4, 0.2],
        )
        self.assertFalse(
            self.bundle["oof_evidence"]["negative_control"][
                "used_for_score_or_threshold_selection"
            ]
        )
        final_state = self.bundle["final_score_state"]
        self.assertEqual(
            final_state["jackknife_member_holdout_motion_block_ids"],
            ["neutral", "left", "right", "near", "far"],
        )
        self.assertEqual(len(final_state["jackknife_member_proofs"]), 5)

    def test_zero_coverage_blocks_are_explicit_not_silently_dropped(self) -> None:
        rows = []
        for block_index, block_id in enumerate(BLOCKS):
            for target_index in range(13):
                rows.append(
                    {
                        "sample_id": f"{block_id}-{target_index}",
                        "outer_holdout_group_id": block_id,
                        "uncertainty_score": float(block_index * 13 + target_index),
                        "spatial_error_px": float(target_index + 1),
                    }
                )
        evidence = build_fixed_coverage_risk(rows)
        lowest_coverage = evidence["coverage_risk"][-1]
        self.assertEqual(
            set(lowest_coverage["motion_blocks_with_zero_coverage"]),
            {"left", "right", "near", "far"},
        )
        self.assertIsNone(lowest_coverage["motion_block_macro_all_blocks"])
        self.assertIsNone(lowest_coverage["worst_motion_block_mean_px"])
        self.assertEqual(
            lowest_coverage["motion_block_macro_nonempty_blocks"]["availability"],
            "descriptive_nonempty_blocks_only",
        )

    def test_heldout_target_mutation_changes_residual_not_prediction_or_score(self) -> None:
        mutated = list(self.inputs)
        mutated_targets = np.array(mutated[3], copy=True)
        neutral_indices = [
            index for index, block in enumerate(mutated[5]) if block == "neutral"
        ]
        mutated_targets[neutral_indices, 0] += 0.05
        mutated[3] = mutated_targets
        changed_bundle = build_uncertainty_v2_bundle(
            *mutated,
            [IDENTITY_STAGE],
        )
        original_rows = {
            row["sample_id"]: row
            for row in self.bundle["oof_evidence"]["rows"]
            if row["outer_holdout_group_id"] == "neutral"
        }
        changed_rows = {
            row["sample_id"]: row
            for row in changed_bundle["oof_evidence"]["rows"]
            if row["outer_holdout_group_id"] == "neutral"
        }
        residual_changed = False
        for sample_id, original in original_rows.items():
            changed = changed_rows[sample_id]
            self.assertEqual(
                (original["oof_prediction_x_norm"], original["oof_prediction_y_norm"]),
                (changed["oof_prediction_x_norm"], changed["oof_prediction_y_norm"]),
            )
            self.assertEqual(original["uncertainty_score"], changed["uncertainty_score"])
            self.assertEqual(original["selected_model"], changed["selected_model"])
            residual_changed = residual_changed or (
                original["spatial_error_px"] != changed["spatial_error_px"]
            )
        self.assertTrue(residual_changed)

    def test_viewport_only_scales_disagreement_covariance_units(self) -> None:
        gaze, head, face, _, _, blocks, _, _ = self.inputs
        state = self.bundle["final_score_state"]
        first = score_samples(
            gaze[:1],
            [IDENTITY_STAGE],
            state,
            [[1920.0, 1080.0]],
            head_pitch_yaw=head[:1],
            face_geometry=face[:1],
        )
        doubled = score_samples(
            gaze[:1],
            [IDENTITY_STAGE],
            state,
            [[3840.0, 2160.0]],
            head_pitch_yaw=head[:1],
            face_geometry=face[:1],
        )
        self.assertEqual(len(set(blocks)), 5)
        np.testing.assert_allclose(
            doubled["jackknife_disagreement_covariance_px"],
            first["jackknife_disagreement_covariance_px"] * 4.0,
        )
        np.testing.assert_allclose(
            doubled["uncertainty_score"], first["uncertainty_score"]
        )

    def test_stable_runtime_observation_is_finite_bounded_and_psd(self) -> None:
        gaze, head, face, _, _, _, _, _ = self.inputs
        scored = score_samples(
            gaze[:1],
            [IDENTITY_STAGE],
            self.bundle["final_score_state"],
            [[1920.0, 1080.0]],
            head_pitch_yaw=head[:1],
            face_geometry=face[:1],
        )
        observation = normalize_uncertainty_observation(
            {
                "schema_version": 1,
                "status": scored["status"],
                "definition_sha256": scored["definition_sha256"],
                "score": scored["uncertainty_score"][0],
                "components": {
                    name: {
                        "value": values[0],
                        "percentile": scored["component_percentiles"][name][0],
                    }
                    for name, values in scored["components"].items()
                },
                "jackknife_disagreement_covariance_norm": scored[
                    "jackknife_disagreement_covariance_norm"
                ][0],
                "jackknife_disagreement_covariance_px": scored[
                    "jackknife_disagreement_covariance_px"
                ][0],
                "abstention": {"status": "not_selected", "threshold": None},
            },
            viewport=(1920.0, 1080.0),
        )
        json.dumps(observation, allow_nan=False)
        self.assertGreaterEqual(observation["score"], 0.0)
        self.assertLessEqual(observation["score"], 1.0)
        for component in observation["components"].values():
            self.assertGreaterEqual(component["value"], 0.0)
            self.assertGreaterEqual(component["percentile"], 0.0)
            self.assertLessEqual(component["percentile"], 1.0)
        for field in (
            "jackknife_disagreement_covariance_norm",
            "jackknife_disagreement_covariance_px",
        ):
            covariance = np.asarray(observation[field])
            self.assertEqual(covariance.shape, (2, 2))
            np.testing.assert_allclose(covariance, covariance.T, atol=1e-9, rtol=0)
            self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(covariance))), -1e-9)

    def test_runtime_validator_rejects_non_psd_or_label_arguments(self) -> None:
        self.assertEqual(
            set(inspect.signature(normalize_uncertainty_observation).parameters),
            {"observation", "viewport"},
        )
        definition_sha = load_frozen_definition()["definition_sha256"]
        bad = {
            "schema_version": 1,
            "status": "scored_no_threshold",
            "definition_sha256": definition_sha,
            "score": 0.5,
            "components": {
                name: {"value": 0.1, "percentile": 0.5}
                for name in ("ood", "leverage", "disagreement")
            },
            "jackknife_disagreement_covariance_norm": [[1.0, 2.0], [2.0, 1.0]],
            "jackknife_disagreement_covariance_px": [[1.0, 0.0], [0.0, 1.0]],
            "abstention": {"status": "not_selected", "threshold": None},
        }
        with self.assertRaisesRegex(ValueError, "PSD"):
            normalize_uncertainty_observation(
                bad,
                viewport=(1920.0, 1080.0),
            )

        boolean_score = deepcopy(bad)
        boolean_score["score"] = True
        boolean_score["jackknife_disagreement_covariance_norm"] = [
            [0.01, 0.0],
            [0.0, 0.01],
        ]
        boolean_score["jackknife_disagreement_covariance_px"] = [
            [9216.0, 0.0],
            [0.0, 2916.0],
        ]
        boolean_score["components"]["disagreement"]["value"] = 0.1
        with self.assertRaisesRegex(ValueError, "must not be boolean"):
            normalize_uncertainty_observation(
                boolean_score,
                viewport=(1920.0, 1080.0),
            )

    def test_runtime_validator_binds_definition_score_and_viewport(self) -> None:
        gaze, head, face, _, _, _, _, _ = self.inputs
        scored = score_samples(
            gaze[:1],
            [IDENTITY_STAGE],
            self.bundle["final_score_state"],
            [[1920.0, 1080.0]],
            head_pitch_yaw=head[:1],
            face_geometry=face[:1],
        )
        raw = {
            "schema_version": 1,
            "status": scored["status"],
            "definition_sha256": scored["definition_sha256"],
            "score": scored["uncertainty_score"][0],
            "components": {
                name: {
                    "value": values[0],
                    "percentile": scored["component_percentiles"][name][0],
                }
                for name, values in scored["components"].items()
            },
            "jackknife_disagreement_covariance_norm": scored[
                "jackknife_disagreement_covariance_norm"
            ][0],
            "jackknife_disagreement_covariance_px": scored[
                "jackknife_disagreement_covariance_px"
            ][0],
            "abstention": {"status": "not_selected", "threshold": None},
        }

        wrong_definition = deepcopy(raw)
        wrong_definition["definition_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "does not match frozen"):
            normalize_uncertainty_observation(
                wrong_definition,
                viewport=(1920.0, 1080.0),
            )

        wrong_score = deepcopy(raw)
        wrong_score["score"] = 1.0 - float(raw["score"])
        if np.isclose(wrong_score["score"], raw["score"]):
            wrong_score["score"] = 0.123456789
        with self.assertRaisesRegex(ValueError, "maximum component percentile"):
            normalize_uncertainty_observation(
                wrong_score,
                viewport=(1920.0, 1080.0),
            )

        wrong_disagreement = deepcopy(raw)
        wrong_disagreement["components"]["disagreement"]["value"] += 0.1
        with self.assertRaisesRegex(ValueError, "sqrt"):
            normalize_uncertainty_observation(
                wrong_disagreement,
                viewport=(1920.0, 1080.0),
            )

        wrong_pixels = deepcopy(raw)
        wrong_pixels["jackknife_disagreement_covariance_px"] = np.asarray(
            wrong_pixels["jackknife_disagreement_covariance_px"]
        ).copy()
        wrong_pixels["jackknife_disagreement_covariance_px"][0, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "viewport transform"):
            normalize_uncertainty_observation(
                wrong_pixels,
                viewport=(1920.0, 1080.0),
            )

    def test_unavailable_observation_is_bounded_and_sanitized(self) -> None:
        observation = unavailable_uncertainty(
            "unavailable_invalid_state",
            "bad\nC:\\private\\runtime\\model.json " + "x" * 400,
        )
        self.assertEqual(observation["schema_version"], 1)
        self.assertLessEqual(len(observation["reason"]), 240)
        self.assertNotIn("C:\\private", observation["reason"])
        self.assertEqual(
            normalize_uncertainty_observation(observation),
            observation,
        )
        with self.assertRaisesRegex(ValueError, "allowlisted"):
            unavailable_uncertainty("unavailable_arbitrary_server_error", "bad")

    def test_legacy_models_fail_closed_without_fabricated_confidence(self) -> None:
        status = uncertainty_status_for_model({"stages": [IDENTITY_STAGE]})
        self.assertEqual(status["status"], "unavailable_legacy_stage_state")


class TrainingSourceContractTests(unittest.TestCase):
    def test_training_persists_bundle_for_shared_runtime_inference(self) -> None:
        training_source = (ROOT / "core/gaze_core/training.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("build_uncertainty_v2_bundle", training_source)
        self.assertIn("validate_complete_motion_grid", training_source)
        self.assertIn('calibration_data["uncertainty_v2"]', training_source)

    def test_inference_uses_shared_stage_pipeline_without_inline_duplicate(self) -> None:
        inference_source = (ROOT / "core/gaze_core/inference.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("from .stage_pipeline import apply_stage_chain", inference_source)
        self.assertIn("pred_xy = apply_stage_chain(", inference_source)
        self.assertNotIn("for stage_meta in stages", inference_source)
        self.assertNotIn("feat @ W_stage", inference_source)
        self.assertNotIn("motion_conditioned_features", inference_source)

    def test_runtime_contract_is_score_only_and_fail_soft(self) -> None:
        inference_source = (ROOT / "core/gaze_core/inference.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"unavailable_viewport_missing"', inference_source)
        self.assertIn('"unavailable_invalid_state"', inference_source)
        self.assertIn('"status": scored["status"]', inference_source)
        self.assertIn('"threshold": None', inference_source)
        self.assertNotIn('"confidence":', inference_source)
        self.assertNotIn('"accepted":', inference_source)
        self.assertNotIn('"quality_band":', inference_source)


if __name__ == "__main__":
    unittest.main()

"""CPU-only contracts for the subject-heldout gaze-diversity experiment."""

from __future__ import annotations

import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.gaze_diversity.data import (
    SubjectSplit,
    build_nested_subject_splits,
    fit_pose_standardization,
    gaze_vector_to_angles,
    parse_sample_list,
    permute_targets_within_subjects,
    pose_rotation_vector_to_angles,
    standardize_pose,
)
from scripts.gaze_diversity.metrics import (
    aggregate_experiment_results,
    angles_to_unit_vectors,
    angular_errors_degrees,
    constant_train_mean_prediction,
    fit_pose_only_ridge,
    paired_participant_bootstrap,
    predict_pose_only_ridge,
)
from scripts.run_subject_holdout_gaze_diversity import (
    _completed_job_seconds,
    _deny_network,
    _evaluate_gates,
    _load_or_run_job,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT
    / "docs"
    / "experiments"
    / "protocols"
    / "2026-08-08-subject-heldout-gaze-diversity-v1.json"
)


def _job(subject: str, seed: int, value: float) -> dict[str, object]:
    return {
        "test_subject": subject,
        "seed": seed,
        "test": {"mean_degrees": value},
    }


class SubjectHeldoutDataContractTests(unittest.TestCase):
    def test_nested_schedule_is_disjoint_and_tests_every_subject_once(self) -> None:
        subjects = tuple(f"p{index:02d}" for index in range(15))
        splits = build_nested_subject_splits(subjects)

        self.assertEqual([split.test_subject for split in splits], list(subjects))
        for index, split in enumerate(splits):
            self.assertEqual(split.validation_subject, subjects[(index + 1) % 15])
            self.assertEqual(len(split.train_subjects), 13)
            roles = (
                set(split.train_subjects),
                {split.validation_subject},
                {split.test_subject},
            )
            self.assertFalse(roles[0] & roles[1])
            self.assertFalse(roles[0] & roles[2])
            self.assertFalse(roles[1] & roles[2])
            self.assertEqual(set.union(*roles), set(subjects))

    def test_sample_list_parser_preserves_official_rows_and_normalizes_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-gaze-list-") as name:
            path = Path(name) / "p00.txt"
            path.write_text(
                "day01/0001.jpg left\nday02\\0002.jpg right\n",
                encoding="utf-8",
            )
            references = parse_sample_list(path, "p00")
            self.assertEqual(
                [reference.identity for reference in references],
                ["p00/day01/0001.jpg:left", "p00/day02/0002.jpg:right"],
            )
            path.write_text(
                "day01/0001.jpg left\nday01/0001.jpg left\n",
                encoding="utf-8",
            )
            repeated = parse_sample_list(path, "p00")
            self.assertEqual(len(repeated), 2)
            self.assertEqual(repeated[0], repeated[1])

    def test_gaze_angle_conversion_round_trips_unit_vectors(self) -> None:
        angles = np.array(
            [[0.0, 0.0], [0.2, -0.4], [-0.3, 0.25]],
            dtype=np.float64,
        )
        vectors = angles_to_unit_vectors(angles)
        recovered = gaze_vector_to_angles(vectors)

        np.testing.assert_allclose(recovered, angles, atol=1e-7)
        np.testing.assert_allclose(
            angular_errors_degrees(recovered, angles),
            np.zeros(len(angles)),
            atol=1e-6,
        )

    def test_pose_conversion_matches_dataset_readme_definition(self) -> None:
        zero = pose_rotation_vector_to_angles(np.zeros((1, 3)))
        pitch_rotation = pose_rotation_vector_to_angles(np.array([[0.2, 0.0, 0.0]]))

        np.testing.assert_allclose(zero, [[0.0, 0.0]], atol=1e-8)
        np.testing.assert_allclose(pitch_rotation, [[-0.2, 0.0]], atol=1e-7)

    def test_pose_standardization_uses_only_supplied_training_rows(self) -> None:
        training = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
        held_out = np.array([[1000.0, -1000.0]])
        mean, scale = fit_pose_standardization(training)
        transformed_training = standardize_pose(training, mean, scale)

        np.testing.assert_allclose(mean, [3.0, 14.0])
        np.testing.assert_allclose(transformed_training.mean(axis=0), 0.0, atol=1e-7)
        self.assertGreater(abs(standardize_pose(held_out, mean, scale)[0, 0]), 100)

    def test_sentinel_permutation_preserves_each_subject_distribution(self) -> None:
        targets = np.arange(24, dtype=np.float32).reshape(12, 2)
        subject_ids = np.repeat([4, 9], 6)
        permuted = permute_targets_within_subjects(targets, subject_ids, seed=20260811)

        self.assertFalse(np.array_equal(permuted, targets))
        for subject_id in (4, 9):
            selected = subject_ids == subject_id
            original_rows = sorted(map(tuple, targets[selected].tolist()))
            permuted_rows = sorted(map(tuple, permuted[selected].tolist()))
            self.assertEqual(permuted_rows, original_rows)


class SubjectHeldoutMetricContractTests(unittest.TestCase):
    def test_fold_job_is_atomic_resumable_and_integrity_bound(self) -> None:
        class FakeMonitor:
            def __init__(self) -> None:
                self.snapshots: list[dict[str, float | str]] = []

            def checkpoint(self, *, process_memory_bytes: int, label: str) -> None:
                self.snapshots.append(
                    {
                        "label": label,
                        "temperature_celsius": 60.0,
                        "utilization_percent": 10.0,
                        "gpu_memory_used_mib": process_memory_bytes / 1024**2,
                    }
                )

        def fake_train_one_fold(**kwargs: object) -> dict[str, object]:
            callback = kwargs["resource_checkpoint"]
            assert callable(callback)
            callback(1024**2, "fake-epoch")
            return {
                "parameter_count": 81570,
                "epochs_completed": 1,
                "best_epoch": 1,
                "best_validation_mean_degrees": 5.0,
                "history": [{"epoch": 1}],
                "test_predictions": np.zeros((2, 2)),
                "peak_process_memory_bytes": 1024**2,
            }

        split = SubjectSplit(
            fold=0,
            train_subjects=("p02",),
            validation_subject="p01",
            test_subject="p00",
        )
        prepared = {
            "train_images": np.zeros((2, 1, 36, 60), dtype=np.uint8),
            "train_poses": np.zeros((2, 2), dtype=np.float32),
            "train_targets": np.zeros((2, 2), dtype=np.float32),
            "validation_images": np.zeros((2, 1, 36, 60), dtype=np.uint8),
            "validation_poses": np.zeros((2, 2), dtype=np.float32),
            "validation_targets": np.zeros((2, 2), dtype=np.float32),
            "test_images": np.zeros((2, 1, 36, 60), dtype=np.uint8),
            "test_poses": np.zeros((2, 2), dtype=np.float32),
            "test_targets": np.zeros((2, 2), dtype=np.float32),
            "test_days": np.array(["day01", "day01"]),
        }
        integrity = {
            "protocol_sha256": "protocol",
            "implementation_sha256": "implementation",
            "data_sha256": "data",
        }
        monitor = FakeMonitor()
        with tempfile.TemporaryDirectory(prefix="lexigaze-gaze-job-") as name:
            work_dir = Path(name)
            with mock.patch(
                "scripts.gaze_diversity.train.train_one_fold",
                side_effect=fake_train_one_fold,
            ) as patched:
                first = _load_or_run_job(
                    kind="candidate",
                    seed=20260808,
                    split=split,
                    prepared=prepared,
                    training_targets=prepared["train_targets"],
                    train_config={},
                    monitor=monitor,
                    work_dir=work_dir,
                    integrity=integrity,
                )
                resumed = _load_or_run_job(
                    kind="candidate",
                    seed=20260808,
                    split=split,
                    prepared=prepared,
                    training_targets=prepared["train_targets"],
                    train_config={},
                    monitor=monitor,
                    work_dir=work_dir,
                    integrity=integrity,
                )

            self.assertEqual(patched.call_count, 1)
            self.assertEqual(first, resumed)
            self.assertGreaterEqual(
                _completed_job_seconds(work_dir, **integrity),
                0.0,
            )

    def test_decision_gates_require_evidence_and_unchanged_production(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        aggregate = {
            "candidate": {
                "macro_mean_degrees": 5.0,
                "seed_macro_standard_deviation_degrees": 0.2,
                "worst_subject_mean_degrees": 8.0,
            },
            "pose_only": {"macro_mean_degrees": 7.0},
            "shuffled_label_sentinel": {"macro_mean_degrees": 9.0},
            "candidate_minus_pose_only": {"ci95_upper_degrees": -1.0},
            "subjects_candidate_beats_pose_only": 15,
        }
        resources = {
            "peak_process_memory_gib": 5.0,
            "peak_temperature_celsius": 70.0,
            "model_execution_hours": 1.0,
        }
        gates = _evaluate_gates(
            protocol,
            {"status": "passed", "split_overlap_count": 0},
            aggregate,
            resources,
            production_unchanged=True,
        )
        changed_gates = _evaluate_gates(
            protocol,
            {"status": "passed", "split_overlap_count": 0},
            aggregate,
            resources,
            production_unchanged=False,
        )

        self.assertTrue(all(gates.values()))
        self.assertFalse(changed_gates["production_model_unchanged"])

    def test_formal_run_network_guard_blocks_name_resolution(self) -> None:
        attempts: list[str] = []
        with (
            _deny_network(attempts),
            self.assertRaisesRegex(RuntimeError, "network disabled"),
        ):
            socket.getaddrinfo("example.invalid", 443)

        self.assertEqual(attempts, ["'example.invalid'"])

    def test_frozen_baselines_produce_expected_shapes(self) -> None:
        poses = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0]])
        targets = poses * np.array([0.2, 0.1])
        coefficients = fit_pose_only_ridge(poses, targets, alpha=0.0)
        prediction = predict_pose_only_ridge(poses, coefficients)
        constant = constant_train_mean_prediction(targets, row_count=5)

        np.testing.assert_allclose(prediction, targets, atol=1e-10)
        self.assertEqual(coefficients.shape, (3, 2))
        self.assertEqual(constant.shape, (5, 2))

    def test_participant_bootstrap_is_seed_deterministic(self) -> None:
        differences = np.array([-1.0, -0.5, 0.25, -0.75])
        first = paired_participant_bootstrap(differences, resamples=1000, seed=20260812)
        second = paired_participant_bootstrap(
            differences, resamples=1000, seed=20260812
        )

        self.assertEqual(first, second)
        self.assertLess(first["mean_difference_degrees"], 0.0)

    def test_aggregate_requires_exact_frozen_seed_schedule(self) -> None:
        subjects = ("p00", "p01")
        seeds = (10, 11, 12)
        baselines = {
            subject: {
                "pose_only": {"mean_degrees": 7.0},
                "constant_train_mean": {"mean_degrees": 10.0},
            }
            for subject in subjects
        }
        candidates = [
            _job(subject, seed, 6.0) for subject in subjects for seed in seeds
        ]
        sentinels = [_job(subject, 13, 12.0) for subject in subjects]

        result = aggregate_experiment_results(
            subjects=subjects,
            expected_candidate_seeds=seeds,
            expected_sentinel_seed=13,
            candidate_jobs=candidates,
            sentinel_jobs=sentinels,
            baseline_by_subject=baselines,
            bootstrap_resamples=100,
            bootstrap_seed=14,
        )
        self.assertEqual(result["subjects_candidate_beats_pose_only"], 2)
        with self.assertRaisesRegex(ValueError, "seed schedule"):
            aggregate_experiment_results(
                subjects=subjects,
                expected_candidate_seeds=seeds,
                expected_sentinel_seed=13,
                candidate_jobs=candidates[:-1],
                sentinel_jobs=sentinels,
                baseline_by_subject=baselines,
                bootstrap_resamples=100,
                bootstrap_seed=14,
            )

    def test_protocol_excludes_private_and_question_answer_outcomes(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        excluded = set(protocol["data"]["excluded_inputs"])
        leakage = protocol["leakage_controls"]

        self.assertEqual(protocol["data"]["expected_total_samples"], 45000)
        self.assertEqual(len(protocol["splits"]["candidate_seeds"]), 3)
        self.assertIn("question-answer datasets", excluded)
        self.assertIn("LexiGaze participant-study data", excluded)
        self.assertFalse(leakage["question_answer_dataset_used"])
        self.assertFalse(leakage["lexigaze_participant_data_used"])
        self.assertFalse(leakage["post_result_parameter_tuning_allowed"])


if __name__ == "__main__":
    unittest.main()

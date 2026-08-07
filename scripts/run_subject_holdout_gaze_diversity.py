"""Run the frozen MPIIGaze nested subject-heldout diversity experiment."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.gaze_diversity.data import (
    EvaluationData,
    SubjectSplit,
    build_nested_subject_splits,
    fit_pose_standardization,
    load_evaluation_data,
    permute_targets_within_subjects,
    standardize_pose,
)
from scripts.gaze_diversity.metrics import (
    aggregate_experiment_results,
    angular_errors_degrees,
    constant_train_mean_prediction,
    fit_pose_only_ridge,
    predict_pose_only_ridge,
    summarize_days,
    summarize_errors,
)
from scripts.gaze_diversity.resources import ResourceMonitor

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "subject-heldout-gaze-diversity-v1"
DEFAULT_PROTOCOL = (
    ROOT
    / "docs"
    / "experiments"
    / "protocols"
    / "2026-08-08-subject-heldout-gaze-diversity-v1.json"
)
DEFAULT_DATASET_ROOT = ROOT / "data" / "eai" / "MPIIGaze"
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "experiments" / "results"
DEFAULT_WORK_DIR = ROOT / "output" / "subject-heldout-gaze-diversity-v1-run-001"
IMPLEMENTATION_PATHS = (
    ROOT / "scripts" / "gaze_diversity" / "data.py",
    ROOT / "scripts" / "gaze_diversity" / "metrics.py",
    ROOT / "scripts" / "gaze_diversity" / "model.py",
    ROOT / "scripts" / "gaze_diversity" / "resources.py",
    ROOT / "scripts" / "gaze_diversity" / "train.py",
    Path(__file__).resolve(),
)
PRODUCTION_ROOTS = (
    ROOT / "core" / "gaze_core",
    ROOT / "core" / "unigaze_personalization",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate source data and splits without importing Torch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    _validate_protocol(protocol)
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    implementation_sha256 = _combined_sha256(IMPLEMENTATION_PATHS)
    protocol_commit = _protocol_commit(protocol_path)
    production_sha256_before = _production_sha256()

    subjects = tuple(str(value) for value in protocol["data"]["subjects"])
    data, data_audit = load_evaluation_data(
        args.dataset_root,
        subjects,
        expected_samples_per_subject=int(protocol["data"]["samples_per_subject"]),
    )
    audit_envelope = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_sha256": protocol_sha256,
        "protocol_commit": protocol_commit,
        "data_audit": data_audit,
        "model_runtime_imported": "torch" in sys.modules,
    }
    audit_path = (
        args.output_dir / "2026-08-08-subject-heldout-gaze-diversity-v1-data-audit.json"
    )
    _atomic_json_write(audit_path, audit_envelope)
    print(
        f"DATA_AUDIT status={data_audit['status']} "
        f"subjects={data_audit['subject_count']} "
        f"rows={data_audit['total_samples']} "
        f"overlap={data_audit['split_overlap_count']} "
        f"sha256={data_audit['source_sha256']}",
        flush=True,
    )
    if args.audit_only:
        print(f"AUDIT_JSON={audit_path}")
        return 0

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    compute = protocol["compute"]
    candidate = protocol["candidate"]
    training = candidate["training"]
    optimizer = candidate["optimizer"]
    loss = candidate["loss"]
    train_config: dict[str, Any] = {
        "cpu_threads": int(compute["cpu_threads"]),
        "memory_fraction": float(compute["per_process_memory_fraction"]),
        "learning_rate": float(optimizer["learning_rate"]),
        "weight_decay": float(optimizer["weight_decay"]),
        "loss_beta_radians": float(loss["beta_radians"]),
        "batch_size": int(training["batch_size"]),
        "max_epochs": int(training["max_epochs"]),
        "early_stopping_patience": int(training["early_stopping_patience"]),
        "early_stopping_min_delta_degrees": float(
            training["early_stopping_min_delta_degrees"]
        ),
        "gradient_norm_clip": float(training["gradient_norm_clip"]),
    }
    completed_seconds = _completed_job_seconds(
        args.work_dir,
        protocol_sha256=protocol_sha256,
        implementation_sha256=implementation_sha256,
        data_sha256=str(data_audit["source_sha256"]),
    )
    monitor = ResourceMonitor(
        maximum_temperature_celsius=float(compute["maximum_temperature_celsius"]),
        maximum_wall_time_hours=float(compute["maximum_wall_time_hours"]),
        started_monotonic=time.monotonic() - completed_seconds,
    )
    monitor.checkpoint(process_memory_bytes=0, label="formal-run-start")
    candidate_jobs: list[dict[str, Any]] = []
    sentinel_jobs: list[dict[str, Any]] = []
    baseline_by_subject: dict[str, dict[str, Any]] = {}
    splits = build_nested_subject_splits(subjects)
    network_attempts: list[str] = []
    torch_was_loaded = "torch" in sys.modules

    with _deny_network(network_attempts):
        for split in splits:
            prepared = _prepare_fold(data, split)
            baseline_by_subject[split.test_subject] = _evaluate_baselines(
                prepared,
                ridge_alpha=float(protocol["baselines"]["pose_only_ridge"]["alpha"]),
            )
            for seed in protocol["splits"]["candidate_seeds"]:
                job = _load_or_run_job(
                    kind="candidate",
                    seed=int(seed),
                    split=split,
                    prepared=prepared,
                    training_targets=prepared["train_targets"],
                    train_config=train_config,
                    monitor=monitor,
                    work_dir=args.work_dir,
                    integrity={
                        "protocol_sha256": protocol_sha256,
                        "implementation_sha256": implementation_sha256,
                        "data_sha256": str(data_audit["source_sha256"]),
                    },
                )
                candidate_jobs.append(job)

            sentinel_targets = permute_targets_within_subjects(
                prepared["train_targets"],
                prepared["train_subject_ids"],
                seed=int(protocol["splits"]["shuffled_label_seed"]),
            )
            sentinel_jobs.append(
                _load_or_run_job(
                    kind="shuffled_label_sentinel",
                    seed=int(protocol["splits"]["shuffled_label_seed"]),
                    split=split,
                    prepared=prepared,
                    training_targets=sentinel_targets,
                    train_config=train_config,
                    monitor=monitor,
                    work_dir=args.work_dir,
                    integrity={
                        "protocol_sha256": protocol_sha256,
                        "implementation_sha256": implementation_sha256,
                        "data_sha256": str(data_audit["source_sha256"]),
                    },
                )
            )

    monitor.checkpoint(
        process_memory_bytes=monitor.peak_process_memory_bytes,
        label="formal-run-complete",
    )
    aggregate = aggregate_experiment_results(
        subjects=subjects,
        expected_candidate_seeds=tuple(
            int(seed) for seed in protocol["splits"]["candidate_seeds"]
        ),
        expected_sentinel_seed=int(protocol["splits"]["shuffled_label_seed"]),
        candidate_jobs=candidate_jobs,
        sentinel_jobs=sentinel_jobs,
        baseline_by_subject=baseline_by_subject,
        bootstrap_resamples=int(protocol["metrics"]["participant_bootstrap_resamples"]),
        bootstrap_seed=int(protocol["metrics"]["participant_bootstrap_seed"]),
    )
    resources = _aggregate_resource_results(
        monitor=monitor,
        candidate_jobs=candidate_jobs,
        sentinel_jobs=sentinel_jobs,
    )
    production_sha256_after = _production_sha256()
    production_unchanged = production_sha256_before == production_sha256_after
    gates = _evaluate_gates(
        protocol,
        data_audit,
        aggregate,
        resources,
        production_unchanged=production_unchanged,
    )
    passed = all(gates.values())
    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "subject-heldout-gaze-diversity-v1",
        "run_id": str(args.run_id),
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_sha256,
        "protocol_commit": protocol_commit,
        "implementation_sha256": implementation_sha256,
        "production_sha256_before": production_sha256_before,
        "production_sha256_after": production_sha256_after,
        "data_audit": data_audit,
        "configuration": {
            "candidate": candidate,
            "splits": protocol["splits"],
            "baselines": protocol["baselines"],
        },
        "baseline_by_subject": baseline_by_subject,
        "candidate_jobs": candidate_jobs,
        "sentinel_jobs": sentinel_jobs,
        "aggregate": aggregate,
        "resources": resources,
        "network_attempts": network_attempts,
        "compute": {
            "device": "cuda:0",
            "torch_imported_before_training": torch_was_loaded,
            "torch_imported_after_training": "torch" in sys.modules,
            "tf32_allowed": False,
            "automatic_mixed_precision": False,
            "network_used": bool(network_attempts),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "leakage_controls": protocol["leakage_controls"],
        "gates": {**gates, "passed": passed},
        "decision": {
            "status": "passed" if passed else "failed",
            "action": (
                protocol["decision_gate"]["pass_action"]
                if passed
                else protocol["decision_gate"]["fail_action"]
            ),
            "production_model_changed": not production_unchanged,
            "real_webcam_generalization_claimed": False,
        },
    }
    stem = f"2026-08-08-subject-heldout-gaze-diversity-v1-{args.run_id}"
    result_path = args.output_dir / f"{stem}.json"
    report_path = args.output_dir.parent / f"{stem}.md"
    _atomic_json_write(result_path, summary)
    _atomic_text_write(report_path, _markdown_report(summary))
    print(
        "FINAL_RESULT "
        f"passed={passed} "
        f"candidate={aggregate['candidate']['macro_mean_degrees']:.4f} "
        f"pose={aggregate['pose_only']['macro_mean_degrees']:.4f} "
        f"sentinel={aggregate['shuffled_label_sentinel']['macro_mean_degrees']:.4f}",
        flush=True,
    )
    print(f"RESULT_JSON={result_path}")
    print(f"REPORT={report_path}")
    return 0


def _validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected experiment protocol")
    if protocol.get("status") != "frozen_before_model_execution":
        raise ValueError("experiment protocol is not frozen")
    data = protocol["data"]
    splits = protocol["splits"]
    compute = protocol["compute"]
    leakage = protocol["leakage_controls"]
    if len(data["subjects"]) != 15 or int(data["expected_total_samples"]) != 45000:
        raise ValueError("frozen MPIIGaze subject/sample contract changed")
    if len(splits["candidate_seeds"]) != 3:
        raise ValueError("frozen candidate seed schedule changed")
    if compute["device"] != "cuda:0" or not compute["cuda_required"]:
        raise ValueError("frozen CUDA contract changed")
    if compute["network_allowed_during_run"]:
        raise ValueError("network must remain disabled during the run")
    forbidden_true = (
        leakage["question_answer_dataset_used"],
        leakage["lexigaze_participant_data_used"],
        leakage["legacy_checkpoint_used"],
        leakage["production_unigaze_checkpoint_used"],
        leakage["post_result_parameter_tuning_allowed"],
    )
    if any(bool(value) for value in forbidden_true):
        raise ValueError("leakage controls changed")


def _prepare_fold(data: EvaluationData, split: SubjectSplit) -> dict[str, Any]:
    train_indices = data.indices_for(split.train_subjects)
    validation_indices = data.indices_for([split.validation_subject])
    test_indices = data.indices_for([split.test_subject])
    if (
        np.intersect1d(train_indices, validation_indices).size
        or np.intersect1d(train_indices, test_indices).size
        or np.intersect1d(validation_indices, test_indices).size
    ):
        raise ValueError(f"row overlap detected in fold {split.fold}")
    pose_mean, pose_scale = fit_pose_standardization(data.poses[train_indices])
    return {
        "train_images": data.images[train_indices],
        "train_poses": standardize_pose(
            data.poses[train_indices], pose_mean, pose_scale
        ),
        "train_targets": data.targets[train_indices],
        "train_subject_ids": data.subject_ids[train_indices],
        "validation_images": data.images[validation_indices],
        "validation_poses": standardize_pose(
            data.poses[validation_indices], pose_mean, pose_scale
        ),
        "validation_targets": data.targets[validation_indices],
        "test_images": data.images[test_indices],
        "test_poses": standardize_pose(data.poses[test_indices], pose_mean, pose_scale),
        "test_targets": data.targets[test_indices],
        "test_days": data.days[test_indices],
        "pose_mean": pose_mean,
        "pose_scale": pose_scale,
    }


def _evaluate_baselines(
    prepared: dict[str, Any], *, ridge_alpha: float
) -> dict[str, Any]:
    test_targets = prepared["test_targets"]
    constant_prediction = constant_train_mean_prediction(
        prepared["train_targets"], len(test_targets)
    )
    coefficients = fit_pose_only_ridge(
        prepared["train_poses"],
        prepared["train_targets"],
        alpha=ridge_alpha,
    )
    pose_prediction = predict_pose_only_ridge(prepared["test_poses"], coefficients)
    constant_errors = angular_errors_degrees(constant_prediction, test_targets)
    pose_errors = angular_errors_degrees(pose_prediction, test_targets)
    return {
        "pose_standardization": {
            "mean": prepared["pose_mean"].tolist(),
            "scale": prepared["pose_scale"].tolist(),
        },
        "pose_only_coefficients": coefficients.tolist(),
        "constant_train_mean": {
            **summarize_errors(constant_errors),
            "days": summarize_days(
                constant_errors, prepared["test_days"], minimum_rows=20
            ),
        },
        "pose_only": {
            **summarize_errors(pose_errors),
            "days": summarize_days(pose_errors, prepared["test_days"], minimum_rows=20),
        },
    }


def _load_or_run_job(
    *,
    kind: str,
    seed: int,
    split: SubjectSplit,
    prepared: dict[str, Any],
    training_targets: np.ndarray,
    train_config: dict[str, Any],
    monitor: ResourceMonitor,
    work_dir: Path,
    integrity: dict[str, str],
) -> dict[str, Any]:
    job_label = f"{kind}-seed-{seed}-fold-{split.fold:02d}-{split.test_subject}"
    job_path = work_dir / f"{job_label}.json"
    if job_path.is_file():
        job = json.loads(job_path.read_text(encoding="utf-8"))
        if job.get("integrity") != integrity:
            raise ValueError(f"refusing incompatible resumed job: {job_path}")
        if (
            job.get("kind") != kind
            or int(job.get("seed")) != seed
            or int(job.get("fold")) != split.fold
            or job.get("test_subject") != split.test_subject
        ):
            raise ValueError(f"resumed job identity mismatch: {job_path}")
        print(f"RESUME_JOB {job_label}", flush=True)
        return job

    from scripts.gaze_diversity.train import train_one_fold

    config = {**train_config, "job_label": job_label}
    snapshot_start = len(monitor.snapshots)
    started = time.monotonic()
    train_result = train_one_fold(
        train_images=prepared["train_images"],
        train_poses=prepared["train_poses"],
        train_targets=training_targets,
        validation_images=prepared["validation_images"],
        validation_poses=prepared["validation_poses"],
        validation_targets=prepared["validation_targets"],
        test_images=prepared["test_images"],
        test_poses=prepared["test_poses"],
        seed=seed,
        config=config,
        resource_checkpoint=lambda memory, label: monitor.checkpoint(
            process_memory_bytes=memory,
            label=label,
        ),
    )
    duration_seconds = time.monotonic() - started
    prediction = train_result.pop("test_predictions")
    errors = angular_errors_degrees(prediction, prepared["test_targets"])
    job_snapshots = monitor.snapshots[snapshot_start:]
    job = {
        "kind": kind,
        "seed": seed,
        "fold": split.fold,
        "train_subjects": list(split.train_subjects),
        "validation_subject": split.validation_subject,
        "test_subject": split.test_subject,
        "parameter_count": train_result["parameter_count"],
        "epochs_completed": train_result["epochs_completed"],
        "best_epoch": train_result["best_epoch"],
        "best_validation_mean_degrees": train_result["best_validation_mean_degrees"],
        "history": train_result["history"],
        "test": {
            **summarize_errors(errors),
            "days": summarize_days(errors, prepared["test_days"], minimum_rows=20),
        },
        "resource": {
            "duration_seconds": duration_seconds,
            "peak_process_memory_bytes": train_result["peak_process_memory_bytes"],
            "peak_temperature_celsius": max(
                (float(snapshot["temperature_celsius"]) for snapshot in job_snapshots),
                default=0.0,
            ),
            "peak_utilization_percent": max(
                (float(snapshot["utilization_percent"]) for snapshot in job_snapshots),
                default=0.0,
            ),
            "peak_gpu_memory_used_mib": max(
                (float(snapshot["gpu_memory_used_mib"]) for snapshot in job_snapshots),
                default=0.0,
            ),
        },
        "integrity": integrity,
    }
    _atomic_json_write(job_path, job)
    print(
        f"JOB_RESULT {job_label} test_deg={job['test']['mean_degrees']:.4f} "
        f"best_epoch={job['best_epoch']} seconds={duration_seconds:.1f}",
        flush=True,
    )
    return job


def _aggregate_resource_results(
    *,
    monitor: ResourceMonitor,
    candidate_jobs: list[dict[str, Any]],
    sentinel_jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    jobs = candidate_jobs + sentinel_jobs
    monitor_summary = monitor.summary()
    return {
        "job_count": len(jobs),
        "model_execution_seconds": float(
            sum(float(job["resource"]["duration_seconds"]) for job in jobs)
        ),
        "model_execution_hours": float(
            sum(float(job["resource"]["duration_seconds"]) for job in jobs) / 3600.0
        ),
        "peak_process_memory_bytes": int(
            max(int(job["resource"]["peak_process_memory_bytes"]) for job in jobs)
        ),
        "peak_process_memory_gib": float(
            max(int(job["resource"]["peak_process_memory_bytes"]) for job in jobs)
            / (1024**3)
        ),
        "peak_temperature_celsius": float(
            max(float(job["resource"]["peak_temperature_celsius"]) for job in jobs)
        ),
        "peak_utilization_percent": float(
            max(float(job["resource"]["peak_utilization_percent"]) for job in jobs)
        ),
        "peak_gpu_memory_used_mib": float(
            max(float(job["resource"]["peak_gpu_memory_used_mib"]) for job in jobs)
        ),
        "monitor": monitor_summary,
    }


def _evaluate_gates(
    protocol: dict[str, Any],
    audit: dict[str, Any],
    aggregate: dict[str, Any],
    resources: dict[str, Any],
    *,
    production_unchanged: bool,
) -> dict[str, bool]:
    spec = protocol["decision_gate"]
    candidate = aggregate["candidate"]
    pose = aggregate["pose_only"]
    sentinel = aggregate["shuffled_label_sentinel"]
    bootstrap = aggregate["candidate_minus_pose_only"]
    return {
        "data_audit_passes": audit["status"] == "passed",
        "subject_overlap_count_equals_zero": (
            int(audit["split_overlap_count"])
            == int(spec["subject_overlap_count_equals"])
        ),
        "candidate_macro_mean_less_than_pose_only": (
            float(candidate["macro_mean_degrees"]) < float(pose["macro_mean_degrees"])
        ),
        "candidate_minus_pose_only_bootstrap_ci_upper_less_than_zero": (
            float(bootstrap["ci95_upper_degrees"])
            < float(
                spec[
                    "candidate_minus_pose_only_participant_bootstrap_ci_upper_less_than"
                ]
            )
        ),
        "subjects_candidate_beats_pose_only_at_least": (
            int(aggregate["subjects_candidate_beats_pose_only"])
            >= int(spec["subjects_candidate_beats_pose_only_at_least"])
        ),
        "candidate_seed_macro_standard_deviation_within_limit": (
            float(candidate["seed_macro_standard_deviation_degrees"])
            <= float(spec["candidate_seed_macro_standard_deviation_at_most_degrees"])
        ),
        "candidate_worst_subject_mean_within_limit": (
            float(candidate["worst_subject_mean_degrees"])
            <= float(spec["candidate_worst_subject_mean_at_most_degrees"])
        ),
        "shuffled_label_macro_worse_than_candidate_by_margin": (
            float(sentinel["macro_mean_degrees"])
            >= float(candidate["macro_mean_degrees"])
            + float(
                spec[
                    "shuffled_label_macro_mean_greater_than_candidate_by_at_least_degrees"
                ]
            )
        ),
        "shuffled_label_does_not_beat_pose_only": (
            float(sentinel["macro_mean_degrees"]) >= float(pose["macro_mean_degrees"])
        ),
        "gpu_peak_process_memory_within_limit": (
            float(resources["peak_process_memory_gib"])
            <= float(spec["gpu_peak_process_memory_at_most_gib"])
        ),
        "gpu_temperature_below_limit": (
            float(resources["peak_temperature_celsius"])
            < float(spec["gpu_temperature_must_remain_below_celsius"])
        ),
        "wall_time_within_limit": (
            float(resources["model_execution_hours"])
            <= float(spec["wall_time_at_most_hours"])
        ),
        "production_model_unchanged": (
            not bool(spec["production_model_may_change"]) and production_unchanged
        ),
    }


def _completed_job_seconds(
    work_dir: Path,
    *,
    protocol_sha256: str,
    implementation_sha256: str,
    data_sha256: str,
) -> float:
    if not work_dir.is_dir():
        return 0.0
    expected = {
        "protocol_sha256": protocol_sha256,
        "implementation_sha256": implementation_sha256,
        "data_sha256": data_sha256,
    }
    total = 0.0
    for path in work_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("integrity") != expected:
            raise ValueError(f"incompatible work artifact: {path}")
        total += float(payload["resource"]["duration_seconds"])
    return total


@contextlib.contextmanager
def _deny_network(attempts: list[str]) -> Iterator[None]:
    original_socket = socket.socket
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo
    original_gethostbyname = socket.gethostbyname
    original_gethostbyname_ex = socket.gethostbyname_ex

    class GuardedSocket(original_socket):
        def connect(self, address: Any) -> Any:
            attempts.append(repr(address))
            raise RuntimeError(f"network disabled by experiment: {address!r}")

        def connect_ex(self, address: Any) -> int:
            attempts.append(repr(address))
            raise RuntimeError(f"network disabled by experiment: {address!r}")

        def sendto(self, *args: Any, **kwargs: Any) -> int:
            address = args[-1] if args else kwargs.get("address", "unknown")
            attempts.append(repr(address))
            raise RuntimeError(f"network disabled by experiment: {address!r}")

    def blocked_create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
        attempts.append(repr(address))
        raise RuntimeError(f"network disabled by experiment: {address!r}")

    def blocked_lookup(*args: Any, **kwargs: Any) -> Any:
        address = args[0] if args else kwargs.get("host", "unknown")
        attempts.append(repr(address))
        raise RuntimeError(f"network disabled by experiment: {address!r}")

    socket.socket = GuardedSocket
    socket.create_connection = blocked_create_connection
    socket.getaddrinfo = blocked_lookup
    socket.gethostbyname = blocked_lookup
    socket.gethostbyname_ex = blocked_lookup
    try:
        yield
    finally:
        socket.socket = original_socket
        socket.create_connection = original_create_connection
        socket.getaddrinfo = original_getaddrinfo
        socket.gethostbyname = original_gethostbyname
        socket.gethostbyname_ex = original_gethostbyname_ex


def _markdown_report(summary: dict[str, Any]) -> str:
    aggregate = summary["aggregate"]
    candidate = aggregate["candidate"]
    pose = aggregate["pose_only"]
    constant = aggregate["constant_train_mean"]
    sentinel = aggregate["shuffled_label_sentinel"]
    lines = [
        "# Subject-Heldout Gaze Diversity v1 - Run 001",
        "",
        f"- Protocol: `{summary['protocol_id']}`",
        f"- Protocol commit: `{summary['protocol_commit']}`",
        "- Data: official MPIIGaze 15-person evaluation subset; 45,000 rows",
        f"- Official repeated rows retained: `{summary['data_audit']['duplicate_reference_rows']}`",
        "- Split: 15 outer held-out people, nested validation person, 13 training people",
        "- Production model changed: **no**",
        f"- Decision: **`{summary['decision']['status']}`**",
        "",
        "## Aggregate result",
        "",
        "| Model | Macro subject angular error (deg) |",
        "| --- | ---: |",
        f"| Constant training mean | {constant['macro_mean_degrees']:.4f} |",
        f"| Pose-only ridge | {pose['macro_mean_degrees']:.4f} |",
        f"| EyePoseTinyCNN-v1 | {candidate['macro_mean_degrees']:.4f} |",
        f"| Shuffled-label sentinel | {sentinel['macro_mean_degrees']:.4f} |",
        "",
        "## Held-out subjects",
        "",
        "| Subject | Constant | Pose-only | Candidate (3-seed mean) | Sentinel | Candidate - pose |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for subject in summary["data_audit"]["subjects"]:
        candidate_value = candidate["per_subject_mean_degrees"][subject]
        pose_value = pose["per_subject_mean_degrees"][subject]
        lines.append(
            f"| {subject} | "
            f"{constant['per_subject_mean_degrees'][subject]:.4f} | "
            f"{pose_value:.4f} | {candidate_value:.4f} | "
            f"{sentinel['per_subject_mean_degrees'][subject]:.4f} | "
            f"{candidate_value - pose_value:+.4f} |"
        )
    bootstrap = aggregate["candidate_minus_pose_only"]
    lines.extend(
        [
            "",
            "## Paired participant inference",
            "",
            f"- Candidate - pose-only macro difference: `{bootstrap['mean_difference_degrees']:+.4f}` degrees",
            f"- Participant-bootstrap 95% CI: `[{bootstrap['ci95_lower_degrees']:+.4f}, {bootstrap['ci95_upper_degrees']:+.4f}]`",
            f"- Held-out subjects improved: `{aggregate['subjects_candidate_beats_pose_only']}/15`",
            f"- Candidate seed macro SD: `{candidate['seed_macro_standard_deviation_degrees']:.4f}` degrees",
            "",
            "## Decision gates",
            "",
            *[
                f"- [{'x' if passed else ' '}] `{name}`"
                for name, passed in summary["gates"].items()
                if name != "passed"
            ],
            "",
            "## Hardware and integrity",
            "",
            f"- Model execution: `{summary['resources']['model_execution_hours']:.3f}` hours",
            f"- Peak process VRAM: `{summary['resources']['peak_process_memory_gib']:.3f}` GiB",
            f"- Peak GPU temperature: `{summary['resources']['peak_temperature_celsius']:.1f}` C",
            f"- Peak observed utilization: `{summary['resources']['peak_utilization_percent']:.1f}%`",
            f"- Network attempts: `{len(summary['network_attempts'])}`",
            f"- Protocol SHA-256: `{summary['protocol_sha256']}`",
            f"- Data SHA-256: `{summary['data_audit']['source_sha256']}`",
            f"- Implementation SHA-256: `{summary['implementation_sha256']}`",
            f"- Production SHA-256 before: `{summary['production_sha256_before']}`",
            f"- Production SHA-256 after: `{summary['production_sha256_after']}`",
            "",
            "## Interpretation boundary",
            "",
            "This experiment measures cross-person generalization on the balanced public MPIIGaze eye-image subset. It does not demonstrate improvement on LexiGaze webcam captures, does not compare independently against the production UniGaze joint checkpoint, and cannot change the production default. Any follow-up must use a newly frozen cross-dataset or real-capture protocol.",
            "",
        ]
    )
    return "\n".join(lines)


def _protocol_commit(protocol_path: Path) -> str:
    relative = protocol_path.relative_to(ROOT).as_posix()
    completed = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", relative],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip()


def _combined_sha256(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _production_sha256() -> str:
    paths = tuple(
        sorted(
            path
            for root in PRODUCTION_ROOTS
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix.lower() != ".pyc"
        )
    )
    if not paths:
        raise RuntimeError("production gaze paths are missing")
    return _combined_sha256(paths)


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text_write(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _atomic_text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())

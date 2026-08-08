"""Run the frozen Columbia cross-domain gaze v2 confirmation experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.columbia_gaze.data import SourceBundle, audit_columbia_source
from scripts.columbia_gaze.metrics import (
    fuse_angle_predictions,
    summarize_model,
    zero_gaze_summary,
)
from scripts.columbia_gaze.prepare import prepare_inputs
from scripts.columbia_gaze.train import (
    predict_candidate,
    state_dict_sha256,
    train_or_load_final_candidate,
)
from scripts.gaze_diversity.data import (
    fit_pose_standardization,
    load_evaluation_data,
    standardize_pose,
)
from scripts.gaze_diversity.metrics import angular_errors_degrees, summarize_errors
from scripts.gaze_diversity.resources import ResourceMonitor
from scripts.run_subject_holdout_gaze_diversity import (
    _atomic_json_write,
    _atomic_text_write,
    _combined_sha256,
    _deny_network,
    _production_sha256,
    _protocol_commit,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "columbia-cross-domain-gaze-v2"
BASE_PROTOCOL_ID = "columbia-cross-domain-gaze-v1"
DEFAULT_PROTOCOL = (
    ROOT
    / "docs"
    / "experiments"
    / "protocols"
    / "2026-08-08-columbia-cross-domain-gaze-v2.json"
)
DEFAULT_BASE_PROTOCOL = (
    ROOT
    / "docs"
    / "experiments"
    / "protocols"
    / "2026-08-08-columbia-cross-domain-gaze-v1.json"
)
DEFAULT_DATASET_ROOT = ROOT / "data" / "external" / "columbia_gaze"
DEFAULT_MPII_ROOT = ROOT / "data" / "eai" / "MPIIGaze"
DEFAULT_WORK_DIR = ROOT / "output" / "columbia-cross-domain-gaze-v2-run-001"
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "experiments" / "results"
LANDMARK_MODEL = ROOT / "web" / "static" / "face_landmarker.task"
IMPLEMENTATION_PATHS = (
    ROOT / "scripts" / "columbia_gaze" / "data.py",
    ROOT / "scripts" / "columbia_gaze" / "metrics.py",
    ROOT / "scripts" / "columbia_gaze" / "prepare.py",
    ROOT / "scripts" / "columbia_gaze" / "train.py",
    ROOT / "scripts" / "gaze_diversity" / "data.py",
    ROOT / "scripts" / "gaze_diversity" / "metrics.py",
    ROOT / "scripts" / "gaze_diversity" / "model.py",
    ROOT / "scripts" / "gaze_diversity" / "resources.py",
    ROOT / "scripts" / "gaze_diversity" / "train.py",
    ROOT / "core" / "unigaze_personalization" / "model.py",
    ROOT / "core" / "unigaze_personalization" / "preprocess.py",
    ROOT / "core" / "unigaze_personalization" / "transforms.py",
    Path(__file__).resolve(),
)
SOURCE_AUDIT_IMPLEMENTATION_PATHS = (ROOT / "scripts" / "columbia_gaze" / "data.py",)
MPII_SUBJECTS = tuple(f"p{index:02d}" for index in range(15))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--base-protocol", type=Path, default=DEFAULT_BASE_PROTOCOL)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--mpii-root", type=Path, default=DEFAULT_MPII_ROOT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run the full source decode audit without importing Torch.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Also prepare frozen candidate and production inputs, without Torch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.audit_only and args.prepare_only:
        raise ValueError("choose at most one staged-execution flag")
    protocol_path = args.protocol.resolve()
    base_protocol_path = args.base_protocol.resolve()
    protocol_bytes = protocol_path.read_bytes()
    base_protocol_bytes = base_protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    base_protocol = json.loads(base_protocol_bytes.decode("utf-8"))
    _validate_protocols(protocol, base_protocol, base_protocol_bytes)
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    base_protocol_sha256 = hashlib.sha256(base_protocol_bytes).hexdigest()
    implementation_sha256 = _combined_sha256(IMPLEMENTATION_PATHS)
    source_audit_implementation_sha256 = _combined_sha256(
        SOURCE_AUDIT_IMPLEMENTATION_PATHS
    )
    production_sha256_before = _production_sha256()
    archive_path = args.dataset_root / "columbia_gaze_data_set.zip"
    annotation_archive_path = args.dataset_root / "eye_corner_locations.zip"
    archive_sha256 = _file_sha256(archive_path)
    annotation_archive_sha256 = _file_sha256(annotation_archive_path)
    _validate_source_files(
        protocol,
        archive_path,
        annotation_archive_path,
        archive_sha256,
        annotation_archive_sha256,
    )
    source_integrity = {
        "protocol_sha256": protocol_sha256,
        "base_protocol_sha256": base_protocol_sha256,
        "archive_sha256": archive_sha256,
        "annotation_archive_sha256": annotation_archive_sha256,
        "source_audit_implementation_sha256": source_audit_implementation_sha256,
    }
    image_root = args.dataset_root / "extracted" / "Columbia Gaze Data Set"
    annotation_csv = (
        args.dataset_root
        / "extracted"
        / "eye_corner_locations"
        / "eye_corner_locations.csv"
    )
    bundle = _load_or_run_source_audit(
        image_root=image_root,
        annotation_csv=annotation_csv,
        work_dir=args.work_dir,
        integrity=source_integrity,
    )
    audit_envelope = {
        "schema_version": 1,
        "experiment": PROTOCOL_ID,
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_sha256": protocol_sha256,
        "base_protocol_sha256": base_protocol_sha256,
        "archive_sha256": archive_sha256,
        "annotation_archive_sha256": annotation_archive_sha256,
        "source_audit_implementation_sha256": source_audit_implementation_sha256,
        "data_audit": bundle.audit,
        "model_runtime_imported": "torch" in sys.modules,
    }
    audit_path = (
        args.output_dir / "2026-08-08-columbia-cross-domain-gaze-v2-data-audit.json"
    )
    _atomic_json_write(audit_path, audit_envelope)
    print(
        "DATA_AUDIT "
        f"status={bundle.audit['status']} "
        f"subjects={bundle.audit['subject_count']} "
        f"images={bundle.audit['image_count']} "
        f"annotations={bundle.audit['official_annotation_count']} "
        f"missing={bundle.audit['missing_annotation_count']}",
        flush=True,
    )
    if args.audit_only:
        print(f"AUDIT_JSON={audit_path}")
        return 0

    landmark_sha256 = _file_sha256(LANDMARK_MODEL)
    prepare_integrity = {
        "protocol_sha256": protocol_sha256,
        "implementation_sha256": implementation_sha256,
        "archive_sha256": archive_sha256,
        "annotation_archive_sha256": annotation_archive_sha256,
        "landmark_model_sha256": landmark_sha256,
        "production_sha256": production_sha256_before,
    }
    candidate_path, production_path, preprocessing = prepare_inputs(
        bundle=bundle,
        work_dir=args.work_dir,
        landmark_model_path=LANDMARK_MODEL,
        integrity=prepare_integrity,
    )
    print(
        "PREPARED "
        f"candidate={preprocessing['candidate']['rows']} "
        f"fallback={preprocessing['candidate']['fallback']['succeeded']} "
        f"production={preprocessing['production']['succeeded']} "
        f"coverage={preprocessing['production']['coverage']:.6f}",
        flush=True,
    )
    if args.prepare_only:
        print(f"PREPARE_MANIFEST={candidate_path.parent / 'manifest.json'}")
        return 0

    return _run_models_and_report(
        args=args,
        protocol=protocol,
        base_protocol=base_protocol,
        protocol_path=protocol_path,
        base_protocol_path=base_protocol_path,
        protocol_sha256=protocol_sha256,
        base_protocol_sha256=base_protocol_sha256,
        implementation_sha256=implementation_sha256,
        production_sha256_before=production_sha256_before,
        archive_sha256=archive_sha256,
        annotation_archive_sha256=annotation_archive_sha256,
        landmark_sha256=landmark_sha256,
        bundle=bundle,
        preprocessing=preprocessing,
        candidate_path=candidate_path,
        production_path=production_path,
    )


def _run_models_and_report(
    *,
    args: argparse.Namespace,
    protocol: dict[str, Any],
    base_protocol: dict[str, Any],
    protocol_path: Path,
    base_protocol_path: Path,
    protocol_sha256: str,
    base_protocol_sha256: str,
    implementation_sha256: str,
    production_sha256_before: str,
    archive_sha256: str,
    annotation_archive_sha256: str,
    landmark_sha256: str,
    bundle: SourceBundle,
    preprocessing: dict[str, Any],
    candidate_path: Path,
    production_path: Path,
) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    compute = base_protocol["compute"]
    monitor = ResourceMonitor(
        maximum_temperature_celsius=float(compute["maximum_temperature_celsius"]),
        maximum_wall_time_hours=float(compute["maximum_model_wall_time_hours"]),
    )
    monitor.checkpoint(process_memory_bytes=0, label="formal-model-start")

    mpii_data, mpii_audit = load_evaluation_data(
        args.mpii_root,
        MPII_SUBJECTS,
        expected_samples_per_subject=3000,
    )
    pose_mean, pose_scale = fit_pose_standardization(mpii_data.poses)
    standardized_train_poses = standardize_pose(
        mpii_data.poses,
        pose_mean,
        pose_scale,
    )
    with np.load(candidate_path, allow_pickle=False) as prepared:
        eye_images = prepared["eye_images"]
        raw_eye_poses = prepared["raw_eye_poses"]
        targets = prepared["targets"].astype(np.float64)
        subject_indices = prepared["subject_indices"]
        head_poses = prepared["head_poses"]
        vertical_gazes = prepared["vertical_gazes"]
        horizontal_gazes = prepared["horizontal_gazes"]
        fallback_mask = prepared["fallback_mask"]
        production_success = prepared["production_success"]
    eye_poses = standardize_pose(
        raw_eye_poses.reshape(-1, 2),
        pose_mean,
        pose_scale,
    ).reshape(raw_eye_poses.shape)

    production_model, production_model_meta = _load_production_model(
        monitor=monitor,
        memory_fraction=float(compute["per_process_memory_fraction"]),
    )
    import torch

    production_model.to("cpu")
    torch.cuda.empty_cache()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    network_attempts: list[str] = []
    training_config = {
        "cpu_threads": int(compute["cpu_threads"]),
        "memory_fraction": float(compute["per_process_memory_fraction"]),
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "loss_beta_radians": 0.05,
        "batch_size": 512,
        "epochs": 11,
        "gradient_norm_clip": 1.0,
    }
    checkpoint_integrity = {
        "protocol_sha256": protocol_sha256,
        "base_protocol_sha256": base_protocol_sha256,
        "implementation_sha256": implementation_sha256,
        "mpii_source_sha256": str(mpii_audit["source_sha256"]),
    }
    checkpoint_payloads: list[dict[str, Any]] = []
    candidate_training: list[dict[str, Any]] = []
    with _deny_network(network_attempts):
        for seed in base_protocol["research_candidate"]["seeds"]:
            payload, summary = train_or_load_final_candidate(
                train_images=mpii_data.images,
                train_poses=standardized_train_poses,
                train_targets=mpii_data.targets,
                seed=int(seed),
                config=training_config,
                checkpoint_path=args.work_dir
                / "checkpoints"
                / f"candidate-seed-{seed}.pt",
                integrity=checkpoint_integrity,
                resource_checkpoint=lambda memory, label: monitor.checkpoint(
                    process_memory_bytes=memory,
                    label=label,
                ),
            )
            checkpoint_payloads.append(payload)
            candidate_training.append(summary)

        member_predictions = predict_candidate(
            checkpoint_payloads=checkpoint_payloads,
            eye_images=eye_images,
            eye_poses=eye_poses,
            batch_size=512,
            resource_checkpoint=lambda memory, label: monitor.checkpoint(
                process_memory_bytes=memory,
                label=label,
            ),
        )
        candidate_prediction = fuse_angle_predictions(member_predictions)
        production_prediction = _predict_production(
            model=production_model,
            production_path=production_path,
            success_mask=production_success,
            batch_size=int(compute["production_inference_batch_size"]),
            monitor=monitor,
        )
    monitor.checkpoint(
        process_memory_bytes=monitor.peak_process_memory_bytes,
        label="formal-model-complete",
    )

    evidence_path = args.work_dir / "prediction-evidence.npz"
    evidence_integrity = {
        "protocol_sha256": protocol_sha256,
        "base_protocol_sha256": base_protocol_sha256,
        "implementation_sha256": implementation_sha256,
        "candidate_state_dict_sha256": [
            str(summary["state_dict_sha256"]) for summary in candidate_training
        ],
        "production_state_dict_sha256": production_model_meta["state_dict_sha256"],
    }
    _write_prediction_evidence(
        evidence_path,
        integrity=evidence_integrity,
        candidate_prediction=candidate_prediction,
        production_prediction=production_prediction,
        targets=targets,
        subject_indices=subject_indices,
        head_poses=head_poses,
        vertical_gazes=vertical_gazes,
        horizontal_gazes=horizontal_gazes,
        fallback_mask=fallback_mask,
        production_success=production_success,
    )
    evidence_sha256 = _file_sha256(evidence_path)

    zero_all, zero_subject_means = zero_gaze_summary(
        targets,
        subject_indices,
        head_poses=head_poses,
        vertical_gazes=vertical_gazes,
        horizontal_gazes=horizontal_gazes,
    )
    candidate_summary = summarize_model(
        candidate_prediction,
        targets,
        subject_indices,
        head_poses=head_poses,
        vertical_gazes=vertical_gazes,
        horizontal_gazes=horizontal_gazes,
        zero_subject_means=zero_subject_means,
        bootstrap_resamples=20000,
        bootstrap_seed=20260813,
    )
    candidate_errors = angular_errors_degrees(candidate_prediction, targets)
    candidate_summary["eye_corner_strata"] = {
        "official_annotation_5865": summarize_errors(candidate_errors[~fallback_mask]),
        "mediapipe_fallback_15": summarize_errors(candidate_errors[fallback_mask]),
    }

    successful = np.asarray(production_success, dtype=bool)
    production_targets = targets[successful]
    production_subjects = subject_indices[successful]
    production_heads = head_poses[successful]
    production_verticals = vertical_gazes[successful]
    production_horizontals = horizontal_gazes[successful]
    production_zero, production_zero_subject_means = zero_gaze_summary(
        production_targets,
        production_subjects,
        head_poses=production_heads,
        vertical_gazes=production_verticals,
        horizontal_gazes=production_horizontals,
    )
    production_summary = summarize_model(
        production_prediction,
        production_targets,
        production_subjects,
        head_poses=production_heads,
        vertical_gazes=production_verticals,
        horizontal_gazes=production_horizontals,
        zero_subject_means=production_zero_subject_means,
        bootstrap_resamples=20000,
        bootstrap_seed=20260813,
    )
    candidate_effectiveness = _effectiveness_decision(
        candidate_summary,
        zero_all,
        minimum_subjects=42,
    )
    production_effectiveness = _effectiveness_decision(
        production_summary,
        production_zero,
        minimum_subjects=42,
    )
    production_effectiveness["coverage_at_least_0_95"] = (
        float(preprocessing["production"]["coverage"]) >= 0.95
    )
    production_effectiveness["passed"] = all(
        value for key, value in production_effectiveness.items() if key != "passed"
    )

    production_sha256_after = _production_sha256()
    resources = monitor.summary()
    metrics_payload = {
        "zero_gaze_all_rows": zero_all,
        "candidate": candidate_summary,
        "production_zero_gaze_success_population": production_zero,
        "production": production_summary,
    }
    finite_metrics = _all_finite(metrics_payload)
    gates = {
        "source_audit_passed": bundle.audit["status"] == "passed",
        "official_annotation_count_exact": (
            bundle.audit["official_annotation_count"] == 5865
        ),
        "fallback_eligible_count_exact": (
            preprocessing["candidate"]["fallback"]["eligible"] == 15
        ),
        "fallback_success_count_exact": (
            preprocessing["candidate"]["fallback"]["succeeded"] == 15
        ),
        "all_candidate_images_scored": len(candidate_prediction) == 5880,
        "network_attempts_zero": not network_attempts,
        "finite_metrics": finite_metrics,
        "hashes_recorded": all(
            len(value) == 64
            for value in (
                protocol_sha256,
                base_protocol_sha256,
                implementation_sha256,
                archive_sha256,
                annotation_archive_sha256,
                landmark_sha256,
                production_model_meta["state_dict_sha256"],
            )
        ),
        "peak_process_memory_within_budget": (
            float(resources["peak_process_memory_gib"]) <= 6.0
        ),
        "temperature_within_budget": (
            float(resources["peak_temperature_celsius"]) < 82.0
        ),
        "model_wall_time_within_budget": float(resources["elapsed_hours"]) <= 8.0,
        "production_source_unchanged": (
            production_sha256_before == production_sha256_after
        ),
    }
    gates["passed"] = all(gates.values())
    result = {
        "schema_version": 1,
        "experiment": PROTOCOL_ID,
        "run_id": str(args.run_id),
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "id": PROTOCOL_ID,
            "sha256": protocol_sha256,
            "commit": _protocol_commit(protocol_path),
            "base_id": BASE_PROTOCOL_ID,
            "base_sha256": base_protocol_sha256,
            "base_commit": _protocol_commit(base_protocol_path),
        },
        "implementation_sha256": implementation_sha256,
        "production_sha256_before": production_sha256_before,
        "production_sha256_after": production_sha256_after,
        "sources": {
            "archive_sha256": archive_sha256,
            "annotation_archive_sha256": annotation_archive_sha256,
            "landmark_model_sha256": landmark_sha256,
            "data_audit": bundle.audit,
            "mpii_training_audit": _aggregate_mpii_audit(mpii_audit),
        },
        "preprocessing": preprocessing,
        "configuration": {
            "candidate": base_protocol["research_candidate"],
            "production": base_protocol["production_baseline"],
            "missing_annotation_fallback": protocol["missing_annotation_fallback"],
        },
        "candidate_training": candidate_training,
        "production_model": production_model_meta,
        "local_prediction_evidence": {
            "file_name": evidence_path.name,
            "sha256": evidence_sha256,
            "candidate_rows": len(candidate_prediction),
            "production_rows": len(production_prediction),
            "committed": False,
        },
        "metrics": metrics_payload,
        "resources": resources,
        "network_attempts_during_model_execution": network_attempts,
        "compute": {
            "device": "cuda:0",
            "platform": platform.platform(),
            "python": platform.python_version(),
            "tf32_allowed": False,
            "automatic_mixed_precision": False,
        },
        "execution_gates": gates,
        "effectiveness": {
            "candidate": candidate_effectiveness,
            "production": production_effectiveness,
        },
        "decision": {
            "execution_status": "passed" if gates["passed"] else "failed",
            "candidate_status": (
                "confirmed_research_baseline"
                if candidate_effectiveness["passed"]
                else "not_confirmed_no_columbia_tuning_allowed"
            ),
            "production_status": (
                "external_baseline_supported"
                if production_effectiveness["passed"]
                else "external_baseline_not_supported_no_columbia_tuning_allowed"
            ),
            "production_model_changed": (
                production_sha256_before != production_sha256_after
            ),
            "real_webcam_generalization_claimed": False,
            "participant_pilot_authorized": False,
        },
    }
    stem = f"2026-08-08-columbia-cross-domain-gaze-v2-{args.run_id}"
    result_path = args.output_dir / f"{stem}.json"
    report_path = args.output_dir.parent / f"{stem}.md"
    _atomic_json_write(result_path, result)
    _atomic_text_write(report_path, _markdown_report(result))
    print(
        "FINAL_RESULT "
        f"execution={gates['passed']} "
        f"candidate={candidate_summary['macro_subject_mean_degrees']:.4f} "
        f"production={production_summary['macro_subject_mean_degrees']:.4f} "
        f"zero={zero_all['macro_subject_mean_degrees']:.4f} "
        f"coverage={preprocessing['production']['coverage']:.6f}",
        flush=True,
    )
    print(f"RESULT_JSON={result_path}")
    print(f"REPORT={report_path}")
    return 0 if gates["passed"] else 1


def _load_production_model(
    *,
    monitor: ResourceMonitor,
    memory_fraction: float,
) -> tuple[Any, dict[str, Any]]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("formal run requires exactly one visible CUDA GPU")
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    from core.unigaze_personalization.model import (
        UniGazeFeatureWrapper,
        count_parameters,
        load_unigaze_b16,
    )

    model = UniGazeFeatureWrapper(load_unigaze_b16("cuda")).to("cuda").eval()
    state_hash = state_dict_sha256(dict(model.state_dict()))
    with torch.inference_mode():
        smoke = model(torch.zeros((1, 3, 224, 224), device="cuda"))
    if smoke.shape != (1, 2) or not torch.isfinite(smoke).all():
        raise RuntimeError("production model synthetic smoke failed")
    peak = int(torch.cuda.max_memory_reserved(torch.device("cuda:0")))
    monitor.checkpoint(process_memory_bytes=peak, label="production-model-smoke")
    return model, {
        "name": "unigaze_b16_joint",
        "parameter_count": count_parameters(model),
        "state_dict_sha256": state_hash,
        "synthetic_smoke_passed": True,
    }


def _predict_production(
    *,
    model: Any,
    production_path: Path,
    success_mask: np.ndarray,
    batch_size: int,
    monitor: ResourceMonitor,
) -> np.ndarray:
    import torch

    from core.unigaze_personalization.transforms import to_unigaze_tensor

    faces = np.load(production_path, mmap_mode="r", allow_pickle=False)
    indices = np.flatnonzero(success_mask)
    model.to("cuda").eval()
    predictions: list[np.ndarray] = []
    torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            tensor = torch.stack(
                [to_unigaze_tensor(np.asarray(faces[index])) for index in selected]
            ).to("cuda", non_blocking=True)
            predictions.append(model(tensor).detach().cpu().numpy())
            if start % (batch_size * 20) == 0 or start + batch_size >= len(indices):
                peak = int(torch.cuda.max_memory_reserved(torch.device("cuda:0")))
                monitor.checkpoint(
                    process_memory_bytes=peak,
                    label=f"production-inference-{min(start + batch_size, len(indices))}",
                )
    result = np.concatenate(predictions, axis=0).astype(np.float64, copy=False)
    if result.shape != (len(indices), 2) or not np.isfinite(result).all():
        raise ValueError("production prediction output is invalid")
    return result


def _load_or_run_source_audit(
    *,
    image_root: Path,
    annotation_csv: Path,
    work_dir: Path,
    integrity: dict[str, str],
) -> SourceBundle:
    fast = audit_columbia_source(
        image_root,
        annotation_csv,
        decode_images=False,
    )
    cache_path = work_dir / "source-audit.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("integrity") == integrity
            and cached.get("audit", {}).get("decoded_all_images") is True
        ):
            return SourceBundle(
                samples=fast.samples,
                official_corners=fast.official_corners,
                missing_annotation_identities=fast.missing_annotation_identities,
                audit=dict(cached["audit"]),
            )
    audited = audit_columbia_source(
        image_root,
        annotation_csv,
        decode_images=True,
    )
    _atomic_json_write(
        cache_path,
        {
            "schema_version": 1,
            "integrity": dict(integrity),
            "audit": audited.audit,
        },
    )
    return audited


def _validate_protocols(
    protocol: dict[str, Any],
    base: dict[str, Any],
    base_bytes: bytes,
) -> None:
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected v2 protocol")
    if base.get("protocol_id") != BASE_PROTOCOL_ID:
        raise ValueError("unexpected base protocol")
    if protocol.get("status") != (
        "frozen_after_source_metadata_preflight_before_archive_extraction_"
        "image_decode_or_model_execution"
    ):
        raise ValueError("v2 protocol is not frozen")
    if base.get("status") != "frozen_before_external_data_download_or_model_execution":
        raise ValueError("base protocol is not frozen")
    expected_base_hash = hashlib.sha256(base_bytes).hexdigest()
    if protocol["base_protocol"]["sha256"] != expected_base_hash:
        raise ValueError("v2 base protocol hash mismatch")
    if not protocol["allowed_delta_from_v1"][
        "all_other_v1_model_data_label_crop_training_metric_effectiveness_compute_and_mutation_fields_unchanged"
    ]:
        raise ValueError("v2 inheritance boundary changed")
    if protocol["source_audit_override"]["official_annotation_rows_equals"] != 5865:
        raise ValueError("unexpected v2 official annotation count")
    if (
        protocol["source_audit_override"]["official_missing_annotation_rows_equals"]
        != 15
    ):
        raise ValueError("unexpected v2 missing annotation count")
    if base["research_candidate"]["training"]["epochs"] != 11:
        raise ValueError("candidate epoch contract changed")
    if base["research_candidate"]["seeds"] != [20260808, 20260809, 20260810]:
        raise ValueError("candidate seed contract changed")
    if base["data"]["expected"]["total_images"] != 5880:
        raise ValueError("Columbia sample contract changed")
    if base["confirmation_policy"]["post_result_parameter_tuning_allowed"]:
        raise ValueError("post-result tuning must remain disabled")


def _validate_source_files(
    protocol: dict[str, Any],
    archive_path: Path,
    annotation_path: Path,
    archive_sha256: str,
    annotation_sha256: str,
) -> None:
    expected = protocol["source_audit_override"]
    if archive_path.stat().st_size != int(expected["archive_bytes_equals"]):
        raise ValueError("Columbia archive byte length mismatch")
    if annotation_path.stat().st_size != int(
        expected["eye_corner_archive_bytes_equals"]
    ):
        raise ValueError("eye-corner archive byte length mismatch")
    if archive_sha256 != expected["archive_sha256_equals"]:
        raise ValueError("Columbia archive hash mismatch")
    if annotation_sha256 != expected["eye_corner_archive_sha256_equals"]:
        raise ValueError("eye-corner archive hash mismatch")


def _effectiveness_decision(
    model: dict[str, Any],
    zero: dict[str, Any],
    *,
    minimum_subjects: int,
) -> dict[str, bool]:
    result = {
        "macro_mean_less_than_zero_gaze": (
            float(model["macro_subject_mean_degrees"])
            < float(zero["macro_subject_mean_degrees"])
        ),
        "bootstrap_ci_upper_less_than_zero": (
            float(model["model_minus_zero_subject_bootstrap"]["ci95_upper_degrees"])
            < 0.0
        ),
        "subjects_beating_zero_gaze_at_least_required": (
            int(model["subjects_beating_zero_gaze"]) >= minimum_subjects
        ),
    }
    result["passed"] = all(result.values())
    return result


def _aggregate_mpii_audit(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": audit["status"],
        "subject_count": audit["subject_count"],
        "total_samples": audit["total_samples"],
        "split_overlap_count": audit["split_overlap_count"],
        "source_file_count": audit["source_file_count"],
        "source_sha256": audit["source_sha256"],
    }


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(value))
    return False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_prediction_evidence(
    path: Path,
    *,
    integrity: dict[str, Any],
    candidate_prediction: np.ndarray,
    production_prediction: np.ndarray,
    targets: np.ndarray,
    subject_indices: np.ndarray,
    head_poses: np.ndarray,
    vertical_gazes: np.ndarray,
    horizontal_gazes: np.ndarray,
    fallback_mask: np.ndarray,
    production_success: np.ndarray,
) -> None:
    """Persist ignored row-level evidence for aggregate result recomputation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    metadata_json = json.dumps(integrity, sort_keys=True, separators=(",", ":"))
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata_json=np.asarray(metadata_json),
            candidate_prediction=np.asarray(candidate_prediction, dtype=np.float64),
            production_prediction=np.asarray(production_prediction, dtype=np.float64),
            targets=np.asarray(targets, dtype=np.float64),
            subject_indices=np.asarray(subject_indices, dtype=np.int16),
            head_poses=np.asarray(head_poses, dtype=np.int8),
            vertical_gazes=np.asarray(vertical_gazes, dtype=np.int8),
            horizontal_gazes=np.asarray(horizontal_gazes, dtype=np.int8),
            fallback_mask=np.asarray(fallback_mask, dtype=bool),
            production_success=np.asarray(production_success, dtype=bool),
        )
    os.replace(temporary, path)


def _markdown_report(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    zero = metrics["zero_gaze_all_rows"]
    candidate = metrics["candidate"]
    production = metrics["production"]
    preprocessing = result["preprocessing"]
    effectiveness = result["effectiveness"]
    resources = result["resources"]
    return "\n".join(
        (
            "# Columbia Cross-Domain Gaze v2 — Run 001",
            "",
            "## Outcome",
            "",
            f"- Execution integrity: `{result['execution_gates']['passed']}`",
            f"- Candidate confirmed: `{effectiveness['candidate']['passed']}`",
            f"- Production external baseline supported: `{effectiveness['production']['passed']}`",
            f"- Zero-gaze macro subject error: `{zero['macro_subject_mean_degrees']:.4f}` degrees",
            f"- TinyCNN candidate macro subject error: `{candidate['macro_subject_mean_degrees']:.4f}` degrees",
            f"- UniGaze production macro subject error: `{production['macro_subject_mean_degrees']:.4f}` degrees",
            f"- Production preprocessing coverage: `{preprocessing['production']['coverage']:.2%}`",
            "",
            "## Integrity and resources",
            "",
            f"- Candidate rows: `{preprocessing['candidate']['rows']}`",
            f"- MediaPipe fallback: `{preprocessing['candidate']['fallback']['succeeded']}/15`",
            f"- Model-run time: `{resources['elapsed_hours']:.3f}` hours",
            f"- Peak process VRAM: `{resources['peak_process_memory_gib']:.3f}` GiB",
            f"- Peak GPU temperature: `{resources['peak_temperature_celsius']:.1f}` C",
            f"- Network attempts during model execution: `{len(result['network_attempts_during_model_execution'])}`",
            f"- Production source unchanged: `{result['execution_gates']['production_source_unchanged']}`",
            "",
            "## Claim boundary",
            "",
            "This is one-time external public-dataset engineering evidence. It does not establish LexiGaze webcam accuracy, does not authorize a participant pilot, does not measure reading or English ability, and does not change the production model. Columbia outcomes may not be used for a tuned rerun presented as independent confirmation.",
            "",
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

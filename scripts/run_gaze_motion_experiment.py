"""Run a preregistered, CPU-only motion-block gaze calibration experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=ROOT / "data" / "sessions",
    )
    parser.add_argument(
        "--output-model-name",
        default="motion_run_001_nested_cpu",
    )
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=8)
    return parser.parse_args()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_fingerprint(session_dir: Path) -> dict[str, Any]:
    manifest = session_dir / "manifest.jsonl"
    image_digest = hashlib.sha256()
    image_hashes: list[str] = []
    valid_rows = 0
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        relative_path = record.get("normalized_face_path")
        if not relative_path:
            continue
        image_path = (session_dir / str(relative_path)).resolve()
        if session_dir not in image_path.parents:
            raise ValueError(
                f"normalized image escapes session at manifest line {line_number}"
            )
        if not image_path.is_file():
            raise ValueError(f"normalized image missing at manifest line {line_number}")
        file_hash = _sha256_file(image_path)
        image_hashes.append(file_hash)
        image_digest.update(line_number.to_bytes(8, "big"))
        image_digest.update(bytes.fromhex(file_hash))
        valid_rows += 1
    if valid_rows == 0:
        raise ValueError("manifest contains no normalized images")
    return {
        "manifest_sha256": _sha256_file(manifest),
        "normalized_image_count": valid_rows,
        "normalized_images_sha256": image_digest.hexdigest(),
        "unique_normalized_image_count": len(set(image_hashes)),
    }


def _parse_csv_rows(output: str) -> list[list[str]]:
    return [
        [field.strip() for field in line.split(",")]
        for line in output.splitlines()
        if line.strip()
    ]


def _optional_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _gpu_snapshot() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False}
    device_result = subprocess.run(
        [
            executable,
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    process_result = subprocess.run(
        [
            executable,
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    devices = []
    for row in _parse_csv_rows(device_result.stdout):
        if len(row) < 5:
            continue
        utilization = _optional_float(row[2])
        memory_used = _optional_float(row[3])
        memory_total = _optional_float(row[4])
        devices.append(
            {
                "index": int(row[0]),
                "name": row[1],
                "utilization_percent": utilization,
                "memory_used_mib": memory_used,
                "memory_total_mib": memory_total,
            }
        )
    processes = []
    for row in _parse_csv_rows(process_result.stdout):
        if len(row) < 2 or not row[0].isdigit():
            continue
        processes.append(
            {
                "pid": int(row[0]),
                "memory_used_mib": _optional_float(row[1]),
            }
        )
    return {
        "available": device_result.returncode == 0,
        "devices": devices,
        "compute_processes": processes,
    }


def _monitor_gpu(
    stop_event: threading.Event,
    observations: dict[str, Any],
) -> None:
    current_pid = os.getpid()
    observations.update(
        {
            "poll_count": 0,
            "current_process_observed": False,
            "peak_current_process_memory_mib": 0.0,
            "peak_device_memory_used_mib": 0.0,
            "peak_device_utilization_percent": 0.0,
        }
    )
    while not stop_event.wait(0.5):
        try:
            snapshot = _gpu_snapshot()
        except (OSError, subprocess.SubprocessError, ValueError):
            continue
        observations["poll_count"] += 1
        for device in snapshot.get("devices", []):
            if device["memory_used_mib"] is not None:
                observations["peak_device_memory_used_mib"] = max(
                    observations["peak_device_memory_used_mib"],
                    device["memory_used_mib"],
                )
            if device["utilization_percent"] is not None:
                observations["peak_device_utilization_percent"] = max(
                    observations["peak_device_utilization_percent"],
                    device["utilization_percent"],
                )
        for process in snapshot.get("compute_processes", []):
            if process["pid"] == current_pid:
                observations["current_process_observed"] = True
                if process["memory_used_mib"] is not None:
                    observations["peak_current_process_memory_mib"] = max(
                        observations["peak_current_process_memory_mib"],
                        process["memory_used_mib"],
                    )


def main() -> int:
    args = _parse_args()
    if args.cpu_threads <= 0:
        raise SystemExit("--cpu-threads must be positive")
    sessions_dir = args.sessions_dir.resolve()
    session_dir = (sessions_dir / args.session_id).resolve()
    if session_dir.parent != sessions_dir:
        raise SystemExit("--session-id must name one direct child session")
    manifest_path = session_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise SystemExit("session manifest not found")

    from core.gaze_core.model_registry import model_path
    from core.gaze_core.motion_robustness import audit_payload, load_motion_samples

    output_model_path = model_path(ROOT, args.output_model_name)
    if output_model_path.exists():
        raise SystemExit(f"refusing to overwrite existing model: {output_model_path.name}")

    samples, diagnostics = load_motion_samples(
        args.sessions_dir.resolve(),
        session_ids=(args.session_id,),
    )
    coverage = audit_payload(samples, diagnostics)
    fingerprint = _input_fingerprint(session_dir)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "error",
        "experiment": "real_single_capture_motion_shift",
        "evidence_scope": (
            "one real capture; supports motion-block robustness for this run, "
            "not cross-person or cross-session generalization"
        ),
        "question_answer_datasets_used": False,
        "input": {
            **fingerprint,
            "motion_manifest_source_sha256": diagnostics.source_sha256,
            "session_count": diagnostics.sessions_seen,
        },
        "coverage_gate": coverage,
        "overfit_controls": {
            "candidate_models_fixed_before_encoder_run": True,
            "confirmation_capture_used": False,
            "outer_split_unit": "motion_block_id",
            "outer_validation_target_visible_to_hyperparameter_search": False,
            "promotion_requires_absolute_and_relative_margin": True,
        },
    }
    if coverage["status"] != "ready":
        result["status"] = "not_ready"
        _atomic_write_json(args.json_output.resolve(), result)
        print("GAZE_MOTION_EXPERIMENT=" + json.dumps(result, sort_keys=True))
        return 2

    os.environ.update(
        {
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "OMP_NUM_THREADS": str(args.cpu_threads),
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    gpu_before = _gpu_snapshot()
    gpu_observations: dict[str, Any] = {}
    stop_monitor = threading.Event()
    monitor = threading.Thread(
        target=_monitor_gpu,
        args=(stop_monitor, gpu_observations),
        name="gaze-gpu-monitor",
        daemon=True,
    )
    monitor.start()
    started = time.perf_counter()
    response: dict[str, Any] = {"ok": False, "error": "training did not start"}
    status_code = 500
    training_exception: str | None = None
    torch_cuda_initialized = False
    try:
        import numpy as np
        import torch

        np.random.seed(0)
        torch.manual_seed(0)
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(1)
        from core.gaze_core.training import train_placeholder

        response, status_code = train_placeholder(
            ROOT,
            {
                "allow_cuda": False,
                "base_model_name": "0",
                "data_session_id": args.session_id,
                "output_model_name": args.output_model_name,
            },
        )
        torch_cuda_initialized = torch.cuda.is_initialized()
    except Exception as exc:
        training_exception = type(exc).__name__
        response = {"ok": False, "error": training_exception}
    finally:
        stop_monitor.set()
        monitor.join(timeout=5)
    duration_seconds = time.perf_counter() - started
    gpu_after = _gpu_snapshot()

    if status_code != 200 or not response.get("ok") or not output_model_path.is_file():
        result.update(
            {
                "status": "failed",
                "duration_seconds": duration_seconds,
                "gpu": {
                    "after": gpu_after,
                    "before": gpu_before,
                    "monitor": gpu_observations,
                    "torch_cuda_initialized": torch_cuda_initialized,
                },
                "training_exception": training_exception,
                "training_response": response,
            }
        )
        _atomic_write_json(args.json_output.resolve(), result)
        print("GAZE_MOTION_EXPERIMENT=" + json.dumps(result, sort_keys=True))
        return 1

    model = json.loads(output_model_path.read_text(encoding="utf-8"))
    comparison = model.get("candidate_comparison") or {}
    selected_stage = model["stages"][-1]
    selected_model = selected_stage.get("calibrator_type", "gaze_polynomial")
    cpu_contract_passed = (
        model.get("training_device") == "cpu"
        and not torch_cuda_initialized
        and not gpu_observations.get("current_process_observed", False)
    )
    experiment_valid = (
        comparison.get("validation_scheme")
        == "nested_leave_one_motion_block_out"
        and cpu_contract_passed
    )
    result.update(
        {
            "status": "passed" if experiment_valid else "failed",
            "duration_seconds": duration_seconds,
            "runtime": {
                "cpu_threads": args.cpu_threads,
                "numpy": np.__version__,
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
            },
            "evaluation": comparison,
            "decision": {
                "optimization_adopted": selected_model
                == "motion_conditioned_ridge_v1",
                "selected_model": selected_model,
                "model_artifact": output_model_path.name,
                "model_artifact_sha256": _sha256_file(output_model_path),
            },
            "gpu": {
                "after": gpu_after,
                "before": gpu_before,
                "cpu_contract_passed": cpu_contract_passed,
                "monitor": gpu_observations,
                "torch_cuda_initialized": torch_cuda_initialized,
            },
        }
    )
    _atomic_write_json(args.json_output.resolve(), result)
    print("GAZE_MOTION_EXPERIMENT=" + json.dumps(result, sort_keys=True))
    return 0 if experiment_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

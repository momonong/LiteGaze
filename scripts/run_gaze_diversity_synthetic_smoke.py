"""Run a tiny synthetic CUDA smoke before the frozen real-data experiment."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.gaze_diversity.resources import ResourceMonitor, gpu_snapshot
from scripts.run_subject_holdout_gaze_diversity import (
    DEFAULT_PROTOCOL,
    _deny_network,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "output" / "subject-heldout-gaze-diversity-v1-synthetic-smoke.json"
)
EXPECTED_PARAMETER_COUNT = 81570


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    compute = protocol["compute"]
    candidate = protocol["candidate"]
    training = candidate["training"]
    optimizer = candidate["optimizer"]
    loss = candidate["loss"]

    preflight = []
    for sample_index in range(3):
        preflight.append(gpu_snapshot())
        if sample_index < 2:
            time.sleep(2.0)
    peak_preflight_utilization = max(
        float(sample["utilization_percent"]) for sample in preflight
    )
    peak_preflight_temperature = max(
        float(sample["temperature_celsius"]) for sample in preflight
    )
    if peak_preflight_utilization > 20.0 or peak_preflight_temperature >= 75.0:
        print(
            "SYNTHETIC_SMOKE_DEFERRED "
            f"peak_utilization={peak_preflight_utilization} "
            f"peak_temperature={peak_preflight_temperature}",
            flush=True,
        )
        return 2

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    arrays = _synthetic_arrays(seed=20260808)
    monitor = ResourceMonitor(
        maximum_temperature_celsius=float(compute["maximum_temperature_celsius"]),
        maximum_wall_time_hours=0.1,
    )
    attempts: list[str] = []
    with _deny_network(attempts):
        monitor.checkpoint(process_memory_bytes=0, label="synthetic-smoke-start")
        from scripts.gaze_diversity.train import train_one_fold

        result = train_one_fold(
            **arrays,
            seed=20260808,
            config={
                "job_label": "synthetic-smoke",
                "cpu_threads": int(compute["cpu_threads"]),
                "memory_fraction": float(compute["per_process_memory_fraction"]),
                "learning_rate": float(optimizer["learning_rate"]),
                "weight_decay": float(optimizer["weight_decay"]),
                "loss_beta_radians": float(loss["beta_radians"]),
                "batch_size": min(128, int(training["batch_size"])),
                "max_epochs": 2,
                "early_stopping_patience": 2,
                "early_stopping_min_delta_degrees": 0.0,
                "gradient_norm_clip": float(training["gradient_norm_clip"]),
            },
            resource_checkpoint=lambda memory, label: monitor.checkpoint(
                process_memory_bytes=memory,
                label=label,
            ),
        )

    predictions = result.pop("test_predictions")
    passed = (
        result["parameter_count"] == EXPECTED_PARAMETER_COUNT
        and predictions.shape == (128, 2)
        and np.isfinite(predictions).all()
        and not attempts
        and monitor.peak_process_memory_bytes <= 6 * 1024**3
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "passed" if passed else "failed",
        "synthetic_only": True,
        "real_dataset_accessed": False,
        "candidate_outcome_generated": False,
        "preflight": preflight,
        "parameter_count": result["parameter_count"],
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
        "epochs_completed": result["epochs_completed"],
        "best_epoch": result["best_epoch"],
        "best_validation_mean_degrees": result["best_validation_mean_degrees"],
        "test_prediction_shape": list(predictions.shape),
        "test_predictions_finite": bool(np.isfinite(predictions).all()),
        "resource": monitor.summary(),
        "network_attempts": attempts,
    }
    _atomic_json_write(args.output, payload)
    print(
        "SYNTHETIC_SMOKE_RESULT "
        f"status={payload['status']} "
        f"parameters={payload['parameter_count']} "
        f"peak_vram_gib={payload['resource']['peak_process_memory_gib']:.3f} "
        f"temperature={payload['resource']['peak_temperature_celsius']:.1f}",
        flush=True,
    )
    print(f"SMOKE_JSON={args.output.resolve()}")
    return 0 if passed else 1


def _synthetic_arrays(*, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def make(rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        images = rng.integers(0, 256, size=(rows, 1, 36, 60), dtype=np.uint8)
        poses = rng.normal(0.0, 0.7, size=(rows, 2)).astype(np.float32)
        left = images[:, :, :, :30].mean(axis=(1, 2, 3))
        right = images[:, :, :, 30:].mean(axis=(1, 2, 3))
        contrast = ((right - left) / 255.0).astype(np.float32)
        targets = np.column_stack(
            (
                0.15 * poses[:, 0] + 0.05 * contrast,
                0.15 * poses[:, 1] - 0.05 * contrast,
            )
        ).astype(np.float32)
        return images, poses, targets

    train_images, train_poses, train_targets = make(768)
    validation_images, validation_poses, validation_targets = make(128)
    test_images, test_poses, _ = make(128)
    return {
        "train_images": train_images,
        "train_poses": train_poses,
        "train_targets": train_targets,
        "validation_images": validation_images,
        "validation_poses": validation_poses,
        "validation_targets": validation_targets,
        "test_images": test_images,
        "test_poses": test_poses,
    }


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())

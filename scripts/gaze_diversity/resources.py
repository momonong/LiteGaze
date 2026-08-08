"""Fail-closed GPU and wall-clock supervision for the long experiment."""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResourceMonitor:
    """Collect hardware telemetry and stop before frozen limits are exceeded."""

    maximum_temperature_celsius: float
    maximum_wall_time_hours: float
    started_monotonic: float = field(default_factory=time.monotonic)
    snapshots: list[dict[str, float | str]] = field(default_factory=list)
    peak_process_memory_bytes: int = 0

    def checkpoint(self, *, process_memory_bytes: int, label: str) -> None:
        self.peak_process_memory_bytes = max(
            self.peak_process_memory_bytes,
            int(process_memory_bytes),
        )
        elapsed_hours = (time.monotonic() - self.started_monotonic) / 3600.0
        if elapsed_hours > self.maximum_wall_time_hours:
            raise RuntimeError("wall-time budget exceeded")
        snapshot = gpu_snapshot()
        snapshot["label"] = label
        snapshot["elapsed_hours"] = elapsed_hours
        self.snapshots.append(snapshot)
        temperature = float(snapshot["temperature_celsius"])
        if temperature >= self.maximum_temperature_celsius:
            raise RuntimeError(
                f"GPU temperature reached {temperature:.1f} C; "
                "stopping at the frozen safety boundary"
            )

    def summary(self) -> dict[str, Any]:
        elapsed_hours = (time.monotonic() - self.started_monotonic) / 3600.0
        temperatures = [
            float(snapshot["temperature_celsius"]) for snapshot in self.snapshots
        ]
        utilization = [
            float(snapshot["utilization_percent"]) for snapshot in self.snapshots
        ]
        total_memory = [
            float(snapshot["gpu_memory_used_mib"]) for snapshot in self.snapshots
        ]
        return {
            "elapsed_hours": elapsed_hours,
            "telemetry_samples": len(self.snapshots),
            "peak_temperature_celsius": max(temperatures, default=0.0),
            "peak_utilization_percent": max(utilization, default=0.0),
            "peak_gpu_memory_used_mib": max(total_memory, default=0.0),
            "peak_process_memory_bytes": self.peak_process_memory_bytes,
            "peak_process_memory_gib": self.peak_process_memory_bytes / (1024**3),
            "first_snapshot": self.snapshots[0] if self.snapshots else None,
            "last_snapshot": self.snapshots[-1] if self.snapshots else None,
        }


def gpu_snapshot() -> dict[str, float | str]:
    """Read one NVIDIA GPU snapshot without importing a model runtime."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        raise RuntimeError("nvidia-smi is required by the frozen protocol")
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        raise RuntimeError("nvidia-smi telemetry failed")
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(
            "the frozen protocol requires exactly one visible NVIDIA GPU"
        )
    parts = [part.strip() for part in rows[0].split(",")]
    if len(parts) != 5:
        raise RuntimeError("unexpected nvidia-smi telemetry row")
    return {
        "name": parts[0],
        "temperature_celsius": float(parts[1]),
        "utilization_percent": float(parts[2]),
        "gpu_memory_used_mib": float(parts[3]),
        "power_draw_watts": float(parts[4]),
    }

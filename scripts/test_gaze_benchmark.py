from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_gaze_inference import (
    atomic_write_json,
    gpu_preflight,
    parse_nvidia_smi_row,
    percentile,
    summarize_latencies,
    torch_device_spec,
)


def _gpu_sample(*, utilization: float, memory: float, temperature: float) -> dict:
    return {
        "index": 0,
        "name": "Test GPU",
        "uuid": "GPU-test",
        "driver_version": "1.2.3",
        "memory_used_mib": memory,
        "memory_total_mib": 24_000.0,
        "gpu_utilization_percent": utilization,
        "memory_utilization_percent": 1.0,
        "temperature_c": temperature,
        "power_w": 20.0,
        "performance_state": "P8",
        "sm_clock_mhz": 210.0,
        "memory_clock_mhz": 405.0,
    }


class GazeBenchmarkUnitTests(unittest.TestCase):
    def test_parses_nvidia_smi_row(self):
        row = parse_nvidia_smi_row(
            "0, NVIDIA Test GPU, GPU-123, 581.57, 1024, 24576, 3, 2, 54, "
            "31.5, P8, 210, 405"
        )

        self.assertEqual(row["index"], 0)
        self.assertEqual(row["name"], "NVIDIA Test GPU")
        self.assertEqual(row["memory_used_mib"], 1024.0)
        self.assertEqual(row["gpu_utilization_percent"], 3.0)
        self.assertEqual(row["power_w"], 31.5)

    def test_latency_summary_uses_interpolated_percentiles(self):
        values = [1.0, 2.0, 3.0, 4.0, 10.0]
        summary = summarize_latencies(values)

        self.assertEqual(percentile(values, 0.5), 3.0)
        self.assertAlmostEqual(percentile(values, 0.95), 8.8)
        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["p50_ms"], 3.0)
        self.assertEqual(summary["p95_ms"], 8.8)
        self.assertEqual(summary["max_ms"], 10.0)

    def test_gpu_guard_refuses_busy_samples(self):
        samples = iter([
            _gpu_sample(utilization=2, memory=1000, temperature=55),
            _gpu_sample(utilization=20, memory=3000, temperature=60),
        ])
        guard = gpu_preflight(
            sample_count=2,
            interval_seconds=0,
            max_utilization_percent=5,
            max_memory_mib=2048,
            max_temperature_c=82,
            allow_busy=False,
            sampler=lambda: next(samples),
            sleeper=lambda _seconds: None,
        )

        self.assertFalse(guard["allowed"])
        self.assertFalse(guard["clean"])
        self.assertTrue(guard["contaminated"])
        self.assertEqual(guard["observed_max"]["gpu_utilization_percent"], 20)
        self.assertEqual(len(guard["reasons"]), 2)

    def test_gpu_guard_marks_explicit_override_contaminated(self):
        guard = gpu_preflight(
            sample_count=1,
            interval_seconds=0,
            max_utilization_percent=5,
            max_memory_mib=2048,
            max_temperature_c=82,
            allow_busy=True,
            sampler=lambda: _gpu_sample(
                utilization=30, memory=4096, temperature=70
            ),
            sleeper=lambda _seconds: None,
        )

        self.assertTrue(guard["allowed"])
        self.assertTrue(guard["contaminated"])
        self.assertTrue(guard["override_used"])

    def test_atomic_json_write_replaces_result_without_temp_file(self):
        with tempfile.TemporaryDirectory(prefix="lexigaze-gaze-benchmark-") as name:
            path = Path(name) / "result.json"
            atomic_write_json(path, {"status": "first"})
            atomic_write_json(path, {"status": "passed", "schema_version": 1})

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"status": "passed", "schema_version": 1},
            )
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_import_does_not_load_torch(self):
        self.assertNotIn("torch", sys.modules)

    def test_cuda_device_spec_has_an_explicit_index(self):
        self.assertEqual(torch_device_spec("cuda"), "cuda:0")
        self.assertEqual(torch_device_spec("cpu"), "cpu")


if __name__ == "__main__":
    unittest.main()

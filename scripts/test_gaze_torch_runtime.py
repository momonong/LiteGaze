from __future__ import annotations

import sys
import unittest

from core.gaze_core.torch_runtime import (
    cuda_runtime_available,
    enable_cuda_tf32,
    restore_matmul_precision,
)


class _FakeCuda:
    def __init__(self, capability=(12, 0), error: Exception | None = None):
        self.capability = capability
        self.error = error
        self.queries: list[str] = []
        self.availability_queries = 0

    def is_available(self):
        self.availability_queries += 1
        return True

    def get_device_capability(self, device: str):
        self.queries.append(device)
        if self.error is not None:
            raise self.error
        return self.capability


class _FakeTorch:
    def __init__(self, *, capability=(12, 0), error: Exception | None = None):
        self.cuda = _FakeCuda(capability, error)
        self.precision = "highest"
        self.precision_changes: list[str] = []

    def get_float32_matmul_precision(self):
        return self.precision

    def set_float32_matmul_precision(self, precision: str):
        self.precision_changes.append(precision)
        self.precision = precision


class GazeTorchRuntimeTests(unittest.TestCase):
    def test_explicitly_hidden_cuda_skips_runtime_probe(self):
        torch_module = _FakeTorch()

        self.assertFalse(
            cuda_runtime_available(
                torch_module,
                {"CUDA_VISIBLE_DEVICES": "-1"},
            )
        )
        self.assertFalse(
            cuda_runtime_available(
                torch_module,
                {"CUDA_VISIBLE_DEVICES": "  "},
            )
        )
        self.assertEqual(torch_module.cuda.availability_queries, 0)

    def test_visible_cuda_uses_runtime_probe(self):
        torch_module = _FakeTorch()

        self.assertTrue(
            cuda_runtime_available(
                torch_module,
                {"CUDA_VISIBLE_DEVICES": "0"},
            )
        )
        self.assertEqual(torch_module.cuda.availability_queries, 1)

    def test_supported_cuda_device_enables_high_precision_policy(self):
        torch_module = _FakeTorch(capability=(12, 0))

        previous = enable_cuda_tf32(torch_module, "cuda")

        self.assertEqual(previous, "highest")
        self.assertEqual(torch_module.cuda.queries, ["cuda"])
        self.assertEqual(torch_module.precision, "high")

    def test_cpu_preserves_existing_precision_without_querying_cuda(self):
        torch_module = _FakeTorch()

        previous = enable_cuda_tf32(torch_module, "cpu")

        self.assertIsNone(previous)
        self.assertEqual(torch_module.cuda.queries, [])
        self.assertEqual(torch_module.precision_changes, [])

    def test_pre_ampere_cuda_device_preserves_existing_precision(self):
        torch_module = _FakeTorch(capability=(7, 5))

        previous = enable_cuda_tf32(torch_module, "cuda:0")

        self.assertIsNone(previous)
        self.assertEqual(torch_module.precision_changes, [])

    def test_capability_probe_failure_preserves_existing_precision(self):
        torch_module = _FakeTorch(error=RuntimeError("driver unavailable"))

        previous = enable_cuda_tf32(torch_module, "cuda")

        self.assertIsNone(previous)
        self.assertEqual(torch_module.precision_changes, [])

    def test_restore_only_changes_an_enabled_policy(self):
        torch_module = _FakeTorch()
        torch_module.precision = "high"

        restore_matmul_precision(torch_module, "highest")
        restore_matmul_precision(torch_module, None)

        self.assertEqual(torch_module.precision_changes, ["highest"])
        self.assertEqual(torch_module.precision, "highest")

    def test_module_does_not_import_torch(self):
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()

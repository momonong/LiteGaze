import os
import unittest
from unittest.mock import patch

from core.device import (
    DEVICE_ENV_VAR,
    _cuda_kernels_work,
    configured_device,
    resolve_torch_device,
)


class TestRuntimeDevicePolicy(unittest.TestCase):
    def tearDown(self):
        _cuda_kernels_work.cache_clear()

    def test_cpu_policy_does_not_probe_cuda(self):
        with (
            patch.dict(os.environ, {DEVICE_ENV_VAR: "cpu"}),
            patch("core.device.torch.cuda.is_available") as cuda_available,
        ):
            device = resolve_torch_device()

        self.assertEqual(device.type, "cpu")
        cuda_available.assert_not_called()

    def test_auto_falls_back_to_cpu_when_cuda_is_unavailable(self):
        with (
            patch.dict(os.environ, {DEVICE_ENV_VAR: "auto"}),
            patch("core.device.torch.cuda.is_available", return_value=False),
        ):
            device = resolve_torch_device()

        self.assertEqual(device.type, "cpu")

    def test_explicit_argument_takes_precedence_over_environment(self):
        with (
            patch.dict(os.environ, {DEVICE_ENV_VAR: "cuda"}),
            patch("core.device.torch.cuda.is_available") as cuda_available,
        ):
            device = resolve_torch_device("cpu")

        self.assertEqual(device.type, "cpu")
        cuda_available.assert_not_called()

    def test_invalid_policy_has_actionable_error(self):
        with patch.dict(os.environ, {DEVICE_ENV_VAR: "quantum"}):
            with self.assertRaisesRegex(ValueError, DEVICE_ENV_VAR):
                configured_device()


if __name__ == "__main__":
    unittest.main()

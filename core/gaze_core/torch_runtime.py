"""Hardware-aware PyTorch runtime policy for production gaze inference."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

CUDA_MATMUL_PRECISION = "high"
TF32_MINIMUM_COMPUTE_CAPABILITY = 8


def cuda_runtime_available(
    torch_module: Any,
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Check CUDA availability without probing an explicitly hidden runtime."""

    current_environment = os.environ if environment is None else environment
    visible_devices = current_environment.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None and visible_devices.strip() in {"", "-1"}:
        return False
    return bool(torch_module.cuda.is_available())


def enable_cuda_tf32(torch_module: Any, device: str) -> str | None:
    """Enable TF32-backed float32 matmuls on supported CUDA devices.

    Returns the previous process-wide precision so a caller can restore it if
    CUDA model initialization fails. CPU and pre-Ampere devices remain on the
    existing precision policy.
    """

    if not str(device).startswith("cuda"):
        return None

    try:
        major, _minor = torch_module.cuda.get_device_capability(device)
    except (AttributeError, RuntimeError):
        return None

    if major < TF32_MINIMUM_COMPUTE_CAPABILITY:
        return None

    previous = torch_module.get_float32_matmul_precision()
    torch_module.set_float32_matmul_precision(CUDA_MATMUL_PRECISION)
    return previous


def restore_matmul_precision(torch_module: Any, previous: str | None) -> None:
    """Restore a precision value returned by :func:`enable_cuda_tf32`."""

    if previous is not None:
        torch_module.set_float32_matmul_precision(previous)

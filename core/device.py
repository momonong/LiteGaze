"""Shared PyTorch device selection for LexiGaze runtime components.

Set ``LEXIGAZE_DEVICE`` to ``cpu``, ``cuda``/``cuda:N``, or ``auto``.  An
explicit function argument takes precedence unless it is ``auto``.  CUDA is
only probed when it is a candidate, so selecting CPU does not initialize or
allocate on the GPU.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch


DEVICE_ENV_VAR = "LEXIGAZE_DEVICE"
_SUPPORTED_DEVICE_TYPES = {"cpu", "cuda"}


def configured_device(requested: str | None = None) -> str:
    """Return the normalized device request without probing any hardware."""
    candidate = (requested or "auto").strip().lower()
    if candidate == "auto":
        candidate = os.environ.get(DEVICE_ENV_VAR, "auto").strip().lower() or "auto"

    if candidate == "auto":
        return candidate

    try:
        device = torch.device(candidate)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(
            f"invalid {DEVICE_ENV_VAR} value {candidate!r}; expected auto, cpu, cuda, or cuda:N"
        ) from exc

    if device.type not in _SUPPORTED_DEVICE_TYPES:
        raise ValueError(
            f"unsupported {DEVICE_ENV_VAR} value {candidate!r}; expected auto, cpu, cuda, or cuda:N"
        )
    return str(device)


@lru_cache(maxsize=8)
def _cuda_kernels_work(device_name: str) -> bool:
    """Verify a convolution kernel, catching CUDA/architecture incompatibility."""
    if not torch.cuda.is_available():
        return False

    device = torch.device(device_name)
    try:
        with torch.inference_mode():
            sample = torch.zeros((1, 3, 224, 224), device=device)
            convolution = torch.nn.Conv2d(3, 16, kernel_size=16, stride=16).to(device)
            convolution(sample)
        return True
    except (RuntimeError, AssertionError):
        return False


def resolve_torch_device(requested: str | None = None) -> torch.device:
    """Resolve a safe runtime device, falling back to CPU when CUDA cannot run."""
    candidate = configured_device(requested)
    if candidate == "cpu":
        return torch.device("cpu")

    cuda_candidate = "cuda" if candidate == "auto" else candidate
    if _cuda_kernels_work(cuda_candidate):
        return torch.device(cuda_candidate)
    return torch.device("cpu")

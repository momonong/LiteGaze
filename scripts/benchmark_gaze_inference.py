"""Benchmark LexiGaze gaze inference with explicit hardware safeguards."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1

VARIANTS = (
    "eager",
    "inference",
    "amp-fp16",
    "amp-bf16",
    "compile-default",
    "compile-reduce-overhead",
)
WORKLOADS = ("model", "pipeline")
DEVICES = ("cpu", "cuda")

DEFAULT_WARMUP = 10
DEFAULT_ITERATIONS = 30
MAX_WARMUP = 100
MAX_ITERATIONS = 500
DEFAULT_GUARD_SAMPLES = 5
DEFAULT_GUARD_INTERVAL_MS = 500
DEFAULT_MAX_GPU_UTILIZATION = 5.0
DEFAULT_MAX_GPU_MEMORY_MIB = 2048.0
DEFAULT_MAX_GPU_TEMPERATURE_C = 82.0
DEFAULT_TIMEOUT_SECONDS = 300

GPU_QUERY_FIELDS = (
    "index",
    "name",
    "uuid",
    "driver_version",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "utilization.memory",
    "temperature.gpu",
    "power.draw",
    "pstate",
    "clocks.current.sm",
    "clocks.current.memory",
)


class BenchmarkRefused(RuntimeError):
    """Raised when a safety precondition blocks a hardware run."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=DEVICES, default="cpu")
    parser.add_argument("--variant", choices=VARIANTS, default="eager")
    parser.add_argument("--workload", choices=WORKLOADS, default="model")
    parser.add_argument(
        "--image",
        type=Path,
        help="Explicit local input. Required for the pipeline workload.",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument(
        "--guard-samples", type=int, default=DEFAULT_GUARD_SAMPLES
    )
    parser.add_argument(
        "--guard-interval-ms", type=int, default=DEFAULT_GUARD_INTERVAL_MS
    )
    parser.add_argument(
        "--max-background-gpu-utilization",
        type=float,
        default=DEFAULT_MAX_GPU_UTILIZATION,
    )
    parser.add_argument(
        "--max-background-gpu-memory-mib",
        type=float,
        default=DEFAULT_MAX_GPU_MEMORY_MIB,
    )
    parser.add_argument(
        "--max-start-temperature-c",
        type=float,
        default=DEFAULT_MAX_GPU_TEMPERATURE_C,
    )
    parser.add_argument("--max-abs-error", type=float)
    parser.add_argument("--max-relative-error", type=float)
    parser.add_argument(
        "--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow model libraries to access their remote caches.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--summary-path", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not 0 <= args.warmup <= MAX_WARMUP:
        raise ValueError(f"--warmup must be between 0 and {MAX_WARMUP}")
    if not 1 <= args.iterations <= MAX_ITERATIONS:
        raise ValueError(f"--iterations must be between 1 and {MAX_ITERATIONS}")
    if not 1 <= args.guard_samples <= 20:
        raise ValueError("--guard-samples must be between 1 and 20")
    if not 0 <= args.guard_interval_ms <= 5000:
        raise ValueError("--guard-interval-ms must be between 0 and 5000")
    if not 1 <= args.timeout_seconds <= 1800:
        raise ValueError("--timeout-seconds must be between 1 and 1800")
    if args.workload == "pipeline" and args.image is None:
        raise ValueError("--image is required for the pipeline workload")
    if args.image is not None and not args.image.is_file():
        raise ValueError(f"input image does not exist: {args.image}")
    if args.variant.startswith(("amp-", "compile-")) and args.device != "cuda":
        raise ValueError(f"{args.variant} requires --device cuda")
    for name in (
        "max_background_gpu_utilization",
        "max_background_gpu_memory_mib",
        "max_start_temperature_c",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    for name in ("max_abs_error", "max_relative_error"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    if args.worker and args.summary_path is None:
        raise ValueError("--summary-path is required in worker mode")


def _coerce_number(value: str) -> float | None:
    stripped = value.strip()
    if not stripped or stripped.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def parse_nvidia_smi_row(line: str) -> dict[str, Any]:
    """Parse the stable no-header/no-units query used by the GPU guard."""

    columns = [column.strip() for column in line.split(",")]
    if len(columns) != len(GPU_QUERY_FIELDS):
        raise ValueError(
            f"Expected {len(GPU_QUERY_FIELDS)} GPU columns, received {len(columns)}"
        )
    values = dict(zip(GPU_QUERY_FIELDS, columns, strict=True))
    return {
        "index": int(values["index"]),
        "name": values["name"],
        "uuid": values["uuid"],
        "driver_version": values["driver_version"],
        "memory_used_mib": _coerce_number(values["memory.used"]),
        "memory_total_mib": _coerce_number(values["memory.total"]),
        "gpu_utilization_percent": _coerce_number(values["utilization.gpu"]),
        "memory_utilization_percent": _coerce_number(
            values["utilization.memory"]
        ),
        "temperature_c": _coerce_number(values["temperature.gpu"]),
        "power_w": _coerce_number(values["power.draw"]),
        "performance_state": values["pstate"],
        "sm_clock_mhz": _coerce_number(values["clocks.current.sm"]),
        "memory_clock_mhz": _coerce_number(values["clocks.current.memory"]),
    }


def query_nvidia_smi(device_index: int = 0) -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        raise RuntimeError("nvidia-smi is unavailable")
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=" + ",".join(GPU_QUERY_FIELDS),
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or "nvidia-smi query failed"
        raise RuntimeError(message)
    rows = [
        parse_nvidia_smi_row(line)
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    for row in rows:
        if row["index"] == device_index:
            return row
    raise RuntimeError(f"GPU index {device_index} was not reported")


def gpu_preflight(
    *,
    sample_count: int,
    interval_seconds: float,
    max_utilization_percent: float,
    max_memory_mib: float,
    max_temperature_c: float,
    allow_busy: bool,
    sampler: Callable[[], dict[str, Any]] = query_nvidia_smi,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Sample the GPU before Torch import and decide whether a run is clean."""

    samples: list[dict[str, Any]] = []
    query_errors: list[str] = []
    for index in range(sample_count):
        try:
            sample = dict(sampler())
            sample["sample"] = index + 1
            samples.append(sample)
        except Exception as exc:
            query_errors.append(f"{type(exc).__name__}: {exc}")
        if index + 1 < sample_count and interval_seconds:
            sleeper(interval_seconds)

    reasons: list[str] = []
    if query_errors:
        reasons.append("gpu telemetry unavailable")
    if not samples:
        reasons.append("no GPU telemetry samples")

    def observed_max(field: str) -> float | None:
        values = [sample.get(field) for sample in samples]
        numeric = [float(value) for value in values if value is not None]
        return max(numeric) if numeric else None

    maxima = {
        "gpu_utilization_percent": observed_max("gpu_utilization_percent"),
        "memory_used_mib": observed_max("memory_used_mib"),
        "temperature_c": observed_max("temperature_c"),
    }
    comparisons = (
        (
            "gpu_utilization_percent",
            max_utilization_percent,
            "background GPU utilization",
            "%",
        ),
        ("memory_used_mib", max_memory_mib, "existing GPU memory", " MiB"),
        ("temperature_c", max_temperature_c, "starting GPU temperature", " C"),
    )
    for field, limit, label, unit in comparisons:
        value = maxima[field]
        if value is not None and value > limit:
            reasons.append(f"{label} {value:g}{unit} exceeds {limit:g}{unit}")

    clean = not reasons
    return {
        "allowed": clean or allow_busy,
        "clean": clean,
        "contaminated": not clean,
        "override_used": bool(allow_busy and not clean),
        "reasons": reasons,
        "query_errors": query_errors,
        "thresholds": {
            "gpu_utilization_percent": max_utilization_percent,
            "memory_used_mib": max_memory_mib,
            "temperature_c": max_temperature_c,
        },
        "observed_max": maxima,
        "samples": samples,
    }


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_latencies(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("latency summary requires at least one value")
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "min_ms": round(min(numeric), 6),
        "p50_ms": round(percentile(numeric, 0.50), 6),
        "p95_ms": round(percentile(numeric, 0.95), 6),
        "p99_ms": round(percentile(numeric, 0.99), 6),
        "max_ms": round(max(numeric), 6),
        "mean_ms": round(statistics.fmean(numeric), 6),
        "stdev_ms": round(statistics.pstdev(numeric), 6),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, ensure_ascii=False, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_revision() -> dict[str, Any]:
    result: dict[str, Any] = {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        result = {
            "commit": commit.stdout.strip(),
            "dirty": bool(status.stdout.strip()),
        }
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def _minimal_environment() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
    }


def _full_environment(torch: Any, device: Any) -> dict[str, Any]:
    environment = _minimal_environment()
    cudnn_version = None
    if device.type == "cuda":
        cudnn_version = torch.backends.cudnn.version()
    environment.update({
        "packages": {
            "mediapipe": _package_version("mediapipe"),
            "numpy": _package_version("numpy"),
            "opencv_python": _package_version("opencv-python"),
            "torch": torch.__version__,
            "unigaze": _package_version("unigaze"),
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": {
            "cuda_build": torch.version.cuda,
            "cudnn": cudnn_version,
            "num_interop_threads": torch.get_num_interop_threads(),
            "num_threads": torch.get_num_threads(),
        },
    })
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        environment["accelerator"] = {
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(device)),
            "multiprocessors": properties.multi_processor_count,
            "total_memory_mib": round(properties.total_memory / (1024 * 1024), 3),
        }
    return environment


def _synthetic_rgb() -> Any:
    import numpy as np

    height = width = 224
    y, x = np.mgrid[0:height, 0:width]
    return np.stack(
        (
            (x * 255 // (width - 1)),
            (y * 255 // (height - 1)),
            ((x + y) * 255 // (height + width - 2)),
        ),
        axis=2,
    ).astype(np.uint8)


def _load_input(args: argparse.Namespace, cv2: Any, np: Any) -> dict[str, Any]:
    if args.image is None:
        rgb = _synthetic_rgb()
        return {
            "rgb": rgb,
            "encoded": None,
            "metadata": {
                "kind": "synthetic-gradient",
                "sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "shape": list(rgb.shape),
                "encoded_bytes": None,
                "suffix": None,
            },
        }

    encoded = args.image.read_bytes()
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("OpenCV could not decode the input image")
    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return {
        "rgb": rgb,
        "encoded": encoded,
        "metadata": {
            "kind": "local-file",
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "shape": list(decoded.shape),
            "encoded_bytes": len(encoded),
            "suffix": args.image.suffix.lower(),
        },
    }


@contextlib.contextmanager
def _execution_context(torch: Any, variant: str) -> Iterator[None]:
    gradient_context = torch.no_grad() if variant == "eager" else torch.inference_mode()
    with gradient_context:
        if variant == "amp-fp16":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                yield
        elif variant == "amp-bf16":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                yield
        else:
            yield


def _configure_variant(torch: Any, model: Any, variant: str) -> tuple[Any, float]:
    started = time.perf_counter()
    if variant == "compile-default":
        candidate = torch.compile(model, mode="default")
    elif variant == "compile-reduce-overhead":
        candidate = torch.compile(model, mode="reduce-overhead")
    else:
        candidate = model
    return candidate, (time.perf_counter() - started) * 1000


def _run_iteration(
    *,
    args: argparse.Namespace,
    torch: Any,
    cv2: Any,
    np: Any,
    device: Any,
    candidate: Any,
    input_data: dict[str, Any],
    preprocessor: Any,
    tensor_transform: Callable[[Any], Any],
) -> tuple[dict[str, float], Any]:
    stages: dict[str, float] = {}
    total_started = time.perf_counter()

    if args.workload == "pipeline":
        decode_started = time.perf_counter()
        image_bgr = cv2.imdecode(
            np.frombuffer(input_data["encoded"], dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image_bgr is None:
            raise ValueError("OpenCV could not decode the benchmark image")
        stages["decode_ms"] = (time.perf_counter() - decode_started) * 1000

        preprocess_started = time.perf_counter()
        processed = preprocessor.process(image_bgr)
        image_rgb = processed.image_rgb
        stages["preprocess_ms"] = (
            time.perf_counter() - preprocess_started
        ) * 1000
    else:
        image_rgb = input_data["rgb"]

    transform_started = time.perf_counter()
    cpu_tensor = tensor_transform(image_rgb).unsqueeze(0)
    stages["tensor_transform_ms"] = (
        time.perf_counter() - transform_started
    ) * 1000

    if device.type == "cuda":
        transfer_start = torch.cuda.Event(enable_timing=True)
        transfer_end = torch.cuda.Event(enable_timing=True)
        transfer_start.record()
        device_tensor = cpu_tensor.to(device)
        transfer_end.record()
    else:
        transfer_started = time.perf_counter()
        device_tensor = cpu_tensor.to(device)
        stages["host_to_device_ms"] = (
            time.perf_counter() - transfer_started
        ) * 1000

    def forward() -> Any:
        with _execution_context(torch, args.variant):
            return candidate(device_tensor)

    if device.type == "cuda":
        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        forward_start.record()
        output = forward()
        forward_end.record()
    else:
        forward_started = time.perf_counter()
        output = forward()
        stages["model_forward_ms"] = (
            time.perf_counter() - forward_started
        ) * 1000

    if device.type == "cuda":
        copy_start = torch.cuda.Event(enable_timing=True)
        copy_end = torch.cuda.Event(enable_timing=True)
        copy_start.record()
        output_cpu = output.detach().float().cpu()
        copy_end.record()
        # One synchronization makes every stage event readable while keeping the
        # end-to-end path free of artificial synchronization between stages.
        copy_end.synchronize()
        stages["host_to_device_ms"] = float(
            transfer_start.elapsed_time(transfer_end)
        )
        stages["model_forward_ms"] = float(
            forward_start.elapsed_time(forward_end)
        )
        stages["device_to_host_ms"] = float(copy_start.elapsed_time(copy_end))
    else:
        copy_started = time.perf_counter()
        output_cpu = output.detach().float().cpu()
        stages["device_to_host_ms"] = (
            time.perf_counter() - copy_started
        ) * 1000
    stages["end_to_end_ms"] = (time.perf_counter() - total_started) * 1000
    return stages, output_cpu


def _parity_tolerances(args: argparse.Namespace) -> tuple[float, float]:
    approximate = args.variant in {"amp-fp16", "amp-bf16"}
    absolute = args.max_abs_error
    relative = args.max_relative_error
    if absolute is None:
        absolute = 5e-3 if approximate else 1e-5
    if relative is None:
        relative = 1e-2 if approximate else 1e-4
    return absolute, relative


def _measure_parity(
    *,
    args: argparse.Namespace,
    torch: Any,
    device: Any,
    base_model: Any,
    candidate: Any,
    rgb: Any,
    tensor_transform: Callable[[Any], Any],
) -> dict[str, Any]:
    tensor = tensor_transform(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        reference = base_model(tensor)
    with _execution_context(torch, args.variant):
        observed = candidate(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    reference_cpu = reference.detach().float().cpu()
    observed_cpu = observed.detach().float().cpu()
    difference = (reference_cpu - observed_cpu).abs()
    max_abs = float(difference.max().item())
    mean_abs = float(difference.mean().item())
    absolute_tolerance, relative_tolerance = _parity_tolerances(args)
    allclose = bool(
        torch.allclose(
            reference_cpu,
            observed_cpu,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
    )
    return {
        "allclose": allclose,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "reference": reference_cpu.squeeze(0).tolist(),
        "observed": observed_cpu.squeeze(0).tolist(),
    }


def _run_benchmark(args: argparse.Namespace, guard: dict[str, Any] | None) -> dict[str, Any]:
    if not args.allow_model_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import cv2
    import numpy as np
    import torch

    from core.unigaze_personalization.model import (
        UniGazeFeatureWrapper,
        load_unigaze_b16,
    )
    from core.unigaze_personalization.transforms import to_unigaze_tensor

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise BenchmarkRefused("PyTorch reports that CUDA is unavailable")

    input_started = time.perf_counter()
    input_data = _load_input(args, cv2, np)
    input_load_ms = (time.perf_counter() - input_started) * 1000

    preprocessor = None
    preprocessor_init_ms = 0.0
    if args.workload == "pipeline":
        asset = ROOT / "web" / "static" / "face_landmarker.task"
        if not asset.is_file():
            raise BenchmarkRefused(
                "face_landmarker.task is missing; benchmark will not download it"
            )
        from core.unigaze_personalization.preprocess import (
            MediaPipeUniGazePreprocessor,
        )

        started = time.perf_counter()
        preprocessor = MediaPipeUniGazePreprocessor()
        preprocessor_init_ms = (time.perf_counter() - started) * 1000

    memory_before: dict[str, float] | None = None
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        memory_before = {
            "allocated_mib": torch.cuda.memory_allocated(device) / (1024 * 1024),
            "reserved_mib": torch.cuda.memory_reserved(device) / (1024 * 1024),
        }

    model_started = time.perf_counter()
    base_model = UniGazeFeatureWrapper(load_unigaze_b16(args.device)).to(device).eval()
    model_load_ms = (time.perf_counter() - model_started) * 1000
    candidate, variant_setup_ms = _configure_variant(torch, base_model, args.variant)

    first_stages, _ = _run_iteration(
        args=args,
        torch=torch,
        cv2=cv2,
        np=np,
        device=device,
        candidate=candidate,
        input_data=input_data,
        preprocessor=preprocessor,
        tensor_transform=to_unigaze_tensor,
    )

    for _ in range(args.warmup):
        _run_iteration(
            args=args,
            torch=torch,
            cv2=cv2,
            np=np,
            device=device,
            candidate=candidate,
            input_data=input_data,
            preprocessor=preprocessor,
            tensor_transform=to_unigaze_tensor,
        )

    observations: dict[str, list[float]] = {}
    last_output = None
    for _ in range(args.iterations):
        stages, last_output = _run_iteration(
            args=args,
            torch=torch,
            cv2=cv2,
            np=np,
            device=device,
            candidate=candidate,
            input_data=input_data,
            preprocessor=preprocessor,
            tensor_transform=to_unigaze_tensor,
        )
        for name, value in stages.items():
            observations.setdefault(name, []).append(value)

    if args.workload == "pipeline":
        decoded = cv2.imdecode(
            np.frombuffer(input_data["encoded"], dtype=np.uint8), cv2.IMREAD_COLOR
        )
        parity_rgb = preprocessor.process(decoded).image_rgb
    else:
        parity_rgb = input_data["rgb"]
    parity = _measure_parity(
        args=args,
        torch=torch,
        device=device,
        base_model=base_model,
        candidate=candidate,
        rgb=parity_rgb,
        tensor_transform=to_unigaze_tensor,
    )

    stages_summary = {
        name: summarize_latencies(values) for name, values in observations.items()
    }
    end_to_end = stages_summary["end_to_end_ms"]
    system_gpu_after: dict[str, Any] | None = None
    telemetry_error: str | None = None
    if device.type == "cuda":
        try:
            system_gpu_after = query_nvidia_smi()
        except Exception as exc:
            telemetry_error = f"{type(exc).__name__}: {exc}"
    resources: dict[str, Any] = {
        "system_gpu_after": system_gpu_after,
        "telemetry_error": telemetry_error,
        "torch_cuda": None,
    }
    if device.type == "cuda":
        resources["torch_cuda"] = {
            "memory_before": memory_before,
            "end_allocated_mib": round(
                torch.cuda.memory_allocated(device) / (1024 * 1024), 3
            ),
            "end_reserved_mib": round(
                torch.cuda.memory_reserved(device) / (1024 * 1024), 3
            ),
            "peak_allocated_mib": round(
                torch.cuda.max_memory_allocated(device) / (1024 * 1024), 3
            ),
            "peak_reserved_mib": round(
                torch.cuda.max_memory_reserved(device) / (1024 * 1024), 3
            ),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if parity["allclose"] else "failed",
        "variant": args.variant,
        "workload": args.workload,
        "device": args.device,
        "revision": _git_revision(),
        "input": input_data["metadata"],
        "environment": _full_environment(torch, device),
        "guard": guard,
        "setup_ms": {
            "input_load": round(input_load_ms, 6),
            "preprocessor_init": round(preprocessor_init_ms, 6),
            "model_load": round(model_load_ms, 6),
            "variant_wrapper": round(variant_setup_ms, 6),
        },
        "first_iteration_ms": {
            name: round(value, 6) for name, value in first_stages.items()
        },
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "latency": stages_summary,
        "throughput_fps": {
            "from_p50": round(1000 / float(end_to_end["p50_ms"]), 6),
            "from_p95": round(1000 / float(end_to_end["p95_ms"]), 6),
        },
        "parity": parity,
        "resources": resources,
        "last_output_shape": list(last_output.shape),
    }


def _base_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "error",
        "variant": args.variant,
        "workload": args.workload,
        "device": args.device,
        "revision": None,
        "environment": _minimal_environment(),
    }


def _failure_summary(
    args: argparse.Namespace,
    exc: Exception,
    *,
    include_traceback: bool,
) -> dict[str, Any]:
    summary = _base_summary(args)
    summary.update({
        "status": "refused" if isinstance(exc, BenchmarkRefused) else "error",
        "failure": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
    })
    if include_traceback:
        summary["failure"]["traceback"] = traceback.format_exc()
    return summary


def _run_worker(args: argparse.Namespace) -> int:
    try:
        _validate_args(args)
        summary = _run_benchmark(args, guard=None)
        exit_code = 0 if summary["status"] == "passed" else 1
    except BenchmarkRefused as exc:
        summary = _failure_summary(args, exc, include_traceback=False)
        exit_code = 2
    except Exception as exc:
        summary = _failure_summary(args, exc, include_traceback=True)
        exit_code = 1
    atomic_write_json(args.summary_path, summary)
    return exit_code


def _worker_command(args: argparse.Namespace, summary_path: Path) -> list[str]:
    command = [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "scripts.benchmark_gaze_inference",
        "--worker",
        "--summary-path",
        str(summary_path),
        "--device",
        args.device,
        "--variant",
        args.variant,
        "--workload",
        args.workload,
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    if args.image is not None:
        command.extend(("--image", str(args.image.resolve())))
    if args.max_abs_error is not None:
        command.extend(("--max-abs-error", str(args.max_abs_error)))
    if args.max_relative_error is not None:
        command.extend(("--max-relative-error", str(args.max_relative_error)))
    if args.allow_model_download:
        command.append("--allow-model-download")
    return command


def _stop_process_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    else:
        os.killpg(process.pid, 9)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()


def _run_supervisor(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    _validate_args(args)
    guard = None
    if args.device == "cuda":
        guard = gpu_preflight(
            sample_count=args.guard_samples,
            interval_seconds=args.guard_interval_ms / 1000,
            max_utilization_percent=args.max_background_gpu_utilization,
            max_memory_mib=args.max_background_gpu_memory_mib,
            max_temperature_c=args.max_start_temperature_c,
            allow_busy=args.allow_busy_gpu,
        )
        if not guard["allowed"]:
            summary = _base_summary(args)
            summary.update({
                "status": "refused",
                "guard": guard,
                "failure": {
                    "type": "BusyGpu",
                    "message": "; ".join(guard["reasons"]),
                },
            })
            return summary, 2

    with tempfile.TemporaryDirectory(prefix="lexigaze-gaze-benchmark-") as name:
        summary_path = Path(name) / "worker-summary.json"
        command = _worker_command(args, summary_path)
        environment = os.environ.copy()
        environment.setdefault("PYTHONFAULTHANDLER", "1")
        if args.device == "cpu":
            environment["CUDA_VISIBLE_DEVICES"] = "-1"
        process_options: dict[str, Any] = {
            "cwd": ROOT,
            "env": environment,
        }
        if os.name == "nt":
            process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            process_options["start_new_session"] = True

        started = time.perf_counter()
        process = subprocess.Popen(command, **process_options)
        timed_out = False
        try:
            worker_exit = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop_process_tree(process)
            worker_exit = 124
        duration = time.perf_counter() - started

        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = _base_summary(args)
            summary.update({
                "status": "timeout" if timed_out else "error",
                "failure": {
                    "type": "Timeout" if timed_out else "MissingWorkerSummary",
                    "message": (
                        f"Benchmark exceeded {args.timeout_seconds} seconds"
                        if timed_out
                        else "Benchmark worker did not produce a summary"
                    ),
                },
            })
        summary["guard"] = guard
        summary["supervisor"] = {
            "duration_seconds": round(duration, 3),
            "timed_out": timed_out,
            "timeout_seconds": args.timeout_seconds,
            "worker_exit": worker_exit,
        }
        expected_exit = {
            "passed": 0,
            "refused": 2,
        }.get(summary.get("status"), 1)
        exit_code = 124 if timed_out else expected_exit
        return summary, exit_code


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.worker:
        return _run_worker(args)

    try:
        summary, exit_code = _run_supervisor(args)
    except BenchmarkRefused as exc:
        summary = _failure_summary(args, exc, include_traceback=False)
        exit_code = 2
    except Exception as exc:
        summary = _failure_summary(args, exc, include_traceback=True)
        exit_code = 1

    if args.json_output:
        atomic_write_json(args.json_output, summary)
    print("GAZE_BENCHMARK_RESULT=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

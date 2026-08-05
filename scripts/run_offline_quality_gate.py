"""Run LexiGaze's fast regression lane without GPU, secrets, or network."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIMEOUT_SECONDS = 180

# Explicit IDs keep heavyweight/model tests out of this lane. In particular,
# cross-attention imports PyTorch and belongs in the separately managed heavy
# suite; fatigue-adaptive fusion exercises the same Flask route without it.
TEST_TARGETS = (
    "scripts.test_app_factory",
    "scripts.test_gaze_benchmark",
    "scripts.test_gaze_calibration_regression",
    "scripts.test_gaze_motion_experiment",
    "scripts.test_gaze_motion_robustness",
    "scripts.test_gaze_session_independence",
    "scripts.test_gaze_torch_runtime",
    "scripts.test_adaptive_stepper",
    "scripts.test_cognitive_policy",
    "scripts.test_cognitive_inspector",
    "scripts.test_onestop_confirmation_preparation",
    "scripts.test_fusion_routes.TestFusionRoutes.test_fatigue_adaptive_method",
)

PROVIDER_CREDENTIALS = (
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "OPENAI_API_KEY",
)

ARTIFACT_SCOPES = (
    # Session JSON files live directly under data/. Gaze frame corpora can
    # contain hundreds of thousands of unrelated files and are not touched by
    # this lane, so scanning them would make the fast gate slower than tests.
    (ROOT / "data", "*.json"),
    (ROOT / "docs" / "cognitive_reports", "**/*"),
    (ROOT / "docs" / "fusion_reports", "**/*"),
    (ROOT / "models", "**/*"),
    (ROOT / "output", "**/*"),
)


class OfflineNetworkAttempt(RuntimeError):
    """Raised whenever code in the quality gate attempts socket I/O."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Maximum worker runtime before its complete process tree is stopped.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optionally persist the final machine-readable summary.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--summary-path", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _worker_environment(temp_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    for name in PROVIDER_CREDENTIALS:
        # Keep an explicit empty value so python-dotenv cannot repopulate a key
        # from a developer's local .env file.
        env[name] = ""

    env.update({
        "CUDA_VISIBLE_DEVICES": "-1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": str(temp_root / "huggingface"),
        "HF_HUB_OFFLINE": "1",
        "LEXIGAZE_QUALITY_GATE": "1",
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NO_PROXY": "*",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "no_proxy": "*",
    })
    return env


def _gpu_snapshot() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False}

    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": type(exc).__name__}

    return {
        "available": completed.returncode == 0,
        "rows": [line.strip() for line in completed.stdout.splitlines() if line.strip()],
    }


def _snapshot_artifacts() -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for root, pattern in ARTIFACT_SCOPES:
        if not root.exists():
            continue
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            snapshot[path.relative_to(ROOT).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def _artifact_changes(
    before: dict[str, tuple[int, int]],
    after: dict[str, tuple[int, int]],
) -> list[str]:
    keys = sorted(set(before) | set(after))
    return [key for key in keys if before.get(key) != after.get(key)]


@contextlib.contextmanager
def _deny_network() -> Iterator[list[str]]:
    attempts: list[str] = []

    def blocked_network(*args: Any, **kwargs: Any) -> Any:
        address = args[-1] if args else kwargs.get("address", "unknown")
        attempts.append(repr(address))
        raise OfflineNetworkAttempt(f"network disabled by quality gate: {address!r}")

    def blocked_process(*args: Any, **kwargs: Any) -> Any:
        attempts.append("<subprocess>")
        raise OfflineNetworkAttempt("process spawning disabled by quality gate")

    original_popen = subprocess.Popen

    class BlockedPopen(original_popen):
        """Remain subclassable for asyncio imports, but never start a process."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            blocked_process(*args, **kwargs)

    replacements = (
        (socket, "create_connection", blocked_network),
        (socket, "getaddrinfo", blocked_network),
        (socket, "gethostbyname", blocked_network),
        (socket, "gethostbyname_ex", blocked_network),
        (socket.socket, "connect", blocked_network),
        (socket.socket, "connect_ex", blocked_network),
        (socket.socket, "sendto", blocked_network),
        (subprocess, "call", blocked_process),
        (subprocess, "check_call", blocked_process),
        (subprocess, "check_output", blocked_process),
        (subprocess, "Popen", BlockedPopen),
        (subprocess, "run", blocked_process),
    )
    originals = [(owner, name, getattr(owner, name)) for owner, name, _ in replacements]
    for owner, name, replacement in replacements:
        setattr(owner, name, replacement)
    try:
        yield attempts
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run_worker(summary_path: Path) -> int:
    started = time.perf_counter()
    before = _snapshot_artifacts()
    # On Windows, platform.uname() may invoke the local `ver` command once.
    # Warm that standard-library cache before process spawning is denied so
    # CPU packages such as pandas can inspect the cached architecture safely.
    platform.uname()
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "error",
        "test_targets": list(TEST_TARGETS),
    }

    try:
        with _deny_network() as network_attempts:
            try:
                socket.create_connection(("example.invalid", 443), timeout=0.01)
            except OfflineNetworkAttempt:
                network_probe_blocked = True
            else:  # pragma: no cover - the guard itself would be broken
                network_probe_blocked = False
            network_attempts.clear()

            try:
                subprocess.run(["quality-gate-probe"], check=False)
            except OfflineNetworkAttempt:
                process_probe_blocked = True
            else:  # pragma: no cover - the guard itself would be broken
                process_probe_blocked = False
            network_attempts.clear()

            suite = unittest.defaultTestLoader.loadTestsFromNames(TEST_TARGETS)
            result = unittest.TextTestRunner(verbosity=2).run(suite)

        after = _snapshot_artifacts()
        artifact_changes = _artifact_changes(before, after)
        torch_imported = "torch" in sys.modules
        safeguards_ok = (
            network_probe_blocked
            and process_probe_blocked
            and not network_attempts
            and not artifact_changes
            and not torch_imported
            and all(os.environ.get(name, "") == "" for name in PROVIDER_CREDENTIALS)
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
        )
        passed = result.wasSuccessful() and safeguards_ok
        summary.update({
            "status": "passed" if passed else "failed",
            "duration_seconds": round(time.perf_counter() - started, 3),
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
            "unexpected_successes": len(result.unexpectedSuccesses),
            "safeguards": {
                "artifact_changes": artifact_changes,
                "credentials_cleared": all(
                    os.environ.get(name, "") == "" for name in PROVIDER_CREDENTIALS
                ),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "network_attempts": network_attempts,
                "network_probe_blocked": network_probe_blocked,
                "process_probe_blocked": process_probe_blocked,
                "torch_imported": torch_imported,
            },
        })
        return_code = 0 if passed else 1
    except Exception as exc:
        summary.update({
            "status": "error",
            "duration_seconds": round(time.perf_counter() - started, 3),
            "exception": type(exc).__name__,
            "traceback": traceback.format_exc(),
        })
        return_code = 1

    _atomic_write_json(summary_path, summary)
    return return_code


def _stop_process_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    else:
        os.killpg(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def _run_supervisor(args: argparse.Namespace) -> int:
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")

    with tempfile.TemporaryDirectory(prefix="lexigaze-quality-gate-") as temp_name:
        temp_root = Path(temp_name)
        summary_path = temp_root / "summary.json"
        command = [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "scripts.run_offline_quality_gate",
            "--worker",
            "--summary-path",
            str(summary_path),
        ]
        popen_options: dict[str, Any] = {
            "cwd": ROOT,
            "env": _worker_environment(temp_root),
        }
        if os.name == "nt":
            popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_options["start_new_session"] = True

        gpu_before = _gpu_snapshot()
        started = time.perf_counter()
        process = subprocess.Popen(command, **popen_options)
        timed_out = False
        try:
            worker_exit = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop_process_tree(process)
            worker_exit = 124
        elapsed = round(time.perf_counter() - started, 3)
        gpu_after = _gpu_snapshot()

        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = {
                "schema_version": 1,
                "status": "timeout" if timed_out else "error",
                "tests_run": 0,
            }
        summary["supervisor"] = {
            "duration_seconds": elapsed,
            "gpu_after": gpu_after,
            "gpu_before": gpu_before,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "timed_out": timed_out,
            "timeout_seconds": args.timeout_seconds,
            "worker_exit": worker_exit,
        }

        if args.json_output:
            _atomic_write_json(args.json_output.resolve(), summary)
        print("QUALITY_GATE_RESULT=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))

        return 0 if worker_exit == 0 and summary.get("status") == "passed" else 1


def main() -> int:
    args = _parse_args()
    if args.worker:
        if args.summary_path is None:
            raise SystemExit("--summary-path is required for worker mode")
        return _run_worker(args.summary_path)
    return _run_supervisor(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the dedicated loopback-only webcam measurement-ceiling surface."""

from __future__ import annotations

import os

# These process-wide guards must be set before importing Flask routes or any
# model runtime.  The measurement protocol is intentionally CPU-only/offline.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse
import sys
import webbrowser
from collections.abc import Sequence
from pathlib import Path

from core.gaze_core.base_inference_bundle import build_base_inference_bundle
from core.gaze_core.measurement_preflight import MeasurementPreflightRegistry
from core.gaze_core.measurement_runner import MeasurementRunner
from web import create_app


DEFAULT_PORT = 8099


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Dedicated 127.0.0.1 port (default: 8099).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help=(
            "Repository/data root for the dedicated run registry and temporary "
            "calibration session. Defaults to the primary checkout on D:."
        ),
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the measurement page automatically.",
    )
    return parser.parse_args(argv)


def _primary_checkout_root(code_root: Path) -> Path:
    """Resolve a linked worktree back to its primary checkout when possible."""

    root = code_root.resolve()
    for ancestor in root.parents:
        if ancestor.name == "workspaces" and ancestor.parent.name == ".tmp":
            candidate = ancestor.parent.parent.resolve()
            if (candidate / "core" / "gaze_core").is_dir():
                return candidate
    marker = root / ".git"
    if marker.is_dir() or not marker.is_file():
        return root
    try:
        first_line = marker.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, UnicodeError, IndexError):
        return root
    prefix = "gitdir:"
    if not first_line.lower().startswith(prefix):
        return root
    raw = first_line[len(prefix) :].strip()
    git_dir = Path(raw)
    if not git_dir.is_absolute():
        git_dir = (root / git_dir).resolve()
    else:
        git_dir = git_dir.resolve()
    # <primary>/.git/worktrees/<name>
    if git_dir.parent.name == "worktrees" and git_dir.parent.parent.name == ".git":
        return git_dir.parent.parent.parent.resolve()
    return root


def _validated_port(value: int) -> int:
    if isinstance(value, bool) or not 1 <= value <= 65535:
        raise ValueError("--port must be an integer from 1 through 65535")
    return value


def _validated_data_root(value: Path) -> Path:
    root = value.resolve()
    if root.drive.upper() == "C:":
        raise ValueError(
            "measurement data root may not use C:; choose the repository on D:"
        )
    if not root.is_dir():
        raise ValueError(f"measurement data root does not exist: {root}")
    return root


def build_measurement_app(*, code_root: Path, data_root: Path, port: int):
    """Build the isolated app without starting a listener or browser."""

    bundle = build_base_inference_bundle(
        repository_root=code_root,
        local_files_only=True,
    )
    runner = MeasurementRunner(data_root, code_root=code_root)
    preflight = MeasurementPreflightRegistry(
        data_root,
        base_inference_bundle=bundle,
    )
    return create_app(
        {
            "LEXIGAZE_BLUEPRINTS": ("measurement",),
            "LEXIGAZE_MEASUREMENT_CEILING_MODE": True,
            "LEXIGAZE_MEASUREMENT_AUTHORITY": f"127.0.0.1:{port}",
            "LEXIGAZE_MEASUREMENT_RUNNER": runner,
            "LEXIGAZE_MEASUREMENT_PREFLIGHT": preflight,
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        port = _validated_port(args.port)
        code_root = Path(__file__).resolve().parents[1]
        data_root = _validated_data_root(
            args.data_root or _primary_checkout_root(code_root)
        )
        app = build_measurement_app(
            code_root=code_root,
            data_root=data_root,
            port=port,
        )
        from waitress import serve
    except Exception as exc:
        print(f"measurement server startup failed: {exc}", file=sys.stderr)
        return 2

    url = f"http://127.0.0.1:{port}/measurement-ceiling"
    print("LexiGaze dedicated measurement-ceiling surface")
    print(f"URL: {url}")
    print(f"Code root: {code_root}")
    print(f"Data root: {data_root}")
    print("Runtime: loopback-only, offline, CPU-only (CUDA_VISIBLE_DEVICES=-1)")
    print(
        "Calibration imagery and encrypted crash spool are temporary dedicated "
        "run data; ledger/final artifacts contain no images."
    )
    if not args.no_browser:
        webbrowser.open(url, new=2)
    serve(app, host="127.0.0.1", port=port, threads=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

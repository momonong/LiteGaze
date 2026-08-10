"""Analyze one verified 193-row webcam gaze acquisition artifact, CPU-only."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gaze_core.measurement_run_analysis import (  # noqa: E402
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    analyze_measurement_run,
    render_measurement_run_markdown,
)
from core.gaze_core.measurement_schedule import (  # noqa: E402
    MeasurementScheduleError,
    deterministic_json,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-artifact",
        type=Path,
        required=True,
        help="Path to the sealed 193-row capture_artifact.json.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        required=True,
        help="Destination for deterministic machine-readable analysis.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        required=True,
        help="Destination for the compact human-readable report.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
        help="Use 20000 for the frozen analysis; other values are diagnostic only.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="Use 20260810 for the frozen analysis.",
    )
    return parser.parse_args(argv)


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MeasurementScheduleError(
            f"capture artifact is unavailable or invalid: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise MeasurementScheduleError("capture artifact must be a JSON object")
    return payload


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    source = args.capture_artifact.resolve()
    json_output = args.json_output.resolve()
    markdown_output = args.markdown_output.resolve()
    if len({source, json_output, markdown_output}) != 3:
        print(
            "capture input and output paths must be distinct",
            file=sys.stderr,
        )
        return 2
    try:
        result = analyze_measurement_run(
            _load_object(source),
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
        json_bytes = deterministic_json(result).encode("utf-8")
        markdown_bytes = render_measurement_run_markdown(result).encode("utf-8")
        _atomic_write(json_output, json_bytes)
        _atomic_write(markdown_output, markdown_bytes)
    except (MeasurementScheduleError, OSError, ValueError) as exc:
        print(f"measurement analysis failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "status": result["status"],
                "analysis_sha256": result["analysis_sha256"],
                "measurement_claim_authorized": False,
                "json_output": str(json_output),
                "markdown_output": str(markdown_output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

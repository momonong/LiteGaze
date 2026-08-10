"""Generate or verify the frozen 193-row webcam-gaze acquisition manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from core.gaze_core.measurement_schedule import (
    DEFAULT_PROTOCOL_PATH,
    MeasurementScheduleError,
    build_run_manifest,
    deterministic_json,
    verify_capture_artifact,
    verify_run_manifest,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--create-manifest",
        type=Path,
        metavar="PATH",
        help="Atomically create and reverify one scheduled-uncollected manifest.",
    )
    mode.add_argument(
        "--verify-manifest",
        type=Path,
        metavar="PATH",
        help="Verify an existing manifest against the frozen protocol.",
    )
    mode.add_argument(
        "--verify-capture-artifact",
        type=Path,
        metavar="PATH",
        help=(
            "Verify one complete 193-row capture artifact; this checks only the "
            "acquisition contract and never authorizes a measurement claim."
        ),
    )
    parser.add_argument(
        "--capture-run-id",
        help="Required with --create-manifest; becomes part of every shuffle seed.",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL_PATH,
        help=(
            "Frozen v1 protocol path. Any canonical SHA-256 mutation fails closed."
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optionally persist the verification summary as strict JSON.",
    )
    return parser.parse_args(argv)


def _atomic_write(path: Path, text: str) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(target)


def _load_json_object(path: Path, *, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MeasurementScheduleError(
            f"unable to load {label}: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise MeasurementScheduleError(f"{label} must be a JSON object")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        if args.create_manifest is not None:
            if not args.capture_run_id:
                raise MeasurementScheduleError(
                    "--capture-run-id is required with --create-manifest"
                )
            manifest = build_run_manifest(
                args.capture_run_id,
                protocol_path=args.protocol,
            )
            _atomic_write(args.create_manifest, deterministic_json(manifest))
            persisted = _load_json_object(
                args.create_manifest.resolve(), label="run manifest"
            )
            summary = verify_run_manifest(
                persisted,
                protocol_path=args.protocol,
            )
            mode = "create"
            input_path = args.create_manifest.resolve()
        elif args.verify_manifest is not None:
            if args.capture_run_id:
                raise MeasurementScheduleError(
                    "--capture-run-id is only valid with --create-manifest"
                )
            persisted = _load_json_object(
                args.verify_manifest.resolve(), label="run manifest"
            )
            summary = verify_run_manifest(
                persisted,
                protocol_path=args.protocol,
            )
            mode = "verify_manifest"
            input_path = args.verify_manifest.resolve()
        else:
            if args.capture_run_id:
                raise MeasurementScheduleError(
                    "--capture-run-id is only valid with --create-manifest"
                )
            persisted = _load_json_object(
                args.verify_capture_artifact.resolve(), label="capture artifact"
            )
            summary = verify_capture_artifact(
                persisted,
                protocol_path=args.protocol,
            )
            mode = "verify_capture_artifact"
            input_path = args.verify_capture_artifact.resolve()
        summary = {
            "schema_version": 1,
            "status": "passed",
            "mode": mode,
            "cpu_only": True,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "network_used": False,
            "torch_imported": "torch" in sys.modules,
            "input_path": str(input_path),
            **summary,
        }
        if args.summary_output is not None:
            _atomic_write(args.summary_output, deterministic_json(summary))
        print(
            "WEBCAM_GAZE_ACQUISITION_DRY_RUN="
            f"status=passed mode={mode} "
            f"rows={summary.get('total_sample_count', summary.get('sample_count'))} "
            f"protocol_sha256={summary['protocol_sha256']} "
            "measurement_claim_authorized=false "
            f"manifest_sha256={summary.get('manifest_sha256', summary.get('run_manifest_sha256'))}"
        )
        return 0
    except MeasurementScheduleError as exc:
        print(
            f"WEBCAM_GAZE_ACQUISITION_DRY_RUN=status=failed reason={exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

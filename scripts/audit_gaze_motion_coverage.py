"""Audit calibration-motion coverage without reading images or loading a model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from core.gaze_core.motion_robustness import audit_payload, load_motion_samples

ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=ROOT / "data" / "sessions",
        help="Directory containing session folders with manifest.jsonl files.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optionally write the aggregate-only result atomically.",
    )
    parser.add_argument(
        "--session-id",
        action="append",
        default=None,
        help="Audit only this session ID; repeat to include multiple sessions.",
    )
    parser.add_argument(
        "--fail-on-not-ready",
        action="store_true",
        help="Return exit code 1 when a frozen coverage requirement is unmet.",
    )
    return parser.parse_args()


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = _parse_args()
    samples, diagnostics = load_motion_samples(
        args.sessions_dir,
        session_ids=args.session_id,
    )
    payload = audit_payload(samples, diagnostics)
    if args.json_output:
        _atomic_write(args.json_output.resolve(), payload)
    print("GAZE_MOTION_AUDIT=" + json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if args.fail_on_not_ready and payload["status"] != "ready":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

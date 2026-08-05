"""Audit whether gaze sessions represent independent physical capture runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.gaze_core.session_independence import (
    IndependenceRequirements,
    audit_capture_independence,
    load_capture_sessions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "sessions",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--min-repeat-labels", type=int, default=5)
    parser.add_argument("--min-independent-capture-runs", type=int, default=10)
    parser.add_argument(
        "--legacy-independence-gap-hours",
        type=float,
        default=24.0,
    )
    parser.add_argument("--fail-on-not-ready", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requirements = IndependenceRequirements(
        min_repeat_labels=args.min_repeat_labels,
        min_independent_capture_runs=args.min_independent_capture_runs,
        legacy_independence_gap_seconds=(
            args.legacy_independence_gap_hours * 60.0 * 60.0
        ),
    )
    sessions, diagnostics = load_capture_sessions(args.sessions_root)
    payload = audit_capture_independence(
        sessions,
        diagnostics,
        requirements=requirements,
    )
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    print(f"GAZE_SESSION_INDEPENDENCE_AUDIT={rendered}")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return int(args.fail_on_not_ready and payload["status"] != "ready")


if __name__ == "__main__":
    raise SystemExit(main())

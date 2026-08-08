"""Audit an outcome-blind independent capture plan without model or media access."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from core.participant_study.independent_capture import (
    AUDIT_TARGETS,
    audit_independent_capture_plan,
    load_capture_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument(
        "--target",
        choices=AUDIT_TARGETS,
        default="template",
        help="Readiness level whose result determines the exit code.",
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    try:
        plan = load_capture_plan(args.plan)
        result = audit_independent_capture_plan(plan, target=args.target)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        result = {
            "schema_version": 1,
            "target": args.target,
            "status": "not_ready",
            "plan_sha256": None,
            "summary": {},
            "issues": [
                {
                    "code": "PLAN_LOAD_FAILED",
                    "severity": "error",
                    "message": f"The plan could not be loaded: {type(exc).__name__}.",
                }
            ],
            "warning": (
                "Engineering readiness is not ethics approval, recruitment "
                "authorization, or evidence of model effectiveness."
            ),
        }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output:
        _atomic_json(args.json_output.resolve(), result)
    return 0 if result["status"] != "not_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())

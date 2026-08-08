"""Audit the fixed-form rehearsal without opening images or using a GPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.participant_study import ParticipantStudyStore
from core.participant_study.general_collection import (
    load_general_bank,
    load_general_protocol,
    validate_general_design,
)


def build_audit(root: Path, settings: dict[str, object]) -> dict[str, object]:
    protocol = load_general_protocol()
    bank = load_general_bank()
    design = validate_general_design(protocol, bank)
    store = ParticipantStudyStore(root, settings=settings)
    rehearsal_root = store._study_root("rehearsal")  # noqa: SLF001 - audit only
    sessions = list(rehearsal_root.glob("ST-*/session.json"))
    started = 0
    completed = 0
    mismatched: list[str] = []
    for path in sessions:
        try:
            session = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            mismatched.append(path.parent.name)
            continue
        collection = dict(session.get("general_collection") or {})
        if collection.get("assessment_id"):
            started += 1
        if session.get("state") == "completed":
            completed += 1
        assignment = dict(session.get("collection_assignment") or {})
        if assignment and (
            assignment.get("protocol_sha256") != design["protocol_sha256"]
            or assignment.get("bank_sha256") != design["bank_sha256"]
        ):
            mismatched.append(str(session.get("study_session_id")))
    software_ready = bool(
        design["ok"]
        and store.activation.get("rehearsal_ready")
        and not mismatched
    )
    review = dict(bank["review"])
    formal_blockers = list(protocol["formal_collection_blockers"])
    return {
        "ok": True,
        "audit_type": "general_collection_readiness_v1",
        "design": design,
        "rehearsal": {
            "software_ready": software_ready,
            "activation_ready": store.activation.get("rehearsal_ready") is True,
            "missing_requirements": store.activation.get(
                "rehearsal_missing_requirements", []
            ),
            "scope": store.activation.get("rehearsal_scope"),
            "existing_session_count": len(sessions),
            "started_session_count": started,
            "completed_session_count": completed,
            "digest_mismatch_sessions": sorted(mismatched),
        },
        "formal_collection": {
            "ready": False,
            "bank_human_reviews": {
                "required": review["required_independent_reviewers"],
                "completed": review["completed_independent_reviewers"],
            },
            "blockers": formal_blockers,
        },
        "compute": {
            "audit_device": "cpu",
            "gpu_used": False,
            "language_model_inference_used": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--require-rehearsal-ready", action="store_true")
    args = parser.parse_args()
    audit = build_audit(args.root.resolve(), dict(os.environ))
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    if args.require_rehearsal_ready and not audit["rehearsal"]["software_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

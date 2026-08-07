"""Audit LexiGaze participant-study readiness without using GPU or network."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.cognitive_inspector.adaptive import validate_item_bank  # noqa: E402
from core.participant_study.protocol import (  # noqa: E402
    activation_status,
    load_protocol,
    public_protocol,
)

REQUIRED_FILES = (
    "core/participant_study/protocol_v1.json",
    "web/routes/study.py",
    "web/templates/participant_study.html",
    "web/templates/participant_assessment.html",
    "web/static/participant_study.js",
    "web/static/participant_assessment.js",
    "docs/participant_study/PROTOCOL.md",
    "docs/participant_study/CONSENT_DRAFT.md",
    "docs/participant_study/RESEARCHER_RUNBOOK.md",
    "docs/participant_study/DATA_MANAGEMENT_PLAN.md",
    "docs/participant_study/INCIDENT_RESPONSE.md",
    "docs/participant_study/READINESS_CHECKLIST.md",
)


def audit() -> dict:
    protocol = load_protocol()
    activation = activation_status(os.environ, protocol=protocol)
    public = public_protocol(os.environ, protocol=protocol)
    bank = validate_item_bank()
    missing_files = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    contacts = public["research_contacts"]
    governance = public["data_governance"]
    engineering_checks = {
        "required_files_present": not missing_files,
        "item_bank_structure_valid": bank["ok"],
        "answer_keys_absent_from_public_passages": "answer_key_leakage"
        not in bank["errors"],
        "fixed_layout_during_ability_estimation": bank["layout"]
        == {"font_size": 16, "line_width": 650, "line_height": 1.7},
        "dry_run_available_without_camera_or_storage": activation["dry_run_ready"],
        "direct_identifiers_excluded_by_contract": True,
        "full_video_collection_disabled": not protocol["optional_scopes"],
        "participant_contacts_configured": all(contacts.values()),
        "retention_and_location_configured": bool(governance["location"])
        and bool(governance["retention_days"])
        and bool(governance["raw_frame_retention_hours"]),
    }
    dry_run_ready = all(
        engineering_checks[key]
        for key in (
            "required_files_present",
            "item_bank_structure_valid",
            "answer_keys_absent_from_public_passages",
            "fixed_layout_during_ability_estimation",
            "dry_run_available_without_camera_or_storage",
            "direct_identifiers_excluded_by_contract",
            "full_video_collection_disabled",
        )
    )
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_version": protocol["protocol_version"],
        "protocol_digest_sha256": activation["protocol_digest_sha256"],
        "consent_digest_sha256": public["consent_digest_sha256"],
        "dry_run_ready": dry_run_ready,
        "pilot_ready": dry_run_ready and activation["pilot_ready"],
        "pilot_missing_requirements": activation["missing_requirements"],
        "engineering_checks": engineering_checks,
        "missing_files": missing_files,
        "item_bank": bank,
        "automated_test_command": (
            r".\.venv\Scripts\python.exe -m unittest "
            "scripts.test_participant_study scripts.test_adaptive_stepper -v"
        ),
        "warning": (
            "Engineering readiness does not replace institutional ethics review, "
            "legal advice, or authorization to recruit participants."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("dry-run", "pilot"),
        default="dry-run",
        help="Readiness level whose status determines the exit code.",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = audit()
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output:
        output = args.json_output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        os.replace(temporary, output)
    ready = result["pilot_ready"] if args.target == "pilot" else result["dry_run_ready"]
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())

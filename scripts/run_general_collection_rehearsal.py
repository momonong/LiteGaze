"""Run an invite-only localhost rehearsal after explicit privacy confirmations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.participant_study import ParticipantStudyStore
from web import create_app


def resolve_data_location(root: Path, declared: Path) -> Path:
    """Require the governance label to name the directory actually written."""
    actual = (root.resolve() / "data").resolve()
    if declared.resolve() != actual:
        raise ValueError(
            "--data-location must exactly match the actual storage directory: "
            f"{actual}"
        )
    return actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "localhost"))
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--data-location", type=Path, required=True)
    parser.add_argument("--retention-days", type=int, default=7)
    parser.add_argument("--raw-frame-retention-hours", type=int, default=1)
    parser.add_argument("--create-invite-pairs", type=int, default=0)
    parser.add_argument("--acknowledge-development-only", action="store_true")
    storage = parser.add_mutually_exclusive_group(required=True)
    storage.add_argument("--confirm-encrypted-storage", action="store_true")
    storage.add_argument(
        "--allow-unencrypted-self-development-data",
        action="store_true",
    )
    parser.add_argument(
        "--retain-until-manual-deletion",
        action="store_true",
    )
    args = parser.parse_args()
    if not args.acknowledge_development_only:
        parser.error("--acknowledge-development-only is required")
    if args.confirm_encrypted_storage:
        if args.retain_until_manual_deletion:
            parser.error(
                "--retain-until-manual-deletion is only for the explicit "
                "unencrypted self-development mode"
            )
        if not 1 <= args.retention_days <= 30:
            parser.error("--retention-days must be between 1 and 30")
        retention_days = args.retention_days
        retention_policy = "fixed_days"
        storage_encrypted = "1"
        unencrypted_self_development = "0"
    else:
        if not args.retain_until_manual_deletion:
            parser.error(
                "--retain-until-manual-deletion is required for unencrypted "
                "self-development data"
            )
        if args.create_invite_pairs > 1:
            parser.error(
                "unencrypted self-development mode permits only one invite pair"
            )
        retention_days = 0
        retention_policy = "manual_until_researcher_deletes"
        storage_encrypted = "0"
        unencrypted_self_development = "1"
    if not 1 <= args.raw_frame_retention_hours <= 24:
        parser.error("--raw-frame-retention-hours must be between 1 and 24")
    if not 0 <= args.create_invite_pairs <= 100:
        parser.error("--create-invite-pairs must be between 0 and 100")

    root = args.root.resolve()
    try:
        data_location = resolve_data_location(root, args.data_location)
    except ValueError as exc:
        parser.error(str(exc))
    base_url = f"http://{args.host}:{args.port}"
    config: dict[str, object] = {
        "LEXIGAZE_STUDY_ROOT": str(root),
        "LEXIGAZE_GAZE_ROOT": str(root),
        "LEXIGAZE_STUDY_MODE": "rehearsal",
        "LEXIGAZE_STUDY_REHEARSAL_MODE": "1",
        "LEXIGAZE_REHEARSAL_ACKNOWLEDGED_DEVELOPMENT_ONLY": "1",
        "LEXIGAZE_REHEARSAL_INVITES_ONLY": "1",
        "LEXIGAZE_REQUEST_BODY_LOGGING_DISABLED": "1",
        "LEXIGAZE_STORAGE_ENCRYPTED": storage_encrypted,
        "LEXIGAZE_UNENCRYPTED_SELF_DEVELOPMENT": unencrypted_self_development,
        "LEXIGAZE_DATA_LOCATION": str(data_location),
        "LEXIGAZE_PUBLIC_BASE_URL": base_url,
        "LEXIGAZE_DATA_RETENTION_DAYS": str(retention_days),
        "LEXIGAZE_DATA_RETENTION_POLICY": retention_policy,
        "LEXIGAZE_RAW_FRAME_RETENTION_HOURS": str(args.raw_frame_retention_hours),
        "LEXIGAZE_PUBLIC_STUDY_MODE": "1",
    }
    store = ParticipantStudyStore(root, settings=config)
    if not store.activation.get("rehearsal_ready"):
        print(
            json.dumps(
                {
                    "ok": False,
                    "missing_requirements": store.activation.get(
                        "rehearsal_missing_requirements", []
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    if args.create_invite_pairs:
        pairs = store.create_collection_invite_pairs(args.create_invite_pairs)
        print(
            json.dumps(
                {
                    "ok": True,
                    "plaintext_invitation_codes_shown_once": True,
                    "pairs": pairs,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    print(
        f"General collection rehearsal: {base_url}/study\n"
        f"Scope: {store.activation['rehearsal_scope']}; "
        "no formal participant collection or confirmation promotion.",
        flush=True,
    )
    create_app(config).run(
        host=args.host,
        port=args.port,
        debug=False,
        use_reloader=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

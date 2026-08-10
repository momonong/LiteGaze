"""Rotate one lost, unused rehearsal invite; plaintext prints once."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.participant_study import ParticipantStudyStore, StudyError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--visit-index", required=True, type=int, choices=(1, 2))
    args = parser.parse_args()
    try:
        rotated = ParticipantStudyStore(
            args.root.resolve(),
            settings=dict(os.environ),
        ).rotate_unused_collection_invite(args.pair_id, args.visit_index)
    except StudyError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "warning": (
                    "The previous unused code is now invalid. This replacement "
                    "code is shown once; keep it outside the tracked repository."
                ),
                "invitation": rotated,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Create one-time local rehearsal invitation pairs; plaintext codes print once."""

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
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        pairs = ParticipantStudyStore(
            args.root.resolve(),
            settings=dict(os.environ),
        ).create_collection_invite_pairs(args.count)
    except StudyError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "warning": (
                    "These plaintext invitation codes are shown once. Store them "
                    "outside the tracked repository and send each code privately."
                ),
                "pairs": pairs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

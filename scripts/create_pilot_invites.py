"""Create one-time pilot invite codes only after every activation gate passes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.participant_study import ParticipantStudyStore, StudyError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    try:
        codes = ParticipantStudyStore(ROOT).create_invites(args.count)
    except StudyError as exc:
        print(f"Pilot invite creation refused: {exc}", file=sys.stderr)
        return 1
    print("Each code is shown once. Distribute it privately; do not commit it.")
    for code in codes:
        print(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

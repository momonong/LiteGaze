"""CPU-only tests for capture-run independence auditing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.gaze_core.session_independence import (
    IndependenceRequirements,
    audit_capture_independence,
    load_capture_sessions,
)


def _write_session(
    root: Path,
    *,
    session_id: str,
    participant_id: str,
    timestamp: float,
    capture_run_id: str | None = None,
    capture_source: str | None = None,
    source_session_id: str | None = None,
    video: bool = False,
) -> None:
    session_dir = root / session_id
    session_dir.mkdir(parents=True)
    meta: dict[str, object] = {
        "session_id": session_id,
        "participant_id": participant_id,
    }
    if capture_run_id:
        meta["capture_run_id"] = capture_run_id
    if capture_source:
        meta["capture_source"] = capture_source
    if source_session_id:
        meta["source_session_id"] = source_session_id
    (session_dir / "session.json").write_text(
        json.dumps(meta),
        encoding="utf-8",
    )
    record = {
        "created_at_unix": timestamp,
        "capture_run_id": capture_run_id,
        "capture_source": capture_source,
        "source_session_id": source_session_id,
        "extracted_from_video": video,
    }
    (session_dir / "manifest.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )
    if video:
        (session_dir / "raw_video.webm").write_bytes(b"test")


class CaptureIndependenceTests(unittest.TestCase):
    def test_direct_and_video_artifacts_share_one_capture_group(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-capture-audit-") as name:
            root = Path(name)
            _write_session(
                root,
                session_id="direct",
                participant_id="private-label",
                timestamp=100.0,
                capture_run_id="capture-a",
                capture_source="direct-frame",
            )
            _write_session(
                root,
                session_id="video",
                participant_id="private-label",
                timestamp=120.0,
                capture_run_id="capture-a",
                capture_source="video-extracted",
                source_session_id="direct",
                video=True,
            )
            _write_session(
                root,
                session_id="confirmation",
                participant_id="private-label",
                timestamp=200.0,
                capture_run_id="capture-b",
                capture_source="direct-frame",
            )
            sessions, diagnostics = load_capture_sessions(root)
            payload = audit_capture_independence(
                sessions,
                diagnostics,
                requirements=IndependenceRequirements(1, 2),
            )

        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["summary"]["capture_groups"], 2)
        self.assertEqual(payload["summary"]["shared_capture_groups"], 1)
        self.assertEqual(payload["summary"]["repeat_labels"], 1)
        self.assertNotIn("private-label", json.dumps(payload))

    def test_legacy_sessions_inside_one_day_are_conservatively_linked(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-capture-audit-") as name:
            root = Path(name)
            for session_id, timestamp in (
                ("legacy-a", 100.0),
                ("legacy-b", 3700.0),
                ("legacy-c", 200000.0),
            ):
                _write_session(
                    root,
                    session_id=session_id,
                    participant_id="legacy-label",
                    timestamp=timestamp,
                )
            sessions, diagnostics = load_capture_sessions(root)
            payload = audit_capture_independence(
                sessions,
                diagnostics,
                requirements=IndependenceRequirements(1, 2),
            )

        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["summary"]["capture_groups"], 2)
        self.assertEqual(payload["summary"]["legacy_links_applied"], 1)
        self.assertEqual(payload["summary"]["missing_provenance_sessions"], 3)

    def test_conflicting_provenance_is_a_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-capture-audit-") as name:
            root = Path(name)
            _write_session(
                root,
                session_id="conflict",
                participant_id="label",
                timestamp=100.0,
                capture_run_id="capture-meta",
            )
            manifest = root / "conflict" / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "created_at_unix": 100.0,
                        "capture_run_id": "capture-row",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            sessions, diagnostics = load_capture_sessions(root)
            payload = audit_capture_independence(sessions, diagnostics)

        self.assertEqual(payload["status"], "not_ready")
        self.assertIn(
            "PROVENANCE_CONFLICTS",
            {issue["code"] for issue in payload["issues"]},
        )


if __name__ == "__main__":
    unittest.main()

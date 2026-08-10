"""Standalone cross-process regression for rehearsal invitation rotation.

This module intentionally stays outside the offline gate because that gate
blocks child-process creation. Run it directly with the project Python.
"""

from __future__ import annotations

import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path
from queue import Empty

import core.participant_study.store as store_module
from core.participant_study import ParticipantStudyStore


def _settings() -> dict[str, object]:
    return {
        "LEXIGAZE_STUDY_MODE": "rehearsal",
        "LEXIGAZE_STUDY_REHEARSAL_MODE": "1",
        "LEXIGAZE_REHEARSAL_ACKNOWLEDGED_DEVELOPMENT_ONLY": "1",
        "LEXIGAZE_REHEARSAL_INVITES_ONLY": "1",
        "LEXIGAZE_REQUEST_BODY_LOGGING_DISABLED": "1",
        "LEXIGAZE_STORAGE_ENCRYPTED": "1",
        "LEXIGAZE_DATA_LOCATION": "encrypted-test-volume",
        "LEXIGAZE_PUBLIC_BASE_URL": "http://127.0.0.1:8080",
        "LEXIGAZE_DATA_RETENTION_DAYS": "30",
        "LEXIGAZE_RAW_FRAME_RETENTION_HOURS": "1",
    }


def _registry_path(root: Path) -> Path:
    return (
        root
        / "data"
        / "participant_studies"
        / "lexigaze-reader-pilot"
        / "rehearsals"
        / "collection_invites.json"
    )


def _consume_worker(
    root: str,
    settings: dict[str, object],
    invite_code: str,
    about_to_call,
    entered_atomic,
    release_atomic,
    result_queue,
) -> None:
    original_atomic = store_module._atomic_json

    def paused_atomic(path, payload):
        if path.name == "collection_invites.json":
            entered_atomic.set()
            if not release_atomic.wait(10):
                raise RuntimeError("test timed out before releasing consumer")
        return original_atomic(path, payload)

    store_module._atomic_json = paused_atomic
    try:
        if about_to_call is not None:
            about_to_call.set()
        result = ParticipantStudyStore(
            Path(root), settings=settings
        )._consume_invite(invite_code, "ST-LOCK-CONSUME", mode="rehearsal")
        result_queue.put(("ok", result["study_session_id"]))
    except Exception as exc:  # noqa: BLE001 - child reports exact outcome
        result_queue.put(("error", str(exc)))


def _rotate_worker(
    root: str,
    settings: dict[str, object],
    pair_id: str,
    about_to_call,
    entered_atomic,
    release_atomic,
    result_queue,
) -> None:
    original_atomic = store_module._atomic_json

    if entered_atomic is not None:
        def paused_atomic(path, payload):
            if path.name == "collection_invites.json":
                entered_atomic.set()
                if not release_atomic.wait(10):
                    raise RuntimeError("test timed out before releasing rotator")
            return original_atomic(path, payload)

        store_module._atomic_json = paused_atomic
    try:
        if about_to_call is not None:
            about_to_call.set()
        result = ParticipantStudyStore(
            Path(root), settings=settings
        ).rotate_unused_collection_invite(pair_id, 1)
        result_queue.put(("ok", result["invite_code"]))
    except Exception as exc:  # noqa: BLE001 - child reports exact outcome
        result_queue.put(("error", str(exc)))


def _join_or_fail(test: unittest.TestCase, process, label: str) -> None:
    process.join(15)
    if process.is_alive():
        process.terminate()
        process.join(5)
        test.fail(f"{label} process did not finish")
    test.assertEqual(process.exitcode, 0, f"{label} process failed")


class CollectionInviteInterprocessLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = multiprocessing.get_context("spawn")
        self.temp_dir = tempfile.TemporaryDirectory(
            prefix="lexigaze-invite-process-lock-"
        )
        self.root = Path(self.temp_dir.name)
        self.settings = _settings()
        self.pair = ParticipantStudyStore(
            self.root, settings=self.settings
        ).create_collection_invite_pairs(1)[0]
        self.original_code = self.pair["visits"][0]["invite_code"]

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_consume_wins_and_rotation_cannot_resurrect_stale_state(self) -> None:
        consumer_entered = self.context.Event()
        release_consumer = self.context.Event()
        consumer_results = self.context.Queue()
        rotation_results = self.context.Queue()
        consumer = self.context.Process(
            target=_consume_worker,
            args=(
                str(self.root),
                self.settings,
                self.original_code,
                None,
                consumer_entered,
                release_consumer,
                consumer_results,
            ),
        )
        consumer.start()
        self.assertTrue(consumer_entered.wait(10))

        rotator_started = self.context.Event()
        rotator_entered_atomic = self.context.Event()
        release_rotator = self.context.Event()
        rotator = self.context.Process(
            target=_rotate_worker,
            args=(
                str(self.root),
                self.settings,
                self.pair["pair_id"],
                rotator_started,
                rotator_entered_atomic,
                release_rotator,
                rotation_results,
            ),
        )
        rotator.start()
        self.assertTrue(rotator_started.wait(10))
        self.assertFalse(rotator_entered_atomic.wait(0.5))
        with self.assertRaises(Empty):
            rotation_results.get(timeout=0.2)

        release_rotator.set()
        release_consumer.set()
        _join_or_fail(self, consumer, "consumer")
        _join_or_fail(self, rotator, "rotator")
        self.assertEqual(consumer_results.get(timeout=2), ("ok", "ST-LOCK-CONSUME"))
        rotation_status, rotation_message = rotation_results.get(timeout=2)
        self.assertEqual(rotation_status, "error")
        self.assertIn("used invitation cannot be rotated", rotation_message)
        registry = json.loads(_registry_path(self.root).read_text(encoding="utf-8"))
        first = next(item for item in registry["invites"] if item["visit_index"] == 1)
        self.assertEqual(first["study_session_id"], "ST-LOCK-CONSUME")
        self.assertNotIn("code_rotation_history", first)

    def test_rotation_wins_and_old_code_cannot_overwrite_it(self) -> None:
        rotator_entered = self.context.Event()
        release_rotator = self.context.Event()
        rotation_results = self.context.Queue()
        consumer_results = self.context.Queue()
        rotator = self.context.Process(
            target=_rotate_worker,
            args=(
                str(self.root),
                self.settings,
                self.pair["pair_id"],
                None,
                rotator_entered,
                release_rotator,
                rotation_results,
            ),
        )
        rotator.start()
        self.assertTrue(rotator_entered.wait(10))

        consumer_started = self.context.Event()
        consumer_entered_atomic = self.context.Event()
        release_consumer = self.context.Event()
        consumer = self.context.Process(
            target=_consume_worker,
            args=(
                str(self.root),
                self.settings,
                self.original_code,
                consumer_started,
                consumer_entered_atomic,
                release_consumer,
                consumer_results,
            ),
        )
        consumer.start()
        self.assertTrue(consumer_started.wait(10))
        self.assertFalse(consumer_entered_atomic.wait(0.5))
        with self.assertRaises(Empty):
            consumer_results.get(timeout=0.2)

        release_consumer.set()
        release_rotator.set()
        _join_or_fail(self, rotator, "rotator")
        _join_or_fail(self, consumer, "consumer")
        rotation_status, replacement_code = rotation_results.get(timeout=2)
        self.assertEqual(rotation_status, "ok")
        self.assertNotEqual(replacement_code, self.original_code)
        consumer_status, consumer_message = consumer_results.get(timeout=2)
        self.assertEqual(consumer_status, "error")
        self.assertIn("invalid or already-used invitation code", consumer_message)
        registry_text = _registry_path(self.root).read_text(encoding="utf-8")
        self.assertNotIn(self.original_code, registry_text)
        self.assertNotIn(replacement_code, registry_text)
        registry = json.loads(registry_text)
        first = next(item for item in registry["invites"] if item["visit_index"] == 1)
        self.assertEqual(first["code_rotation_count"], 1)
        self.assertIsNone(first["study_session_id"])


if __name__ == "__main__":
    unittest.main()

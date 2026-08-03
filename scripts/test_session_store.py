from __future__ import annotations

import json
import logging
import math
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from unittest import mock
from uuid import UUID, uuid4

from web import create_app
from web.session_store import (
    CorruptSession,
    InvalidSessionId,
    InvalidSessionPayload,
    SessionNotFound,
    SessionStore,
    SessionTooLarge,
    SessionWriteError,
)


class SessionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="lexigaze-session-store-")
        self.root = Path(self.temp_dir.name)
        self.store = SessionStore(self.root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_round_trip_summary_and_legacy_timestamp(self):
        result = self.store.create({
            "filename": "閱讀材料.md",
            "filetype": "md",
            "items": [{"text": "hello", "index": 0}],
        })

        UUID(result["id"])
        self.assertEqual(datetime.fromisoformat(result["created_at"]).tzinfo, UTC)
        stored = self.store.get(result["id"])
        self.assertEqual(stored["filename"], "閱讀材料.md")
        self.assertEqual(stored["item_count"], 1)
        self.assertEqual(stored["items"][0]["text"], "hello")

        scan = self.store.scan()
        self.assertEqual(scan.corrupt_count, 0)
        self.assertEqual(scan.total_files, 1)
        self.assertEqual(scan.sessions, [{
            "id": result["id"],
            "filename": "閱讀材料.md",
            "filetype": "md",
            "created_at": result["created_at"],
            "item_count": 1,
        }])

        stored["created_at"] = "2026-06-22T15:00:11.077962"
        path = self.root / f"{result['id']}.json"
        path.write_text(json.dumps(stored), encoding="utf-8")
        self.assertEqual(self.store.get(result["id"])["created_at"], stored["created_at"])

    def test_rejects_invalid_payloads_without_artifacts(self):
        cases = (
            [],
            {"items": {}},
            {"items": ["not-an-object"]},
            {"filename": 123, "items": []},
            {"filetype": "x" * 65, "items": []},
            {"filename": "bad\x00name", "items": []},
            {"items": [{"score": math.nan}]},
            {"items": [{"bad": object()}]},
        )
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(InvalidSessionPayload):
                    self.store.create(payload)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_enforces_item_and_encoded_size_limits(self):
        item_limited = SessionStore(self.root, max_items=1)
        with self.assertRaises(SessionTooLarge):
            item_limited.create({"items": [{}, {}]})

        byte_limited = SessionStore(self.root, max_bytes=128)
        with self.assertRaises(SessionTooLarge):
            byte_limited.create({"items": [{"text": "x" * 256}]})
        self.assertEqual(list(self.root.iterdir()), [])

    def test_rejects_traversal_and_malformed_ids(self):
        for value in ("", "not-a-uuid", "../escape", "..\\escape", "0" * 35):
            with self.subTest(value=value):
                with self.assertRaises(InvalidSessionId):
                    self.store.get(value)
                with self.assertRaises(InvalidSessionId):
                    self.store.delete(value)

        missing = str(uuid4())
        with self.assertRaises(SessionNotFound):
            self.store.get(missing)

        result = self.store.create({"items": []})
        braced_uppercase = "{" + result["id"].upper() + "}"
        self.assertEqual(self.store.get(braced_uppercase)["id"], result["id"])

    def test_corrupt_files_are_observable_and_skipped(self):
        session_id = str(uuid4())
        path = self.root / f"{session_id}.json"
        path.write_text("{not-json", encoding="utf-8")

        nonfinite_id = str(uuid4())
        nonfinite_path = self.root / f"{nonfinite_id}.json"
        nonfinite_path.write_text(
            json.dumps({
                "id": nonfinite_id,
                "filename": "nonfinite.md",
                "created_at": "2026-08-04T00:00:00+00:00",
                "item_count": 1,
                "items": [{"score": math.nan}],
            }),
            encoding="utf-8",
        )

        with self.assertLogs("web.session_store", level=logging.WARNING) as captured:
            scan = self.store.scan()
        self.assertEqual(scan.sessions, [])
        self.assertEqual(scan.corrupt_count, 2)
        self.assertEqual(scan.total_files, 2)
        self.assertIn(path.name, "\n".join(captured.output))
        self.assertIn(nonfinite_path.name, "\n".join(captured.output))
        with self.assertRaises(CorruptSession):
            self.store.get(session_id)
        with self.assertRaises(CorruptSession):
            self.store.get(nonfinite_id)

    def test_rejects_symbolic_session_files_when_supported(self):
        target = self.root / "outside.txt"
        target.write_text("{}", encoding="utf-8")
        session_id = str(uuid4())
        link = self.root / f"{session_id}.json"
        try:
            link.symlink_to(target)
        except OSError:
            self.skipTest("symbolic links are unavailable in this environment")
        with self.assertRaises(CorruptSession):
            self.store.get(session_id)

    def test_publish_failure_leaves_no_partial_or_temporary_file(self):
        with (
            mock.patch("web.session_store.os.replace", side_effect=OSError("simulated")),
            self.assertRaises(SessionWriteError),
        ):
            self.store.create({"filename": "atomic.md", "items": []})

        self.assertEqual(list(self.root.glob("*.json")), [])
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_concurrent_crud_does_not_corrupt_or_leak_files(self):
        stores = [SessionStore(self.root) for _ in range(4)]
        start = Event()

        def create_one(index: int) -> str:
            start.wait()
            return stores[index % len(stores)].create({
                "filename": f"session-{index}.md",
                "items": [{"index": index}],
            })["id"]

        def scan_while_creating() -> list[int]:
            start.wait()
            return [stores[-1].scan().corrupt_count for _ in range(32)]

        with ThreadPoolExecutor(max_workers=16) as executor:
            scan_future = executor.submit(scan_while_creating)
            create_futures = [executor.submit(create_one, index) for index in range(64)]
            start.set()
            ids = [future.result() for future in create_futures]
            corrupt_counts = scan_future.result()
        self.assertEqual(len(set(ids)), 64)
        self.assertEqual(corrupt_counts, [0] * 32)
        self.assertEqual(len(self.store.scan().sessions), 64)

        with ThreadPoolExecutor(max_workers=8) as executor:
            records = list(
                executor.map(
                    lambda pair: stores[pair[0] % len(stores)].get(pair[1]),
                    enumerate(ids),
                )
            )
        self.assertEqual({record["id"] for record in records}, set(ids))

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(
                executor.map(
                    lambda pair: stores[pair[0] % len(stores)].delete(pair[1]),
                    enumerate(ids),
                )
            )
        self.assertEqual(self.store.scan().sessions, [])
        self.assertEqual(list(self.root.glob("*.tmp")), [])


class SessionApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="lexigaze-session-api-")
        self.root = Path(self.temp_dir.name)
        self.app = create_app({
            "TESTING": True,
            "LEXIGAZE_BLUEPRINTS": (),
            "LEXIGAZE_DATA_DIR": self.root,
            "LEXIGAZE_SESSION_MAX_BYTES": 4096,
            "LEXIGAZE_SESSION_MAX_ITEMS": 4,
        })
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_successful_crud_preserves_response_shapes(self):
        created = self.client.post("/api/sessions", json={
            "filename": "sample.md",
            "filetype": "md",
            "items": [{"text": "sample"}],
        })
        self.assertEqual(created.status_code, 201)
        self.assertEqual(set(created.json), {"id", "created_at"})
        session_id = created.json["id"]

        fetched = self.client.get(f"/api/sessions/{session_id}")
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(fetched.json["id"], session_id)
        self.assertEqual(fetched.json["item_count"], 1)

        listed = self.client.get("/api/sessions")
        self.assertEqual(listed.status_code, 200)
        self.assertIsInstance(listed.json, list)
        self.assertEqual(set(listed.json[0]), {
            "id", "filename", "filetype", "created_at", "item_count"
        })

        ping = self.client.get("/api/ping")
        self.assertEqual(ping.json, {
            "ok": True, "sessions": 1, "corrupt_sessions": 0
        })

        deleted = self.client.delete(f"/api/sessions/{session_id}")
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(deleted.json, {"ok": True})
        self.assertEqual(self.client.get(f"/api/sessions/{session_id}").status_code, 404)

    def test_invalid_requests_return_json_and_create_nothing(self):
        cases = (
            self.client.post("/api/sessions", data="{}"),
            self.client.post(
                "/api/sessions", data="{", content_type="application/json"
            ),
            self.client.post("/api/sessions", json=[]),
            self.client.post("/api/sessions", json={"items": {}}),
            self.client.post("/api/sessions", json={"items": [{}, {}, {}, {}, {}]}),
            self.client.post(
                "/api/sessions",
                data=json.dumps({"items": [{"text": "x" * 5000}]}),
                content_type="application/json",
            ),
        )
        self.assertEqual(
            [response.status_code for response in cases],
            [415, 400, 400, 400, 413, 413],
        )
        for response in cases:
            self.assertEqual(response.content_type, "application/json")
            self.assertIn("error", response.json)
            self.assertIn("message", response.json)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_invalid_missing_and_corrupt_ids_have_stable_errors(self):
        invalid = self.client.get("/api/sessions/not-a-uuid")
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(invalid.json["error"], "invalid_session_id")

        missing = self.client.get(f"/api/sessions/{uuid4()}")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json["error"], "not_found")

        corrupt_id = str(uuid4())
        (self.root / f"{corrupt_id}.json").write_text("{broken", encoding="utf-8")
        corrupt = self.client.get(f"/api/sessions/{corrupt_id}")
        self.assertEqual(corrupt.status_code, 500)
        self.assertEqual(corrupt.json["error"], "corrupt_session")
        self.assertEqual(self.client.get("/api/sessions").json, [])
        self.assertEqual(self.client.get("/api/ping").json["corrupt_sessions"], 1)


if __name__ == "__main__":
    unittest.main()

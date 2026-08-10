"""Pure-standard-library tests for the read-only Visit 2 preflight."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
import urllib.parse
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.preflight_general_collection_visit import (
    EXPECTED_BRANCH,
    _canonical_sha256,
    build_preflight,
)


ROOT = Path(__file__).resolve().parents[1]
SESSION_ID = "ST-AAAAAAAAAAAAAAAAAAAA"
PAIR_ID = "PAIR-1111111111111111"
PARTICIPANT_ID = "GP-222222222222"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GeneralCollectionVisitPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="lexigaze-visit-preflight-")
        self.addCleanup(temporary.cleanup)
        sandbox = Path(temporary.name)
        self.code_root = sandbox / "code"
        self.study_root = sandbox / "study"
        for relative in (
            "core/participant_study/general_collection_v1.json",
            "core/participant_study/general_collection_bank_v1.json",
            "core/participant_study/protocol_v1.json",
            "web/static/participant_study.js",
            "web/static/participant_collection.js",
            "web/static/gaze_calibration_feedback.js",
            "core/participant_study/store.py",
            "web/__init__.py",
            "web/routes/study.py",
            "web/routes/gaze.py",
        ):
            target = self.code_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        self.now = datetime(2026, 8, 10, 22, 0, tzinfo=UTC)
        self.protocol = json.loads(
            (self.code_root / "core/participant_study/general_collection_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.bank = json.loads(
            (
                self.code_root
                / "core/participant_study/general_collection_bank_v1.json"
            ).read_text(encoding="utf-8")
        )
        parent = json.loads(
            (self.code_root / "core/participant_study/protocol_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.rehearsal_root = (
            self.study_root
            / "data"
            / "participant_studies"
            / parent["protocol_id"]
            / "rehearsals"
        )
        self.rehearsal_root.mkdir(parents=True)
        self.registry_path = self.rehearsal_root / "collection_invites.json"
        self.session_path = self.rehearsal_root / SESSION_ID / "session.json"
        self.session_path.parent.mkdir()
        self.gaze_root = self.study_root / "data" / "sessions" / "GAZE-FIXTURE"
        self.gaze_root.mkdir(parents=True)
        self._write_fixture(completed_at=self.now - timedelta(hours=20))

    def _write_fixture(self, *, completed_at: datetime) -> None:
        protocol_sha256 = _canonical_sha256(self.protocol)
        bank_sha256 = _canonical_sha256(self.bank)
        registry = {
            "schema_version": 1,
            "protocol_sha256": protocol_sha256,
            "bank_sha256": bank_sha256,
            "invites": [
                {
                    "code_sha256": "a" * 64,
                    "used_at_utc": (self.now - timedelta(hours=80)).isoformat(),
                    "study_session_id": SESSION_ID,
                    "pair_id": PAIR_ID,
                    "participant_id": PARTICIPANT_ID,
                    "schedule_cell": 0,
                    "sequence": "A_then_B",
                    "order_cell": 0,
                    "visit_index": 1,
                    "form_id": "A",
                    "protocol_sha256": protocol_sha256,
                    "bank_sha256": bank_sha256,
                },
                {
                    "code_sha256": "c" * 64,
                    "code_rotation_history": [
                        {
                            "code_sha256": "b" * 64,
                            "rotated_at_utc": "2026-08-10T00:00:00+00:00",
                        }
                    ],
                    "code_rotation_count": 1,
                    "used_at_utc": None,
                    "study_session_id": None,
                    "pair_id": PAIR_ID,
                    "participant_id": PARTICIPANT_ID,
                    "schedule_cell": 0,
                    "sequence": "A_then_B",
                    "order_cell": 0,
                    "visit_index": 2,
                    "form_id": "B",
                    "protocol_sha256": protocol_sha256,
                    "bank_sha256": bank_sha256,
                },
            ],
        }
        self.registry_path.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        session = {
            "study_session_id": SESSION_ID,
            "participant_id": PARTICIPANT_ID,
            "state": "completed",
            "events": [
                {
                    "event": "general_collection_completed",
                    "at_utc": completed_at.astimezone(UTC).isoformat(),
                }
            ],
            "collection_assignment": {
                key: registry["invites"][0][key]
                for key in (
                    "pair_id",
                    "schedule_cell",
                    "sequence",
                    "order_cell",
                    "visit_index",
                    "form_id",
                    "protocol_sha256",
                    "bank_sha256",
                )
            },
            "linked_data": {"gaze_session_id": "GAZE-FIXTURE"},
            "quality": {
                "general_system_check": {
                    "device": {
                        "device_class": "desktop",
                        "browser_family": "chromium",
                        "viewport_width": 1697,
                        "viewport_height": 888,
                        "device_pixel_ratio_bucket": "1_25_1_74",
                        "camera_width": 640,
                        "camera_height": 480,
                        "estimated_camera_fps_band": "15_23",
                    }
                },
                "calibration": {"calibration_images_purged": True},
            },
        }
        self.session_path.write_text(
            json.dumps(session, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _git_probe(_: Path) -> dict[str, object]:
        return {
            "head": "f" * 40,
            "branch": EXPECTED_BRANCH,
            "capture_critical_files_match_head": True,
            "capture_critical_mismatch_count": 0,
        }

    @staticmethod
    def _listener_probe(_: str, __: int) -> dict[str, object]:
        return {
            "listener_count": 1,
            "pid": 1234,
            "started_at_utc": "2100-01-01T00:00:00+00:00",
            "unexpected_bindings": 0,
        }

    def _getter(
        self,
        url: str,
        *,
        health_extra: bool = False,
        asset_tamper: str | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        path = urllib.parse.urlsplit(url).path
        headers = {"cache-control": "no-store"}
        if path == "/api/gaze/health":
            payload = {"ok": True}
            if health_extra:
                payload["backend"] = "forbidden"
            return 200, headers, json.dumps(payload).encode("utf-8")
        if path == "/api/study/protocol":
            payload = {
                "ok": True,
                "protocol": {
                    "protocol_digest_sha256": "d" * 64,
                    "activation": {
                        "configured_mode": "rehearsal",
                        "rehearsal_ready": True,
                        "pilot_ready": False,
                        "rehearsal_scope": (
                            "local_invited_self_development_unencrypted"
                        ),
                        "rehearsal_self_only": True,
                    },
                    "data_governance": {
                        "location": str((self.study_root / "data").resolve()),
                        "raw_frame_retention_hours": 1,
                        "retention_policy": "manual_until_researcher_deletes",
                        "self_only": True,
                        "formal_promotion_allowed": False,
                    },
                },
            }
            return 200, headers, json.dumps(payload).encode("utf-8")
        if path == "/api/study/general-collection/protocol":
            payload = {
                "ok": True,
                "design_audit": {
                    "ok": True,
                    "protocol_sha256": _canonical_sha256(self.protocol),
                    "bank_sha256": _canonical_sha256(self.bank),
                },
            }
            return 200, headers, json.dumps(payload).encode("utf-8")
        if path == "/study":
            return (
                200,
                headers,
                b'<input id="inviteCode"><button id="startAnotherInviteBtn">',
            )
        if path.startswith("/static/"):
            name = path.rsplit("/", 1)[-1]
            body = (self.code_root / "web" / "static" / name).read_bytes()
            if asset_tamper == name:
                body += b"tampered"
            return 200, headers, body
        if path in {"/api/gaze/datasets", "/api/ping", "/api/gaze/health/extra"}:
            return 403, headers, b'{"ok":false}'
        raise AssertionError(f"unexpected fixture URL: {url}")

    def _build(self, **kwargs: object) -> dict[str, object]:
        return build_preflight(
            self.code_root,
            self.study_root,
            now=self.now,
            git_probe=self._git_probe,
            listener_probe=self._listener_probe,
            getter=kwargs.pop("getter", self._getter),
            **kwargs,
        )

    def test_machine_ready_output_is_read_only_and_secret_free(self) -> None:
        before = (_sha256(self.registry_path), _sha256(self.session_path))
        result = self._build(expected_head="f" * 40)
        after = (_sha256(self.registry_path), _sha256(self.session_path))
        self.assertEqual(result["status"], "machine_ready")
        self.assertEqual(before, after)
        self.assertFalse(result["privacy"]["plaintext_invite_read"])
        self.assertFalse(result["privacy"]["registry_or_session_write_performed"])
        rendered = json.dumps(result, ensure_ascii=False)
        for forbidden in (
            PAIR_ID,
            PARTICIPANT_ID,
            SESSION_ID,
            "a" * 64,
            "b" * 64,
            "c" * 64,
        ):
            self.assertNotIn(forbidden, rendered)

    def test_window_uses_completion_event_not_old_invite_consumption(self) -> None:
        self._write_fixture(completed_at=self.now - timedelta(hours=17))
        result = self._build()
        self.assertEqual(result["status"], "waiting_for_window")
        self.assertEqual(result["visit"]["window"]["state"], "too_early")
        self.assertEqual(
            result["visit"]["window"]["basis"],
            "visit_1_general_collection_completed_event",
        )

    def test_plaintext_registry_field_fails_closed_without_echo(self) -> None:
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        registry["invites"][1]["invite_code"] = "SENTINEL-PLAINTEXT"
        self.registry_path.write_text(json.dumps(registry), encoding="utf-8")
        result = self._build()
        self.assertEqual(result["status"], "failed")
        self.assertIn(
            "registry contains a forbidden plaintext invite field",
            result["failures"],
        )
        self.assertNotIn("SENTINEL-PLAINTEXT", json.dumps(result))

    def test_public_health_and_served_asset_contracts_fail_closed(self) -> None:
        health_result = self._build(
            getter=lambda url: self._getter(url, health_extra=True)
        )
        self.assertEqual(health_result["status"], "failed")
        self.assertIn(
            "participant-safe gaze health exposes unexpected fields",
            health_result["failures"],
        )
        asset_result = self._build(
            getter=lambda url: self._getter(
                url,
                asset_tamper="participant_study.js",
            )
        )
        self.assertEqual(asset_result["status"], "failed")
        self.assertIn(
            "served participant_study.js differs from the selected code root",
            asset_result["failures"],
        )

    def test_non_exact_loopback_url_is_never_requested(self) -> None:
        requested: list[str] = []

        def forbidden_getter(url: str) -> tuple[int, dict[str, str], bytes]:
            requested.append(url)
            raise AssertionError("non-loopback getter must not be called")

        for base_url in (
            "https://example.invalid:8098",
            "http://127.0.0.1:8098/extra",
            "http://user@127.0.0.1:8098",
        ):
            with self.subTest(base_url=base_url):
                result = self._build(
                    base_url=base_url,
                    getter=forbidden_getter,
                )
                self.assertEqual(result["status"], "failed")
        self.assertEqual(requested, [])

    def test_legacy_raw_directory_is_warning_only_and_never_deleted(self) -> None:
        legacy_raw = self.study_root / "data" / "sessions" / "LEGACY" / "raw"
        legacy_raw.mkdir(parents=True)
        marker = legacy_raw / "keep.jpg"
        marker.write_bytes(b"do-not-delete")
        result = self._build()
        self.assertEqual(result["status"], "machine_ready")
        self.assertTrue(marker.exists())
        self.assertTrue(
            any("warning-only" in warning for warning in result["warnings"])
        )
        self.assertFalse(result["privacy"]["legacy_raw_deleted"])


if __name__ == "__main__":
    unittest.main()

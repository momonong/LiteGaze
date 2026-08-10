"""Pure-stdlib contracts for offline gaze preprocessing assets."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.unigaze_personalization.runtime_assets import (
    resolve_face_landmarker_asset,
)


class GazeRuntimeAssetTests(unittest.TestCase):
    def test_repository_fallback_is_local_and_available(self) -> None:
        path = resolve_face_landmarker_asset()
        self.assertTrue(path.is_file())
        self.assertEqual(path.name, "face_landmarker.task")
        self.assertIn("archive", path.parts)

    def test_missing_asset_fails_without_network_or_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "network download is disabled"):
                resolve_face_landmarker_asset(root)
            self.assertEqual(list(root.rglob("*")), [])

    def test_resolver_source_has_no_network_client(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "core"
            / "unigaze_personalization"
            / "runtime_assets.py"
        ).read_text(encoding="utf-8")
        for forbidden in ("urllib", "requests", "http://", "https://"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()

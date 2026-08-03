import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.experiment_manifest import (
    _git_metadata,
    fingerprint_files,
    write_experiment_manifest,
)


class TestExperimentManifest(unittest.TestCase):
    def test_fingerprint_records_hash_and_missing_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "data" / "sample.csv"
            dataset.parent.mkdir()
            dataset.write_bytes(b"word,value\nhello,1\n")

            records = fingerprint_files([dataset, root / "missing.csv"], root)

        self.assertEqual(records[0]["path"], "data/sample.csv")
        self.assertEqual(
            records[0]["sha256"], hashlib.sha256(b"word,value\nhello,1\n").hexdigest()
        )
        self.assertFalse(records[1]["exists"])
        self.assertNotIn("sha256", records[1])

    def test_manifest_is_atomic_and_contains_reproduction_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "inputs" / "source.csv"
            artifact = root / "output" / "result.csv"
            destination = root / "output" / "manifest.json"
            dataset.parent.mkdir()
            artifact.parent.mkdir()
            dataset.write_text("x\n1\n", encoding="utf-8")
            artifact.write_text("score\n0.9\n", encoding="utf-8")

            with (
                patch(
                    "scripts.experiment_manifest._git_metadata",
                    return_value={"commit": "abc123", "branch": "test", "dirty": False},
                ),
                patch("scripts.experiment_manifest._gpu_inventory", return_value=[]),
                patch(
                    "scripts.experiment_manifest._package_versions",
                    return_value={"numpy": "test-version"},
                ),
            ):
                written = write_experiment_manifest(
                    destination,
                    "unit_test_experiment",
                    root=root,
                    datasets=[dataset],
                    artifacts=[artifact],
                    config={"sigma": 40.0},
                    metrics={"accuracy": 0.9},
                    seed=42,
                    duration_seconds=1.25,
                )

            manifest = json.loads(written.read_text(encoding="utf-8"))
            temporary_files = list(
                destination.parent.glob(f".{destination.name}.*.tmp")
            )

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["experiment"]["name"], "unit_test_experiment")
        self.assertEqual(manifest["experiment"]["seed"], 42)
        self.assertEqual(manifest["source"]["commit"], "abc123")
        self.assertTrue(manifest["source"]["files"])
        self.assertTrue(manifest["source"]["files"][0]["exists"])
        self.assertEqual(manifest["datasets"][0]["path"], "inputs/source.csv")
        self.assertEqual(manifest["artifacts"][0]["path"], "output/result.csv")
        self.assertEqual(manifest["config"]["sigma"], 40.0)
        self.assertEqual(manifest["metrics"]["accuracy"], 0.9)
        self.assertEqual(temporary_files, [])

    def test_git_metadata_fingerprints_tracked_diff(self):
        diff = "diff --git a/example.py b/example.py\n+print('changed')"
        with patch(
            "scripts.experiment_manifest._run_command",
            side_effect=[
                "abc123",
                "codex/feat/example",
                " M example.py\n?? notes.txt",
                " M example.py",
                diff,
            ],
        ):
            metadata = _git_metadata(Path("."))

        self.assertTrue(metadata["dirty"])
        self.assertTrue(metadata["tracked_changes"])
        self.assertEqual(
            metadata["tracked_diff_from_head_sha256"],
            hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        )

    def test_pre_run_source_snapshot_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            destination = root / "manifest.json"
            snapshot = {
                "commit": "protocol-lock",
                "branch": "test/generalization",
                "dirty": False,
                "files": [{"path": "study.py", "sha256": "source-hash"}],
            }
            with (
                patch("scripts.experiment_manifest._gpu_inventory", return_value=[]),
                patch("scripts.experiment_manifest._package_versions", return_value={}),
            ):
                write_experiment_manifest(
                    destination,
                    "snapshot_test",
                    root=root,
                    source_snapshot=snapshot,
                )

            manifest = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(manifest["source"], snapshot)


if __name__ == "__main__":
    unittest.main()

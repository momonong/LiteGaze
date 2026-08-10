"""Contracts for the offline, recomputable UniGaze base inference identity."""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from core.gaze_core.base_inference_bundle import (
    BUNDLE_TYPE,
    CHECKPOINT_FILENAME,
    FACE_LANDMARKER_LOGICAL_PATH,
    HF_REPO_ID,
    MODEL_NAME,
    REPOSITORY_RUNTIME_FILES,
    RUNTIME_DISTRIBUTIONS,
    BaseInferenceBundleError,
    build_base_inference_bundle,
    canonical_json_bytes,
    canonical_sha256,
    verify_base_inference_bundle,
)


COMMIT = "1234567890abcdef1234567890abcdef12345678"
VERSIONS = {
    "unigaze": "0.1.3",
    "torch": "2.9.1+cu130",
    "mediapipe": "0.10.35",
    "opencv-python": "4.13.0.92",
    "numpy": "2.4.4",
    "safetensors": "0.7.0",
}


class FakeIdentityEnvironment:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.repository_root = root / "repository"
        self.cache_root = root / "huggingface" / "hub"
        self.repo_cache = (
            self.cache_root / "models--UniGaze--UniGaze-models"
        )
        self.snapshot = self.repo_cache / "snapshots" / COMMIT
        self.checkpoint = self.snapshot / CHECKPOINT_FILENAME
        self.package_root = root / "site-packages" / "unigaze"
        self.version_calls: list[str] = []

        for index, relative in enumerate(REPOSITORY_RUNTIME_FILES):
            path = self.repository_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"runtime-file-{index}:{relative}\n".encode("utf-8"))
        face_asset = self.repository_root / FACE_LANDMARKER_LOGICAL_PATH
        face_asset.parent.mkdir(parents=True, exist_ok=True)
        face_asset.write_bytes(b"fake-mediapipe-face-landmarker-task")
        (self.repo_cache / "refs").mkdir(parents=True, exist_ok=True)
        (self.repo_cache / "refs" / "main").write_text(
            COMMIT + "\n", encoding="utf-8"
        )
        self.snapshot.mkdir(parents=True, exist_ok=True)
        self.checkpoint.write_bytes(b"fake-safetensors-checkpoint\x00\x01")
        (self.package_root / "models").mkdir(parents=True, exist_ok=True)
        (self.package_root / "__init__.py").write_text(
            "from .loader import load\n", encoding="utf-8"
        )
        (self.package_root / "loader.py").write_text(
            "def load(name, device='cpu'): return name, device\n",
            encoding="utf-8",
        )
        (self.package_root / "models" / "mae.py").write_text(
            "class MAE: pass\n", encoding="utf-8"
        )

    def version_provider(self, name: str) -> str:
        self.version_calls.append(name)
        return VERSIONS[name]

    def kwargs(self) -> dict:
        return {
            "repository_root": self.repository_root,
            "hf_cache_root": self.cache_root,
            "checkpoint_path": self.checkpoint,
            "unigaze_package_root": self.package_root,
            "version_provider": self.version_provider,
            "local_files_only": True,
        }


def _rehash_bundle(bundle: dict) -> None:
    core = deepcopy(bundle)
    core.pop("bundle_sha256", None)
    core.pop("model_sha256", None)
    digest = canonical_sha256(core)
    bundle["bundle_sha256"] = digest
    bundle["model_sha256"] = digest


class GazeBaseInferenceBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.environment = FakeIdentityEnvironment(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact_identity_hashes_checkpoint_code_assets_package_and_versions(
        self,
    ) -> None:
        bundle = build_base_inference_bundle(**self.environment.kwargs())
        expected_checkpoint_sha = hashlib.sha256(
            self.environment.checkpoint.read_bytes()
        ).hexdigest()
        self.assertEqual(bundle["bundle_type"], BUNDLE_TYPE)
        self.assertEqual(bundle["status"], "offline_identity_complete")
        self.assertEqual(bundle["model_name"], MODEL_NAME)
        self.assertEqual(bundle["model_id"], f"{MODEL_NAME}@{COMMIT}")
        self.assertEqual(bundle["resolved_revision"], COMMIT)
        self.assertEqual(bundle["model_sha256"], bundle["bundle_sha256"])
        self.assertEqual(
            bundle["components"]["checkpoint"]["repo_id"], HF_REPO_ID
        )
        self.assertEqual(
            bundle["components"]["checkpoint"]["resolved_snapshot_commit"],
            COMMIT,
        )
        self.assertEqual(bundle["checkpoint_sha256"], expected_checkpoint_sha)
        self.assertEqual(
            [
                item["relative_path"]
                for item in bundle["components"]["repository_runtime"]["files"]
            ],
            [*REPOSITORY_RUNTIME_FILES, FACE_LANDMARKER_LOGICAL_PATH],
        )
        self.assertEqual(
            [
                item["relative_path"]
                for item in bundle["components"]["installed_unigaze_package"][
                    "python_files"
                ]
            ],
            ["__init__.py", "loader.py", "models/mae.py"],
        )
        self.assertEqual(bundle["package_versions"], VERSIONS)
        self.assertEqual(
            self.environment.version_calls, list(RUNTIME_DISTRIBUTIONS)
        )
        self.assertNotIn(str(self.environment.root), json.dumps(bundle))
        self.assertFalse(bundle["identity_contract"]["network_allowed"])
        self.assertFalse(bundle["identity_contract"]["gpu_used"])
        self.assertFalse(bundle["identity_contract"]["model_loaded"])
        self.assertTrue(
            {
                "schema_version",
                "status",
                "model_name",
                "model_id",
                "resolved_revision",
                "checkpoint_sha256",
                "components",
                "package_versions",
                "bundle_sha256",
            }.issubset(bundle)
        )

        summary = verify_base_inference_bundle(
            bundle, **self.environment.kwargs()
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["model_sha256"], bundle["model_sha256"])
        self.assertEqual(summary["repository_file_count"], 8)
        self.assertEqual(summary["unigaze_python_file_count"], 3)
        self.assertFalse(summary["gpu_used"])
        self.assertFalse(summary["model_loaded"])

    def test_bundle_is_deterministic_and_sha_does_not_contain_itself(self) -> None:
        first = build_base_inference_bundle(**self.environment.kwargs())
        second = build_base_inference_bundle(**self.environment.kwargs())
        self.assertEqual(first, second)
        core = deepcopy(first)
        core.pop("bundle_sha256")
        core.pop("model_sha256")
        self.assertEqual(canonical_sha256(core), first["bundle_sha256"])
        self.assertNotIn(
            first["bundle_sha256"], canonical_json_bytes(core).decode("utf-8")
        )

    def test_fresh_verification_detects_checkpoint_repository_and_package_mutation(
        self,
    ) -> None:
        mutations = (
            self.environment.checkpoint,
            self.environment.repository_root / REPOSITORY_RUNTIME_FILES[0],
            self.environment.repository_root / FACE_LANDMARKER_LOGICAL_PATH,
            self.environment.package_root / "loader.py",
        )
        for path in mutations:
            with self.subTest(path=path.name):
                before = path.read_bytes()
                bundle = build_base_inference_bundle(**self.environment.kwargs())
                path.write_bytes(before + b"mutation")
                with self.assertRaisesRegex(
                    BaseInferenceBundleError, "differs from fresh local evidence"
                ):
                    verify_base_inference_bundle(
                        bundle, **self.environment.kwargs()
                    )
                changed = build_base_inference_bundle(**self.environment.kwargs())
                self.assertNotEqual(
                    changed["bundle_sha256"], bundle["bundle_sha256"]
                )
                path.write_bytes(before)

    def test_semantically_rehashed_bundle_still_fails_recomputation(self) -> None:
        bundle = build_base_inference_bundle(**self.environment.kwargs())
        bundle["identity_contract"]["network_allowed"] = True
        _rehash_bundle(bundle)
        with self.assertRaisesRegex(
            BaseInferenceBundleError, "differs from fresh local evidence"
        ):
            verify_base_inference_bundle(
                bundle, **self.environment.kwargs()
            )

        bundle = build_base_inference_bundle(**self.environment.kwargs())
        bundle["model_sha256"] = "f" * 64
        with self.assertRaisesRegex(BaseInferenceBundleError, "must equal"):
            verify_base_inference_bundle(
                bundle, **self.environment.kwargs()
            )

    def test_missing_checkpoint_asset_and_package_source_fail_closed(self) -> None:
        cases = (
            (self.environment.checkpoint, "checkpoint is missing"),
            (
                self.environment.repository_root / FACE_LANDMARKER_LOGICAL_PATH,
                "network download is disabled",
            ),
            (
                self.environment.package_root / "__init__.py",
                None,
            ),
        )
        for path, expected_message in cases:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.unlink()
                if expected_message is None:
                    # Other .py files remain, so removing one source changes the
                    # identity rather than making the package empty.
                    old_bundle = None
                    path.write_bytes(original)
                    old_bundle = build_base_inference_bundle(
                        **self.environment.kwargs()
                    )
                    path.unlink()
                    changed = build_base_inference_bundle(
                        **self.environment.kwargs()
                    )
                    self.assertNotEqual(
                        old_bundle["bundle_sha256"], changed["bundle_sha256"]
                    )
                else:
                    with self.assertRaisesRegex(
                        BaseInferenceBundleError, expected_message
                    ):
                        build_base_inference_bundle(**self.environment.kwargs())
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(original)

    def test_checkpoint_path_escape_and_invalid_ref_are_rejected(self) -> None:
        outside = self.environment.root / CHECKPOINT_FILENAME
        outside.write_bytes(self.environment.checkpoint.read_bytes())
        escaped = self.environment.kwargs()
        escaped["checkpoint_path"] = outside
        with self.assertRaisesRegex(BaseInferenceBundleError, "escapes"):
            build_base_inference_bundle(**escaped)

        invalid_ref = self.environment.repo_cache / "refs" / "main"
        invalid_ref.write_text("../../outside\n", encoding="utf-8")
        kwargs = self.environment.kwargs()
        kwargs.pop("checkpoint_path")
        with self.assertRaisesRegex(BaseInferenceBundleError, "40 lowercase hex"):
            build_base_inference_bundle(**kwargs)

    def test_offline_face_landmarker_resolver_hashes_tracked_archive_fallback(
        self,
    ) -> None:
        primary = self.environment.repository_root / FACE_LANDMARKER_LOGICAL_PATH
        primary.unlink()
        fallback = (
            self.environment.repository_root
            / "archive"
            / "shengwen"
            / "face_landmarker.task"
        )
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_bytes(b"tracked-offline-fallback")
        bundle = build_base_inference_bundle(**self.environment.kwargs())
        face_component = bundle["components"]["repository_runtime"]["files"][-1]
        self.assertEqual(
            face_component["logical_path"], FACE_LANDMARKER_LOGICAL_PATH
        )
        self.assertEqual(
            face_component["relative_path"],
            "archive/shengwen/face_landmarker.task",
        )
        self.assertEqual(
            face_component["sha256"], hashlib.sha256(fallback.read_bytes()).hexdigest()
        )

    def test_literal_offline_boolean_and_strict_types_are_enforced(self) -> None:
        for value in (False, 0, 1, "true", None):
            with self.subTest(local_files_only=value):
                kwargs = self.environment.kwargs()
                kwargs["local_files_only"] = value
                with self.assertRaisesRegex(
                    BaseInferenceBundleError, "literal boolean true"
                ):
                    build_base_inference_bundle(**kwargs)
        kwargs = self.environment.kwargs()
        kwargs["resolved_snapshot_commit"] = True
        with self.assertRaisesRegex(BaseInferenceBundleError, "must be a string"):
            build_base_inference_bundle(**kwargs)

        def boolean_version(name: str):
            return True if name == "torch" else VERSIONS[name]

        kwargs = self.environment.kwargs()
        kwargs["version_provider"] = boolean_version
        with self.assertRaisesRegex(BaseInferenceBundleError, "version is invalid"):
            build_base_inference_bundle(**kwargs)
        with self.assertRaisesRegex(BaseInferenceBundleError, "non-string"):
            canonical_json_bytes({1: "invalid key"})
        with self.assertRaisesRegex(BaseInferenceBundleError, "non-finite"):
            canonical_json_bytes({"value": float("nan")})

    def test_hf_home_cache_resolution_is_offline_and_import_free(self) -> None:
        kwargs = self.environment.kwargs()
        kwargs.pop("hf_cache_root")
        kwargs.pop("checkpoint_path")
        imported_before = {
            name: name in sys.modules
            for name in ("torch", "mediapipe", "cv2", "numpy", "safetensors")
        }
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in {
                "torch",
                "mediapipe",
                "cv2",
                "numpy",
                "safetensors",
                "huggingface_hub",
            }:
                raise AssertionError(f"runtime package imported: {name}")
            return original_import(name, *args, **kwargs)

        environment = {
            "HF_HOME": str(self.environment.cache_root.parent),
            "HF_HUB_OFFLINE": "1",
            "HUGGINGFACE_HUB_CACHE": "",
            "HF_HUB_CACHE": "",
            "CUDA_VISIBLE_DEVICES": "-1",
        }
        with mock.patch.dict(os.environ, environment, clear=False), mock.patch(
            "builtins.__import__", side_effect=guarded_import
        ):
            bundle = build_base_inference_bundle(**kwargs)
        self.assertEqual(bundle["resolved_revision"], COMMIT)
        self.assertEqual(
            imported_before,
            {
                name: name in sys.modules
                for name in imported_before
            },
        )

    def test_source_contains_no_network_or_runtime_model_import(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "core"
            / "gaze_core"
            / "base_inference_bundle.py"
        ).read_text(encoding="utf-8")
        for forbidden in (
            "hf_hub_download",
            "import cv2",
            "import mediapipe",
            "import numpy",
            "import requests",
            "import safetensors",
            "import socket",
            "import torch",
            "urllib.request",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()

"""Offline identity for the frozen UniGaze base inference bundle.

The calibration ledger needs an identity for the *pre-fit* predictor that
produced its raw gaze and screen prediction fields.  A distribution version or
the string ``before`` is not sufficient: checkpoint bytes, repository runtime
code/assets, installed UniGaze source, and relevant distribution versions all
affect that result.

This module only reads and hashes local files.  It does not import UniGaze,
Torch, MediaPipe, OpenCV, NumPy, or safetensors; it does not instantiate a model;
and it has no network fallback.  Missing local evidence fails closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from importlib import metadata
from pathlib import Path
from typing import Any

from core.unigaze_personalization.runtime_assets import (
    resolve_face_landmarker_asset,
)


SCHEMA_VERSION = 1
BUNDLE_TYPE = "lexigaze_frozen_base_inference_identity_v1"
MODEL_NAME = "unigaze_b16_joint"
HF_REPO_ID = "UniGaze/UniGaze-models"
HF_REVISION = "main"
CHECKPOINT_FILENAME = "unigaze_b16_joint.safetensors"
HF_REPO_CACHE_DIRECTORY = "models--UniGaze--UniGaze-models"
DEFAULT_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

REPOSITORY_RUNTIME_FILES = (
    "core/gaze_core/inference.py",
    "core/gaze_core/capture_contract.py",
    "core/unigaze_personalization/preprocess.py",
    "core/unigaze_personalization/runtime_assets.py",
    "core/unigaze_personalization/model.py",
    "core/unigaze_personalization/transforms.py",
    "core/unigaze_personalization/assets/face_model.txt",
)
FACE_LANDMARKER_LOGICAL_PATH = "web/static/face_landmarker.task"

RUNTIME_DISTRIBUTIONS = (
    "unigaze",
    "torch",
    "mediapipe",
    "opencv-python",
    "numpy",
    "safetensors",
)

LOWER_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
HF_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
VERSION_PATTERN = re.compile(r"^[^\x00-\x1f\x7f]{1,256}$")


class BaseInferenceBundleError(ValueError):
    """Raised when base-inference identity evidence is absent or inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return strict, portable JSON bytes used by every bundle hash."""

    _validate_json_value(value, location="$")
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise BaseInferenceBundleError("bundle is not strict canonical JSON") from exc
    return rendered.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def build_base_inference_bundle(
    *,
    repository_root: str | Path | None = None,
    hf_cache_root: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    resolved_snapshot_commit: str | None = None,
    unigaze_package_root: str | Path | None = None,
    version_provider: Callable[[str], str] | None = None,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Recompute the complete base predictor identity from local evidence.

    ``hf_cache_root`` is the Hugging Face ``hub`` directory, not ``HF_HOME``.
    Test and audit callers may inject an exact checkpoint and package root, but
    an injected checkpoint must still be the frozen file inside the resolved
    repository snapshot; arbitrary paths are rejected.
    """

    if local_files_only is not True:
        raise BaseInferenceBundleError(
            "local_files_only must be the literal boolean true; network is forbidden"
        )

    repo_root = _directory(
        repository_root or DEFAULT_REPOSITORY_ROOT,
        field="repository_root",
    )
    cache_root = _directory(
        hf_cache_root or _default_hf_cache_root(),
        field="hf_cache_root",
    )
    checkpoint = _resolve_checkpoint(
        cache_root,
        checkpoint_path=checkpoint_path,
        resolved_snapshot_commit=resolved_snapshot_commit,
    )
    package_root = (
        _directory(unigaze_package_root, field="unigaze_package_root")
        if unigaze_package_root is not None
        else _installed_unigaze_package_root()
    )
    provider = version_provider or metadata.version

    repository_files = [
        _hash_relative_file(repo_root, relative_path)
        for relative_path in REPOSITORY_RUNTIME_FILES
    ]
    try:
        face_landmarker_path = resolve_face_landmarker_asset(repo_root)
    except FileNotFoundError as exc:
        raise BaseInferenceBundleError(str(exc)) from exc
    repository_files.append(
        _hash_resolved_repository_file(
            repo_root,
            face_landmarker_path,
            logical_path=FACE_LANDMARKER_LOGICAL_PATH,
        )
    )
    package_files = _hash_python_package(package_root)
    runtime_versions = _runtime_versions(provider)
    resolved_commit = checkpoint["resolved_snapshot_commit"]
    model_id = f"{MODEL_NAME}@{resolved_commit}"

    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "offline_identity_complete",
        "bundle_type": BUNDLE_TYPE,
        "model_name": MODEL_NAME,
        "model_id": model_id,
        "resolved_revision": resolved_commit,
        "checkpoint_sha256": checkpoint["sha256"],
        "components": {
            "checkpoint": checkpoint,
            "repository_runtime": {
                "files": repository_files,
                "files_sha256": canonical_sha256(repository_files),
            },
            "installed_unigaze_package": {
                "distribution_name": "unigaze",
                "python_file_count": len(package_files),
                "python_files": package_files,
                "python_files_sha256": canonical_sha256(package_files),
            },
        },
        "package_versions": runtime_versions,
        "identity_contract": {
            "local_files_only": True,
            "network_allowed": False,
            "gpu_used": False,
            "model_loaded": False,
            "runtime_packages_imported": False,
            "hash_semantics": "unkeyed_reproducible_integrity_identity",
        },
    }
    digest = canonical_sha256(bundle)
    bundle["bundle_sha256"] = digest
    # The persistent acquisition store uses this exact bundle digest as its
    # calibration model SHA; it is not merely the checkpoint byte digest.
    bundle["model_sha256"] = digest
    return bundle


def verify_base_inference_bundle(
    bundle: Mapping[str, Any],
    *,
    repository_root: str | Path | None = None,
    hf_cache_root: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    resolved_snapshot_commit: str | None = None,
    unigaze_package_root: str | Path | None = None,
    version_provider: Callable[[str], str] | None = None,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Fail closed unless a stored bundle equals a fresh local recomputation."""

    if not isinstance(bundle, Mapping):
        raise BaseInferenceBundleError("base inference bundle must be an object")
    actual = deepcopy(dict(bundle))
    stored_bundle_sha = actual.get("bundle_sha256")
    stored_model_sha = actual.get("model_sha256")
    if not isinstance(stored_bundle_sha, str) or not LOWER_SHA256_PATTERN.fullmatch(
        stored_bundle_sha
    ):
        raise BaseInferenceBundleError("bundle SHA-256 is malformed")
    if not isinstance(stored_model_sha, str) or not LOWER_SHA256_PATTERN.fullmatch(
        stored_model_sha
    ):
        raise BaseInferenceBundleError("model SHA-256 is malformed")
    if stored_model_sha != stored_bundle_sha:
        raise BaseInferenceBundleError(
            "model SHA-256 must equal the complete bundle SHA-256"
        )
    core = deepcopy(actual)
    core.pop("bundle_sha256", None)
    core.pop("model_sha256", None)
    if canonical_sha256(core) != stored_bundle_sha:
        raise BaseInferenceBundleError("bundle SHA-256 mismatch")

    expected = build_base_inference_bundle(
        repository_root=repository_root,
        hf_cache_root=hf_cache_root,
        checkpoint_path=checkpoint_path,
        resolved_snapshot_commit=resolved_snapshot_commit,
        unigaze_package_root=unigaze_package_root,
        version_provider=version_provider,
        local_files_only=local_files_only,
    )
    if canonical_json_bytes(actual) != canonical_json_bytes(expected):
        raise BaseInferenceBundleError(
            "stored base inference bundle differs from fresh local evidence"
        )
    return {
        "status": "passed",
        "model_name": MODEL_NAME,
        "model_id": expected["model_id"],
        "model_sha256": expected["model_sha256"],
        "bundle_sha256": expected["bundle_sha256"],
        "resolved_snapshot_commit": expected["resolved_revision"],
        "checkpoint_sha256": expected["checkpoint_sha256"],
        "repository_file_count": len(
            expected["components"]["repository_runtime"]["files"]
        ),
        "unigaze_python_file_count": expected["components"][
            "installed_unigaze_package"
        ]["python_file_count"],
        "local_files_only": True,
        "gpu_used": False,
        "model_loaded": False,
    }


def _validate_json_value(value: Any, *, location: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BaseInferenceBundleError(f"{location} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise BaseInferenceBundleError(
                    f"{location} contains a non-string object key"
                )
            _validate_json_value(nested, location=f"{location}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        for index, nested in enumerate(value):
            _validate_json_value(nested, location=f"{location}[{index}]")
        return
    raise BaseInferenceBundleError(
        f"{location} contains unsupported value type {type(value).__name__}"
    )


def _default_hf_cache_root() -> Path:
    explicit_hub = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get(
        "HF_HUB_CACHE"
    )
    if explicit_hub:
        return Path(explicit_hub)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _directory(value: str | Path, *, field: str) -> Path:
    if isinstance(value, bool):
        raise BaseInferenceBundleError(f"{field} must be a directory path")
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise BaseInferenceBundleError(f"{field} does not exist") from exc
    if not path.is_dir():
        raise BaseInferenceBundleError(f"{field} must be a directory")
    return path


def _installed_unigaze_package_root() -> Path:
    try:
        distribution = metadata.distribution("unigaze")
        candidate = distribution.locate_file("unigaze")
    except metadata.PackageNotFoundError as exc:
        raise BaseInferenceBundleError(
            "installed unigaze distribution is unavailable"
        ) from exc
    return _directory(candidate, field="installed unigaze package root")


def _runtime_versions(provider: Callable[[str], str]) -> dict[str, str]:
    if not callable(provider):
        raise BaseInferenceBundleError("version_provider must be callable")
    versions: dict[str, str] = {}
    for distribution_name in RUNTIME_DISTRIBUTIONS:
        try:
            value = provider(distribution_name)
        except Exception as exc:
            raise BaseInferenceBundleError(
                f"distribution version unavailable: {distribution_name}"
            ) from exc
        if not isinstance(value, str) or not VERSION_PATTERN.fullmatch(value):
            raise BaseInferenceBundleError(
                f"distribution version is invalid: {distribution_name}"
            )
        if value != value.strip():
            raise BaseInferenceBundleError(
                f"distribution version is not normalized: {distribution_name}"
            )
        versions[distribution_name] = value
    return versions


def _resolve_checkpoint(
    cache_root: Path,
    *,
    checkpoint_path: str | Path | None,
    resolved_snapshot_commit: str | None,
) -> dict[str, Any]:
    repo_cache = _directory(
        cache_root / HF_REPO_CACHE_DIRECTORY,
        field="UniGaze Hugging Face repository cache",
    )
    commit = _resolved_commit(repo_cache, resolved_snapshot_commit)
    expected_lexical = (
        repo_cache / "snapshots" / commit / CHECKPOINT_FILENAME
    ).absolute()
    supplied_lexical = (
        Path(checkpoint_path).expanduser().absolute()
        if checkpoint_path is not None
        else expected_lexical
    )
    if os.path.normcase(os.path.normpath(str(supplied_lexical))) != os.path.normcase(
        os.path.normpath(str(expected_lexical))
    ):
        raise BaseInferenceBundleError(
            "checkpoint path escapes the resolved frozen snapshot"
        )
    try:
        resolved_path = supplied_lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BaseInferenceBundleError(
            "frozen UniGaze checkpoint is missing from the local cache"
        ) from exc
    if not resolved_path.is_file():
        raise BaseInferenceBundleError("frozen UniGaze checkpoint is not a file")
    if not resolved_path.is_relative_to(repo_cache):
        raise BaseInferenceBundleError(
            "checkpoint symlink or target escapes its Hugging Face repository cache"
        )
    return {
        "repo_id": HF_REPO_ID,
        "requested_revision": HF_REVISION,
        "resolved_snapshot_commit": commit,
        "filename": CHECKPOINT_FILENAME,
        "snapshot_relative_path": (
            Path("snapshots") / commit / CHECKPOINT_FILENAME
        ).as_posix(),
        "size_bytes": resolved_path.stat().st_size,
        "sha256": _file_sha256(resolved_path),
    }


def _resolved_commit(repo_cache: Path, injected: str | None) -> str:
    if injected is None:
        ref_path = _resolve_relative_file(repo_cache, Path("refs") / HF_REVISION)
        try:
            ref_text = ref_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise BaseInferenceBundleError(
                "Hugging Face main ref is unreadable in the local cache"
            ) from exc
        commit = ref_text.strip()
        if ref_text not in {commit, commit + "\n", commit + "\r\n"}:
            raise BaseInferenceBundleError("Hugging Face main ref is not normalized")
    else:
        if not isinstance(injected, str):
            raise BaseInferenceBundleError(
                "resolved_snapshot_commit must be a string"
            )
        commit = injected
    if not HF_COMMIT_PATTERN.fullmatch(commit):
        raise BaseInferenceBundleError(
            "resolved Hugging Face snapshot commit must be 40 lowercase hex characters"
        )
    return commit


def _resolve_relative_file(root: Path, relative_path: str | Path) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise BaseInferenceBundleError("relative file path escapes its root")
    lexical = root / relative
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BaseInferenceBundleError(
            f"required identity file is missing: {relative.as_posix()}"
        ) from exc
    if not resolved.is_relative_to(root):
        raise BaseInferenceBundleError(
            f"identity file escapes its root: {relative.as_posix()}"
        )
    if not resolved.is_file():
        raise BaseInferenceBundleError(
            f"required identity path is not a file: {relative.as_posix()}"
        )
    return resolved


def _hash_relative_file(root: Path, relative_path: str) -> dict[str, Any]:
    path = _resolve_relative_file(root, relative_path)
    return {
        "relative_path": Path(relative_path).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _hash_resolved_repository_file(
    repository_root: Path,
    path: Path,
    *,
    logical_path: str,
) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BaseInferenceBundleError(
            f"required identity file is missing: {logical_path}"
        ) from exc
    if not resolved.is_file() or not resolved.is_relative_to(repository_root):
        raise BaseInferenceBundleError(
            f"resolved identity asset escapes repository: {logical_path}"
        )
    return {
        "logical_path": logical_path,
        "relative_path": resolved.relative_to(repository_root).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _file_sha256(resolved),
    }


def _hash_python_package(package_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    lexical_files = sorted(
        package_root.rglob("*.py"),
        key=lambda path: path.relative_to(package_root).as_posix(),
    )
    if not lexical_files:
        raise BaseInferenceBundleError(
            "installed unigaze package contains no Python source files"
        )
    seen: set[str] = set()
    for lexical in lexical_files:
        relative = lexical.relative_to(package_root)
        resolved = _resolve_relative_file(package_root, relative)
        relative_text = relative.as_posix()
        if relative_text in seen:
            raise BaseInferenceBundleError(
                "installed unigaze package contains duplicate relative paths"
            )
        seen.add(relative_text)
        records.append(
            {
                "relative_path": relative_text,
                "size_bytes": resolved.stat().st_size,
                "sha256": _file_sha256(resolved),
            }
        )
    return records


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise BaseInferenceBundleError(f"identity file is unreadable: {path.name}") from exc
    return digest.hexdigest()


__all__ = [
    "BUNDLE_TYPE",
    "BaseInferenceBundleError",
    "CHECKPOINT_FILENAME",
    "FACE_LANDMARKER_LOGICAL_PATH",
    "HF_REPO_ID",
    "MODEL_NAME",
    "REPOSITORY_RUNTIME_FILES",
    "RUNTIME_DISTRIBUTIONS",
    "build_base_inference_bundle",
    "canonical_json_bytes",
    "canonical_sha256",
    "verify_base_inference_bundle",
]

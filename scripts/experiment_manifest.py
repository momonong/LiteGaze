"""Reproducibility manifests for LexiGaze benchmark scripts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "seaborn",
    "torch",
    "transformers",
)


def _run_command(command: list[str], root: Path, timeout: float = 5.0) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_metadata(root: Path) -> dict[str, Any]:
    commit = _run_command(["git", "rev-parse", "HEAD"], root)
    branch = _run_command(["git", "branch", "--show-current"], root)
    status = _run_command(["git", "status", "--porcelain"], root)
    tracked_status = _run_command(
        ["git", "status", "--porcelain", "--untracked-files=no"], root
    )
    tracked_diff = _run_command(
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD"], root
    )
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
        "tracked_changes": (
            bool(tracked_status) if tracked_status is not None else None
        ),
        "tracked_diff_from_head_sha256": (
            hashlib.sha256(tracked_diff.encode("utf-8")).hexdigest()
            if tracked_diff
            else None
        ),
    }


def _package_versions(packages: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _gpu_inventory(root: Path) -> list[dict[str, str]]:
    output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        root,
    )
    if not output:
        return []

    devices = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) == 3:
            devices.append(
                {
                    "name": fields[0],
                    "driver_version": fields[1],
                    "memory_total_mib": fields[2],
                }
            )
    return devices


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def fingerprint_files(
    paths: Iterable[str | Path], root: str | Path
) -> list[dict[str, Any]]:
    """Return stable metadata and hashes for inputs or generated artifacts."""
    project_root = Path(root).resolve()
    fingerprints = []
    for value in paths:
        path = Path(value)
        if not path.is_absolute():
            path = project_root / path
        exists = path.is_file()
        item: dict[str, Any] = {
            "path": _display_path(path, project_root),
            "exists": exists,
        }
        if exists:
            item.update({"size_bytes": path.stat().st_size, "sha256": _sha256(path)})
        fingerprints.append(item)
    return fingerprints


def _source_fingerprints(project_root: Path) -> list[dict[str, Any]]:
    """Fingerprint the manifest helper and the benchmark entry point."""
    candidates = [Path(__file__).resolve()]
    if sys.argv:
        entrypoint = Path(sys.argv[0])
        if not entrypoint.is_absolute():
            entrypoint = Path.cwd() / entrypoint
        if entrypoint.is_file():
            candidates.append(entrypoint.resolve())

    unique_candidates = list(dict.fromkeys(candidates))
    return fingerprint_files(unique_candidates, project_root)


def capture_source_snapshot(root: str | Path) -> dict[str, Any]:
    """Capture Git and entry-point state before an experiment writes outputs."""
    project_root = Path(root).resolve()
    source = _git_metadata(project_root)
    source["files"] = _source_fingerprints(project_root)
    return source


def build_experiment_manifest(
    experiment_name: str,
    *,
    root: str | Path,
    datasets: Iterable[str | Path] = (),
    artifacts: Iterable[str | Path] = (),
    config: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    seed: int | None = None,
    status: str = "completed",
    duration_seconds: float | None = None,
    packages: Iterable[str] = DEFAULT_PACKAGES,
    source_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable manifest without importing ML frameworks."""
    project_root = Path(root).resolve()
    source = (
        dict(source_snapshot)
        if source_snapshot is not None
        else capture_source_snapshot(project_root)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": {
            "name": experiment_name,
            "status": status,
            "created_at": datetime.now(UTC).isoformat(),
            "duration_seconds": duration_seconds,
            "seed": seed,
            "command": sys.argv,
        },
        "source": source,
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device_policy": os.environ.get("LEXIGAZE_DEVICE", "auto"),
            "gpus": _gpu_inventory(project_root),
            "packages": _package_versions(packages),
        },
        "datasets": fingerprint_files(datasets, project_root),
        "config": dict(config or {}),
        "metrics": dict(metrics or {}),
        "artifacts": fingerprint_files(artifacts, project_root),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"cannot serialize {type(value).__name__} in experiment manifest")


def write_experiment_manifest(
    output_path: str | Path,
    experiment_name: str,
    **manifest_kwargs: Any,
) -> Path:
    """Build and atomically write a manifest next to benchmark outputs."""
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_experiment_manifest(experiment_name, **manifest_kwargs)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                manifest,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
            handle.write("\n")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return destination

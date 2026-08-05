"""Aggregate-only provenance audit for independent gaze capture runs.

The audit intentionally reads only session metadata and JSONL manifests.  It
does not open participant images or videos.  Its purpose is to keep a direct
capture session and a video-extracted session from the same browser recording
out of opposite sides of a validation split.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaptureSession:
    session_id: str
    participant_id: str | None
    capture_run_id: str | None
    capture_source: str | None
    source_session_id: str | None
    start_unix: float | None
    end_unix: float | None
    sample_count: int
    video_derived: bool


@dataclass(frozen=True)
class IndependenceDiagnostics:
    sessions_seen: int
    manifests_seen: int
    manifest_lines: int
    malformed_lines: int
    provenance_conflicts: int
    source_sha256: str


@dataclass(frozen=True)
class IndependenceRequirements:
    min_repeat_labels: int = 5
    min_independent_capture_runs: int = 10
    legacy_independence_gap_seconds: float = 24.0 * 60.0 * 60.0

    def __post_init__(self) -> None:
        if self.min_repeat_labels <= 0:
            raise ValueError("min_repeat_labels must be positive")
        if self.min_independent_capture_runs < self.min_repeat_labels * 2:
            raise ValueError(
                "min_independent_capture_runs must permit two runs per label"
            )
        if (
            not isfinite(self.legacy_independence_gap_seconds)
            or self.legacy_independence_gap_seconds <= 0
        ):
            raise ValueError(
                "legacy_independence_gap_seconds must be finite and positive"
            )


@dataclass(frozen=True)
class IndependenceIssue:
    code: str
    severity: str
    message: str


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:128] if text else None


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _legacy_folder_time(session_id: str) -> float | None:
    try:
        parsed = datetime.strptime(session_id[:15], "%Y%m%d_%H%M%S")
    except (TypeError, ValueError):
        return None
    return parsed.timestamp()


def _consistent_value(values: list[str | None]) -> tuple[str | None, bool]:
    populated = {value for value in values if value}
    if not populated:
        return None, False
    return sorted(populated)[0], len(populated) > 1


def load_capture_sessions(
    sessions_root: Path,
) -> tuple[tuple[CaptureSession, ...], IndependenceDiagnostics]:
    """Load bounded provenance fields without opening image or video content."""

    root = Path(sessions_root)
    digest = hashlib.sha256()
    sessions: list[CaptureSession] = []
    sessions_seen = 0
    manifests_seen = 0
    manifest_lines = 0
    malformed_lines = 0
    provenance_conflicts = 0

    if root.is_dir():
        for session_dir in sorted(root.iterdir(), key=lambda path: path.name):
            if not session_dir.is_dir():
                continue
            sessions_seen += 1
            manifest_path = session_dir / "manifest.jsonl"
            if not manifest_path.is_file():
                continue
            manifests_seen += 1

            session_meta: Mapping[str, Any] = {}
            session_path = session_dir / "session.json"
            raw_session = b""
            if session_path.is_file():
                raw_session = session_path.read_bytes()
                try:
                    decoded = json.loads(raw_session.decode("utf-8"))
                    if isinstance(decoded, Mapping):
                        session_meta = decoded
                except (UnicodeDecodeError, json.JSONDecodeError):
                    malformed_lines += 1

            raw_manifest = manifest_path.read_bytes()
            for relative_name, raw_bytes in (
                (session_path.name, raw_session),
                (manifest_path.name, raw_manifest),
            ):
                digest.update(session_dir.name.encode("utf-8"))
                digest.update(b"\0")
                digest.update(relative_name.encode("utf-8"))
                digest.update(b"\0")
                digest.update(raw_bytes)
                digest.update(b"\0")

            timestamps: list[float] = []
            run_values = [_optional_text(session_meta.get("capture_run_id"))]
            source_values = [_optional_text(session_meta.get("capture_source"))]
            parent_values = [_optional_text(session_meta.get("source_session_id"))]
            sample_count = 0
            video_derived = any(session_dir.glob("raw_video.*"))

            for raw_line in raw_manifest.decode("utf-8", errors="replace").splitlines():
                if not raw_line.strip():
                    continue
                manifest_lines += 1
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    malformed_lines += 1
                    continue
                if not isinstance(record, Mapping):
                    malformed_lines += 1
                    continue
                sample_count += 1
                timestamp = _finite_float(record.get("created_at_unix"))
                if timestamp is not None:
                    timestamps.append(timestamp)
                run_values.append(_optional_text(record.get("capture_run_id")))
                source_values.append(_optional_text(record.get("capture_source")))
                parent_values.append(_optional_text(record.get("source_session_id")))
                video_derived = video_derived or record.get("extracted_from_video") is True

            capture_run_id, run_conflict = _consistent_value(run_values)
            capture_source, source_conflict = _consistent_value(source_values)
            source_session_id, parent_conflict = _consistent_value(parent_values)
            provenance_conflicts += int(run_conflict or source_conflict or parent_conflict)
            start_unix = min(timestamps) if timestamps else _legacy_folder_time(session_dir.name)
            end_unix = max(timestamps) if timestamps else start_unix
            sessions.append(
                CaptureSession(
                    session_id=session_dir.name,
                    participant_id=_optional_text(session_meta.get("participant_id")),
                    capture_run_id=capture_run_id,
                    capture_source=capture_source,
                    source_session_id=source_session_id,
                    start_unix=start_unix,
                    end_unix=end_unix,
                    sample_count=sample_count,
                    video_derived=video_derived,
                )
            )

    return (
        tuple(sessions),
        IndependenceDiagnostics(
            sessions_seen=sessions_seen,
            manifests_seen=manifests_seen,
            manifest_lines=manifest_lines,
            malformed_lines=malformed_lines,
            provenance_conflicts=provenance_conflicts,
            source_sha256=digest.hexdigest(),
        ),
    )


def summarize_capture_independence(
    sessions: tuple[CaptureSession, ...],
    diagnostics: IndependenceDiagnostics,
    *,
    requirements: IndependenceRequirements | None = None,
) -> dict[str, Any]:
    """Build identity-free capture-group counts for validation readiness."""

    requirements = requirements or IndependenceRequirements()
    parent = list(range(len(sessions)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_run: dict[str, list[int]] = defaultdict(list)
    by_session_id = {session.session_id: index for index, session in enumerate(sessions)}
    for index, session in enumerate(sessions):
        if session.capture_run_id:
            by_run[session.capture_run_id].append(index)
        if session.source_session_id in by_session_id:
            union(index, by_session_id[session.source_session_id])
    for indices in by_run.values():
        for index in indices[1:]:
            union(indices[0], index)

    legacy_links = 0
    legacy_by_label: dict[str, list[int]] = defaultdict(list)
    for index, session in enumerate(sessions):
        if session.participant_id and not session.capture_run_id:
            legacy_by_label[session.participant_id].append(index)
    for indices in legacy_by_label.values():
        ordered = sorted(
            indices,
            key=lambda index: (
                sessions[index].start_unix is None,
                sessions[index].start_unix or 0.0,
                sessions[index].session_id,
            ),
        )
        for left, right in zip(ordered, ordered[1:], strict=False):
            left_time = sessions[left].start_unix
            right_time = sessions[right].start_unix
            if left_time is None or right_time is None:
                continue
            if right_time - left_time < requirements.legacy_independence_gap_seconds:
                union(left, right)
                legacy_links += 1

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(sessions)):
        groups[find(index)].append(index)
    label_groups: dict[str, set[int]] = defaultdict(set)
    for index, session in enumerate(sessions):
        if session.participant_id:
            label_groups[session.participant_id].add(find(index))
    repeat_group_counts = [len(group_ids) for group_ids in label_groups.values() if len(group_ids) >= 2]
    shared_groups = [indices for indices in groups.values() if len(indices) >= 2]

    return {
        "sessions": len(sessions),
        "manifests_seen": diagnostics.manifests_seen,
        "manifest_lines": diagnostics.manifest_lines,
        "malformed_lines": diagnostics.malformed_lines,
        "provenance_conflicts": diagnostics.provenance_conflicts,
        "participant_label_count": len(label_groups),
        "explicit_provenance_sessions": sum(
            session.capture_run_id is not None for session in sessions
        ),
        "missing_provenance_sessions": sum(
            session.capture_run_id is None for session in sessions
        ),
        "video_derived_sessions": sum(session.video_derived for session in sessions),
        "capture_groups": len(groups),
        "shared_capture_groups": len(shared_groups),
        "sessions_in_shared_capture_groups": sum(map(len, shared_groups)),
        "legacy_links_applied": legacy_links,
        "repeat_labels": len(repeat_group_counts),
        "independent_capture_runs_for_repeat_labels": sum(repeat_group_counts),
        "max_capture_runs_per_label": max(
            (len(group_ids) for group_ids in label_groups.values()),
            default=0,
        ),
        "source_sha256": diagnostics.source_sha256,
    }

def audit_capture_independence(
    sessions: tuple[CaptureSession, ...],
    diagnostics: IndependenceDiagnostics,
    *,
    requirements: IndependenceRequirements | None = None,
) -> dict[str, Any]:
    requirements = requirements or IndependenceRequirements()
    summary = summarize_capture_independence(
        sessions,
        diagnostics,
        requirements=requirements,
    )
    issues: list[IndependenceIssue] = []
    if summary["provenance_conflicts"]:
        issues.append(
            IndependenceIssue(
                "PROVENANCE_CONFLICTS",
                "error",
                "One or more sessions contain conflicting capture provenance values.",
            )
        )
    if summary["malformed_lines"]:
        issues.append(
            IndependenceIssue(
                "MALFORMED_METADATA",
                "error",
                "One or more session or manifest records could not be parsed.",
            )
        )
    if summary["repeat_labels"] < requirements.min_repeat_labels:
        issues.append(
            IndependenceIssue(
                "INSUFFICIENT_REPEAT_LABELS",
                "error",
                (
                    f"Only {summary['repeat_labels']} labels have independent capture runs; "
                    f"at least {requirements.min_repeat_labels} are required."
                ),
            )
        )
    if (
        summary["independent_capture_runs_for_repeat_labels"]
        < requirements.min_independent_capture_runs
    ):
        issues.append(
            IndependenceIssue(
                "INSUFFICIENT_INDEPENDENT_CAPTURE_RUNS",
                "error",
                (
                    "Only "
                    f"{summary['independent_capture_runs_for_repeat_labels']} independent "
                    "runs belong to repeat labels; at least "
                    f"{requirements.min_independent_capture_runs} are required."
                ),
            )
        )
    if summary["missing_provenance_sessions"]:
        issues.append(
            IndependenceIssue(
                "MISSING_CAPTURE_PROVENANCE",
                "warning",
                (
                    f"{summary['missing_provenance_sessions']} sessions lack capture_run_id; "
                    "legacy sessions within the frozen time window are conservatively linked."
                ),
            )
        )
    if summary["shared_capture_groups"]:
        issues.append(
            IndependenceIssue(
                "SHARED_CAPTURE_ARTIFACTS",
                "warning",
                (
                    f"{summary['sessions_in_shared_capture_groups']} sessions collapse into "
                    f"{summary['shared_capture_groups']} shared capture groups."
                ),
            )
        )

    return {
        "status": (
            "ready"
            if not any(issue.severity == "error" for issue in issues)
            else "not_ready"
        ),
        "requirements": asdict(requirements),
        "summary": summary,
        "issues": [asdict(issue) for issue in issues],
    }

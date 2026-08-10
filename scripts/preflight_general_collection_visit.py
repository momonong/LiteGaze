"""Read-only, standard-library preflight for a paired localhost Visit 2.

The command never accepts or reads a plaintext invitation code.  It validates
the local registry by metadata and hashes, verifies the single localhost
listener and participant-safe HTTP surface, and reports manual camera/browser
gates separately from machine-verifiable readiness.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "experiment/webcam-gaze-measurement-ceiling-v1"
EXPECTED_SCOPE = "local_invited_self_development_unencrypted"
MANUAL_GATES = (
    "plaintext_visit_2_invite_secured_outside_git",
    "same_physical_device_camera_and_chromium_browser",
    "camera_permission_private_space_and_even_front_light",
    "viewport_at_least_1024x700_and_stable_after_system_check",
    "neutral_left_right_near_far_calibration_completed",
    "receipt_bound_start_and_end_five_point_validation_completed",
)
RUNTIME_FILES = (
    "core/participant_study/store.py",
    "web/__init__.py",
    "web/routes/study.py",
    "web/routes/gaze.py",
)
CAPTURE_CRITICAL_FILES = (
    *RUNTIME_FILES,
    "core/participant_study/general_collection_v1.json",
    "core/participant_study/general_collection_bank_v1.json",
    "core/gaze_core/participant_gaze_measurement_contract_v1.json",
    "web/static/participant_study.js",
    "web/static/participant_collection.js",
    "web/static/gaze_calibration_feedback.js",
)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _aware_utc(value: object, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value or ""))
    except ValueError as exc:
        raise ValueError(f"{field} is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} is not timezone-aware")
    return parsed.astimezone(UTC)


def _has_plaintext_invite_key(value: object) -> bool:
    if isinstance(value, Mapping):
        if "invite_code" in value:
            return True
        return any(_has_plaintext_invite_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_plaintext_invite_key(item) for item in value)
    return False


def _valid_sha256(value: object) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", str(value or "")) is not None


def _visit_completion_event(session: Mapping[str, object]) -> datetime:
    events = session.get("events")
    if not isinstance(events, list):
        raise ValueError("visit 1 completion event is unavailable")
    matches = [
        event
        for event in events
        if isinstance(event, Mapping)
        and event.get("event") == "general_collection_completed"
    ]
    if len(matches) != 1:
        raise ValueError("visit 1 must contain exactly one completion event")
    return _aware_utc(matches[0].get("at_utc"), field="visit 1 completion event")


def _safe_session_path(rehearsal_root: Path, session_id: object) -> Path:
    text = str(session_id or "")
    if re.fullmatch(r"ST-[A-F0-9]{20}", text) is None:
        raise ValueError("visit 1 session reference is invalid")
    root = rehearsal_root.resolve()
    candidate = (root / text / "session.json").resolve()
    if candidate.parent.parent != root:
        raise ValueError("visit 1 session reference escapes the study root")
    return candidate


def inspect_visit_two(
    code_root: Path,
    study_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, object]:
    """Inspect paired-session state without returning identifiers or secrets."""

    failures: list[str] = []
    warnings: list[str] = []
    active_now = (now or datetime.now(UTC)).astimezone(UTC)
    protocol_path = code_root / "core/participant_study/general_collection_v1.json"
    bank_path = code_root / "core/participant_study/general_collection_bank_v1.json"
    parent_protocol_path = code_root / "core/participant_study/protocol_v1.json"
    try:
        protocol = _read_json(protocol_path)
        bank = _read_json(bank_path)
        parent_protocol = _read_json(parent_protocol_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "failures": [f"local frozen design is unreadable: {type(exc).__name__}"],
            "warnings": warnings,
            "window": {"state": "unavailable"},
        }
    protocol_sha256 = _canonical_sha256(protocol)
    bank_sha256 = _canonical_sha256(bank)
    rehearsal_root = (
        study_root
        / "data"
        / "participant_studies"
        / str(parent_protocol.get("protocol_id") or "")
        / "rehearsals"
    )
    registry_path = rehearsal_root / "collection_invites.json"
    try:
        registry = _read_json(registry_path)
    except FileNotFoundError:
        return {
            "failures": ["collection invitation registry is unavailable"],
            "warnings": warnings,
            "window": {"state": "unavailable"},
        }
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "failures": [f"collection invitation registry is invalid: {type(exc).__name__}"],
            "warnings": warnings,
            "window": {"state": "unavailable"},
        }

    if not isinstance(registry, Mapping):
        failures.append("collection invitation registry must be an object")
        invites: list[Mapping[str, object]] = []
    else:
        raw_invites = registry.get("invites")
        invites = (
            [item for item in raw_invites if isinstance(item, Mapping)]
            if isinstance(raw_invites, list)
            else []
        )
        if not isinstance(raw_invites, list) or len(invites) != len(raw_invites):
            failures.append("collection invitation entries are malformed")
        if registry.get("protocol_sha256") != protocol_sha256:
            failures.append("registry protocol digest differs from frozen local design")
        if registry.get("bank_sha256") != bank_sha256:
            failures.append("registry bank digest differs from frozen local bank")
    if _has_plaintext_invite_key(registry):
        failures.append("registry contains a forbidden plaintext invite field")
    if len(invites) != 2:
        failures.append("self-only registry must contain exactly two visit invites")
    pair_ids = {str(item.get("pair_id") or "") for item in invites}
    if len(pair_ids) != 1 or not next(iter(pair_ids), ""):
        failures.append("registry must contain exactly one complete invite pair")
    by_visit = {item.get("visit_index"): item for item in invites}
    if set(by_visit) != {1, 2}:
        failures.append("registry must contain exactly Visit 1 and Visit 2")
        return {
            "failures": sorted(set(failures)),
            "warnings": warnings,
            "window": {"state": "unavailable"},
            "registry": {"pair_count": len(pair_ids), "invite_count": len(invites)},
        }

    first = by_visit[1]
    second = by_visit[2]
    for visit_index, invite in ((1, first), (2, second)):
        if not _valid_sha256(invite.get("code_sha256")):
            failures.append(f"Visit {visit_index} current invite hash is malformed")
        history = invite.get("code_rotation_history", [])
        if not isinstance(history, list) or any(
            not isinstance(item, Mapping)
            or not _valid_sha256(item.get("code_sha256"))
            for item in history
        ):
            failures.append(f"Visit {visit_index} rotation history is malformed")
            history = []
        if invite.get("code_sha256") in {
            item.get("code_sha256") for item in history if isinstance(item, Mapping)
        }:
            failures.append(f"Visit {visit_index} current invite hash was not rotated")
        rotation_count = invite.get("code_rotation_count", len(history))
        if rotation_count != len(history):
            failures.append(f"Visit {visit_index} rotation count is inconsistent")
    paired_fields = (
        "pair_id",
        "participant_id",
        "schedule_cell",
        "sequence",
        "order_cell",
        "protocol_sha256",
        "bank_sha256",
    )
    if any(first.get(field) != second.get(field) for field in paired_fields):
        failures.append("Visit 1 and Visit 2 frozen pair assignments differ")
    if first.get("form_id") == second.get("form_id"):
        failures.append("Visit 1 and Visit 2 must use alternate forms")
    if not first.get("used_at_utc") or not first.get("study_session_id"):
        failures.append("Visit 1 invite has not been consumed")
    if second.get("used_at_utc") or second.get("study_session_id"):
        failures.append("Visit 2 invite is already used")

    window: dict[str, object] = {"state": "unavailable"}
    device: dict[str, object] = {}
    target_artifacts_clean = False
    try:
        session_path = _safe_session_path(rehearsal_root, first.get("study_session_id"))
        session = _read_json(session_path)
        if not isinstance(session, Mapping):
            raise ValueError("visit 1 session must be an object")
        if session.get("state") != "completed":
            failures.append("Visit 1 session is not completed")
        assignment = session.get("collection_assignment")
        if not isinstance(assignment, Mapping) or any(
            assignment.get(field) != first.get(field)
            for field in (
                "pair_id",
                "schedule_cell",
                "sequence",
                "order_cell",
                "visit_index",
                "form_id",
                "protocol_sha256",
                "bank_sha256",
            )
            if field in first
        ):
            failures.append("Visit 1 session assignment differs from its registry entry")
        if session.get("participant_id") != first.get("participant_id"):
            failures.append("Visit 1 participant differs from its registry entry")
        completed_at = _visit_completion_event(session)
        sessions_spec = protocol.get("sessions")
        if not isinstance(sessions_spec, Mapping):
            raise ValueError("frozen visit interval is unavailable")
        minimum_hours = int(sessions_spec["minimum_interval_hours"])
        maximum_hours = int(sessions_spec["maximum_interval_hours"])
        earliest = completed_at + timedelta(hours=minimum_hours)
        latest = completed_at + timedelta(hours=maximum_hours)
        state = (
            "too_early"
            if active_now < earliest
            else "expired"
            if active_now > latest
            else "open"
        )
        window = {
            "basis": "visit_1_general_collection_completed_event",
            "minimum_interval_hours": minimum_hours,
            "maximum_interval_hours": maximum_hours,
            "visit_1_completed_at_utc": completed_at.isoformat(),
            "earliest_utc": earliest.isoformat(),
            "latest_utc": latest.isoformat(),
            "state": state,
        }
        quality = session.get("quality")
        system_check = (
            quality.get("general_system_check")
            if isinstance(quality, Mapping)
            else None
        )
        raw_device = (
            system_check.get("device")
            if isinstance(system_check, Mapping)
            else None
        )
        if isinstance(raw_device, Mapping):
            device = {
                field: raw_device.get(field)
                for field in (
                    "device_class",
                    "browser_family",
                    "viewport_width",
                    "viewport_height",
                    "device_pixel_ratio_bucket",
                    "camera_width",
                    "camera_height",
                    "estimated_camera_fps_band",
                )
            }
        if not device.get("device_class") or not device.get("browser_family"):
            failures.append("Visit 1 coarse device policy evidence is unavailable")
        calibration = quality.get("calibration") if isinstance(quality, Mapping) else None
        linked = session.get("linked_data")
        gaze_id = linked.get("gaze_session_id") if isinstance(linked, Mapping) else None
        sessions_root = (study_root / "data" / "sessions").resolve()
        gaze_text = str(gaze_id or "")
        if re.fullmatch(r"[A-Za-z0-9_-]+", gaze_text):
            gaze_root = (sessions_root / gaze_text).resolve()
            if gaze_root.parent != sessions_root:
                raise ValueError("Visit 1 gaze session reference escapes its root")
            artifact_dirs = [gaze_root / name for name in ("raw", "crop", "normalized_face")]
            target_artifacts_clean = all(not path.exists() for path in artifact_dirs)
        if not isinstance(calibration, Mapping) or calibration.get(
            "calibration_images_purged"
        ) is not True:
            target_artifacts_clean = False
        if not target_artifacts_clean:
            failures.append("Visit 1 linked calibration image artifacts are not purged")
        legacy_raw_count = sum(
            1
            for path in sessions_root.glob("*/raw")
            if path.is_dir()
        )
        if legacy_raw_count:
            warnings.append(
                "legacy raw directories exist outside the target linked session; "
                "they are warning-only and were not opened or deleted"
            )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        failures.append(f"Visit 1 evidence is invalid: {exc}")

    return {
        "failures": sorted(set(failures)),
        "warnings": sorted(set(warnings)),
        "window": window,
        "registry": {
            "schema_version": registry.get("schema_version"),
            "pair_count": len(pair_ids),
            "invite_count": len(invites),
            "visit_1_state": "completed" if first.get("used_at_utc") else "unused",
            "visit_2_state": "unused" if not second.get("used_at_utc") else "used",
            "visit_2_rotation_count": second.get("code_rotation_count", 0),
            "plaintext_invite_field_absent": not _has_plaintext_invite_key(registry),
            "frozen_protocol_sha256": protocol_sha256,
            "frozen_bank_sha256": bank_sha256,
        },
        "visit_1_device": device,
        "target_linked_calibration_artifacts_purged": target_artifacts_clean,
    }


def _run_text(command: list[str], *, timeout: float = 5.0) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return completed.stdout.strip()


def _git_directory(code_root: Path) -> Path:
    marker = code_root / ".git"
    if marker.is_dir():
        return marker
    if marker.is_file():
        text = marker.read_text(encoding="utf-8").strip()
        if not text.startswith("gitdir: "):
            raise ValueError("invalid .git indirection")
        return (code_root / text[8:]).resolve()
    raise ValueError("selected code root has no Git metadata")


def _git_ref(git_dir: Path, name: str) -> str:
    loose = git_dir / name
    if loose.is_file():
        value = loose.read_text(encoding="ascii").strip()
        if re.fullmatch(r"[0-9a-f]{40}", value):
            return value
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="ascii").splitlines():
            if line.startswith(("#", "^")):
                continue
            value, _, ref_name = line.partition(" ")
            if ref_name == name and re.fullmatch(r"[0-9a-f]{40}", value):
                return value
    raise ValueError(f"Git ref is unavailable: {name}")


def _git_object(git_dir: Path, oid: str) -> tuple[str, bytes]:
    path = git_dir / "objects" / oid[:2] / oid[2:]
    if not path.is_file():
        raise ValueError("capture preflight requires loose HEAD Git objects")
    decoded = zlib.decompress(path.read_bytes())
    header, separator, payload = decoded.partition(b"\0")
    if not separator:
        raise ValueError("Git object header is malformed")
    kind, _, size_text = header.partition(b" ")
    if int(size_text) != len(payload):
        raise ValueError("Git object size is inconsistent")
    return kind.decode("ascii"), payload


def _tree_entry(git_dir: Path, tree_oid: str, name: str) -> tuple[str, str]:
    kind, payload = _git_object(git_dir, tree_oid)
    if kind != "tree":
        raise ValueError("Git path traversal expected a tree")
    cursor = 0
    while cursor < len(payload):
        space = payload.index(b" ", cursor)
        nul = payload.index(b"\0", space + 1)
        mode = payload[cursor:space].decode("ascii")
        entry_name = payload[space + 1 : nul].decode("utf-8", errors="surrogateescape")
        oid = payload[nul + 1 : nul + 21].hex()
        cursor = nul + 21
        if entry_name == name:
            return mode, oid
    raise ValueError(f"Git HEAD does not contain capture file component: {name}")


def _head_blob(git_dir: Path, head_oid: str, relative: str) -> bytes:
    kind, commit = _git_object(git_dir, head_oid)
    if kind != "commit":
        raise ValueError("Git HEAD is not a commit")
    first_line = commit.splitlines()[0].decode("ascii")
    if not first_line.startswith("tree "):
        raise ValueError("Git commit tree is unavailable")
    current_oid = first_line[5:]
    parts = Path(relative).as_posix().split("/")
    for index, part in enumerate(parts):
        mode, current_oid = _tree_entry(git_dir, current_oid, part)
        if index < len(parts) - 1 and mode not in {"40000", "040000"}:
            raise ValueError(f"Git capture path is not a tree: {relative}")
    kind, blob = _git_object(git_dir, current_oid)
    if kind != "blob":
        raise ValueError(f"Git capture path is not a blob: {relative}")
    return blob


def probe_git(code_root: Path) -> dict[str, object]:
    git_dir = _git_directory(code_root)
    head_text = (git_dir / "HEAD").read_text(encoding="ascii").strip()
    if not head_text.startswith("ref: "):
        raise ValueError("detached HEAD is not allowed for capture")
    ref_name = head_text[5:]
    head_oid = _git_ref(git_dir, ref_name)
    mismatches: list[str] = []
    for relative in CAPTURE_CRITICAL_FILES:
        working = (code_root / relative).read_bytes()
        committed = _head_blob(git_dir, head_oid, relative)
        if working == committed:
            continue
        if b"\0" not in working and working.replace(b"\r\n", b"\n") == committed:
            continue
        mismatches.append(relative)
    return {
        "head": head_oid,
        "branch": ref_name.removeprefix("refs/heads/"),
        "capture_critical_files_match_head": not mismatches,
        "capture_critical_mismatch_count": len(mismatches),
    }


class _FileTime(ctypes.Structure):
    _fields_ = (("low", ctypes.c_uint32), ("high", ctypes.c_uint32))


def _process_started_at_utc(pid: int) -> datetime | None:
    if os.name != "nt":
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetProcessTimes.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
    )
    kernel32.GetProcessTimes.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return None
    try:
        created = _FileTime()
        exited = _FileTime()
        kernel = _FileTime()
        user = _FileTime()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            return None
        ticks = (created.high << 32) | created.low
        unix_seconds = ticks / 10_000_000 - 11_644_473_600
        return datetime.fromtimestamp(unix_seconds, UTC)
    finally:
        kernel32.CloseHandle(handle)


def probe_listener(host: str, port: int) -> dict[str, object]:
    if os.name != "nt":
        raise RuntimeError("single-listener enumeration is currently Windows-only")
    output = _run_text(["netstat", "-ano", "-p", "TCP"])
    rows: list[dict[str, object]] = []
    pattern = re.compile(
        r"^\s*TCP\s+(\S+):(\d+)\s+\S+\s+LISTENING\s+(\d+)\s*$",
        re.IGNORECASE,
    )
    for line in output.splitlines():
        match = pattern.match(line)
        if match and int(match.group(2)) == port:
            rows.append(
                {
                    "address": match.group(1),
                    "port": port,
                    "pid": int(match.group(3)),
                }
            )
    exact = [row for row in rows if row["address"] == host]
    exposed = [row for row in rows if row["address"] != host]
    started = (
        _process_started_at_utc(int(exact[0]["pid"])) if len(exact) == 1 else None
    )
    return {
        "listener_count": len(exact),
        "pid": exact[0]["pid"] if len(exact) == 1 else None,
        "started_at_utc": started.isoformat() if started else None,
        "unexpected_bindings": len(exposed),
    }


def http_get(url: str, *, timeout: float = 3.0) -> tuple[int, dict[str, str], bytes]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, method="GET")
    try:
        with opener.open(request, timeout=timeout) as response:
            return (
                int(response.status),
                {key.lower(): value for key, value in response.headers.items()},
                response.read(),
            )
    except urllib.error.HTTPError as exc:
        return (
            int(exc.code),
            {key.lower(): value for key, value in exc.headers.items()},
            exc.read(),
        )


def _json_body(body: bytes, *, endpoint: str) -> Mapping[str, object]:
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{endpoint} returned invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{endpoint} JSON must be an object")
    return value


def probe_service(
    code_root: Path,
    study_root: Path,
    base_url: str,
    *,
    getter: Callable[[str], tuple[int, dict[str, str], bytes]] = http_get,
) -> dict[str, object]:
    failures: list[str] = []

    def get(path: str) -> tuple[int, dict[str, str], bytes]:
        status, headers, body = getter(base_url.rstrip("/") + path)
        if "no-store" not in headers.get("cache-control", "").lower():
            failures.append(f"{path} does not disable caching")
        return status, headers, body

    health_status, _, health_body = get("/api/gaze/health")
    if health_status != 200:
        failures.append("participant-safe gaze health is unavailable")
    else:
        health = _json_body(health_body, endpoint="gaze health")
        if set(health) != {"ok"} or health.get("ok") is not True:
            failures.append("participant-safe gaze health exposes unexpected fields")

    protocol_status, _, protocol_body = get("/api/study/protocol")
    protocol_digest = None
    if protocol_status != 200:
        failures.append("public study protocol is unavailable")
    else:
        envelope = _json_body(protocol_body, endpoint="study protocol")
        public = envelope.get("protocol")
        if not isinstance(public, Mapping):
            failures.append("public study protocol payload is malformed")
        else:
            activation = public.get("activation")
            governance = public.get("data_governance")
            expected_data = str((study_root / "data").resolve())
            if not isinstance(activation, Mapping) or not isinstance(
                governance, Mapping
            ):
                failures.append("public activation/governance payload is malformed")
            else:
                expected_activation = {
                    "configured_mode": "rehearsal",
                    "rehearsal_ready": True,
                    "pilot_ready": False,
                    "rehearsal_scope": EXPECTED_SCOPE,
                    "rehearsal_self_only": True,
                }
                if any(
                    activation.get(key) != value
                    for key, value in expected_activation.items()
                ):
                    failures.append("public rehearsal activation is not capture-safe")
                if (
                    str(governance.get("location") or "") != expected_data
                    or governance.get("raw_frame_retention_hours") != 1
                    or governance.get("retention_policy")
                    != "manual_until_researcher_deletes"
                    or governance.get("self_only") is not True
                    or governance.get("formal_promotion_allowed") is not False
                ):
                    failures.append("public data governance differs from the runbook")
            protocol_digest = public.get("protocol_digest_sha256")

    general_status, _, general_body = get(
        "/api/study/general-collection/protocol"
    )
    design_digests: dict[str, object] = {}
    if general_status != 200:
        failures.append("general collection protocol is unavailable")
    else:
        envelope = _json_body(general_body, endpoint="general collection protocol")
        design = envelope.get("design_audit")
        local_protocol = _read_json(
            code_root / "core/participant_study/general_collection_v1.json"
        )
        local_bank = _read_json(
            code_root / "core/participant_study/general_collection_bank_v1.json"
        )
        if not isinstance(design, Mapping) or design.get("ok") is not True:
            failures.append("served general collection design is invalid")
        else:
            design_digests = {
                "protocol_sha256": design.get("protocol_sha256"),
                "bank_sha256": design.get("bank_sha256"),
            }
            if design.get("protocol_sha256") != _canonical_sha256(local_protocol):
                failures.append("served general protocol differs from local code root")
            if design.get("bank_sha256") != _canonical_sha256(local_bank):
                failures.append("served general bank differs from local code root")

    study_status, _, study_body = get("/study")
    if study_status != 200 or b'id="inviteCode"' not in study_body or b"startAnotherInviteBtn" not in study_body:
        failures.append("study page lacks the current Visit handoff controls")
    asset_matches: dict[str, bool] = {}
    for asset in (
        "participant_study.js",
        "participant_collection.js",
        "gaze_calibration_feedback.js",
    ):
        status, _, served = get(f"/static/{asset}")
        local = (code_root / "web" / "static" / asset).read_bytes()
        matched = status == 200 and hashlib.sha256(served).digest() == hashlib.sha256(
            local
        ).digest()
        asset_matches[asset] = matched
        if not matched:
            failures.append(f"served {asset} differs from the selected code root")
    blocked_statuses: dict[str, int] = {}
    for path in ("/api/gaze/datasets", "/api/ping", "/api/gaze/health/extra"):
        status, _, _ = get(path)
        blocked_statuses[path] = status
        if status != 403:
            failures.append(f"public surface did not fail closed for {path}")
    return {
        "failures": sorted(set(failures)),
        "health_exact": health_status == 200 and not any(
            "gaze health" in failure for failure in failures
        ),
        "protocol_digest_sha256": protocol_digest,
        "general_design": design_digests,
        "asset_matches": asset_matches,
        "blocked_statuses": blocked_statuses,
    }


def build_preflight(
    code_root: Path,
    study_root: Path,
    *,
    base_url: str = "http://127.0.0.1:8098",
    expected_head: str | None = None,
    now: datetime | None = None,
    git_probe: Callable[[Path], dict[str, object]] = probe_git,
    listener_probe: Callable[[str, int], dict[str, object]] = probe_listener,
    getter: Callable[[str], tuple[int, dict[str, str], bytes]] = http_get,
) -> dict[str, object]:
    code_root = code_root.resolve()
    study_root = study_root.resolve()
    active_now = (now or datetime.now(UTC)).astimezone(UTC)
    failures: list[str] = []
    warnings: list[str] = []
    base_url_valid = base_url == "http://127.0.0.1:8098"
    if not base_url_valid:
        failures.append("base URL must be exactly loopback HTTP on port 8098")

    try:
        git = git_probe(code_root)
        if git.get("branch") != EXPECTED_BRANCH:
            failures.append("selected code root is not on the dedicated experiment branch")
        if git.get("capture_critical_files_match_head") is not True:
            failures.append("capture-critical files differ from Git HEAD")
        if expected_head and git.get("head") != expected_head:
            failures.append("selected experiment commit differs from --expected-head")
    except (OSError, ValueError, zlib.error, subprocess.SubprocessError) as exc:
        git = {"available": False}
        failures.append(f"git preflight failed: {type(exc).__name__}")

    try:
        listener_first = listener_probe("127.0.0.1", 8098)
        time.sleep(0.05)
        listener_second = listener_probe("127.0.0.1", 8098)
        listener = listener_second
        if listener_first != listener_second:
            failures.append("localhost listener changed during the preflight")
        if listener.get("listener_count") != 1:
            failures.append("exactly one 127.0.0.1:8098 listener is required")
        if listener.get("unexpected_bindings") != 0:
            failures.append("port 8098 has a non-loopback or unexpected binding")
        started_text = listener.get("started_at_utc")
        if started_text:
            started = _aware_utc(started_text, field="listener start time")
            newest_runtime = max(
                (code_root / relative).stat().st_mtime for relative in RUNTIME_FILES
            )
            newest_runtime_at = datetime.fromtimestamp(newest_runtime, UTC)
            listener["newest_runtime_source_mtime_utc"] = newest_runtime_at.isoformat()
            listener["started_after_runtime_sources"] = started > newest_runtime_at
            if started <= newest_runtime_at:
                failures.append("listener predates changed runtime source; restart it")
        else:
            failures.append("listener process start time is unavailable")
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        listener = {"available": False}
        failures.append(f"listener preflight failed: {type(exc).__name__}")

    visit = inspect_visit_two(code_root, study_root, now=active_now)
    failures.extend(str(item) for item in visit.get("failures", []))
    warnings.extend(str(item) for item in visit.get("warnings", []))
    if base_url_valid:
        try:
            service = probe_service(
                code_root,
                study_root,
                base_url,
                getter=getter,
            )
            failures.extend(str(item) for item in service.get("failures", []))
        except (OSError, ValueError, urllib.error.URLError) as exc:
            service = {"available": False}
            failures.append(f"HTTP preflight failed: {type(exc).__name__}")
    else:
        service = {"available": False, "skipped_non_loopback_url": True}

    window = visit.get("window", {})
    window_state = window.get("state") if isinstance(window, Mapping) else None
    if window_state == "expired":
        failures.append("Visit 2 frozen completion-anchored interval has expired")
    failures = sorted(set(failures))
    warnings = sorted(set(warnings))
    status = (
        "failed"
        if failures
        else "waiting_for_window"
        if window_state == "too_early"
        else "machine_ready"
        if window_state == "open"
        else "failed"
    )
    return {
        "schema_version": 1,
        "audit_type": "general_collection_visit_preflight_v1",
        "checked_at_utc": active_now.isoformat(),
        "status": status,
        "code": git,
        "listener": listener,
        "service": service,
        "visit": {
            key: value
            for key, value in visit.items()
            if key not in {"failures", "warnings"}
        },
        "failures": failures,
        "warnings": warnings,
        "manual_gates_required": list(MANUAL_GATES),
        "privacy": {
            "plaintext_invite_read": False,
            "plaintext_invite_output": False,
            "registry_or_session_write_performed": False,
            "legacy_raw_deleted": False,
        },
        "compute": {
            "standard_library_only": True,
            "model_inference_used": False,
            "gpu_used": False,
            "torch_imported": "torch" in sys.modules,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, default=ROOT)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8098")
    parser.add_argument("--expected-head")
    args = parser.parse_args()
    result = build_preflight(
        args.code_root,
        args.study_root,
        base_url=args.base_url,
        expected_head=args.expected_head,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] == "machine_ready":
        return 0
    if result["status"] == "waiting_for_window":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

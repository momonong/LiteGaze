"""Persistent, fail-closed runner state for the frozen 193-sample capture.

This store is deliberately separate from participant-study state and from the
legacy five-point calibration flow.  It owns server-authoritative scheduling,
single-use challenges, derived-sample ledgers, and acquisition-artifact
integrity only.  It never stores raw frames and never authorizes a physical,
accuracy, reading, or cognitive claim.

The caller remains responsible for ephemeral frame processing, model training,
and image purging.  A typical adapter saves/processes a frame with
``sample_store``, turns the result of ``inference.predict`` into the flat
observation accepted here, trains after calibration is sealed, binds the exact
model digest, and finally purges the image-bearing calibration session.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shutil
import threading
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import measurement_schedule as _schedule
from .capture_contract import compare_capture_contracts, normalize_capture_contract


STATE_SCHEMA_VERSION = 1
STATE_TYPE = "webcam_gaze_measurement_ceiling_persistent_run_state_v1"
LEDGER_RECORD_TYPE = "webcam_gaze_measurement_ceiling_derived_sample_v1"
STORE_RELATIVE_PATH = Path("data") / "measurement_ceiling_runs"
RUN_MANIFEST_FILENAME = "run_manifest.json"
STATE_FILENAME = "state.json"
CALIBRATION_LEDGER_FILENAME = "calibration_ledger.jsonl"
EVALUATION_LEDGER_FILENAME = "evaluation_ledger.jsonl"
CAPTURE_ARTIFACT_FILENAME = "capture_artifact.json"
CREATE_REGISTRY_FILENAME = "create_registry.json"
CREATE_REGISTRY_TYPE = "webcam_gaze_measurement_ceiling_create_registry_v1"
ATTEMPT_SIDECAR_FILENAME = "attempt_sidecar.json"
ATTEMPT_SIDECAR_TYPE = "webcam_gaze_measurement_ceiling_attempt_sidecar_v1"

PHASE_SCHEDULED = "scheduled"
PHASE_CALIBRATION_COLLECTING = "calibration_collecting"
PHASE_CALIBRATION_SEALED = "calibration_sealed"
PHASE_MODEL_BOUND = "model_bound"
PHASE_EVALUATION_COLLECTING = "evaluation_collecting"
PHASE_CAPTURE_SEALED = "capture_sealed"
PHASE_ARTIFACT_VERIFIED = "artifact_verified"
PHASE_ABORTED = "aborted"
PHASE_FAILED_INTEGRITY = "failed_integrity"

ALL_PHASES = frozenset(
    {
        PHASE_SCHEDULED,
        PHASE_CALIBRATION_COLLECTING,
        PHASE_CALIBRATION_SEALED,
        PHASE_MODEL_BOUND,
        PHASE_EVALUATION_COLLECTING,
        PHASE_CAPTURE_SEALED,
        PHASE_ARTIFACT_VERIFIED,
        PHASE_ABORTED,
        PHASE_FAILED_INTEGRITY,
    }
)
TERMINAL_PHASES = frozenset(
    {PHASE_ARTIFACT_VERIFIED, PHASE_ABORTED, PHASE_FAILED_INTEGRITY}
)

LOWER_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SAFE_RUN_ID_PATTERN = re.compile(r"^WGMC-[A-Za-z0-9][A-Za-z0-9._:-]{0,122}$")
NORMALIZED_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,127}$")
CREATE_REQUEST_ID_PATTERN = re.compile(r"^WGMCREQ-[0-9a-f]{32}$")
CLIENT_RUN_TOKEN_PATTERN = re.compile(r"^wgmr_client_[0-9a-f]{64}$")
SPOOL_FILENAME_PATTERN = re.compile(r"^ch-[0-9a-f]{24}\.aesgcm$")
EMPTY_FILE_SHA256 = hashlib.sha256(b"").hexdigest()
DEDICATED_IMAGE_DIRECTORIES = ("raw", "crop", "normalized_face")

# Exact keys are denied recursively.  Substring matching would incorrectly
# reject legitimate fields such as ``camera_frame_rate``.
RAW_MEDIA_KEYS = frozenset(
    {
        "base64",
        "blob",
        "canvas_data_url",
        "data_url",
        "frame",
        "frame_base64",
        "frame_bytes",
        "image",
        "image_base64",
        "image_bytes",
        "image_data",
        "jpeg",
        "png",
        "raw_frame",
        "raw_image",
        "raw_media",
        "raw_video",
        "video",
        "webcam_frame",
    }
)

LEDGER_FILES = {
    "calibration": CALIBRATION_LEDGER_FILENAME,
    "evaluation": EVALUATION_LEDGER_FILENAME,
}
EXPECTED_ROLE_COUNTS = {
    "calibration": _schedule.EXPECTED_CALIBRATION_SAMPLE_COUNT,
    "evaluation": _schedule.EXPECTED_EVALUATION_SAMPLE_COUNT,
}


class MeasurementRunError(RuntimeError):
    """Base class for persistent-run failures."""


class MeasurementRunAuthenticationError(MeasurementRunError):
    """Raised when the one-time-returned run token does not authenticate."""


class MeasurementRunStateError(MeasurementRunError):
    """Raised when an operation is invalid for the current phase."""


class MeasurementRunChallengeError(MeasurementRunError):
    """Raised for an absent, stale, or otherwise unusable challenge."""


class MeasurementRunValidationError(MeasurementRunError, ValueError):
    """Raised before persistence when a derived observation is invalid."""


class MeasurementRunIntegrityError(MeasurementRunError):
    """Raised when persisted state, schedule, ledger, or artifact fails closed."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _secret_token(prefix: str, byte_count: int = 32) -> str:
    material = base64.urlsafe_b64encode(secrets.token_bytes(byte_count)).decode("ascii")
    return prefix + material.rstrip("=")


def _token_sha256(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _canonical_line(payload: Mapping[str, Any]) -> bytes:
    return _schedule.canonical_json_bytes(payload) + b"\n"


def _state_without_hash(state: Mapping[str, Any]) -> dict[str, Any]:
    core = deepcopy(dict(state))
    core.pop("state_sha256", None)
    return core


def _record_without_hash(record: Mapping[str, Any]) -> dict[str, Any]:
    core = deepcopy(dict(record))
    core.pop("record_sha256", None)
    return core


def _registry_without_hash(registry: Mapping[str, Any]) -> dict[str, Any]:
    core = deepcopy(dict(registry))
    core.pop("registry_sha256", None)
    return core


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _assert_no_raw_media(value: Any, *, location: str = "observation") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = _normalized_key(key)
            if normalized in RAW_MEDIA_KEYS:
                raise MeasurementRunValidationError(
                    f"{location} contains forbidden raw-media field {key!r}"
                )
            _assert_no_raw_media(nested, location=f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, nested in enumerate(value):
            _assert_no_raw_media(nested, location=f"{location}[{index}]")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raise MeasurementRunValidationError(
            f"{location} contains forbidden raw binary media"
        )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = _schedule.deterministic_json(payload).encode("utf-8")
    _atomic_bytes(path, rendered)


@contextmanager
def _exclusive_file_lock(lock_path: Path):
    """Serialize one run across threads and OS processes."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    locked = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            locked = True
        except OSError as exc:
            raise MeasurementRunStateError("measurement run is busy; retry") from exc
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise MeasurementRunIntegrityError(f"{field} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementRunIntegrityError(f"{field} must be finite") from exc
    if not math.isfinite(number):
        raise MeasurementRunIntegrityError(f"{field} must be finite")
    return number


class MeasurementRunStore:
    """Atomic store for one dedicated self-development acquisition namespace."""

    _process_lock = threading.RLock()

    def __init__(
        self,
        root: Path,
        *,
        protocol_path: str | Path | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.store_root = (self.root / STORE_RELATIVE_PATH).resolve()
        expected_parent = (self.root / "data").resolve()
        if self.store_root.parent != expected_parent:
            raise MeasurementRunStateError("invalid measurement-run storage root")
        self.protocol_path = Path(protocol_path).resolve() if protocol_path else None

    def lookup_create_request(
        self, *, create_request_id: str, run_token: str
    ) -> dict[str, Any]:
        """Look up an idempotent create authority without allocating a run.

        The client-generated values are authenticated secrets.  Only their
        SHA-256 digests are persisted.  This method lets an HTTP layer recover
        a response-lost create *before* consuming another in-memory camera
        preflight receipt.
        """

        request = self._normalized_create_request_id(create_request_id)
        token = self._normalized_client_run_token(run_token)
        request_sha = _token_sha256(request)
        token_sha = _token_sha256(token)
        with self._process_lock:
            self.store_root.mkdir(parents=True, exist_ok=True)
            with _exclusive_file_lock(self.store_root / ".registry.lock"):
                registry = self._read_create_registry_unlocked()
                entry = self._find_create_entry_unlocked(
                    registry,
                    request_sha=request_sha,
                    token_sha=token_sha,
                    repair_registry=True,
                )
                if entry is None:
                    return {
                        "ok": True,
                        "exists": False,
                        "idempotent": False,
                        "measurement_claim_authorized": False,
                    }
                state = self._read_state_unlocked(
                    self._run_dir(str(entry["capture_run_id"]))
                )
                public = self._public_state(state)
                public.update(
                    {
                        "exists": True,
                        "idempotent": True,
                        "created_new": False,
                        "run_token_client_supplied": True,
                        "run_token_returned_once": False,
                    }
                )
                return public

    def create_run(
        self, *, create_request_id: str, run_token: str
    ) -> dict[str, Any]:
        """Create or recover one run from a client-owned create authority.

        A response loss is safe: the same request ID and 256-bit secret returns
        the same run across processes and restarts.  Reusing a request ID with
        another secret fails authentication.  Plaintext authorities are never
        written to disk.
        """

        request = self._normalized_create_request_id(create_request_id)
        token = self._normalized_client_run_token(run_token)
        request_sha = _token_sha256(request)
        token_sha = _token_sha256(token)

        with self._process_lock:
            self.store_root.mkdir(parents=True, exist_ok=True)
            with _exclusive_file_lock(self.store_root / ".registry.lock"):
                registry = self._read_create_registry_unlocked()
                existing = self._find_create_entry_unlocked(
                    registry,
                    request_sha=request_sha,
                    token_sha=token_sha,
                    repair_registry=True,
                )
                if existing is not None:
                    state = self._read_state_unlocked(
                        self._run_dir(str(existing["capture_run_id"]))
                    )
                    public = self._public_state(state)
                    public.update(
                        {
                            "run_token_client_supplied": True,
                            "run_token_returned_once": False,
                            "created_new": False,
                            "idempotent": True,
                        }
                    )
                    return public
                capture_run_id = self._new_capture_run_id()
                manifest = _schedule.build_run_manifest(
                    capture_run_id, protocol_path=self.protocol_path
                )
                now = _utc_now()
                state = {
                    "schema_version": STATE_SCHEMA_VERSION,
                    "state_type": STATE_TYPE,
                    "capture_run_id": capture_run_id,
                    "created_at_utc": now,
                    "updated_at_utc": now,
                    "phase": PHASE_SCHEDULED,
                    "measurement_claim_authorized": False,
                    "physical_capture_claim_authorized": False,
                    "acquisition_artifact_verified": False,
                    "capture_contract_binding_verified": False,
                    "run_token_sha256": token_sha,
                    "create_request_sha256": request_sha,
                    "protocol_sha256": manifest["protocol"]["canonical_sha256"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "manifest_rows_sha256": manifest["rows_sha256"],
                    "expected_counts": deepcopy(manifest["expected_counts"]),
                    "progress": {
                        "next_sequence_index": 0,
                        "calibration_count": 0,
                        "evaluation_count": 0,
                    },
                    "ledgers": {
                        role: {
                            "count": 0,
                            "head_sha256": None,
                            "file_sha256": EMPTY_FILE_SHA256,
                            "sealed_sha256": None,
                        }
                        for role in LEDGER_FILES
                    },
                    "active_challenge": None,
                    "last_consumption": None,
                    "calibration_model_binding": None,
                    "model_binding": None,
                    "runner": {
                        "runtime_binding": None,
                        "calibration_write": None,
                        "calibration_manifest_bindings": [],
                        "capture_contract_proofs": [],
                        "frame_spool": None,
                        "inference_intent": None,
                        "base_bundle_checks": [],
                        "training_artifact_intent": None,
                        "trained_artifact": None,
                        "calibration_image_purge": None,
                    },
                    "capture_artifact": None,
                    "failure": None,
                    "abort": None,
                }
                state["state_sha256"] = _schedule.canonical_sha256(state)

                run_dir = self._run_dir(capture_run_id)
                staging = self.store_root / f".creating-{secrets.token_hex(16)}"
                staging.mkdir(parents=False, exist_ok=False)
                try:
                    _atomic_json(staging / RUN_MANIFEST_FILENAME, manifest)
                    _atomic_bytes(staging / CALIBRATION_LEDGER_FILENAME, b"")
                    _atomic_bytes(staging / EVALUATION_LEDGER_FILENAME, b"")
                    _atomic_json(staging / STATE_FILENAME, state)
                    os.replace(staging, run_dir)
                finally:
                    if staging.exists():
                        # A failed create has never returned a token or run id.
                        for child in staging.iterdir():
                            child.unlink()
                        staging.rmdir()

                # State is intentionally durable first.  If the process dies
                # here, the next authenticated retry recovers it by scanning
                # the hashed creation binding and repairs this registry.
                registry["entries"].append(
                    {
                        "create_request_sha256": request_sha,
                        "run_token_sha256": token_sha,
                        "capture_run_id": capture_run_id,
                        "created_at_utc": now,
                    }
                )
                self._write_create_registry_unlocked(registry)

        public = self._public_state(state)
        public.update(
            {
                "ok": True,
                "run_token_client_supplied": True,
                "run_token_returned_once": False,
                "created_new": True,
                "idempotent": False,
            }
        )
        return public

    def get_status(self, capture_run_id: str, run_token: str) -> dict[str, Any]:
        with self._locked_run(capture_run_id) as run_dir:
            state, _, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            return self._public_state(state)

    @contextmanager
    def adapter_operation_lock(self, capture_run_id: str):
        """Serialize an adapter operation that spans persistence callbacks.

        Store methods retain their narrower state lock.  This second lock is
        intentionally distinct so a runner can safely call inference,
        ``save_sample``, training, or purge between store transactions while
        excluding a second process from repeating the same external effect.
        """

        normalized = self._normalized_run_id(capture_run_id)
        run_dir = self._run_dir(normalized)
        if not run_dir.is_dir():
            raise MeasurementRunStateError("measurement run not found")
        with _exclusive_file_lock(run_dir / ".adapter.lock"):
            yield

    def bind_runtime_context(
        self,
        capture_run_id: str,
        run_token: str,
        *,
        calibration_session_id: str,
        capture_contract: Mapping[str, Any],
        viewport_width: float,
        viewport_height: float,
        device_pixel_ratio: float,
        base_model_id: str,
        base_model_name: str,
        base_bundle_sha256: str,
        base_model_sha256: str,
        base_checkpoint_sha256: str,
        readiness_preflight: Mapping[str, Any],
        base_inference_selector: str = "before",
    ) -> dict[str, Any]:
        """Bind immutable, non-participant runtime identity before capture."""

        session_id = self._normalized_identifier(
            calibration_session_id, field="calibration_session_id"
        )
        model_id = self._normalized_identifier(base_model_id, field="base_model_id")
        model_name = self._normalized_identifier(
            base_model_name, field="base_model_name"
        )
        inference_selector = self._normalized_identifier(
            base_inference_selector, field="base_inference_selector"
        )
        bundle_sha = self._normalized_sha256(
            base_bundle_sha256, field="base_bundle_sha256"
        )
        model_sha = self._normalized_sha256(
            base_model_sha256, field="base_model_sha256"
        )
        checkpoint_sha = self._normalized_sha256(
            base_checkpoint_sha256, field="base_checkpoint_sha256"
        )
        if not isinstance(readiness_preflight, Mapping):
            raise MeasurementRunValidationError(
                "readiness_preflight must be an object"
            )
        preflight = deepcopy(dict(readiness_preflight))
        _assert_no_raw_media(preflight, location="readiness_preflight")
        try:
            _schedule.canonical_json_bytes(preflight)
        except _schedule.MeasurementScheduleError as exc:
            raise MeasurementRunValidationError(str(exc)) from exc
        if not isinstance(capture_contract, Mapping):
            raise MeasurementRunValidationError("capture_contract must be an object")
        try:
            normalized_contract = normalize_capture_contract(capture_contract)
        except ValueError as exc:
            raise MeasurementRunValidationError(str(exc)) from exc
        _assert_no_raw_media(normalized_contract, location="capture_contract")
        try:
            _schedule.canonical_json_bytes(normalized_contract)
        except _schedule.MeasurementScheduleError as exc:
            raise MeasurementRunValidationError(str(exc)) from exc
        viewport = {
            "width": self._positive_finite(viewport_width, field="viewport_width"),
            "height": self._positive_finite(viewport_height, field="viewport_height"),
            "device_pixel_ratio": self._positive_finite(
                device_pixel_ratio, field="device_pixel_ratio"
            ),
        }
        binding = {
            "calibration_session_id": session_id,
            "capture_contract": normalized_contract,
            "capture_contract_sha256": _schedule.canonical_sha256(
                normalized_contract
            ),
            "viewport": viewport,
            "base_inference_bundle": {
                "model_id": model_id,
                "model_name": model_name,
                "inference_selector": inference_selector,
                "bundle_sha256": bundle_sha,
                "model_sha256": model_sha,
                "checkpoint_sha256": checkpoint_sha,
            },
            "readiness_preflight": preflight,
            "readiness_preflight_sha256": _schedule.canonical_sha256(preflight),
        }
        with self._locked_run(capture_run_id) as run_dir:
            state, _, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            self._require_active_run(state)
            if state["progress"]["next_sequence_index"] != 0:
                raise MeasurementRunStateError(
                    "runtime context must be frozen before the first sample"
                )
            if state.get("active_challenge") is not None:
                raise MeasurementRunStateError(
                    "runtime context must be frozen before issuing a challenge"
                )
            runner = self._runner_state(state)
            existing = runner.get("runtime_binding")
            if existing is not None:
                existing_core = deepcopy(dict(existing))
                existing_core.pop("bound_at_utc", None)
                if _schedule.canonical_json_bytes(
                    existing_core
                ) == _schedule.canonical_json_bytes(binding):
                    return self._public_state(state)
                raise MeasurementRunStateError("a different runtime context is bound")
            binding["bound_at_utc"] = _utc_now()
            runner["runtime_binding"] = binding
            self._write_state(run_dir, state)
            return self._public_state(state)

    def inspect_challenge(
        self,
        capture_run_id: str,
        run_token: str,
        challenge_token: str,
    ) -> dict[str, Any]:
        """Resolve a challenge server-side without accepting client labels."""

        challenge_sha = _token_sha256(self._normalized_secret(challenge_token))
        with self._locked_run(capture_run_id) as run_dir:
            state, manifest, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            last = state.get("last_consumption")
            if isinstance(last, Mapping) and hmac.compare_digest(
                challenge_sha, str(last.get("challenge_token_sha256") or "")
            ):
                return {
                    "ok": True,
                    "status": "committed",
                    "receipt": self._receipt_from_last(state, idempotent=True),
                    "runtime_binding": deepcopy(
                        self._runner_state(state).get("runtime_binding")
                    ),
                }
            self._require_active_run(state)
            active = state.get("active_challenge")
            if not isinstance(active, Mapping) or not hmac.compare_digest(
                challenge_sha, str(active.get("challenge_token_sha256") or "")
            ):
                raise MeasurementRunChallengeError("challenge is invalid or stale")
            ordinal = int(active["ordinal"])
            proofs = self._runner_state(state)["capture_contract_proofs"]
            spool = self._runner_state(state).get("frame_spool")
            return {
                "ok": True,
                "status": "active",
                "capture_run_id": capture_run_id,
                "challenge_id": active["challenge_id"],
                "ordinal": ordinal,
                "block_role": active["ledger_role"],
                "schedule_row": deepcopy(manifest["rows"][ordinal]),
                "runtime_binding": deepcopy(
                    self._runner_state(state).get("runtime_binding")
                ),
                "calibration_write": deepcopy(
                    self._runner_state(state).get("calibration_write")
                ),
                "prepared_attempt": (
                    deepcopy(proofs[ordinal])
                    if len(proofs) == ordinal + 1
                    and proofs[ordinal].get("status") == "prepared"
                    else None
                ),
                "spooled_frame": (
                    {
                        "status": spool.get("status"),
                        "challenge_id": spool.get("challenge_id"),
                        "ordinal": spool.get("ordinal"),
                        "frame_sha256": spool.get("frame_sha256"),
                        "capture_contract_evidence": deepcopy(
                            spool.get("capture_contract_evidence")
                        ),
                        "server_receive_context": deepcopy(
                            spool.get("server_receive_context")
                        ),
                    }
                    if isinstance(spool, Mapping)
                    else None
                ),
                "inference_intent": deepcopy(
                    self._runner_state(state).get("inference_intent")
                ),
                "model_binding": deepcopy(state.get("model_binding")),
                "measurement_claim_authorized": False,
                "physical_capture_claim_authorized": False,
            }

    def record_base_bundle_check(
        self,
        capture_run_id: str,
        run_token: str,
        *,
        stage: str,
        model_id: str,
        bundle_sha256: str,
    ) -> dict[str, Any]:
        allowed_stages = {
            "run_created",
            "first_calibration_inference",
            "calibration_sealed_pre_training",
        }
        if stage not in allowed_stages:
            raise MeasurementRunValidationError("base bundle check stage is invalid")
        normalized_model = self._normalized_identifier(model_id, field="model_id")
        normalized_sha = self._normalized_sha256(
            bundle_sha256, field="bundle_sha256"
        )
        with self._locked_run(capture_run_id) as run_dir:
            state, _, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            runtime = self._runner_state(state).get("runtime_binding")
            if not isinstance(runtime, Mapping):
                raise MeasurementRunStateError("runtime context is not bound")
            base = runtime["base_inference_bundle"]
            if (normalized_model, normalized_sha) != (
                base["model_id"],
                base["bundle_sha256"],
            ):
                raise MeasurementRunIntegrityError(
                    "fresh base bundle check differs from runtime binding"
                )
            checks = self._runner_state(state)["base_bundle_checks"]
            existing = next(
                (item for item in checks if item.get("stage") == stage), None
            )
            core = {
                "stage": stage,
                "model_id": normalized_model,
                "bundle_sha256": normalized_sha,
                "gpu_used": False,
            }
            if existing is not None:
                existing_core = deepcopy(dict(existing))
                existing_core.pop("checked_at_utc", None)
                if existing_core != core:
                    raise MeasurementRunIntegrityError(
                        "base bundle check changed"
                    )
                return deepcopy(existing)
            core["checked_at_utc"] = _utc_now()
            checks.append(core)
            self._write_state(run_dir, state)
            return deepcopy(core)

    def prepare_calibration_write(
        self,
        capture_run_id: str,
        run_token: str,
        challenge_token: str,
        *,
        frame_sha256: str,
    ) -> dict[str, Any]:
        """Durably reserve the exact manifest ordinal before ``save_sample``."""

        frame_sha = self._normalized_sha256(frame_sha256, field="frame_sha256")
        challenge_sha = _token_sha256(self._normalized_secret(challenge_token))
        with self._locked_run(capture_run_id) as run_dir:
            state, _, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            self._require_active_run(state)
            active = state.get("active_challenge")
            if not isinstance(active, Mapping) or not hmac.compare_digest(
                challenge_sha, str(active.get("challenge_token_sha256") or "")
            ):
                raise MeasurementRunChallengeError("challenge is invalid or stale")
            if active.get("ledger_role") != "calibration":
                raise MeasurementRunStateError(
                    "calibration writes are unavailable for evaluation"
                )
            runner = self._runner_state(state)
            if runner.get("runtime_binding") is None:
                raise MeasurementRunStateError("runtime context is not bound")
            ordinal = int(active["ordinal"])
            existing = runner.get("calibration_write")
            if existing is not None:
                if (
                    existing.get("challenge_id") == active["challenge_id"]
                    and existing.get("ordinal") == ordinal
                    and existing.get("frame_sha256") == frame_sha
                    and existing.get("status")
                    in {
                        "prepared",
                        "saved",
                        "no_face_reclassification_pending",
                    }
                ):
                    return deepcopy(existing)
                raise MeasurementRunStateError(
                    "a different calibration frame write is already prepared"
                )
            expected_manifest_index = len(
                runner["calibration_manifest_bindings"]
            )
            candidate = {
                "status": "prepared",
                "challenge_id": active["challenge_id"],
                "ordinal": ordinal,
                "frame_sha256": frame_sha,
                "expected_manifest_index": expected_manifest_index,
            }
            candidate["prepared_at_utc"] = _utc_now()
            runner["calibration_write"] = candidate
            self._write_state(run_dir, state)
            return deepcopy(candidate)

    def persist_encrypted_frame_spool(
        self,
        capture_run_id: str,
        run_token: str,
        challenge_token: str,
        *,
        frame_bytes: bytes,
        capture_contract_evidence: Mapping[str, Any],
        server_receive_context: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Persist one run-token-encrypted frame for crash-safe adapter replay."""

        if not isinstance(frame_bytes, bytes) or not frame_bytes:
            raise MeasurementRunValidationError("frame spool bytes are invalid")
        if len(frame_bytes) > 10 * 1024 * 1024:
            raise MeasurementRunValidationError("frame spool exceeds 10 MiB")
        normalized_run_token = self._normalized_secret(run_token)
        challenge_sha = _token_sha256(self._normalized_secret(challenge_token))
        frame_sha = hashlib.sha256(frame_bytes).hexdigest()
        with self._locked_run(capture_run_id) as run_dir:
            state, _, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            self._require_active_run(state)
            active = state.get("active_challenge")
            if not isinstance(active, Mapping) or not hmac.compare_digest(
                challenge_sha, str(active.get("challenge_token_sha256") or "")
            ):
                raise MeasurementRunChallengeError("challenge is invalid or stale")
            runner = self._runner_state(state)
            runtime = runner.get("runtime_binding")
            if not isinstance(runtime, Mapping):
                raise MeasurementRunStateError("runtime context is not bound")
            capture_evidence = self._validated_capture_contract_evidence(
                runtime, capture_contract_evidence
            )
            receive_context = self._validated_server_receive_context(
                server_receive_context
            )
            existing = runner.get("frame_spool")
            if isinstance(existing, Mapping):
                if (
                    existing.get("challenge_id") != active.get("challenge_id")
                    or existing.get("ordinal") != active.get("ordinal")
                    or existing.get("frame_sha256") != frame_sha
                    or existing.get("capture_contract_evidence")
                    != capture_evidence
                    or existing.get("server_receive_context")
                    != receive_context
                ):
                    raise MeasurementRunStateError(
                        "a different encrypted frame is already spooled"
                    )
                recovered = self._read_frame_spool_unlocked(
                    run_dir, state, normalized_run_token
                )
                if not hmac.compare_digest(
                    hashlib.sha256(recovered).hexdigest(), frame_sha
                ):
                    raise MeasurementRunIntegrityError(
                        "encrypted frame spool plaintext changed"
                    )
                return {
                    "ok": True,
                    "frame_sha256": frame_sha,
                    "ordinal": active["ordinal"],
                    "encrypted": True,
                }
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            key = hashlib.sha256(
                b"lexigaze-wgmc-spool-v1\0"
                + normalized_run_token.encode("utf-8")
            ).digest()
            nonce = secrets.token_bytes(12)
            aad = self._frame_spool_aad(state, active, frame_sha)
            ciphertext = nonce + AESGCM(key).encrypt(nonce, frame_bytes, aad)
            relative = Path("spool") / f"{active['challenge_id']}.aesgcm"
            path = (run_dir / relative).resolve()
            if path.parent != (run_dir / "spool").resolve():
                raise MeasurementRunIntegrityError("frame spool path is unsafe")
            _atomic_bytes(path, ciphertext)
            runner["frame_spool"] = {
                "status": "active",
                "challenge_id": active["challenge_id"],
                "ordinal": active["ordinal"],
                "frame_sha256": frame_sha,
                "ciphertext_sha256": hashlib.sha256(ciphertext).hexdigest(),
                "relative_path": relative.as_posix(),
                "encryption": "AES-256-GCM-keyed-by-run-token-v1",
                "capture_contract_evidence": capture_evidence,
                "server_receive_context": receive_context,
                "persisted_at_utc": _utc_now(),
            }
            self._write_state(run_dir, state)
            return {
                "ok": True,
                "frame_sha256": frame_sha,
                "ordinal": active["ordinal"],
                "encrypted": True,
            }

    def read_encrypted_frame_spool(
        self, capture_run_id: str, run_token: str
    ) -> bytes:
        normalized_run_token = self._normalized_secret(run_token)
        with self._locked_run(capture_run_id) as run_dir:
            state, _, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            return self._read_frame_spool_unlocked(
                run_dir, state, normalized_run_token
            )

    def delete_encrypted_frame_spool(
        self, capture_run_id: str, run_token: str
    ) -> dict[str, Any]:
        with self._locked_run(capture_run_id) as run_dir:
            state, _, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            runner = self._runner_state(state)
            metadata = runner.get("frame_spool")
            if isinstance(metadata, Mapping) and metadata.get("status") == "active":
                metadata["status"] = "cleanup_pending_uncommitted"
                metadata["cleanup_requested_at_utc"] = _utc_now()
                self._write_state(run_dir, state)
            elif isinstance(metadata, Mapping) and metadata.get("status") not in {
                "cleanup_pending_uncommitted",
                "cleanup_pending_committed",
            }:
                raise MeasurementRunIntegrityError(
                    "encrypted frame spool status is invalid"
                )
            expected_path = (
                self._frame_spool_path(run_dir, metadata)
                if isinstance(metadata, Mapping)
                else None
            )
            spool_dir = (run_dir / "spool").resolve()
            if spool_dir.parent != run_dir:
                raise MeasurementRunIntegrityError("frame spool root is unsafe")
            children = list(spool_dir.iterdir()) if spool_dir.is_dir() else []
            unexpected = [
                child
                for child in children
                if not child.is_file()
                or child.parent.resolve() != spool_dir
                or not SPOOL_FILENAME_PATTERN.fullmatch(child.name)
            ]
            if unexpected:
                raise MeasurementRunIntegrityError(
                    "encrypted frame spool contains an unexpected path"
                )
            deleted = False
            for child in children:
                child.unlink()
                deleted = True
            if expected_path is not None and expected_path.exists():
                raise MeasurementRunIntegrityError(
                    "encrypted frame spool could not be removed"
                )
            if spool_dir.is_dir() and any(spool_dir.iterdir()):
                raise MeasurementRunIntegrityError(
                    "encrypted frame spool is not empty"
                )
            runner["frame_spool"] = None
            self._write_state(run_dir, state)
            return {
                "ok": True,
                "deleted": deleted or metadata is not None,
                "orphan_files_deleted": len(children) if metadata is None else 0,
                "absence_verified": True,
            }

    @staticmethod
    def _frame_spool_aad(
        state: Mapping[str, Any], active: Mapping[str, Any], frame_sha256: str
    ) -> bytes:
        return _schedule.canonical_json_bytes(
            {
                "capture_run_id": state["capture_run_id"],
                "challenge_id": active["challenge_id"],
                "ordinal": active["ordinal"],
                "frame_sha256": frame_sha256,
            }
        )

    def _frame_spool_path(
        self, run_dir: Path, metadata: Mapping[str, Any]
    ) -> Path:
        relative = Path(str(metadata.get("relative_path") or ""))
        path = (run_dir / relative).resolve()
        if path.parent != (run_dir / "spool").resolve():
            raise MeasurementRunIntegrityError("frame spool path is unsafe")
        return path

    def _read_frame_spool_unlocked(
        self,
        run_dir: Path,
        state: Mapping[str, Any],
        normalized_run_token: str,
    ) -> bytes:
        runner = self._runner_state(state)
        metadata = runner.get("frame_spool")
        active = state.get("active_challenge")
        if not isinstance(metadata, Mapping) or not isinstance(active, Mapping):
            raise MeasurementRunStateError("encrypted frame spool is unavailable")
        if (
            metadata.get("challenge_id") != active.get("challenge_id")
            or metadata.get("ordinal") != active.get("ordinal")
        ):
            raise MeasurementRunIntegrityError("frame spool challenge binding changed")
        path = self._frame_spool_path(run_dir, metadata)
        try:
            ciphertext = path.read_bytes()
        except OSError as exc:
            raise MeasurementRunIntegrityError(
                "encrypted frame spool is unreadable"
            ) from exc
        if hashlib.sha256(ciphertext).hexdigest() != metadata.get(
            "ciphertext_sha256"
        ):
            raise MeasurementRunIntegrityError(
                "encrypted frame spool ciphertext changed"
            )
        if len(ciphertext) < 29:
            raise MeasurementRunIntegrityError("encrypted frame spool is truncated")
        nonce, encrypted = ciphertext[:12], ciphertext[12:]
        from cryptography.exceptions import InvalidTag
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        key = hashlib.sha256(
            b"lexigaze-wgmc-spool-v1\0"
            + normalized_run_token.encode("utf-8")
        ).digest()
        aad = self._frame_spool_aad(
            state, active, str(metadata.get("frame_sha256") or "")
        )
        try:
            plaintext = AESGCM(key).decrypt(nonce, encrypted, aad)
        except InvalidTag as exc:
            raise MeasurementRunIntegrityError(
                "encrypted frame spool authentication failed"
            ) from exc
        if hashlib.sha256(plaintext).hexdigest() != metadata.get("frame_sha256"):
            raise MeasurementRunIntegrityError(
                "encrypted frame spool plaintext hash changed"
            )
        return plaintext

    def reset_unwritten_calibration_write(
        self,
        capture_run_id: str,
        run_token: str,
        challenge_token: str,
        *,
        frame_sha256: str,
    ) -> dict[str, Any]:
        """Release only a prepared write whose manifest row is proven absent."""

        frame_sha = self._normalized_sha256(frame_sha256, field="frame_sha256")
        challenge_sha = _token_sha256(self._normalized_secret(challenge_token))
        with self._locked_run(capture_run_id) as run_dir:
            state, _, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            active = state.get("active_challenge")
            runner = self._runner_state(state)
            pending = runner.get("calibration_write")
            if (
                not isinstance(active, Mapping)
                or not isinstance(pending, Mapping)
                or not hmac.compare_digest(
                    challenge_sha, str(active.get("challenge_token_sha256") or "")
                )
                or pending.get("challenge_id") != active.get("challenge_id")
                or pending.get("frame_sha256") != frame_sha
            ):
                raise MeasurementRunStateError("prepared calibration write changed")
            if pending.get("status") != "prepared":
                raise MeasurementRunStateError("saved calibration write cannot reset")
            runner["calibration_write"] = None
            self._write_state(run_dir, state)
            return {"ok": True, "reset": True, "ordinal": pending["ordinal"]}

    def record_calibration_sample_saved(
        self,
        capture_run_id: str,
        run_token: str,
        challenge_token: str,
        *,
        frame_sha256: str,
        sample_index: int,
        manifest_record_sha256: str,
        raw_frame_sha256: str,
        normalized_face_sha256: str,
        crop_sha256: str,
        face_detected: bool,
    ) -> dict[str, Any]:
        """Bind one dedicated manifest row to its server challenge and frame."""

        frame_sha = self._normalized_sha256(frame_sha256, field="frame_sha256")
        manifest_sha = self._normalized_sha256(
            manifest_record_sha256, field="manifest_record_sha256"
        )
        raw_sha = self._normalized_sha256(
            raw_frame_sha256, field="raw_frame_sha256"
        )
        normalized_sha = self._normalized_sha256(
            normalized_face_sha256, field="normalized_face_sha256"
        )
        crop_sha = self._normalized_sha256(crop_sha256, field="crop_sha256")
        if raw_sha != frame_sha:
            raise MeasurementRunIntegrityError(
                "saved raw frame differs from the prepared frame"
            )
        if not isinstance(sample_index, int) or isinstance(sample_index, bool):
            raise MeasurementRunValidationError("sample_index must be an integer")
        if not isinstance(face_detected, bool):
            raise MeasurementRunValidationError("face_detected must be boolean")
        challenge_sha = _token_sha256(self._normalized_secret(challenge_token))
        with self._locked_run(capture_run_id) as run_dir:
            state, _, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            active = state.get("active_challenge")
            runner = self._runner_state(state)
            pending = runner.get("calibration_write")
            if (
                not isinstance(active, Mapping)
                or not isinstance(pending, Mapping)
                or not hmac.compare_digest(
                    challenge_sha, str(active.get("challenge_token_sha256") or "")
                )
                or pending.get("challenge_id") != active.get("challenge_id")
                or pending.get("frame_sha256") != frame_sha
            ):
                raise MeasurementRunStateError("calibration write binding changed")
            ordinal = int(active["ordinal"])
            expected_manifest_index = len(
                runner["calibration_manifest_bindings"]
            )
            if (
                sample_index != expected_manifest_index
                or pending.get("expected_manifest_index") != expected_manifest_index
            ):
                raise MeasurementRunIntegrityError(
                    "dedicated calibration manifest index is not exact"
                )
            binding = {
                "status": "saved",
                "challenge_id": active["challenge_id"],
                "ordinal": ordinal,
                "sample_index": sample_index,
                "frame_sha256": frame_sha,
                "raw_frame_sha256": raw_sha,
                "normalized_face_sha256": normalized_sha,
                "crop_sha256": crop_sha,
                "manifest_record_sha256": manifest_sha,
                "face_detected": face_detected,
            }
            bindings = runner["calibration_manifest_bindings"]
            existing_for_ordinal = next(
                (item for item in bindings if item.get("ordinal") == ordinal),
                None,
            )
            if existing_for_ordinal is None:
                binding["saved_at_utc"] = _utc_now()
                bindings.append(binding)
            else:
                existing = deepcopy(dict(existing_for_ordinal))
                existing.pop("saved_at_utc", None)
                existing.pop("committed_at_utc", None)
                existing.pop("sample_sha256", None)
                existing.pop("ledger_record_sha256", None)
                if existing != binding:
                    raise MeasurementRunIntegrityError(
                        "calibration manifest binding changed"
                    )
            pending.update(binding)
            self._write_state(run_dir, state)
            return deepcopy(
                next(item for item in bindings if item.get("ordinal") == ordinal)
            )

    def begin_inference_attempt(
        self,
        capture_run_id: str,
        run_token: str,
        challenge_token: str,
        *,
        frame_sha256: str,
        model_id: str,
        model_sha256: str,
        model_selector: str,
        capture_contract_evidence: Mapping[str, Any],
        server_receive_context: Mapping[str, Any],
        predict_started_monotonic_ms: float,
    ) -> dict[str, Any]:
        """Durably mark a non-repeatable predictor call before it starts."""

        frame_sha = self._normalized_sha256(frame_sha256, field="frame_sha256")
        normalized_model_id = self._normalized_identifier(model_id, field="model_id")
        normalized_model_sha = self._normalized_sha256(
            model_sha256, field="model_sha256"
        )
        normalized_selector = self._normalized_identifier(
            model_selector, field="model_selector"
        )
        challenge_sha = _token_sha256(self._normalized_secret(challenge_token))
        with self._locked_run(capture_run_id) as run_dir:
            state, manifest, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            self._require_active_run(state)
            active = state.get("active_challenge")
            if not isinstance(active, Mapping) or not hmac.compare_digest(
                challenge_sha, str(active.get("challenge_token_sha256") or "")
            ):
                raise MeasurementRunChallengeError("challenge is invalid or stale")
            runner = self._runner_state(state)
            runtime = runner.get("runtime_binding")
            spool = runner.get("frame_spool")
            if not isinstance(runtime, Mapping) or not isinstance(spool, Mapping):
                raise MeasurementRunStateError(
                    "inference intent requires runtime and encrypted frame spool"
                )
            ordinal = int(active["ordinal"])
            if (
                spool.get("status") != "active"
                or spool.get("challenge_id") != active.get("challenge_id")
                or spool.get("ordinal") != ordinal
                or spool.get("frame_sha256") != frame_sha
            ):
                raise MeasurementRunIntegrityError(
                    "inference intent frame-spool binding changed"
                )
            if len(runner["capture_contract_proofs"]) != ordinal:
                raise MeasurementRunStateError(
                    "inference cannot restart after an observation is prepared"
                )
            row = manifest["rows"][ordinal]
            if row["block_role"] == "calibration":
                base = runtime["base_inference_bundle"]
                expected_model = (
                    base["model_id"],
                    base["bundle_sha256"],
                    base["inference_selector"],
                )
            else:
                binding = state.get("model_binding")
                if not isinstance(binding, Mapping):
                    raise MeasurementRunStateError(
                        "evaluation inference model is not bound"
                    )
                expected_model = (
                    binding["model_id"],
                    binding["model_sha256"],
                    binding["model_id"],
                )
            if (
                normalized_model_id,
                normalized_model_sha,
                normalized_selector,
            ) != expected_model:
                raise MeasurementRunIntegrityError(
                    "inference intent model binding changed"
                )
            evidence = self._validated_capture_contract_evidence(
                runtime, capture_contract_evidence
            )
            receive = self._validated_server_receive_context(server_receive_context)
            started = self._positive_finite(
                predict_started_monotonic_ms,
                field="predict_started_monotonic_ms",
            )
            if started < float(receive["decode_completed_monotonic_ms"]):
                raise MeasurementRunValidationError(
                    "predict start precedes server decode completion"
                )
            if runner.get("inference_intent") is not None:
                raise MeasurementRunStateError(
                    "an unsealed inference intent already exists"
                )
            core = {
                "status": "inference_in_progress",
                "challenge_id": active["challenge_id"],
                "ordinal": ordinal,
                "frame_sha256": frame_sha,
                "model_id": normalized_model_id,
                "model_sha256": normalized_model_sha,
                "model_selector": normalized_selector,
                "capture_contract_evidence_sha256": _schedule.canonical_sha256(
                    evidence
                ),
                "server_receive_context_sha256": _schedule.canonical_sha256(
                    receive
                ),
                "predict_started_monotonic_ms": started,
            }
            core["intent_sha256"] = _schedule.canonical_sha256(core)
            core["started_at_utc"] = _utc_now()
            runner["inference_intent"] = core
            self._write_state(run_dir, state)
            return deepcopy(core)

    def clear_inference_intent_after_hard_error(
        self,
        capture_run_id: str,
        run_token: str,
        challenge_token: str,
        *,
        frame_sha256: str,
    ) -> dict[str, Any]:
        """Acknowledge a returned hard response before allowing a new frame."""

        frame_sha = self._normalized_sha256(frame_sha256, field="frame_sha256")
        challenge_sha = _token_sha256(self._normalized_secret(challenge_token))
        with self._locked_run(capture_run_id) as run_dir:
            state, _, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            active = state.get("active_challenge")
            runner = self._runner_state(state)
            intent = runner.get("inference_intent")
            if (
                not isinstance(active, Mapping)
                or not isinstance(intent, Mapping)
                or not hmac.compare_digest(
                    challenge_sha, str(active.get("challenge_token_sha256") or "")
                )
                or intent.get("challenge_id") != active.get("challenge_id")
                or intent.get("frame_sha256") != frame_sha
            ):
                raise MeasurementRunStateError("inference intent binding changed")
            runner["inference_intent"] = None
            self._write_state(run_dir, state)
            return {"ok": True, "cleared": True, "ordinal": active["ordinal"]}

    def record_attempt_observation(
        self,
        capture_run_id: str,
        run_token: str,
        challenge_token: str,
        *,
        frame_sha256: str,
        observation: Mapping[str, Any],
        capture_contract_evidence: Mapping[str, Any],
        server_timing_evidence: Mapping[str, Any],
        disposition: str,
    ) -> dict[str, Any]:
        """Persist the exact label-free observation before ledger commit."""

        frame_sha = self._normalized_sha256(frame_sha256, field="frame_sha256")
        if disposition not in {"success", "no_face_detected"}:
            raise MeasurementRunValidationError("attempt disposition is invalid")
        if not isinstance(capture_contract_evidence, Mapping):
            raise MeasurementRunValidationError(
                "capture_contract_evidence must be an object"
            )
        challenge_sha = _token_sha256(self._normalized_secret(challenge_token))
        with self._locked_run(capture_run_id) as run_dir:
            state, manifest, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            self._require_active_run(state)
            active = state.get("active_challenge")
            if not isinstance(active, Mapping) or not hmac.compare_digest(
                challenge_sha, str(active.get("challenge_token_sha256") or "")
            ):
                raise MeasurementRunChallengeError("challenge is invalid or stale")
            runner = self._runner_state(state)
            runtime = runner.get("runtime_binding")
            if not isinstance(runtime, Mapping):
                raise MeasurementRunStateError("runtime context is not bound")
            ordinal = int(active["ordinal"])
            row = dict(manifest["rows"][ordinal])
            sample = self._build_sample(
                observation, row, manifest=manifest, ordinal=ordinal
            )
            validation = self._validate_sample(sample, row, manifest, ordinal)
            self._validate_model_binding(state, validation, role=str(row["block_role"]))
            success = sample.get("prediction_success") is True
            if success != (disposition == "success"):
                raise MeasurementRunValidationError(
                    "prediction_success differs from attempt disposition"
                )
            if disposition == "no_face_detected" and any(
                sample.get(field) is not None
                for field in (
                    "raw_gaze_pitch_yaw",
                    "predicted_x_px",
                    "predicted_y_px",
                    "head_pose_pitch_yaw",
                    "normalized_face_bbox",
                )
            ):
                raise MeasurementRunValidationError(
                    "no-face observation must contain null sensor outputs"
                )
            evidence = self._validated_capture_contract_evidence(
                runtime, capture_contract_evidence
            )
            timing = self._validated_server_timing_evidence(
                server_timing_evidence
            )
            if (
                float(sample["frame_capture_monotonic_ms"])
                != timing["predict_started_monotonic_ms"]
                or float(sample["inference_completed_monotonic_ms"])
                != timing["predict_completed_monotonic_ms"]
            ):
                raise MeasurementRunValidationError(
                    "sample inference timing differs from server timing evidence"
                )
            intent = runner.get("inference_intent")
            receive_core = {
                "server_request_received_monotonic_ms": timing[
                    "server_request_received_monotonic_ms"
                ],
                "decode_completed_monotonic_ms": timing[
                    "decode_completed_monotonic_ms"
                ],
                "frame_source": timing["frame_source"],
            }
            expected_selector = (
                runtime["base_inference_bundle"]["inference_selector"]
                if row["block_role"] == "calibration"
                else state["model_binding"]["model_id"]
            )
            if not isinstance(intent, Mapping) or (
                intent.get("status") != "inference_in_progress"
                or intent.get("challenge_id") != active.get("challenge_id")
                or intent.get("ordinal") != ordinal
                or intent.get("frame_sha256") != frame_sha
                or intent.get("model_id") != sample.get("model_id")
                or intent.get("model_sha256") != sample.get("model_sha256")
                or intent.get("model_selector") != expected_selector
                or intent.get("capture_contract_evidence_sha256")
                != _schedule.canonical_sha256(evidence)
                or intent.get("server_receive_context_sha256")
                != _schedule.canonical_sha256(receive_core)
                or float(intent.get("predict_started_monotonic_ms", -1.0))
                != timing["predict_started_monotonic_ms"]
            ):
                raise MeasurementRunIntegrityError(
                    "attempt observation lacks its non-repeatable inference intent"
                )
            proof = {
                "status": "prepared",
                "challenge_id": active["challenge_id"],
                "ordinal": ordinal,
                "ledger_role": row["block_role"],
                "frame_sha256": frame_sha,
                "disposition": disposition,
                "capture_contract_evidence": evidence,
                "server_timing_evidence": timing,
                "observation": sample,
                "observation_sha256": _schedule.canonical_sha256(sample),
            }
            proofs = runner["capture_contract_proofs"]
            if len(proofs) == ordinal:
                proof["prepared_at_utc"] = _utc_now()
                proofs.append(proof)
            elif len(proofs) == ordinal + 1:
                existing = deepcopy(dict(proofs[ordinal]))
                existing.pop("prepared_at_utc", None)
                existing.pop("committed_at_utc", None)
                existing.pop("sample_sha256", None)
                existing.pop("ledger_record_sha256", None)
                if existing != proof:
                    raise MeasurementRunIntegrityError(
                        "prepared attempt observation changed"
                    )
            else:
                raise MeasurementRunIntegrityError(
                    "capture-contract proof order is not exact"
                )
            runner["inference_intent"] = None
            self._write_state(run_dir, state)
            return deepcopy(proofs[ordinal])

    def mark_calibration_no_face_reclassification_pending(
        self,
        capture_run_id: str,
        run_token: str,
        challenge_token: str,
        *,
        frame_sha256: str,
        sample_index: int,
        manifest_record_sha256: str,
        cleanup_relative_paths: Mapping[str, str | None],
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Make a sample-store no-face result irreversible before truncation."""

        frame_sha = self._normalized_sha256(frame_sha256, field="frame_sha256")
        manifest_sha = self._normalized_sha256(
            manifest_record_sha256, field="manifest_record_sha256"
        )
        if not isinstance(sample_index, int) or isinstance(sample_index, bool):
            raise MeasurementRunValidationError("sample_index must be an integer")
        if set(cleanup_relative_paths) != {
            "raw",
            "crop",
            "normalized_face",
        }:
            raise MeasurementRunValidationError(
                "no-face cleanup path fields changed"
            )
        normalized_paths: dict[str, str | None] = {}
        for key, value in cleanup_relative_paths.items():
            if value is not None and (
                not isinstance(value, str)
                or not value
                or Path(value).is_absolute()
                or ".." in Path(value).parts
            ):
                raise MeasurementRunValidationError(
                    "no-face cleanup relative path is invalid"
                )
            normalized_paths[key] = value
        challenge_sha = _token_sha256(self._normalized_secret(challenge_token))
        with self._locked_run(capture_run_id) as run_dir:
            state, manifest, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            active = state.get("active_challenge")
            runner = self._runner_state(state)
            pending = runner.get("calibration_write")
            if (
                not isinstance(active, Mapping)
                or not isinstance(pending, dict)
                or not hmac.compare_digest(
                    challenge_sha, str(active.get("challenge_token_sha256") or "")
                )
                or pending.get("challenge_id") != active.get("challenge_id")
                or pending.get("frame_sha256") != frame_sha
            ):
                raise MeasurementRunStateError("calibration write binding changed")
            ordinal = int(active["ordinal"])
            if sample_index != pending.get("expected_manifest_index"):
                raise MeasurementRunIntegrityError(
                    "no-face manifest index differs from prepared write"
                )
            proofs = runner["capture_contract_proofs"]
            if len(proofs) != ordinal + 1:
                raise MeasurementRunStateError("prepared observation is unavailable")
            proof = proofs[ordinal]
            if (
                proof.get("status") != "prepared"
                or proof.get("disposition") != "success"
                or proof.get("frame_sha256") != frame_sha
            ):
                raise MeasurementRunIntegrityError(
                    "no-face reclassification source proof changed"
                )
            row = dict(manifest["rows"][ordinal])
            sample = self._build_sample(
                observation, row, manifest=manifest, ordinal=ordinal
            )
            validation = self._validate_sample(sample, row, manifest, ordinal)
            self._validate_model_binding(state, validation, role="calibration")
            if sample.get("prediction_success") is not False or any(
                sample.get(field) is not None
                for field in (
                    "raw_gaze_pitch_yaw",
                    "predicted_x_px",
                    "predicted_y_px",
                    "head_pose_pitch_yaw",
                    "normalized_face_bbox",
                )
            ):
                raise MeasurementRunValidationError(
                    "no-face reclassification must contain null sensor outputs"
                )
            core = {
                "status": "no_face_reclassification_pending",
                "challenge_id": active["challenge_id"],
                "ordinal": ordinal,
                "frame_sha256": frame_sha,
                "expected_manifest_index": sample_index,
                "manifest_record_sha256": manifest_sha,
                "cleanup_relative_paths": normalized_paths,
                "no_face_observation": sample,
                "no_face_observation_sha256": _schedule.canonical_sha256(sample),
            }
            if pending.get("status") == "no_face_reclassification_pending":
                existing = deepcopy(dict(pending))
                existing.pop("pending_at_utc", None)
                if existing != core:
                    raise MeasurementRunIntegrityError(
                        "no-face reclassification intent changed"
                    )
                return deepcopy(pending)
            if pending.get("status") != "prepared":
                raise MeasurementRunStateError(
                    "calibration write cannot enter no-face reclassification"
                )
            core["pending_at_utc"] = _utc_now()
            runner["calibration_write"] = core
            self._write_state(run_dir, state)
            return deepcopy(core)

    def complete_calibration_no_face_reclassification(
        self,
        capture_run_id: str,
        run_token: str,
        challenge_token: str,
        *,
        frame_sha256: str,
    ) -> dict[str, Any]:
        """Atomically apply an already-durable no-face disposition."""

        frame_sha = self._normalized_sha256(frame_sha256, field="frame_sha256")
        challenge_sha = _token_sha256(self._normalized_secret(challenge_token))
        with self._locked_run(capture_run_id) as run_dir:
            state, _, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            active = state.get("active_challenge")
            runner = self._runner_state(state)
            pending = runner.get("calibration_write")
            if (
                not isinstance(active, Mapping)
                or not isinstance(pending, Mapping)
                or not hmac.compare_digest(
                    challenge_sha, str(active.get("challenge_token_sha256") or "")
                )
                or pending.get("status") != "no_face_reclassification_pending"
                or pending.get("challenge_id") != active.get("challenge_id")
                or pending.get("frame_sha256") != frame_sha
            ):
                raise MeasurementRunStateError(
                    "no-face reclassification intent is unavailable"
                )
            ordinal = int(active["ordinal"])
            proofs = runner["capture_contract_proofs"]
            if len(proofs) != ordinal + 1:
                raise MeasurementRunStateError("prepared observation is unavailable")
            proof = proofs[ordinal]
            sample = pending.get("no_face_observation")
            if (
                proof.get("status") != "prepared"
                or proof.get("disposition") != "success"
                or proof.get("frame_sha256") != frame_sha
                or not isinstance(sample, Mapping)
                or _schedule.canonical_sha256(sample)
                != pending.get("no_face_observation_sha256")
            ):
                raise MeasurementRunIntegrityError(
                    "no-face reclassification proof binding changed"
                )
            proof["disposition"] = "no_face_detected"
            proof["observation"] = deepcopy(dict(sample))
            proof["observation_sha256"] = pending[
                "no_face_observation_sha256"
            ]
            proof["reclassified_at_utc"] = _utc_now()
            runner["calibration_write"] = None
            self._write_state(run_dir, state)
            return deepcopy(proof)

    def replace_prepared_observation_with_no_face(
        self,
        capture_run_id: str,
        run_token: str,
        challenge_token: str,
        *,
        frame_sha256: str,
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Resolve a second-pass preprocessor disagreement without replacement."""

        frame_sha = self._normalized_sha256(frame_sha256, field="frame_sha256")
        challenge_sha = _token_sha256(self._normalized_secret(challenge_token))
        with self._locked_run(capture_run_id) as run_dir:
            state, manifest, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            active = state.get("active_challenge")
            if not isinstance(active, Mapping) or not hmac.compare_digest(
                challenge_sha, str(active.get("challenge_token_sha256") or "")
            ):
                raise MeasurementRunChallengeError("challenge is invalid or stale")
            ordinal = int(active["ordinal"])
            runner = self._runner_state(state)
            proofs = runner["capture_contract_proofs"]
            if len(proofs) != ordinal + 1:
                raise MeasurementRunStateError("prepared observation is unavailable")
            proof = proofs[ordinal]
            if (
                proof.get("status") != "prepared"
                or proof.get("challenge_id") != active.get("challenge_id")
                or proof.get("frame_sha256") != frame_sha
                or proof.get("disposition") != "success"
            ):
                raise MeasurementRunIntegrityError(
                    "prepared success observation binding changed"
                )
            row = dict(manifest["rows"][ordinal])
            sample = self._build_sample(
                observation, row, manifest=manifest, ordinal=ordinal
            )
            validation = self._validate_sample(sample, row, manifest, ordinal)
            self._validate_model_binding(state, validation, role=str(row["block_role"]))
            if sample.get("prediction_success") is not False or any(
                sample.get(field) is not None
                for field in (
                    "raw_gaze_pitch_yaw",
                    "predicted_x_px",
                    "predicted_y_px",
                    "head_pose_pitch_yaw",
                    "normalized_face_bbox",
                )
            ):
                raise MeasurementRunValidationError(
                    "replacement must be a null-sensor no-face observation"
                )
            if any(
                item.get("ordinal") == ordinal
                for item in runner["calibration_manifest_bindings"]
            ):
                raise MeasurementRunIntegrityError(
                    "unusable row was not removed from the training manifest"
                )
            proof["disposition"] = "no_face_detected"
            proof["observation"] = sample
            proof["observation_sha256"] = _schedule.canonical_sha256(sample)
            proof["reclassified_at_utc"] = _utc_now()
            runner["calibration_write"] = None
            self._write_state(run_dir, state)
            return deepcopy(proof)

    def issue_next_challenge(
        self, capture_run_id: str, run_token: str
    ) -> dict[str, Any]:
        """Issue the next ordinal once; only the client receives its secret."""

        with self._locked_run(capture_run_id) as run_dir:
            state, manifest, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            self._require_active_run(state)
            if state["active_challenge"] is not None:
                raise MeasurementRunChallengeError(
                    "an unconsumed challenge already exists; rotate it explicitly"
                )
            if self._runner_state(state).get("frame_spool") is not None:
                raise MeasurementRunStateError(
                    "encrypted frame spool cleanup must finish before the next challenge"
                )
            ordinal = int(state["progress"]["next_sequence_index"])
            if ordinal >= _schedule.EXPECTED_SAMPLE_COUNT:
                raise MeasurementRunStateError("all 193 samples are already collected")

            row = dict(manifest["rows"][ordinal])
            role = str(row["block_role"])
            runner = self._runner_state(state)
            if role == "calibration":
                if state["phase"] not in {
                    PHASE_SCHEDULED,
                    PHASE_CALIBRATION_COLLECTING,
                }:
                    raise MeasurementRunStateError(
                        "calibration challenges are unavailable in this phase"
                    )
                state["phase"] = PHASE_CALIBRATION_COLLECTING
            else:
                if state["phase"] not in {
                    PHASE_MODEL_BOUND,
                    PHASE_EVALUATION_COLLECTING,
                }:
                    raise MeasurementRunStateError(
                        "bind the sealed calibration model before evaluation"
                    )
                if runner.get("runtime_binding") is not None:
                    purge = runner.get("calibration_image_purge")
                    if not isinstance(purge, Mapping) or purge.get("status") != "verified":
                        raise MeasurementRunStateError(
                            "evaluation is unavailable until dedicated calibration images are purged"
                        )
                state["phase"] = PHASE_EVALUATION_COLLECTING

            challenge = self._new_challenge(ordinal=ordinal, row=row, role=role)
            state["active_challenge"] = challenge["persisted"]
            self._write_state(run_dir, state)
            return {
                "ok": True,
                "capture_run_id": capture_run_id,
                "phase": state["phase"],
                "challenge_id": challenge["persisted"]["challenge_id"],
                "challenge_token": challenge["plaintext_token"],
                "challenge_token_returned_once": True,
                "ordinal": ordinal,
                "block_role": role,
                "schedule_row": row,
                "schedule_row_sha256": challenge["persisted"][
                    "schedule_row_sha256"
                ],
                "manifest_sha256": state["manifest_sha256"],
                "protocol_sha256": state["protocol_sha256"],
                "measurement_claim_authorized": False,
                "physical_capture_claim_authorized": False,
            }

    def rotate_unconsumed_challenge(
        self, capture_run_id: str, run_token: str
    ) -> dict[str, Any]:
        """Replace a lost, still-unconsumed challenge without storing plaintext."""

        with self._locked_run(capture_run_id) as run_dir:
            state, manifest, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            self._require_active_run(state)
            active = state.get("active_challenge")
            if not isinstance(active, Mapping):
                raise MeasurementRunChallengeError(
                    "there is no unconsumed challenge to rotate"
                )
            ordinal = int(active["ordinal"])
            runner = self._runner_state(state)
            if (
                runner.get("inference_intent") is not None
                or runner.get("calibration_write") is not None
                or len(
                runner["capture_contract_proofs"]
                ) > ordinal
            ):
                raise MeasurementRunChallengeError(
                    "challenge has durable server work and cannot be rotated"
                )
            row = dict(manifest["rows"][ordinal])
            replacement = self._new_challenge(
                ordinal=ordinal,
                row=row,
                role=str(active["ledger_role"]),
                rotation_count=int(active["rotation_count"]) + 1,
            )
            state["active_challenge"] = replacement["persisted"]
            self._write_state(run_dir, state)
            return {
                "ok": True,
                "capture_run_id": capture_run_id,
                "phase": state["phase"],
                "challenge_id": replacement["persisted"]["challenge_id"],
                "challenge_token": replacement["plaintext_token"],
                "challenge_token_returned_once": True,
                "ordinal": ordinal,
                "block_role": active["ledger_role"],
                "schedule_row": row,
                "schedule_row_sha256": replacement["persisted"][
                    "schedule_row_sha256"
                ],
                "rotation_count": replacement["persisted"]["rotation_count"],
                "measurement_claim_authorized": False,
                "physical_capture_claim_authorized": False,
            }

    def consume_challenge(
        self,
        capture_run_id: str,
        run_token: str,
        challenge_token: str,
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Validate and atomically append one derived observation."""

        if not isinstance(observation, Mapping):
            raise MeasurementRunValidationError("observation must be an object")
        _assert_no_raw_media(observation)
        challenge_sha256 = _token_sha256(self._normalized_secret(challenge_token))

        with self._locked_run(capture_run_id) as run_dir:
            state, manifest, calibration, evaluation = (
                self._load_authenticated_unlocked(
                    run_dir, capture_run_id, run_token
                )
            )

            last = state.get("last_consumption")
            if isinstance(last, Mapping) and hmac.compare_digest(
                challenge_sha256, str(last.get("challenge_token_sha256") or "")
            ):
                ordinal = int(last["ordinal"])
                sample = self._build_sample(
                    observation,
                    dict(manifest["rows"][ordinal]),
                    manifest=manifest,
                    ordinal=ordinal,
                )
                sample_sha256 = _schedule.canonical_sha256(sample)
                if not hmac.compare_digest(
                    sample_sha256, str(last.get("sample_sha256") or "")
                ):
                    self._mark_failed_unlocked(
                        run_dir,
                        state,
                        code="conflicting_consumed_challenge_replay",
                        detail=(
                            "a consumed challenge was replayed with a different "
                            "derived observation"
                        ),
                    )
                    raise MeasurementRunIntegrityError(
                        "consumed challenge replay differs from persisted sample"
                    )
                return self._receipt_from_last(state, idempotent=True)

            self._require_active_run(state)

            active = state.get("active_challenge")
            if not isinstance(active, Mapping):
                raise MeasurementRunChallengeError("no challenge is awaiting a sample")
            if not hmac.compare_digest(
                challenge_sha256, str(active.get("challenge_token_sha256") or "")
            ):
                raise MeasurementRunChallengeError("challenge is invalid or stale")
            return self._commit_active_observation_unlocked(
                run_dir,
                state,
                manifest,
                calibration,
                evaluation,
                observation,
            )

    def commit_prepared_observation(
        self, capture_run_id: str, run_token: str
    ) -> dict[str, Any]:
        """Commit a durable adapter observation after a crash lost its token."""

        with self._locked_run(capture_run_id) as run_dir:
            state, manifest, calibration, evaluation = (
                self._load_authenticated_unlocked(
                    run_dir, capture_run_id, run_token
                )
            )
            self._require_active_run(state)
            active = state.get("active_challenge")
            if not isinstance(active, Mapping):
                raise MeasurementRunChallengeError("no challenge is awaiting a sample")
            ordinal = int(active["ordinal"])
            proofs = self._runner_state(state)["capture_contract_proofs"]
            if len(proofs) != ordinal + 1:
                raise MeasurementRunStateError("prepared observation is unavailable")
            proof = proofs[ordinal]
            if (
                proof.get("status") != "prepared"
                or proof.get("challenge_id") != active.get("challenge_id")
                or not isinstance(proof.get("observation"), Mapping)
            ):
                raise MeasurementRunIntegrityError(
                    "prepared observation binding changed"
                )
            return self._commit_active_observation_unlocked(
                run_dir,
                state,
                manifest,
                calibration,
                evaluation,
                proof["observation"],
            )

    def _commit_active_observation_unlocked(
        self,
        run_dir: Path,
        state: dict[str, Any],
        manifest: Mapping[str, Any],
        calibration: Mapping[str, Any],
        evaluation: Mapping[str, Any],
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        active = state.get("active_challenge")
        if not isinstance(active, Mapping):
            raise MeasurementRunChallengeError("no challenge is awaiting a sample")
        ordinal = int(active["ordinal"])
        row = dict(manifest["rows"][ordinal])
        sample = self._build_sample(
            observation, row, manifest=manifest, ordinal=ordinal
        )
        role = str(row["block_role"])
        validation = self._validate_sample(sample, row, manifest, ordinal)
        self._validate_model_binding(state, validation, role=role)

        runner = self._runner_state(state)
        runtime_bound = runner.get("runtime_binding") is not None
        if runtime_bound:
            proofs = runner["capture_contract_proofs"]
            if len(proofs) != ordinal + 1:
                raise MeasurementRunStateError(
                    "adapter observation was not durably prepared"
                )
            proof = proofs[ordinal]
            if (
                proof.get("status") != "prepared"
                or proof.get("challenge_id") != active.get("challenge_id")
                or proof.get("observation_sha256")
                != _schedule.canonical_sha256(sample)
            ):
                raise MeasurementRunIntegrityError(
                    "adapter observation differs from durable preparation"
                )
            if role == "calibration" and sample.get("prediction_success") is True:
                saved = next(
                    (
                        item
                        for item in runner["calibration_manifest_bindings"]
                        if item.get("ordinal") == ordinal
                    ),
                    None,
                )
                if (
                    not isinstance(saved, Mapping)
                    or saved.get("status") != "saved"
                    or saved.get("frame_sha256") != proof.get("frame_sha256")
                    or saved.get("face_detected") is not True
                ):
                    raise MeasurementRunStateError(
                        "successful calibration attempt lacks its exact usable row"
                    )
            if role == "calibration" and sample.get("prediction_success") is False:
                if any(
                    item.get("ordinal") == ordinal
                    for item in runner["calibration_manifest_bindings"]
                ):
                    raise MeasurementRunIntegrityError(
                        "no-face calibration attempt entered the training manifest"
                    )

        records = calibration["records"] if role == "calibration" else evaluation[
            "records"
        ]
        previous_head = records[-1]["record_sha256"] if records else None
        record = {
            "schema_version": STATE_SCHEMA_VERSION,
            "record_type": LEDGER_RECORD_TYPE,
            "capture_run_id": state["capture_run_id"],
            "ledger_role": role,
            "ledger_ordinal": len(records),
            "sequence_index": ordinal,
            "schedule_row_sha256": active["schedule_row_sha256"],
            "challenge_id": active["challenge_id"],
            "challenge_token_sha256": active["challenge_token_sha256"],
            "sample": sample,
            "sample_sha256": _schedule.canonical_sha256(sample),
            "previous_record_sha256": previous_head,
            "recorded_at_utc": _utc_now(),
        }
        if runtime_bound:
            proof = runner["capture_contract_proofs"][ordinal]
            record["attempt_disposition"] = proof["disposition"]
            record["capture_contract_evidence_sha256"] = _schedule.canonical_sha256(
                proof["capture_contract_evidence"]
            )
            record["server_timing_evidence_sha256"] = _schedule.canonical_sha256(
                proof["server_timing_evidence"]
            )
            record["frame_sha256"] = proof["frame_sha256"]
        record["record_sha256"] = _schedule.canonical_sha256(record)

        # Ledger first, state second.  Restart reconciliation is bound to the
        # challenge, exact prepared observation, and (for calibration) manifest.
        self._append_ledger_unlocked(run_dir, role, records, record)
        calibration, evaluation = self._read_and_verify_ledgers(
            run_dir, manifest, state
        )
        self._apply_consumed_record(
            run_dir,
            state,
            manifest,
            calibration,
            evaluation,
            record,
            validation,
        )
        self._write_state(run_dir, state)
        return self._receipt_from_last(state, idempotent=False)

    def bind_model(
        self,
        capture_run_id: str,
        run_token: str,
        *,
        model_id: str,
        model_sha256: str,
        calibration_ledger_sha256: str,
    ) -> dict[str, Any]:
        """Bind one trained model to the immutable 65-row calibration ledger."""

        normalized_model_id = self._normalized_identifier(model_id, field="model_id")
        normalized_model_sha = self._normalized_sha256(
            model_sha256, field="model_sha256"
        )
        normalized_ledger_sha = self._normalized_sha256(
            calibration_ledger_sha256, field="calibration_ledger_sha256"
        )
        with self._locked_run(capture_run_id) as run_dir:
            state, _, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            self._require_active_run(state)
            if self._runner_state(state).get("runtime_binding") is not None:
                raise MeasurementRunStateError(
                    "dedicated runs require bind_trained_model with artifact provenance"
                )
            if state["phase"] == PHASE_MODEL_BOUND:
                existing = state.get("model_binding") or {}
                same = (
                    existing.get("model_id") == normalized_model_id
                    and existing.get("model_sha256") == normalized_model_sha
                    and existing.get("calibration_ledger_sha256")
                    == normalized_ledger_sha
                )
                if same:
                    return self._public_state(state)
                raise MeasurementRunStateError("a different model is already bound")
            if state["phase"] != PHASE_CALIBRATION_SEALED:
                raise MeasurementRunStateError(
                    "model binding requires exactly 65 sealed calibration samples"
                )
            sealed_sha = state["ledgers"]["calibration"]["sealed_sha256"]
            if not hmac.compare_digest(normalized_ledger_sha, str(sealed_sha or "")):
                raise MeasurementRunIntegrityError(
                    "model binding does not reference the sealed calibration ledger"
                )
            state["model_binding"] = {
                "model_id": normalized_model_id,
                "model_sha256": normalized_model_sha,
                "calibration_ledger_sha256": normalized_ledger_sha,
                "bound_at_utc": _utc_now(),
            }
            state["phase"] = PHASE_MODEL_BOUND
            self._write_state(run_dir, state)
            return self._public_state(state)

    def reserve_training_artifact(
        self,
        capture_run_id: str,
        run_token: str,
        *,
        model_id: str,
        artifact_relative_path: str,
    ) -> dict[str, Any]:
        """Reserve one initially absent deterministic path before CPU training.

        Once this intent is durable, arbitrary partial bytes at the exact path
        are attributable to this run and may be removed after a trainer crash.
        No other model path receives that authority.
        """

        normalized_model_id = self._normalized_identifier(model_id, field="model_id")
        relative = Path(str(artifact_relative_path or ""))
        if relative.is_absolute() or not relative.parts:
            raise MeasurementRunValidationError("artifact_relative_path is invalid")
        artifact_path = (self.root / relative).resolve()
        expected_parent = (self.root / "examples" / "models").resolve()
        expected_model_id = "wgmc_" + "".join(
            character.lower() if character.isalnum() else "_"
            for character in capture_run_id
        )
        if (
            artifact_path.parent != expected_parent
            or artifact_path.suffix != ".json"
            or artifact_path.name != f"{normalized_model_id}.json"
            or normalized_model_id != expected_model_id
        ):
            raise MeasurementRunValidationError(
                "training artifact reservation is not the run-deterministic JSON path"
            )
        with self._locked_run(capture_run_id) as run_dir:
            state, _, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            self._require_active_run(state)
            if state["phase"] != PHASE_CALIBRATION_SEALED:
                raise MeasurementRunStateError(
                    "training artifact reservation requires sealed calibration"
                )
            runner = self._runner_state(state)
            runtime = runner.get("runtime_binding")
            if not isinstance(runtime, Mapping):
                raise MeasurementRunStateError("runtime context is not bound")
            core = {
                "status": "reserved",
                "model_id": normalized_model_id,
                "artifact_relative_path": relative.as_posix(),
                "calibration_session_id": runtime["calibration_session_id"],
                "path_initially_absent_verified": True,
            }
            existing = runner.get("training_artifact_intent")
            if isinstance(existing, Mapping):
                comparable = deepcopy(dict(existing))
                for field in (
                    "reserved_at_utc",
                    "bound_at_utc",
                    "cleanup_verified_at_utc",
                    "model_sha256",
                    "training_provenance_sha256",
                ):
                    comparable.pop(field, None)
                if comparable != core:
                    raise MeasurementRunIntegrityError(
                        "training artifact reservation changed"
                    )
                return self._public_state(state)
            if artifact_path.exists():
                raise MeasurementRunStateError(
                    "deterministic training artifact path was not initially absent"
                )
            intent = deepcopy(core)
            intent["reserved_at_utc"] = _utc_now()
            runner["training_artifact_intent"] = intent
            self._write_state(run_dir, state)
            return self._public_state(state)

    def delete_reserved_training_artifact(
        self, capture_run_id: str, run_token: str
    ) -> dict[str, Any]:
        """Delete arbitrary partial bytes only from a durably reserved path."""

        with self._locked_run(capture_run_id) as run_dir:
            state, _, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            runner = self._runner_state(state)
            if runner.get("trained_artifact") is not None:
                raise MeasurementRunStateError(
                    "a bound trained artifact is not a partial training output"
                )
            intent = runner.get("training_artifact_intent")
            if not isinstance(intent, dict):
                raise MeasurementRunStateError(
                    "training artifact path was not durably reserved"
                )
            relative = Path(str(intent.get("artifact_relative_path") or ""))
            artifact_path = (self.root / relative).resolve()
            expected_parent = (self.root / "examples" / "models").resolve()
            if (
                relative.is_absolute()
                or not relative.parts
                or artifact_path.parent != expected_parent
                or artifact_path.name != f"{intent.get('model_id')}.json"
                or intent.get("path_initially_absent_verified") is not True
            ):
                raise MeasurementRunIntegrityError(
                    "reserved training artifact path binding is invalid"
                )
            if artifact_path.exists():
                if not artifact_path.is_file():
                    raise MeasurementRunIntegrityError(
                        "reserved training artifact path is not a file"
                    )
                artifact_path.unlink()
            if artifact_path.exists():
                raise MeasurementRunIntegrityError(
                    "reserved partial training artifact could not be removed"
                )
            intent["status"] = "cleanup_verified"
            intent["cleanup_verified_at_utc"] = _utc_now()
            self._write_state(run_dir, state)
            return {
                "ok": True,
                "artifact_relative_path": relative.as_posix(),
                "model_absence_verified": True,
            }

    def bind_trained_model(
        self,
        capture_run_id: str,
        run_token: str,
        *,
        model_id: str,
        model_sha256: str,
        artifact_relative_path: str,
        calibration_ledger_sha256: str,
        training_provenance_sha256: str,
    ) -> dict[str, Any]:
        """Verify and bind the actual run-owned personalized JSON artifact."""

        normalized_model_id = self._normalized_identifier(model_id, field="model_id")
        normalized_model_sha = self._normalized_sha256(
            model_sha256, field="model_sha256"
        )
        normalized_ledger_sha = self._normalized_sha256(
            calibration_ledger_sha256, field="calibration_ledger_sha256"
        )
        provenance_sha = self._normalized_sha256(
            training_provenance_sha256, field="training_provenance_sha256"
        )
        relative = Path(str(artifact_relative_path or ""))
        if relative.is_absolute() or not relative.parts:
            raise MeasurementRunValidationError("artifact_relative_path is invalid")
        artifact_path = (self.root / relative).resolve()
        expected_parent = (self.root / "examples" / "models").resolve()
        if artifact_path.parent != expected_parent or artifact_path.suffix != ".json":
            raise MeasurementRunValidationError(
                "trained artifact must be one JSON file in examples/models"
            )
        with self._locked_run(capture_run_id) as run_dir:
            state, _, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            self._require_active_run(state)
            runner = self._runner_state(state)
            if runner.get("runtime_binding") is None:
                raise MeasurementRunStateError("runtime context is not bound")
            if state["phase"] == PHASE_MODEL_BOUND:
                existing = runner.get("trained_artifact") or {}
                if (
                    existing.get("model_id") == normalized_model_id
                    and existing.get("model_sha256") == normalized_model_sha
                    and existing.get("artifact_relative_path") == relative.as_posix()
                    and existing.get("training_provenance_sha256") == provenance_sha
                ):
                    self._verify_bound_model_file(state)
                    return self._public_state(state)
                raise MeasurementRunStateError("a different trained artifact is bound")
            if state["phase"] != PHASE_CALIBRATION_SEALED:
                raise MeasurementRunStateError(
                    "trained-model binding requires sealed calibration attempts"
                )
            intent = runner.get("training_artifact_intent")
            if not isinstance(intent, dict) or (
                intent.get("status") != "reserved"
                or intent.get("model_id") != normalized_model_id
                or intent.get("artifact_relative_path") != relative.as_posix()
                or intent.get("path_initially_absent_verified") is not True
            ):
                raise MeasurementRunIntegrityError(
                    "trained artifact lacks its durable path reservation"
                )
            if len(runner["calibration_manifest_bindings"]) != EXPECTED_ROLE_COUNTS[
                "calibration"
            ]:
                raise MeasurementRunStateError(
                    "training requires exactly 65 usable dedicated manifest rows"
                )
            if any(
                item.get("status") != "committed"
                for item in runner["calibration_manifest_bindings"]
            ):
                raise MeasurementRunIntegrityError(
                    "calibration manifest bindings are not committed"
                )
            sealed_sha = state["ledgers"]["calibration"]["sealed_sha256"]
            if not hmac.compare_digest(normalized_ledger_sha, str(sealed_sha or "")):
                raise MeasurementRunIntegrityError(
                    "trained artifact does not reference the sealed calibration ledger"
                )
            try:
                artifact_bytes = artifact_path.read_bytes()
                artifact = json.loads(artifact_bytes.decode("utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise MeasurementRunIntegrityError(
                    "trained model artifact is unreadable"
                ) from exc
            if hashlib.sha256(artifact_bytes).hexdigest() != normalized_model_sha:
                raise MeasurementRunIntegrityError(
                    "trained model artifact SHA-256 changed before binding"
                )
            if not isinstance(artifact, Mapping):
                raise MeasurementRunIntegrityError(
                    "trained model artifact must be an object"
                )
            provenance = artifact.get("measurement_ceiling_provenance")
            if not isinstance(provenance, Mapping) or _schedule.canonical_sha256(
                provenance
            ) != provenance_sha:
                raise MeasurementRunIntegrityError(
                    "trained model provenance binding is invalid"
                )
            runtime = runner["runtime_binding"]
            expected_provenance = {
                "capture_run_id": capture_run_id,
                "protocol_sha256": state["protocol_sha256"],
                "manifest_sha256": state["manifest_sha256"],
                "calibration_ledger_sha256": normalized_ledger_sha,
                "calibration_session_id": runtime["calibration_session_id"],
                "capture_contract_sha256": runtime["capture_contract_sha256"],
                "calibration_ordinals": list(
                    range(EXPECTED_ROLE_COUNTS["calibration"])
                ),
            }
            for field, expected in expected_provenance.items():
                if provenance.get(field) != expected:
                    raise MeasurementRunIntegrityError(
                        f"trained model provenance {field} changed"
                    )
            fitting_sentinels = {
                "training_role": "calibration_only",
                "train_samples": EXPECTED_ROLE_COUNTS["calibration"],
                "evaluation_labels_used": False,
                "evaluation_rows_used": 0,
                "evaluation_targets_excluded": True,
                "calibration_evaluation_target_intersection_count": 0,
                "allow_cuda": False,
                "training_device_required": "cpu",
                "collection_protocol": "motion-diverse-v1",
            }
            for field, expected in fitting_sentinels.items():
                if provenance.get(field) != expected:
                    raise MeasurementRunIntegrityError(
                        f"trained model fitting sentinel {field} changed"
                    )
            manifest_bindings = provenance.get("calibration_manifest_bindings")
            if not isinstance(manifest_bindings, list) or len(
                manifest_bindings
            ) != EXPECTED_ROLE_COUNTS["calibration"]:
                raise MeasurementRunIntegrityError(
                    "trained model lacks exact calibration manifest bindings"
                )
            if provenance.get(
                "calibration_manifest_bindings_sha256"
            ) != _schedule.canonical_sha256(manifest_bindings):
                raise MeasurementRunIntegrityError(
                    "trained model calibration manifest binding hash changed"
                )
            immutable_bindings = runner["calibration_manifest_bindings"]
            if manifest_bindings != immutable_bindings:
                raise MeasurementRunIntegrityError(
                    "trained model manifest bindings differ from persistent store"
                )
            image_bindings = provenance.get("training_image_bindings")
            if not isinstance(image_bindings, list) or len(image_bindings) != (
                EXPECTED_ROLE_COUNTS["calibration"]
            ):
                raise MeasurementRunIntegrityError(
                    "trained model lacks exact training image bindings"
                )
            if provenance.get(
                "training_image_bindings_sha256"
            ) != _schedule.canonical_sha256(image_bindings):
                raise MeasurementRunIntegrityError(
                    "trained model training-image binding hash changed"
                )
            consumed_rows: list[dict[str, Any]] = []
            for index, (stored, image) in enumerate(
                zip(immutable_bindings, image_bindings, strict=True)
            ):
                if not isinstance(image, Mapping) or set(image) != {
                    "sample_index",
                    "normalized_face_path",
                    "normalized_face_sha256",
                    "raw_path",
                    "raw_frame_sha256",
                    "crop_path",
                    "crop_sha256",
                }:
                    raise MeasurementRunIntegrityError(
                        "trained model image-binding fields changed"
                    )
                if (
                    image.get("sample_index") != index
                    or image.get("normalized_face_sha256")
                    != stored.get("normalized_face_sha256")
                    or image.get("raw_frame_sha256")
                    != stored.get("raw_frame_sha256")
                    or image.get("crop_sha256") != stored.get("crop_sha256")
                ):
                    raise MeasurementRunIntegrityError(
                        "trained model image bytes differ from capture-time binding"
                    )
                consumed_rows.append(
                    {
                        "sequence_index": index,
                        "manifest_sample_index": index,
                        "manifest_record_sha256": stored[
                            "manifest_record_sha256"
                        ],
                        "frame_sha256": stored["frame_sha256"],
                        "normalized_face_path": image["normalized_face_path"],
                        "normalized_face_sha256": stored[
                            "normalized_face_sha256"
                        ],
                    }
                )
            expected_rows_sha = _schedule.canonical_sha256(consumed_rows)
            training_base_bundle = {
                field: runtime["base_inference_bundle"][field]
                for field in (
                    "model_id",
                    "model_name",
                    "model_sha256",
                    "bundle_sha256",
                    "checkpoint_sha256",
                )
            }
            expected_binding_core = {
                "schema_version": 1,
                "binding_type": (
                    "webcam_gaze_measurement_ceiling_training_input_binding_v1"
                ),
                "data_session_id": runtime["calibration_session_id"],
                "capture_run_id": capture_run_id,
                "manifest_sha256": provenance.get(
                    "calibration_manifest_sha256"
                ),
                "base_inference_bundle": training_base_bundle,
                "rows": consumed_rows,
                "rows_sha256": expected_rows_sha,
            }
            expected_binding_sha = _schedule.canonical_sha256(
                expected_binding_core
            )
            if (
                provenance.get("consumed_training_rows_sha256")
                != expected_rows_sha
                or provenance.get("measurement_training_binding_sha256")
                != expected_binding_sha
            ):
                raise MeasurementRunIntegrityError(
                    "trained model consumed-byte provenance changed"
                )
            if artifact.get("measurement_training_input_binding") != {
                "binding_sha256": expected_binding_sha,
                "rows_sha256": expected_rows_sha,
                "row_count": EXPECTED_ROLE_COUNTS["calibration"],
                "capture_run_id": capture_run_id,
                "base_inference_bundle": training_base_bundle,
            }:
                raise MeasurementRunIntegrityError(
                    "trained model artifact lacks its exact consumed-byte binding"
                )
            if provenance.get(
                "training_consumed_base_inference_bundle"
            ) != training_base_bundle or provenance.get(
                "post_training_base_inference_bundle_verified"
            ) is not True:
                raise MeasurementRunIntegrityError(
                    "trained model base-bundle provenance changed"
                )
            if artifact.get("train_samples") != EXPECTED_ROLE_COUNTS["calibration"]:
                raise MeasurementRunIntegrityError(
                    "trained model does not contain exactly 65 samples"
                )
            if artifact.get("training_device") != "cpu":
                raise MeasurementRunIntegrityError("trained model was not fit on CPU")
            uncertainty = artifact.get("uncertainty_v2")
            if not isinstance(uncertainty, Mapping) or uncertainty.get("status") != (
                "scored_no_threshold"
            ):
                raise MeasurementRunIntegrityError(
                    "trained model lacks uncertainty_v2 scored_no_threshold"
                )
            runner["trained_artifact"] = {
                "model_id": normalized_model_id,
                "model_sha256": normalized_model_sha,
                "artifact_relative_path": relative.as_posix(),
                "calibration_ledger_sha256": normalized_ledger_sha,
                "training_provenance_sha256": provenance_sha,
                "bound_at_utc": _utc_now(),
            }
            intent["status"] = "bound"
            intent["model_sha256"] = normalized_model_sha
            intent["training_provenance_sha256"] = provenance_sha
            intent["bound_at_utc"] = _utc_now()
            state["model_binding"] = {
                "model_id": normalized_model_id,
                "model_sha256": normalized_model_sha,
                "calibration_ledger_sha256": normalized_ledger_sha,
                "bound_at_utc": _utc_now(),
            }
            state["phase"] = PHASE_MODEL_BOUND
            self._write_state(run_dir, state)
            return self._public_state(state)

    def read_calibration_training_binding(
        self, capture_run_id: str, run_token: str
    ) -> dict[str, Any]:
        """Return immutable, derived-only identities required by CPU training."""

        with self._locked_run(capture_run_id) as run_dir:
            state, manifest, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            if state["phase"] not in {PHASE_CALIBRATION_SEALED, PHASE_MODEL_BOUND}:
                raise MeasurementRunStateError(
                    "calibration training binding is unavailable in this phase"
                )
            bindings = deepcopy(
                self._runner_state(state)["calibration_manifest_bindings"]
            )
            if len(bindings) != EXPECTED_ROLE_COUNTS["calibration"] or any(
                item.get("status") != "committed" for item in bindings
            ):
                raise MeasurementRunStateError(
                    "exact 65-row calibration training binding is unavailable"
                )
            rows = [
                deepcopy(row)
                for row in manifest["rows"]
                if row["block_role"] == "calibration"
            ]
            return {
                "capture_run_id": capture_run_id,
                "calibration_ledger_sha256": state["ledgers"]["calibration"][
                    "sealed_sha256"
                ],
                "calibration_manifest_bindings": bindings,
                "calibration_manifest_bindings_sha256": _schedule.canonical_sha256(
                    bindings
                ),
                "calibration_schedule_rows_sha256": _schedule.canonical_sha256(rows),
                "measurement_claim_authorized": False,
            }

    def _verify_bound_model_file(self, state: Mapping[str, Any]) -> None:
        trained = self._runner_state(state).get("trained_artifact")
        if not isinstance(trained, Mapping):
            raise MeasurementRunIntegrityError("trained artifact binding is missing")
        relative = Path(str(trained.get("artifact_relative_path") or ""))
        artifact_path = (self.root / relative).resolve()
        expected_parent = (self.root / "examples" / "models").resolve()
        if artifact_path.parent != expected_parent or not artifact_path.is_file():
            raise MeasurementRunIntegrityError("bound trained artifact is missing")
        observed = _file_sha256(artifact_path)
        expected = str(trained.get("model_sha256") or "")
        if not LOWER_SHA256_PATTERN.fullmatch(expected) or not hmac.compare_digest(
            observed, expected
        ):
            raise MeasurementRunIntegrityError(
                "bound trained artifact SHA-256 changed"
            )

    def record_calibration_image_purge(
        self,
        capture_run_id: str,
        run_token: str,
        *,
        calibration_session_id: str,
        removed_directories: Sequence[str],
        postcondition_verified: bool,
    ) -> dict[str, Any]:
        """Persist the exact dedicated-image purge as the evaluation gate."""

        session_id = self._normalized_identifier(
            calibration_session_id, field="calibration_session_id"
        )
        removed = sorted(str(item) for item in removed_directories)
        expected = sorted(DEDICATED_IMAGE_DIRECTORIES)
        if postcondition_verified is not True:
            raise MeasurementRunStateError("calibration image purge is unverified")
        if any(item not in DEDICATED_IMAGE_DIRECTORIES for item in removed):
            raise MeasurementRunValidationError(
                "purge response contains an unexpected directory"
            )
        with self._locked_run(capture_run_id) as run_dir:
            state, _, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            self._require_active_run(state)
            if state["phase"] not in {PHASE_CALIBRATION_SEALED, PHASE_MODEL_BOUND}:
                raise MeasurementRunStateError(
                    "calibration images may be sealed only after calibration ends"
                )
            runner = self._runner_state(state)
            runtime = runner.get("runtime_binding") or {}
            if runtime.get("calibration_session_id") != session_id:
                raise MeasurementRunIntegrityError(
                    "purge targeted a different calibration session"
                )
            existing = runner.get("calibration_image_purge")
            proof = {
                "status": "verified",
                "calibration_session_id": session_id,
                "required_directories": expected,
                "removed_directories": removed,
                "postcondition_verified": True,
            }
            if existing is not None:
                existing_core = deepcopy(dict(existing))
                existing_core.pop("verified_at_utc", None)
                if existing_core == proof:
                    return self._public_state(state)
                raise MeasurementRunIntegrityError("calibration purge proof changed")
            proof["verified_at_utc"] = _utc_now()
            runner["calibration_image_purge"] = proof
            self._write_state(run_dir, state)
            return self._public_state(state)

    def fail_integrity(
        self,
        capture_run_id: str,
        run_token: str,
        *,
        code: str,
        detail: str,
    ) -> None:
        """Fail one authenticated dedicated run without touching other data."""

        normalized_code = _normalized_key(code)
        if not normalized_code or len(normalized_code) > 96:
            raise MeasurementRunValidationError("failure code is invalid")
        with self._locked_run(capture_run_id) as run_dir:
            state = self._read_state_unlocked(run_dir)
            self._authenticate(state, run_token)
            self._mark_failed_unlocked(
                run_dir, state, code=normalized_code, detail=str(detail)
            )

    def verify_sealed_artifact(
        self, capture_run_id: str, run_token: str
    ) -> dict[str, Any]:
        """Re-read and independently verify the exact 193-row artifact."""

        with self._locked_run(capture_run_id) as run_dir:
            state, _, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            self._require_active_run(state, allow_verified=True)
            if state["phase"] not in {
                PHASE_CAPTURE_SEALED,
                PHASE_ARTIFACT_VERIFIED,
            }:
                raise MeasurementRunStateError(
                    "capture artifact is unavailable until all 193 samples are sealed"
                )
            runner = self._runner_state(state)
            runtime_bound = runner.get("runtime_binding") is not None
            spool_dir = (run_dir / "spool").resolve()
            if runtime_bound and (
                runner.get("frame_spool") is not None
                or (spool_dir.is_dir() and any(spool_dir.iterdir()))
            ):
                raise MeasurementRunStateError(
                    "capture artifact verification waits for encrypted spool cleanup"
                )
            if runtime_bound and state.get(
                "capture_contract_binding_verified"
            ) is not True:
                raise MeasurementRunStateError(
                    "capture artifact lacks all 193 decoded capture-contract proofs"
                )
            purge = runner.get("calibration_image_purge")
            if runtime_bound and (
                not isinstance(purge, Mapping) or purge.get("status") != "verified"
            ):
                raise MeasurementRunStateError(
                    "capture artifact lacks verified calibration-image purge"
                )
            if runtime_bound and not isinstance(
                runner.get("trained_artifact"), Mapping
            ):
                raise MeasurementRunStateError(
                    "capture artifact lacks its personalized model provenance"
                )
            if runtime_bound and {
                item.get("stage") for item in runner["base_bundle_checks"]
            } != {
                "run_created",
                "first_calibration_inference",
                "calibration_sealed_pre_training",
            }:
                raise MeasurementRunStateError(
                    "capture artifact lacks exact base-bundle checkpoints"
                )
            artifact = self._read_artifact_unlocked(run_dir, state)
            try:
                summary = _schedule.verify_capture_artifact(
                    artifact, protocol_path=self.protocol_path
                )
            except _schedule.MeasurementScheduleError as exc:
                self._mark_failed_unlocked(
                    run_dir,
                    state,
                    code="capture_artifact_verification_failed",
                    detail=str(exc),
                )
                raise MeasurementRunIntegrityError(str(exc)) from exc
            if state["phase"] != PHASE_ARTIFACT_VERIFIED:
                state["phase"] = PHASE_ARTIFACT_VERIFIED
                state["acquisition_artifact_verified"] = True
                state["capture_artifact"]["verified_at_utc"] = _utc_now()
                self._write_state(run_dir, state)
            return {
                **summary,
                "phase": state["phase"],
                "acquisition_artifact_verified": True,
                "capture_contract_binding_verified": state[
                    "capture_contract_binding_verified"
                ],
                "measurement_claim_authorized": False,
                "physical_capture_claim_authorized": False,
            }

    def read_sealed_artifact(
        self, capture_run_id: str, run_token: str
    ) -> dict[str, Any]:
        with self._locked_run(capture_run_id) as run_dir:
            state, _, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            if state["phase"] not in {
                PHASE_CAPTURE_SEALED,
                PHASE_ARTIFACT_VERIFIED,
            }:
                raise MeasurementRunStateError("capture artifact is not sealed")
            return deepcopy(self._read_artifact_unlocked(run_dir, state))

    def read_sealed_attempt_sidecar(
        self, capture_run_id: str, run_token: str
    ) -> dict[str, Any]:
        with self._locked_run(capture_run_id) as run_dir:
            state, _, calibration, evaluation = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            if state["phase"] not in {
                PHASE_CAPTURE_SEALED,
                PHASE_ARTIFACT_VERIFIED,
            }:
                raise MeasurementRunStateError("attempt sidecar is not sealed")
            return deepcopy(
                self._read_attempt_sidecar_unlocked(
                    run_dir, state, calibration, evaluation
                )
            )

    def abort_run(
        self,
        capture_run_id: str,
        run_token: str,
        *,
        reason: str,
        cleanup_proof: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._locked_run(capture_run_id) as run_dir:
            state, _, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            if state["phase"] in TERMINAL_PHASES:
                if state["phase"] == PHASE_ABORTED:
                    return self._public_state(state)
                raise MeasurementRunStateError("terminal run cannot be aborted")
            normalized_reason = str(reason or "").strip()
            if not normalized_reason or len(normalized_reason) > 256:
                raise MeasurementRunValidationError(
                    "abort reason must contain 1 to 256 characters"
                )
            verified_cleanup = self._verified_terminal_cleanup_unlocked(
                run_dir,
                state,
                cleanup_proof,
            )
            state["active_challenge"] = None
            state["phase"] = PHASE_ABORTED
            state["abort"] = {
                "reason": normalized_reason,
                "at_utc": _utc_now(),
                "cleanup": verified_cleanup,
            }
            self._write_state(run_dir, state)
            return self._public_state(state)

    def complete_failed_integrity_cleanup(
        self, capture_run_id: str, run_token: str
    ) -> dict[str, Any]:
        """Verify cleanup for a failed run without rewriting its conclusion."""

        with self._locked_run(capture_run_id) as run_dir:
            state, _, _, _ = self._load_authenticated_unlocked(
                run_dir, capture_run_id, run_token
            )
            if state["phase"] != PHASE_FAILED_INTEGRITY:
                raise MeasurementRunStateError(
                    "terminal-failure cleanup requires failed_integrity"
                )
            proof = self._verified_terminal_cleanup_unlocked(
                run_dir, state, None
            )
            failure = state.get("failure")
            if not isinstance(failure, dict):
                raise MeasurementRunIntegrityError(
                    "failed run lacks failure evidence"
                )
            failure["cleanup"] = proof
            failure["cleanup_verified_at_utc"] = _utc_now()
            state["active_challenge"] = None
            self._write_state(run_dir, state)
            return self._public_state(state)

    def _verified_terminal_cleanup_unlocked(
        self,
        run_dir: Path,
        state: Mapping[str, Any],
        supplied: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        runner = self._runner_state(state)
        if runner.get("frame_spool") is not None:
            raise MeasurementRunStateError(
                "encrypted frame spool must be cleaned before abort"
            )
        spool_dir = (run_dir / "spool").resolve()
        if spool_dir.parent != run_dir or (
            spool_dir.is_dir() and any(spool_dir.iterdir())
        ):
            raise MeasurementRunStateError(
                "encrypted frame spool absence is not verified"
            )
        session_ids = self._dedicated_session_ids_for_run(
            str(state["capture_run_id"])
        )
        runtime = runner.get("runtime_binding")
        if isinstance(runtime, Mapping) and runtime.get(
            "calibration_session_id"
        ) not in session_ids:
            raise MeasurementRunIntegrityError(
                "bound dedicated calibration session metadata is missing"
            )
        for session_id in session_ids:
            session_dir = (
                self.root / "data" / "sessions" / session_id
            ).resolve()
            sessions_root = (self.root / "data" / "sessions").resolve()
            if session_dir.parent != sessions_root:
                raise MeasurementRunIntegrityError(
                    "dedicated calibration session path is unsafe"
                )
            if any((session_dir / name).exists() for name in DEDICATED_IMAGE_DIRECTORIES):
                raise MeasurementRunStateError(
                    "dedicated calibration images remain; abort is retryable"
                )
        model_name = "wgmc_" + "".join(
            character.lower() if character.isalnum() else "_"
            for character in str(state["capture_run_id"])
        )
        model_path = (
            self.root / "examples" / "models" / f"{model_name}.json"
        ).resolve()
        if model_path.parent != (self.root / "examples" / "models").resolve():
            raise MeasurementRunIntegrityError(
                "dedicated personalized-model path is unsafe"
            )
        if model_path.exists():
            raise MeasurementRunStateError(
                "dedicated personalized model remains; abort is retryable"
            )
        expected = {
            "cleanup_verified": True,
            "calibration_session_ids": session_ids,
            "required_image_directories": sorted(DEDICATED_IMAGE_DIRECTORIES),
            "image_directories_absent": True,
            "spool_absence_verified": True,
            "model_absence_verified": True,
        }
        if supplied is not None and deepcopy(dict(supplied)) != expected:
            raise MeasurementRunValidationError(
                "abort cleanup proof differs from server verification"
            )
        return expected

    def _dedicated_session_ids_for_run(self, capture_run_id: str) -> list[str]:
        sessions_root = (self.root / "data" / "sessions").resolve()
        if not sessions_root.is_dir():
            return []
        matches: list[str] = []
        for child in sorted(sessions_root.iterdir(), key=lambda path: path.name):
            if not child.is_dir() or child.parent.resolve() != sessions_root:
                continue
            metadata_path = child / "session.json"
            if not metadata_path.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if (
                isinstance(metadata, Mapping)
                and metadata.get("capture_run_id") == capture_run_id
                and metadata.get("capture_source")
                == "direct-webcam-self-development"
                and metadata.get("participant_id") == f"wgmc_{capture_run_id}"
                and metadata.get("session_id") == child.name
            ):
                matches.append(child.name)
        return matches

    @contextmanager
    def _locked_run(self, capture_run_id: str):
        normalized = self._normalized_run_id(capture_run_id)
        run_dir = self._run_dir(normalized)
        if not run_dir.is_dir():
            raise MeasurementRunStateError("measurement run not found")
        with self._process_lock:
            with _exclusive_file_lock(run_dir / ".lock"):
                yield run_dir

    def _new_capture_run_id(self) -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        for _ in range(32):
            candidate = f"WGMC-{timestamp}-{secrets.token_hex(12)}"
            if not self._run_dir(candidate).exists():
                return candidate
        raise MeasurementRunStateError("could not allocate a unique capture run id")

    @staticmethod
    def _normalized_create_request_id(value: object) -> str:
        if isinstance(value, bool):
            raise MeasurementRunAuthenticationError(
                "create_request_id must use WGMCREQ plus 128 random bits"
            )
        text = str(value or "")
        if not CREATE_REQUEST_ID_PATTERN.fullmatch(text):
            raise MeasurementRunAuthenticationError(
                "create_request_id must use WGMCREQ plus 128 lowercase hex bits"
            )
        return text

    @staticmethod
    def _normalized_client_run_token(value: object) -> str:
        if isinstance(value, bool):
            raise MeasurementRunAuthenticationError(
                "run_token must contain 256 random bits"
            )
        text = str(value or "")
        if not CLIENT_RUN_TOKEN_PATTERN.fullmatch(text):
            raise MeasurementRunAuthenticationError(
                "run_token must use wgmr_client plus 256 lowercase hex bits"
            )
        return text

    def _read_create_registry_unlocked(self) -> dict[str, Any]:
        path = self.store_root / CREATE_REGISTRY_FILENAME
        if not path.exists():
            registry: dict[str, Any] = {
                "schema_version": 1,
                "registry_type": CREATE_REGISTRY_TYPE,
                "entries": [],
            }
            registry["registry_sha256"] = _schedule.canonical_sha256(registry)
            return registry
        try:
            registry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MeasurementRunIntegrityError(
                "create registry is unreadable"
            ) from exc
        if not isinstance(registry, dict) or set(registry) != {
            "schema_version",
            "registry_type",
            "entries",
            "registry_sha256",
        }:
            raise MeasurementRunIntegrityError("create registry fields changed")
        if (
            registry.get("schema_version") != 1
            or registry.get("registry_type") != CREATE_REGISTRY_TYPE
        ):
            raise MeasurementRunIntegrityError("create registry identity changed")
        stored_sha = str(registry.get("registry_sha256") or "")
        if not LOWER_SHA256_PATTERN.fullmatch(stored_sha) or not hmac.compare_digest(
            stored_sha,
            _schedule.canonical_sha256(_registry_without_hash(registry)),
        ):
            raise MeasurementRunIntegrityError("create registry SHA-256 mismatch")
        entries = registry.get("entries")
        if not isinstance(entries, list):
            raise MeasurementRunIntegrityError("create registry entries are invalid")
        request_hashes: set[str] = set()
        run_ids: set[str] = set()
        for entry in entries:
            if not isinstance(entry, Mapping) or set(entry) != {
                "create_request_sha256",
                "run_token_sha256",
                "capture_run_id",
                "created_at_utc",
            }:
                raise MeasurementRunIntegrityError(
                    "create registry entry fields changed"
                )
            for field in ("create_request_sha256", "run_token_sha256"):
                if not LOWER_SHA256_PATTERN.fullmatch(str(entry.get(field) or "")):
                    raise MeasurementRunIntegrityError(
                        f"create registry {field} is malformed"
                    )
            run_id = self._normalized_run_id(entry.get("capture_run_id"))
            request_sha = str(entry["create_request_sha256"])
            if request_sha in request_hashes or run_id in run_ids:
                raise MeasurementRunIntegrityError(
                    "create registry contains a duplicate binding"
                )
            request_hashes.add(request_sha)
            run_ids.add(run_id)
        return registry

    def _write_create_registry_unlocked(self, registry: dict[str, Any]) -> None:
        registry["entries"] = sorted(
            registry["entries"], key=lambda item: item["create_request_sha256"]
        )
        registry["registry_sha256"] = _schedule.canonical_sha256(
            _registry_without_hash(registry)
        )
        _atomic_json(self.store_root / CREATE_REGISTRY_FILENAME, registry)

    def _find_create_entry_unlocked(
        self,
        registry: dict[str, Any],
        *,
        request_sha: str,
        token_sha: str,
        repair_registry: bool,
    ) -> dict[str, Any] | None:
        matches = [
            entry
            for entry in registry["entries"]
            if hmac.compare_digest(
                str(entry["create_request_sha256"]), request_sha
            )
        ]
        if len(matches) > 1:
            raise MeasurementRunIntegrityError(
                "create authority maps to multiple registry entries"
            )
        if matches:
            entry = matches[0]
            if not hmac.compare_digest(str(entry["run_token_sha256"]), token_sha):
                raise MeasurementRunAuthenticationError(
                    "create request is already bound to another run secret"
                )
            run_dir = self._run_dir(str(entry["capture_run_id"]))
            if not run_dir.is_dir():
                raise MeasurementRunIntegrityError(
                    "create registry references a missing run"
                )
            state = self._read_state_unlocked(run_dir)
            if (
                state.get("create_request_sha256") != request_sha
                or state.get("run_token_sha256") != token_sha
            ):
                raise MeasurementRunIntegrityError(
                    "create registry differs from run state"
                )
            return dict(entry)

        recovered: list[dict[str, Any]] = []
        if self.store_root.is_dir():
            for run_dir in sorted(self.store_root.iterdir(), key=lambda path: path.name):
                if not run_dir.is_dir() or not SAFE_RUN_ID_PATTERN.fullmatch(
                    run_dir.name
                ):
                    continue
                try:
                    state = self._read_state_unlocked(run_dir)
                except MeasurementRunIntegrityError:
                    # A corrupt unrelated run must not prevent authenticated
                    # recovery, but a matching digest cannot be safely ignored.
                    try:
                        raw = json.loads(
                            (run_dir / STATE_FILENAME).read_text(encoding="utf-8")
                        )
                    except Exception:
                        continue
                    if raw.get("create_request_sha256") == request_sha:
                        raise
                    continue
                if state.get("create_request_sha256") == request_sha:
                    recovered.append(
                        {
                            "create_request_sha256": request_sha,
                            "run_token_sha256": str(state.get("run_token_sha256") or ""),
                            "capture_run_id": str(state.get("capture_run_id") or ""),
                            "created_at_utc": str(state.get("created_at_utc") or ""),
                        }
                    )
        if len(recovered) > 1:
            raise MeasurementRunIntegrityError(
                "create authority maps to multiple durable runs"
            )
        if not recovered:
            return None
        entry = recovered[0]
        if not hmac.compare_digest(str(entry["run_token_sha256"]), token_sha):
            raise MeasurementRunAuthenticationError(
                "create request is already bound to another run secret"
            )
        if repair_registry:
            registry["entries"].append(entry)
            self._write_create_registry_unlocked(registry)
        return entry

    def _run_dir(self, capture_run_id: str) -> Path:
        candidate = (self.store_root / capture_run_id).resolve()
        if candidate.parent != self.store_root:
            raise MeasurementRunStateError("invalid capture run path")
        return candidate

    @staticmethod
    def _normalized_run_id(value: object) -> str:
        text = str(value or "").strip()
        if not SAFE_RUN_ID_PATTERN.fullmatch(text):
            raise MeasurementRunStateError("invalid capture run id")
        return text

    @staticmethod
    def _normalized_secret(value: object) -> str:
        text = str(value or "")
        if not text or len(text) > 256 or text != text.strip():
            raise MeasurementRunAuthenticationError("invalid secret token")
        return text

    @staticmethod
    def _normalized_identifier(value: object, *, field: str) -> str:
        text = str(value or "").strip()
        if not NORMALIZED_IDENTIFIER_PATTERN.fullmatch(text):
            raise MeasurementRunValidationError(f"{field} is invalid")
        return text

    @staticmethod
    def _normalized_sha256(value: object, *, field: str) -> str:
        text = str(value or "")
        if not LOWER_SHA256_PATTERN.fullmatch(text):
            raise MeasurementRunValidationError(f"{field} must be lowercase SHA-256")
        return text

    @staticmethod
    def _positive_finite(value: object, *, field: str) -> float:
        if isinstance(value, bool):
            raise MeasurementRunValidationError(f"{field} must be positive")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise MeasurementRunValidationError(
                f"{field} must be positive"
            ) from exc
        if not math.isfinite(number) or number <= 0:
            raise MeasurementRunValidationError(f"{field} must be positive")
        return number

    @staticmethod
    def _runner_state(state: Mapping[str, Any]) -> dict[str, Any]:
        runner = state.get("runner")
        if not isinstance(runner, dict):
            raise MeasurementRunIntegrityError("runner state is invalid")
        expected = {
            "runtime_binding",
            "calibration_write",
            "calibration_manifest_bindings",
            "capture_contract_proofs",
            "frame_spool",
            "inference_intent",
            "base_bundle_checks",
            "training_artifact_intent",
            "trained_artifact",
            "calibration_image_purge",
        }
        if set(runner) != expected:
            raise MeasurementRunIntegrityError("runner state fields changed")
        return runner

    def _validated_capture_contract_evidence(
        self,
        runtime_binding: Mapping[str, Any],
        evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        contract = runtime_binding.get("capture_contract")
        if not isinstance(contract, Mapping):
            raise MeasurementRunIntegrityError("runtime capture contract is invalid")
        normalized = deepcopy(dict(evidence))
        required = {
            "observed_capture_contract",
            "observed_capture_contract_sha256",
            "decoded_transport_width_px",
            "decoded_transport_height_px",
            "contract_comparison",
            "transport_frame_validated",
            "viewport",
            "viewport_sha256",
        }
        if set(normalized) != required:
            raise MeasurementRunValidationError(
                "capture-contract evidence fields are not exact"
            )
        try:
            observed = normalize_capture_contract(
                normalized["observed_capture_contract"]
            )
        except ValueError as exc:
            raise MeasurementRunValidationError(str(exc)) from exc
        comparison = compare_capture_contracts(contract, observed)
        if comparison["compatible"] is not True:
            raise MeasurementRunValidationError(
                "observed capture contract is incompatible with frozen binding"
            )
        expected = {
            "observed_capture_contract": observed,
            "observed_capture_contract_sha256": _schedule.canonical_sha256(
                observed
            ),
            "decoded_transport_width_px": observed["transport_width_px"],
            "decoded_transport_height_px": observed["transport_height_px"],
            "contract_comparison": comparison,
            "transport_frame_validated": True,
            "viewport": runtime_binding["viewport"],
            "viewport_sha256": _schedule.canonical_sha256(
                runtime_binding["viewport"]
            ),
        }
        if normalized != expected:
            raise MeasurementRunValidationError(
                "capture-contract/viewport evidence differs from server verification"
            )
        return normalized

    @staticmethod
    def _validated_server_receive_context(
        value: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise MeasurementRunValidationError(
                "server_receive_context must be an object"
            )
        normalized = deepcopy(dict(value))
        if set(normalized) != {
            "server_request_received_monotonic_ms",
            "decode_completed_monotonic_ms",
            "frame_source",
        }:
            raise MeasurementRunValidationError(
                "server receive timing fields are not exact"
            )
        received = _finite_number(
            normalized["server_request_received_monotonic_ms"],
            field="server_request_received_monotonic_ms",
        )
        decoded = _finite_number(
            normalized["decode_completed_monotonic_ms"],
            field="decode_completed_monotonic_ms",
        )
        if received < 0 or decoded < received:
            raise MeasurementRunValidationError(
                "server receive timing order is invalid"
            )
        if normalized["frame_source"] not in {
            "request_decode",
            "encrypted_spool_recovery",
        }:
            raise MeasurementRunValidationError("frame source is invalid")
        normalized["server_request_received_monotonic_ms"] = received
        normalized["decode_completed_monotonic_ms"] = decoded
        return normalized

    @classmethod
    def _validated_server_timing_evidence(
        cls, value: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise MeasurementRunValidationError(
                "server_timing_evidence must be an object"
            )
        normalized = deepcopy(dict(value))
        required = {
            "schema_version",
            "timing_semantics",
            "server_request_received_monotonic_ms",
            "decode_completed_monotonic_ms",
            "predict_started_monotonic_ms",
            "predict_completed_monotonic_ms",
            "frame_source",
            "camera_exposure_timestamp_available",
            "client_timing_used_for_integrity",
        }
        if set(normalized) != required:
            raise MeasurementRunValidationError(
                "server timing evidence fields are not exact"
            )
        receive = cls._validated_server_receive_context(
            {
                field: normalized[field]
                for field in (
                    "server_request_received_monotonic_ms",
                    "decode_completed_monotonic_ms",
                    "frame_source",
                )
            }
        )
        started = _finite_number(
            normalized["predict_started_monotonic_ms"],
            field="predict_started_monotonic_ms",
        )
        completed = _finite_number(
            normalized["predict_completed_monotonic_ms"],
            field="predict_completed_monotonic_ms",
        )
        if started < receive["decode_completed_monotonic_ms"] or completed < started:
            raise MeasurementRunValidationError(
                "server inference timing order is invalid"
            )
        if normalized.get("schema_version") != 1 or normalized.get(
            "timing_semantics"
        ) != "v1_sample_frame_capture_is_predict_start_proxy_not_camera_exposure":
            raise MeasurementRunValidationError(
                "server timing semantics are invalid"
            )
        if normalized.get("camera_exposure_timestamp_available") is not False or (
            normalized.get("client_timing_used_for_integrity") is not False
        ):
            raise MeasurementRunValidationError(
                "server timing claim boundary changed"
            )
        normalized.update(receive)
        normalized["predict_started_monotonic_ms"] = started
        normalized["predict_completed_monotonic_ms"] = completed
        return normalized

    @staticmethod
    def _new_challenge(
        *,
        ordinal: int,
        row: Mapping[str, Any],
        role: str,
        rotation_count: int = 0,
    ) -> dict[str, Any]:
        token = _secret_token("wgmc_ch_")
        persisted = {
            "challenge_id": f"ch-{secrets.token_hex(12)}",
            "challenge_token_sha256": _token_sha256(token),
            "ordinal": ordinal,
            "ledger_role": role,
            "schedule_row_sha256": _schedule.canonical_sha256(row),
            "issued_at_utc": _utc_now(),
            "rotation_count": rotation_count,
        }
        return {"plaintext_token": token, "persisted": persisted}

    def _read_state_unlocked(self, run_dir: Path) -> dict[str, Any]:
        path = run_dir / STATE_FILENAME
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MeasurementRunIntegrityError("run state is unreadable") from exc
        if not isinstance(state, dict):
            raise MeasurementRunIntegrityError("run state must be an object")
        stored_sha = str(state.get("state_sha256") or "")
        if not LOWER_SHA256_PATTERN.fullmatch(stored_sha):
            raise MeasurementRunIntegrityError("run state SHA-256 is malformed")
        if not hmac.compare_digest(
            stored_sha, _schedule.canonical_sha256(_state_without_hash(state))
        ):
            raise MeasurementRunIntegrityError("run state SHA-256 mismatch")
        if state.get("schema_version") != STATE_SCHEMA_VERSION:
            raise MeasurementRunIntegrityError("run state schema changed")
        if state.get("state_type") != STATE_TYPE:
            raise MeasurementRunIntegrityError("run state type changed")
        if state.get("phase") not in ALL_PHASES:
            raise MeasurementRunIntegrityError("run state phase is invalid")
        if state.get("measurement_claim_authorized") is not False:
            raise MeasurementRunIntegrityError("measurement claim boundary changed")
        if state.get("physical_capture_claim_authorized") is not False:
            raise MeasurementRunIntegrityError("physical capture claim boundary changed")
        if not LOWER_SHA256_PATTERN.fullmatch(
            str(state.get("create_request_sha256") or "")
        ):
            raise MeasurementRunIntegrityError(
                "create request binding SHA-256 is malformed"
            )
        if not LOWER_SHA256_PATTERN.fullmatch(
            str(state.get("run_token_sha256") or "")
        ):
            raise MeasurementRunIntegrityError(
                "run token binding SHA-256 is malformed"
            )
        return state

    def _write_state(self, run_dir: Path, state: dict[str, Any]) -> None:
        state["updated_at_utc"] = _utc_now()
        state["state_sha256"] = _schedule.canonical_sha256(
            _state_without_hash(state)
        )
        _atomic_json(run_dir / STATE_FILENAME, state)

    def _authenticate(self, state: Mapping[str, Any], run_token: str) -> None:
        supplied = _token_sha256(self._normalized_secret(run_token))
        expected = str(state.get("run_token_sha256") or "")
        if not LOWER_SHA256_PATTERN.fullmatch(expected) or not hmac.compare_digest(
            supplied, expected
        ):
            raise MeasurementRunAuthenticationError("run token is invalid")

    def _load_authenticated_unlocked(
        self, run_dir: Path, capture_run_id: str, run_token: str
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        state = self._read_state_unlocked(run_dir)
        self._authenticate(state, run_token)
        if state.get("capture_run_id") != capture_run_id:
            raise MeasurementRunIntegrityError("state capture run id changed")
        try:
            manifest = self._read_manifest_unlocked(run_dir, state)
            calibration, evaluation = self._read_and_verify_ledgers(
                run_dir, manifest, state
            )
            self._reconcile_unlocked(
                run_dir, state, manifest, calibration, evaluation
            )
            self._validate_state_against_persistence(
                run_dir, state, manifest, calibration, evaluation
            )
        except (MeasurementRunIntegrityError, _schedule.MeasurementScheduleError) as exc:
            self._mark_failed_unlocked(
                run_dir,
                state,
                code="persisted_run_integrity_failed",
                detail=str(exc),
            )
            if isinstance(exc, MeasurementRunIntegrityError):
                raise
            raise MeasurementRunIntegrityError(str(exc)) from exc
        return state, manifest, calibration, evaluation

    def _read_manifest_unlocked(
        self, run_dir: Path, state: Mapping[str, Any]
    ) -> dict[str, Any]:
        try:
            manifest = json.loads(
                (run_dir / RUN_MANIFEST_FILENAME).read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MeasurementRunIntegrityError("run manifest is unreadable") from exc
        summary = _schedule.verify_run_manifest(
            manifest, protocol_path=self.protocol_path
        )
        checks = {
            "capture_run_id": summary["capture_run_id"],
            "protocol_sha256": summary["protocol_sha256"],
            "manifest_sha256": summary["manifest_sha256"],
            "manifest_rows_sha256": manifest["rows_sha256"],
        }
        for field, expected in checks.items():
            observed = state.get(field)
            if observed != expected:
                raise MeasurementRunIntegrityError(
                    f"state {field} differs from immutable run manifest"
                )
        return dict(manifest)

    def _read_and_verify_ledgers(
        self,
        run_dir: Path,
        manifest: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        calibration = self._read_ledger_unlocked(
            run_dir, "calibration", manifest, state
        )
        evaluation = self._read_ledger_unlocked(
            run_dir, "evaluation", manifest, state
        )
        if evaluation["count"] and calibration["count"] != EXPECTED_ROLE_COUNTS[
            "calibration"
        ]:
            raise MeasurementRunIntegrityError(
                "evaluation ledger exists before calibration is complete"
            )
        samples = [
            record["sample"]
            for record in calibration["records"] + evaluation["records"]
        ]
        self._validate_cross_sample_contract(samples)
        bound = state.get("model_binding")
        if evaluation["count"]:
            if not isinstance(bound, Mapping):
                raise MeasurementRunIntegrityError(
                    "evaluation ledger exists without a bound model"
                )
            expected_binding = (bound.get("model_id"), bound.get("model_sha256"))
            for record in evaluation["records"]:
                sample = record["sample"]
                if (sample.get("model_id"), sample.get("model_sha256")) != expected_binding:
                    raise MeasurementRunIntegrityError(
                        "evaluation sample differs from bound model"
                    )
        return calibration, evaluation

    def _read_ledger_unlocked(
        self,
        run_dir: Path,
        role: str,
        manifest: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        path = run_dir / LEDGER_FILES[role]
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise MeasurementRunIntegrityError(f"{role} ledger is unreadable") from exc
        if payload and not payload.endswith(b"\n"):
            raise MeasurementRunIntegrityError(
                f"{role} ledger is not complete JSON Lines"
            )
        records: list[dict[str, Any]] = []
        previous_head: str | None = None
        role_offset = 0 if role == "calibration" else EXPECTED_ROLE_COUNTS[
            "calibration"
        ]
        for local_index, raw_line in enumerate(payload.splitlines()):
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise MeasurementRunIntegrityError(
                    f"{role} ledger row {local_index} is unreadable"
                ) from exc
            if not isinstance(record, dict):
                raise MeasurementRunIntegrityError(
                    f"{role} ledger row {local_index} must be an object"
                )
            if _canonical_line(record).rstrip(b"\n") != raw_line:
                raise MeasurementRunIntegrityError(
                    f"{role} ledger row {local_index} is not canonical"
                )
            expected_sequence = role_offset + local_index
            if record.get("schema_version") != STATE_SCHEMA_VERSION:
                raise MeasurementRunIntegrityError("ledger schema changed")
            if record.get("record_type") != LEDGER_RECORD_TYPE:
                raise MeasurementRunIntegrityError("ledger record type changed")
            if record.get("capture_run_id") != manifest["capture_run_id"]:
                raise MeasurementRunIntegrityError("ledger capture run id changed")
            if record.get("ledger_role") != role:
                raise MeasurementRunIntegrityError("ledger role changed")
            if record.get("ledger_ordinal") != local_index:
                raise MeasurementRunIntegrityError("ledger ordinal is not exact")
            if record.get("sequence_index") != expected_sequence:
                raise MeasurementRunIntegrityError("ledger schedule order is not exact")
            if record.get("previous_record_sha256") != previous_head:
                raise MeasurementRunIntegrityError("ledger hash chain is broken")
            stored_record_sha = str(record.get("record_sha256") or "")
            if not LOWER_SHA256_PATTERN.fullmatch(stored_record_sha):
                raise MeasurementRunIntegrityError("ledger record SHA-256 is malformed")
            if not hmac.compare_digest(
                stored_record_sha,
                _schedule.canonical_sha256(_record_without_hash(record)),
            ):
                raise MeasurementRunIntegrityError("ledger record SHA-256 mismatch")
            row = dict(manifest["rows"][expected_sequence])
            if record.get("schedule_row_sha256") != _schedule.canonical_sha256(row):
                raise MeasurementRunIntegrityError(
                    "ledger schedule-row binding changed"
                )
            challenge_sha = str(record.get("challenge_token_sha256") or "")
            if not LOWER_SHA256_PATTERN.fullmatch(challenge_sha):
                raise MeasurementRunIntegrityError(
                    "ledger challenge SHA-256 is malformed"
                )
            sample = record.get("sample")
            if not isinstance(sample, Mapping):
                raise MeasurementRunIntegrityError("ledger sample must be an object")
            try:
                _assert_no_raw_media(sample, location="persisted sample")
            except MeasurementRunValidationError as exc:
                raise MeasurementRunIntegrityError(str(exc)) from exc
            if record.get("sample_sha256") != _schedule.canonical_sha256(sample):
                raise MeasurementRunIntegrityError("ledger sample SHA-256 mismatch")
            try:
                self._validate_sample(dict(sample), row, manifest, expected_sequence)
            except MeasurementRunValidationError as exc:
                raise MeasurementRunIntegrityError(str(exc)) from exc
            runner = self._runner_state(state)
            if runner.get("runtime_binding") is not None:
                proofs = runner["capture_contract_proofs"]
                if len(proofs) <= expected_sequence:
                    raise MeasurementRunIntegrityError(
                        "ledger record lacks its capture-contract proof"
                    )
                proof = proofs[expected_sequence]
                if (
                    record.get("attempt_disposition") != proof.get("disposition")
                    or record.get("frame_sha256") != proof.get("frame_sha256")
                    or record.get("capture_contract_evidence_sha256")
                    != _schedule.canonical_sha256(
                        proof.get("capture_contract_evidence")
                    )
                    or record.get("server_timing_evidence_sha256")
                    != _schedule.canonical_sha256(
                        proof.get("server_timing_evidence")
                    )
                    or record.get("sample_sha256")
                    != proof.get("observation_sha256")
                ):
                    raise MeasurementRunIntegrityError(
                        "ledger attempt sidecar differs from durable proof"
                    )
            records.append(record)
            previous_head = stored_record_sha
        if len(records) > EXPECTED_ROLE_COUNTS[role]:
            raise MeasurementRunIntegrityError(f"{role} ledger exceeds frozen count")
        return {
            "role": role,
            "records": records,
            "count": len(records),
            "head_sha256": previous_head,
            "file_sha256": hashlib.sha256(payload).hexdigest(),
        }

    def _validate_sample(
        self,
        sample: Mapping[str, Any],
        row: Mapping[str, Any],
        manifest: Mapping[str, Any],
        ordinal: int,
    ) -> dict[str, Any]:
        required = manifest["schedule_contract"]["required_sample_fields"]
        try:
            return _schedule._validate_sample(  # noqa: SLF001
                sample,
                row,
                index=ordinal,
                required_fields=required,
                optional_fields=manifest["schedule_contract"][
                    "optional_sensor_uncertainty_fields"
                ],
            )
        except _schedule.MeasurementScheduleError as exc:
            raise MeasurementRunValidationError(str(exc)) from exc

    @staticmethod
    def _validate_cross_sample_contract(samples: Sequence[Mapping[str, Any]]) -> None:
        if not samples:
            return
        sequence = [sample.get("sequence_index") for sample in samples]
        if sequence != list(range(len(samples))):
            raise MeasurementRunIntegrityError("persisted sample order is not exact")
        capture_times = [
            _finite_number(
                sample.get("frame_capture_monotonic_ms"),
                field="frame_capture_monotonic_ms",
            )
            for sample in samples
        ]
        if capture_times != sorted(capture_times):
            raise MeasurementRunIntegrityError("capture monotonic order changed")
        for field in (
            "capture_source",
            "viewport_width",
            "viewport_height",
            "device_pixel_ratio",
        ):
            if len({_schedule.canonical_json_bytes(sample.get(field)) for sample in samples}) != 1:
                raise MeasurementRunIntegrityError(
                    f"capture field {field} changed within run"
                )
        ratios = [
            _finite_number(sample.get("camera_width"), field="camera_width")
            / _finite_number(sample.get("camera_height"), field="camera_height")
            for sample in samples
        ]
        if max(ratios) - min(ratios) > 0.02:
            raise MeasurementRunIntegrityError(
                "camera aspect ratio changed by more than 0.02"
            )
        for role in ("calibration", "evaluation"):
            bindings = {
                (sample.get("model_id"), sample.get("model_sha256"))
                for sample in samples
                if sample.get("block_role") == role
            }
            if len(bindings) > 1:
                raise MeasurementRunIntegrityError(
                    f"{role} model binding changed within run"
                )

    def _build_sample(
        self,
        observation: Mapping[str, Any],
        row: Mapping[str, Any],
        *,
        manifest: Mapping[str, Any],
        ordinal: int,
    ) -> dict[str, Any]:
        protocol, _ = _schedule.load_frozen_protocol(self.protocol_path)
        sample_contract = dict(protocol["sample_contract"])
        allowed_fields = (
            set(manifest["schedule_contract"]["required_sample_fields"])
            | set(sample_contract["optional_sensor_uncertainty_fields"])
            | set(_schedule.SCHEDULE_ROW_FIELDS)
        )
        unexpected = sorted(set(observation) - allowed_fields)
        if unexpected:
            raise MeasurementRunValidationError(
                f"observation contains fields outside the frozen sample contract: {unexpected}"
            )
        sample = deepcopy(dict(observation))
        for field in _schedule.SCHEDULE_ROW_FIELDS:
            if field in sample:
                try:
                    same = (
                        _schedule.canonical_json_bytes(sample[field])
                        == _schedule.canonical_json_bytes(row[field])
                    )
                except _schedule.MeasurementScheduleError as exc:
                    raise MeasurementRunValidationError(str(exc)) from exc
                if not same:
                    raise MeasurementRunValidationError(
                        f"observation attempts to override server schedule field {field}"
                    )
            sample[field] = deepcopy(row[field])
        if sample.get("sequence_index") != ordinal:
            raise MeasurementRunValidationError("sample ordinal differs from challenge")
        try:
            # Reject NaN, infinity, non-JSON objects, and non-string mapping keys.
            _schedule.canonical_json_bytes(sample)
        except _schedule.MeasurementScheduleError as exc:
            raise MeasurementRunValidationError(str(exc)) from exc
        return sample

    @staticmethod
    def _validate_model_binding(
        state: dict[str, Any], validation: Mapping[str, Any], *, role: str
    ) -> None:
        observed = tuple(validation["model_binding"])
        if role == "calibration":
            runner = state.get("runner")
            runtime = (
                runner.get("runtime_binding")
                if isinstance(runner, Mapping)
                else None
            )
            if isinstance(runtime, Mapping):
                base = runtime.get("base_inference_bundle")
                expected = (
                    base.get("model_id"),
                    base.get("bundle_sha256"),
                ) if isinstance(base, Mapping) else (None, None)
                if observed != expected:
                    raise MeasurementRunValidationError(
                        "calibration observation differs from frozen base bundle"
                    )
            existing = state.get("calibration_model_binding")
            if existing is not None and (
                existing.get("model_id"), existing.get("model_sha256")
            ) != observed:
                raise MeasurementRunValidationError(
                    "calibration model binding changed within run"
                )
        else:
            bound = state.get("model_binding")
            if not isinstance(bound, Mapping):
                raise MeasurementRunStateError("evaluation model is not bound")
            if (bound.get("model_id"), bound.get("model_sha256")) != observed:
                raise MeasurementRunValidationError(
                    "evaluation observation differs from bound model"
                )

    def _append_ledger_unlocked(
        self,
        run_dir: Path,
        role: str,
        existing: Sequence[Mapping[str, Any]],
        record: Mapping[str, Any],
    ) -> None:
        payload = b"".join(_canonical_line(item) for item in existing)
        current_path = run_dir / LEDGER_FILES[role]
        if hashlib.sha256(payload).hexdigest() != _file_sha256(current_path):
            raise MeasurementRunIntegrityError(
                f"{role} ledger changed before atomic append"
            )
        _atomic_bytes(current_path, payload + _canonical_line(record))

    def _apply_consumed_record(
        self,
        run_dir: Path,
        state: dict[str, Any],
        manifest: Mapping[str, Any],
        calibration: Mapping[str, Any],
        evaluation: Mapping[str, Any],
        record: Mapping[str, Any],
        validation: Mapping[str, Any],
    ) -> None:
        active = state.get("active_challenge")
        if not isinstance(active, Mapping):
            raise MeasurementRunIntegrityError(
                "committed ledger record has no active challenge"
            )
        for field in (
            "challenge_id",
            "challenge_token_sha256",
            "schedule_row_sha256",
        ):
            if active.get(field) != record.get(field):
                raise MeasurementRunIntegrityError(
                    "committed ledger record differs from active challenge"
                )
        if active.get("ordinal") != record.get("sequence_index"):
            raise MeasurementRunIntegrityError(
                "committed ledger ordinal differs from active challenge"
            )
        role = str(record["ledger_role"])
        runner = self._runner_state(state)
        if runner.get("runtime_binding") is not None:
            ordinal = int(record["sequence_index"])
            proofs = runner["capture_contract_proofs"]
            if len(proofs) != ordinal + 1:
                raise MeasurementRunIntegrityError(
                    "committed record lacks its capture-contract proof"
                )
            proof = proofs[ordinal]
            if (
                proof.get("challenge_id") != record.get("challenge_id")
                or proof.get("observation_sha256") != record.get("sample_sha256")
                or proof.get("frame_sha256") != record.get("frame_sha256")
                or _schedule.canonical_sha256(
                    proof.get("capture_contract_evidence")
                )
                != record.get("capture_contract_evidence_sha256")
                or _schedule.canonical_sha256(
                    proof.get("server_timing_evidence")
                )
                != record.get("server_timing_evidence_sha256")
            ):
                raise MeasurementRunIntegrityError(
                    "committed record differs from prepared capture proof"
                )
            proof["status"] = "committed"
            proof["sample_sha256"] = record["sample_sha256"]
            proof["ledger_record_sha256"] = record["record_sha256"]
            proof["committed_at_utc"] = _utc_now()
            proof.pop("observation", None)
            spool = runner.get("frame_spool")
            if not isinstance(spool, dict) or (
                spool.get("status") != "active"
                or spool.get("challenge_id") != record.get("challenge_id")
                or spool.get("ordinal") != ordinal
                or spool.get("frame_sha256") != record.get("frame_sha256")
            ):
                raise MeasurementRunIntegrityError(
                    "committed attempt lacks its encrypted frame spool binding"
                )
            spool["status"] = "cleanup_pending_committed"
            spool["ledger_record_sha256"] = record["record_sha256"]
            spool["cleanup_requested_at_utc"] = _utc_now()
            if role == "calibration" and record["sample"].get(
                "prediction_success"
            ) is True:
                saved = next(
                    (
                        item
                        for item in runner["calibration_manifest_bindings"]
                        if item.get("ordinal") == ordinal
                    ),
                    None,
                )
                if not isinstance(saved, dict):
                    raise MeasurementRunIntegrityError(
                        "usable calibration manifest binding is missing"
                    )
                saved["status"] = "committed"
                saved["sample_sha256"] = record["sample_sha256"]
                saved["ledger_record_sha256"] = record["record_sha256"]
                saved["committed_at_utc"] = _utc_now()
            pending_write = runner.get("calibration_write")
            if (
                isinstance(pending_write, Mapping)
                and pending_write.get("ordinal") == ordinal
            ):
                runner["calibration_write"] = None
        if role == "calibration" and state.get("calibration_model_binding") is None:
            model_id, model_sha256 = validation["model_binding"]
            state["calibration_model_binding"] = {
                "model_id": model_id,
                "model_sha256": model_sha256,
            }
        state["active_challenge"] = None
        state["last_consumption"] = {
            "challenge_id": record["challenge_id"],
            "challenge_token_sha256": record["challenge_token_sha256"],
            "ordinal": record["sequence_index"],
            "ledger_role": role,
            "sample_sha256": record["sample_sha256"],
            "record_sha256": record["record_sha256"],
        }
        self._update_progress_and_ledgers(state, calibration, evaluation)
        total = int(state["progress"]["next_sequence_index"])
        if total == _schedule.EXPECTED_SAMPLE_COUNT:
            proofs = runner["capture_contract_proofs"]
            state["capture_contract_binding_verified"] = (
                len(proofs) == _schedule.EXPECTED_SAMPLE_COUNT
                and all(item.get("status") == "committed" for item in proofs)
            )
        if total < EXPECTED_ROLE_COUNTS["calibration"]:
            state["phase"] = PHASE_CALIBRATION_COLLECTING
        elif total == EXPECTED_ROLE_COUNTS["calibration"]:
            state["phase"] = PHASE_CALIBRATION_SEALED
        elif total < _schedule.EXPECTED_SAMPLE_COUNT:
            state["phase"] = PHASE_EVALUATION_COLLECTING
        elif total == _schedule.EXPECTED_SAMPLE_COUNT:
            self._seal_capture_unlocked(
                run_dir, state, manifest, calibration, evaluation
            )
        else:
            raise MeasurementRunIntegrityError("run exceeds 193 samples")

    def _reconcile_unlocked(
        self,
        run_dir: Path,
        state: dict[str, Any],
        manifest: Mapping[str, Any],
        calibration: Mapping[str, Any],
        evaluation: Mapping[str, Any],
    ) -> None:
        actual_total = int(calibration["count"]) + int(evaluation["count"])
        progress = state.get("progress")
        if not isinstance(progress, Mapping):
            raise MeasurementRunIntegrityError("run progress is invalid")
        state_total = progress.get("next_sequence_index")
        if not isinstance(state_total, int) or isinstance(state_total, bool):
            raise MeasurementRunIntegrityError("next sequence index is invalid")
        lag = actual_total - state_total
        if lag == 0:
            return
        if lag != 1:
            raise MeasurementRunIntegrityError(
                "state and immutable ledgers differ by more than one transaction"
            )
        all_records = calibration["records"] + evaluation["records"]
        record = all_records[-1]
        if record["sequence_index"] != state_total:
            raise MeasurementRunIntegrityError(
                "ledger-ahead transaction is not the exact next ordinal"
            )
        row = dict(manifest["rows"][state_total])
        validation = self._validate_sample(
            record["sample"], row, manifest, state_total
        )
        self._validate_model_binding(
            state, validation, role=str(record["ledger_role"])
        )
        self._apply_consumed_record(
            run_dir,
            state,
            manifest,
            calibration,
            evaluation,
            record,
            validation,
        )
        self._write_state(run_dir, state)

    def _update_progress_and_ledgers(
        self,
        state: dict[str, Any],
        calibration: Mapping[str, Any],
        evaluation: Mapping[str, Any],
    ) -> None:
        calibration_count = int(calibration["count"])
        evaluation_count = int(evaluation["count"])
        state["progress"] = {
            "next_sequence_index": calibration_count + evaluation_count,
            "calibration_count": calibration_count,
            "evaluation_count": evaluation_count,
        }
        state["ledgers"] = {}
        for role, summary in (
            ("calibration", calibration),
            ("evaluation", evaluation),
        ):
            sealed = (
                summary["file_sha256"]
                if summary["count"] == EXPECTED_ROLE_COUNTS[role]
                else None
            )
            state["ledgers"][role] = {
                "count": int(summary["count"]),
                "head_sha256": summary["head_sha256"],
                "file_sha256": summary["file_sha256"],
                "sealed_sha256": sealed,
            }

    def _seal_capture_unlocked(
        self,
        run_dir: Path,
        state: dict[str, Any],
        manifest: Mapping[str, Any],
        calibration: Mapping[str, Any],
        evaluation: Mapping[str, Any],
    ) -> None:
        samples = [
            deepcopy(record["sample"])
            for record in calibration["records"] + evaluation["records"]
        ]
        try:
            artifact = _schedule.build_capture_artifact(
                manifest,
                samples,
                evidence_class="physical_self_development",
                protocol_path=self.protocol_path,
            )
        except _schedule.MeasurementScheduleError as exc:
            raise MeasurementRunIntegrityError(str(exc)) from exc
        _atomic_json(run_dir / CAPTURE_ARTIFACT_FILENAME, artifact)
        records = calibration["records"] + evaluation["records"]
        persisted_proofs = self._runner_state(state)["capture_contract_proofs"]
        if self._runner_state(state).get("runtime_binding") is None:
            sidecar_proofs: list[Mapping[str, Any]] = [
                {} for _ in records
            ]
        else:
            if len(persisted_proofs) != len(records):
                raise MeasurementRunIntegrityError(
                    "sealed capture lacks exact attempt proofs"
                )
            sidecar_proofs = persisted_proofs
        sidecar: dict[str, Any] = {
            "schema_version": 1,
            "sidecar_type": ATTEMPT_SIDECAR_TYPE,
            "capture_run_id": state["capture_run_id"],
            "protocol_sha256": state["protocol_sha256"],
            "manifest_sha256": state["manifest_sha256"],
            "capture_artifact_sha256": artifact["artifact_sha256"],
            "entries": [
                {
                    "sequence_index": record["sequence_index"],
                    "ledger_role": record["ledger_role"],
                    "prediction_success": record["sample"]["prediction_success"],
                    "failure_code": (
                        "no_face_detected"
                        if record.get("attempt_disposition") == "no_face_detected"
                        else None
                    ),
                    "sample_sha256": record["sample_sha256"],
                    "ledger_record_sha256": record["record_sha256"],
                    "capture_contract_evidence_sha256": record.get(
                        "capture_contract_evidence_sha256"
                    ),
                    "server_timing_evidence": deepcopy(
                        proof.get("server_timing_evidence")
                    ),
                    "server_timing_evidence_sha256": record.get(
                        "server_timing_evidence_sha256"
                    ),
                    "frame_sha256": record.get("frame_sha256"),
                }
                for record, proof in zip(
                    records,
                    sidecar_proofs,
                    strict=True,
                )
            ],
            "measurement_claim_authorized": False,
            "physical_capture_claim_authorized": False,
        }
        sidecar["entries_sha256"] = _schedule.canonical_sha256(sidecar["entries"])
        sidecar["sidecar_sha256"] = _schedule.canonical_sha256(sidecar)
        _atomic_json(run_dir / ATTEMPT_SIDECAR_FILENAME, sidecar)
        state["capture_artifact"] = {
            "artifact_sha256": artifact["artifact_sha256"],
            "samples_sha256": artifact["samples_sha256"],
            "sample_count": len(samples),
            "attempt_sidecar_sha256": sidecar["sidecar_sha256"],
            "attempt_sidecar_entries_sha256": sidecar["entries_sha256"],
            "sealed_at_utc": _utc_now(),
            "verified_at_utc": None,
        }
        state["acquisition_artifact_verified"] = False
        state["phase"] = PHASE_CAPTURE_SEALED

    def _read_artifact_unlocked(
        self, run_dir: Path, state: Mapping[str, Any]
    ) -> dict[str, Any]:
        metadata = state.get("capture_artifact")
        if not isinstance(metadata, Mapping):
            raise MeasurementRunIntegrityError("sealed artifact metadata is missing")
        try:
            artifact = json.loads(
                (run_dir / CAPTURE_ARTIFACT_FILENAME).read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MeasurementRunIntegrityError("sealed artifact is unreadable") from exc
        if not isinstance(artifact, dict):
            raise MeasurementRunIntegrityError("sealed artifact must be an object")
        if artifact.get("artifact_sha256") != metadata.get("artifact_sha256"):
            raise MeasurementRunIntegrityError("sealed artifact binding changed")
        return artifact

    def _read_attempt_sidecar_unlocked(
        self,
        run_dir: Path,
        state: Mapping[str, Any],
        calibration: Mapping[str, Any],
        evaluation: Mapping[str, Any],
    ) -> dict[str, Any]:
        metadata = state.get("capture_artifact")
        if not isinstance(metadata, Mapping):
            raise MeasurementRunIntegrityError("sealed artifact metadata is missing")
        try:
            sidecar = json.loads(
                (run_dir / ATTEMPT_SIDECAR_FILENAME).read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MeasurementRunIntegrityError("attempt sidecar is unreadable") from exc
        if not isinstance(sidecar, dict):
            raise MeasurementRunIntegrityError("attempt sidecar must be an object")
        stored_sha = str(sidecar.get("sidecar_sha256") or "")
        core = deepcopy(sidecar)
        core.pop("sidecar_sha256", None)
        if not LOWER_SHA256_PATTERN.fullmatch(stored_sha) or not hmac.compare_digest(
            stored_sha, _schedule.canonical_sha256(core)
        ):
            raise MeasurementRunIntegrityError("attempt sidecar SHA-256 mismatch")
        entries = sidecar.get("entries")
        if not isinstance(entries, list) or len(entries) != _schedule.EXPECTED_SAMPLE_COUNT:
            raise MeasurementRunIntegrityError("attempt sidecar count changed")
        if sidecar.get("sidecar_type") != ATTEMPT_SIDECAR_TYPE:
            raise MeasurementRunIntegrityError("attempt sidecar type changed")
        if sidecar.get("capture_run_id") != state["capture_run_id"]:
            raise MeasurementRunIntegrityError("attempt sidecar run changed")
        if sidecar.get("capture_artifact_sha256") != metadata.get("artifact_sha256"):
            raise MeasurementRunIntegrityError(
                "attempt sidecar artifact binding changed"
            )
        if sidecar.get("entries_sha256") != _schedule.canonical_sha256(entries):
            raise MeasurementRunIntegrityError("attempt sidecar entries changed")
        if metadata.get("attempt_sidecar_sha256") != stored_sha or metadata.get(
            "attempt_sidecar_entries_sha256"
        ) != sidecar.get("entries_sha256"):
            raise MeasurementRunIntegrityError("attempt sidecar state binding changed")
        records = calibration["records"] + evaluation["records"]
        persisted_proofs = self._runner_state(state)["capture_contract_proofs"]
        runtime_bound = self._runner_state(state).get("runtime_binding") is not None
        for index, (entry, record) in enumerate(zip(entries, records, strict=True)):
            proof = persisted_proofs[index] if runtime_bound else {}
            expected = {
                "sequence_index": index,
                "ledger_role": record["ledger_role"],
                "prediction_success": record["sample"]["prediction_success"],
                "failure_code": (
                    "no_face_detected"
                    if record.get("attempt_disposition") == "no_face_detected"
                    else None
                ),
                "sample_sha256": record["sample_sha256"],
                "ledger_record_sha256": record["record_sha256"],
                "capture_contract_evidence_sha256": record.get(
                    "capture_contract_evidence_sha256"
                ),
                "server_timing_evidence": deepcopy(
                    proof.get("server_timing_evidence")
                ),
                "server_timing_evidence_sha256": record.get(
                    "server_timing_evidence_sha256"
                ),
                "frame_sha256": record.get("frame_sha256"),
            }
            if entry != expected:
                raise MeasurementRunIntegrityError(
                    f"attempt sidecar entry {index} differs from ledger"
                )
        return sidecar

    def _validate_state_against_persistence(
        self,
        run_dir: Path,
        state: Mapping[str, Any],
        manifest: Mapping[str, Any],
        calibration: Mapping[str, Any],
        evaluation: Mapping[str, Any],
    ) -> None:
        runner = self._runner_state(state)
        runtime = runner.get("runtime_binding")
        if runtime is not None:
            if not isinstance(runtime, Mapping):
                raise MeasurementRunIntegrityError("runtime binding is invalid")
            try:
                normalized_contract = normalize_capture_contract(
                    runtime.get("capture_contract")
                )
            except ValueError as exc:
                raise MeasurementRunIntegrityError(str(exc)) from exc
            if runtime.get("capture_contract") != normalized_contract:
                raise MeasurementRunIntegrityError(
                    "runtime capture contract is not normalized"
                )
            if runtime.get("capture_contract_sha256") != _schedule.canonical_sha256(
                normalized_contract
            ):
                raise MeasurementRunIntegrityError(
                    "runtime capture contract SHA-256 changed"
                )
            base = runtime.get("base_inference_bundle")
            if not isinstance(base, Mapping):
                raise MeasurementRunIntegrityError(
                    "runtime base inference bundle binding is invalid"
                )
            for field in (
                "bundle_sha256",
                "model_sha256",
                "checkpoint_sha256",
            ):
                if not LOWER_SHA256_PATTERN.fullmatch(str(base.get(field) or "")):
                    raise MeasurementRunIntegrityError(
                        f"runtime base inference bundle {field} is malformed"
                    )
        elif any(
            runner.get(field)
            for field in (
                "calibration_write",
                "calibration_manifest_bindings",
                "capture_contract_proofs",
                "frame_spool",
                "inference_intent",
                "base_bundle_checks",
                "training_artifact_intent",
                "trained_artifact",
                "calibration_image_purge",
            )
        ):
            raise MeasurementRunIntegrityError(
                "runner evidence exists without a runtime binding"
            )
        expected_counts = manifest["expected_counts"]
        if state.get("expected_counts") != expected_counts:
            raise MeasurementRunIntegrityError("state expected counts changed")
        actual_progress = {
            "next_sequence_index": calibration["count"] + evaluation["count"],
            "calibration_count": calibration["count"],
            "evaluation_count": evaluation["count"],
        }
        if state.get("progress") != actual_progress:
            raise MeasurementRunIntegrityError("state progress differs from ledgers")
        for role, summary in (
            ("calibration", calibration),
            ("evaluation", evaluation),
        ):
            observed = state.get("ledgers", {}).get(role)
            expected = {
                "count": summary["count"],
                "head_sha256": summary["head_sha256"],
                "file_sha256": summary["file_sha256"],
                "sealed_sha256": (
                    summary["file_sha256"]
                    if summary["count"] == EXPECTED_ROLE_COUNTS[role]
                    else None
                ),
            }
            if observed != expected:
                raise MeasurementRunIntegrityError(
                    f"state {role} ledger summary changed"
                )
        total = actual_progress["next_sequence_index"]
        proofs = runner["capture_contract_proofs"]
        if runtime is not None and len(proofs) not in {total, total + 1}:
            raise MeasurementRunIntegrityError(
                "capture-contract proof count differs from run progress"
            )
        for index, proof in enumerate(proofs):
            if not isinstance(proof, Mapping) or proof.get("ordinal") != index:
                raise MeasurementRunIntegrityError(
                    "capture-contract proof order is not exact"
                )
            try:
                timing = self._validated_server_timing_evidence(
                    proof.get("server_timing_evidence")
                )
            except (MeasurementRunValidationError, MeasurementRunIntegrityError) as exc:
                raise MeasurementRunIntegrityError(str(exc)) from exc
            if timing != proof.get("server_timing_evidence"):
                raise MeasurementRunIntegrityError(
                    "server timing evidence is not normalized"
                )
            expected_status = "committed" if index < total else "prepared"
            if proof.get("status") != expected_status:
                raise MeasurementRunIntegrityError(
                    "capture-contract proof status differs from progress"
                )
            if expected_status == "committed" and proof.get("observation") is not None:
                raise MeasurementRunIntegrityError(
                    "committed proof retained duplicate observation"
                )
            if expected_status == "prepared" and not isinstance(
                proof.get("observation"), Mapping
            ):
                raise MeasurementRunIntegrityError(
                    "prepared proof lacks its replay observation"
                )
        calibration_write = runner.get("calibration_write")
        if calibration_write is not None:
            active_for_write = state.get("active_challenge")
            if not isinstance(calibration_write, Mapping) or not isinstance(
                active_for_write, Mapping
            ):
                raise MeasurementRunIntegrityError(
                    "calibration write lacks its active challenge"
                )
            if (
                calibration_write.get("challenge_id")
                != active_for_write.get("challenge_id")
                or calibration_write.get("ordinal") != total
                or calibration_write.get("frame_sha256")
                != (runner.get("frame_spool") or {}).get("frame_sha256")
            ):
                raise MeasurementRunIntegrityError(
                    "calibration write active-frame binding changed"
                )
            write_status = calibration_write.get("status")
            if write_status not in {
                "prepared",
                "saved",
                "no_face_reclassification_pending",
            }:
                raise MeasurementRunIntegrityError(
                    "calibration write status is invalid"
                )
            if write_status == "no_face_reclassification_pending":
                sample = calibration_write.get("no_face_observation")
                paths = calibration_write.get("cleanup_relative_paths")
                if (
                    len(proofs) != total + 1
                    or proofs[total].get("disposition") != "success"
                    or not isinstance(sample, Mapping)
                    or _schedule.canonical_sha256(sample)
                    != calibration_write.get("no_face_observation_sha256")
                    or not isinstance(paths, Mapping)
                    or set(paths) != {"raw", "crop", "normalized_face"}
                    or not LOWER_SHA256_PATTERN.fullmatch(
                        str(calibration_write.get("manifest_record_sha256") or "")
                    )
                ):
                    raise MeasurementRunIntegrityError(
                        "no-face reclassification intent changed"
                    )
        bindings = runner["calibration_manifest_bindings"]
        if len(bindings) > EXPECTED_ROLE_COUNTS["calibration"]:
            raise MeasurementRunIntegrityError(
                "dedicated calibration manifest exceeds 65 usable rows"
            )
        if [item.get("sample_index") for item in bindings] != list(
            range(len(bindings))
        ):
            raise MeasurementRunIntegrityError(
                "dedicated calibration manifest indices are not exact"
            )
        if len({item.get("ordinal") for item in bindings}) != len(bindings):
            raise MeasurementRunIntegrityError(
                "dedicated calibration manifest ordinal binding is duplicated"
            )
        for binding in bindings:
            ordinal = binding.get("ordinal")
            if not isinstance(ordinal, int) or not 0 <= ordinal < EXPECTED_ROLE_COUNTS[
                "calibration"
            ]:
                raise MeasurementRunIntegrityError(
                    "dedicated calibration manifest ordinal is invalid"
                )
            expected_status = "committed" if ordinal < total else "saved"
            if binding.get("status") != expected_status:
                raise MeasurementRunIntegrityError(
                    "dedicated calibration manifest status differs from progress"
                )
        intent = runner.get("training_artifact_intent")
        if intent is not None:
            if not isinstance(intent, Mapping) or intent.get("status") not in {
                "reserved",
                "bound",
                "cleanup_verified",
            }:
                raise MeasurementRunIntegrityError(
                    "training artifact intent is invalid"
                )
            relative = Path(str(intent.get("artifact_relative_path") or ""))
            artifact_path = (self.root / relative).resolve()
            expected_parent = (self.root / "examples" / "models").resolve()
            expected_model_id = "wgmc_" + "".join(
                character.lower() if character.isalnum() else "_"
                for character in str(state["capture_run_id"])
            )
            runtime_binding = runner.get("runtime_binding")
            if (
                relative.is_absolute()
                or not relative.parts
                or artifact_path.parent != expected_parent
                or artifact_path.name != f"{intent.get('model_id')}.json"
                or intent.get("model_id") != expected_model_id
                or intent.get("path_initially_absent_verified") is not True
                or not isinstance(runtime_binding, Mapping)
                or intent.get("calibration_session_id")
                != runtime_binding.get("calibration_session_id")
            ):
                raise MeasurementRunIntegrityError(
                    "training artifact intent binding changed"
                )
            if intent.get("status") == "cleanup_verified" and artifact_path.exists():
                raise MeasurementRunIntegrityError(
                    "cleaned training artifact path exists"
                )
        if runner.get("trained_artifact") is not None:
            model_binding = state.get("model_binding")
            trained = runner["trained_artifact"]
            if not isinstance(model_binding, Mapping) or (
                model_binding.get("model_id"), model_binding.get("model_sha256")
            ) != (trained.get("model_id"), trained.get("model_sha256")):
                raise MeasurementRunIntegrityError(
                    "trained artifact differs from model binding"
                )
            if not isinstance(intent, Mapping) or (
                intent.get("status") != "bound"
                or intent.get("model_id") != trained.get("model_id")
                or intent.get("artifact_relative_path")
                != trained.get("artifact_relative_path")
                or intent.get("model_sha256") != trained.get("model_sha256")
                or intent.get("training_provenance_sha256")
                != trained.get("training_provenance_sha256")
            ):
                raise MeasurementRunIntegrityError(
                    "bound artifact differs from its training intent"
                )
            if state.get("phase") not in {PHASE_ABORTED, PHASE_FAILED_INTEGRITY}:
                self._verify_bound_model_file(state)
        active = state.get("active_challenge")
        if active is not None:
            if not isinstance(active, Mapping):
                raise MeasurementRunIntegrityError("active challenge is invalid")
            if active.get("ordinal") != total:
                raise MeasurementRunIntegrityError("active challenge ordinal changed")
            if total >= _schedule.EXPECTED_SAMPLE_COUNT:
                raise MeasurementRunIntegrityError("challenge exists after final sample")
            row = manifest["rows"][total]
            checks = {
                "ledger_role": row["block_role"],
                "schedule_row_sha256": _schedule.canonical_sha256(row),
            }
            for field, expected in checks.items():
                if active.get(field) != expected:
                    raise MeasurementRunIntegrityError(
                        f"active challenge {field} changed"
                    )
        spool = runner.get("frame_spool")
        if spool is not None:
            if not isinstance(spool, Mapping):
                raise MeasurementRunIntegrityError("frame spool state is invalid")
            spool_status = spool.get("status")
            if spool_status in {"active", "cleanup_pending_uncommitted"}:
                if not isinstance(active, Mapping) or (
                    spool.get("challenge_id") != active.get("challenge_id")
                    or spool.get("ordinal") != active.get("ordinal")
                ):
                    raise MeasurementRunIntegrityError(
                        "frame spool active-challenge binding changed"
                    )
            elif spool_status == "cleanup_pending_committed":
                last = state.get("last_consumption")
                if not isinstance(last, Mapping) or (
                    spool.get("challenge_id") != last.get("challenge_id")
                    or spool.get("ordinal") != last.get("ordinal")
                    or spool.get("ledger_record_sha256")
                    != last.get("record_sha256")
                ):
                    raise MeasurementRunIntegrityError(
                        "committed frame spool cleanup binding changed"
                    )
            else:
                raise MeasurementRunIntegrityError("frame spool status is invalid")
            for field in ("frame_sha256", "ciphertext_sha256"):
                if not LOWER_SHA256_PATTERN.fullmatch(str(spool.get(field) or "")):
                    raise MeasurementRunIntegrityError(
                        f"frame spool {field} is malformed"
                    )
            self._frame_spool_path(run_dir, spool)
        inference_intent = runner.get("inference_intent")
        if inference_intent is not None:
            if (
                not isinstance(inference_intent, Mapping)
                or inference_intent.get("status") != "inference_in_progress"
                or not isinstance(active, Mapping)
                or not isinstance(spool, Mapping)
                or spool.get("status") != "active"
                or inference_intent.get("challenge_id")
                != active.get("challenge_id")
                or inference_intent.get("ordinal") != total
                or inference_intent.get("frame_sha256")
                != spool.get("frame_sha256")
                or len(proofs) != total
            ):
                raise MeasurementRunIntegrityError(
                    "inference intent active-frame binding changed"
                )
            for field in (
                "frame_sha256",
                "model_sha256",
                "capture_contract_evidence_sha256",
                "server_receive_context_sha256",
            ):
                if not LOWER_SHA256_PATTERN.fullmatch(
                    str(inference_intent.get(field) or "")
                ):
                    raise MeasurementRunIntegrityError(
                        f"inference intent {field} is malformed"
                    )
            intent_core = deepcopy(dict(inference_intent))
            stored_intent_sha = intent_core.pop("intent_sha256", None)
            intent_core.pop("started_at_utc", None)
            if stored_intent_sha != _schedule.canonical_sha256(intent_core):
                raise MeasurementRunIntegrityError(
                    "inference intent hash binding changed"
                )
        if isinstance(active, Mapping):
            if not LOWER_SHA256_PATTERN.fullmatch(
                str(active.get("challenge_token_sha256") or "")
            ):
                raise MeasurementRunIntegrityError(
                    "active challenge token hash is malformed"
                )
        if total == 0 and state.get("last_consumption") is not None:
            raise MeasurementRunIntegrityError("last consumption exists on empty run")
        if total:
            records = calibration["records"] + evaluation["records"]
            last_record = records[-1]
            last = state.get("last_consumption")
            if not isinstance(last, Mapping):
                raise MeasurementRunIntegrityError("last consumption is missing")
            checks = {
                "challenge_id": last_record["challenge_id"],
                "challenge_token_sha256": last_record["challenge_token_sha256"],
                "ordinal": last_record["sequence_index"],
                "ledger_role": last_record["ledger_role"],
                "sample_sha256": last_record["sample_sha256"],
                "record_sha256": last_record["record_sha256"],
            }
            for field, expected in checks.items():
                if last.get(field) != expected:
                    raise MeasurementRunIntegrityError(
                        f"last consumption {field} changed"
                    )

        if state["phase"] not in {PHASE_ABORTED, PHASE_FAILED_INTEGRITY}:
            if total < EXPECTED_ROLE_COUNTS["calibration"]:
                allowed = {PHASE_SCHEDULED, PHASE_CALIBRATION_COLLECTING}
            elif total == EXPECTED_ROLE_COUNTS["calibration"]:
                allowed = (
                    {PHASE_CALIBRATION_SEALED}
                    if state.get("model_binding") is None
                    else {PHASE_MODEL_BOUND, PHASE_EVALUATION_COLLECTING}
                )
            elif total < _schedule.EXPECTED_SAMPLE_COUNT:
                allowed = {PHASE_EVALUATION_COLLECTING}
            else:
                allowed = {PHASE_CAPTURE_SEALED, PHASE_ARTIFACT_VERIFIED}
            if state["phase"] not in allowed:
                raise MeasurementRunIntegrityError(
                    "run phase differs from immutable progress"
                )
        if evaluation["count"] and state.get("model_binding") is None:
            raise MeasurementRunIntegrityError("evaluation lacks model binding")
        if total == _schedule.EXPECTED_SAMPLE_COUNT:
            artifact = self._read_artifact_unlocked(run_dir, state)
            summary = _schedule.verify_capture_artifact(
                artifact, protocol_path=self.protocol_path
            )
            if summary["sample_count"] != _schedule.EXPECTED_SAMPLE_COUNT:
                raise MeasurementRunIntegrityError("sealed artifact count changed")
            self._read_attempt_sidecar_unlocked(
                run_dir, state, calibration, evaluation
            )
            expected_verified = state["phase"] == PHASE_ARTIFACT_VERIFIED
            if state.get("acquisition_artifact_verified") is not expected_verified:
                raise MeasurementRunIntegrityError(
                    "capture contract verification state changed"
                )
        elif state.get("capture_artifact") is not None:
            raise MeasurementRunIntegrityError("artifact exists before 193 samples")
        expected_capture_binding = (
            runtime is not None
            and state.get("phase") not in {PHASE_ABORTED, PHASE_FAILED_INTEGRITY}
            and total == _schedule.EXPECTED_SAMPLE_COUNT
            and len(proofs) == _schedule.EXPECTED_SAMPLE_COUNT
            and all(item.get("status") == "committed" for item in proofs)
        )
        if state.get("capture_contract_binding_verified") is not expected_capture_binding:
            raise MeasurementRunIntegrityError(
                "capture-contract binding verification state changed"
            )

    def _mark_failed_unlocked(
        self,
        run_dir: Path,
        state: dict[str, Any],
        *,
        code: str,
        detail: str,
    ) -> None:
        if state.get("phase") == PHASE_FAILED_INTEGRITY:
            return
        runner = self._runner_state(state)
        spool_absent = False
        try:
            spool_dir = (run_dir / "spool").resolve()
            if spool_dir.parent != run_dir:
                raise MeasurementRunIntegrityError("frame spool root is unsafe")
            children = list(spool_dir.iterdir()) if spool_dir.is_dir() else []
            if any(
                not child.is_file()
                or child.parent.resolve() != spool_dir
                or not SPOOL_FILENAME_PATTERN.fullmatch(child.name)
                for child in children
            ):
                raise MeasurementRunIntegrityError(
                    "frame spool contains an unexpected failure-cleanup path"
                )
            for child in children:
                child.unlink()
            spool_absent = not (spool_dir.is_dir() and any(spool_dir.iterdir()))
            if spool_absent:
                runner["frame_spool"] = None
        except Exception:
            spool_absent = False

        image_absent = True
        cleaned_session_id: str | None = None
        runtime = runner.get("runtime_binding")
        if isinstance(runtime, Mapping):
            session_id = str(runtime.get("calibration_session_id") or "")
            sessions_root = (self.root / "data" / "sessions").resolve()
            session_dir = (sessions_root / session_id).resolve()
            metadata_path = session_dir / "session.json"
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if (
                    session_dir.parent != sessions_root
                    or metadata.get("session_id") != session_id
                    or metadata.get("capture_run_id") != state["capture_run_id"]
                    or metadata.get("capture_source")
                    != "direct-webcam-self-development"
                ):
                    raise MeasurementRunIntegrityError(
                        "failure cleanup session binding is invalid"
                    )
                for name in DEDICATED_IMAGE_DIRECTORIES:
                    target = (session_dir / name).resolve()
                    if target.parent != session_dir:
                        raise MeasurementRunIntegrityError(
                            "failure cleanup image path is unsafe"
                        )
                    if target.is_dir():
                        shutil.rmtree(target)
                image_absent = all(
                    not (session_dir / name).exists()
                    for name in DEDICATED_IMAGE_DIRECTORIES
                )
                cleaned_session_id = session_id
            except Exception:
                image_absent = False
        intent = runner.get("training_artifact_intent")
        model_absent = False
        if isinstance(intent, dict) and runner.get("trained_artifact") is None:
            relative = Path(str(intent.get("artifact_relative_path") or ""))
            model_path = (self.root / relative).resolve()
            expected_parent = (self.root / "examples" / "models").resolve()
            try:
                if (
                    relative.is_absolute()
                    or not relative.parts
                    or model_path.parent != expected_parent
                    or model_path.name != f"{intent.get('model_id')}.json"
                    or intent.get("path_initially_absent_verified") is not True
                ):
                    raise MeasurementRunIntegrityError(
                        "failure cleanup training-artifact intent is invalid"
                    )
                if model_path.exists():
                    if not model_path.is_file():
                        raise MeasurementRunIntegrityError(
                            "reserved training artifact is not a file"
                        )
                    model_path.unlink()
                model_absent = not model_path.exists()
                if model_absent:
                    intent["status"] = "cleanup_verified"
                    intent["cleanup_verified_at_utc"] = _utc_now()
            except Exception:
                model_absent = False
        else:
            model_name = "wgmc_" + "".join(
                character.lower() if character.isalnum() else "_"
                for character in str(state["capture_run_id"])
            )
            model_path = (
                self.root / "examples" / "models" / f"{model_name}.json"
            ).resolve()
            model_absent = (
                model_path.parent
                == (self.root / "examples" / "models").resolve()
                and not model_path.exists()
            )
        state["phase"] = PHASE_FAILED_INTEGRITY
        if spool_absent:
            state["active_challenge"] = None
            runner["calibration_write"] = None
            runner["inference_intent"] = None
        state["acquisition_artifact_verified"] = False
        state["capture_contract_binding_verified"] = False
        state["failure"] = {
            "code": code,
            "detail": str(detail)[:512],
            "detail_sha256": _token_sha256(str(detail)),
            "at_utc": _utc_now(),
            "cleanup": {
                "cleanup_verified": spool_absent and image_absent and model_absent,
                "spool_absence_verified": spool_absent,
                "image_directories_absent": image_absent,
                "calibration_session_id": cleaned_session_id,
                "model_absence_verified": model_absent,
            },
        }
        self._write_state(run_dir, state)

    @staticmethod
    def _require_active_run(
        state: Mapping[str, Any], *, allow_verified: bool = False
    ) -> None:
        terminal = set(TERMINAL_PHASES)
        if allow_verified:
            terminal.discard(PHASE_ARTIFACT_VERIFIED)
        if state.get("phase") in terminal:
            raise MeasurementRunStateError(
                f"run is terminal: {state.get('phase')}"
            )

    @staticmethod
    def _receipt_from_last(
        state: Mapping[str, Any], *, idempotent: bool
    ) -> dict[str, Any]:
        last = state.get("last_consumption")
        if not isinstance(last, Mapping):
            raise MeasurementRunIntegrityError("consumption receipt is unavailable")
        return {
            "ok": True,
            "status": "already_committed" if idempotent else "committed",
            "idempotent": idempotent,
            "capture_run_id": state["capture_run_id"],
            "phase": state["phase"],
            "ordinal": last["ordinal"],
            "block_role": last["ledger_role"],
            "sample_sha256": last["sample_sha256"],
            "record_sha256": last["record_sha256"],
            "progress": deepcopy(state["progress"]),
            "acquisition_artifact_verified": state[
                "acquisition_artifact_verified"
            ],
            "capture_contract_binding_verified": state[
                "capture_contract_binding_verified"
            ],
            "measurement_claim_authorized": False,
            "physical_capture_claim_authorized": False,
        }

    @staticmethod
    def _public_state(state: Mapping[str, Any]) -> dict[str, Any]:
        ledgers = deepcopy(state.get("ledgers", {}))
        active = state.get("active_challenge")
        runner = state.get("runner") if isinstance(state.get("runner"), Mapping) else {}
        proofs = runner.get("capture_contract_proofs", [])
        manifest_bindings = runner.get("calibration_manifest_bindings", [])
        return {
            "ok": True,
            "capture_run_id": state["capture_run_id"],
            "phase": state["phase"],
            "created_at_utc": state["created_at_utc"],
            "updated_at_utc": state["updated_at_utc"],
            "protocol_sha256": state["protocol_sha256"],
            "manifest_sha256": state["manifest_sha256"],
            "manifest_rows_sha256": state["manifest_rows_sha256"],
            "expected_counts": deepcopy(state["expected_counts"]),
            "progress": deepcopy(state["progress"]),
            "ledgers": ledgers,
            "challenge_outstanding": active is not None,
            "challenge_ordinal": active.get("ordinal") if isinstance(active, Mapping) else None,
            "calibration_model_binding": deepcopy(
                state.get("calibration_model_binding")
            ),
            "model_binding": deepcopy(state.get("model_binding")),
            "runner": {
                "runtime_binding": deepcopy(runner.get("runtime_binding")),
                "calibration_write_pending": runner.get("calibration_write")
                is not None,
                "calibration_usable_manifest_count": len(manifest_bindings),
                "capture_contract_proof_count": len(proofs),
                "base_bundle_checks": deepcopy(
                    runner.get("base_bundle_checks", [])
                ),
                "prepared_observation_pending": bool(
                    proofs
                    and isinstance(active, Mapping)
                    and len(proofs) == int(active.get("ordinal", -1)) + 1
                    and proofs[-1].get("status") == "prepared"
                ),
                "inference_in_progress": runner.get("inference_intent")
                is not None,
                "trained_artifact": deepcopy(runner.get("trained_artifact")),
                "training_artifact_intent": deepcopy(
                    runner.get("training_artifact_intent")
                ),
                "calibration_image_purge": deepcopy(
                    runner.get("calibration_image_purge")
                ),
            },
            "capture_artifact": deepcopy(state.get("capture_artifact")),
            "failure": deepcopy(state.get("failure")),
            "abort": deepcopy(state.get("abort")),
            "acquisition_artifact_verified": state[
                "acquisition_artifact_verified"
            ],
            "capture_contract_binding_verified": state[
                "capture_contract_binding_verified"
            ],
            "measurement_claim_authorized": False,
            "physical_capture_claim_authorized": False,
        }


__all__ = [
    "ATTEMPT_SIDECAR_FILENAME",
    "CAPTURE_ARTIFACT_FILENAME",
    "CALIBRATION_LEDGER_FILENAME",
    "EVALUATION_LEDGER_FILENAME",
    "MeasurementRunAuthenticationError",
    "MeasurementRunChallengeError",
    "MeasurementRunError",
    "MeasurementRunIntegrityError",
    "MeasurementRunStateError",
    "MeasurementRunStore",
    "MeasurementRunValidationError",
    "PHASE_ABORTED",
    "PHASE_ARTIFACT_VERIFIED",
    "PHASE_CALIBRATION_SEALED",
    "PHASE_CAPTURE_SEALED",
    "PHASE_FAILED_INTEGRITY",
    "PHASE_MODEL_BOUND",
    "STATE_FILENAME",
    "STORE_RELATIVE_PATH",
]

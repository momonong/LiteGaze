"""Trusted adapter for the frozen 193-attempt webcam measurement run.

The browser supplies only opaque tokens, webcam bytes, and a normalized
capture contract.  Targets, model identities, timing, outcomes, and persisted
observations are reconstructed server-side.  Participant-study state is never
read or mutated here.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from . import measurement_schedule as _schedule
from .capture_contract import (
    compare_capture_contracts,
    normalize_capture_contract,
    validate_transport_frame,
)
from .measurement_run_store import (
    DEDICATED_IMAGE_DIRECTORIES,
    MeasurementRunIntegrityError,
    MeasurementRunStateError,
    MeasurementRunStore,
    MeasurementRunValidationError,
    PHASE_CALIBRATION_SEALED,
    PHASE_MODEL_BOUND,
)
from .measurement_preflight import normalize_base_bundle_identity


CAPTURE_SOURCE = "direct-webcam-self-development"
BASE_INFERENCE_SELECTOR = "before"
CALIBRATION_PROTOCOL = "motion-diverse-v1"
CALIBRATION_LABEL_AUTHORITY = "server_frozen_measurement_ceiling_v1"
MAX_FRAME_BYTES = 10 * 1024 * 1024

MOTION_BLOCKS = {
    "calibration_neutral": "neutral",
    "calibration_left": "left",
    "calibration_right": "right",
    "calibration_near": "near",
    "calibration_far": "far",
}


class MeasurementRunnerError(RuntimeError):
    """A trusted-adapter operation could not complete."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_bytes(path: Path, payload: bytes) -> None:
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


def _default_decode_frame(image_data: str) -> tuple[bytes, int, int]:
    if not isinstance(image_data, str) or not image_data:
        raise MeasurementRunnerError("missing image_data")
    encoded = image_data.split(",", 1)[1] if "," in image_data else image_data
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, TypeError, ValueError) as exc:
        raise MeasurementRunnerError("invalid image_data") from exc
    if not raw or len(raw) > MAX_FRAME_BYTES:
        raise MeasurementRunnerError("image payload size is invalid")
    try:
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as exc:  # pragma: no cover - environment-specific backend
        raise MeasurementRunnerError("frame decoder is unavailable") from exc
    if image is None:
        raise MeasurementRunnerError("cannot decode image")
    return raw, int(image.shape[1]), int(image.shape[0])


def _default_create_session(root: Path, **kwargs: Any) -> Mapping[str, Any]:
    from .sample_store import create_session

    return create_session(root, **kwargs)


def _default_find_session(root: Path, capture_run_id: str) -> Mapping[str, Any] | None:
    """Recover the unique dedicated sample namespace after a create crash."""

    sessions_root = (root / "data" / "sessions").resolve()
    if not sessions_root.is_dir():
        return None
    matches: list[dict[str, Any]] = []
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
            and metadata.get("capture_source") == CAPTURE_SOURCE
            and metadata.get("participant_id") == f"wgmc_{capture_run_id}"
            and metadata.get("session_id") == child.name
        ):
            matches.append(dict(metadata))
    if len(matches) > 1:
        raise MeasurementRunIntegrityError(
            "multiple dedicated sessions exist for one measurement run"
        )
    return matches[0] if matches else None


def _default_save_sample(root: Path, payload: dict[str, Any]) -> tuple[dict, int]:
    from .sample_store import save_sample

    return save_sample(root, payload)


def _default_train(root: Path, payload: dict[str, Any]) -> tuple[dict, int]:
    from .training import train_placeholder

    return train_placeholder(root, payload)


def _default_predict(root: Path, payload: dict[str, Any]) -> tuple[dict, int]:
    from .inference import predict

    return predict(root, payload)


def _default_purge(root: Path, session_id: str) -> Mapping[str, Any]:
    from .sample_store import purge_session_images

    return purge_session_images(root, session_id)


def _default_model_path(root: Path, model_name: str) -> Path:
    from .model_registry import model_path

    return model_path(root, model_name)


def _default_base_bundle(code_root: Path) -> Mapping[str, Any]:
    from .base_inference_bundle import build_base_inference_bundle

    return build_base_inference_bundle(repository_root=code_root)


def _default_verify_base_bundle(
    bundle: Mapping[str, Any], code_root: Path
) -> Mapping[str, Any]:
    from .base_inference_bundle import verify_base_inference_bundle

    return verify_base_inference_bundle(bundle, repository_root=code_root)


def _default_reset_inference_cache() -> None:
    from . import inference
    from core.unigaze_personalization.model import load_unigaze_b16

    # The base identity is meaningless if a pre-run in-memory model survives.
    with inference._model_cache_lock:  # noqa: SLF001
        inference._model_cache.clear()  # noqa: SLF001
        load_unigaze_b16.cache_clear()


class MeasurementRunner:
    """Crash-recoverable CPU adapter over :class:`MeasurementRunStore`."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        code_root: str | Path | None = None,
        store: MeasurementRunStore | None = None,
        create_session_fn: Callable[..., Mapping[str, Any]] | None = None,
        find_session_fn: Callable[[Path, str], Mapping[str, Any] | None]
        | None = None,
        save_sample_fn: Callable[[Path, dict[str, Any]], tuple[dict, int]] | None = None,
        train_fn: Callable[[Path, dict[str, Any]], tuple[dict, int]] | None = None,
        predict_fn: Callable[[Path, dict[str, Any]], tuple[dict, int]] | None = None,
        purge_fn: Callable[[Path, str], Mapping[str, Any]] | None = None,
        model_path_fn: Callable[[Path, str], Path] | None = None,
        base_bundle_fn: Callable[[Path], Mapping[str, Any]] | None = None,
        verify_base_bundle_fn: Callable[
            [Mapping[str, Any], Path], Mapping[str, Any]
        ]
        | None = None,
        reset_inference_cache_fn: Callable[[], None] | None = None,
        decode_frame_fn: Callable[[str], tuple[bytes, int, int]] | None = None,
        monotonic_ms_fn: Callable[[], float] | None = None,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.code_root = Path(
            code_root or Path(__file__).resolve().parents[2]
        ).resolve()
        protocol_path = self.code_root / _schedule.PROTOCOL_RELATIVE_PATH
        self.store = store or MeasurementRunStore(
            self.data_root, protocol_path=protocol_path
        )
        self._create_session = create_session_fn or _default_create_session
        self._find_session = find_session_fn or _default_find_session
        self._save_sample = save_sample_fn or _default_save_sample
        self._train = train_fn or _default_train
        self._predict = predict_fn or _default_predict
        self._purge = purge_fn or _default_purge
        self._model_path = model_path_fn or _default_model_path
        self._base_bundle = base_bundle_fn or _default_base_bundle
        self._verify_base_bundle = (
            verify_base_bundle_fn or _default_verify_base_bundle
        )
        self._reset_inference_cache = (
            reset_inference_cache_fn or _default_reset_inference_cache
        )
        self._decode_frame = decode_frame_fn or _default_decode_frame
        self._monotonic_ms = monotonic_ms_fn or (
            lambda: time.perf_counter() * 1000.0
        )
        self._cache_reset_runs: set[str] = set()

    def create_run(
        self,
        *,
        create_request_id: str,
        run_token: str,
        capture_contract: Mapping[str, Any],
        viewport_width: float,
        viewport_height: float,
        device_pixel_ratio: float,
        readiness_preflight: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Create or repair one idempotent non-participant run namespace."""

        normalized_contract = normalize_capture_contract(capture_contract)
        viewport = self._viewport(
            viewport_width, viewport_height, device_pixel_ratio
        )
        existing = self.store.lookup_create_request(
            create_request_id=create_request_id, run_token=run_token
        )
        if existing.get("exists") is True:
            runtime = existing["runner"]["runtime_binding"]
            if isinstance(runtime, Mapping):
                if runtime["capture_contract"] != normalized_contract:
                    raise MeasurementRunnerError(
                        "recovered create capture contract differs"
                    )
                if viewport != runtime["viewport"]:
                    raise MeasurementRunnerError(
                        "recovered create viewport differs"
                    )
                # A new server process/wrapper must never inherit an opaque
                # pre-run model object.  Clear both wrapper and loader caches,
                # then rehash the frozen bytes before this instance may infer.
                self._reset_inference_cache()
                fresh = self._assert_fresh_base_bundle(runtime)
                checked = {
                    item["stage"]
                    for item in existing["runner"].get("base_bundle_checks", [])
                }
                if "run_created" not in checked:
                    self.store.record_base_bundle_check(
                        str(existing["capture_run_id"]),
                        run_token,
                        stage="run_created",
                        model_id=str(fresh["model_id"]),
                        bundle_sha256=str(fresh["bundle_sha256"]),
                    )
                    existing = self.store.get_status(
                        str(existing["capture_run_id"]), run_token
                    )
                self._cache_reset_runs.add(str(existing["capture_run_id"]))
                return {
                    **existing,
                    "run_token_client_supplied": True,
                    "run_token_returned_once": False,
                    "created_new": False,
                    "idempotent": True,
                    "initialization_recovered": True,
                }
        self._reset_inference_cache()
        bundle = self._fresh_base_bundle()
        preflight = self._validated_readiness_preflight(
            readiness_preflight,
            normalized_contract,
            viewport,
            bundle,
        )
        created = self.store.create_run(
            create_request_id=create_request_id, run_token=run_token
        )
        run_id = str(created["capture_run_id"])
        token = run_token
        with self.store.adapter_operation_lock(run_id):
            current = self.store.get_status(run_id, token)
            bound_runtime = current["runner"]["runtime_binding"]
            if isinstance(bound_runtime, Mapping):
                if bound_runtime["capture_contract"] != normalized_contract or (
                    viewport != bound_runtime["viewport"]
                ):
                    raise MeasurementRunnerError(
                        "concurrent recovered create configuration differs"
                    )
                bound_base = bound_runtime["base_inference_bundle"]
                if (
                    bound_base["model_id"],
                    bound_base["bundle_sha256"],
                ) != (bundle["model_id"], bundle["bundle_sha256"]):
                    raise MeasurementRunIntegrityError(
                        "concurrent recovered create base bundle differs"
                    )
                recovered_session = True
                status = current
            else:
                session = self._find_session(self.data_root, run_id)
                recovered_session = session is not None
                if session is None:
                    session = self._create_session(
                        self.data_root,
                        participant_id=f"wgmc_{run_id}",
                        capture_run_id=run_id,
                        capture_source=CAPTURE_SOURCE,
                    )
                if session.get("ok") is not True or not session.get("session_id"):
                    # Default recovered metadata has no transport ``ok`` field.
                    if not recovered_session or not session.get("session_id"):
                        raise MeasurementRunnerError(
                            "dedicated calibration session failed"
                        )
                session_id = str(session["session_id"])
                status = self.store.bind_runtime_context(
                    run_id,
                    token,
                    calibration_session_id=session_id,
                    capture_contract=normalized_contract,
                    viewport_width=viewport["width"],
                    viewport_height=viewport["height"],
                    device_pixel_ratio=viewport["device_pixel_ratio"],
                    base_model_id=str(bundle["model_id"]),
                    base_model_name=str(bundle["model_name"]),
                    base_bundle_sha256=str(bundle["bundle_sha256"]),
                    base_model_sha256=str(bundle["model_sha256"]),
                    base_checkpoint_sha256=str(bundle["checkpoint_sha256"]),
                    readiness_preflight=preflight,
                    base_inference_selector=BASE_INFERENCE_SELECTOR,
                )
            checked = {
                item["stage"]
                for item in status["runner"].get("base_bundle_checks", [])
            }
            if "run_created" not in checked:
                self.store.record_base_bundle_check(
                    run_id,
                    token,
                    stage="run_created",
                    model_id=str(bundle["model_id"]),
                    bundle_sha256=str(bundle["bundle_sha256"]),
                )
                status = self.store.get_status(run_id, token)
            self._cache_reset_runs.add(run_id)
            return {
                **status,
                "run_token_client_supplied": True,
                "run_token_returned_once": False,
                "created_new": bool(created.get("created_new")),
                "idempotent": bool(created.get("idempotent")),
                "initialization_recovered": bool(
                    created.get("idempotent") or recovered_session
                ),
            }

    def get_status(self, capture_run_id: str, run_token: str) -> dict[str, Any]:
        return self.store.get_status(capture_run_id, run_token)

    def lookup_create_request(
        self, *, create_request_id: str, run_token: str
    ) -> dict[str, Any]:
        """Authenticated persistent lookup used before consuming preflight."""

        result = self.store.lookup_create_request(
            create_request_id=create_request_id,
            run_token=run_token,
        )
        result.pop("run_token", None)
        return result

    def inspect_challenge(
        self,
        capture_run_id: str,
        run_token: str,
        challenge_token: str,
    ) -> dict[str, Any]:
        """Return only server schedule/render context needed by the HTTP gate."""

        with self.store.adapter_operation_lock(capture_run_id):
            inspected = self.store.inspect_challenge(
                capture_run_id, run_token, challenge_token
            )
            if inspected["status"] == "committed":
                return {
                    "ok": True,
                    "status": "committed",
                    "receipt": inspected["receipt"],
                    "measurement_claim_authorized": False,
                }
            if isinstance(inspected.get("inference_intent"), Mapping):
                return self._fail_unsealed_inference_intent(
                    capture_run_id, run_token
                )
            if isinstance(inspected.get("spooled_frame"), Mapping):
                try:
                    self.store.read_encrypted_frame_spool(
                        capture_run_id, run_token
                    )
                except MeasurementRunIntegrityError as exc:
                    return self._fail_spool_integrity(
                        capture_run_id, run_token, exc
                    )
            runtime = self._runtime(inspected)
            return {
                "ok": True,
                "status": "active",
                "capture_run_id": capture_run_id,
                "challenge_id": inspected["challenge_id"],
                "ordinal": inspected["ordinal"],
                "block_role": inspected["block_role"],
                "schedule_row": deepcopy(inspected["schedule_row"]),
                "viewport": deepcopy(runtime["viewport"]),
                "prepared_observation_pending": isinstance(
                    inspected.get("prepared_attempt"), Mapping
                ),
                "server_spool_available": isinstance(
                    inspected.get("spooled_frame"), Mapping
                ),
                "measurement_claim_authorized": False,
                "physical_capture_claim_authorized": False,
            }

    def issue_next_challenge(
        self, capture_run_id: str, run_token: str
    ) -> dict[str, Any]:
        with self.store.adapter_operation_lock(capture_run_id):
            status = self.store.get_status(capture_run_id, run_token)
            if status["runner"]["runtime_binding"] is None:
                raise MeasurementRunStateError("runtime context is not bound")
            return self.store.issue_next_challenge(capture_run_id, run_token)

    def rotate_unconsumed_challenge(
        self, capture_run_id: str, run_token: str
    ) -> dict[str, Any]:
        """Replace only a challenge with no durable frame or inference work."""

        with self.store.adapter_operation_lock(capture_run_id):
            return self.store.rotate_unconsumed_challenge(
                capture_run_id, run_token
            )

    def submit_frame(
        self,
        capture_run_id: str,
        run_token: str,
        challenge_token: str,
        *,
        image_data: str | None,
        observed_capture_contract: Mapping[str, Any],
        observed_viewport_width: float,
        observed_viewport_height: float,
        observed_device_pixel_ratio: float,
    ) -> dict[str, Any]:
        """Process one server-scheduled attempt; never accepts a target label."""

        request_received_ms = self._finite_monotonic()
        with self.store.adapter_operation_lock(capture_run_id):
            try:
                inspected = self.store.inspect_challenge(
                    capture_run_id, run_token, challenge_token
                )
            except MeasurementRunIntegrityError:
                status = self.store.get_status(capture_run_id, run_token)
                if status.get("phase") == "failed_integrity":
                    runtime = status["runner"].get("runtime_binding")
                    try:
                        self.store.delete_encrypted_frame_spool(
                            capture_run_id, run_token
                        )
                    except Exception:
                        pass
                    if isinstance(runtime, Mapping):
                        try:
                            self._purge_session_images_verified(
                                str(runtime["calibration_session_id"])
                            )
                        except Exception:
                            pass
                    try:
                        self._delete_owned_model_verified(
                            self._model_path(
                                self.data_root,
                                self._personal_model_name(capture_run_id),
                            ).resolve(),
                            capture_run_id,
                            expected_session_id=(
                                str(runtime["calibration_session_id"])
                                if isinstance(runtime, Mapping)
                                else None
                            ),
                        )
                    except Exception:
                        pass
                    try:
                        self.store.complete_failed_integrity_cleanup(
                            capture_run_id, run_token
                        )
                    except Exception:
                        pass
                raise
            if inspected["status"] == "committed":
                self.store.delete_encrypted_frame_spool(
                    capture_run_id, run_token
                )
                return {
                    "ok": True,
                    "classification": "exact_retry",
                    "consumed": True,
                    "retryable": False,
                    "receipt": inspected["receipt"],
                    "inference_replayed": False,
                    "measurement_claim_authorized": False,
                    "physical_capture_claim_authorized": False,
                }
            if isinstance(inspected.get("inference_intent"), Mapping):
                return self._fail_unsealed_inference_intent(
                    capture_run_id, run_token
                )
            prepared = inspected.get("prepared_attempt")
            if isinstance(prepared, Mapping):
                return self._resume_prepared_attempt(
                    inspected,
                    run_token,
                    challenge_token,
                    image_data=image_data,
                    observed_capture_contract=observed_capture_contract,
                )

            runtime = self._runtime(inspected)
            spooled = inspected.get("spooled_frame")
            if isinstance(spooled, Mapping) and spooled.get("status") == (
                "cleanup_pending_uncommitted"
            ):
                self.store.delete_encrypted_frame_spool(
                    capture_run_id, run_token
                )
                return self._hard_error(
                    "prior_hard_error_cleanup_recovered",
                    "encrypted frame cleanup recovered; submit a new frame",
                )
            if isinstance(spooled, Mapping):
                try:
                    raw = self.store.read_encrypted_frame_spool(
                        capture_run_id, run_token
                    )
                except MeasurementRunIntegrityError as exc:
                    return self._fail_spool_integrity(
                        capture_run_id, run_token, exc
                    )
                evidence = deepcopy(spooled["capture_contract_evidence"])
                observed_contract = deepcopy(
                    evidence["observed_capture_contract"]
                )
                comparison = deepcopy(evidence["contract_comparison"])
                observed_viewport = deepcopy(evidence["viewport"])
                width = int(evidence["decoded_transport_width_px"])
                height = int(evidence["decoded_transport_height_px"])
                receive_context = deepcopy(spooled["server_receive_context"])
                inference_image_data = "data:image/jpeg;base64," + base64.b64encode(
                    raw
                ).decode("ascii")
            else:
                try:
                    observed_viewport = self._viewport(
                        observed_viewport_width,
                        observed_viewport_height,
                        observed_device_pixel_ratio,
                    )
                    if observed_viewport != runtime["viewport"]:
                        raise MeasurementRunnerError(
                            "viewport or device-pixel ratio changed during the run"
                        )
                    raw, width, height, observed_contract, comparison = (
                        self._decode_and_verify_contract(
                            image_data,
                            observed_capture_contract,
                            runtime,
                        )
                    )
                    decode_completed_ms = self._finite_monotonic(
                        minimum=request_received_ms
                    )
                except (MeasurementRunnerError, ValueError) as exc:
                    return self._hard_error("capture_hard_error", str(exc))
                inference_image_data = str(image_data)
                evidence = self._capture_contract_evidence(
                    runtime,
                    observed_contract,
                    comparison,
                    width,
                    height,
                    observed_viewport,
                )
                receive_context = {
                    "server_request_received_monotonic_ms": request_received_ms,
                    "decode_completed_monotonic_ms": decode_completed_ms,
                    "frame_source": "request_decode",
                }
            frame_sha = _sha256_bytes(raw)
            self.store.persist_encrypted_frame_spool(
                capture_run_id,
                run_token,
                challenge_token,
                frame_bytes=raw,
                capture_contract_evidence=evidence,
                server_receive_context=receive_context,
            )
            row = dict(inspected["schedule_row"])
            role = str(row["block_role"])
            model_id, model_sha, selector = self._model_for_role(
                capture_run_id, role, runtime, inspected.get("model_binding")
            )
            self._ensure_fresh_cache(capture_run_id)
            if role == "calibration":
                check_status = self.store.get_status(capture_run_id, run_token)
                checked_stages = {
                    item["stage"]
                    for item in check_status["runner"]["base_bundle_checks"]
                }
                if "first_calibration_inference" not in checked_stages:
                    fresh = self._assert_fresh_base_bundle(runtime)
                    self.store.record_base_bundle_check(
                        capture_run_id,
                        run_token,
                        stage="first_calibration_inference",
                        model_id=str(fresh["model_id"]),
                        bundle_sha256=str(fresh["bundle_sha256"]),
                    )
            elif not self._rehash_personal_model(inspected):
                self._fail_evaluation_model_integrity(
                    inspected,
                    run_token,
                    code="personal_model_hash_mismatch_before_inference",
                    detail="personalized model changed before evaluation inference",
                )
                raise MeasurementRunIntegrityError(
                    "personalized model changed before evaluation inference"
                )

            started_ms = self._finite_monotonic(
                minimum=float(receive_context["decode_completed_monotonic_ms"])
            )
            self.store.begin_inference_attempt(
                capture_run_id,
                run_token,
                challenge_token,
                frame_sha256=frame_sha,
                model_id=model_id,
                model_sha256=model_sha,
                model_selector=selector,
                capture_contract_evidence=evidence,
                server_receive_context=receive_context,
                predict_started_monotonic_ms=started_ms,
            )
            response, status_code = self._predict(
                self.data_root,
                {
                    "image_data": inference_image_data,
                    "capture_contract": observed_contract,
                    "model_name": selector,
                    "viewport_width": runtime["viewport"]["width"],
                    "viewport_height": runtime["viewport"]["height"],
                    "allow_cuda": False,
                },
            )
            completed_ms = self._finite_monotonic(minimum=started_ms)
            if role != "calibration" and not self._rehash_personal_model(inspected):
                self._fail_evaluation_model_integrity(
                    inspected,
                    run_token,
                    code="personal_model_hash_mismatch_after_inference",
                    detail="personalized model changed during evaluation inference",
                )
                raise MeasurementRunIntegrityError(
                    "personalized model changed during evaluation inference"
                )

            classification = self._prediction_classification(
                response,
                status_code,
                expected_model_name=selector,
            )
            if classification == "hard_error":
                self.store.clear_inference_intent_after_hard_error(
                    capture_run_id,
                    run_token,
                    challenge_token,
                    frame_sha256=frame_sha,
                )
                self.store.delete_encrypted_frame_spool(
                    capture_run_id, run_token
                )
                return self._hard_error(
                    str(response.get("failure_stage") or "inference_hard_error"),
                    str(response.get("error") or "inference failed"),
                )
            no_face = classification == "attributable_sensor_failure"
            observation = self._observation(
                row=row,
                runtime=runtime,
                observed_contract=observed_contract,
                model_id=model_id,
                model_sha256=model_sha,
                response=response,
                started_ms=started_ms,
                completed_ms=completed_ms,
                no_face=no_face,
            )
            self.store.record_attempt_observation(
                capture_run_id,
                run_token,
                challenge_token,
                frame_sha256=frame_sha,
                observation=observation,
                capture_contract_evidence=evidence,
                server_timing_evidence={
                    "schema_version": 1,
                    "timing_semantics": (
                        "v1_sample_frame_capture_is_predict_start_proxy_not_camera_exposure"
                    ),
                    **receive_context,
                    "predict_started_monotonic_ms": started_ms,
                    "predict_completed_monotonic_ms": completed_ms,
                    "camera_exposure_timestamp_available": False,
                    "client_timing_used_for_integrity": False,
                },
                disposition="no_face_detected" if no_face else "success",
            )
            if role == "calibration" and not no_face:
                saved = self._persist_calibration_sample(
                    inspected,
                    run_token,
                    challenge_token,
                    image_data=inference_image_data,
                    raw=raw,
                    frame_sha256=frame_sha,
                    observed_contract=observed_contract,
                    success_observation=observation,
                )
                if saved.get("ok") is not True:
                    return saved
                if saved.get("classification") == "attributable_sensor_failure":
                    no_face = True
            receipt = self.store.commit_prepared_observation(
                capture_run_id, run_token
            )
            self.store.delete_encrypted_frame_spool(capture_run_id, run_token)
            return self._committed_result(receipt, no_face=no_face)

    def finalize_calibration(
        self, capture_run_id: str, run_token: str
    ) -> dict[str, Any]:
        """Finalize calibration and convert external-effect crashes to evidence."""

        try:
            return self._finalize_calibration_once(capture_run_id, run_token)
        except Exception as exc:
            with self.store.adapter_operation_lock(capture_run_id):
                status = self.store.get_status(capture_run_id, run_token)
                if status["phase"] not in {
                    PHASE_CALIBRATION_SEALED,
                    PHASE_MODEL_BOUND,
                    "failed_integrity",
                }:
                    raise
                cleanup_verified = True
                cleanup_errors: list[str] = []
                runtime = status["runner"].get("runtime_binding")
                try:
                    self.store.delete_encrypted_frame_spool(
                        capture_run_id, run_token
                    )
                except Exception as cleanup_exc:
                    cleanup_verified = False
                    cleanup_errors.append(f"spool: {cleanup_exc}")
                if isinstance(runtime, Mapping):
                    try:
                        self._purge_session_images_verified(
                            str(runtime["calibration_session_id"])
                        )
                    except Exception as cleanup_exc:
                        cleanup_verified = False
                        cleanup_errors.append(f"images: {cleanup_exc}")
                try:
                    if (
                        isinstance(
                            status["runner"].get("training_artifact_intent"),
                            Mapping,
                        )
                        and status["runner"].get("trained_artifact") is None
                    ):
                        self.store.delete_reserved_training_artifact(
                            capture_run_id, run_token
                        )
                    else:
                        self._delete_owned_model_verified(
                            self._model_path(
                                self.data_root,
                                self._personal_model_name(capture_run_id),
                            ).resolve(),
                            capture_run_id,
                            expected_session_id=(
                                str(runtime["calibration_session_id"])
                                if isinstance(runtime, Mapping)
                                else None
                            ),
                        )
                except Exception as cleanup_exc:
                    cleanup_verified = False
                    cleanup_errors.append(f"model: {cleanup_exc}")
                if status["phase"] != "failed_integrity":
                    self.store.fail_integrity(
                        capture_run_id,
                        run_token,
                        code="calibration_finalization_failed",
                        detail=str(exc),
                    )
                failed = self.store.get_status(capture_run_id, run_token)
                persisted_cleanup = bool(
                    (failed.get("failure") or {}).get("cleanup", {}).get(
                        "cleanup_verified"
                    )
                )
                return {
                    **failed,
                    "ok": False,
                    "classification": "calibration_finalization_failed",
                    "detail": str(exc)[:512],
                    "cleanup_verified": cleanup_verified and persisted_cleanup,
                    "cleanup_errors": cleanup_errors,
                    "training_device": "cpu",
                    "measurement_claim_authorized": False,
                }

    def _finalize_calibration_once(
        self, capture_run_id: str, run_token: str
    ) -> dict[str, Any]:
        """Train on exactly 65 usable rows, bind, then purge all imagery."""

        with self.store.adapter_operation_lock(capture_run_id):
            status = self.store.get_status(capture_run_id, run_token)
            if status["phase"] == PHASE_MODEL_BOUND:
                self._purge_and_record(capture_run_id, run_token, status)
                return self.store.get_status(capture_run_id, run_token)
            if status["phase"] != PHASE_CALIBRATION_SEALED:
                raise MeasurementRunStateError(
                    "calibration must contain exactly 65 scheduled attempts"
                )
            usable = int(status["runner"]["calibration_usable_manifest_count"])
            if usable != _schedule.EXPECTED_CALIBRATION_SAMPLE_COUNT:
                self._purge_and_record(capture_run_id, run_token, status)
                self.store.fail_integrity(
                    capture_run_id,
                    run_token,
                    code="calibration_usable_count_below_65",
                    detail=(
                        f"only {usable} of 65 scheduled calibration attempts "
                        "contained a usable face; replacement capture is forbidden"
                    ),
                )
                failed = self.store.get_status(capture_run_id, run_token)
                return {
                    **failed,
                    "ok": False,
                    "classification": "negative_calibration_result",
                    "calibration_attempt_count": 65,
                    "usable_calibration_count": usable,
                    "required_calibration_count": 65,
                    "training_started": False,
                    "images_purged": True,
                    "cleanup_verified": bool(
                        (failed.get("failure") or {}).get("cleanup", {}).get(
                            "cleanup_verified"
                        )
                    ),
                    "terminal": True,
                    "measurement_claim_authorized": False,
                }
            runtime = status["runner"]["runtime_binding"]
            session_id = str(runtime["calibration_session_id"])
            fresh = self._assert_fresh_base_bundle(runtime)
            self.store.record_base_bundle_check(
                capture_run_id,
                run_token,
                stage="calibration_sealed_pre_training",
                model_id=str(fresh["model_id"]),
                bundle_sha256=str(fresh["bundle_sha256"]),
            )
            status = self.store.get_status(capture_run_id, run_token)
            training_binding = self.store.read_calibration_training_binding(
                capture_run_id, run_token
            )
            records, manifest_sha, row_hashes, image_bindings = (
                self._verified_training_manifest(
                session_id, capture_run_id, runtime
                )
            )
            if len(records) != 65:
                raise MeasurementRunIntegrityError(
                    "dedicated training manifest is not exactly 65 rows"
                )
            self._assert_manifest_matches_store_binding(
                training_binding,
                row_hashes=row_hashes,
                image_bindings=image_bindings,
            )
            model_name = self._personal_model_name(capture_run_id)
            artifact_path = self._model_path(self.data_root, model_name).resolve()
            relative = artifact_path.relative_to(self.data_root).as_posix()
            self.store.reserve_training_artifact(
                capture_run_id,
                run_token,
                model_id=model_name,
                artifact_relative_path=relative,
            )
            measurement_training_binding = self._measurement_training_binding(
                capture_run_id=capture_run_id,
                session_id=session_id,
                manifest_sha256=manifest_sha,
                store_binding=training_binding,
                image_bindings=image_bindings,
                base_inference_bundle=runtime["base_inference_bundle"],
            )
            provenance = self._training_provenance(
                capture_run_id,
                status,
                manifest_sha=manifest_sha,
                manifest_row_sha256s=row_hashes,
                image_bindings=image_bindings,
                training_binding=training_binding,
                measurement_training_binding=measurement_training_binding,
            )
            provenance_sha = _schedule.canonical_sha256(provenance)
            existing = self._matching_owned_artifact(
                artifact_path, provenance_sha, capture_run_id
            )
            if existing is None:
                response, response_status = self._train(
                    self.data_root,
                    {
                        "data_session_id": session_id,
                        "base_model_name": "0",
                        "output_model_name": model_name,
                        "allow_cuda": False,
                        "measurement_training_binding": measurement_training_binding,
                    },
                )
                if response_status != 200 or response.get("ok") is not True:
                    self._purge_and_record(capture_run_id, run_token, status)
                    self.store.delete_reserved_training_artifact(
                        capture_run_id, run_token
                    )
                    self.store.fail_integrity(
                        capture_run_id,
                        run_token,
                        code="cpu_training_failed",
                        detail=str(response.get("error") or "training failed"),
                    )
                    return {
                        "ok": False,
                        "classification": "training_hard_error",
                        "training_device": "cpu",
                        "images_purged": True,
                        "measurement_claim_authorized": False,
                    }
                self._validate_training_response(
                    response,
                    model_name,
                    expected_consumed_sha256=measurement_training_binding[
                        "rows_sha256"
                    ],
                    expected_binding_sha256=measurement_training_binding[
                        "binding_sha256"
                    ],
                    expected_base_inference_bundle=measurement_training_binding[
                        "base_inference_bundle"
                    ],
                )
                post_training_base = self._assert_fresh_base_bundle(runtime)
                if normalize_base_bundle_identity(post_training_base) != (
                    normalize_base_bundle_identity(
                        measurement_training_binding["base_inference_bundle"]
                    )
                ):
                    raise MeasurementRunIntegrityError(
                        "base inference bundle changed during CPU training"
                    )
                post_records, post_manifest_sha, post_row_hashes, post_images = (
                    self._verified_training_manifest(
                        session_id, capture_run_id, runtime
                    )
                )
                if (
                    post_manifest_sha != manifest_sha
                    or post_row_hashes != row_hashes
                    or post_images != image_bindings
                    or len(post_records) != len(records)
                ):
                    raise MeasurementRunIntegrityError(
                        "training inputs changed while CPU training was running"
                    )
                post_binding = self.store.read_calibration_training_binding(
                    capture_run_id, run_token
                )
                if post_binding != training_binding:
                    raise MeasurementRunIntegrityError(
                        "persistent calibration binding changed during training"
                    )
                artifact = self._load_model_artifact(artifact_path)
                artifact["measurement_ceiling_provenance"] = provenance
                _atomic_json(artifact_path, artifact)
            model_sha = _sha256_bytes(artifact_path.read_bytes())
            artifact = self._load_model_artifact(artifact_path)
            self._validate_final_model_artifact(
                artifact,
                provenance_sha=provenance_sha,
                expected_binding_sha256=measurement_training_binding[
                    "binding_sha256"
                ],
                expected_rows_sha256=measurement_training_binding["rows_sha256"],
                capture_run_id=capture_run_id,
                expected_base_inference_bundle=measurement_training_binding[
                    "base_inference_bundle"
                ],
            )
            status = self.store.bind_trained_model(
                capture_run_id,
                run_token,
                model_id=model_name,
                model_sha256=model_sha,
                artifact_relative_path=relative,
                calibration_ledger_sha256=status["ledgers"]["calibration"][
                    "sealed_sha256"
                ],
                training_provenance_sha256=provenance_sha,
            )
            self._purge_and_record(capture_run_id, run_token, status)
            return self.store.get_status(capture_run_id, run_token)

    def verify_artifact(
        self, capture_run_id: str, run_token: str
    ) -> dict[str, Any]:
        return self.store.verify_sealed_artifact(capture_run_id, run_token)

    def read_verified_analysis_evidence(
        self, capture_run_id: str, run_token: str
    ) -> dict[str, Any]:
        """Return a reverified, frame-free evidence bundle for analysis."""

        with self.store.adapter_operation_lock(capture_run_id):
            verification = self.store.verify_sealed_artifact(
                capture_run_id, run_token
            )
            status = self.store.get_status(capture_run_id, run_token)
            if (
                status["phase"] != "artifact_verified"
                or status.get("acquisition_artifact_verified") is not True
                or status.get("capture_contract_binding_verified") is not True
            ):
                raise MeasurementRunIntegrityError(
                    "analysis evidence is not fully verified"
                )
            trained = status["runner"].get("trained_artifact")
            purge = status["runner"].get("calibration_image_purge")
            if not isinstance(trained, Mapping) or not isinstance(purge, Mapping):
                raise MeasurementRunIntegrityError(
                    "analysis evidence lacks model or purge provenance"
                )
            model_path = (
                self.data_root / str(trained["artifact_relative_path"])
            ).resolve()
            try:
                model_bytes = model_path.read_bytes()
            except OSError as exc:
                raise MeasurementRunIntegrityError(
                    "analysis evidence model is unreadable"
                ) from exc
            if _sha256_bytes(model_bytes) != trained.get("model_sha256"):
                raise MeasurementRunIntegrityError(
                    "analysis evidence model SHA-256 changed"
                )
            try:
                model = json.loads(model_bytes.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise MeasurementRunIntegrityError(
                    "analysis evidence model is invalid JSON"
                ) from exc
            if not isinstance(model, dict):
                raise MeasurementRunIntegrityError(
                    "analysis evidence model must be an object"
                )
            provenance = model.get("measurement_ceiling_provenance")
            if not isinstance(provenance, Mapping):
                raise MeasurementRunIntegrityError(
                    "analysis evidence lacks training provenance"
                )
            runtime = status["runner"].get("runtime_binding")
            if not isinstance(runtime, Mapping) or (
                purge.get("calibration_session_id")
                != runtime.get("calibration_session_id")
            ):
                raise MeasurementRunIntegrityError(
                    "analysis evidence purge session binding changed"
                )
            session_dir = self._safe_session_dir(
                str(runtime["calibration_session_id"])
            )
            remaining_image_directories = [
                name
                for name in DEDICATED_IMAGE_DIRECTORIES
                if (session_dir / name).exists()
            ]
            if remaining_image_directories:
                raise MeasurementRunIntegrityError(
                    "analysis evidence calibration-image purge no longer holds"
                )
            bundle: dict[str, Any] = {
                "schema_version": 1,
                "evidence_type": (
                    "webcam_gaze_measurement_ceiling_verified_analysis_evidence_v1"
                ),
                "capture_run_id": capture_run_id,
                "verification": verification,
                "status": status,
                "capture_artifact": self.store.read_sealed_artifact(
                    capture_run_id, run_token
                ),
                "attempt_sidecar": self.store.read_sealed_attempt_sidecar(
                    capture_run_id, run_token
                ),
                "training_provenance": deepcopy(dict(provenance)),
                "model_sha256": trained["model_sha256"],
                "calibration_image_purge": deepcopy(dict(purge)),
                "calibration_image_absence_reverified": True,
                "spool_absence_verified": True,
                "raw_frames_included": False,
                "run_token_included": False,
                "measurement_claim_authorized": False,
                "physical_capture_claim_authorized": False,
            }
            bundle["evidence_sha256"] = _schedule.canonical_sha256(bundle)
            return bundle

    def analyze_verified_run(
        self,
        capture_run_id: str,
        run_token: str,
    ) -> dict[str, Any]:
        """Run the frozen descriptive analysis through this canonical runner.

        The authority-bearing entry point deliberately lives on
        ``MeasurementRunner``.  The analysis module only exposes a private
        evidence consumer, so an arbitrary duck-typed object cannot promote a
        self-consistent file bundle to live-runner provenance.
        """

        from core.gaze_core.measurement_run_analysis import (
            _analyze_reverified_live_evidence,
        )

        evidence = self.read_verified_analysis_evidence(
            capture_run_id,
            run_token,
        )
        return _analyze_reverified_live_evidence(evidence)

    def abort_and_cleanup(
        self,
        capture_run_id: str,
        run_token: str,
        *,
        reason: str,
    ) -> dict[str, Any]:
        with self.store.adapter_operation_lock(capture_run_id):
            status = self.store.get_status(capture_run_id, run_token)
            if status["phase"] == "aborted":
                return {
                    **status,
                    "cleanup_verified": bool(
                        (status.get("abort") or {}).get("cleanup", {}).get(
                            "cleanup_verified"
                        )
                    ),
                    "idempotent": True,
                }
            if status["runner"].get("inference_in_progress") is True:
                self.store.fail_integrity(
                    capture_run_id,
                    run_token,
                    code="unsealed_inference_result_before_abort",
                    detail=(
                        "authenticated cleanup found a predictor call without "
                        "a durable response observation"
                    ),
                )
                status = self.store.get_status(capture_run_id, run_token)
            if status["phase"] == "artifact_verified":
                raise MeasurementRunStateError(
                    "terminal run cannot be aborted or have its evidence removed"
                )
            try:
                spool = self.store.delete_encrypted_frame_spool(
                    capture_run_id, run_token
                )
            except Exception as exc:
                return {
                    **self._hard_error(
                        "spool_cleanup_failed",
                        str(exc),
                    ),
                    "cleanup_verified": False,
                    "terminal": False,
                }
            runtime = status["runner"]["runtime_binding"]
            session: Mapping[str, Any] | None = None
            if isinstance(runtime, Mapping):
                session = {
                    "session_id": str(runtime["calibration_session_id"])
                }
            else:
                session = self._find_session(self.data_root, capture_run_id)
            if isinstance(session, Mapping) and session.get("session_id"):
                try:
                    self._purge_session_images_verified(
                        str(session["session_id"])
                    )
                except Exception as exc:
                    return {
                        **self._hard_error(
                            "dedicated_image_cleanup_failed",
                            str(exc),
                        ),
                        "cleanup_verified": False,
                        "spool_absence_verified": bool(
                            spool.get("absence_verified")
                        ),
                        "terminal": False,
                    }
            model_path = self._model_path(
                self.data_root, self._personal_model_name(capture_run_id)
            ).resolve()
            try:
                if (
                    isinstance(
                        status["runner"].get("training_artifact_intent"),
                        Mapping,
                    )
                    and status["runner"].get("trained_artifact") is None
                ):
                    self.store.delete_reserved_training_artifact(
                        capture_run_id, run_token
                    )
                else:
                    self._delete_owned_model_verified(
                        model_path,
                        capture_run_id,
                        expected_session_id=(
                            str(runtime["calibration_session_id"])
                            if isinstance(runtime, Mapping)
                            else None
                        ),
                    )
            except Exception as exc:
                return {
                    **self._hard_error(
                        "dedicated_model_cleanup_failed",
                        str(exc),
                    ),
                    "cleanup_verified": False,
                    "spool_absence_verified": True,
                    "image_directories_absent": True,
                    "model_absence_verified": False,
                    "terminal": False,
                }
            if status["phase"] == "failed_integrity":
                aborted = self.store.complete_failed_integrity_cleanup(
                    capture_run_id, run_token
                )
                terminal_classification = "failed_integrity_cleanup_verified"
            else:
                aborted = self.store.abort_run(
                    capture_run_id,
                    run_token,
                    reason=reason,
                )
                terminal_classification = "aborted_cleanup_verified"
            return {
                **aborted,
                "classification": terminal_classification,
                "cleanup_verified": True,
                "spool_absence_verified": True,
                "model_absence_verified": True,
                "terminal": True,
                "idempotent": False,
            }

    # ---- internal contract helpers -------------------------------------------------

    def _fresh_base_bundle(self) -> dict[str, Any]:
        bundle = deepcopy(dict(self._base_bundle(self.code_root)))
        verified = self._verify_base_bundle(bundle, self.code_root)
        if verified.get("status") != "passed":
            raise MeasurementRunnerError("base inference bundle verification failed")
        for field in ("model_id", "model_name", "bundle_sha256", "model_sha256"):
            if not bundle.get(field):
                raise MeasurementRunnerError(f"base bundle lacks {field}")
        if bundle["bundle_sha256"] != bundle["model_sha256"]:
            raise MeasurementRunnerError("base bundle and model SHA differ")
        return bundle

    @staticmethod
    def _validated_readiness_preflight(
        proof: Mapping[str, Any] | None,
        capture_contract: Mapping[str, Any],
        viewport: Mapping[str, Any],
        bundle: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(proof, Mapping):
            raise MeasurementRunnerError("readiness preflight proof is required")
        normalized = deepcopy(dict(proof))
        required = {
            "schema_version",
            "implementation_id",
            "preflight_id",
            "capture_contract_sha256",
            "consumed_capture_contract_sha256",
            "capture_contract_warnings",
            "viewport_sha256",
            "consecutive_successes",
            "distinct_frame_count",
            "distinct_frame_set_sha256",
            "target_free",
            "persistent_storage_used",
            "fixed_model_name",
            "base_inference_bundle",
            "allow_cuda",
            "measurement_claim_authorized",
            "receipt_type",
            "receipt_sha256",
        }
        if set(normalized) != required:
            raise MeasurementRunnerError("readiness preflight fields are not exact")
        expected = {
            "schema_version": 1,
            "implementation_id": "target-free-camera-readiness-v1",
            "consumed_capture_contract_sha256": _schedule.canonical_sha256(
                capture_contract
            ),
            "viewport_sha256": _schedule.canonical_sha256(viewport),
            "consecutive_successes": 3,
            "target_free": True,
            "persistent_storage_used": False,
            "fixed_model_name": "before",
            "allow_cuda": False,
            "measurement_claim_authorized": False,
            "receipt_type": "target_free_camera_readiness_receipt_v1",
        }
        for field, value in expected.items():
            if normalized.get(field) != value:
                raise MeasurementRunnerError(
                    f"readiness preflight {field} differs from run configuration"
                )
        receipt_id = str(normalized.get("preflight_id") or "")
        if not receipt_id.startswith("PF-") or len(receipt_id) > 128:
            raise MeasurementRunnerError("readiness preflight receipt ID is invalid")
        for field in (
            "capture_contract_sha256",
            "distinct_frame_set_sha256",
        ):
            value = str(normalized.get(field) or "")
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise MeasurementRunnerError(
                    f"readiness preflight {field} is invalid"
                )
        warnings = normalized.get("capture_contract_warnings")
        if (
            not isinstance(warnings, list)
            or any(not isinstance(item, str) for item in warnings)
            or warnings != sorted(set(warnings))
        ):
            raise MeasurementRunnerError(
                "readiness preflight capture contract warnings are invalid"
            )
        distinct_count = normalized.get("distinct_frame_count")
        if (
            isinstance(distinct_count, bool)
            or not isinstance(distinct_count, int)
            or distinct_count < 3
        ):
            raise MeasurementRunnerError(
                "readiness preflight requires three distinct frames"
            )
        proof_bundle = normalized.get("base_inference_bundle")
        if not isinstance(proof_bundle, Mapping) or set(proof_bundle) != {
            "model_id",
            "model_name",
            "model_sha256",
            "bundle_sha256",
            "checkpoint_sha256",
        }:
            raise MeasurementRunnerError(
                "readiness preflight base bundle fields are not exact"
            )
        if normalize_base_bundle_identity(proof_bundle) != (
            normalize_base_bundle_identity(bundle)
        ):
            raise MeasurementRunnerError(
                "readiness preflight base bundle differs from fresh run identity"
            )
        stored_sha = str(normalized.get("receipt_sha256") or "")
        core = deepcopy(normalized)
        core.pop("receipt_sha256", None)
        if stored_sha != _schedule.canonical_sha256(core):
            raise MeasurementRunnerError("readiness preflight receipt hash mismatch")
        return normalized

    def _assert_fresh_base_bundle(
        self, runtime: Mapping[str, Any]
    ) -> dict[str, Any]:
        fresh = self._fresh_base_bundle()
        bound = runtime["base_inference_bundle"]
        if normalize_base_bundle_identity(fresh) != normalize_base_bundle_identity(
            bound
        ):
            raise MeasurementRunIntegrityError(
                "fresh base inference bundle differs from frozen run binding"
            )
        return fresh

    def _ensure_fresh_cache(self, capture_run_id: str) -> None:
        if capture_run_id not in self._cache_reset_runs:
            self._reset_inference_cache()
            self._cache_reset_runs.add(capture_run_id)

    @staticmethod
    def _viewport(width: Any, height: Any, dpr: Any) -> dict[str, float]:
        values: dict[str, float] = {}
        for field, value in (
            ("width", width),
            ("height", height),
            ("device_pixel_ratio", dpr),
        ):
            if isinstance(value, bool):
                raise ValueError(f"viewport {field} must be positive")
            number = float(value)
            if not math.isfinite(number) or number <= 0:
                raise ValueError(f"viewport {field} must be positive")
            values[field] = number
        return values

    @staticmethod
    def _runtime(inspected: Mapping[str, Any]) -> dict[str, Any]:
        runtime = inspected.get("runtime_binding")
        if not isinstance(runtime, Mapping):
            raise MeasurementRunStateError("runtime context is not bound")
        return deepcopy(dict(runtime))

    def _decode_and_verify_contract(
        self,
        image_data: str | None,
        observed_contract: Mapping[str, Any],
        runtime: Mapping[str, Any],
    ) -> tuple[bytes, int, int, dict[str, Any], dict[str, Any]]:
        normalized = normalize_capture_contract(observed_contract)
        comparison = compare_capture_contracts(
            runtime["capture_contract"], normalized
        )
        if comparison["compatible"] is not True:
            raise MeasurementRunnerError(
                "capture contract mismatch: " + ", ".join(comparison["reasons"])
            )
        raw, width, height = self._decode_frame(str(image_data or ""))
        validate_transport_frame(
            normalized, frame_width_px=width, frame_height_px=height
        )
        return raw, width, height, normalized, comparison

    @staticmethod
    def _capture_contract_evidence(
        runtime: Mapping[str, Any],
        observed_contract: Mapping[str, Any],
        comparison: Mapping[str, Any],
        width: int,
        height: int,
        observed_viewport: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "observed_capture_contract": deepcopy(dict(observed_contract)),
            "observed_capture_contract_sha256": _schedule.canonical_sha256(
                observed_contract
            ),
            "decoded_transport_width_px": width,
            "decoded_transport_height_px": height,
            "contract_comparison": deepcopy(dict(comparison)),
            "transport_frame_validated": True,
            "viewport": deepcopy(dict(observed_viewport)),
            "viewport_sha256": _schedule.canonical_sha256(observed_viewport),
        }

    @staticmethod
    def _prediction_classification(
        response: Mapping[str, Any],
        status_code: int,
        *,
        expected_model_name: str,
    ) -> str:
        if response.get("model_name") != expected_model_name:
            return "hard_error"
        if status_code == 200 and response.get("ok") is True:
            return "success"
        if (
            status_code == 400
            and response.get("ok") is False
            and response.get("failure_code") == "no_face_detected"
            and response.get("failure_stage") == "attributable_sensor_failure"
        ):
            return "attributable_sensor_failure"
        return "hard_error"

    def _model_for_role(
        self,
        capture_run_id: str,
        role: str,
        runtime: Mapping[str, Any],
        model_binding: Any,
    ) -> tuple[str, str, str]:
        if role == "calibration":
            base = runtime["base_inference_bundle"]
            return (
                str(base["model_id"]),
                str(base["bundle_sha256"]),
                str(base["inference_selector"]),
            )
        if not isinstance(model_binding, Mapping):
            raise MeasurementRunStateError("evaluation model is not bound")
        return (
            str(model_binding["model_id"]),
            str(model_binding["model_sha256"]),
            str(model_binding["model_id"]),
        )

    def _observation(
        self,
        *,
        row: Mapping[str, Any],
        runtime: Mapping[str, Any],
        observed_contract: Mapping[str, Any],
        model_id: str,
        model_sha256: str,
        response: Mapping[str, Any],
        started_ms: float,
        completed_ms: float,
        no_face: bool,
    ) -> dict[str, Any]:
        viewport = runtime["viewport"]
        contract = observed_contract
        width = float(viewport["width"])
        height = float(viewport["height"])
        target_x = float(math.floor(row["target_x_viewport_fraction"] * width + 0.5))
        target_y = float(math.floor(row["target_y_viewport_fraction"] * height + 0.5))
        if no_face:
            sensor = {
                "prediction_success": False,
                "raw_gaze_pitch_yaw": None,
                "predicted_x_px": None,
                "predicted_y_px": None,
                "head_pose_pitch_yaw": None,
                "normalized_face_bbox": None,
            }
        else:
            bbox = response.get("face_bbox")
            if not isinstance(bbox, Mapping):
                raise MeasurementRunnerError("inference response lacks face_bbox")
            sensor = {
                "prediction_success": True,
                "raw_gaze_pitch_yaw": self._vector(
                    response.get("gaze_pitch_yaw"), 2, "gaze_pitch_yaw"
                ),
                "predicted_x_px": self._vector(
                    response.get("screen_xy_px"), 2, "screen_xy_px"
                )[0],
                "predicted_y_px": self._vector(
                    response.get("screen_xy_px"), 2, "screen_xy_px"
                )[1],
                "head_pose_pitch_yaw": self._vector(
                    response.get("head_pose_pitch_yaw"), 2, "head_pose_pitch_yaw"
                ),
                "normalized_face_bbox": [
                    float(bbox["x_norm"]),
                    float(bbox["y_norm"]),
                    float(bbox["x_norm"]) + float(bbox["w_norm"]),
                    float(bbox["y_norm"]) + float(bbox["h_norm"]),
                ],
            }
        observation: dict[str, Any] = {
            "capture_source": CAPTURE_SOURCE,
            "target_x_px": target_x,
            "target_y_px": target_y,
            "frame_capture_monotonic_ms": started_ms,
            "inference_completed_monotonic_ms": completed_ms,
            "inference_latency_ms": completed_ms - started_ms,
            "model_id": model_id,
            "model_sha256": model_sha256,
            **sensor,
            "camera_width": contract["source_width_px"],
            "camera_height": contract["source_height_px"],
            "camera_frame_rate": (
                contract["source_frame_rate_hz"]
                or contract["intent_frame_rate_hz"]
            ),
            "viewport_width": width,
            "viewport_height": height,
            "device_pixel_ratio": viewport["device_pixel_ratio"],
        }
        if not no_face:
            uncertainty = response.get("uncertainty")
            if isinstance(uncertainty, Mapping) and uncertainty.get("status") == (
                "scored_no_threshold"
            ):
                score = uncertainty.get("score")
                covariance = uncertainty.get(
                    "jackknife_disagreement_covariance_px"
                )
                if score is not None:
                    observation["sensor_uncertainty_score"] = float(score)
                if covariance is not None:
                    observation["prediction_covariance_px"] = deepcopy(covariance)
        return observation

    @staticmethod
    def _vector(value: Any, length: int, field: str) -> list[float]:
        if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
            raise MeasurementRunnerError(f"{field} must contain {length} values")
        if len(value) != length:
            raise MeasurementRunnerError(f"{field} must contain {length} values")
        result = [float(item) for item in value]
        if not all(math.isfinite(item) for item in result):
            raise MeasurementRunnerError(f"{field} must be finite")
        return result

    def _finite_monotonic(self, *, minimum: float = 0.0) -> float:
        value = float(self._monotonic_ms())
        if not math.isfinite(value) or value < minimum:
            raise MeasurementRunnerError("server monotonic clock is invalid")
        return value

    def _persist_calibration_sample(
        self,
        inspected: Mapping[str, Any],
        run_token: str,
        challenge_token: str,
        *,
        image_data: str,
        raw: bytes,
        frame_sha256: str,
        observed_contract: Mapping[str, Any],
        success_observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        run_id = str(inspected["capture_run_id"])
        runtime = self._runtime(inspected)
        row = dict(inspected["schedule_row"])
        session_id = str(runtime["calibration_session_id"])
        prepared = self.store.prepare_calibration_write(
            run_id,
            run_token,
            challenge_token,
            frame_sha256=frame_sha256,
        )
        if prepared.get("status") == "no_face_reclassification_pending":
            self._complete_pending_no_face_reclassification(
                run_id,
                run_token,
                challenge_token,
                session_id=session_id,
                pending=prepared,
                frame_sha256=frame_sha256,
            )
            return {
                "ok": True,
                "classification": "attributable_sensor_failure",
                "consumed": False,
                "retryable": False,
            }
        existing = self._recover_saved_manifest_row(
            session_id,
            expected_index=int(prepared["expected_manifest_index"]),
            frame_sha256=frame_sha256,
            row=row,
            runtime=runtime,
            observed_contract=observed_contract,
            require_present=False,
        )
        if existing is None:
            try:
                response, status_code = self._save_sample(
                    self.data_root,
                    self._calibration_payload(
                        session_id,
                        run_id,
                        row,
                        runtime,
                        observed_contract,
                        image_data,
                    ),
                )
            except Exception as exc:
                recovered = self._recover_saved_manifest_row(
                    session_id,
                    expected_index=int(prepared["expected_manifest_index"]),
                    frame_sha256=frame_sha256,
                    row=row,
                    runtime=runtime,
                    observed_contract=observed_contract,
                    require_present=False,
                )
                if recovered is None:
                    return self._hard_error(
                        "sample_store_hard_error",
                        str(exc),
                        server_spool_retry_available=True,
                    )
                existing = recovered
            else:
                existing = self._recover_saved_manifest_row(
                    session_id,
                    expected_index=int(prepared["expected_manifest_index"]),
                    frame_sha256=frame_sha256,
                    row=row,
                    runtime=runtime,
                    observed_contract=observed_contract,
                    require_present=True,
                )
                if status_code != 200 or response.get("ok") is not True:
                    if existing is None:
                        return self._hard_error(
                            "sample_store_hard_error",
                            str(response.get("error") or "save_sample failed"),
                            server_spool_retry_available=True,
                        )
        if existing is None:
            return self._hard_error(
                "sample_store_hard_error",
                "successful inference did not create a calibration manifest row",
                server_spool_retry_available=True,
            )
        if existing.get("face_detected", True) is not True:
            pending = self.store.mark_calibration_no_face_reclassification_pending(
                run_id,
                run_token,
                challenge_token,
                frame_sha256=frame_sha256,
                sample_index=int(existing["sample_index"]),
                manifest_record_sha256=_schedule.canonical_sha256(existing),
                cleanup_relative_paths={
                    "raw": existing.get("raw_path"),
                    "crop": existing.get("crop_path"),
                    "normalized_face": existing.get("normalized_face_path"),
                },
                observation=self._as_no_face_observation(success_observation),
            )
            self._complete_pending_no_face_reclassification(
                run_id,
                run_token,
                challenge_token,
                session_id=session_id,
                pending=pending,
                frame_sha256=frame_sha256,
            )
            return {
                "ok": True,
                "classification": "attributable_sensor_failure",
                "consumed": False,
                "retryable": False,
            }
        normalized_sha, crop_sha = self._saved_derived_hashes(
            session_id, existing
        )
        self.store.record_calibration_sample_saved(
            run_id,
            run_token,
            challenge_token,
            frame_sha256=frame_sha256,
            sample_index=int(existing["sample_index"]),
            manifest_record_sha256=_schedule.canonical_sha256(existing),
            raw_frame_sha256=_sha256_bytes(raw),
            normalized_face_sha256=normalized_sha,
            crop_sha256=crop_sha,
            face_detected=True,
        )
        return {"ok": True, "classification": "saved"}

    def _discard_unusable_manifest_row(
        self,
        session_id: str,
        *,
        expected_index: int,
        record: Mapping[str, Any],
        frame_sha256: str,
    ) -> None:
        session_dir = self._safe_session_dir(session_id)
        records = self._manifest_records(session_id)
        if len(records) != expected_index + 1 or records[-1] != record:
            raise MeasurementRunIntegrityError(
                "unusable manifest row is not the exact final dedicated row"
            )
        raw_path = (session_dir / str(record.get("raw_path") or "")).resolve()
        if raw_path.parent != (session_dir / "raw").resolve():
            raise MeasurementRunIntegrityError("unusable raw path is unsafe")
        if not raw_path.is_file() or _sha256_bytes(raw_path.read_bytes()) != frame_sha256:
            raise MeasurementRunIntegrityError(
                "unusable manifest raw frame binding changed"
            )
        targets: list[Path] = []
        for field, directory in (
            ("raw_path", "raw"),
            ("crop_path", "crop"),
            ("normalized_face_path", "normalized_face"),
        ):
            relative = record.get(field)
            if not relative:
                continue
            target = (session_dir / str(relative)).resolve()
            if target.parent != (session_dir / directory).resolve():
                raise MeasurementRunIntegrityError(
                    "unusable derived image path is unsafe"
                )
            targets.append(target)
        manifest_path = session_dir / "manifest.jsonl"
        payload = b"".join(
            json.dumps(item, ensure_ascii=False).encode("utf-8") + b"\n"
            for item in records[:-1]
        )
        # The manifest is the canonical training authority.  Remove its exact
        # final row atomically before deleting derived files so a crash can at
        # worst leave purgeable orphans, never a row that points at missing
        # training bytes.
        _atomic_bytes(manifest_path, payload)
        for target in targets:
            try:
                if target.is_file():
                    target.unlink()
            except OSError:
                # Finalize/abort performs an exact directory purge and verifies
                # absence.  The now-unreferenced file cannot enter training.
                pass

    def _complete_pending_no_face_reclassification(
        self,
        run_id: str,
        run_token: str,
        challenge_token: str,
        *,
        session_id: str,
        pending: Mapping[str, Any],
        frame_sha256: str,
    ) -> None:
        """Finish an irreversible no-face intent without preprocessing again."""

        expected_index = int(pending["expected_manifest_index"])
        records = self._manifest_records(session_id)
        if len(records) == expected_index + 1:
            record = records[-1]
            if (
                int(record.get("sample_index", -1)) != expected_index
                or record.get("face_detected", True) is not False
                or _schedule.canonical_sha256(record)
                != pending.get("manifest_record_sha256")
            ):
                raise MeasurementRunIntegrityError(
                    "pending no-face manifest record changed"
                )
            expected_paths = pending.get("cleanup_relative_paths")
            observed_paths = {
                "raw": record.get("raw_path"),
                "crop": record.get("crop_path"),
                "normalized_face": record.get("normalized_face_path"),
            }
            if expected_paths != observed_paths:
                raise MeasurementRunIntegrityError(
                    "pending no-face cleanup paths changed"
                )
            self._discard_unusable_manifest_row(
                session_id,
                expected_index=expected_index,
                record=record,
                frame_sha256=frame_sha256,
            )
        elif len(records) == expected_index:
            # A crash after atomic manifest truncation can leave only orphaned
            # derived files.  Remove solely the paths frozen in the intent.
            session_dir = self._safe_session_dir(session_id)
            for key, directory in (
                ("raw", "raw"),
                ("crop", "crop"),
                ("normalized_face", "normalized_face"),
            ):
                relative = pending["cleanup_relative_paths"].get(key)
                if relative is None:
                    continue
                target = (session_dir / str(relative)).resolve()
                if target.parent != (session_dir / directory).resolve():
                    raise MeasurementRunIntegrityError(
                        "pending no-face cleanup path is unsafe"
                    )
                try:
                    if target.is_file():
                        target.unlink()
                except OSError:
                    # The manifest is already authoritative and no longer
                    # references the file; terminal purge verifies absence.
                    pass
        else:
            raise MeasurementRunIntegrityError(
                "pending no-face manifest cardinality changed"
            )
        self.store.complete_calibration_no_face_reclassification(
            run_id,
            run_token,
            challenge_token,
            frame_sha256=frame_sha256,
        )

    @staticmethod
    def _as_no_face_observation(
        observation: Mapping[str, Any]
    ) -> dict[str, Any]:
        result = deepcopy(dict(observation))
        result["prediction_success"] = False
        for field in (
            "raw_gaze_pitch_yaw",
            "predicted_x_px",
            "predicted_y_px",
            "head_pose_pitch_yaw",
            "normalized_face_bbox",
        ):
            result[field] = None
        for field in (
            "prediction_covariance_px",
            "sensor_uncertainty_score",
            "blur_score",
            "exposure_score",
            "eye_scale",
            "pose_ood_score",
        ):
            result.pop(field, None)
        return result

    def _resume_prepared_attempt(
        self,
        inspected: Mapping[str, Any],
        run_token: str,
        challenge_token: str,
        *,
        image_data: str,
        observed_capture_contract: Mapping[str, Any],
    ) -> dict[str, Any]:
        prepared = inspected["prepared_attempt"]
        runtime = self._runtime(inspected)
        disposition = prepared["disposition"]
        try:
            raw = self.store.read_encrypted_frame_spool(
                str(inspected["capture_run_id"]), run_token
            )
        except MeasurementRunIntegrityError as exc:
            return self._fail_spool_integrity(
                str(inspected["capture_run_id"]), run_token, exc
            )
        if _sha256_bytes(raw) != prepared["frame_sha256"]:
            raise MeasurementRunIntegrityError(
                "prepared encrypted frame differs from observation"
            )
        recovered_image_data = "data:image/jpeg;base64," + base64.b64encode(
            raw
        ).decode("ascii")
        if inspected["block_role"] == "calibration" and disposition == "success":
            saved = self._persist_calibration_sample(
                inspected,
                run_token,
                challenge_token,
                image_data=recovered_image_data,
                raw=raw,
                frame_sha256=prepared["frame_sha256"],
                observed_contract=prepared["capture_contract_evidence"][
                    "observed_capture_contract"
                ],
                success_observation=prepared["observation"],
            )
            if saved.get("ok") is not True:
                return saved
            if saved.get("classification") == "attributable_sensor_failure":
                disposition = "no_face_detected"
        receipt = self.store.commit_prepared_observation(
            str(inspected["capture_run_id"]), run_token
        )
        self.store.delete_encrypted_frame_spool(
            str(inspected["capture_run_id"]), run_token
        )
        return self._committed_result(
            receipt, no_face=disposition == "no_face_detected"
        )

    def _calibration_payload(
        self,
        session_id: str,
        run_id: str,
        row: Mapping[str, Any],
        runtime: Mapping[str, Any],
        observed_contract: Mapping[str, Any],
        image_data: str,
    ) -> dict[str, Any]:
        motion_id = MOTION_BLOCKS.get(str(row["block_id"]))
        if motion_id is None:
            raise MeasurementRunIntegrityError("calibration block mapping changed")
        viewport = runtime["viewport"]
        contract = observed_contract
        return {
            "session_id": session_id,
            "image_data": image_data,
            "capture_contract": deepcopy(dict(observed_contract)),
            "phase": "calibration",
            "point_index": int(row["target_index"]),
            "repeat_index": int(row["repeat_index"]),
            "target_x": float(row["target_x_viewport_fraction"])
            * float(viewport["width"]),
            "target_y": float(row["target_y_viewport_fraction"])
            * float(viewport["height"]),
            "target_x_norm": float(row["target_x_norm"]),
            "target_y_norm": float(row["target_y_norm"]),
            "viewport_width": viewport["width"],
            "viewport_height": viewport["height"],
            "collect_mode": "motion_robust",
            "collection_protocol": CALIBRATION_PROTOCOL,
            "motion_block_id": motion_id,
            "posture_condition": row["posture"],
            "distance_condition": row["distance"],
            "lighting_condition": "ambient",
            "capture_burst_id": (
                f"{session_id}:{motion_id}:r{int(row['repeat_index'])}"
            ),
            "capture_run_id": run_id,
            "capture_source": CAPTURE_SOURCE,
            "calibration_label_authority": CALIBRATION_LABEL_AUTHORITY,
            "target_pixel_role": "server_frozen_measurement_target",
            "camera_width": contract["source_width_px"],
            "camera_height": contract["source_height_px"],
            "camera_frame_rate": (
                contract["source_frame_rate_hz"]
                or contract["intent_frame_rate_hz"]
            ),
        }

    def _recover_saved_manifest_row(
        self,
        session_id: str,
        *,
        expected_index: int,
        frame_sha256: str,
        row: Mapping[str, Any],
        runtime: Mapping[str, Any],
        observed_contract: Mapping[str, Any],
        require_present: bool,
    ) -> dict[str, Any] | None:
        records = self._manifest_records(session_id)
        if len(records) <= expected_index:
            if require_present:
                raise MeasurementRunIntegrityError(
                    "save_sample returned without its dedicated manifest row"
                )
            return None
        if len(records) != expected_index + 1:
            raise MeasurementRunIntegrityError(
                "dedicated calibration manifest contains an unexpected extra row"
            )
        record = records[expected_index]
        if int(record.get("sample_index", -1)) != expected_index:
            raise MeasurementRunIntegrityError("manifest sample index changed")
        expected_fields = {
            "phase": "calibration",
            "point_index": int(row["target_index"]),
            "repeat_index": int(row["repeat_index"]),
            "target_x_norm": float(row["target_x_norm"]),
            "target_y_norm": float(row["target_y_norm"]),
            "collection_protocol": CALIBRATION_PROTOCOL,
            "motion_block_id": MOTION_BLOCKS[str(row["block_id"])],
            "posture_condition": row["posture"],
            "distance_condition": row["distance"],
            "capture_run_id": row["capture_run_id"],
            "capture_source": CAPTURE_SOURCE,
            "calibration_label_authority": CALIBRATION_LABEL_AUTHORITY,
        }
        for field, expected in expected_fields.items():
            if record.get(field) != expected:
                raise MeasurementRunIntegrityError(
                    f"dedicated manifest field {field} changed"
                )
        if record.get("capture_contract") != observed_contract:
            raise MeasurementRunIntegrityError(
                "dedicated manifest capture contract changed"
            )
        raw_path = self._safe_session_dir(session_id) / str(record.get("raw_path") or "")
        raw_path = raw_path.resolve()
        if raw_path.parent != (self._safe_session_dir(session_id) / "raw").resolve():
            raise MeasurementRunIntegrityError("manifest raw path is unsafe")
        if not raw_path.is_file() or _sha256_bytes(raw_path.read_bytes()) != frame_sha256:
            raise MeasurementRunIntegrityError("manifest raw frame binding changed")
        return record

    def _saved_derived_hashes(
        self, session_id: str, record: Mapping[str, Any]
    ) -> tuple[str, str]:
        session_dir = self._safe_session_dir(session_id)
        normalized = (session_dir / str(
            record.get("normalized_face_path") or ""
        )).resolve()
        crop = (session_dir / str(record.get("crop_path") or "")).resolve()
        if normalized.parent != (session_dir / "normalized_face").resolve():
            raise MeasurementRunIntegrityError("normalized-face path is unsafe")
        if crop.parent != (session_dir / "crop").resolve():
            raise MeasurementRunIntegrityError("face-crop path is unsafe")
        if not normalized.is_file() or not crop.is_file():
            raise MeasurementRunIntegrityError(
                "usable calibration row lacks derived face images"
            )
        return (
            _sha256_bytes(normalized.read_bytes()),
            _sha256_bytes(crop.read_bytes()),
        )

    def _manifest_records(self, session_id: str) -> list[dict[str, Any]]:
        path = self._safe_session_dir(session_id) / "manifest.jsonl"
        if not path.exists():
            return []
        payload = path.read_bytes()
        if payload and not payload.endswith(b"\n"):
            raise MeasurementRunIntegrityError("dedicated manifest is incomplete")
        result: list[dict[str, Any]] = []
        for raw in payload.splitlines():
            try:
                item = json.loads(raw.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise MeasurementRunIntegrityError(
                    "dedicated manifest row is unreadable"
                ) from exc
            if not isinstance(item, dict):
                raise MeasurementRunIntegrityError(
                    "dedicated manifest row must be an object"
                )
            result.append(item)
        return result

    def _safe_session_dir(self, session_id: str) -> Path:
        sessions = (self.data_root / "data" / "sessions").resolve()
        candidate = (sessions / session_id).resolve()
        if candidate.parent != sessions or not candidate.is_dir():
            raise MeasurementRunIntegrityError(
                "dedicated calibration session is missing or unsafe"
            )
        return candidate

    def _verified_training_manifest(
        self,
        session_id: str,
        capture_run_id: str,
        runtime: Mapping[str, Any],
    ) -> tuple[
        list[dict[str, Any]],
        str,
        list[str],
        list[dict[str, Any]],
    ]:
        records = self._manifest_records(session_id)
        if len(records) != 65:
            raise MeasurementRunIntegrityError(
                "training manifest must contain exactly 65 usable rows"
            )
        image_bindings: list[dict[str, Any]] = []
        session_dir = self._safe_session_dir(session_id)
        for index, record in enumerate(records):
            if record.get("face_detected", True) is not True:
                raise MeasurementRunIntegrityError(
                    "training manifest contains an unusable face row"
                )
            if record.get("capture_run_id") != capture_run_id:
                raise MeasurementRunIntegrityError(
                    "training manifest capture run changed"
                )
            if record.get("collection_protocol") != CALIBRATION_PROTOCOL:
                raise MeasurementRunIntegrityError(
                    "training manifest collection protocol changed"
                )
            try:
                comparison = compare_capture_contracts(
                    runtime["capture_contract"], record.get("capture_contract")
                )
            except ValueError as exc:
                raise MeasurementRunIntegrityError(
                    "training manifest capture contract is invalid"
                ) from exc
            if comparison.get("compatible") is not True:
                raise MeasurementRunIntegrityError(
                    "training manifest capture contract is incompatible"
                )
            if int(record.get("sample_index", -1)) != index:
                raise MeasurementRunIntegrityError(
                    "training manifest indices are not exact"
                )
            normalized_path = (session_dir / str(
                record.get("normalized_face_path") or ""
            )).resolve()
            crop_path = (session_dir / str(record.get("crop_path") or "")).resolve()
            raw_path = (session_dir / str(record.get("raw_path") or "")).resolve()
            if normalized_path.parent != (session_dir / "normalized_face").resolve():
                raise MeasurementRunIntegrityError(
                    "normalized training image path is unsafe"
                )
            if raw_path.parent != (session_dir / "raw").resolve():
                raise MeasurementRunIntegrityError("raw training image path is unsafe")
            if crop_path.parent != (session_dir / "crop").resolve():
                raise MeasurementRunIntegrityError("crop training image path is unsafe")
            if (
                not normalized_path.is_file()
                or not raw_path.is_file()
                or not crop_path.is_file()
            ):
                raise MeasurementRunIntegrityError("training image is missing")
            image_bindings.append(
                {
                    "sample_index": index,
                    "normalized_face_path": normalized_path.relative_to(
                        session_dir
                    ).as_posix(),
                    "normalized_face_sha256": _sha256_bytes(
                        normalized_path.read_bytes()
                    ),
                    "raw_path": raw_path.relative_to(session_dir).as_posix(),
                    "raw_frame_sha256": _sha256_bytes(raw_path.read_bytes()),
                    "crop_path": crop_path.relative_to(session_dir).as_posix(),
                    "crop_sha256": _sha256_bytes(crop_path.read_bytes()),
                }
            )
        manifest_path = self._safe_session_dir(session_id) / "manifest.jsonl"
        return (
            records,
            _sha256_bytes(manifest_path.read_bytes()),
            [_schedule.canonical_sha256(item) for item in records],
            image_bindings,
        )

    @staticmethod
    def _assert_manifest_matches_store_binding(
        training_binding: Mapping[str, Any],
        *,
        row_hashes: Sequence[str],
        image_bindings: Sequence[Mapping[str, Any]],
    ) -> None:
        store_bindings = training_binding.get("calibration_manifest_bindings")
        if not isinstance(store_bindings, list) or len(store_bindings) != 65:
            raise MeasurementRunIntegrityError(
                "persistent calibration manifest binding is incomplete"
            )
        for index, (stored, row_sha, image) in enumerate(
            zip(store_bindings, row_hashes, image_bindings, strict=True)
        ):
            expected = {
                "sample_index": index,
                "ordinal": index,
                "manifest_record_sha256": row_sha,
                "raw_frame_sha256": image["raw_frame_sha256"],
                "normalized_face_sha256": image["normalized_face_sha256"],
                "crop_sha256": image["crop_sha256"],
            }
            for field, value in expected.items():
                if stored.get(field) != value:
                    raise MeasurementRunIntegrityError(
                        f"training input binding {field} changed at row {index}"
                    )

    def _training_provenance(
        self,
        capture_run_id: str,
        status: Mapping[str, Any],
        *,
        manifest_sha: str,
        manifest_row_sha256s: Sequence[str],
        image_bindings: Sequence[Mapping[str, Any]],
        training_binding: Mapping[str, Any],
        measurement_training_binding: Mapping[str, Any],
    ) -> dict[str, Any]:
        runtime = status["runner"]["runtime_binding"]
        manifest = _schedule.build_run_manifest(
            capture_run_id,
            protocol_path=self.code_root / _schedule.PROTOCOL_RELATIVE_PATH,
        )
        calibration_rows = [
            row for row in manifest["rows"] if row["block_role"] == "calibration"
        ]
        evaluation_rows = [
            row for row in manifest["rows"] if row["block_role"] == "evaluation"
        ]
        calibration_target_ids = sorted({row["target_id"] for row in calibration_rows})
        evaluation_target_ids = sorted({row["target_id"] for row in evaluation_rows})
        if set(calibration_target_ids) & set(evaluation_target_ids):
            raise MeasurementRunIntegrityError(
                "calibration and evaluation target identities overlap"
            )
        manifest_bindings = deepcopy(
            training_binding["calibration_manifest_bindings"]
        )
        return {
            "schema_version": 1,
            "provenance_type": "webcam_gaze_measurement_ceiling_training_v1",
            "capture_run_id": capture_run_id,
            "protocol_sha256": status["protocol_sha256"],
            "manifest_sha256": status["manifest_sha256"],
            "calibration_ledger_sha256": status["ledgers"]["calibration"][
                "sealed_sha256"
            ],
            "calibration_session_id": runtime["calibration_session_id"],
            "capture_contract_sha256": runtime["capture_contract_sha256"],
            "base_inference_bundle": deepcopy(runtime["base_inference_bundle"]),
            "training_consumed_base_inference_bundle": deepcopy(
                measurement_training_binding["base_inference_bundle"]
            ),
            "post_training_base_inference_bundle_verified": True,
            "base_bundle_checks": deepcopy(
                status["runner"]["base_bundle_checks"]
            ),
            "calibration_manifest_sha256": manifest_sha,
            "calibration_manifest_row_sha256s": list(manifest_row_sha256s),
            "calibration_manifest_bindings": manifest_bindings,
            "calibration_manifest_bindings_sha256": _schedule.canonical_sha256(
                manifest_bindings
            ),
            "training_image_bindings": deepcopy(list(image_bindings)),
            "training_image_bindings_sha256": _schedule.canonical_sha256(
                image_bindings
            ),
            "measurement_training_binding_sha256": measurement_training_binding[
                "binding_sha256"
            ],
            "consumed_training_rows_sha256": measurement_training_binding[
                "rows_sha256"
            ],
            "calibration_ordinals": list(range(65)),
            "training_role": "calibration_only",
            "calibration_schedule_rows_sha256": _schedule.canonical_sha256(
                calibration_rows
            ),
            "evaluation_schedule_rows_sha256": _schedule.canonical_sha256(
                evaluation_rows
            ),
            "calibration_target_ids_sha256": _schedule.canonical_sha256(
                calibration_target_ids
            ),
            "evaluation_target_ids_sha256": _schedule.canonical_sha256(
                evaluation_target_ids
            ),
            "train_samples": 65,
            "allow_cuda": False,
            "training_device_required": "cpu",
            "collection_protocol": CALIBRATION_PROTOCOL,
            "evaluation_labels_used": False,
            "evaluation_rows_used": 0,
            "evaluation_targets_excluded": True,
            "calibration_evaluation_target_intersection_count": 0,
            "text_cursor_cognitive_inputs_used": False,
            "measurement_claim_authorized": False,
        }

    @staticmethod
    def _measurement_training_binding(
        *,
        capture_run_id: str,
        session_id: str,
        manifest_sha256: str,
        store_binding: Mapping[str, Any],
        image_bindings: Sequence[Mapping[str, Any]],
        base_inference_bundle: Mapping[str, Any],
    ) -> dict[str, Any]:
        persisted = store_binding["calibration_manifest_bindings"]
        rows = [
            {
                "sequence_index": index,
                "manifest_sample_index": index,
                "manifest_record_sha256": persisted[index][
                    "manifest_record_sha256"
                ],
                "frame_sha256": persisted[index]["frame_sha256"],
                "normalized_face_path": image_bindings[index][
                    "normalized_face_path"
                ],
                "normalized_face_sha256": persisted[index][
                    "normalized_face_sha256"
                ],
            }
            for index in range(65)
        ]
        binding: dict[str, Any] = {
            "schema_version": 1,
            "binding_type": (
                "webcam_gaze_measurement_ceiling_training_input_binding_v1"
            ),
            "data_session_id": session_id,
            "capture_run_id": capture_run_id,
            "manifest_sha256": manifest_sha256,
            "base_inference_bundle": {
                field: base_inference_bundle[field]
                for field in (
                    "model_id",
                    "model_name",
                    "model_sha256",
                    "bundle_sha256",
                    "checkpoint_sha256",
                )
            },
            "rows": rows,
            "rows_sha256": _schedule.canonical_sha256(rows),
        }
        binding["binding_sha256"] = _schedule.canonical_sha256(binding)
        return binding

    @staticmethod
    def _validate_training_response(
        response: Mapping[str, Any],
        model_name: str,
        *,
        expected_consumed_sha256: str,
        expected_binding_sha256: str,
        expected_base_inference_bundle: Mapping[str, Any],
    ) -> None:
        if response.get("model_name") != model_name:
            raise MeasurementRunIntegrityError("training returned a different model")
        if response.get("train_samples") != 65:
            raise MeasurementRunIntegrityError("training did not use exactly 65 rows")
        if response.get("training_device") != "cpu":
            raise MeasurementRunIntegrityError("training was not CPU-only")
        uncertainty = response.get("uncertainty_v2")
        if not isinstance(uncertainty, Mapping) or uncertainty.get("status") != (
            "scored_no_threshold"
        ):
            raise MeasurementRunIntegrityError(
                "training response lacks uncertainty_v2"
            )
        if response.get("consumed_training_rows_sha256") != expected_consumed_sha256:
            raise MeasurementRunIntegrityError(
                "training response consumed a different byte-bound input set"
            )
        if response.get(
            "measurement_training_binding_sha256"
        ) != expected_binding_sha256:
            raise MeasurementRunIntegrityError(
                "training response used a different input binding"
            )
        if response.get("base_inference_bundle") != expected_base_inference_bundle:
            raise MeasurementRunIntegrityError(
                "training response consumed a different base inference bundle"
            )

    @staticmethod
    def _load_model_artifact(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MeasurementRunIntegrityError("model artifact is unreadable") from exc
        if not isinstance(payload, dict):
            raise MeasurementRunIntegrityError("model artifact must be an object")
        return payload

    @staticmethod
    def _validate_final_model_artifact(
        artifact: Mapping[str, Any],
        *,
        provenance_sha: str,
        expected_binding_sha256: str,
        expected_rows_sha256: str,
        capture_run_id: str,
        expected_base_inference_bundle: Mapping[str, Any],
    ) -> None:
        if artifact.get("train_samples") != 65 or artifact.get("training_device") != "cpu":
            raise MeasurementRunIntegrityError("final model training contract changed")
        uncertainty = artifact.get("uncertainty_v2")
        if not isinstance(uncertainty, Mapping) or uncertainty.get("status") != (
            "scored_no_threshold"
        ):
            raise MeasurementRunIntegrityError("final model lacks uncertainty_v2")
        provenance = artifact.get("measurement_ceiling_provenance")
        if not isinstance(provenance, Mapping) or _schedule.canonical_sha256(
            provenance
        ) != provenance_sha:
            raise MeasurementRunIntegrityError("final model provenance changed")
        consumed = artifact.get("measurement_training_input_binding")
        expected_consumed = {
            "binding_sha256": expected_binding_sha256,
            "rows_sha256": expected_rows_sha256,
            "row_count": 65,
            "capture_run_id": capture_run_id,
            "base_inference_bundle": deepcopy(
                dict(expected_base_inference_bundle)
            ),
        }
        if consumed != expected_consumed:
            raise MeasurementRunIntegrityError(
                "final model consumed-input binding changed"
            )
        if (
            provenance.get("measurement_training_binding_sha256")
            != expected_binding_sha256
            or provenance.get("consumed_training_rows_sha256")
            != expected_rows_sha256
        ):
            raise MeasurementRunIntegrityError(
                "final model provenance differs from consumed training bytes"
            )

    def _matching_owned_artifact(
        self, path: Path, provenance_sha: str, capture_run_id: str
    ) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        artifact = self._load_model_artifact(path)
        provenance = artifact.get("measurement_ceiling_provenance")
        if not isinstance(provenance, Mapping):
            return None
        if provenance.get("capture_run_id") != capture_run_id:
            raise MeasurementRunIntegrityError(
                "deterministic model path is owned by a different run"
            )
        if _schedule.canonical_sha256(provenance) != provenance_sha:
            raise MeasurementRunIntegrityError(
                "existing run model has different training provenance"
            )
        self._validate_final_model_artifact(
            artifact,
            provenance_sha=provenance_sha,
            expected_binding_sha256=str(
                provenance.get("measurement_training_binding_sha256") or ""
            ),
            expected_rows_sha256=str(
                provenance.get("consumed_training_rows_sha256") or ""
            ),
            capture_run_id=capture_run_id,
            expected_base_inference_bundle=provenance.get(
                "training_consumed_base_inference_bundle"
            )
            or {},
        )
        return artifact

    @staticmethod
    def _personal_model_name(capture_run_id: str) -> str:
        return "wgmc_" + "".join(
            character.lower() if character.isalnum() else "_"
            for character in capture_run_id
        )

    def _rehash_personal_model(self, inspected: Mapping[str, Any]) -> bool:
        binding = inspected.get("model_binding")
        runtime = inspected.get("runtime_binding")
        if not isinstance(binding, Mapping) or not isinstance(runtime, Mapping):
            return False
        path = self._model_path(self.data_root, str(binding["model_id"])).resolve()
        return path.is_file() and _sha256_bytes(path.read_bytes()) == binding.get(
            "model_sha256"
        )

    def _fail_evaluation_model_integrity(
        self,
        inspected: Mapping[str, Any],
        run_token: str,
        *,
        code: str,
        detail: str,
    ) -> None:
        run_id = str(inspected["capture_run_id"])
        runtime = self._runtime(inspected)
        try:
            self.store.delete_encrypted_frame_spool(run_id, run_token)
        except Exception:
            pass
        try:
            self._delete_owned_model_verified(
                self._model_path(
                    self.data_root,
                    self._personal_model_name(run_id),
                ).resolve(),
                run_id,
                expected_session_id=str(runtime["calibration_session_id"]),
            )
        except Exception:
            pass
        self.store.fail_integrity(
            run_id,
            run_token,
            code=code,
            detail=detail,
        )

    def _purge_and_record(
        self,
        capture_run_id: str,
        run_token: str,
        status: Mapping[str, Any],
    ) -> None:
        runtime = status["runner"]["runtime_binding"]
        session_id = str(runtime["calibration_session_id"])
        response = self._purge_session_images_verified(session_id)
        spool = self.store.delete_encrypted_frame_spool(
            capture_run_id, run_token
        )
        if spool.get("absence_verified") is not True:
            raise MeasurementRunIntegrityError(
                "encrypted frame spool remains after calibration"
            )
        self.store.record_calibration_image_purge(
            capture_run_id,
            run_token,
            calibration_session_id=session_id,
            removed_directories=list(response.get("removed_directories") or []),
            postcondition_verified=True,
        )

    def _purge_session_images_verified(
        self, session_id: str
    ) -> Mapping[str, Any]:
        response = self._purge(self.data_root, session_id)
        if response.get("ok") is not True:
            raise MeasurementRunIntegrityError("dedicated image purge failed")
        session_dir = self._safe_session_dir(session_id)
        absent = all(
            not (session_dir / name).exists()
            for name in DEDICATED_IMAGE_DIRECTORIES
        )
        if not absent:
            raise MeasurementRunIntegrityError(
                "dedicated calibration images remain after purge"
            )
        return response

    def _purge_images_best_effort(self, session_id: str) -> None:
        try:
            self._purge(self.data_root, session_id)
        except Exception:
            pass

    def _delete_owned_model(self, path: Path, capture_run_id: str) -> None:
        if not path.is_file():
            return
        try:
            artifact = self._load_model_artifact(path)
        except MeasurementRunIntegrityError:
            return
        provenance = artifact.get("measurement_ceiling_provenance")
        if isinstance(provenance, Mapping) and provenance.get(
            "capture_run_id"
        ) == capture_run_id:
            path.unlink()

    def _delete_owned_model_verified(
        self,
        path: Path,
        capture_run_id: str,
        *,
        expected_session_id: str | None,
    ) -> None:
        expected_parent = (self.data_root / "examples" / "models").resolve()
        if path.parent != expected_parent or path.suffix != ".json":
            raise MeasurementRunIntegrityError(
                "dedicated personalized-model path is unsafe"
            )
        if not path.exists():
            return
        artifact = self._load_model_artifact(path)
        provenance = artifact.get("measurement_ceiling_provenance")
        proven = isinstance(provenance, Mapping) and provenance.get(
            "capture_run_id"
        ) == capture_run_id
        deterministic_unbound = (
            provenance is None
            and artifact.get("name") == self._personal_model_name(capture_run_id)
            and expected_session_id is not None
            and artifact.get("data_session_id") == expected_session_id
        )
        if not proven and not deterministic_unbound:
            raise MeasurementRunIntegrityError(
                "personalized model is not owned by this measurement run"
            )
        path.unlink()
        if path.exists():
            raise MeasurementRunIntegrityError(
                "dedicated personalized model could not be removed"
            )

    @staticmethod
    def _hard_error(
        classification: str,
        detail: str,
        *,
        exact_frame_retry_required: bool = False,
        server_spool_retry_available: bool = False,
        abort_required: bool = False,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "classification": classification,
            "detail": detail[:512],
            "consumed": False,
            "retryable": not abort_required,
            "exact_frame_retry_required": exact_frame_retry_required,
            "server_spool_retry_available": server_spool_retry_available,
            "new_frame_retry_allowed": (
                not exact_frame_retry_required
                and not server_spool_retry_available
                and not abort_required
            ),
            "abort_required": abort_required,
            "inference_replayed": False,
            "measurement_claim_authorized": False,
            "physical_capture_claim_authorized": False,
        }

    def _fail_spool_integrity(
        self,
        capture_run_id: str,
        run_token: str,
        error: Exception,
    ) -> dict[str, Any]:
        """Poisoned encrypted replay state is terminal, never retryable."""

        self.store.fail_integrity(
            capture_run_id,
            run_token,
            code="encrypted_frame_spool_integrity_failed",
            detail=str(error),
        )
        failed = self.store.get_status(capture_run_id, run_token)
        if failed.get("phase") != "failed_integrity":
            raise MeasurementRunIntegrityError(
                "spool integrity failure did not become durable"
            )
        cleanup = (failed.get("failure") or {}).get("cleanup") or {}
        return {
            **self._hard_error(
                "encrypted_frame_spool_integrity_failed",
                str(error),
                abort_required=True,
            ),
            "phase": "failed_integrity",
            "terminal": True,
            "cleanup_verified": bool(cleanup.get("cleanup_verified")),
        }

    def _fail_unsealed_inference_intent(
        self, capture_run_id: str, run_token: str
    ) -> dict[str, Any]:
        """Never repeat a predictor call whose returned result was not sealed."""

        detail = (
            "predictor execution began but no durable response observation exists; "
            "repeating the model call is forbidden"
        )
        self.store.fail_integrity(
            capture_run_id,
            run_token,
            code="unsealed_inference_result_after_restart",
            detail=detail,
        )
        failed = self.store.get_status(capture_run_id, run_token)
        if failed.get("phase") != "failed_integrity":
            raise MeasurementRunIntegrityError(
                "unsealed inference intent did not become durable failure"
            )
        cleanup = (failed.get("failure") or {}).get("cleanup") or {}
        return {
            **self._hard_error(
                "unsealed_inference_result_after_restart",
                detail,
                abort_required=True,
            ),
            "phase": "failed_integrity",
            "terminal": True,
            "cleanup_verified": bool(cleanup.get("cleanup_verified")),
        }

    @staticmethod
    def _committed_result(
        receipt: Mapping[str, Any], *, no_face: bool
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "classification": (
                "attributable_sensor_failure" if no_face else "success"
            ),
            "consumed": True,
            "retryable": False,
            "prediction_success": not no_face,
            "receipt": deepcopy(dict(receipt)),
            "measurement_claim_authorized": False,
            "physical_capture_claim_authorized": False,
        }


__all__ = [
    "BASE_INFERENCE_SELECTOR",
    "CALIBRATION_PROTOCOL",
    "MeasurementRunner",
    "MeasurementRunnerError",
]

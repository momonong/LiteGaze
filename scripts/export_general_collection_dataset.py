"""Export completed rehearsal sessions into a versioned private analysis bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.participant_study.general_collection import (
    load_general_bank,
    load_general_protocol,
    validate_general_design,
)
from core.participant_study.protocol import load_protocol


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_cell(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def export_bundle(
    root: Path,
    output: Path,
    *,
    include_incomplete: bool = False,
) -> dict[str, Any]:
    general_protocol = load_general_protocol()
    bank = load_general_bank()
    design = validate_general_design(general_protocol, bank)
    consent_protocol = load_protocol()
    rehearsal_root = (
        root
        / "data"
        / "participant_studies"
        / consent_protocol["protocol_id"]
        / "rehearsals"
    )
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory already exists and is not empty")
    output.mkdir(parents=True, exist_ok=True)

    participants: dict[str, dict[str, Any]] = {}
    session_rows: list[dict[str, Any]] = []
    passage_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    layout_rows: list[dict[str, Any]] = []
    telemetry_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    reading_video_rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    storage_security_modes: set[str] = set()
    retention_policies: set[str] = set()

    for session_path in sorted(rehearsal_root.glob("ST-*/session.json")):
        session = json.loads(session_path.read_text(encoding="utf-8"))
        session_id = str(session.get("study_session_id") or session_path.parent.name)
        if session.get("state") == "withdrawn":
            excluded.append({"study_session_id": session_id, "reason": "withdrawn"})
            continue
        if not include_incomplete and session.get("state") != "completed":
            excluded.append({"study_session_id": session_id, "reason": "incomplete"})
            continue
        assignment = dict(session.get("collection_assignment") or {})
        collection = dict(session.get("general_collection") or {})
        if not assignment or not collection.get("assessment_id"):
            excluded.append({"study_session_id": session_id, "reason": "not_general_collection"})
            continue
        if assignment.get("protocol_sha256") != design["protocol_sha256"]:
            raise ValueError(f"session {session_id} has a different protocol digest")
        if assignment.get("bank_sha256") != design["bank_sha256"]:
            raise ValueError(f"session {session_id} has a different bank digest")
        participant_id = str(session["participant_id"])
        governance = dict(session.get("data_governance") or {})
        storage_security_modes.add(
            str(governance.get("storage_security") or "legacy_unspecified")
        )
        retention_policies.add(
            str(governance.get("retention_policy") or "legacy_unspecified")
        )
        profile = dict(collection.get("profile") or {})
        participant_row = {
            "participant_id": participant_id,
            "pair_id": assignment.get("pair_id"),
            "schedule_cell": assignment.get("schedule_cell"),
            "sequence": assignment.get("sequence"),
            **profile,
        }
        existing_participant = participants.get(participant_id)
        if existing_participant and existing_participant != participant_row:
            raise ValueError(f"participant {participant_id} changed frozen profile or assignment")
        participants[participant_id] = participant_row
        system_check = dict(
            dict(session.get("quality") or {}).get("general_system_check") or {}
        )
        device = dict(system_check.get("device", {}))
        quality = dict(dict(session.get("quality") or {}).get("general_collection") or {})
        session_rows.append(
            {
                "participant_id": participant_id,
                "study_session_id": session_id,
                "visit_index": assignment.get("visit_index"),
                "capture_session_id": dict(session.get("linked_data") or {}).get("gaze_session_id"),
                "form_id": assignment.get("form_id"),
                "order_cell": assignment.get("order_cell"),
                "state": session.get("state"),
                "storage_security": governance.get("storage_security"),
                "retention_policy": governance.get("retention_policy"),
                "self_only": governance.get("self_only"),
                "formal_promotion_allowed": False,
                "created_at_utc": session.get("created_at_utc"),
                "completed_at_utc": next(
                    (
                        event.get("at_utc")
                        for event in reversed(session.get("events", []))
                        if event.get("event") == "general_collection_completed"
                    ),
                    None,
                ),
                "device_class": device.get("device_class"),
                "browser_family": device.get("browser_family"),
                "viewport_width": device.get("viewport_width"),
                "viewport_height": device.get("viewport_height"),
                "camera_width": device.get("camera_width"),
                "camera_height": device.get("camera_height"),
                "estimated_camera_fps_band": device.get("estimated_camera_fps_band"),
                "gaze_quality_band": quality.get("gaze_quality_band"),
                "median_spatial_error_px": quality.get("median_spatial_error_px"),
                "p90_spatial_error_px": quality.get("p90_spatial_error_px"),
                "precision_rms_px": quality.get("precision_rms_px"),
                "prediction_success_fraction": quality.get("prediction_success_fraction"),
                "effective_sampling_hz": quality.get("effective_sampling_hz"),
                "head_pose_range": _json_cell(quality.get("head_pose_range")),
                "face_scale_range": quality.get("face_scale_range"),
                "drift_change_px": quality.get("drift_change_px"),
            }
        )

        for round_path in sorted((session_path.parent / "collection" / "rounds").glob("R*.json")):
            observation = json.loads(round_path.read_text(encoding="utf-8"))
            base = {
                "participant_id": participant_id,
                "study_session_id": session_id,
                "visit_index": assignment.get("visit_index"),
                "form_id": assignment.get("form_id"),
                "round_number": observation.get("round_number"),
                "passage_id": observation.get("passage_id"),
                "passage_family_id": observation.get("passage_family_id"),
            }
            report = dict(observation.get("passage_self_report") or {})
            passage_rows.append(
                {
                    **base,
                    "difficulty_band": observation.get("difficulty_band"),
                    "reading_elapsed_ms": observation.get("reading_elapsed_ms"),
                    "scroll_occurred": observation.get("scroll_occurred"),
                    "zoom_ratio": observation.get("zoom_ratio"),
                    "understanding": report.get("understanding"),
                    "mental_effort": report.get("mental_effort"),
                    "read_complete": report.get("read_complete"),
                    "interrupted": report.get("interrupted"),
                    "word_layout_sha256": observation.get("word_layout_sha256"),
                    "probe_order_sha256": observation.get("probe_order_sha256"),
                }
            )
            for order_index, review in enumerate(observation.get("word_reviews", [])):
                review_rows.append(
                    {
                        **base,
                        "probe_order_index": order_index,
                        "probe_id": review.get("probe_id"),
                        "surface": review.get("surface"),
                        "stratum": review.get("stratum"),
                        "label": review.get("label"),
                    }
                )
            for layout in observation.get("word_layout", []):
                layout_rows.append({**base, **layout})

        telemetry_root = session_path.parent / "collection" / "telemetry"
        for batch_path in sorted(telemetry_root.glob("*/*.json")):
            batch = json.loads(batch_path.read_text(encoding="utf-8"))
            for sample_index, sample in enumerate(batch.get("samples", [])):
                telemetry_rows.append(
                    {
                        "participant_id": participant_id,
                        "study_session_id": session_id,
                        "visit_index": assignment.get("visit_index"),
                        "capture_session_id": batch.get("capture_session_id"),
                        "passage_id": batch.get("passage_id"),
                        "batch_id": batch.get("batch_id"),
                        "sample_index": sample_index,
                        "monotonic_elapsed_ms": sample.get("monotonic_elapsed_ms"),
                        "prediction_success": sample.get("prediction_success"),
                        "coarse_failure_code": sample.get("coarse_failure_code"),
                        "screen_xy_norm": _json_cell(sample.get("screen_xy_norm")),
                        "screen_xy_px": _json_cell(sample.get("screen_xy_px")),
                        "gaze_pitch_yaw": _json_cell(sample.get("gaze_pitch_yaw")),
                        "head_pose_pitch_yaw": _json_cell(sample.get("head_pose_pitch_yaw")),
                        "normalized_face_bbox": _json_cell(sample.get("normalized_face_bbox")),
                        "nearest_word_index": sample.get("nearest_word_index"),
                        "viewport": _json_cell(batch.get("viewport")),
                    }
                )
        for phase, summary in dict(collection.get("validations") or {}).items():
            for sample_index, sample in enumerate(dict(summary).get("samples", [])):
                validation_rows.append(
                    {
                        "participant_id": participant_id,
                        "study_session_id": session_id,
                        "visit_index": assignment.get("visit_index"),
                        "phase": phase,
                        "sample_index": sample_index,
                        **sample,
                    }
                )

        reading_video_root = session_path.parent / "collection" / "reading_video"
        for metadata_path in sorted(reading_video_root.glob("R*.json")):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            media_paths = [
                candidate
                for candidate in metadata_path.parent.glob(f"{metadata_path.stem}.*")
                if candidate.suffix.lower() != ".json"
            ]
            if len(media_paths) != 1:
                raise ValueError(
                    f"session {session_id} reading video {metadata_path.stem} "
                    "does not have exactly one media file"
                )
            media_path = media_paths[0]
            if media_path.stat().st_size != int(metadata.get("bytes", -1)):
                raise ValueError(f"reading video size mismatch: {media_path}")
            if _sha256(media_path) != metadata.get("sha256"):
                raise ValueError(f"reading video digest mismatch: {media_path}")
            reading_video_rows.append(
                {
                    "participant_id": participant_id,
                    "study_session_id": session_id,
                    "visit_index": assignment.get("visit_index"),
                    "round_number": metadata.get("round_number"),
                    "passage_id": metadata.get("passage_id"),
                    "recording_id": metadata.get("recording_id"),
                    "duration_ms": metadata.get("duration_ms"),
                    "mime_type": metadata.get("mime_type"),
                    "bytes": metadata.get("bytes"),
                    "sha256": metadata.get("sha256"),
                    "video_track_count": metadata.get("video_track_count"),
                    "audio_track_count": metadata.get("audio_track_count"),
                    "storage_security": metadata.get("storage_security"),
                    "dataset_role": metadata.get("dataset_role"),
                    "source_relative_path": media_path.relative_to(root).as_posix(),
                }
            )

    tables: dict[str, tuple[list[str], list[dict[str, Any]]]] = {
        "participants.csv": (
            [
                "participant_id", "pair_id", "schedule_cell", "sequence", "english_l1",
                "english_age_of_acquisition_band", "weekly_english_reading_band",
                "vision_correction", "education_band",
            ],
            sorted(participants.values(), key=lambda row: str(row["participant_id"])),
        ),
        "sessions.csv": (
            [
                "participant_id", "study_session_id", "visit_index", "capture_session_id",
                "form_id", "order_cell", "state", "storage_security",
                "retention_policy", "self_only", "formal_promotion_allowed",
                "created_at_utc", "completed_at_utc",
                "device_class", "browser_family", "viewport_width", "viewport_height",
                "camera_width", "camera_height", "estimated_camera_fps_band",
                "gaze_quality_band", "median_spatial_error_px", "p90_spatial_error_px",
                "precision_rms_px", "prediction_success_fraction", "effective_sampling_hz",
                "head_pose_range", "face_scale_range", "drift_change_px",
            ],
            session_rows,
        ),
        "passages.csv": (
            [
                "participant_id", "study_session_id", "visit_index", "form_id", "round_number",
                "passage_id", "passage_family_id", "difficulty_band", "reading_elapsed_ms",
                "scroll_occurred", "zoom_ratio", "understanding", "mental_effort",
                "read_complete", "interrupted", "word_layout_sha256", "probe_order_sha256",
            ],
            passage_rows,
        ),
        "word_reviews.csv": (
            [
                "participant_id", "study_session_id", "visit_index", "form_id", "round_number",
                "passage_id", "passage_family_id", "probe_order_index", "probe_id", "surface",
                "stratum", "label",
            ],
            review_rows,
        ),
        "word_layout.csv": (
            [
                "participant_id", "study_session_id", "visit_index", "form_id", "round_number",
                "passage_id", "passage_family_id", "word_index", "left_px", "top_px",
                "right_px", "bottom_px",
            ],
            layout_rows,
        ),
        "gaze_telemetry.csv": (
            [
                "participant_id", "study_session_id", "visit_index", "capture_session_id",
                "passage_id", "batch_id", "sample_index", "monotonic_elapsed_ms",
                "prediction_success", "coarse_failure_code", "screen_xy_norm", "screen_xy_px",
                "gaze_pitch_yaw", "head_pose_pitch_yaw", "normalized_face_bbox",
                "nearest_word_index", "viewport",
            ],
            telemetry_rows,
        ),
        "validation_samples.csv": (
            [
                "participant_id", "study_session_id", "visit_index", "phase", "sample_index",
                "target_id", "target_x_px", "target_y_px", "prediction_success",
                "predicted_x_px", "predicted_y_px", "spatial_error_px",
            ],
            validation_rows,
        ),
        "reading_video_index.csv": (
            [
                "participant_id", "study_session_id", "visit_index", "round_number",
                "passage_id", "recording_id", "duration_ms", "mime_type", "bytes",
                "sha256", "video_track_count", "audio_track_count", "storage_security",
                "dataset_role", "source_relative_path",
            ],
            reading_video_rows,
        ),
        "excluded_sessions.csv": (
            ["study_session_id", "reason"],
            excluded,
        ),
    }
    for filename, (fields, rows) in tables.items():
        _write_csv(output / filename, fields, rows)
    files = {
        filename: {
            "sha256": _sha256(output / filename),
            "row_count": len(rows),
        }
        for filename, (_, rows) in tables.items()
    }
    manifest = {
        "schema_version": 1,
        "exported_at_utc": datetime.now(UTC).isoformat(),
        "dataset_role": general_protocol["dataset_role"],
        "formal_promotion_allowed": False,
        "claim_boundary": general_protocol["claim_boundary"],
        "protocol_id": general_protocol["protocol_id"],
        "protocol_version": general_protocol["protocol_version"],
        "protocol_sha256": design["protocol_sha256"],
        "bank_id": bank["bank_id"],
        "bank_version": bank["bank_version"],
        "bank_sha256": design["bank_sha256"],
        "bank_status": bank["status"],
        "participant_count": len(participants),
        "session_count": len(session_rows),
        "files": files,
        "split_policy": {
            "same_participant_same_partition": True,
            "participant_holdout_required": True,
            "passage_family_holdout_required": True,
            "probe_holdout_required": True,
            "capture_session_and_device_holdout_required": True,
            "split_manifest_status": "not_created_by_exporter",
        },
        "privacy": {
            "direct_identifiers_exported": False,
            "raw_images_or_video_exported": False,
            "bundle_classification": "private_pseudonymous_research_data",
        },
        "source_reading_videos": {
            "present": bool(reading_video_rows),
            "count": len(reading_video_rows),
            "raw_media_files_exported": False,
            "index_exported": True,
            "dataset_role": "self_development_only_not_confirmation",
        },
        "storage_governance": {
            "security_modes": sorted(storage_security_modes),
            "retention_policies": sorted(retention_policies),
            "unencrypted_self_development_present": (
                "unencrypted_self_development" in storage_security_modes
            ),
            "formal_promotion_allowed": False,
        },
    }
    manifest_path = output / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-incomplete", action="store_true")
    args = parser.parse_args()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = args.root.resolve()
    output = (
        args.output or (root / "artifacts" / f"general_collection_{timestamp}")
    ).resolve()
    manifest = export_bundle(
        root,
        output,
        include_incomplete=args.include_incomplete,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output),
                "participant_count": manifest["participant_count"],
                "session_count": manifest["session_count"],
                "formal_promotion_allowed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

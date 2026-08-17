"""Audit the current participant export as a CHI pipeline smoke test.

This command deliberately does not fit a model or estimate an effect.  It only
checks that the current self-development export can supply the independent
word-review outcome and that ineligible gaze is routed to the text/person
fallback branch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_LABELS = ("no_review", "unsure", "review_needed")
EXPECTED_REVIEWS_PER_PASSAGE = 8


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _is_true(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"expected strict true/false boolean, received {value!r}")


def _required_columns(
    rows: list[dict[str, str]], required: Iterable[str], name: str
) -> None:
    if not rows:
        raise ValueError(f"{name} is empty")
    missing = set(required) - set(rows[0])
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _required_csv_header(path: Path, required: Iterable[str], name: str) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        fields = set(csv.DictReader(handle).fieldnames or ())
    missing = set(required) - fields
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def audit_export(export_dir: Path) -> dict[str, Any]:
    """Return a privacy-minimal, deterministic aggregate smoke result."""

    root = export_dir.resolve()
    review_path = root / "word_reviews.csv"
    session_path = root / "sessions.csv"
    manifest_path = root / "dataset_manifest.json"
    gaze_path = root / "gaze_telemetry.csv"
    validation_path = root / "validation_samples.csv"
    unverified_path = root / "reading_telemetry_unverified.csv"
    for path in (
        review_path,
        session_path,
        manifest_path,
        gaze_path,
        validation_path,
        unverified_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    reviews = _read_csv(review_path)
    sessions = _read_csv(session_path)
    gaze_rows = _read_csv(gaze_path)
    validation_rows = _read_csv(validation_path)
    unverified_rows = _read_csv(unverified_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("dataset manifest must be a JSON object")
    gaze_provenance = manifest.get("gaze_provenance")
    if not isinstance(gaze_provenance, Mapping):
        raise ValueError("dataset manifest gaze_provenance is unavailable")

    _required_columns(
        reviews,
        (
            "study_session_id",
            "passage_id",
            "passage_family_id",
            "probe_id",
            "surface",
            "stratum",
            "label",
        ),
        "word_reviews.csv",
    )
    _required_columns(
        sessions,
        (
            "study_session_id",
            "state",
            "formal_promotion_allowed",
            "gaze_quality_band",
            "gaze_export_status",
            "reading_gaze_export_eligible",
            "validation_gaze_export_eligible",
            "uncertainty_evidence_eligible",
        ),
        "sessions.csv",
    )
    _required_csv_header(
        gaze_path,
        (
            "evidence_status",
            "reading_gaze_export_eligible",
            "formal_evidence_eligible",
        ),
        "gaze_telemetry.csv",
    )
    _required_csv_header(
        validation_path,
        (
            "evidence_status",
            "validation_gaze_export_eligible",
            "formal_evidence_eligible",
        ),
        "validation_samples.csv",
    )
    _required_csv_header(
        unverified_path,
        (
            "evidence_status",
            "prediction_receipt_bound",
            "reading_gaze_export_eligible",
            "formal_evidence_eligible",
        ),
        "reading_telemetry_unverified.csv",
    )

    labels = Counter(row["label"] for row in reviews)
    unknown_labels = sorted(set(labels) - set(ALLOWED_LABELS))
    identities = [
        (row["study_session_id"], row["passage_id"], row["probe_id"])
        for row in reviews
    ]
    session_ids = [row["study_session_id"] for row in sessions]
    session_id_set = set(session_ids)
    review_session_ids = {row["study_session_id"] for row in reviews}
    session_passage_counts = Counter(
        (row["study_session_id"], row["passage_id"]) for row in reviews
    )
    passage_ids = {row["passage_id"] for row in reviews}
    family_count = len({row["passage_family_id"] for row in reviews})
    empty_required_fields = sum(
        not row[field].strip()
        for row in reviews
        for field in (
            "study_session_id",
            "passage_id",
            "passage_family_id",
            "probe_id",
            "surface",
            "stratum",
            "label",
        )
    )
    all_completed = all(row["state"] == "completed" for row in sessions)
    formal_promotion_disabled = all(
        not _is_true(row["formal_promotion_allowed"]) for row in sessions
    )
    gaze_rows_eligible = sum(
        _is_true(row["reading_gaze_export_eligible"]) for row in sessions
    )
    validation_rows_eligible = sum(
        _is_true(row["validation_gaze_export_eligible"]) for row in sessions
    )
    uncertainty_rows_eligible = sum(
        _is_true(row["uncertainty_evidence_eligible"]) for row in sessions
    )
    behavioral_only_sessions = sum(
        row["gaze_export_status"] == "behavioral_only" for row in sessions
    )
    unverified_rows_are_non_evidence = all(
        row.get("evidence_status") == "client_roundtrip_unverified"
        and not _is_true(row.get("prediction_receipt_bound"))
        and not _is_true(row.get("reading_gaze_export_eligible"))
        and not _is_true(row.get("formal_evidence_eligible"))
        for row in unverified_rows
    )
    manifest_counts_match = (
        int(gaze_provenance.get("eligible_reading_gaze_table_row_count", -1))
        == len(gaze_rows)
        and int(
            gaze_provenance.get(
                "client_roundtrip_unverified_reading_row_count", -1
            )
        )
        == len(unverified_rows)
    )
    manifest_separation_contract = (
        gaze_provenance.get("gaze_tables_contain_only_eligible_sessions") is True
        and gaze_provenance.get(
            "unverified_reading_telemetry_separate_from_eligible_gaze"
        )
        is True
        and gaze_provenance.get("legacy_or_unavailable_gaze_mixed_with_eligible")
        is False
    )

    gates = {
        "sessions_nonempty_and_completed": bool(sessions) and all_completed,
        "session_identity_unique_nonempty": len(session_ids) == len(session_id_set)
        and all(value.strip() for value in session_ids),
        "review_session_foreign_key_valid": review_session_ids.issubset(
            session_id_set
        ),
        "every_session_has_word_reviews": session_id_set.issubset(
            review_session_ids
        ),
        "three_class_labels_allowlisted": not unknown_labels,
        "review_identity_unique": len(identities) == len(set(identities)),
        "required_review_fields_nonempty": empty_required_fields == 0,
        "eight_reviews_per_session_passage": bool(session_passage_counts)
        and all(
            count == EXPECTED_REVIEWS_PER_PASSAGE
            for count in session_passage_counts.values()
        ),
        "formal_promotion_disabled": formal_promotion_disabled,
        "all_sessions_behavioral_only": behavioral_only_sessions == len(sessions),
        "ineligible_gaze_not_exported_as_evidence": gaze_rows_eligible == 0,
        "ineligible_validation_not_exported_as_evidence": (
            validation_rows_eligible == 0
        ),
        "unavailable_uncertainty_not_exported_as_evidence": (
            uncertainty_rows_eligible == 0
        ),
        "eligible_gaze_tables_empty_for_behavioral_only_smoke": not gaze_rows
        and not validation_rows,
        "unverified_reading_rows_are_non_evidence": (
            unverified_rows_are_non_evidence
        ),
        "manifest_gaze_counts_match_export_tables": manifest_counts_match,
        "manifest_gaze_separation_contract": manifest_separation_contract,
    }
    status = "passed_pipeline_smoke" if all(gates.values()) else "failed_schema_gate"
    review_count = len(reviews)
    class_fractions = {
        label: labels.get(label, 0) / review_count if review_count else 0.0
        for label in ALLOWED_LABELS
    }
    return {
        "schema_version": 1,
        "audit_id": "chi-selective-fusion-current-data-smoke-v1",
        "status": status,
        "scope": "self_development_export_pipeline_only",
        "participant_count": int(manifest.get("participant_count", len(sessions))),
        "session_count": len(sessions),
        "passage_count": len(passage_ids),
        "passage_family_count": family_count,
        "word_review_count": review_count,
        "label_counts": {label: labels.get(label, 0) for label in ALLOWED_LABELS},
        "label_fractions": class_fractions,
        "rare_review_needed_warning": class_fractions["review_needed"] < 0.10,
        "gaze": {
            "behavioral_only_session_count": behavioral_only_sessions,
            "reading_gaze_eligible_session_count": gaze_rows_eligible,
            "validation_gaze_eligible_session_count": validation_rows_eligible,
            "uncertainty_evidence_eligible_session_count": (
                uncertainty_rows_eligible
            ),
            "client_roundtrip_unverified_reading_row_count": int(
                len(unverified_rows)
            ),
            "required_runtime_branch": "F1_text_person_fallback",
        },
        "gates": gates,
        "input_sha256": {
            "word_reviews_csv": _sha256(review_path),
            "sessions_csv": _sha256(session_path),
            "dataset_manifest_evidence_canonical": _canonical_sha256(
                {
                    "participant_count": manifest.get("participant_count"),
                    "gaze_provenance": gaze_provenance,
                }
            ),
            "gaze_telemetry_csv": _sha256(gaze_path),
            "validation_samples_csv": _sha256(validation_path),
            "reading_telemetry_unverified_csv": _sha256(unverified_path),
        },
        "claim_boundary": {
            "model_fitting_authorized": False,
            "effect_estimation_authorized": False,
            "webcam_accuracy_claim_authorized": False,
            "user_benefit_claim_authorized": False,
            "formal_promotion_authorized": False,
            "pipeline_smoke_claim_authorized": status == "passed_pipeline_smoke",
        },
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    counts = result["label_counts"]
    gaze = result["gaze"]
    gate_lines = [
        f"- [{'x' if passed else ' '}] `{name}`"
        for name, passed in result["gates"].items()
    ]
    return "\n".join(
        [
            "# CHI Selective-Fusion Current-Data Smoke v1",
            "",
            f"Status: **`{result['status']}`**",
            "",
            "This is a pipeline/schema smoke test only. It does not fit a model, "
            "estimate an effect, or authorize a webcam-accuracy or user-benefit claim.",
            "",
            "## Current data",
            "",
            f"- Participants / sessions: `{result['participant_count']}` / "
            f"`{result['session_count']}`",
            f"- Passages / families: `{result['passage_count']}` / "
            f"`{result['passage_family_count']}`",
            f"- Word reviews: `{result['word_review_count']}` "
            f"(`no_review={counts['no_review']}`, `unsure={counts['unsure']}`, "
            f"`review_needed={counts['review_needed']}`)",
            f"- Eligible reading-gaze sessions: "
            f"`{gaze['reading_gaze_eligible_session_count']}`",
            f"- Unverified reading telemetry rows retained separately: "
            f"`{gaze['client_roundtrip_unverified_reading_row_count']}`",
            f"- Required runtime branch: **`{gaze['required_runtime_branch']}`**",
            "",
            "The rare positive class and single self-development participant make "
            "effect fitting invalid. The useful result is that behavioral labels "
            "remain exportable while ineligible gaze stays outside the evidence table.",
            "",
            "## Gates",
            "",
            *gate_lines,
            "",
            "## Decision",
            "",
            "Keep the export path for future dress rehearsals. Do not train F1/F2 or "
            "select an abstention threshold from this session.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    if args.export_dir is not None:
        result = audit_export(args.export_dir)
    else:
        from scripts.export_general_collection_dataset import export_bundle

        with tempfile.TemporaryDirectory(prefix="lexigaze-chi-smoke-") as directory:
            export_dir = Path(directory)
            export_bundle(args.root.resolve(), export_dir)
            result = audit_export(export_dir)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "passed_pipeline_smoke" else 1


if __name__ == "__main__":
    raise SystemExit(main())

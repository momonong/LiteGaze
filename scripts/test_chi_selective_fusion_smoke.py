from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_chi_selective_fusion_smoke import audit_export, render_markdown


class ChiSelectiveFusionSmokeTests(unittest.TestCase):
    def _fixture(self, root: Path, *, eligible: bool = False) -> None:
        reviews = []
        labels = ["no_review"] * 6 + ["unsure", "review_needed"]
        for index, label in enumerate(labels):
            reviews.append(
                {
                    "study_session_id": "private-session",
                    "passage_id": "passage-a",
                    "passage_family_id": "family-a",
                    "probe_id": f"probe-{index}",
                    "surface": f"word-{index}",
                    "stratum": "control",
                    "label": label,
                }
            )
        self._write_csv(root / "word_reviews.csv", reviews)
        self._write_csv(
            root / "sessions.csv",
            [
                {
                    "study_session_id": "private-session",
                    "state": "completed",
                    "formal_promotion_allowed": "False",
                    "gaze_quality_band": "behavioral_only",
                    "gaze_export_status": "behavioral_only",
                    "reading_gaze_export_eligible": str(eligible),
                    "validation_gaze_export_eligible": "False",
                    "uncertainty_evidence_eligible": "False",
                }
            ],
        )
        self._write_header(
            root / "gaze_telemetry.csv",
            [
                "evidence_status",
                "reading_gaze_export_eligible",
                "formal_evidence_eligible",
            ],
        )
        self._write_header(
            root / "validation_samples.csv",
            [
                "evidence_status",
                "validation_gaze_export_eligible",
                "formal_evidence_eligible",
            ],
        )
        self._write_csv(
            root / "reading_telemetry_unverified.csv",
            [
                {
                    "evidence_status": "client_roundtrip_unverified",
                    "prediction_receipt_bound": "False",
                    "reading_gaze_export_eligible": "False",
                    "formal_evidence_eligible": "False",
                }
            ],
        )
        (root / "dataset_manifest.json").write_text(
            json.dumps(
                {
                    "participant_count": 1,
                    "gaze_provenance": {
                        "eligible_reading_gaze_table_row_count": 0,
                        "client_roundtrip_unverified_reading_row_count": 1,
                        "gaze_tables_contain_only_eligible_sessions": True,
                        "unverified_reading_telemetry_separate_from_eligible_gaze": True,
                        "legacy_or_unavailable_gaze_mixed_with_eligible": False,
                    },
                }
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_header(path: Path, fields: list[str]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()

    def test_behavioral_only_export_passes_without_authorizing_fit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            result = audit_export(root)
        self.assertEqual(result["status"], "passed_pipeline_smoke")
        self.assertEqual(result["word_review_count"], 8)
        self.assertEqual(result["gaze"]["required_runtime_branch"], "F1_text_person_fallback")
        self.assertFalse(result["claim_boundary"]["model_fitting_authorized"])
        self.assertFalse(result["claim_boundary"]["effect_estimation_authorized"])
        self.assertNotIn("private-session", json.dumps(result))
        self.assertIn("pipeline/schema smoke test only", render_markdown(result))

    def test_eligible_reading_gaze_fails_this_fallback_smoke_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root, eligible=True)
            result = audit_export(root)
        self.assertEqual(result["status"], "failed_schema_gate")
        self.assertFalse(
            result["gates"]["ineligible_gaze_not_exported_as_evidence"]
        )

    def test_unknown_label_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            with (root / "word_reviews.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["label"] = "attention_failure"
            self._write_csv(root / "word_reviews.csv", rows)
            result = audit_export(root)
        self.assertEqual(result["status"], "failed_schema_gate")
        self.assertFalse(result["gates"]["three_class_labels_allowlisted"])

    def test_same_passage_for_two_sessions_is_counted_per_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            with (root / "word_reviews.csv").open(encoding="utf-8") as handle:
                reviews = list(csv.DictReader(handle))
            second_reviews = [dict(row, study_session_id="private-session-2") for row in reviews]
            self._write_csv(root / "word_reviews.csv", reviews + second_reviews)
            with (root / "sessions.csv").open(encoding="utf-8") as handle:
                sessions = list(csv.DictReader(handle))
            sessions.append(dict(sessions[0], study_session_id="private-session-2"))
            self._write_csv(root / "sessions.csv", sessions)
            manifest = json.loads(
                (root / "dataset_manifest.json").read_text(encoding="utf-8")
            )
            manifest["participant_count"] = 2
            (root / "dataset_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            result = audit_export(root)
        self.assertEqual(result["status"], "passed_pipeline_smoke")
        self.assertEqual(result["session_count"], 2)
        self.assertEqual(result["passage_count"], 1)
        self.assertEqual(result["word_review_count"], 16)

    def test_review_session_foreign_key_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            with (root / "word_reviews.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["study_session_id"] = "unknown-session"
            self._write_csv(root / "word_reviews.csv", rows)
            result = audit_export(root)
        self.assertEqual(result["status"], "failed_schema_gate")
        self.assertFalse(result["gates"]["review_session_foreign_key_valid"])

    def test_blank_review_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            with (root / "word_reviews.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["probe_id"] = ""
            self._write_csv(root / "word_reviews.csv", rows)
            result = audit_export(root)
        self.assertEqual(result["status"], "failed_schema_gate")
        self.assertFalse(result["gates"]["required_review_fields_nonempty"])

    def test_unverified_reading_row_cannot_claim_formal_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            with (root / "reading_telemetry_unverified.csv").open(
                encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["formal_evidence_eligible"] = "True"
            self._write_csv(root / "reading_telemetry_unverified.csv", rows)
            result = audit_export(root)
        self.assertEqual(result["status"], "failed_schema_gate")
        self.assertFalse(
            result["gates"]["unverified_reading_rows_are_non_evidence"]
        )

    def test_manifest_row_count_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            manifest_path = root / "dataset_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["gaze_provenance"][
                "client_roundtrip_unverified_reading_row_count"
            ] = 2
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = audit_export(root)
        self.assertEqual(result["status"], "failed_schema_gate")
        self.assertFalse(
            result["gates"]["manifest_gaze_counts_match_export_tables"]
        )

    def test_empty_eligible_table_still_requires_evidence_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            self._write_header(root / "gaze_telemetry.csv", ["evidence_status"])
            with self.assertRaisesRegex(ValueError, "gaze_telemetry.csv"):
                audit_export(root)

    def test_malformed_boolean_cannot_fail_open_as_false(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            with (root / "sessions.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["reading_gaze_export_eligible"] = "corrupt"
            self._write_csv(root / "sessions.csv", rows)
            with self.assertRaisesRegex(ValueError, "strict true/false"):
                audit_export(root)

    def test_session_without_reviews_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            with (root / "sessions.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows.append(dict(rows[0], study_session_id="private-session-2"))
            self._write_csv(root / "sessions.csv", rows)
            result = audit_export(root)
        self.assertEqual(result["status"], "failed_schema_gate")
        self.assertFalse(result["gates"]["every_session_has_word_reviews"])

    def test_non_behavioral_session_fails_behavioral_only_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            with (root / "sessions.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["gaze_export_status"] = "unavailable"
            self._write_csv(root / "sessions.csv", rows)
            result = audit_export(root)
        self.assertEqual(result["status"], "failed_schema_gate")
        self.assertFalse(result["gates"]["all_sessions_behavioral_only"])


if __name__ == "__main__":
    unittest.main()

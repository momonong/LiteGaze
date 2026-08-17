"""Tests for the participant-facing dress-rehearsal readiness audit."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts import audit_dress_rehearsal_readiness as readiness


def _write_material_binding_fixture(root: Path) -> tuple[dict[str, str], dict, str, str]:
    environment = {
        "LEXIGAZE_INVESTIGATOR_EMAIL": "approved.researcher@example.edu",
        "LEXIGAZE_PARTICIPANT_RIGHTS_CONTACT": "rights@example.edu",
        "LEXIGAZE_ETHICS_REFERENCE": "IRB-APPROVED-2026-001",
    }
    compensation_text = "每次完成後提供新台幣 500 元；中途停止按核定比例提供。"
    shared_text = "\n".join(
        (
            environment["LEXIGAZE_INVESTIGATOR_EMAIL"],
            environment["LEXIGAZE_PARTICIPANT_RIGHTS_CONTACT"],
            environment["LEXIGAZE_ETHICS_REFERENCE"],
            compensation_text,
        )
    )
    contents = {
        readiness.INVITATION: f"# Final invitation\n{shared_text}\n",
        readiness.DEBRIEF: f"# Final debrief\n{shared_text}\n",
        readiness.FINAL_CONSENT: (
            f"# Final consent\n{shared_text}\n"
            "兩次 Visit；每次 50 分鐘；Visit 2 間隔 18–72 小時；"
            "每次 6 篇文章、每篇 8 個單字。\n"
        ),
        readiness.COMPENSATION_POLICY: f"# Approved compensation\n{compensation_text}\n",
    }
    for relative_path, body in contents.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")

    rehearsal_sha256 = "d" * 64
    runtime_consent_digest_sha256 = "c" * 64
    evidence = {
        "schema_version": 1,
        "status": "approved_for_external_rehearsal",
        "canonical_protocol_id": "test-protocol",
        "canonical_protocol_version": "test-v1",
        "rehearsal_protocol_sha256": rehearsal_sha256,
        "runtime_consent_digest_sha256": runtime_consent_digest_sha256,
        "approved_bindings": {
            "investigator_email": environment["LEXIGAZE_INVESTIGATOR_EMAIL"],
            "participant_rights_contact": environment[
                "LEXIGAZE_PARTICIPANT_RIGHTS_CONTACT"
            ],
            "ethics_reference": environment["LEXIGAZE_ETHICS_REFERENCE"],
            "compensation_policy": {
                "artifact_path": readiness.COMPENSATION_POLICY,
                "artifact_sha256": hashlib.sha256(
                    (root / readiness.COMPENSATION_POLICY).read_bytes()
                ).hexdigest(),
                "participant_facing_text_zh": compensation_text,
            },
        },
        "materials": {
            label: {
                "path": relative_path,
                "sha256": hashlib.sha256((root / relative_path).read_bytes()).hexdigest(),
            }
            for label, relative_path in {
                "invitation": readiness.INVITATION,
                "debrief": readiness.DEBRIEF,
                "consent": readiness.FINAL_CONSENT,
            }.items()
        },
    }
    evidence_path = root / readiness.MATERIAL_APPROVAL_EVIDENCE
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    return (
        environment,
        {"protocol_id": "test-protocol", "protocol_version": "test-v1"},
        rehearsal_sha256,
        runtime_consent_digest_sha256,
    )


class DressRehearsalReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = readiness.audit(environment={})

    def test_additive_material_contract_is_complete(self) -> None:
        self.assertTrue(self.result["documentation_ready"])
        self.assertTrue(all(self.result["automated_checks"].values()))
        self.assertEqual(self.result["missing_files"], [])

    def test_frozen_flow_values_match_two_visit_50min_6x8_contract(self) -> None:
        self.assertEqual(
            self.result["canonical_flow"],
            {
                "estimated_minutes_per_visit": 50,
                "required_visits": 2,
                "minimum_interval_hours": 18,
                "maximum_interval_hours": 72,
                "passages_per_visit": 6,
                "word_probes_per_passage": 8,
                "minimum_age": 18,
            },
        )

    def test_current_external_rehearsal_fails_closed(self) -> None:
        self.assertFalse(self.result["external_rehearsal_ready"])
        self.assertEqual(self.result["current_decision"], "NO_GO_external_participants")
        blockers = set(self.result["blockers"])
        self.assertIn("canonical_protocol_not_approved_for_pilot", blockers)
        self.assertIn(
            "consent_draft_not_aligned_to_current_two_visit_50min_6x8_flow",
            blockers,
        )
        self.assertIn("participant_invitation_or_debrief_has_unresolved_fields", blockers)
        self.assertIn("independent_passage_review_evidence_missing", blockers)
        self.assertIn("manual_browser_visual_qa_evidence_missing", blockers)
        self.assertIn("approved_material_binding_evidence_missing", blockers)

    def test_rehearsal_contract_freezes_exact_process_only_eight_slot_boundary(self) -> None:
        protocol = json.loads(
            (readiness.ROOT / readiness.REHEARSAL_PROTOCOL).read_text(encoding="utf-8")
        )
        result = readiness._rehearsal_contract_status(
            protocol,
            readiness.FROZEN_REHEARSAL_PROTOCOL_SHA256,
        )
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["slot_ids"], [f"DR{index:02d}" for index in range(1, 9)])
        self.assertEqual(
            result["participant_count_boundary"],
            {"minimum": 5, "maximum": 8},
        )
        self.assertIs(result["extension_to_chase_a_gate_allowed"], False)
        self.assertEqual(result["analysis_role"], "process_and_measurement_only")
        self.assertIs(
            result["rehearsal_rows_may_enter_formal_confirmation"],
            False,
        )
        wrong_revision = readiness._rehearsal_contract_status(protocol, "a" * 64)
        self.assertFalse(wrong_revision["valid"])
        self.assertIn(
            "rehearsal_protocol_sha256_not_frozen_revision",
            wrong_revision["errors"],
        )

        mutations = (
            (
                ("participant_scope", "minimum_started_participants"),
                True,
                "rehearsal_minimum_started_participants_not_five",
            ),
            (
                ("participant_scope", "maximum_started_participants"),
                9,
                "rehearsal_maximum_started_participants_not_eight",
            ),
            (
                ("participant_scope", "extension_to_chase_a_gate_allowed"),
                True,
                "rehearsal_extension_to_chase_gate_not_forbidden",
            ),
            (
                ("participant_scope", "analysis_role"),
                "effect_estimation",
                "rehearsal_analysis_role_not_process_only",
            ),
            (
                ("participant_scope", "rehearsal_rows_may_enter_formal_confirmation"),
                True,
                "rehearsal_rows_not_excluded_from_confirmation",
            ),
            (
                ("participant_scope", "outcome_blind_slot_plan", "slot_ids"),
                ["DR01"] * 8,
                "rehearsal_slot_ids_not_exactly_DR01_through_DR08",
            ),
            (
                (
                    "participant_scope",
                    "outcome_blind_slot_plan",
                    "maximum_activated_slots",
                ),
                9,
                "rehearsal_maximum_activated_slots_not_eight",
            ),
            (
                (
                    "participant_scope",
                    "outcome_blind_slot_plan",
                    "pseudonymous_slots_only",
                ),
                False,
                "rehearsal_pseudonymous_slot_boundary_not_confirmed",
            ),
            (
                (
                    "participant_scope",
                    "outcome_blind_slot_plan",
                    "frozen_before_target_outcomes",
                ),
                False,
                "rehearsal_slot_plan_not_frozen_before_outcomes",
            ),
            (
                (
                    "participant_scope",
                    "outcome_blind_slot_plan",
                    "replacement_or_extension_based_on_outcomes_allowed",
                ),
                True,
                "rehearsal_outcome_based_slot_replacement_or_extension_not_forbidden",
            ),
        )
        for field_path, invalid_value, expected_error in mutations:
            with self.subTest(field_path=field_path):
                candidate = copy.deepcopy(protocol)
                target = candidate
                for key in field_path[:-1]:
                    target = target[key]
                target[field_path[-1]] = invalid_value
                invalid = readiness._rehearsal_contract_status(
                    candidate,
                    readiness.FROZEN_REHEARSAL_PROTOCOL_SHA256,
                )
                self.assertFalse(invalid["valid"])
                self.assertIn(expected_error, invalid["errors"])

    def test_known_ui_constraints_are_not_hidden(self) -> None:
        constraints = self.result["ui_constraints"]
        self.assertFalse(constraints["mid_passage_pause_control_present"])
        self.assertFalse(constraints["collection_page_withdraw_control_present"])
        self.assertFalse(constraints["completion_page_discloses_visit2_window"])
        self.assertFalse(constraints["participant_facing_8min_auto_stop_present"])
        self.assertFalse(constraints["camera_stop_control_present_on_setup"])
        self.assertTrue(constraints["moderated_mitigation_documented"])

    def test_audit_reads_no_participant_data_paths(self) -> None:
        self.assertTrue(self.result["automated_checks"]["participant_data_paths_not_read"])
        self.assertTrue(
            all(not path.startswith("data/") for path in self.result["files_read"])
        )

    def test_materials_cli_succeeds_but_external_cli_stays_blocked(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(readiness.main([]), 1)
            self.assertEqual(readiness.main(["--target", "materials"]), 0)
            self.assertEqual(readiness.main(["--target", "external-rehearsal"]), 1)

    def test_json_output_is_strict_and_matches_decision(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-dress-audit-") as temp_name:
            output = Path(temp_name) / "readiness.json"
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = readiness.main(
                    ["--target", "materials", "--json-output", str(output)]
                )
            self.assertEqual(exit_code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(payload["documentation_ready"])
            self.assertFalse(payload["external_rehearsal_ready"])
            self.assertNotIn("NaN", output.read_text(encoding="utf-8"))

    def test_visual_qa_evidence_is_version_bound_and_complete(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-visual-qa-") as temp_name:
            root = Path(temp_name)
            path = root / readiness.VISUAL_QA_EVIDENCE
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "protocol_version": "test-v1",
                        "browser_family": "chromium",
                        "checks": {
                            "consent_readable": True,
                            "keyboard_order_checked": True,
                            "camera_preview_checked": True,
                            "calibration_stop_path_checked": True,
                            "error_messages_checked": True,
                            "withdrawal_path_checked": True,
                            "completion_and_visit2_debrief_checked": True,
                            "minimum_viewport_checked": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = readiness._visual_qa_status(
                root, {"protocol_version": "test-v1"}
            )
            self.assertTrue(result["valid"])
            self.assertEqual(result["errors"], [])

    def test_passage_review_evidence_is_bound_to_bank_hash(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-passage-review-") as temp_name:
            root = Path(temp_name)
            path = root / readiness.PASSAGE_REVIEW_EVIDENCE
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "bank_sha256": "a" * 64,
                        "reviewers_independent": True,
                        "reviewer_count": 2,
                        "checks": {
                            "factual_accuracy": True,
                            "naturalness": True,
                            "accessibility": True,
                            "sensitive_content": True,
                            "difficulty_and_probe_clarity": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(readiness._passage_review_status(root, "a" * 64)["valid"])
            mismatch = readiness._passage_review_status(root, "b" * 64)
            self.assertFalse(mismatch["valid"])
            self.assertIn("passage_review_bank_sha256_mismatch", mismatch["errors"])

    def test_approved_material_binding_rejects_self_consistent_wrong_contact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-material-binding-") as temp_name:
            root = Path(temp_name)
            environment, protocol, rehearsal_sha256, consent_digest = (
                _write_material_binding_fixture(root)
            )
            valid = readiness._material_approval_status(
                root,
                environment,
                canonical_protocol=protocol,
                rehearsal_protocol_sha256=rehearsal_sha256,
                runtime_consent_digest_sha256=consent_digest,
            )
            self.assertTrue(valid["valid"], valid["errors"])

            invitation_path = root / readiness.INVITATION
            invitation_path.write_text(
                invitation_path.read_text(encoding="utf-8").replace(
                    environment["LEXIGAZE_INVESTIGATOR_EMAIL"],
                    "arbitrary.nonplaceholder@example.edu",
                ),
                encoding="utf-8",
            )
            evidence_path = root / readiness.MATERIAL_APPROVAL_EVIDENCE
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["materials"]["invitation"]["sha256"] = hashlib.sha256(
                invitation_path.read_bytes()
            ).hexdigest()
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            mismatch = readiness._material_approval_status(
                root,
                environment,
                canonical_protocol=protocol,
                rehearsal_protocol_sha256=rehearsal_sha256,
                runtime_consent_digest_sha256=consent_digest,
            )
            self.assertFalse(mismatch["valid"])
            self.assertIn(
                "approved_material_invitation_investigator_email_mismatch",
                mismatch["errors"],
            )

    def test_approved_material_binding_rejects_runtime_and_policy_tampering(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-material-binding-") as temp_name:
            root = Path(temp_name)
            environment, protocol, rehearsal_sha256, consent_digest = (
                _write_material_binding_fixture(root)
            )
            runtime_mutations = (
                (
                    "LEXIGAZE_INVESTIGATOR_EMAIL",
                    "wrong@example.edu",
                    "approved_material_investigator_email_runtime_mismatch",
                ),
                (
                    "LEXIGAZE_PARTICIPANT_RIGHTS_CONTACT",
                    "wrong-rights@example.edu",
                    "approved_material_participant_rights_contact_runtime_mismatch",
                ),
                (
                    "LEXIGAZE_ETHICS_REFERENCE",
                    "IRB-WRONG-999",
                    "approved_material_ethics_reference_runtime_mismatch",
                ),
            )
            for environment_key, invalid_value, expected_error in runtime_mutations:
                with self.subTest(environment_key=environment_key):
                    runtime_mismatch = readiness._material_approval_status(
                        root,
                        {**environment, environment_key: invalid_value},
                        canonical_protocol=protocol,
                        rehearsal_protocol_sha256=rehearsal_sha256,
                        runtime_consent_digest_sha256=consent_digest,
                    )
                    self.assertFalse(runtime_mismatch["valid"])
                    self.assertIn(expected_error, runtime_mismatch["errors"])

            compensation_path = root / readiness.COMPENSATION_POLICY
            compensation_path.write_text(
                compensation_path.read_text(encoding="utf-8") + "tampered\n",
                encoding="utf-8",
            )
            policy_mismatch = readiness._material_approval_status(
                root,
                environment,
                canonical_protocol=protocol,
                rehearsal_protocol_sha256=rehearsal_sha256,
                runtime_consent_digest_sha256=consent_digest,
            )
            self.assertFalse(policy_mismatch["valid"])
            self.assertIn(
                "approved_compensation_policy_sha256_mismatch",
                policy_mismatch["errors"],
            )


if __name__ == "__main__":
    unittest.main()

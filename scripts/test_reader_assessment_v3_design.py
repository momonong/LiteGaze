"""Regression tests for the Reader Assessment v3 design contract."""

from __future__ import annotations

import copy
import json
import unittest

from scripts.audit_reader_assessment_v3_design import (
    DEFAULT_PROTOCOL,
    audit_protocol,
)
from scripts.run_reader_assessment_v3_coverage import CANDIDATES, run_grid


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


class ReaderAssessmentV3DesignTests(unittest.TestCase):
    def test_frozen_protocol_passes_every_design_safeguard(self) -> None:
        result = audit_protocol(_protocol())
        self.assertTrue(result["design_contract_valid"])
        self.assertFalse(result["collection_ready"])
        self.assertEqual(result["failed_checks"], [])
        self.assertTrue(result["readiness_blockers"])
        self.assertTrue(all(result["checks"].values()))

    def test_latent_claim_promotion_is_rejected(self) -> None:
        protocol = copy.deepcopy(_protocol())
        protocol["claim_registry"]["cefr"]["status"] = "estimated"
        result = audit_protocol(protocol)
        self.assertFalse(result["design_contract_valid"])
        self.assertIn("latent_claims_abstain", result["failed_checks"])

    def test_adaptive_routing_before_calibration_is_rejected(self) -> None:
        protocol = copy.deepcopy(_protocol())
        protocol["stimulus_design"]["adaptive_routing_enabled"] = True
        result = audit_protocol(protocol)
        self.assertIn(
            "adaptive_routing_disabled_before_calibration", result["failed_checks"]
        )

    def test_question_answer_target_is_rejected(self) -> None:
        protocol = copy.deepcopy(_protocol())
        prohibited = protocol["outcomes"][
            "prohibited_training_or_selection_targets"
        ]
        prohibited.remove("question_answer_correctness")
        result = audit_protocol(protocol)
        self.assertIn("qa_and_circular_targets_prohibited", result["failed_checks"])

    def test_missing_joint_holdout_axis_is_rejected(self) -> None:
        protocol = copy.deepcopy(_protocol())
        protocol["split_policy"]["holdout_axes"].remove("passage_family_id")
        result = audit_protocol(protocol)
        self.assertIn(
            "participant_and_content_holdouts_frozen", result["failed_checks"]
        )

    def test_cognitive_add_on_cannot_be_folded_into_reading_session(self) -> None:
        protocol = copy.deepcopy(_protocol())
        protocol["modules"][-1]["session"] = "core"
        result = audit_protocol(protocol)
        self.assertIn(
            "cognition_is_separate_and_non_composite", result["failed_checks"]
        )

    def test_text_artifact_fine_tuning_is_rejected(self) -> None:
        protocol = copy.deepcopy(_protocol())
        model = next(
            item
            for item in protocol["model_ladder"]
            if item["id"] == "b2_frozen_text_artifact"
        )
        model["fine_tuning_allowed"] = True
        result = audit_protocol(protocol)
        self.assertIn("text_artifact_remains_frozen", result["failed_checks"])

    def test_reusing_calibration_participants_for_fusion_is_rejected(self) -> None:
        protocol = copy.deepcopy(_protocol())
        protocol["study_tracks"]["reading_measurement_calibration"][
            "participant_overlap_with_fusion"
        ] = True
        result = audit_protocol(protocol)
        self.assertIn(
            "measurement_and_fusion_tracks_are_independent", result["failed_checks"]
        )

    def test_live_v2_bank_cannot_be_promoted_as_calibrated_content(self) -> None:
        protocol = copy.deepcopy(_protocol())
        protocol["study_tracks"]["reading_measurement_calibration"][
            "v2_live_bank_role"
        ] = "measurement_bank"
        result = audit_protocol(protocol)
        self.assertIn(
            "live_v2_bank_is_not_reused_for_measurement", result["failed_checks"]
        )

    def test_preprotocol_data_cannot_become_confirmation_data(self) -> None:
        protocol = copy.deepcopy(_protocol())
        protocol["legacy_data_policy"]["eligible_for_validation_or_confirmation"] = (
            True
        )
        result = audit_protocol(protocol)
        self.assertIn(
            "preprotocol_data_cannot_enter_formal_evidence", result["failed_checks"]
        )

    def test_coverage_grid_is_deterministic_and_keeps_more_content_clusters(
        self,
    ) -> None:
        first = run_grid((60,), replicates=3, seed=17)
        second = run_grid((60,), replicates=3, seed=17)
        first.pop("runtime_seconds")
        second.pop("runtime_seconds")
        self.assertEqual(first, second)
        clusters = {
            row["candidate_id"]: row["confirmation_passage_families"]
            for row in first["rows"]
        }
        self.assertEqual(clusters["compact_18"], 3)
        self.assertEqual(clusters["diverse_48"], 12)
        self.assertEqual(len(CANDIDATES), 3)


if __name__ == "__main__":
    unittest.main()

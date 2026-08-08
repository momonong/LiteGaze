"""CPU-only contracts for independent participant capture planning."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path

from core.participant_study.independent_capture import (
    audit_independent_capture_plan,
    canonical_plan_sha256,
    load_capture_plan,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs" / "participant_study" / "independent_capture_plan.example.json"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _example() -> dict:
    return load_capture_plan(EXAMPLE)


def _frozen(*, bind: bool) -> dict:
    plan = copy.deepcopy(_example())
    plan["status"] = "frozen_before_collection"
    plan["frozen_at_utc"] = "2026-08-08T01:00:00Z"
    for index, article in enumerate(plan["article_slots"]):
        article["content_sha256"] = _digest(f"article-{index}")
        article["authorization_id"] = "synthetic-test-authorization"
    if not bind:
        return plan
    for index, participant in enumerate(plan["participant_slots"]):
        participant["binding_status"] = "bound"
        participant["binding_sha256"] = _digest(f"participant-{index}")
    for index, session in enumerate(plan["session_slots"]):
        session["binding_status"] = "bound"
        session["binding_sha256"] = _digest(f"session-{index}")
    for index, device in enumerate(plan["device_slots"]):
        device["binding_status"] = "bound"
        device["binding_sha256"] = _digest(f"device-{index}")
    return plan


def _codes(result: dict) -> set[str]:
    return {issue["code"] for issue in result["issues"]}


def _add_multiview(plan: dict, *, bound: bool, calibration_bound: bool) -> None:
    plan["claims"]["multiview_incremental_value"] = True
    plan["claims"]["missing_view_policy"] = "primary_only_fallback"
    plan["device_slots"].append(
        {
            "slot_id": "DSLOT-002",
            "analysis_role": "shared",
            "device_role": "phone-secondary",
            "device_class": "phone",
            "binding_status": "bound" if bound else "unbound",
            "binding_sha256": _digest("phone-device") if bound else None,
        }
    )
    for index, capture in enumerate(plan["capture_run_slots"], start=1):
        participant_number = (index + 1) // 2
        session_number = 1 if index % 2 else 2
        capture["source_slots"].append(
            {
                "slot_id": (f"SRC-{participant_number:03d}-{session_number:02d}-02"),
                "device_slot_id": "DSLOT-002",
                "source_role": "phone-secondary",
                "required": False,
            }
        )
        capture["synchronization"] = {
            "required": True,
            "max_offset_ms": 50,
            "clock_strategy": "monotonic-offset",
            "relative_camera_calibration_slot_id": "XCAL-001",
            "relative_camera_calibration_sha256": (
                _digest("relative-camera-calibration") if calibration_bound else None
            ),
        }


class IndependentCapturePlanTests(unittest.TestCase):
    def test_example_is_valid_template_but_not_collection_plan(self) -> None:
        plan = _example()

        template = audit_independent_capture_plan(plan, target="template")
        collection = audit_independent_capture_plan(plan, target="collection")

        self.assertEqual(template["status"], "template_valid")
        self.assertIn("TEMPLATE_NOT_AUTHORIZED", _codes(template))
        self.assertEqual(collection["status"], "not_ready")
        self.assertIn("PLAN_NOT_FROZEN", _codes(collection))

    def test_frozen_plan_passes_collection_before_bindings_exist(self) -> None:
        result = audit_independent_capture_plan(
            _frozen(bind=False),
            target="collection",
        )

        self.assertEqual(result["status"], "collection_ready")
        self.assertEqual(result["summary"]["unbound_participants"], 3)

    def test_bound_plan_passes_evidence_gate(self) -> None:
        result = audit_independent_capture_plan(_frozen(bind=True), target="evidence")

        self.assertEqual(result["status"], "evidence_ready")
        self.assertEqual(result["summary"]["bound_participants"], 3)
        self.assertEqual(result["summary"]["bound_sessions"], 6)

    def test_unknown_outcome_field_is_rejected(self) -> None:
        plan = _example()
        plan["observed_answer_accuracy"] = 1.0

        result = audit_independent_capture_plan(plan)

        self.assertEqual(result["status"], "not_ready")
        self.assertIn("UNKNOWN_OR_OUTCOME_FIELD", _codes(result))

    def test_malformed_nested_values_return_audit_instead_of_crashing(self) -> None:
        plan = _example()
        plan["capture_run_slots"][0]["article_slot_ids"] = [{}]

        result = audit_independent_capture_plan(plan)

        self.assertEqual(result["status"], "not_ready")
        self.assertIn("INVALID_ARTICLE_ASSIGNMENT", _codes(result))

    def test_schema_type_fuzz_never_crashes_the_audit(self) -> None:
        plan = _example()
        bad_values = [
            None,
            True,
            1,
            1.5,
            "",
            [],
            {},
            ["x"],
            {"x": 1},
            float("nan"),
        ]
        locations = [("top", plan)]
        for list_key in (
            "participant_slots",
            "session_slots",
            "device_slots",
            "article_slots",
            "capture_run_slots",
        ):
            locations.append((list_key, plan[list_key][0]))
        locations.append(("source", plan["capture_run_slots"][0]["source_slots"][0]))

        for location, record in locations:
            for key in record:
                for value in bad_values:
                    with self.subTest(location=location, key=key, kind=type(value)):
                        mutated = copy.deepcopy(plan)
                        if location == "top":
                            mutated[key] = value
                        elif location == "source":
                            mutated["capture_run_slots"][0]["source_slots"][0][key] = (
                                value
                            )
                        else:
                            mutated[location][0][key] = value
                        result = audit_independent_capture_plan(mutated)
                        self.assertIn(result["status"], {"template_valid", "not_ready"})

    def test_participant_session_and_capture_roles_cannot_cross(self) -> None:
        plan = _example()
        plan["session_slots"][0]["analysis_role"] = "validation"
        plan["capture_run_slots"][1]["participant_slot_id"] = "PSLOT-002"

        result = audit_independent_capture_plan(plan)

        self.assertEqual(result["status"], "not_ready")
        self.assertIn("CROSS_ROLE_REFERENCE", _codes(result))
        self.assertIn("SESSION_CAPTURE_PARTICIPANT_MISMATCH", _codes(result))

    def test_article_family_cannot_cross_roles(self) -> None:
        plan = _example()
        plan["article_slots"][1]["family_slot_id"] = "AFAMILY-001"

        result = audit_independent_capture_plan(plan)

        self.assertEqual(result["status"], "not_ready")
        self.assertIn("ARTICLE_FAMILY_CROSSES_ROLES", _codes(result))

    def test_identical_article_hash_cannot_hide_behind_different_families(self) -> None:
        plan = _frozen(bind=False)
        plan["article_slots"][1]["content_sha256"] = plan["article_slots"][0][
            "content_sha256"
        ]

        result = audit_independent_capture_plan(plan, target="collection")

        self.assertEqual(result["status"], "not_ready")
        self.assertIn("ARTICLE_CONTENT_CROSSES_ROLES", _codes(result))

    def test_device_holdout_rejects_shared_physical_device(self) -> None:
        plan = _example()
        plan["claims"]["device_generalization"] = True

        result = audit_independent_capture_plan(plan)

        self.assertEqual(result["status"], "not_ready")
        self.assertIn("DEVICE_HOLDOUT_REUSES_SHARED_SLOT", _codes(result))

    def test_multiview_requires_synchronization_and_extrinsics(self) -> None:
        plan = _example()
        plan["device_slots"].append(
            {
                "slot_id": "DSLOT-002",
                "analysis_role": "shared",
                "device_role": "phone-secondary",
                "device_class": "phone",
                "binding_status": "unbound",
                "binding_sha256": None,
            }
        )
        plan["capture_run_slots"][0]["source_slots"].append(
            {
                "slot_id": "SRC-001-01-02",
                "device_slot_id": "DSLOT-002",
                "source_role": "phone-secondary",
                "required": False,
            }
        )

        result = audit_independent_capture_plan(plan)

        self.assertEqual(result["status"], "not_ready")
        self.assertIn("MULTIVIEW_SYNC_MISSING", _codes(result))

    def test_multiview_evidence_requires_hashed_relative_calibration(self) -> None:
        missing = _frozen(bind=True)
        _add_multiview(missing, bound=True, calibration_bound=False)
        complete = _frozen(bind=True)
        _add_multiview(complete, bound=True, calibration_bound=True)

        missing_result = audit_independent_capture_plan(missing, target="evidence")
        complete_result = audit_independent_capture_plan(complete, target="evidence")

        self.assertEqual(missing_result["status"], "not_ready")
        self.assertIn("UNBOUND_MULTIVIEW_CALIBRATION", _codes(missing_result))
        self.assertEqual(complete_result["status"], "evidence_ready")
        self.assertEqual(complete_result["summary"]["multi_view_capture_runs"], 6)

    def test_unbound_and_withdrawn_slots_do_not_count_as_evidence(self) -> None:
        unbound = audit_independent_capture_plan(
            _frozen(bind=False),
            target="evidence",
        )
        withdrawn_plan = _frozen(bind=True)
        withdrawn_plan["participant_slots"][1]["binding_status"] = "withdrawn"
        withdrawn_plan["participant_slots"][1]["binding_sha256"] = None
        for session in withdrawn_plan["session_slots"]:
            if session["participant_slot_id"] == "PSLOT-002":
                session["binding_status"] = "withdrawn"
                session["binding_sha256"] = None
        withdrawn = audit_independent_capture_plan(
            withdrawn_plan,
            target="evidence",
        )

        self.assertEqual(unbound["status"], "not_ready")
        self.assertEqual(withdrawn["status"], "not_ready")
        self.assertIn("INSUFFICIENT_BOUND_PARTICIPANTS", _codes(unbound))
        self.assertIn("INSUFFICIENT_BOUND_PARTICIPANTS", _codes(withdrawn))
        self.assertEqual(withdrawn["summary"]["withdrawn_participants"], 1)

    def test_bound_session_cannot_outlive_its_participant_binding(self) -> None:
        plan = _frozen(bind=True)
        plan["participant_slots"][0]["binding_status"] = "unbound"
        plan["participant_slots"][0]["binding_sha256"] = None

        result = audit_independent_capture_plan(plan, target="evidence")

        self.assertEqual(result["status"], "not_ready")
        self.assertIn("ORPHAN_SESSION_BINDING", _codes(result))

    def test_audit_is_deterministic_and_does_not_echo_identifiers(self) -> None:
        plan = _frozen(bind=True)
        first = audit_independent_capture_plan(plan, target="evidence")
        second = audit_independent_capture_plan(
            json.loads(json.dumps(plan, sort_keys=True)),
            target="evidence",
        )
        rendered = json.dumps(first, sort_keys=True)

        self.assertEqual(first, second)
        self.assertEqual(canonical_plan_sha256(plan), first["plan_sha256"])
        self.assertNotIn("PSLOT-", rendered)
        self.assertNotIn("SSLOT-", rendered)
        self.assertNotIn("SRC-", rendered)
        self.assertNotIn(_digest("participant-0"), rendered)

    def test_module_does_not_import_torch(self) -> None:
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()

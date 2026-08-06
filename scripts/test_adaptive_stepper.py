"""Tests for the v2 adaptive reading-assessment protocol."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.cognitive_inspector.adaptive import (
    MAX_ROUNDS,
    MIN_ROUNDS,
    PASSAGE_BY_ID,
    PASSAGES,
    STANDARD_LAYOUT,
    estimate_theta,
    validate_item_bank,
)
from web import create_app


def _answers(passage_id: str, *, correct: bool = True) -> dict[str, str]:
    responses = {}
    for item in PASSAGE_BY_ID[passage_id]["questions"]:
        if correct:
            responses[item["question_id"]] = item["answer"]
        else:
            responses[item["question_id"]] = next(
                option for option in item["options"] if option != item["answer"]
            )
    return responses


class AdaptiveEngineTests(unittest.TestCase):
    def test_item_bank_has_balanced_coverage_and_no_public_key_leak(self) -> None:
        audit = validate_item_bank()
        self.assertTrue(audit["ok"], audit["errors"])
        self.assertEqual(audit["passage_count"], 6)
        self.assertEqual(audit["question_count"], 18)
        self.assertEqual(set(audit["construct_distribution"].values()), {6})

    def test_all_correct_response_pattern_has_higher_theta(self) -> None:
        correct_history = []
        incorrect_history = []
        for passage in PASSAGES:
            correct_history.append(
                {
                    "passage_id": passage["passage_id"],
                    "item_results": [
                        {"question_id": item["question_id"], "correct": True}
                        for item in passage["questions"]
                    ],
                }
            )
            incorrect_history.append(
                {
                    "passage_id": passage["passage_id"],
                    "item_results": [
                        {"question_id": item["question_id"], "correct": False}
                        for item in passage["questions"]
                    ],
                }
            )
        self.assertGreater(
            estimate_theta(correct_history)["theta"],
            estimate_theta(incorrect_history)["theta"],
        )


class AdaptiveApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="lexigaze-adaptive-v2-")
        self.addCleanup(self.temp_dir.cleanup)
        self.reports_dir = Path(self.temp_dir.name)
        patcher = patch("web.routes.inspector.REPORTS_DIR", self.reports_dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.app = create_app({"TESTING": True, "LEXIGAZE_BLUEPRINTS": ("inspector",)})
        self.client = self.app.test_client()

    def _start(self, assessment_id: str = "test-assessment") -> dict:
        response = self.client.post(
            "/api/inspector/adaptive/start",
            json={"assessment_id": assessment_id, "lang": "en"},
        )
        self.assertEqual(response.status_code, 200)
        return response.get_json()

    def _score(self, passage_id: str, *, correct: bool = True) -> dict:
        response = self.client.post(
            "/api/inspector/adaptive/score",
            json={
                "passage_id": passage_id,
                "responses": _answers(passage_id, correct=correct),
            },
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        return response.get_json()

    def test_start_hides_answers_and_explanations(self) -> None:
        payload = self._start()
        self.assertEqual(payload["min_rounds"], MIN_ROUNDS)
        self.assertEqual(payload["max_rounds"], MAX_ROUNDS)
        self.assertEqual(payload["calibration_status"], "expert_seed_only_uncalibrated")
        self.assertEqual(
            {key: payload[key] for key in STANDARD_LAYOUT}, STANDARD_LAYOUT
        )
        for question in payload["quiz"]:
            self.assertNotIn("answer", question)
            self.assertNotIn("explanation", question)
            self.assertIn("construct", question)

    def test_server_scores_complete_responses_and_rejects_missing_items(self) -> None:
        start = self._start()
        scored = self._score(start["passage_id"], correct=True)
        self.assertEqual(scored["round_result"]["correct"], 3)
        self.assertEqual(scored["round_result"]["total"], 3)
        self.assertTrue(scored["result_token"])

        first_question = start["quiz"][0]
        incomplete = self.client.post(
            "/api/inspector/adaptive/score",
            json={
                "passage_id": start["passage_id"],
                "responses": {
                    first_question["question_id"]: next(iter(first_question["options"]))
                },
            },
        )
        self.assertEqual(incomplete.status_code, 400)

    def test_signed_adaptive_flow_holds_layout_constant_and_finishes(self) -> None:
        assessment_id = "flow-constant-layout"
        current = self._start(assessment_id)
        history = []
        seen_passages = set()
        while True:
            self.assertEqual(
                {key: current[key] for key in STANDARD_LAYOUT}, STANDARD_LAYOUT
            )
            self.assertNotIn(current["passage_id"], seen_passages)
            seen_passages.add(current["passage_id"])
            scored = self._score(current["passage_id"], correct=True)
            history.append(
                {
                    "passage_id": current["passage_id"],
                    "result_token": scored["result_token"],
                    "wpm": 180.0,
                    "regression_rate": 0.08,
                    "data_quality_status": "good",
                }
            )
            next_response = self.client.post(
                "/api/inspector/adaptive/next",
                json={"assessment_id": assessment_id, "history": history},
            )
            self.assertEqual(next_response.status_code, 200)
            next_payload = next_response.get_json()
            if next_payload.get("is_finished"):
                break
            current = next_payload
        self.assertGreaterEqual(len(history), MIN_ROUNDS)
        self.assertLessEqual(len(history), MAX_ROUNDS)

        report_response = self.client.post(
            "/api/inspector/adaptive/report",
            json={
                "assessment_id": assessment_id,
                "participant_id": "adaptive-v2-user",
                "history": history,
                "persist": True,
            },
        )
        self.assertEqual(report_response.status_code, 200)
        report = report_response.get_json()
        self.assertEqual(report["summary"]["claim_status"], "not_estimated")
        self.assertEqual(report["summary"]["typography_status"], "not_estimated")
        self.assertIsNone(report["summary"]["optimal_font_size"])
        self.assertIn("不是 CEFR", report["report_md"])
        self.assertTrue((self.reports_dir / Path(report["report_path"]).name).exists())

    def test_tampered_round_token_is_rejected(self) -> None:
        start = self._start("tamper-test")
        scored = self._score(start["passage_id"])
        token = scored["result_token"]
        tampered = token[:-1] + ("A" if token[-1] != "A" else "B")
        response = self.client.post(
            "/api/inspector/adaptive/next",
            json={
                "assessment_id": "tamper-test",
                "history": [
                    {"passage_id": start["passage_id"], "result_token": tampered}
                ],
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_client_cannot_swap_passage_id_after_scoring(self) -> None:
        start = self._start("swap-test")
        scored = self._score(start["passage_id"])
        other_passage = next(
            passage["passage_id"]
            for passage in PASSAGES
            if passage["passage_id"] != start["passage_id"]
        )
        response = self.client.post(
            "/api/inspector/adaptive/next",
            json={
                "assessment_id": "swap-test",
                "history": [
                    {
                        "passage_id": other_passage,
                        "result_token": scored["result_token"],
                    }
                ],
            },
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()

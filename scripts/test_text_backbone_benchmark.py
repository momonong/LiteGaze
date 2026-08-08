"""CPU-only tests for the frozen text-backbone benchmark contract."""

from __future__ import annotations

import types
import unittest

import torch

from core.cognition.causal_backbone import (
    FrozenCausalBackbone,
    align_token_offsets_to_words,
    build_display_text,
    model_id_has_excluded_prefix,
)
from scripts import run_text_backbone_benchmark as benchmark


class _FakeEncoding(dict):
    pass


class _FakeTokenizer:
    def __call__(self, text, **kwargs):
        if text != "alpha beta":
            raise AssertionError("unexpected display text")
        if kwargs.get("add_special_tokens") is not False:
            raise AssertionError("special tokens must remain disabled")
        return _FakeEncoding(
            input_ids=torch.tensor([[0, 1, 2]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
            offset_mapping=torch.tensor([[[0, 2], [2, 5], [5, 10]]]),
        )


class _FakeModel:
    def __call__(self, **kwargs):
        if kwargs.get("use_cache") is not False:
            raise AssertionError("inference cache must remain disabled")
        logits = torch.zeros(1, 3, 4)
        logits[0, 0, 1] = 4.0
        logits[0, 1, 2] = 2.0
        return types.SimpleNamespace(logits=logits)


class TextBackboneBenchmarkTests(unittest.TestCase):
    def test_display_spans_and_leading_separator_offsets_align(self) -> None:
        text, spans = build_display_text(["Hello", ",", "world!"])

        token_word_ids, counts = align_token_offsets_to_words(
            [(0, 5), (5, 7), (7, 14)],
            spans,
        )

        self.assertEqual(text, "Hello , world!")
        self.assertEqual(spans, [(0, 5), (6, 7), (8, 14)])
        self.assertEqual(token_word_ids, [0, 1, 2])
        self.assertEqual(counts, [1, 1, 1])

    def test_alignment_rejects_a_token_spanning_two_words(self) -> None:
        _, spans = build_display_text(["one", "two"])

        with self.assertRaisesRegex(ValueError, "overlaps 2 displayed words"):
            align_token_offsets_to_words([(0, 7)], spans)

    def test_alignment_rejects_missing_word_coverage(self) -> None:
        _, spans = build_display_text(["one", "two"])

        with self.assertRaisesRegex(ValueError, "without tokenizer coverage"):
            align_token_offsets_to_words([(0, 3)], spans)

    def test_frozen_separator_policy_assigns_exact_gap_to_following_word(self) -> None:
        text, spans = build_display_text(["words:", "erything"])

        token_word_ids, counts = align_token_offsets_to_words(
            [(0, 6), (6, 7), (7, 15)],
            spans,
            text=text,
            separator_policy=(
                "assign_to_following_word_if_exact_unicode_whitespace_gap"
            ),
        )

        self.assertEqual(token_word_ids, [0, 1, 1])
        self.assertEqual(counts, [1, 2])

    def test_separator_policy_still_rejects_non_whitespace_gap_token(self) -> None:
        _, spans = build_display_text(["one", "two"])

        with self.assertRaisesRegex(ValueError, "overlaps 0 displayed words"):
            align_token_offsets_to_words(
                [(0, 3), (3, 4), (4, 7)],
                spans,
                text="one-two",
                separator_policy=(
                    "assign_to_following_word_if_exact_unicode_whitespace_gap"
                ),
            )

    def test_source_prefix_policy_is_case_insensitive(self) -> None:
        excluded = ("Qwen/", "deepseek-ai/", "uer/")

        self.assertTrue(model_id_has_excluded_prefix("qWEN/model", excluded))
        self.assertTrue(model_id_has_excluded_prefix("UER/model", excluded))
        self.assertFalse(
            model_id_has_excluded_prefix("EleutherAI/pythia-410m", excluded)
        )

    def test_frozen_protocol_contains_only_exact_allowed_revisions(self) -> None:
        protocol, specs = benchmark.load_protocol(benchmark.DEFAULT_PROTOCOL_PATH)

        self.assertEqual(specs[0].key, "gpt2")
        self.assertEqual(len(specs), 5)
        self.assertTrue(all(len(spec.revision) == 40 for spec in specs))
        self.assertFalse(protocol["source_policy"]["trust_remote_code"])
        excluded = protocol["source_policy"]["excluded_model_id_prefixes"]
        self.assertFalse(
            any(
                model_id_has_excluded_prefix(spec.model_id, excluded)
                for spec in specs
            )
        )

    def test_first_word_is_a_model_independent_zero_boundary(self) -> None:
        calculator = FrozenCausalBackbone.__new__(FrozenCausalBackbone)
        calculator.spec = types.SimpleNamespace(key="fake")
        calculator.device = torch.device("cpu")
        calculator.context_limit = 16
        calculator.separator_policy = "reject"
        calculator.tokenizer = _FakeTokenizer()
        calculator.model = _FakeModel()
        calculator.inference_seconds = 0.0
        calculator.tokens_scored = 0
        calculator.texts_scored = 0
        calculator.peak_cuda_allocated_bytes = 0
        calculator.peak_cuda_reserved_bytes = 0

        result = calculator.score_words(["alpha", "beta"])

        self.assertEqual(result.surprisals[0], 0.0)
        self.assertGreater(result.surprisals[1], 0.0)
        self.assertEqual(result.subtoken_counts, [2, 1])
        self.assertEqual(result.predicted_token_count, 2)

    def test_label_free_loader_does_not_open_duration_columns(self) -> None:
        protocol, _ = benchmark.load_protocol(benchmark.DEFAULT_PROTOCOL_PATH)

        items, fingerprint = benchmark.load_label_free_provo_items(
            benchmark.DEFAULT_PROVO_PATH,
            protocol,
        )

        self.assertEqual(len(items), 2743)
        self.assertEqual(fingerprint["participant_count"], 84)
        self.assertFalse(fingerprint["outcome_columns_read"])


if __name__ == "__main__":
    unittest.main()

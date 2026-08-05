"""CPU-only validity tests for cognition metric semantics and GPT chunking."""

from __future__ import annotations

import types
import unittest
from unittest import mock

from core.cognition import pipeline as cognition_pipeline
from core.cognition.pipeline import LanguageModelCalculator


class _TokenCountTokenizer:
    def __init__(self, *, character_tokens: bool = False) -> None:
        self.character_tokens = character_tokens

    def __call__(self, words, *, is_split_into_words):
        if not is_split_into_words:
            raise AssertionError("tests require pre-tokenized words")
        if self.character_tokens:
            count = sum(len(word) for word in words)
        else:
            count = len(words)
        return {"input_ids": list(range(count))}


def _fake_window_metrics(self, words):
    values = [int(word.removeprefix("w")) for word in words]
    surprisals = [
        -1.0 if index == 0 else float(values[index - 1] * 100 + value)
        for index, value in enumerate(values)
    ]
    return {
        "surprisals": surprisals,
        "entropies": [value + 0.25 for value in surprisals],
        "renyi_entropies": [value + 0.5 for value in surprisals],
        "attentions": [0.5] * len(words),
    }


class TextModelingValidityTests(unittest.TestCase):
    def _calculator(self) -> LanguageModelCalculator:
        calculator = LanguageModelCalculator.__new__(LanguageModelCalculator)
        calculator.model_type = "gpt2"
        calculator.lang = "en"
        calculator.device = "cpu"
        calculator.use_fp16 = False
        calculator.tokenizer = _TokenCountTokenizer()
        calculator._compute_gpt = types.MethodType(_fake_window_metrics, calculator)
        return calculator

    def test_chunking_preserves_left_context_and_renyi_entropy(self) -> None:
        calculator = self._calculator()
        words = [f"w{index}" for index in range(13)]

        expected = _fake_window_metrics(calculator, words)
        actual = calculator._compute_gpt_chunked(
            words,
            max_tokens=5,
            context_words=2,
        )

        self.assertEqual(actual["surprisals"], expected["surprisals"])
        self.assertEqual(actual["entropies"], expected["entropies"])
        self.assertEqual(actual["renyi_entropies"], expected["renyi_entropies"])
        self.assertEqual(len(actual["attentions"]), len(words))

    def test_chunking_rejects_a_single_word_larger_than_context(self) -> None:
        calculator = self._calculator()
        calculator.tokenizer = _TokenCountTokenizer(character_tokens=True)

        with self.assertRaisesRegex(ValueError, "exceeds the GPT context limit"):
            calculator._compute_gpt_chunked(["toolong"], max_tokens=3)

    def test_cpu_resolution_never_probes_cuda(self) -> None:
        with mock.patch.object(
            cognition_pipeline.torch.cuda,
            "is_available",
            side_effect=AssertionError("CUDA must not be touched"),
        ):
            self.assertEqual(LanguageModelCalculator._resolve_device("cpu"), "cpu")

    def test_compute_uses_resolved_device_not_global_cuda_availability(self) -> None:
        calculator = self._calculator()
        expected = _fake_window_metrics(calculator, ["w0", "w1"])

        context = mock.MagicMock()
        context.__enter__.return_value = None
        context.__exit__.return_value = False
        with mock.patch(
            "core.cognition.pipeline.torch.cuda.is_available",
            return_value=True,
        ), mock.patch(
            "core.cognition.pipeline.torch.amp.autocast",
            return_value=context,
        ) as autocast:
            actual = calculator.compute(["w0", "w1"])

        autocast.assert_called_once_with(device_type="cpu", enabled=False)
        self.assertEqual(actual["surprisals"], expected["surprisals"])
        self.assertEqual(actual["metric_contract"]["surprisal_kind"], "causal")

    def test_masked_and_causal_metrics_have_distinct_contracts(self) -> None:
        calculator = self._calculator()
        causal = calculator.metric_contract()
        calculator.model_type = "bert"
        masked = calculator.metric_contract()

        self.assertEqual(causal["surprisal_kind"], "causal")
        self.assertEqual(causal["context_direction"], "left_only")
        self.assertEqual(masked["surprisal_kind"], "masked_pseudo")
        self.assertEqual(masked["context_direction"], "bidirectional")
        self.assertNotEqual(causal["surprisal_kind"], masked["surprisal_kind"])


if __name__ == "__main__":
    unittest.main()

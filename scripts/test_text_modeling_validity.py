"""CPU-only validity tests for cognition metric semantics and GPT chunking."""

from __future__ import annotations

import types
import unittest
from unittest import mock

import torch

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


class _TensorEncoding(dict):
    def __init__(self) -> None:
        super().__init__({
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        })

    def to(self, device):
        return self

    def word_ids(self):
        return [0, 1, 2]


class _TensorTokenizer:
    def __call__(self, words, *, is_split_into_words, return_tensors):
        if not is_split_into_words or return_tensors != "pt":
            raise AssertionError("unexpected tokenizer contract")
        return _TensorEncoding()


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

    def test_gpt_initialization_pins_eager_attention_on_cpu(self) -> None:
        tokenizer = mock.MagicMock()
        model = mock.MagicMock()
        model.to.return_value = model
        model.eval.return_value = model
        with mock.patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ) as tokenizer_loader, mock.patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=model,
        ) as causal_loader:
            calculator = LanguageModelCalculator(
                model_type="gpt2",
                lang="en",
                device="cpu",
            )

        tokenizer_loader.assert_called_once_with(
            "gpt2",
            add_prefix_space=True,
            trust_remote_code=False,
        )
        causal_loader.assert_called_once_with(
            "gpt2",
            attn_implementation="eager",
            trust_remote_code=False,
        )
        self.assertEqual(calculator.device, "cpu")

    def test_chinese_gpt2_is_rejected_before_model_loading(self) -> None:
        with mock.patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=AssertionError("disallowed model must not be loaded"),
        ) as tokenizer_loader:
            with self.assertRaisesRegex(ValueError, "approved source policy"):
                LanguageModelCalculator(
                    model_type="gpt2",
                    lang="zh",
                    device="cpu",
                )

        tokenizer_loader.assert_not_called()

    def test_active_model_allowlist_excludes_disallowed_sources(self) -> None:
        model_ids = {
            model_id.casefold()
            for languages in LanguageModelCalculator.MODELS.values()
            for model_id in languages.values()
        }
        excluded_prefixes = tuple(
            prefix.casefold()
            for prefix in LanguageModelCalculator.EXCLUDED_MODEL_ID_PREFIXES
        )

        for model_id in model_ids:
            self.assertFalse(model_id.startswith(excluded_prefixes), model_id)

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

    def test_gpt_forward_receives_cpu_attention_mask(self) -> None:
        calculator = self._calculator()
        calculator.tokenizer = _TensorTokenizer()
        calculator.model = mock.MagicMock(
            return_value=types.SimpleNamespace(logits=torch.zeros(1, 3, 4))
        )
        calculator.model.get_input_embeddings.return_value = (
            lambda input_ids: input_ids.float().unsqueeze(-1)
        )

        result = LanguageModelCalculator._compute_gpt(
            calculator,
            ["one", "two", "three"],
        )

        self.assertEqual(calculator.model.call_args.args, ())
        input_embeddings = calculator.model.call_args.kwargs["inputs_embeds"]
        attention_mask = calculator.model.call_args.kwargs["attention_mask"]
        torch.testing.assert_close(
            input_embeddings,
            torch.tensor([[[1.0], [2.0], [3.0]]]),
        )
        torch.testing.assert_close(attention_mask, torch.tensor([[1, 1, 1]]))
        self.assertEqual(len(result["surprisals"]), 3)


if __name__ == "__main__":
    unittest.main()

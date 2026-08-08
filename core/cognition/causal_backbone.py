"""Frozen causal-LM feature extraction for controlled backbone comparisons.

This module is deliberately separate from the production cognition pipeline.
It accepts only an already validated, immutable model specification and exposes
word-level causal surprisal with explicit character-offset alignment.
"""

from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BackboneSpec:
    """Immutable identity and provenance for one experimental backbone."""

    key: str
    model_id: str
    revision: str
    developer: str
    license: str
    role: str


@dataclass(frozen=True)
class WordScoreResult:
    """Causal features and alignment diagnostics for one displayed text."""

    surprisals: list[float]
    subtoken_counts: list[int]
    token_count: int
    predicted_token_count: int
    inference_seconds: float


def build_display_text(words: Sequence[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join display items with spaces and retain exact character spans."""
    if not words:
        return "", []
    pieces: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for index, raw_word in enumerate(words):
        word = str(raw_word)
        if not word:
            raise ValueError(f"display word at index {index} is empty")
        if index:
            pieces.append(" ")
            cursor += 1
        start = cursor
        pieces.append(word)
        cursor += len(word)
        spans.append((start, cursor))
    return "".join(pieces), spans


def align_token_offsets_to_words(
    offsets: Sequence[Sequence[int]],
    word_spans: Sequence[tuple[int, int]],
) -> tuple[list[int], list[int]]:
    """Map each non-special token to exactly one displayed word.

    Fast tokenizers may include a leading separator in a token offset.  The
    overlap test ignores separators while rejecting tokens that span two words,
    because such a token has no unambiguous word-level surprisal assignment.
    """
    token_word_ids: list[int] = []
    subtoken_counts = [0] * len(word_spans)
    previous_word_id = -1
    for token_index, raw_offset in enumerate(offsets):
        if len(raw_offset) != 2:
            raise ValueError(f"token offset {token_index} is not a pair")
        start, end = (int(raw_offset[0]), int(raw_offset[1]))
        if start < 0 or end <= start:
            raise ValueError(
                f"token offset {token_index} is empty or invalid: {(start, end)}"
            )
        overlaps = [
            word_id
            for word_id, (word_start, word_end) in enumerate(word_spans)
            if start < word_end and end > word_start
        ]
        if len(overlaps) != 1:
            raise ValueError(
                f"token offset {token_index} {(start, end)} overlaps "
                f"{len(overlaps)} displayed words"
            )
        word_id = overlaps[0]
        if word_id < previous_word_id:
            raise ValueError("token-to-word alignment is not monotonic")
        previous_word_id = word_id
        token_word_ids.append(word_id)
        subtoken_counts[word_id] += 1

    missing = [index for index, count in enumerate(subtoken_counts) if count < 1]
    if missing:
        raise ValueError(f"display words without tokenizer coverage: {missing[:10]}")
    return token_word_ids, subtoken_counts


def model_id_has_excluded_prefix(
    model_id: str,
    excluded_prefixes: Sequence[str],
) -> bool:
    """Apply the protocol's source exclusion case-insensitively."""
    normalized = str(model_id).casefold()
    return any(normalized.startswith(prefix.casefold()) for prefix in excluded_prefixes)


class FrozenCausalBackbone:
    """Load and score one immutable causal LM on one device at a time."""

    def __init__(
        self,
        spec: BackboneSpec,
        *,
        device: str,
        dtype: str,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.device = torch.device(device)
        self.requested_dtype = dtype
        self.inference_seconds = 0.0
        self.tokens_scored = 0
        self.texts_scored = 0

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        load_dtype = self._resolve_dtype(dtype)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            trust_remote_code=False,
            use_fast=True,
        )
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError(f"{spec.key} did not resolve to a fast tokenizer")
        tokenizer_commit = self.tokenizer.init_kwargs.get("_commit_hash")
        if tokenizer_commit not in {None, spec.revision}:
            raise RuntimeError(
                f"{spec.key} resolved tokenizer commit {tokenizer_commit!r}, "
                f"expected {spec.revision!r}"
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            trust_remote_code=False,
            dtype=load_dtype,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        resolved_commit = getattr(self.model.config, "_commit_hash", None)
        if resolved_commit != spec.revision:
            raise RuntimeError(
                f"{spec.key} resolved model commit {resolved_commit!r}, "
                f"expected {spec.revision!r}"
            )
        self.model.to(self.device).eval()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.load_seconds = time.perf_counter() - started
        self.context_limit = self._context_limit()
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.loaded_cuda_allocated_bytes = self._cuda_memory("allocated")
        self.loaded_cuda_reserved_bytes = self._cuda_memory("reserved")
        self.peak_cuda_allocated_bytes = self._cuda_memory("max_allocated")
        self.peak_cuda_reserved_bytes = self._cuda_memory("max_reserved")

    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        normalized = str(dtype).strip().lower()
        supported = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if normalized not in supported:
            raise ValueError(f"unsupported model dtype: {dtype!r}")
        if self.device.type == "cpu" and normalized != "float32":
            return torch.float32
        return supported[normalized]

    def _context_limit(self) -> int:
        candidates = []
        for attribute in (
            "max_position_embeddings",
            "n_positions",
            "max_sequence_length",
            "seq_length",
        ):
            value = getattr(self.model.config, attribute, None)
            if isinstance(value, int) and 2 <= value < 10_000_000:
                candidates.append(value)
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 2 <= tokenizer_limit < 10_000_000:
            candidates.append(tokenizer_limit)
        if not candidates:
            raise RuntimeError(f"cannot determine context limit for {self.spec.key}")
        return int(min(candidates))

    def _cuda_memory(self, kind: str) -> int:
        if self.device.type != "cuda":
            return 0
        functions = {
            "allocated": torch.cuda.memory_allocated,
            "reserved": torch.cuda.memory_reserved,
            "max_allocated": torch.cuda.max_memory_allocated,
            "max_reserved": torch.cuda.max_memory_reserved,
        }
        return int(functions[kind](self.device))

    @torch.inference_mode()
    def score_words(self, words: Sequence[str]) -> WordScoreResult:
        """Compute left-context word surprisal in nats.

        The first word is a fixed zero-valued boundary sentinel and is excluded
        by the benchmark.  This avoids model-specific partial scoring when the
        first word contains multiple subtokens but no document-left context.
        """
        if not words:
            return WordScoreResult([], [], 0, 0, 0.0)
        text, word_spans = build_display_text(words)
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoding["offset_mapping"][0].tolist()
        token_word_ids, subtoken_counts = align_token_offsets_to_words(
            offsets,
            word_spans,
        )
        input_ids = encoding["input_ids"]
        token_count = int(input_ids.shape[1])
        if token_count > self.context_limit:
            raise ValueError(
                f"{self.spec.key} text has {token_count} tokens, exceeding "
                f"its frozen context limit {self.context_limit}"
            )

        if token_count < 2:
            return WordScoreResult(
                surprisals=[0.0] * len(words),
                subtoken_counts=subtoken_counts,
                token_count=token_count,
                predicted_token_count=0,
                inference_seconds=0.0,
            )

        input_ids = input_ids.to(self.device)
        attention_mask = encoding.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        shift_logits = outputs.logits[:, :-1, :].float().transpose(1, 2)
        shift_labels = input_ids[:, 1:]
        token_nll = F.cross_entropy(
            shift_logits,
            shift_labels,
            reduction="none",
        )[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        duration = time.perf_counter() - started

        nll_values = token_nll.detach().cpu().tolist()
        surprisals = [0.0] * len(words)
        for shifted_index, value in enumerate(nll_values, start=1):
            word_id = token_word_ids[shifted_index]
            surprisals[word_id] += float(value)
        surprisals[0] = 0.0
        if not all(math.isfinite(value) and value >= 0 for value in surprisals):
            raise RuntimeError(f"{self.spec.key} produced invalid word surprisals")

        self.inference_seconds += duration
        self.tokens_scored += len(nll_values)
        self.texts_scored += 1
        self.peak_cuda_allocated_bytes = max(
            self.peak_cuda_allocated_bytes,
            self._cuda_memory("max_allocated"),
        )
        self.peak_cuda_reserved_bytes = max(
            self.peak_cuda_reserved_bytes,
            self._cuda_memory("max_reserved"),
        )
        return WordScoreResult(
            surprisals=surprisals,
            subtoken_counts=subtoken_counts,
            token_count=token_count,
            predicted_token_count=len(nll_values),
            inference_seconds=duration,
        )

    def metadata(self) -> dict[str, Any]:
        """Return auditable identity, alignment, and resource metadata."""
        parameter = next(self.model.parameters())
        tokenizer_commit = self.tokenizer.init_kwargs.get("_commit_hash")
        return {
            "key": self.spec.key,
            "model_id": self.spec.model_id,
            "requested_revision": self.spec.revision,
            "resolved_model_commit": getattr(self.model.config, "_commit_hash", None),
            "resolved_tokenizer_commit": tokenizer_commit,
            "developer": self.spec.developer,
            "license": self.spec.license,
            "role": self.spec.role,
            "model_class": type(self.model).__name__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "parameter_count": int(self.parameter_count),
            "context_limit": self.context_limit,
            "device": str(self.device),
            "dtype": str(parameter.dtype),
            "trust_remote_code": False,
            "special_tokens_used": False,
            "metric_contract": {
                "surprisal_kind": "causal",
                "surprisal_unit": "nats",
                "context_direction": "left_only",
                "word_surprisal_aggregation": "subtoken_sum",
                "first_word_boundary_value": 0.0,
            },
            "load_seconds": self.load_seconds,
            "inference_seconds": self.inference_seconds,
            "texts_scored": self.texts_scored,
            "tokens_scored": self.tokens_scored,
            "tokens_per_second": (
                self.tokens_scored / self.inference_seconds
                if self.inference_seconds > 0
                else None
            ),
            "loaded_cuda_allocated_bytes": self.loaded_cuda_allocated_bytes,
            "loaded_cuda_reserved_bytes": self.loaded_cuda_reserved_bytes,
            "peak_cuda_allocated_bytes": self.peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": self.peak_cuda_reserved_bytes,
        }

    def close(self) -> None:
        """Release the one allowed resident backbone before loading another."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)

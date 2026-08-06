"""Evidence-bounded analysis of a single reading gaze session.

This module deliberately separates observed behaviour from latent-trait claims.
It can describe a session, but it cannot infer cognitive ability, attention,
fatigue, English proficiency, or a CEFR level without an independently
validated assessment design and suitable reference data.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from itertools import pairwise
from statistics import median
from typing import Any

from wordfreq import zipf_frequency

ASSESSMENT_VERSION = "2.0.0"
NOT_ESTIMATED = "not_estimated"


@dataclass(frozen=True)
class GazeFixation:
    word: str
    index: int
    duration_ms: float
    timestamp_ms: float
    end_timestamp_ms: float
    confidence: float
    sample_count: int


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bounded(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = _bounded(probability, 0.0, 1.0) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _wilson_interval(
    successes: int, trials: int, z: float = 1.96
) -> list[float] | None:
    if trials <= 0:
        return None
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return [round(max(0.0, centre - margin), 3), round(min(1.0, centre + margin), 3)]


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator > 0 else None


def _confidence_value(value: Any) -> float:
    if isinstance(value, str):
        normalized = value.strip().lower()
        labels = {
            "high": 0.9,
            "medium": 0.6,
            "low": 0.3,
            "unknown": 0.4,
        }
        if normalized in labels:
            return labels[normalized]
    numeric = _finite_float(value)
    if numeric is None:
        return 0.4
    return _bounded(numeric, 0.0, 1.0)


def _context_number(context: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _finite_float(context.get(key))
        if value is not None and value >= 0:
            return value
    return None


class CognitiveInspector:
    """Describe one gaze session while abstaining from unsupported traits."""

    def __init__(self, sample_rate_hz: int = 8):
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        self.tick_ms = 1000.0 / sample_rate_hz

    def _normalize_hits(
        self, gaze_history: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        missing_timestamps = 0
        invalid_events = 0
        original_timestamps: list[float] = []

        for order, hit in enumerate(gaze_history):
            if not isinstance(hit, dict):
                invalid_events += 1
                continue
            try:
                index = int(hit.get("index", -1))
            except (TypeError, ValueError):
                invalid_events += 1
                continue
            word = str(hit.get("word", "")).strip()
            if index < 0 or not word:
                invalid_events += 1
                continue

            timestamp = _finite_float(hit.get("timestamp_ms"))
            timestamp_is_observed = timestamp is not None
            if timestamp is None:
                timestamp = order * self.tick_ms
                missing_timestamps += 1
            else:
                original_timestamps.append(timestamp)

            explicit_duration = _context_number(hit, "duration_ms", "dwell_ms")
            dwell_count = _finite_float(hit.get("dwell_count"))
            normalized.append(
                {
                    "word": word,
                    "index": index,
                    "timestamp_ms": timestamp,
                    "timestamp_is_observed": timestamp_is_observed,
                    "confidence": _confidence_value(hit.get("confidence")),
                    "explicit_duration_ms": explicit_duration,
                    "dwell_count": max(1.0, dwell_count or 1.0),
                    "order": order,
                }
            )

        monotonic = all(
            later >= earlier for earlier, later in pairwise(original_timestamps)
        )
        if not monotonic:
            normalized.sort(key=lambda item: (item["timestamp_ms"], item["order"]))

        return normalized, {
            "raw_event_count": len(gaze_history),
            "valid_event_count": len(normalized),
            "invalid_event_count": invalid_events,
            "missing_timestamp_count": missing_timestamps,
            "timestamps_monotonic": monotonic,
        }

    def _estimate_tick(self, hits: list[dict[str, Any]]) -> float:
        deltas = [
            later["timestamp_ms"] - earlier["timestamp_ms"]
            for earlier, later in pairwise(hits)
            if later["timestamp_ms"] > earlier["timestamp_ms"]
        ]
        plausible = [delta for delta in deltas if 10.0 <= delta <= 2000.0]
        return float(median(plausible)) if plausible else self.tick_ms

    def group_fixations(self, gaze_history: list[dict[str, Any]]) -> list[GazeFixation]:
        hits, _ = self._normalize_hits(gaze_history)
        return self._group_normalized_hits(hits, self._estimate_tick(hits))

    def _group_normalized_hits(
        self, hits: list[dict[str, Any]], estimated_tick_ms: float
    ) -> list[GazeFixation]:
        if not hits:
            return []
        threshold_ms = max(350.0, estimated_tick_ms * 2.5)
        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []

        for hit in hits:
            if not current:
                current = [hit]
                continue
            gap = hit["timestamp_ms"] - current[-1]["timestamp_ms"]
            if hit["index"] == current[-1]["index"] and 0 <= gap <= threshold_ms:
                current.append(hit)
            else:
                groups.append(current)
                current = [hit]
        if current:
            groups.append(current)

        return [self._aggregate_group(group, estimated_tick_ms) for group in groups]

    def _aggregate_group(
        self, group: list[dict[str, Any]], estimated_tick_ms: float
    ) -> GazeFixation:
        explicit = [
            item["explicit_duration_ms"]
            for item in group
            if item["explicit_duration_ms"] is not None
        ]
        if explicit:
            duration_ms = sum(explicit)
        else:
            represented_samples = sum(item["dwell_count"] for item in group)
            observed_span = group[-1]["timestamp_ms"] - group[0]["timestamp_ms"]
            duration_ms = max(
                represented_samples * estimated_tick_ms,
                observed_span + estimated_tick_ms,
            )

        sample_weight = sum(item["dwell_count"] for item in group)
        confidence = (
            sum(item["confidence"] * item["dwell_count"] for item in group)
            / sample_weight
        )
        first = group[0]
        return GazeFixation(
            word=first["word"],
            index=first["index"],
            duration_ms=float(duration_ms),
            timestamp_ms=float(first["timestamp_ms"]),
            end_timestamp_ms=float(group[-1]["timestamp_ms"]),
            confidence=confidence,
            sample_count=max(1, round(sample_weight)),
        )

    def _lexical_processing_signal(
        self, fixations: list[GazeFixation], lang: str
    ) -> dict[str, Any]:
        rows: list[tuple[float, float, float]] = []
        for fixation in fixations:
            token = re.sub(r"[^\w'-]", "", fixation.word.lower(), flags=re.UNICODE)
            if not token:
                continue
            frequency = zipf_frequency(token, lang)
            if frequency <= 0:
                continue
            rows.append(
                (
                    float(len(token)),
                    float(frequency),
                    math.log(max(1.0, fixation.duration_ms)),
                )
            )

        if len(rows) < 12:
            return {
                "status": "insufficient_data",
                "observation_count": len(rows),
                "reason": "At least 12 lexical fixations are required for a session-level association.",
            }

        lengths = [row[0] for row in rows]
        frequencies = [row[1] for row in rows]
        log_durations = [row[2] for row in rows]
        frequency_range = max(frequencies) - min(frequencies)
        if frequency_range < 1.0:
            return {
                "status": "insufficient_variation",
                "observation_count": len(rows),
                "zipf_range": round(frequency_range, 3),
                "reason": "The passage does not contain enough lexical-frequency variation.",
            }

        length_mean = sum(lengths) / len(lengths)
        duration_mean = sum(log_durations) / len(log_durations)
        length_ss = sum((value - length_mean) ** 2 for value in lengths)
        length_slope = (
            sum(
                (length - length_mean) * (duration - duration_mean)
                for length, duration in zip(lengths, log_durations)
            )
            / length_ss
            if length_ss > 0
            else 0.0
        )
        residuals = [
            duration - (duration_mean + length_slope * (length - length_mean))
            for length, duration in zip(lengths, log_durations)
        ]
        rarity = [6.0 - frequency for frequency in frequencies]
        association = _pearson(rarity, residuals)
        return {
            "status": "session_observation"
            if association is not None
            else "insufficient_variation",
            "observation_count": len(rows),
            "rarity_dwell_association": round(association, 3)
            if association is not None
            else None,
            "zipf_range": round(frequency_range, 3),
            "interpretation": (
                "Positive values mean rarer words tended to receive longer dwell after a simple word-length adjustment. "
                "This is a passage-specific association, not an English proficiency score."
            ),
        }

    def _quality(
        self,
        audit: dict[str, Any],
        fixations: list[GazeFixation],
        estimated_tick_ms: float,
        elapsed_time_ms: float | None,
        explicit_elapsed: bool,
        text_word_count: int | None,
    ) -> dict[str, Any]:
        valid_ratio = audit["valid_event_count"] / max(1, audit["raw_event_count"])
        unique_words = len({fixation.index for fixation in fixations})
        mean_confidence = (
            sum(fixation.confidence for fixation in fixations) / len(fixations)
            if fixations
            else 0.0
        )
        amount_score = 0.5 * min(1.0, len(fixations) / 40.0) + 0.5 * min(
            1.0, unique_words / 30.0
        )
        if 20.0 <= estimated_tick_ms <= 250.0:
            cadence_score = 1.0
        elif estimated_tick_ms <= 500.0:
            cadence_score = 0.55
        else:
            cadence_score = 0.2
        timestamp_score = 1.0
        if not audit["timestamps_monotonic"]:
            timestamp_score -= 0.45
        if audit["missing_timestamp_count"]:
            timestamp_score -= 0.35
        timestamp_score = max(0.0, timestamp_score)
        quality_score = (
            0.30 * amount_score
            + 0.25 * mean_confidence
            + 0.20 * cadence_score
            + 0.15 * timestamp_score
            + 0.10 * valid_ratio
        )

        reasons: list[str] = []
        if len(fixations) < 8:
            reasons.append("too_few_fixations")
        if unique_words < 5:
            reasons.append("too_few_observed_words")
        if estimated_tick_ms > 500.0:
            reasons.append("sampling_too_sparse_for_fixation_timing")
        if mean_confidence < 0.55:
            reasons.append("low_mean_tracking_confidence")
        if not audit["timestamps_monotonic"]:
            reasons.append("timestamps_were_not_monotonic")
        if audit["missing_timestamp_count"]:
            reasons.append("timestamps_were_imputed")
        if not explicit_elapsed:
            reasons.append("reading_elapsed_time_not_explicit")
        if text_word_count is None:
            reasons.append("text_word_count_not_provided")
        if elapsed_time_ms is None:
            reasons.append("elapsed_time_unavailable")

        if len(fixations) < 8 or unique_words < 5 or valid_ratio < 0.5:
            status = "insufficient"
        elif len(fixations) < 24 or unique_words < 15 or quality_score < 0.70:
            status = "limited"
        else:
            status = "good"
        confidence = (
            "high"
            if quality_score >= 0.80
            else "moderate"
            if quality_score >= 0.60
            else "low"
        )
        return {
            "status": status,
            "confidence": confidence,
            "score": round(quality_score, 3),
            "score_is_measurement_quality_not_ability": True,
            "reasons": reasons,
            "checks": {
                **audit,
                "estimated_sample_interval_ms": round(estimated_tick_ms, 2),
                "mean_tracking_confidence": round(mean_confidence, 3),
                "unique_observed_words": unique_words,
                "elapsed_time_source": "explicit_session_context"
                if explicit_elapsed
                else "gaze_event_span",
            },
        }

    def analyze(
        self,
        gaze_history: list[dict[str, Any]],
        lang: str = "en",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(gaze_history, list):
            raise TypeError("gaze_history must be a list")
        context = context if isinstance(context, dict) else {}
        lang = lang if lang in {"en", "zh"} else "en"
        hits, audit = self._normalize_hits(gaze_history)
        estimated_tick_ms = self._estimate_tick(hits)
        fixations = self._group_normalized_hits(hits, estimated_tick_ms)

        explicit_elapsed_value = _context_number(
            context, "elapsed_time_ms", "session_duration_ms", "reading_duration_ms"
        )
        explicit_elapsed = (
            explicit_elapsed_value is not None and explicit_elapsed_value > 0
        )
        if explicit_elapsed:
            elapsed_time_ms = explicit_elapsed_value
        elif len(hits) >= 2:
            elapsed_time_ms = (
                hits[-1]["timestamp_ms"] - hits[0]["timestamp_ms"] + estimated_tick_ms
            )
        else:
            elapsed_time_ms = None

        text_count_value = _context_number(
            context, "text_word_count", "content_word_count", "total_word_count"
        )
        text_word_count = (
            int(text_count_value) if text_count_value and text_count_value > 0 else None
        )
        completed_text = context.get("completed_text") is True

        unique_indices = {fixation.index for fixation in fixations}
        durations = [fixation.duration_ms for fixation in fixations]
        total_dwell_time_ms = sum(durations)
        regressions = sum(
            current.index < previous.index for previous, current in pairwise(fixations)
        )
        transition_count = max(0, len(fixations) - 1)
        regression_rate = regressions / transition_count if transition_count else 0.0
        visited: set[int] = set()
        rereads = 0
        previous_index: int | None = None
        for fixation in fixations:
            if fixation.index != previous_index:
                if fixation.index in visited:
                    rereads += 1
                visited.add(fixation.index)
            previous_index = fixation.index

        observed_word_rate = None
        if elapsed_time_ms and elapsed_time_ms > 0:
            observed_word_rate = len(unique_indices) / (elapsed_time_ms / 60_000.0)
        reading_rate = None
        if explicit_elapsed and completed_text and text_word_count:
            reading_rate = text_word_count / (elapsed_time_ms / 60_000.0)

        quality = self._quality(
            audit,
            fixations,
            estimated_tick_ms,
            elapsed_time_ms,
            explicit_elapsed,
            text_word_count,
        )
        lexical_signal = self._lexical_processing_signal(fixations, lang)

        comprehension = context.get("comprehension")
        if not isinstance(comprehension, dict):
            comprehension = {}
        correct_value = _context_number(comprehension, "correct", "score")
        total_value = _context_number(comprehension, "total", "item_count")
        correct = int(correct_value or 0)
        total = int(total_value or 0)
        correct = min(correct, total)
        comprehension_observation = {
            "status": "session_observation" if total > 0 else "not_collected",
            "correct": correct if total > 0 else None,
            "total": total if total > 0 else None,
            "proportion_correct": round(correct / total, 3) if total > 0 else None,
            "proportion_correct_ci95": _wilson_interval(correct, total),
        }

        if len(fixations) >= 24:
            segment_size = max(1, len(fixations) // 3)
            early = durations[:segment_size]
            late = durations[-segment_size:]
            change_signal = {
                "status": "session_observation",
                "late_to_early_median_duration_ratio": round(
                    median(late) / median(early), 3
                )
                if median(early) > 0
                else None,
                "interpretation": (
                    "An unadjusted within-session change signal. Text order, lexical difficulty, posture, "
                    "and tracking drift can all produce the same pattern; it is not a fatigue diagnosis."
                ),
            }
        else:
            change_signal = {
                "status": "insufficient_data",
                "reason": "At least 24 fixations are required for a coarse early/late comparison.",
            }

        full_session_evidence = (
            quality["status"] != "insufficient"
            and reading_rate is not None
            and total >= 3
        )
        claims = {
            "session_reading_behavior": {
                "status": "session_observation" if fixations else "insufficient_data",
                "confidence": quality["confidence"],
                "scope": "this passage in this session only",
            },
            "reading_fluency": {
                "status": "provisional_session_estimate"
                if full_session_evidence
                else NOT_ESTIMATED,
                "reading_rate_wpm": round(reading_rate, 1)
                if reading_rate is not None
                else None,
                "comprehension": comprehension_observation,
                "reason": None
                if full_session_evidence
                else "Requires explicit full-text timing, text length, comprehension evidence, and usable gaze data.",
                "scope": "session performance; not a norm-referenced reading ability",
            },
            "lexical_processing_signal": lexical_signal,
            "within_session_change": change_signal,
            "cognitive_ability": {
                "status": NOT_ESTIMATED,
                "reason": "A single reading gaze trace is not a validated measure of general cognitive ability.",
            },
            "english_proficiency": {
                "status": NOT_ESTIMATED,
                "reason": "English proficiency requires calibrated multi-item evidence and external validation; lexical dwell alone is insufficient.",
            },
            "attention": {
                "status": NOT_ESTIMATED,
                "reason": "Regressions and rereading have linguistic and oculomotor causes and are not an attention scale.",
            },
            "fatigue": {
                "status": NOT_ESTIMATED,
                "reason": "Early/late gaze changes are confounded by text order, difficulty, tracking drift, and session context.",
            },
            "cognitive_load": {
                "status": NOT_ESTIMATED,
                "reason": "No validated personal or text-conditioned reference model is available for this session.",
            },
        }

        summary = {
            "total_fixations": len(fixations),
            "total_dwell_time_ms": round(total_dwell_time_ms),
            "elapsed_time_ms": round(elapsed_time_ms)
            if elapsed_time_ms is not None
            else None,
            "elapsed_time_source": "explicit_session_context"
            if explicit_elapsed
            else "gaze_event_span",
            "avg_fixation_duration_ms": round(total_dwell_time_ms / len(fixations), 1)
            if fixations
            else None,
            "median_fixation_duration_ms": round(median(durations), 1)
            if durations
            else None,
            "fixation_duration_iqr_ms": [
                round(_quantile(durations, 0.25) or 0.0, 1),
                round(_quantile(durations, 0.75) or 0.0, 1),
            ]
            if durations
            else None,
            "regression_count": regressions,
            "regression_rate": round(regression_rate, 3),
            "regression_rate_ci95": _wilson_interval(regressions, transition_count),
            "reread_count": rereads,
            "unique_words_read": len(unique_indices),
            "text_word_count": text_word_count,
            "observed_word_coverage": round(len(unique_indices) / text_word_count, 3)
            if text_word_count
            else None,
            "observed_word_rate_wpm": round(observed_word_rate, 1)
            if observed_word_rate is not None
            else None,
            "words_per_minute": round(reading_rate, 1)
            if reading_rate is not None
            else None,
            "words_per_minute_basis": "full_text_explicit_elapsed"
            if reading_rate is not None
            else "not_available",
        }

        return {
            "assessment_version": ASSESSMENT_VERSION,
            "language": lang,
            "measurement_scope": "single_session_observation",
            "summary": summary,
            "data_quality": quality,
            "claims": claims,
            "user_profile": {
                "deprecated": True,
                "reading_ability_score": None,
                "reading_ability_level": NOT_ESTIMATED,
                "english_proficiency_score": None,
                "english_proficiency_level": NOT_ESTIMATED,
                "avg_struggle_word_frequency": None,
                "cognitive_load_index": None,
                "fatigue_level": NOT_ESTIMATED,
                "fatigue_label": "Not estimated from a single gaze session",
                "fatigue_ratio": None,
                "attention_index": None,
                "replacement": "Use data_quality, summary, and claims.",
            },
            "methodology": {
                "claim_policy": "abstain_without_validity_evidence",
                "prohibited_interpretations": [
                    "general cognitive ability",
                    "clinical or diagnostic status",
                    "attention score",
                    "fatigue diagnosis",
                    "English proficiency or CEFR level",
                ],
            },
        }

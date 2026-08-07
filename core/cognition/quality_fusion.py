"""Deterministic quality-aware late fusion for the v2 shadow experiment.

The candidate intentionally contains no fitted parameters and never imports a
model runtime.  It combines a stable text prior with gaze evidence only in
proportion to independently supplied tracking-quality signals.  The production
default remains unchanged until a later independent real-capture gate passes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

PROTOCOL_ID = "quality-aware-text-fusion-v2"


@dataclass(frozen=True)
class QualityAwareFusionConfig:
    """Frozen numeric configuration for the v2 shadow candidate."""

    dwell_min_ms: float = 50.0
    dwell_max_ms: float = 3000.0
    fixation_count_cap: float = 6.0
    dwell_weight: float = 0.75
    fixation_weight: float = 0.25
    gaze_weight_power: float = 1.5
    high_confidence: float = 1.0
    medium_confidence: float = 0.55
    low_confidence: float = 0.15
    unknown_confidence: float = 0.0
    text_ood_abstain_below_quality: float = 0.45

    @classmethod
    def from_protocol(cls, protocol: Mapping[str, Any]) -> QualityAwareFusionConfig:
        """Load the exact candidate values from the frozen protocol mapping."""
        if protocol.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("unexpected quality-fusion protocol")
        candidate = _mapping(protocol, "candidate")
        gaze = _mapping(candidate, "gaze_score")
        quality = _mapping(candidate, "quality")
        dwell_bounds = gaze.get("dwell_calibration_ms")
        if not isinstance(dwell_bounds, Sequence) or len(dwell_bounds) != 2:
            raise ValueError("protocol gaze dwell calibration must contain two bounds")
        confidence = _mapping(quality, "confidence_values")
        return cls(
            dwell_min_ms=float(dwell_bounds[0]),
            dwell_max_ms=float(dwell_bounds[1]),
            fixation_count_cap=float(gaze["fixation_count_cap"]),
            dwell_weight=float(gaze["dwell_weight"]),
            fixation_weight=float(gaze["fixation_weight"]),
            gaze_weight_power=float(quality["gaze_weight_power"]),
            high_confidence=float(confidence["high"]),
            medium_confidence=float(confidence["medium"]),
            low_confidence=float(confidence["low"]),
            unknown_confidence=float(confidence["unknown"]),
            text_ood_abstain_below_quality=float(
                quality["text_out_of_distribution_abstain_below_quality"]
            ),
        ).validated()

    def validated(self) -> QualityAwareFusionConfig:
        """Reject invalid or silently non-convex configurations."""
        if not 0 <= self.dwell_min_ms < self.dwell_max_ms:
            raise ValueError("dwell calibration bounds are invalid")
        if self.fixation_count_cap <= 0:
            raise ValueError("fixation_count_cap must be positive")
        if not math.isclose(self.dwell_weight + self.fixation_weight, 1.0):
            raise ValueError("gaze score weights must sum to one")
        if self.dwell_weight < 0 or self.fixation_weight < 0:
            raise ValueError("gaze score weights must be non-negative")
        if self.gaze_weight_power <= 0:
            raise ValueError("gaze_weight_power must be positive")
        bounded = (
            self.high_confidence,
            self.medium_confidence,
            self.low_confidence,
            self.unknown_confidence,
            self.text_ood_abstain_below_quality,
        )
        if any(not 0 <= value <= 1 for value in bounded):
            raise ValueError("quality configuration values must be within [0, 1]")
        if not (
            self.high_confidence
            >= self.medium_confidence
            >= self.low_confidence
            >= self.unknown_confidence
        ):
            raise ValueError("confidence values must be monotonically ordered")
        return self


@dataclass(frozen=True)
class QualityAwareFusionResult:
    """Auditable decomposition of one shadow fusion decision."""

    protocol_id: str
    text_score: float
    gaze_score: float | None
    gaze_quality: float
    gaze_weight: float
    text_weight: float
    fused_score: float
    modality_disagreement: float | None
    abstain: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def confidence_quality(
    confidence_counts: Mapping[str, Any] | None,
    *,
    fallback_label: str | None = None,
    config: QualityAwareFusionConfig | None = None,
) -> float:
    """Return hit-weighted mapping confidence without using the best hit only."""
    active = (config or QualityAwareFusionConfig()).validated()
    values = {
        "high": active.high_confidence,
        "medium": active.medium_confidence,
        "low": active.low_confidence,
        "unknown": active.unknown_confidence,
    }
    weighted = 0.0
    total = 0.0
    for label, raw_count in (confidence_counts or {}).items():
        key = str(label).strip().lower()
        if key not in values:
            key = "unknown"
        count = _nonnegative_float(raw_count, f"confidence_counts.{label}")
        weighted += values[key] * count
        total += count
    if total > 0:
        return _clip01(weighted / total)
    key = str(fallback_label or "unknown").strip().lower()
    return values.get(key, values["unknown"])


def stable_gaze_score(
    dwell_ms: float,
    fixation_count: float,
    *,
    config: QualityAwareFusionConfig | None = None,
) -> float:
    """Map gaze measurements with frozen physical bounds, never request min-max."""
    active = (config or QualityAwareFusionConfig()).validated()
    dwell = _nonnegative_float(dwell_ms, "dwell_ms")
    fixations = _nonnegative_float(fixation_count, "fixation_count")
    if dwell <= 0 and fixations <= 0:
        return 0.0
    dwell_score = _clip01(
        (dwell - active.dwell_min_ms)
        / (active.dwell_max_ms - active.dwell_min_ms)
    )
    fixation_score = _clip01(fixations / active.fixation_count_cap)
    return _clip01(
        active.dwell_weight * dwell_score
        + active.fixation_weight * fixation_score
    )


def aggregate_quality(
    *,
    mapping_confidence: float,
    tracking_coverage: float,
    stability: float,
    calibration_quality: float,
    config: QualityAwareFusionConfig | None = None,
) -> float:
    """Combine independent quality components using the frozen geometric mean."""
    active = (config or QualityAwareFusionConfig()).validated()
    components = (
        _unit_float(mapping_confidence, "mapping_confidence"),
        _unit_float(tracking_coverage, "tracking_coverage"),
        _unit_float(stability, "stability"),
        _unit_float(calibration_quality, "calibration_quality"),
    )
    if any(value == 0 for value in components):
        return 0.0
    geometric_mean = math.prod(components) ** (1.0 / len(components))
    return _clip01(geometric_mean**active.gaze_weight_power)


def fuse_quality_aware(
    *,
    text_score: float,
    gaze_score: float | None,
    mapping_confidence: float,
    tracking_coverage: float,
    stability: float,
    calibration_quality: float,
    text_in_distribution: bool = True,
    config: QualityAwareFusionConfig | None = None,
) -> QualityAwareFusionResult:
    """Fuse one word and preserve an exact text-only fallback for missing gaze."""
    active = (config or QualityAwareFusionConfig()).validated()
    text = _unit_float(text_score, "text_score")
    if gaze_score is None:
        quality = 0.0
        gaze = None
    else:
        gaze = _unit_float(gaze_score, "gaze_score")
        quality = aggregate_quality(
            mapping_confidence=mapping_confidence,
            tracking_coverage=tracking_coverage,
            stability=stability,
            calibration_quality=calibration_quality,
            config=active,
        )

    gaze_weight = quality if gaze is not None else 0.0
    text_weight = 1.0 - gaze_weight
    fused = text if gaze is None else gaze_weight * gaze + text_weight * text
    disagreement = None if gaze is None else abs(gaze - text)
    abstain = bool(
        not text_in_distribution
        and gaze_weight < active.text_ood_abstain_below_quality
    )
    if gaze is None:
        reason = "missing_gaze_text_fallback"
    elif abstain:
        reason = "text_ood_and_low_gaze_quality"
    elif gaze_weight < 0.05:
        reason = "low_gaze_quality_text_dominant"
    else:
        reason = "quality_weighted_shadow_candidate"
    return QualityAwareFusionResult(
        protocol_id=PROTOCOL_ID,
        text_score=text,
        gaze_score=gaze,
        gaze_quality=quality,
        gaze_weight=gaze_weight,
        text_weight=text_weight,
        fused_score=_clip01(fused),
        modality_disagreement=disagreement,
        abstain=abstain,
        reason=reason,
    )


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        raise ValueError(f"protocol field {key!r} must be an object")
    return nested


def _unit_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return number


def _nonnegative_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _clip01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))

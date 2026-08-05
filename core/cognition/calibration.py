"""Stable, model-runtime-free calibration for cognitive reading-time scores."""

from __future__ import annotations

import math


# The committed regressors were trained after filtering GECO reading times to
# this fixed interval.  Reusing it avoids request-local normalization leakage.
TRAINING_TRT_MIN_MS = 50.0
TRAINING_TRT_MAX_MS = 3000.0


def calibrate_reading_time_prediction(
    prediction: float,
    *,
    log_space: bool,
) -> float:
    """Map a predicted reading time to [0, 1] with frozen training bounds."""
    value = float(prediction)
    if not math.isfinite(value):
        raise ValueError("reading-time prediction must be finite")
    lower = TRAINING_TRT_MIN_MS
    upper = TRAINING_TRT_MAX_MS
    if log_space:
        lower = math.log(lower)
        upper = math.log(upper)
    scaled = (value - lower) / (upper - lower)
    return float(min(max(scaled, 0.0), 1.0))

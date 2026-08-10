"""Pure NumPy execution for personalized gaze calibration stage chains.

This module is the shared numerical contract for training, nested evaluation,
and live inference.  It does not load images, models, routes, or participant-
study state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .calibration_regression import (
    motion_conditioned_features,
    polynomial_design,
    standardized_design,
)


GAZE_POLYNOMIAL = "gaze_polynomial"
MOTION_CONDITIONED_RIDGE = "motion_conditioned_ridge_v1"


def _as_gaze_rows(gaze_pitch_yaw: np.ndarray) -> np.ndarray:
    gaze = np.asarray(gaze_pitch_yaw, dtype=np.float64)
    if gaze.ndim != 2 or gaze.shape[1] != 2:
        raise ValueError("gaze_pitch_yaw must have shape (N, 2)")
    if len(gaze) == 0 or not np.isfinite(gaze).all():
        raise ValueError("gaze_pitch_yaw must be non-empty and finite")
    return gaze


def _stage_design(
    *,
    stage_index: int,
    stage: Mapping[str, Any],
    raw_gaze: np.ndarray,
    current: np.ndarray,
    head_pitch_yaw: np.ndarray | None,
    face_geometry: np.ndarray | None,
) -> np.ndarray:
    calibrator_type = str(stage.get("calibrator_type", GAZE_POLYNOMIAL))
    if calibrator_type == MOTION_CONDITIONED_RIDGE:
        if head_pitch_yaw is None or face_geometry is None:
            raise ValueError(
                "motion-conditioned stages require head pose and face geometry"
            )
        features = motion_conditioned_features(
            raw_gaze,
            np.asarray(head_pitch_yaw, dtype=np.float64),
            np.asarray(face_geometry, dtype=np.float64),
        )
        try:
            feature_mean = np.asarray(stage["feature_mean"], dtype=np.float64)
            feature_scale = np.asarray(stage["feature_scale"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "motion-conditioned stage lacks feature normalization"
            ) from exc
        return standardized_design(features, feature_mean, feature_scale)

    # Preserve the legacy live contract: only the explicit motion calibrator
    # takes the motion path; absent or historical type labels remain polynomial.
    try:
        degree = int(stage.get("poly_degree", 2))
    except (TypeError, ValueError) as exc:
        raise ValueError("poly_degree must be 1 or 2") from exc
    return polynomial_design(
        current,
        degree=degree,
        is_stage_1=stage_index == 0,
    )


def evaluate_stage_chain(
    gaze_pitch_yaw: np.ndarray,
    stages: Sequence[Mapping[str, Any]],
    *,
    head_pitch_yaw: np.ndarray | None = None,
    face_geometry: np.ndarray | None = None,
    clamp: bool = True,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Return final normalized coordinates and each exact production design.

    Stage one consumes raw ``[pitch, yaw]``.  Later polynomial stages consume
    the preceding ``[screen_x, screen_y]`` output.  A motion-conditioned stage
    intentionally consumes the original sensor observation, matching the live
    pipeline.  Only the final chain output is clamped.
    """

    raw_gaze = _as_gaze_rows(gaze_pitch_yaw)
    if not stages:
        raise ValueError("at least one calibration stage is required")

    current = raw_gaze.copy()
    designs: list[np.ndarray] = []
    for stage_index, stage in enumerate(stages):
        if not isinstance(stage, Mapping):
            raise ValueError("every calibration stage must be a mapping")
        design = _stage_design(
            stage_index=stage_index,
            stage=stage,
            raw_gaze=raw_gaze,
            current=current,
            head_pitch_yaw=head_pitch_yaw,
            face_geometry=face_geometry,
        )
        try:
            weights = np.asarray(stage["W"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("calibration stage lacks finite weights") from exc
        if (
            weights.ndim != 2
            or weights.shape != (design.shape[1], 2)
            or not np.isfinite(weights).all()
        ):
            raise ValueError("calibration stage weights do not match its design")
        current = design @ weights
        if not np.isfinite(current).all():
            raise ValueError("calibration stage produced non-finite coordinates")
        designs.append(design)

    if clamp:
        current = np.clip(current, -1.0, 1.0)
    return current, tuple(designs)


def apply_stage_chain(
    gaze_pitch_yaw: np.ndarray,
    stages: Sequence[Mapping[str, Any]],
    *,
    head_pitch_yaw: np.ndarray | None = None,
    face_geometry: np.ndarray | None = None,
    clamp: bool = True,
) -> np.ndarray:
    """Apply a calibration chain and return normalized screen coordinates."""

    predictions, _ = evaluate_stage_chain(
        gaze_pitch_yaw,
        stages,
        head_pitch_yaw=head_pitch_yaw,
        face_geometry=face_geometry,
        clamp=clamp,
    )
    return predictions

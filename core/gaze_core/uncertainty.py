"""Leakage-resistant, training-only gaze uncertainty evidence.

The v2 phase-1 contract deliberately produces a continuous, label-free score
without selecting a threshold.  Targets and residuals are accepted only by the
post-hoc OOF evidence helpers; they are not accepted by score fitting or score
inference functions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .stage_pipeline import evaluate_stage_chain
from .uncertainty_contract import (
    ABSTENTION_STATUS,
    DEFINITION_PATH,
    OUTPUT_STATUS,
    RUNTIME_OBSERVATION_SCHEMA_VERSION,
    UNCERTAINTY_SCHEMA_VERSION,
    UNAVAILABLE_STATUSES,
    canonical_json_bytes,
    canonical_sha256,
    load_frozen_definition,
    normalize_uncertainty_observation,
    unavailable_uncertainty,
    verified_definition,
)

_definition = verified_definition


def validate_complete_motion_grid(
    sample_ids: Sequence[str],
    target_ids: Sequence[str],
    motion_blocks: Sequence[str],
    *,
    definition_document: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed unless the exact frozen 13-target by 5-block grid exists."""

    definition, definition_sha = _definition(definition_document)
    required = definition["required_grid"]
    expected_blocks = tuple(str(value) for value in required["motion_block_ids"])
    expected_targets = tuple(str(value) for value in required["target_ids"])
    expected_total = int(required["total_samples"])

    normalized_samples = [str(value).strip() for value in sample_ids]
    normalized_targets = [str(value).strip() for value in target_ids]
    normalized_blocks = [str(value).strip() for value in motion_blocks]
    if not (
        len(normalized_samples)
        == len(normalized_targets)
        == len(normalized_blocks)
        == expected_total
    ):
        raise ValueError(
            f"uncertainty v2 requires exactly {expected_total} processed samples"
        )
    if any(not value for value in normalized_samples + normalized_targets + normalized_blocks):
        raise ValueError("uncertainty v2 grid identifiers must not be blank")
    if len(set(normalized_samples)) != len(normalized_samples):
        raise ValueError("uncertainty v2 sample_ids must be unique")
    if set(normalized_blocks) != set(expected_blocks):
        raise ValueError("uncertainty v2 motion blocks do not match the frozen grid")

    observed_pairs = list(zip(normalized_blocks, normalized_targets, strict=True))
    if len(set(observed_pairs)) != len(observed_pairs):
        raise ValueError("uncertainty v2 grid contains a duplicate block-target pair")
    expected_pairs = {
        (block_id, target_id)
        for block_id in expected_blocks
        for target_id in expected_targets
    }
    if set(observed_pairs) != expected_pairs:
        raise ValueError("uncertainty v2 grid is missing a frozen block-target pair")

    return {
        "status": "complete",
        "definition_sha256": definition_sha,
        "sample_count": expected_total,
        "motion_block_ids": list(expected_blocks),
        "target_ids": list(expected_targets),
        "pair_count": len(expected_pairs),
    }


def _regularized_gram_inverse(design: np.ndarray, alpha: float) -> np.ndarray:
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("stage alpha must be finite and non-negative")
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    return np.linalg.pinv(design.T @ design + alpha * penalty)


def _fit_ood_state(features: np.ndarray, *, shrinkage: float, jitter: float) -> dict[str, Any]:
    mean = features.mean(axis=0)
    centered = features - mean
    covariance = centered.T @ centered / max(len(features) - 1, 1)
    diagonal = np.diag(np.diag(covariance))
    shrunk = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    shrunk = shrunk + np.eye(shrunk.shape[0], dtype=np.float64) * jitter
    precision = np.linalg.pinv(shrunk)
    return {
        "feature_count": int(features.shape[1]),
        "feature_mean": mean.astype(float).tolist(),
        "precision": precision.astype(float).tolist(),
    }


def _stage_signature(stages: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256([dict(stage) for stage in stages])


def _raw_components(
    *,
    gaze_pitch_yaw: np.ndarray,
    stages: Sequence[Mapping[str, Any]],
    stage_states: Sequence[Mapping[str, Any]],
    jackknife_stage_sets: Sequence[Sequence[Mapping[str, Any]]],
    head_pitch_yaw: np.ndarray | None,
    face_geometry: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, designs = evaluate_stage_chain(
        gaze_pitch_yaw,
        stages,
        head_pitch_yaw=head_pitch_yaw,
        face_geometry=face_geometry,
        clamp=True,
    )
    if len(designs) != len(stage_states):
        raise ValueError("uncertainty stage state does not match production stages")

    ood_per_stage: list[np.ndarray] = []
    leverage_per_stage: list[np.ndarray] = []
    for design, stage_state in zip(designs, stage_states, strict=True):
        features = design[:, :-1]
        mean = np.asarray(stage_state["feature_mean"], dtype=np.float64)
        precision = np.asarray(stage_state["precision"], dtype=np.float64)
        gram_inverse = np.asarray(
            stage_state["regularized_gram_inverse"], dtype=np.float64
        )
        if mean.shape != (features.shape[1],) or precision.shape != (
            features.shape[1],
            features.shape[1],
        ):
            raise ValueError("uncertainty OOD state width mismatch")
        if gram_inverse.shape != (design.shape[1], design.shape[1]):
            raise ValueError("uncertainty leverage state width mismatch")
        centered = features - mean
        mahalanobis_squared = np.einsum(
            "ni,ij,nj->n", centered, precision, centered
        )
        ood_per_stage.append(
            np.sqrt(np.maximum(mahalanobis_squared, 0.0) / features.shape[1])
        )
        leverage_per_stage.append(
            np.maximum(
                np.einsum("ni,ij,nj->n", design, gram_inverse, design),
                0.0,
            )
        )

    ood = np.max(np.column_stack(ood_per_stage), axis=1)
    leverage = np.max(np.column_stack(leverage_per_stage), axis=1)

    if len(jackknife_stage_sets) < 2:
        raise ValueError("at least two block-jackknife models are required")
    ensemble = np.stack(
        [
            evaluate_stage_chain(
                gaze_pitch_yaw,
                member_stages,
                head_pitch_yaw=head_pitch_yaw,
                face_geometry=face_geometry,
                clamp=True,
            )[0]
            for member_stages in jackknife_stage_sets
        ],
        axis=0,
    )
    centered_predictions = ensemble - ensemble.mean(axis=0, keepdims=True)
    covariance_norm = np.einsum(
        "mni,mnj->nij", centered_predictions, centered_predictions
    ) / (ensemble.shape[0] - 1)
    disagreement = np.sqrt(
        np.maximum(np.trace(covariance_norm, axis1=1, axis2=2), 0.0) / 2.0
    )
    return ood, leverage, disagreement, covariance_norm


def fit_score_state(
    gaze_pitch_yaw: np.ndarray,
    stages: Sequence[Mapping[str, Any]],
    motion_blocks: Sequence[str],
    jackknife_stage_sets: Sequence[Sequence[Mapping[str, Any]]],
    *,
    jackknife_member_holdout_motion_block_ids: Sequence[str],
    training_sample_ids: Sequence[str],
    head_pitch_yaw: np.ndarray | None = None,
    face_geometry: np.ndarray | None = None,
    fit_scope: str,
    definition_document: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit label-free score state on one training partition.

    This function intentionally has no target or residual parameter.
    """

    definition, definition_sha = _definition(definition_document)
    gaze = np.asarray(gaze_pitch_yaw, dtype=np.float64)
    predictions, designs = evaluate_stage_chain(
        gaze,
        stages,
        head_pitch_yaw=head_pitch_yaw,
        face_geometry=face_geometry,
        clamp=True,
    )
    del predictions
    normalized_blocks = [str(value).strip() for value in motion_blocks]
    if len(normalized_blocks) != len(gaze) or any(not value for value in normalized_blocks):
        raise ValueError("motion_blocks must align with score-state training rows")
    if len(set(normalized_blocks)) < 3:
        raise ValueError("score state requires at least three training motion blocks")
    normalized_sample_ids = [str(value).strip() for value in training_sample_ids]
    if (
        len(normalized_sample_ids) != len(gaze)
        or any(not value for value in normalized_sample_ids)
        or len(set(normalized_sample_ids)) != len(normalized_sample_ids)
    ):
        raise ValueError(
            "training_sample_ids must be unique, non-blank, and align with score rows"
        )
    training_block_ids = sorted(set(normalized_blocks))
    member_holdouts = [
        str(value).strip()
        for value in jackknife_member_holdout_motion_block_ids
    ]
    if (
        len(member_holdouts) != len(jackknife_stage_sets)
        or any(not value for value in member_holdouts)
        or len(set(member_holdouts)) != len(member_holdouts)
        or set(member_holdouts) != set(training_block_ids)
    ):
        raise ValueError(
            "jackknife members must map one-to-one to every training motion block"
        )

    shrinkage = float(definition["ood_component"]["diagonal_shrinkage"])
    jitter = float(definition["ood_component"]["numerical_jitter"])
    stage_states: list[dict[str, Any]] = []
    for stage_index, (stage, design) in enumerate(zip(stages, designs, strict=True)):
        ood_state = _fit_ood_state(
            design[:, :-1],
            shrinkage=shrinkage,
            jitter=jitter,
        )
        alpha = float(stage.get("alpha", 0.0) or 0.0)
        stage_states.append(
            {
                "stage_index": stage_index,
                "calibrator_type": str(
                    stage.get("calibrator_type", "gaze_polynomial")
                ),
                **ood_state,
                "alpha": alpha,
                "regularized_gram_inverse": _regularized_gram_inverse(
                    design, alpha
                ).astype(float).tolist(),
            }
        )

    serializable_jackknife = [
        [dict(stage) for stage in member_stages]
        for member_stages in jackknife_stage_sets
    ]
    jackknife_member_proofs = [
        {
            "member_index": member_index,
            "held_out_training_motion_block_id": holdout_id,
            "fit_motion_block_ids": [
                block_id
                for block_id in training_block_ids
                if block_id != holdout_id
            ],
            "fit_motion_block_ids_sha256": canonical_sha256(
                [
                    block_id
                    for block_id in training_block_ids
                    if block_id != holdout_id
                ]
            ),
            "fit_sample_count": sum(
                block_id != holdout_id for block_id in normalized_blocks
            ),
            "fit_sample_ids_sha256": canonical_sha256(
                sorted(
                    sample_id
                    for sample_id, block_id in zip(
                        normalized_sample_ids,
                        normalized_blocks,
                        strict=True,
                    )
                    if block_id != holdout_id
                )
            ),
            "held_out_sample_count": sum(
                block_id == holdout_id for block_id in normalized_blocks
            ),
            "held_out_sample_ids_sha256": canonical_sha256(
                sorted(
                    sample_id
                    for sample_id, block_id in zip(
                        normalized_sample_ids,
                        normalized_blocks,
                        strict=True,
                    )
                    if block_id == holdout_id
                )
            ),
            "stage_signature_sha256": _stage_signature(
                serializable_jackknife[member_index]
            ),
        }
        for member_index, holdout_id in enumerate(member_holdouts)
    ]
    ood, leverage, disagreement, _ = _raw_components(
        gaze_pitch_yaw=gaze,
        stages=stages,
        stage_states=stage_states,
        jackknife_stage_sets=serializable_jackknife,
        head_pitch_yaw=head_pitch_yaw,
        face_geometry=face_geometry,
    )
    state: dict[str, Any] = {
        "schema_version": UNCERTAINTY_SCHEMA_VERSION,
        "status": OUTPUT_STATUS,
        "definition_sha256": definition_sha,
        "fit_scope": str(fit_scope),
        "training_sample_count": int(len(gaze)),
        "training_sample_ids_sha256": canonical_sha256(
            sorted(normalized_sample_ids)
        ),
        "training_motion_block_ids": training_block_ids,
        "production_stage_signature_sha256": _stage_signature(stages),
        "stage_states": stage_states,
        "jackknife_stage_sets": serializable_jackknife,
        "jackknife_member_count": len(serializable_jackknife),
        "jackknife_member_holdout_motion_block_ids": member_holdouts,
        "jackknife_member_proofs": jackknife_member_proofs,
        "component_reference": {
            "ood": np.sort(ood).astype(float).tolist(),
            "leverage": np.sort(leverage).astype(float).tolist(),
            "disagreement": np.sort(disagreement).astype(float).tolist(),
        },
        "threshold": None,
        "abstention_status": ABSTENTION_STATUS,
        "claim_boundary": definition["claim_boundary"],
    }
    state["state_sha256"] = canonical_sha256(state)
    return state


def _ecdf(reference: Sequence[float], values: np.ndarray) -> np.ndarray:
    sorted_reference = np.asarray(reference, dtype=np.float64)
    if (
        sorted_reference.ndim != 1
        or len(sorted_reference) == 0
        or not np.isfinite(sorted_reference).all()
        or np.any(sorted_reference[:-1] > sorted_reference[1:])
    ):
        raise ValueError("uncertainty component reference must be sorted and finite")
    return np.searchsorted(sorted_reference, values, side="right") / len(
        sorted_reference
    )


def score_samples(
    gaze_pitch_yaw: np.ndarray,
    stages: Sequence[Mapping[str, Any]],
    score_state: Mapping[str, Any],
    viewport_list: Sequence[Sequence[float]],
    *,
    head_pitch_yaw: np.ndarray | None = None,
    face_geometry: np.ndarray | None = None,
    definition_document: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score sensor observations without accepting labels, residuals, or text."""

    _, definition_sha = _definition(definition_document)
    state = dict(score_state)
    stored_state_sha = state.pop("state_sha256", None)
    if stored_state_sha != canonical_sha256(state):
        raise ValueError("uncertainty score state hash mismatch")
    state["state_sha256"] = stored_state_sha
    if state.get("definition_sha256") != definition_sha:
        raise ValueError("uncertainty score state definition hash mismatch")
    if state.get("status") != OUTPUT_STATUS or state.get("threshold") is not None:
        raise ValueError("uncertainty score state must remain scored_no_threshold")
    if state.get("production_stage_signature_sha256") != _stage_signature(stages):
        raise ValueError("uncertainty score state does not match production stages")

    gaze = np.asarray(gaze_pitch_yaw, dtype=np.float64)
    viewports = np.asarray(viewport_list, dtype=np.float64)
    if viewports.shape != (len(gaze), 2):
        raise ValueError("viewport_list must have shape (N, 2)")
    if not np.isfinite(viewports).all() or (viewports <= 0).any():
        raise ValueError("viewport_list must be finite and positive")

    ood, leverage, disagreement, covariance_norm = _raw_components(
        gaze_pitch_yaw=gaze,
        stages=stages,
        stage_states=state["stage_states"],
        jackknife_stage_sets=state["jackknife_stage_sets"],
        head_pitch_yaw=head_pitch_yaw,
        face_geometry=face_geometry,
    )
    references = state["component_reference"]
    percentiles = {
        "ood": _ecdf(references["ood"], ood),
        "leverage": _ecdf(references["leverage"], leverage),
        "disagreement": _ecdf(references["disagreement"], disagreement),
    }
    primary = np.max(np.column_stack(list(percentiles.values())), axis=1)
    scales = np.zeros((len(gaze), 2, 2), dtype=np.float64)
    scales[:, 0, 0] = viewports[:, 0] * 0.5
    scales[:, 1, 1] = viewports[:, 1] * 0.5
    covariance_px = scales @ covariance_norm @ scales
    return {
        "status": OUTPUT_STATUS,
        "definition_sha256": definition_sha,
        "threshold": None,
        "abstention_status": ABSTENTION_STATUS,
        "uncertainty_score": primary,
        "components": {
            "ood": ood,
            "leverage": leverage,
            "disagreement": disagreement,
        },
        "component_percentiles": percentiles,
        "jackknife_disagreement_covariance_norm": covariance_norm,
        "jackknife_disagreement_covariance_px": covariance_px,
    }


def _error_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    if not rows:
        return {"mean_px": None, "median_px": None, "p90_px": None}
    values = np.asarray([float(row["spatial_error_px"]) for row in rows])
    return {
        "mean_px": float(np.mean(values)),
        "median_px": float(np.median(values)),
        "p90_px": float(np.percentile(values, 90)),
    }


def _coverage_entry(
    rows: Sequence[Mapping[str, Any]],
    retained_ids: set[str],
    *,
    requested_coverage: float,
) -> dict[str, Any]:
    retained = [row for row in rows if str(row["sample_id"]) in retained_ids]
    block_ids = sorted({str(row["outer_holdout_group_id"]) for row in rows})
    per_block: dict[str, Any] = {}
    block_metrics: list[dict[str, float | None]] = []
    for block_id in block_ids:
        block_rows = [
            row for row in rows if str(row["outer_holdout_group_id"]) == block_id
        ]
        retained_block = [
            row for row in block_rows if str(row["sample_id"]) in retained_ids
        ]
        metrics = _error_summary(retained_block)
        block_metrics.append(metrics)
        per_block[block_id] = {
            "retained_count": len(retained_block),
            "total_count": len(block_rows),
            "coverage": len(retained_block) / len(block_rows),
            **metrics,
        }

    def macro(metric: str) -> float | None:
        values = [
            float(item[metric])
            for item in block_metrics
            if item[metric] is not None
        ]
        return float(np.mean(values)) if values else None

    block_means = [
        float(item["mean_px"])
        for item in block_metrics
        if item["mean_px"] is not None
    ]
    zero_coverage_blocks = [
        block_id
        for block_id, summary in per_block.items()
        if summary["retained_count"] == 0
    ]
    nonempty_macro = {
        "included_motion_block_count": len(block_metrics)
        - len(zero_coverage_blocks),
        "total_motion_block_count": len(block_metrics),
        "mean_px": macro("mean_px"),
        "median_px": macro("median_px"),
        "p90_px": macro("p90_px"),
    }
    return {
        "requested_coverage": float(requested_coverage),
        "retained_count": len(retained),
        "total_count": len(rows),
        "empirical_coverage": len(retained) / len(rows),
        "retained_sample_ids": sorted(retained_ids),
        "would_abstain_sample_ids": sorted(
            str(row["sample_id"])
            for row in rows
            if str(row["sample_id"]) not in retained_ids
        ),
        "overall": _error_summary(retained),
        "motion_blocks_with_zero_coverage": zero_coverage_blocks,
        "motion_block_macro_all_blocks": (
            None if zero_coverage_blocks else nonempty_macro
        ),
        "motion_block_macro_nonempty_blocks": {
            "availability": (
                "descriptive_nonempty_blocks_only"
                if zero_coverage_blocks
                else "all_blocks_have_nonzero_coverage"
            ),
            **nonempty_macro,
        },
        "worst_motion_block_mean_px": (
            None
            if zero_coverage_blocks
            else max(block_means) if block_means else None
        ),
        "worst_nonempty_motion_block_mean_px": (
            max(block_means) if block_means else None
        ),
        "per_motion_block": per_block,
    }


def build_fixed_coverage_risk(
    rows: Sequence[Mapping[str, Any]],
    *,
    definition_document: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build descriptive risk evidence without selecting any threshold."""

    definition, definition_sha = _definition(definition_document)
    if not rows:
        raise ValueError("OOF rows must not be empty")
    normalized_rows = [dict(row) for row in rows]
    sample_ids = [str(row.get("sample_id", "")).strip() for row in normalized_rows]
    if any(not value for value in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("OOF sample_ids must be unique and non-blank")
    for row in normalized_rows:
        score = float(row["uncertainty_score"])
        error = float(row["spatial_error_px"])
        if not np.isfinite(score) or not np.isfinite(error) or error < 0:
            raise ValueError("OOF scores and errors must be finite")

    ordered = sorted(
        normalized_rows,
        key=lambda row: (float(row["uncertainty_score"]), str(row["sample_id"])),
    )
    coverage_entries: list[dict[str, Any]] = []
    grid = [float(value) for value in definition["coverage_risk"]["coverage_grid"]]
    for coverage in grid:
        retain_count = min(
            len(ordered),
            max(1, int(round(coverage * len(ordered)))),
        )
        retained_ids = {
            str(row["sample_id"]) for row in ordered[:retain_count]
        }
        key = f"{coverage:.1f}"
        for row in normalized_rows:
            row.setdefault("would_abstain_at_fixed_coverage", {})[key] = (
                str(row["sample_id"]) not in retained_ids
            )
        coverage_entries.append(
            _coverage_entry(
                normalized_rows,
                retained_ids,
                requested_coverage=coverage,
            )
        )

    negative_order = sorted(
        normalized_rows,
        key=lambda row: hashlib.sha256(
            f"{definition_sha}{row['sample_id']}".encode("utf-8")
        ).hexdigest(),
    )
    negative_entries: list[dict[str, Any]] = []
    for coverage in grid:
        retain_count = min(
            len(negative_order),
            max(1, int(round(coverage * len(negative_order)))),
        )
        negative_entries.append(
            _coverage_entry(
                normalized_rows,
                {str(row["sample_id"]) for row in negative_order[:retain_count]},
                requested_coverage=coverage,
            )
        )

    return {
        "schema_version": UNCERTAINTY_SCHEMA_VERSION,
        "definition_sha256": definition_sha,
        "threshold_selected": False,
        "threshold": None,
        "coverage_grid": grid,
        "rows": normalized_rows,
        "coverage_risk": coverage_entries,
        "negative_control": {
            "method": definition["negative_control"]["ordering"],
            "used_for_score_or_threshold_selection": False,
            "coverage_risk": negative_entries,
        },
    }


def uncertainty_status_for_model(model: Mapping[str, Any]) -> dict[str, Any]:
    """Return a conservative availability status for legacy/malformed artifacts."""

    bundle = model.get("uncertainty_v2")
    if not isinstance(bundle, Mapping):
        return unavailable_uncertainty(
            "unavailable_legacy_stage_state",
            "model artifact has no uncertainty_v2 state",
        )
    try:
        _, definition_sha = _definition(None)
    except (OSError, ValueError, json.JSONDecodeError):
        return unavailable_uncertainty(
            "unavailable_definition_error",
            "the frozen uncertainty definition could not be verified",
        )
    if bundle.get("definition_sha256") != definition_sha:
        return unavailable_uncertainty(
            "unavailable_definition_mismatch",
            "model uncertainty definition does not match frozen v2",
        )
    if bundle.get("status") != OUTPUT_STATUS or bundle.get("threshold") is not None:
        return unavailable_uncertainty(
            "unavailable_policy_mismatch",
            "model uncertainty state is not scored_no_threshold",
        )
    if not isinstance(bundle.get("final_score_state"), Mapping):
        return unavailable_uncertainty(
            "unavailable_missing_score_state",
            "model uncertainty bundle lacks final score state",
        )
    return {
        "schema_version": RUNTIME_OBSERVATION_SCHEMA_VERSION,
        "status": OUTPUT_STATUS,
        "definition_sha256": definition_sha,
        "threshold": None,
        "abstention_status": ABSTENTION_STATUS,
    }

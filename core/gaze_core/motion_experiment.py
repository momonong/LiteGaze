"""Nested, group-held-out evaluation for motion-diverse gaze calibration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .calibration_regression import (
    MOTION_FEATURE_NAMES,
    MOTION_PROMOTION_MIN_ABSOLUTE_PX,
    MOTION_PROMOTION_MIN_RELATIVE,
    fit_best_stage,
    fit_standardized_ridge,
    motion_challenger_decision,
    motion_conditioned_features,
)
from .stage_pipeline import apply_stage_chain
from .uncertainty import (
    ABSTENTION_STATUS,
    OUTPUT_STATUS,
    UNCERTAINTY_SCHEMA_VERSION,
    build_fixed_coverage_risk,
    canonical_sha256,
    fit_score_state,
    load_frozen_definition,
    score_samples,
    validate_complete_motion_grid,
)

BASELINE_MODEL = "gaze_polynomial"
CHALLENGER_MODEL = "motion_conditioned_ridge_v1"
VALIDATION_SCHEME = "nested_leave_one_motion_block_out"


def _baseline_stage(
    weights: np.ndarray,
    *,
    degree: int,
    alpha: float,
) -> dict[str, Any]:
    return {
        "stage": 1,
        "calibrator_type": BASELINE_MODEL,
        "W": np.asarray(weights, dtype=np.float64).astype(float).tolist(),
        "poly_degree": int(degree),
        "alpha": float(alpha),
    }


def _challenger_stage(
    weights: np.ndarray,
    *,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    return {
        "stage": 1,
        "calibrator_type": CHALLENGER_MODEL,
        "feature_names": list(MOTION_FEATURE_NAMES),
        "feature_mean": np.asarray(feature_mean, dtype=np.float64).astype(float).tolist(),
        "feature_scale": np.asarray(feature_scale, dtype=np.float64).astype(float).tolist(),
        "W": np.asarray(weights, dtype=np.float64).astype(float).tolist(),
        "alpha": float(alpha),
    }


def _fit_partition_selected_pipeline(
    gaze: np.ndarray,
    head: np.ndarray,
    face: np.ndarray,
    targets: np.ndarray,
    viewports: np.ndarray,
    groups: Sequence[str],
) -> dict[str, Any]:
    """Select and fit one whole stage-one pipeline inside a training partition."""

    normalized_groups = [str(value).strip() for value in groups]
    if len(set(normalized_groups)) < 3:
        raise ValueError("pipeline selection requires at least three motion blocks")
    unique_targets = len(np.unique(targets, axis=0))
    baseline_weights, baseline_degree, baseline_alpha, baseline_error = fit_best_stage(
        gaze,
        targets,
        viewports,
        unique_targets,
        is_stage_1=True,
        validation_groups=normalized_groups,
    )
    features = motion_conditioned_features(gaze, head, face)
    (
        challenger_weights,
        feature_mean,
        feature_scale,
        challenger_alpha,
        challenger_error,
    ) = fit_standardized_ridge(
        features,
        targets,
        viewports,
        validation_groups=normalized_groups,
    )
    promote, required, observed = motion_challenger_decision(
        baseline_error,
        challenger_error,
    )
    selected_model = CHALLENGER_MODEL if promote else BASELINE_MODEL
    stage = (
        _challenger_stage(
            challenger_weights,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            alpha=challenger_alpha,
        )
        if promote
        else _baseline_stage(
            baseline_weights,
            degree=baseline_degree,
            alpha=baseline_alpha,
        )
    )
    return {
        "stages": [stage],
        "selection": {
            "selected_model": selected_model,
            "promotion_passed": bool(promote),
            "required_improvement_px": float(required),
            "observed_improvement_px": float(observed),
            "baseline_inner_validation_mean_px": float(baseline_error),
            "challenger_inner_validation_mean_px": float(challenger_error),
            "baseline_hyperparameters": {
                "degree": int(baseline_degree),
                "alpha": float(baseline_alpha),
            },
            "challenger_hyperparameters": {"alpha": float(challenger_alpha)},
        },
    }


def _error_statistics(errors: np.ndarray) -> dict[str, float]:
    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("pixel errors must be a non-empty finite vector")
    return {
        "mean_px": float(np.mean(values)),
        "median_px": float(np.median(values)),
        "p95_px": float(np.percentile(values, 95)),
    }


def _pixel_errors(
    predictions: np.ndarray,
    targets: np.ndarray,
    viewports: np.ndarray,
) -> np.ndarray:
    return np.linalg.norm((predictions - targets) * (viewports * 0.5), axis=1)


def evaluate_motion_candidates(
    gaze_pitch_yaw: np.ndarray,
    head_pitch_yaw: np.ndarray,
    face_geometry: np.ndarray,
    targets: np.ndarray,
    viewport_list: Sequence[Sequence[float]],
    motion_blocks: Sequence[str],
) -> dict[str, Any]:
    """Compare M0/M1 with an outer motion-block holdout and inner CV.

    The complete outer block is invisible to hyperparameter selection. The
    remaining blocks are passed to the existing group-held-out fitters, so
    degree/regularization selection happens only inside the outer training
    partition.
    """

    gaze = np.asarray(gaze_pitch_yaw, dtype=np.float64)
    head = np.asarray(head_pitch_yaw, dtype=np.float64)
    face = np.asarray(face_geometry, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    viewports = np.asarray(viewport_list, dtype=np.float64)
    if gaze.ndim != 2 or gaze.shape[1] != 2:
        raise ValueError("gaze_pitch_yaw must have shape (N, 2)")
    if head.shape != gaze.shape:
        raise ValueError("head_pitch_yaw must have shape (N, 2)")
    if face.shape != (len(gaze), 3):
        raise ValueError("face_geometry must have shape (N, 3)")
    if target_array.shape != gaze.shape:
        raise ValueError("targets must have shape (N, 2)")
    if viewports.shape != gaze.shape:
        raise ValueError("viewport_list must have shape (N, 2)")
    if len(motion_blocks) != len(gaze):
        raise ValueError("motion_blocks must align with samples")
    if not (
        np.isfinite(gaze).all()
        and np.isfinite(head).all()
        and np.isfinite(face).all()
        and np.isfinite(target_array).all()
        and np.isfinite(viewports).all()
    ):
        raise ValueError("experiment arrays must contain only finite values")
    if (viewports <= 0).any():
        raise ValueError("viewport dimensions must be positive")

    normalized_groups = [str(group).strip() for group in motion_blocks]
    if any(not group for group in normalized_groups):
        raise ValueError("motion_blocks must not contain blank values")
    group_names = sorted(set(normalized_groups))
    if len(group_names) < 3:
        raise ValueError("nested validation requires at least three motion blocks")
    group_array = np.asarray(normalized_groups)
    motion_features = motion_conditioned_features(gaze, head, face)

    baseline_errors: list[float] = []
    challenger_errors: list[float] = []
    folds: list[dict[str, Any]] = []
    for outer_group in group_names:
        train_indices = np.flatnonzero(group_array != outer_group)
        validation_indices = np.flatnonzero(group_array == outer_group)
        inner_groups = group_array[train_indices].tolist()
        unique_targets = len(np.unique(target_array[train_indices], axis=0))

        (
            baseline_weights,
            baseline_degree,
            baseline_alpha,
            baseline_inner_error,
        ) = fit_best_stage(
            gaze[train_indices],
            target_array[train_indices],
            viewports[train_indices],
            unique_targets,
            is_stage_1=True,
            validation_groups=inner_groups,
        )
        baseline_predictions = apply_stage_chain(
            gaze[validation_indices],
            [
                _baseline_stage(
                    baseline_weights,
                    degree=baseline_degree,
                    alpha=baseline_alpha,
                )
            ],
            clamp=False,
        )

        (
            challenger_weights,
            feature_mean,
            feature_scale,
            challenger_alpha,
            challenger_inner_error,
        ) = fit_standardized_ridge(
            motion_features[train_indices],
            target_array[train_indices],
            viewports[train_indices],
            validation_groups=inner_groups,
        )
        challenger_predictions = apply_stage_chain(
            gaze[validation_indices],
            [
                _challenger_stage(
                    challenger_weights,
                    feature_mean=feature_mean,
                    feature_scale=feature_scale,
                    alpha=challenger_alpha,
                )
            ],
            head_pitch_yaw=head[validation_indices],
            face_geometry=face[validation_indices],
            clamp=False,
        )

        fold_baseline_errors = _pixel_errors(
            baseline_predictions,
            target_array[validation_indices],
            viewports[validation_indices],
        )
        fold_challenger_errors = _pixel_errors(
            challenger_predictions,
            target_array[validation_indices],
            viewports[validation_indices],
        )
        baseline_errors.extend(fold_baseline_errors.astype(float).tolist())
        challenger_errors.extend(fold_challenger_errors.astype(float).tolist())
        folds.append(
            {
                "outer_motion_block": outer_group,
                "train_motion_block_count": len(set(inner_groups)),
                "train_samples": int(len(train_indices)),
                "validation_samples": int(len(validation_indices)),
                "gaze_polynomial": {
                    "degree": int(baseline_degree),
                    "alpha": float(baseline_alpha),
                    "inner_validation_mean_px": float(baseline_inner_error),
                    **_error_statistics(fold_baseline_errors),
                },
                "motion_conditioned_ridge_v1": {
                    "alpha": float(challenger_alpha),
                    "inner_validation_mean_px": float(challenger_inner_error),
                    **_error_statistics(fold_challenger_errors),
                },
            }
        )

    baseline_array = np.asarray(baseline_errors, dtype=np.float64)
    challenger_array = np.asarray(challenger_errors, dtype=np.float64)
    baseline_macro = float(
        np.mean([fold[BASELINE_MODEL]["mean_px"] for fold in folds])
    )
    challenger_macro = float(
        np.mean([fold[CHALLENGER_MODEL]["mean_px"] for fold in folds])
    )
    promote, required, observed = motion_challenger_decision(
        baseline_macro,
        challenger_macro,
    )
    relative = observed / baseline_macro if baseline_macro > 0 else 0.0
    selected_model = CHALLENGER_MODEL if promote else BASELINE_MODEL

    return {
        "schema_version": 1,
        "validation_scheme": VALIDATION_SCHEME,
        "sample_count": int(len(gaze)),
        "outer_motion_block_count": len(group_names),
        "candidates": {
            BASELINE_MODEL: {
                "macro_mean_px": baseline_macro,
                **_error_statistics(baseline_array),
            },
            CHALLENGER_MODEL: {
                "macro_mean_px": challenger_macro,
                **_error_statistics(challenger_array),
            },
        },
        "promotion_gate": {
            "min_absolute_improvement_px": MOTION_PROMOTION_MIN_ABSOLUTE_PX,
            "min_relative_improvement": MOTION_PROMOTION_MIN_RELATIVE,
            "required_improvement_px": required,
            "observed_improvement_px": observed,
            "observed_relative_improvement": relative,
            "passed": promote,
            "selected_model": selected_model,
        },
        "folds": folds,
    }


def build_uncertainty_v2_bundle(
    gaze_pitch_yaw: np.ndarray,
    head_pitch_yaw: np.ndarray,
    face_geometry: np.ndarray,
    targets: np.ndarray,
    viewport_list: Sequence[Sequence[float]],
    motion_blocks: Sequence[str],
    sample_ids: Sequence[str],
    target_ids: Sequence[str],
    final_stages: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build end-to-end OOF evidence and final label-free score state.

    Every outer holdout is invisible to family selection, hyperparameter
    selection, stage fitting, OOD/leverage fitting, and block-jackknife member
    fitting.  Targets are used only after prediction and scoring to construct
    descriptive OOF residual/risk evidence.
    """

    gaze = np.asarray(gaze_pitch_yaw, dtype=np.float64)
    head = np.asarray(head_pitch_yaw, dtype=np.float64)
    face = np.asarray(face_geometry, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    viewports = np.asarray(viewport_list, dtype=np.float64)
    if gaze.ndim != 2 or gaze.shape[1] != 2:
        raise ValueError("gaze_pitch_yaw must have shape (N, 2)")
    if head.shape != gaze.shape:
        raise ValueError("head_pitch_yaw must have shape (N, 2)")
    if face.shape != (len(gaze), 3):
        raise ValueError("face_geometry must have shape (N, 3)")
    if target_array.shape != gaze.shape:
        raise ValueError("targets must have shape (N, 2)")
    if viewports.shape != gaze.shape:
        raise ValueError("viewport_list must have shape (N, 2)")
    if not (
        np.isfinite(gaze).all()
        and np.isfinite(head).all()
        and np.isfinite(face).all()
        and np.isfinite(target_array).all()
        and np.isfinite(viewports).all()
    ):
        raise ValueError("uncertainty experiment arrays must be finite")
    if (viewports <= 0).any():
        raise ValueError("viewport dimensions must be positive")

    definition_document = load_frozen_definition()
    definition_sha = str(definition_document["definition_sha256"])
    grid_validation = validate_complete_motion_grid(
        sample_ids,
        target_ids,
        motion_blocks,
        definition_document=definition_document,
    )
    normalized_sample_ids = [str(value).strip() for value in sample_ids]
    normalized_target_ids = [str(value).strip() for value in target_ids]
    normalized_groups = [str(value).strip() for value in motion_blocks]
    group_array = np.asarray(normalized_groups)
    group_names = list(
        definition_document["definition"]["required_grid"]["motion_block_ids"]
    )

    fit_cache: dict[tuple[int, ...], dict[str, Any]] = {}

    def fit_indices(indices: np.ndarray) -> dict[str, Any]:
        key = tuple(int(index) for index in np.sort(indices))
        if key not in fit_cache:
            fit_cache[key] = _fit_partition_selected_pipeline(
                gaze[indices],
                head[indices],
                face[indices],
                target_array[indices],
                viewports[indices],
                group_array[indices].tolist(),
            )
        return fit_cache[key]

    rows: list[dict[str, Any]] = []
    fold_evidence: list[dict[str, Any]] = []
    for outer_group in group_names:
        train_indices = np.flatnonzero(group_array != outer_group)
        holdout_indices = np.flatnonzero(group_array == outer_group)
        selected = fit_indices(train_indices)
        stages = selected["stages"]

        jackknife_stage_sets: list[list[dict[str, Any]]] = []
        jackknife_proofs: list[dict[str, Any]] = []
        training_groups = sorted(set(group_array[train_indices].tolist()))
        for jackknife_holdout in training_groups:
            member_indices = train_indices[
                group_array[train_indices] != jackknife_holdout
            ]
            member = fit_indices(member_indices)
            jackknife_stage_sets.append(member["stages"])
            jackknife_proofs.append(
                {
                    "held_out_training_motion_block_id": jackknife_holdout,
                    "fit_motion_block_ids": sorted(
                        set(group_array[member_indices].tolist())
                    ),
                    "fit_motion_block_ids_sha256": canonical_sha256(
                        sorted(set(group_array[member_indices].tolist()))
                    ),
                    "fit_sample_count": int(len(member_indices)),
                    "fit_sample_ids_sha256": canonical_sha256(
                        sorted(
                            normalized_sample_ids[index]
                            for index in member_indices
                        )
                    ),
                    "stage_signature_sha256": canonical_sha256(
                        member["stages"]
                    ),
                    "selected_model": member["selection"]["selected_model"],
                }
            )

        score_state = fit_score_state(
            gaze[train_indices],
            stages,
            group_array[train_indices].tolist(),
            jackknife_stage_sets,
            jackknife_member_holdout_motion_block_ids=training_groups,
            training_sample_ids=[
                normalized_sample_ids[index] for index in train_indices
            ],
            head_pitch_yaw=head[train_indices],
            face_geometry=face[train_indices],
            fit_scope=f"outer_training_partition_excluding_{outer_group}",
            definition_document=definition_document,
        )
        predictions = apply_stage_chain(
            gaze[holdout_indices],
            stages,
            head_pitch_yaw=head[holdout_indices],
            face_geometry=face[holdout_indices],
            clamp=True,
        )
        scored = score_samples(
            gaze[holdout_indices],
            stages,
            score_state,
            viewports[holdout_indices],
            head_pitch_yaw=head[holdout_indices],
            face_geometry=face[holdout_indices],
            definition_document=definition_document,
        )

        train_ids = [normalized_sample_ids[index] for index in train_indices]
        holdout_ids = [normalized_sample_ids[index] for index in holdout_indices]
        intersection = set(train_ids) & set(holdout_ids)
        if intersection:
            raise ValueError("outer training and holdout sample ids overlap")
        partition_proof = {
            "train_sample_count": len(train_ids),
            "holdout_sample_count": len(holdout_ids),
            "train_motion_block_ids": training_groups,
            "holdout_motion_block_id": outer_group,
            "sample_id_intersection_count": 0,
            "train_sample_ids_sha256": canonical_sha256(sorted(train_ids)),
            "holdout_sample_ids_sha256": canonical_sha256(sorted(holdout_ids)),
        }
        fold_evidence.append(
            {
                "outer_fold_id": f"motion_block:{outer_group}",
                "outer_holdout_group_id": outer_group,
                "training_partition_proof": partition_proof,
                "model_selection": selected["selection"],
                "selected_stages": stages,
                "score_state_sha256": score_state["state_sha256"],
                "score_state": score_state,
                "jackknife_members": jackknife_proofs,
            }
        )

        for local_index, sample_index in enumerate(holdout_indices):
            viewport = viewports[sample_index]
            prediction = predictions[local_index]
            target = target_array[sample_index]
            prediction_px = (prediction + 1.0) * 0.5 * viewport
            target_px = (target + 1.0) * 0.5 * viewport
            residual_px = prediction_px - target_px
            spatial_error = float(np.linalg.norm(residual_px))
            covariance_px = scored[
                "jackknife_disagreement_covariance_px"
            ][local_index]
            row = {
                "sample_id": normalized_sample_ids[sample_index],
                "outer_fold_id": f"motion_block:{outer_group}",
                "outer_holdout_group_id": outer_group,
                "target_id": normalized_target_ids[sample_index],
                "oof_prediction_x_norm": float(prediction[0]),
                "oof_prediction_y_norm": float(prediction[1]),
                "oof_prediction_x_px": float(prediction_px[0]),
                "oof_prediction_y_px": float(prediction_px[1]),
                "residual_x_px": float(residual_px[0]),
                "residual_y_px": float(residual_px[1]),
                "spatial_error_px": spatial_error,
                "spatial_error_viewport_diagonal_fraction": float(
                    spatial_error / np.linalg.norm(viewport)
                ),
                "training_only_ood_score": float(
                    scored["components"]["ood"][local_index]
                ),
                "leverage_score": float(
                    scored["components"]["leverage"][local_index]
                ),
                "jackknife_disagreement_score": float(
                    scored["components"]["disagreement"][local_index]
                ),
                "uncertainty_component_percentiles": {
                    name: float(values[local_index])
                    for name, values in scored["component_percentiles"].items()
                },
                "uncertainty_score": float(
                    scored["uncertainty_score"][local_index]
                ),
                "prediction_covariance_px": covariance_px.astype(float).tolist(),
                "prediction_covariance_definition": (
                    "whole_pipeline_block_jackknife_model_disagreement_not_"
                    "posterior_or_guaranteed_prediction_covariance"
                ),
                "score_definition_sha256": definition_sha,
                "selected_model": selected["selection"]["selected_model"],
                "training_partition_proof": partition_proof,
                "threshold": None,
                "abstention_status": ABSTENTION_STATUS,
            }
            rows.append(row)

    coverage = build_fixed_coverage_risk(
        rows,
        definition_document=definition_document,
    )

    all_indices = np.arange(len(gaze))
    final_jackknife_stages = [
        fit_indices(all_indices[group_array != group_name])["stages"]
        for group_name in group_names
    ]
    final_score_state = fit_score_state(
        gaze,
        final_stages,
        normalized_groups,
        final_jackknife_stages,
        jackknife_member_holdout_motion_block_ids=group_names,
        training_sample_ids=normalized_sample_ids,
        head_pitch_yaw=head,
        face_geometry=face,
        fit_scope="all_training_motion_blocks",
        definition_document=definition_document,
    )

    return {
        "schema_version": UNCERTAINTY_SCHEMA_VERSION,
        "status": OUTPUT_STATUS,
        "definition_sha256": definition_sha,
        "claim_boundary": definition_document["definition"]["claim_boundary"],
        "threshold": None,
        "abstention_policy": {
            "status": ABSTENTION_STATUS,
            "threshold": None,
            "quality_band": None,
        },
        "grid_validation": grid_validation,
        "oof_evidence": {
            "validation_scheme": (
                "nested_outer_motion_block_with_fold_local_family_selection"
            ),
            "effective_independent_motion_block_count": len(group_names),
            "fresh_matched_contract_capture_required": True,
            "folds": fold_evidence,
            **coverage,
        },
        "final_score_state": final_score_state,
    }

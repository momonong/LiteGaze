"""CPU-only evaluation primitives for selective three-class word-review fusion.

The class order is frozen as ``no_review < unsure < review_needed``.  This
module scores already-produced probabilities; it does not fit a model, derive
an uncertainty score from outcomes, or choose an abstention threshold.  Lower
uncertainty values must mean more reliable predictions before labels are
opened.

The crossed-cluster interval is deliberately named a *multiplier bootstrap*.
Each replicate gives every participant and every passage family one
deterministic positive exponential multiplier, then gives a row the product of
its two cluster multipliers.  Rows therefore remain linked through both axes,
including in incomplete participant-by-passage designs.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


CLASS_LABELS = ("no_review", "unsure", "review_needed")
F0_MODEL_ID = "F0_always_on_text_person_gaze"
F1_MODEL_ID = "F1_text_person"
F2_MODEL_ID = "F2_selective_exact_F1_fallback"
FIXED_COVERAGES = (1.0, 0.8, 0.6, 0.4, 0.2)
METRIC_NAMES = (
    "negative_log_likelihood",
    "multiclass_brier_score",
    "ranked_probability_score",
)
PROBABILITY_SUM_TOLERANCE = 1e-10
NLL_PROBABILITY_FLOOR = 1e-15
DEFAULT_BOOTSTRAP_RESAMPLES = 2_000
DEFAULT_BOOTSTRAP_SEED = 20_260_817
BOOTSTRAP_ALGORITHM = "sha256_exponential_crossed_cluster_multiplier_v1"
MIN_INFERENTIAL_DIAGNOSTIC_PARTICIPANTS = 8
MIN_INFERENTIAL_DIAGNOSTIC_PASSAGE_FAMILIES = 8
MIN_INFERENTIAL_DIAGNOSTIC_BOOTSTRAP_RESAMPLES = 1_000
STATISTICAL_REVIEW_BOUNDARY = (
    "requires_statistical_method_review_before_CHI_final_inference"
)


@dataclass(frozen=True)
class CrossedClusterWeights:
    """Auditable row and component weights for one bootstrap replicate."""

    row_weights: np.ndarray
    participant_weights: Mapping[str, float]
    passage_family_weights: Mapping[str, float]


def validate_class_probabilities(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    name: str = "probabilities",
) -> np.ndarray:
    """Return a validated ``N x 3`` probability matrix without renormalizing."""

    raw = np.asarray(probabilities)
    if raw.dtype.kind == "b":
        raise ValueError(f"{name} must be numeric probabilities, not booleans")
    try:
        matrix = np.asarray(probabilities, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric N x 3 matrix") from exc
    if matrix.ndim != 2 or matrix.shape[1] != len(CLASS_LABELS):
        raise ValueError(f"{name} must have shape N x {len(CLASS_LABELS)}")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError(f"{name} values must be within [0, 1]")
    row_sums = matrix.sum(axis=1)
    if not np.allclose(
        row_sums,
        np.ones(len(matrix), dtype=np.float64),
        rtol=0.0,
        atol=PROBABILITY_SUM_TOLERANCE,
    ):
        raise ValueError(
            f"{name} rows must sum to one within "
            f"{PROBABILITY_SUM_TOLERANCE:g}"
        )
    return np.ascontiguousarray(matrix)


def encode_class_labels(labels: Sequence[str | int] | np.ndarray) -> np.ndarray:
    """Encode frozen string labels or explicit integer indices ``0..2``."""

    raw = np.asarray(labels, dtype=object)
    if raw.ndim != 1 or len(raw) == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence")
    values = raw.tolist()
    if all(isinstance(value, str) for value in values):
        mapping = {label: index for index, label in enumerate(CLASS_LABELS)}
        unknown = sorted({value for value in values if value not in mapping})
        if unknown:
            raise ValueError(f"labels contain unknown classes: {unknown}")
        return np.asarray([mapping[value] for value in values], dtype=np.int64)
    if all(
        isinstance(value, (int, np.integer)) and not isinstance(value, bool)
        for value in values
    ):
        encoded = np.asarray(values, dtype=np.int64)
        if np.any((encoded < 0) | (encoded >= len(CLASS_LABELS))):
            raise ValueError("integer labels must be in the inclusive range 0..2")
        return encoded
    raise ValueError("labels must be all frozen class names or all integer indices")


def probability_metrics(
    labels: Sequence[str | int] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
) -> dict[str, Any]:
    """Compute NLL, multiclass Brier, and normalized ranked probability score."""

    encoded = encode_class_labels(labels)
    matrix = validate_class_probabilities(probabilities)
    if len(encoded) != len(matrix):
        raise ValueError("labels and probabilities must contain the same rows")
    losses = _per_row_losses(encoded, matrix)
    return {
        "row_count": len(encoded),
        "class_order": list(CLASS_LABELS),
        "negative_log_likelihood": float(
            np.mean(losses["negative_log_likelihood"])
        ),
        # The multiclass Brier definition is the sum across all three classes.
        "multiclass_brier_score": float(
            np.mean(losses["multiclass_brier_score"])
        ),
        # RPS is divided by K-1, so its range is [0, 1].
        "ranked_probability_score": float(
            np.mean(losses["ranked_probability_score"])
        ),
        "ranked_probability_score_definition": (
            "mean_squared_cumulative_probability_error_over_K_minus_1"
        ),
        "nll_probability_floor": NLL_PROBABILITY_FLOOR,
        "lower_is_better": True,
    }


def fixed_coverage_risk_curve(
    labels: Sequence[str | int] | np.ndarray,
    f2_probabilities: Sequence[Sequence[float]] | np.ndarray,
    uncertainty_scores: Sequence[float] | np.ndarray,
    *,
    sample_ids: Sequence[str],
    gaze_eligible: Sequence[bool] | np.ndarray,
    f1_probabilities: Sequence[Sequence[float]] | np.ndarray,
    f0_probabilities: Sequence[Sequence[float]] | np.ndarray,
) -> list[dict[str, Any]]:
    """Score conditional acceptance and deployed hybrid risk on a fixed grid.

    Smaller uncertainty values are selected first *within gaze-eligible F2
    rows*.  Conditional accepted risk scores only that selected eligible
    subset.  Deployed hybrid risk scores all rows, using F2 on selected rows
    and exact F1 everywhere else.  Ties are broken by unique sample ID, never
    by label or loss.  Eligible coverage uses ``ceil(N_eligible * coverage)``.
    """

    encoded = encode_class_labels(labels)
    f2 = validate_class_probabilities(
        f2_probabilities, name="f2_probabilities"
    )
    row_count = len(f2)
    if len(encoded) != row_count:
        raise ValueError("labels and F2 probabilities must align")
    uncertainty = _finite_vector(
        uncertainty_scores, row_count=row_count, name="uncertainty_scores"
    )
    if np.any(uncertainty < 0.0):
        raise ValueError("uncertainty_scores must be non-negative")
    identifiers = _identifier_vector(
        sample_ids, row_count=row_count, name="sample_ids", require_unique=True
    )
    eligible = _boolean_vector(
        gaze_eligible, row_count=row_count, name="gaze_eligible"
    )
    f1 = validate_class_probabilities(
        f1_probabilities,
        name="f1_probabilities",
    )
    f0 = validate_class_probabilities(
        f0_probabilities,
        name="f0_probabilities",
    )
    if len(f1) != row_count or len(f0) != row_count:
        raise ValueError("F0, F1, and F2 probabilities must align")

    eligible_indices = [index for index in range(row_count) if eligible[index]]
    ineligible_indices = [index for index in range(row_count) if not eligible[index]]
    if ineligible_indices and np.any(f2[ineligible_indices] != f1[ineligible_indices]):
        raise ValueError("ineligible F2 rows must equal F1 probabilities exactly")
    eligible_ranking = sorted(
        eligible_indices,
        key=lambda index: (float(uncertainty[index]), identifiers[index]),
    )
    curve: list[dict[str, Any]] = []
    for requested in FIXED_COVERAGES:
        if eligible_ranking:
            selected_eligible_count = int(
                math.ceil(len(eligible_ranking) * requested - 1e-12)
            )
        else:
            selected_eligible_count = 0
        selected_eligible = eligible_ranking[:selected_eligible_count]
        selected = np.asarray(selected_eligible, dtype=np.int64)
        conditional_metrics: dict[str, dict[str, Any]] | None = None
        conditional_differences: dict[str, dict[str, float]] | None = None
        if len(selected):
            selected_labels = encoded[selected]
            conditional_metrics = _three_model_metrics(
                selected_labels,
                f0=f0[selected],
                f1=f1[selected],
                f2=f2[selected],
            )
            conditional_differences = _f2_comparison_differences(
                conditional_metrics
            )

        deployed_hybrid = f1.copy()
        if len(selected):
            deployed_hybrid[selected] = f2[selected]
        deployed_metrics = _three_model_metrics(
            encoded,
            f0=f0,
            f1=f1,
            f2=deployed_hybrid,
        )
        item: dict[str, Any] = {
            "requested_eligible_gaze_coverage": requested,
            "eligible_gaze_pool_count": len(eligible_ranking),
            "selected_eligible_gaze_count": selected_eligible_count,
            "achieved_eligible_gaze_coverage": (
                selected_eligible_count / len(eligible_ranking)
                if eligible_ranking
                else None
            ),
            "ineligible_exact_f1_rows_retained": len(ineligible_indices),
            "unselected_eligible_rows_falling_back_to_f1": (
                len(eligible_ranking) - selected_eligible_count
            ),
            "total_rows_falling_back_to_f1": row_count - selected_eligible_count,
            "overall_evaluation_row_count": row_count,
            "overall_evaluation_coverage": 1.0,
            "maximum_selected_eligible_uncertainty": (
                float(np.max(uncertainty[selected_eligible]))
                if selected_eligible
                else None
            ),
            "selected_eligible_sample_ids": [
                identifiers[index] for index in selected_eligible
            ],
            "conditional_selected_eligible_metrics": conditional_metrics,
            "conditional_selected_eligible_metric_differences": (
                conditional_differences
            ),
            "deployed_hybrid_all_row_metrics": deployed_metrics,
            "deployed_hybrid_all_row_metric_differences": (
                _f2_comparison_differences(deployed_metrics)
            ),
            "selection_rule": (
                "eligible_uncertainty_ascending_then_sample_id;_"
                "F1_fallback_on_every_unselected_row"
            ),
            "conditional_risk_definition": (
                "score_only_selected_gaze_eligible_rows"
            ),
            "deployed_hybrid_risk_definition": (
                "score_all_rows_with_F2_on_selected_eligible_and_F1_elsewhere"
            ),
            "uncertainty_contract": (
                "caller_supplied_label_free_lower_is_more_certain"
            ),
        }
        curve.append(item)
    return curve


def check_exact_f1_fallback(
    f2_probabilities: Sequence[Sequence[float]] | np.ndarray,
    f1_probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    gaze_available: Sequence[bool] | np.ndarray,
    gaze_eligible: Sequence[bool] | np.ndarray,
    sample_ids: Sequence[str],
) -> dict[str, Any]:
    """Require exact F1 fallback whenever gaze is missing or ineligible."""

    f2 = validate_class_probabilities(
        f2_probabilities, name="f2_probabilities"
    )
    f1 = validate_class_probabilities(
        f1_probabilities,
        name="f1_probabilities",
    )
    if f2.shape != f1.shape:
        raise ValueError("F1 and F2 probabilities must have the same shape")
    row_count = len(f2)
    available = _boolean_vector(
        gaze_available, row_count=row_count, name="gaze_available"
    )
    eligible = _boolean_vector(
        gaze_eligible, row_count=row_count, name="gaze_eligible"
    )
    identifiers = _identifier_vector(
        sample_ids, row_count=row_count, name="sample_ids", require_unique=True
    )
    invalid_eligibility = eligible & ~available
    if np.any(invalid_eligibility):
        first = int(np.flatnonzero(invalid_eligibility)[0])
        raise ValueError(
            "gaze cannot be eligible when unavailable; "
            f"first sample={identifiers[first]}"
        )
    fallback_mask = ~(available & eligible)
    mismatches = np.flatnonzero(
        fallback_mask & np.any(f2 != f1, axis=1)
    )
    if len(mismatches):
        first = int(mismatches[0])
        raise ValueError(
            "missing or ineligible gaze must use exact F1 text/person "
            f"probabilities; mismatch_count={len(mismatches)} "
            f"first_sample={identifiers[first]}"
        )
    return {
        "passed": True,
        "comparison": "F2_equals_F1_bit_exact_float64_after_validation",
        "fallback_row_count": int(fallback_mask.sum()),
        "eligible_gaze_row_count": int((available & eligible).sum()),
        "row_count": row_count,
    }


def crossed_cluster_multiplier_weights(
    participant_ids: Sequence[str],
    passage_family_ids: Sequence[str],
    *,
    seed: int,
    replicate: int,
) -> CrossedClusterWeights:
    """Build one deterministic participant-by-passage multiplier replicate."""

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if (
        isinstance(replicate, bool)
        or not isinstance(replicate, (int, np.integer))
        or int(replicate) < 0
    ):
        raise ValueError("replicate must be a non-negative integer")
    participants = _identifier_vector(
        participant_ids,
        row_count=len(participant_ids),
        name="participant_ids",
        require_unique=False,
    )
    passages = _identifier_vector(
        passage_family_ids,
        row_count=len(participants),
        name="passage_family_ids",
        require_unique=False,
    )
    participant_weights = {
        group: _exponential_multiplier(
            seed=int(seed), replicate=int(replicate), axis="participant", group=group
        )
        for group in sorted(set(participants))
    }
    passage_weights = {
        group: _exponential_multiplier(
            seed=int(seed),
            replicate=int(replicate),
            axis="passage_family",
            group=group,
        )
        for group in sorted(set(passages))
    }
    row_weights = np.asarray(
        [
            participant_weights[participant] * passage_weights[passage]
            for participant, passage in zip(participants, passages, strict=True)
        ],
        dtype=np.float64,
    )
    return CrossedClusterWeights(
        row_weights=row_weights,
        participant_weights=participant_weights,
        passage_family_weights=passage_weights,
    )


def crossed_cluster_multiplier_bootstrap_difference(
    labels: Sequence[str | int] | np.ndarray,
    model_probabilities: Sequence[Sequence[float]] | np.ndarray,
    reference_probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    model_id: str,
    reference_id: str,
    participant_ids: Sequence[str],
    passage_family_ids: Sequence[str],
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap model-minus-reference loss through both crossed cluster axes.

    Point differences are always descriptive.  Percentile intervals and the
    bootstrap improvement probability are withheld unless both frozen cluster
    minima and the frozen minimum resample count are met.  Availability means
    only that an inferential diagnostic can be emitted; it does not establish
    sample-size sufficiency or approve this custom method for final inference.
    """

    if (
        isinstance(resamples, bool)
        or not isinstance(resamples, (int, np.integer))
        or int(resamples) <= 0
    ):
        raise ValueError("resamples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    for name, value in (("model_id", model_id), ("reference_id", reference_id)):
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{name} must be a normalized non-blank string")
    if model_id == reference_id:
        raise ValueError("model_id and reference_id must differ")
    encoded = encode_class_labels(labels)
    model = validate_class_probabilities(
        model_probabilities, name="model_probabilities"
    )
    reference = validate_class_probabilities(
        reference_probabilities,
        name="reference_probabilities",
    )
    if model.shape != reference.shape or len(encoded) != len(model):
        raise ValueError(
            "labels, model probabilities, and reference probabilities must align"
        )
    row_count = len(model)
    participants = _identifier_vector(
        participant_ids,
        row_count=row_count,
        name="participant_ids",
        require_unique=False,
    )
    passages = _identifier_vector(
        passage_family_ids,
        row_count=row_count,
        name="passage_family_ids",
        require_unique=False,
    )
    unique_participants = sorted(set(participants))
    unique_passages = sorted(set(passages))
    if len(unique_participants) < 2:
        raise ValueError("crossed-cluster bootstrap requires at least two participants")
    if len(unique_passages) < 2:
        raise ValueError(
            "crossed-cluster bootstrap requires at least two passage families"
        )

    model_losses = _per_row_losses(encoded, model)
    reference_losses = _per_row_losses(encoded, reference)
    loss_difference = np.column_stack(
        [model_losses[name] - reference_losses[name] for name in METRIC_NAMES]
    )
    participant_index = {
        group: index for index, group in enumerate(unique_participants)
    }
    passage_index = {group: index for index, group in enumerate(unique_passages)}
    cells: dict[tuple[int, int], list[int]] = {}
    for row, (participant, passage) in enumerate(
        zip(participants, passages, strict=True)
    ):
        key = (participant_index[participant], passage_index[passage])
        cells.setdefault(key, []).append(row)
    ordered_cells = sorted(cells)
    cell_participant = np.asarray([key[0] for key in ordered_cells], dtype=np.int64)
    cell_passage = np.asarray([key[1] for key in ordered_cells], dtype=np.int64)
    cell_counts = np.asarray(
        [len(cells[key]) for key in ordered_cells], dtype=np.float64
    )
    cell_loss_sums = np.asarray(
        [
            [
                math.fsum(float(loss_difference[row, metric]) for row in cells[key])
                for metric in range(len(METRIC_NAMES))
            ]
            for key in ordered_cells
        ],
        dtype=np.float64,
    )

    replicates = np.empty((int(resamples), len(METRIC_NAMES)), dtype=np.float64)
    for replicate in range(int(resamples)):
        participant_weights = np.asarray(
            [
                _exponential_multiplier(
                    seed=int(seed),
                    replicate=replicate,
                    axis="participant",
                    group=group,
                )
                for group in unique_participants
            ],
            dtype=np.float64,
        )
        passage_weights = np.asarray(
            [
                _exponential_multiplier(
                    seed=int(seed),
                    replicate=replicate,
                    axis="passage_family",
                    group=group,
                )
                for group in unique_passages
            ],
            dtype=np.float64,
        )
        cell_weights = (
            participant_weights[cell_participant] * passage_weights[cell_passage]
        )
        denominator = float(np.dot(cell_weights, cell_counts))
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise RuntimeError("crossed-cluster replicate has invalid total weight")
        replicates[replicate] = (cell_weights @ cell_loss_sums) / denominator

    point = np.mean(loss_difference, axis=0)
    cluster_structure_minima_met = (
        len(unique_participants) >= MIN_INFERENTIAL_DIAGNOSTIC_PARTICIPANTS
        and len(unique_passages)
        >= MIN_INFERENTIAL_DIAGNOSTIC_PASSAGE_FAMILIES
    )
    bootstrap_resample_minimum_met = (
        int(resamples) >= MIN_INFERENTIAL_DIAGNOSTIC_BOOTSTRAP_RESAMPLES
    )
    inferential_diagnostic_available = (
        cluster_structure_minima_met and bootstrap_resample_minimum_met
    )
    metrics: dict[str, Any] = {}
    for index, name in enumerate(METRIC_NAMES):
        values = replicates[:, index]
        metrics[name] = {
            "point_difference_model_minus_reference": float(point[index]),
            "ci95_low": (
                _linear_quantile(values, 0.025)
                if inferential_diagnostic_available
                else None
            ),
            "ci95_high": (
                _linear_quantile(values, 0.975)
                if inferential_diagnostic_available
                else None
            ),
            "bootstrap_probability_model_better": (
                float(np.mean(values < 0.0))
                if inferential_diagnostic_available
                else None
            ),
            "lower_is_better": True,
        }
    return {
        "comparison_id": f"{model_id}_minus_{reference_id}",
        "model_id": model_id,
        "reference_id": reference_id,
        "algorithm": BOOTSTRAP_ALGORITHM,
        "bootstrap_type": "crossed_cluster_multiplier_bootstrap",
        "cluster_axes": ["participant_id", "passage_family_id"],
        "participant_cluster_count": len(unique_participants),
        "passage_family_cluster_count": len(unique_passages),
        "observed_crossed_cell_count": len(ordered_cells),
        "row_count": row_count,
        "resamples": int(resamples),
        "seed": int(seed),
        "inference_evaluability": {
            "inferential_diagnostic_available": (
                inferential_diagnostic_available
            ),
            "status": (
                "inferential_diagnostic_available"
                if inferential_diagnostic_available
                else "inferential_diagnostic_unavailable"
            ),
            "cluster_structure_minima_met": cluster_structure_minima_met,
            "bootstrap_resample_minimum_met": (
                bootstrap_resample_minimum_met
            ),
            "minimum_participant_clusters": (
                MIN_INFERENTIAL_DIAGNOSTIC_PARTICIPANTS
            ),
            "minimum_passage_family_clusters": (
                MIN_INFERENTIAL_DIAGNOSTIC_PASSAGE_FAMILIES
            ),
            "minimum_bootstrap_resamples": (
                MIN_INFERENTIAL_DIAGNOSTIC_BOOTSTRAP_RESAMPLES
            ),
            "intervals_withheld": not inferential_diagnostic_available,
            "sample_size_sufficiency_established": False,
            "warning": (
                "diagnostic_only_not_sample_size_sufficiency"
                if inferential_diagnostic_available
                else (
                    "descriptive_point_differences_only;_inferential_"
                    "diagnostic_withheld_until_all_frozen_minima_are_met"
                )
            ),
        },
        "claim_boundary": STATISTICAL_REVIEW_BOUNDARY,
        "group_integrity": {
            "one_multiplier_shared_by_all_rows_in_participant": True,
            "one_multiplier_shared_by_all_rows_in_passage_family": True,
            "row_weight_is_product_of_crossed_components": True,
        },
        "metrics": metrics,
    }


def deterministic_label_shuffle(
    labels: Sequence[str | int] | np.ndarray,
    *,
    sample_ids: Sequence[str],
    seed: int,
) -> list[str]:
    """Create an order-invariant, class-count-preserving label sentinel."""

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    encoded = encode_class_labels(labels)
    identifiers = _identifier_vector(
        sample_ids, row_count=len(encoded), name="sample_ids", require_unique=True
    )
    assignment_order = sorted(
        range(len(encoded)),
        key=lambda index: _shuffle_digest(int(seed), identifiers[index]),
    )
    shuffled = np.empty(len(encoded), dtype=np.int64)
    for index, label in zip(assignment_order, sorted(encoded.tolist()), strict=True):
        shuffled[index] = label
    return [CLASS_LABELS[int(value)] for value in shuffled]


def label_shuffle_sentinel_metrics(
    labels: Sequence[str | int] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    sample_ids: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    """Score fixed predictions against the deterministic shuffled-label sentinel."""

    shuffled = deterministic_label_shuffle(labels, sample_ids=sample_ids, seed=seed)
    counts = {label: shuffled.count(label) for label in CLASS_LABELS}
    return {
        "sentinel": "sha256_count_preserving_label_shuffle_v1",
        "seed": int(seed),
        "class_counts": counts,
        "metrics": probability_metrics(shuffled, probabilities),
        "interpretation": "sanity_sentinel_not_a_null_population_model",
    }


def evaluate_selective_fusion(
    labels: Sequence[str | int] | np.ndarray,
    f0_probabilities: Sequence[Sequence[float]] | np.ndarray,
    f1_probabilities: Sequence[Sequence[float]] | np.ndarray,
    f2_probabilities: Sequence[Sequence[float]] | np.ndarray,
    uncertainty_scores: Sequence[float] | np.ndarray,
    *,
    sample_ids: Sequence[str],
    participant_ids: Sequence[str],
    passage_family_ids: Sequence[str],
    gaze_available: Sequence[bool] | np.ndarray,
    gaze_eligible: Sequence[bool] | np.ndarray,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run the complete frozen, fit-free selective-fusion evaluation surface."""

    fallback = check_exact_f1_fallback(
        f2_probabilities,
        f1_probabilities,
        gaze_available=gaze_available,
        gaze_eligible=gaze_eligible,
        sample_ids=sample_ids,
    )
    model_metrics = _three_model_metrics(
        labels,
        f0=f0_probabilities,
        f1=f1_probabilities,
        f2=f2_probabilities,
    )
    f2_minus_f1 = crossed_cluster_multiplier_bootstrap_difference(
        labels,
        f2_probabilities,
        f1_probabilities,
        model_id=F2_MODEL_ID,
        reference_id=F1_MODEL_ID,
        participant_ids=participant_ids,
        passage_family_ids=passage_family_ids,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    f2_minus_f0 = crossed_cluster_multiplier_bootstrap_difference(
        labels,
        f2_probabilities,
        f0_probabilities,
        model_id=F2_MODEL_ID,
        reference_id=F0_MODEL_ID,
        participant_ids=participant_ids,
        passage_family_ids=passage_family_ids,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    inferential_diagnostic_available = (
        f2_minus_f1["inference_evaluability"][
            "inferential_diagnostic_available"
        ]
        and f2_minus_f0["inference_evaluability"][
            "inferential_diagnostic_available"
        ]
    )
    evaluability_detail = f2_minus_f1["inference_evaluability"]
    return {
        "schema_version": 2,
        "evaluation_id": "chi-selective-word-review-fusion-v1",
        "class_order": list(CLASS_LABELS),
        "model_ids": [F0_MODEL_ID, F1_MODEL_ID, F2_MODEL_ID],
        "fixed_coverages": list(FIXED_COVERAGES),
        "ranked_probability_score_definition": (
            "mean_squared_cumulative_probability_error_over_K_minus_1"
        ),
        "model_metrics": model_metrics,
        "metric_differences": _f2_comparison_differences(model_metrics),
        "fallback": fallback,
        "coverage_risk": fixed_coverage_risk_curve(
            labels,
            f2_probabilities,
            uncertainty_scores,
            sample_ids=sample_ids,
            gaze_eligible=gaze_eligible,
            f1_probabilities=f1_probabilities,
            f0_probabilities=f0_probabilities,
        ),
        "crossed_cluster_comparisons": {
            "F2_minus_F1": f2_minus_f1,
            "F2_minus_F0": f2_minus_f0,
        },
        "inference_evaluability": {
            "inferential_diagnostic_available": (
                inferential_diagnostic_available
            ),
            "minimum_participant_clusters": (
                MIN_INFERENTIAL_DIAGNOSTIC_PARTICIPANTS
            ),
            "minimum_passage_family_clusters": (
                MIN_INFERENTIAL_DIAGNOSTIC_PASSAGE_FAMILIES
            ),
            "minimum_bootstrap_resamples": (
                MIN_INFERENTIAL_DIAGNOSTIC_BOOTSTRAP_RESAMPLES
            ),
            "cluster_structure_minima_met": evaluability_detail[
                "cluster_structure_minima_met"
            ],
            "bootstrap_resample_minimum_met": evaluability_detail[
                "bootstrap_resample_minimum_met"
            ],
            "intervals_withheld": not inferential_diagnostic_available,
            "status": (
                "inferential_diagnostic_available"
                if inferential_diagnostic_available
                else "inferential_diagnostic_unavailable"
            ),
            "sample_size_sufficiency_established": False,
            "warning": (
                "diagnostic_only_not_sample_size_sufficiency"
                if inferential_diagnostic_available
                else "descriptive_point_differences_only"
            ),
        },
        "claim_boundary": STATISTICAL_REVIEW_BOUNDARY,
        "label_shuffle_sentinel": label_shuffle_sentinel_metrics(
            labels,
            f2_probabilities,
            sample_ids=sample_ids,
            seed=int(bootstrap_seed) + 1,
        ),
        "compute": {"device": "cpu", "gpu_used": False, "model_fit": False},
        "threshold_selected": False,
        "production_model_changed": False,
    }


def _three_model_metrics(
    labels: Sequence[str | int] | np.ndarray,
    *,
    f0: Sequence[Sequence[float]] | np.ndarray,
    f1: Sequence[Sequence[float]] | np.ndarray,
    f2: Sequence[Sequence[float]] | np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Return aligned F0/F1/F2 metrics under frozen comparator IDs."""

    encoded = encode_class_labels(labels)
    matrices = {
        F0_MODEL_ID: validate_class_probabilities(f0, name="f0_probabilities"),
        F1_MODEL_ID: validate_class_probabilities(f1, name="f1_probabilities"),
        F2_MODEL_ID: validate_class_probabilities(f2, name="f2_probabilities"),
    }
    if any(len(matrix) != len(encoded) for matrix in matrices.values()):
        raise ValueError("labels and F0, F1, and F2 probabilities must align")
    return {
        model_id: probability_metrics(encoded, matrix)
        for model_id, matrix in matrices.items()
    }


def _f2_comparison_differences(
    model_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    """Return frozen F2-minus-F1 and F2-minus-F0 loss differences."""

    f2_metrics = model_metrics[F2_MODEL_ID]
    return {
        comparison_id: {
            metric: float(f2_metrics[metric] - model_metrics[reference_id][metric])
            for metric in METRIC_NAMES
        }
        for comparison_id, reference_id in (
            ("F2_minus_F1", F1_MODEL_ID),
            ("F2_minus_F0", F0_MODEL_ID),
        )
    }


def _per_row_losses(
    encoded_labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, np.ndarray]:
    row_indices = np.arange(len(encoded_labels), dtype=np.int64)
    true_probability = probabilities[row_indices, encoded_labels]
    nll = -np.log(np.maximum(true_probability, NLL_PROBABILITY_FLOOR))
    one_hot = np.eye(len(CLASS_LABELS), dtype=np.float64)[encoded_labels]
    brier = np.sum((probabilities - one_hot) ** 2, axis=1)
    predicted_cumulative = np.cumsum(probabilities, axis=1)[:, :-1]
    observed_cumulative = np.cumsum(one_hot, axis=1)[:, :-1]
    rps = np.mean((predicted_cumulative - observed_cumulative) ** 2, axis=1)
    return {
        "negative_log_likelihood": nll,
        "multiclass_brier_score": brier,
        "ranked_probability_score": rps,
    }


def _identifier_vector(
    values: Sequence[str],
    *,
    row_count: int,
    name: str,
    require_unique: bool,
) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of identifiers")
    identifiers = list(values)
    if len(identifiers) != row_count:
        raise ValueError(f"{name} must contain exactly {row_count} rows")
    for value in identifiers:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{name} must contain normalized non-blank strings")
    if require_unique and len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} must be unique")
    return identifiers


def _finite_vector(
    values: Sequence[float] | np.ndarray,
    *,
    row_count: int,
    name: str,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind == "b":
        raise ValueError(f"{name} must be numeric, not boolean")
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if vector.ndim != 1 or len(vector) != row_count:
        raise ValueError(f"{name} must contain exactly {row_count} rows")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _boolean_vector(
    values: Sequence[bool] | np.ndarray,
    *,
    row_count: int,
    name: str,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) != row_count or raw.dtype.kind != "b":
        raise ValueError(f"{name} must contain exactly {row_count} booleans")
    return np.asarray(raw, dtype=bool)


def _exponential_multiplier(
    *,
    seed: int,
    replicate: int,
    axis: str,
    group: str,
) -> float:
    material = (
        f"{seed}\0{replicate}\0{axis}\0{len(group.encode('utf-8'))}\0{group}"
    ).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    uniform = (integer + 0.5) / float(1 << 64)
    return -math.log(uniform)


def _shuffle_digest(seed: int, sample_id: str) -> bytes:
    material = f"{seed}\0{len(sample_id.encode('utf-8'))}\0{sample_id}".encode(
        "utf-8"
    )
    return hashlib.sha256(material).digest()


def _linear_quantile(values: np.ndarray, probability: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if not len(ordered):
        raise ValueError("quantile requires at least one value")
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


__all__ = [
    "BOOTSTRAP_ALGORITHM",
    "CLASS_LABELS",
    "CrossedClusterWeights",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "F0_MODEL_ID",
    "F1_MODEL_ID",
    "F2_MODEL_ID",
    "FIXED_COVERAGES",
    "MIN_INFERENTIAL_DIAGNOSTIC_BOOTSTRAP_RESAMPLES",
    "MIN_INFERENTIAL_DIAGNOSTIC_PARTICIPANTS",
    "MIN_INFERENTIAL_DIAGNOSTIC_PASSAGE_FAMILIES",
    "STATISTICAL_REVIEW_BOUNDARY",
    "METRIC_NAMES",
    "check_exact_f1_fallback",
    "crossed_cluster_multiplier_bootstrap_difference",
    "crossed_cluster_multiplier_weights",
    "deterministic_label_shuffle",
    "encode_class_labels",
    "evaluate_selective_fusion",
    "fixed_coverage_risk_curve",
    "label_shuffle_sentinel_metrics",
    "probability_metrics",
    "validate_class_probabilities",
]

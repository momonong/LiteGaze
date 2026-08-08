"""Run the frozen CPU-only personalized text-backbone feasibility study.

The language-model features are cached and label-free. This runner never loads
PyTorch or a language model: it fits small Ridge models, performs nested
participant/text holdouts, and applies the preregistered abstaining selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.cognition.generalization import (
    fit_standardized_ridge,
    paired_bootstrap_mean_difference,
    predict_standardized_ridge,
    safe_spearman,
    stable_group_folds,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "docs/experiments/protocols/2026-08-08-personalized-backbone-matching-feasibility-v1.json"
)
DEFAULT_GECO_PATH = PROJECT_ROOT / "data/geco/L2ReadingData.csv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/geco/text_modeling/personalized-backbone-matching-feasibility-v1/run-001"
)
DEFAULT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs/experiments/results/2026-08-08-personalized-backbone-matching-feasibility-v1-run-001.json"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "docs/experiments/2026-08-08-personalized-backbone-matching-feasibility-v1-run-001.md"
)

SOURCE_COLUMNS = (
    "PP_NR",
    "GROUP",
    "LANGUAGE",
    "PART",
    "TRIAL",
    "WORD_ID_WITHIN_TRIAL",
    "WORD",
    "WORD_TOTAL_READING_TIME",
)
IDENTITY_COLUMNS = ("Text_ID", "IA_ID", "IA_LABEL")
GECO_PUNCTUATION_REPAIRS = {
    "\u00d4\u00c7\u00f4": "\u2013",
    "\u00d4\u00c7\u00a3": "\u201c",
    "\u00d4\u00c7?": "\u201d",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _normalize_word(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).strip()
    for mojibake, punctuation in GECO_PUNCTUATION_REPAIRS.items():
        normalized = normalized.replace(mojibake, punctuation)
    return normalized


def _prefixed_feature_columns(model_key: str, columns: Sequence[str]) -> list[str]:
    return [f"{model_key}__{column}" for column in columns]


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != "personalized-backbone-matching-feasibility-v1":
        raise ValueError("unexpected personalized-backbone protocol")
    if payload["claim_scope"]["study_role"] != "exploratory_feasibility_only":
        raise ValueError("GECO personalized matching must remain exploratory")
    if payload["source_policy"].get("trust_remote_code") is not False:
        raise ValueError("trust_remote_code must remain false")
    if payload["dataset"].get("question_answer_dataset_used") is not False:
        raise ValueError("question-answer data is prohibited")
    if payload["compute"].get("feasibility_device") != "cpu":
        raise ValueError("the first-stage feasibility gate must remain CPU-only")
    if payload["compute"].get("feasibility_gpu_hours") != 0.0:
        raise ValueError("the first-stage GPU budget must remain zero")

    backbones = payload["frozen_backbones"]
    if [item["key"] for item in backbones] != [
        "gpt2",
        "pythia_410m_deduped_full",
    ]:
        raise ValueError("v1 must contain exactly the frozen incumbent and challenger")
    excluded = tuple(
        prefix.lower() for prefix in payload["source_policy"]["excluded_model_id_prefixes"]
    )
    for item in (*backbones, *payload["conditional_expansion_bank"]):
        if not re.fullmatch(r"[0-9a-f]{40}", str(item["revision"])):
            raise ValueError(f"mutable model revision: {item['key']}")
        model_id = str(item["model_id"]).lower()
        if any(model_id.startswith(prefix) for prefix in excluded):
            raise ValueError(f"excluded model source: {item['model_id']}")

    calibration = payload["analysis"]["calibration_text_selection"]
    if calibration["primary_budget"] not in calibration["budgets"]:
        raise ValueError("primary calibration budget is not in the learning curve")
    return payload


def _metadata_path(feature_path: Path) -> Path:
    suffix = ".features.csv"
    if not feature_path.name.endswith(suffix):
        raise ValueError(f"unexpected feature filename: {feature_path}")
    return feature_path.with_name(feature_path.name.removesuffix(suffix) + ".metadata.json")


def _read_feature_table(
    feature_path: Path,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Read labels literally while preserving numeric empty cells as missing."""
    features = pd.read_csv(
        feature_path,
        usecols=list(columns),
        dtype={"Text_ID": str, "IA_ID": int, "IA_LABEL": str},
        keep_default_na=False,
        low_memory=False,
    )
    for column in columns:
        if column not in IDENTITY_COLUMNS:
            features[column] = pd.to_numeric(features[column], errors="coerce")
    if features["IA_LABEL"].eq("").any():
        raise ValueError(f"empty display label in {feature_path}")
    return features


def _load_feature_cache(
    spec: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_path = PROJECT_ROOT / str(spec["feature_path"])
    metadata_path = _metadata_path(feature_path)
    if _sha256(feature_path) != spec["feature_sha256"]:
        raise RuntimeError(f"feature hash drift for {spec['key']}")
    if _sha256(metadata_path) != spec["metadata_sha256"]:
        raise RuntimeError(f"metadata hash drift for {spec['key']}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("source_sha256") != protocol["dataset"]["sha256"]:
        raise RuntimeError(f"feature source drift for {spec['key']}")
    if metadata.get("feature_sha256") != spec["feature_sha256"]:
        raise RuntimeError(f"metadata feature hash mismatch for {spec['key']}")
    model = metadata.get("model", {})
    if model.get("resolved_model_commit") != spec["revision"]:
        raise RuntimeError(f"resolved model revision drift for {spec['key']}")
    if model.get("trust_remote_code") is not False:
        raise RuntimeError(f"remote code was enabled for {spec['key']}")
    if metadata.get("outcome_columns_read_during_extraction") is not False:
        raise RuntimeError(f"outcomes entered feature extraction for {spec['key']}")
    if metadata.get("question_answer_dataset_used") is not False:
        raise RuntimeError(f"question-answer data entered {spec['key']} features")

    columns = [
        *IDENTITY_COLUMNS,
        *protocol["analysis"]["model_feature_set"],
        "is_context_scored",
    ]
    features = _read_feature_table(feature_path, columns)
    if features.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError(f"duplicate feature identity for {spec['key']}")
    rename = {
        column: f"{spec['key']}__{column}"
        for column in columns
        if column not in IDENTITY_COLUMNS
    }
    return features.rename(columns=rename), {
        "key": spec["key"],
        "model_id": spec["model_id"],
        "revision": spec["revision"],
        "feature_sha256": spec["feature_sha256"],
        "metadata_sha256": spec["metadata_sha256"],
        "rows": len(features),
    }


def load_analysis_frame(
    protocol: Mapping[str, Any],
    *,
    geco_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset = protocol["dataset"]
    source_hash = _sha256(geco_path)
    if source_hash != dataset["sha256"]:
        raise RuntimeError("GECO source hash does not match the frozen protocol")
    raw = pd.read_csv(
        geco_path,
        usecols=list(SOURCE_COLUMNS),
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    if len(raw) != int(dataset["expected_rows"]):
        raise RuntimeError("unexpected GECO row count")
    if set(raw["GROUP"].unique()) != {dataset["reader_group"]}:
        raise RuntimeError("unexpected GECO reader group")
    if set(raw["LANGUAGE"].unique()) != {dataset["stimulus_language"]}:
        raise RuntimeError("unexpected GECO stimulus language")

    raw["Participant_ID"] = raw["PP_NR"].astype(str)
    raw["Text_ID"] = raw["PART"].astype(str) + ":" + raw["TRIAL"].astype(str)
    raw["IA_ID"] = pd.to_numeric(
        raw["WORD_ID_WITHIN_TRIAL"], errors="raise"
    ).astype(int)
    raw["IA_LABEL"] = raw["WORD"].map(_normalize_word)
    raw["duration"] = pd.to_numeric(
        raw[protocol["analysis"]["primary_outcome"]["source_column"]],
        errors="coerce",
    )
    if raw.duplicated(["Participant_ID", "Text_ID", "IA_ID"]).any():
        raise ValueError("duplicate participant/text/item outcome row")
    if raw["Participant_ID"].nunique() != int(dataset["expected_participants"]):
        raise RuntimeError("unexpected GECO participant count")
    if raw["Text_ID"].nunique() != int(dataset["expected_texts"]):
        raise RuntimeError("unexpected GECO text count")

    frame = raw[
        ["Participant_ID", "Text_ID", "IA_ID", "IA_LABEL", "duration"]
    ].copy()
    cache_fingerprints: list[dict[str, Any]] = []
    context_columns: list[str] = []
    for spec in protocol["frozen_backbones"]:
        features, fingerprint = _load_feature_cache(spec, protocol)
        frame = frame.merge(
            features,
            on=list(IDENTITY_COLUMNS),
            how="left",
            validate="many_to_one",
        )
        cache_fingerprints.append(fingerprint)
        context_columns.append(f"{spec['key']}__is_context_scored")

    if frame[context_columns].isna().any().any():
        raise RuntimeError("outcome rows failed to align to a frozen feature cache")
    if not frame[context_columns].nunique(axis=1).eq(1).all():
        raise RuntimeError("backbones disagree on the context-scored row boundary")
    frame = frame.loc[
        frame[context_columns[0]].eq(1)
        & frame["duration"].notna()
        & frame["duration"].gt(0)
    ].copy()
    frame["log_duration"] = np.log1p(frame["duration"].to_numpy(dtype=np.float64))

    feature_set = protocol["analysis"]["model_feature_set"]
    required_features = [
        column
        for spec in protocol["frozen_backbones"]
        for column in _prefixed_feature_columns(spec["key"], feature_set)
    ]
    if frame[required_features].isna().any().any():
        raise RuntimeError("analysis contains missing frozen features")

    assignments = stable_group_folds(
        frame["Text_ID"],
        int(protocol["analysis"]["outer_holdout"]["folds"]),
        seed=int(protocol["analysis"]["outer_holdout"]["seed"]),
    )
    frame["outer_fold"] = frame["Text_ID"].map(assignments).astype(int)
    fingerprint = {
        "path": geco_path.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": source_hash,
        "source_rows": len(raw),
        "analysis_rows": len(frame),
        "participants": int(frame["Participant_ID"].nunique()),
        "texts": int(frame["Text_ID"].nunique()),
        "items": int(raw[["Text_ID", "IA_ID"]].drop_duplicates().shape[0]),
        "feature_caches": cache_fingerprints,
    }
    if fingerprint["items"] != int(dataset["expected_items"]):
        raise RuntimeError("unexpected GECO item count")
    return frame.reset_index(drop=True), fingerprint


def build_nested_masks(
    participant_values: Sequence[str],
    text_values: Sequence[str],
    fold_values: Sequence[int],
    *,
    participant: str,
    outer_fold: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    participants = np.asarray(participant_values, dtype=str)
    texts = np.asarray(text_values, dtype=str)
    folds = np.asarray(fold_values, dtype=int)
    is_target = participants == str(participant)
    is_evaluation_fold = folds == int(outer_fold)
    masks = {
        "fit": (~is_target) & (~is_evaluation_fold),
        "calibration": is_target & (~is_evaluation_fold),
        "evaluation": is_target & is_evaluation_fold,
    }
    if any(not mask.any() for mask in masks.values()):
        raise RuntimeError("nested split produced an empty role")

    evaluation_texts = set(texts[masks["evaluation"]])
    fit_texts = set(texts[masks["fit"]])
    calibration_texts = set(texts[masks["calibration"]])
    diagnostics = {
        "target_participant_rows_in_fit": int(
            (masks["fit"] & is_target).sum()
        ),
        "evaluation_texts_in_fit": len(evaluation_texts.intersection(fit_texts)),
        "evaluation_texts_in_calibration": len(
            evaluation_texts.intersection(calibration_texts)
        ),
        "fit_rows": int(masks["fit"].sum()),
        "calibration_rows": int(masks["calibration"].sum()),
        "evaluation_rows": int(masks["evaluation"].sum()),
    }
    if any(
        diagnostics[key]
        for key in (
            "target_participant_rows_in_fit",
            "evaluation_texts_in_fit",
            "evaluation_texts_in_calibration",
        )
    ):
        raise RuntimeError(f"nested split leakage detected: {diagnostics}")
    return masks, diagnostics


def stable_calibration_text_order(
    text_ids: Sequence[str],
    *,
    participant: str,
    outer_fold: int,
    seed: int,
) -> list[str]:
    unique = sorted({str(value) for value in text_ids})

    def key(text_id: str) -> tuple[str, str]:
        payload = f"{seed}|{participant}|{outer_fold}|{text_id}".encode()
        return hashlib.sha256(payload).hexdigest(), text_id

    return sorted(unique, key=key)


def grouped_spearman(
    group_values: Sequence[str],
    prediction: Sequence[float],
    target: Sequence[float],
) -> pd.Series:
    """Vectorized Spearman correlation per complete text group."""
    groups = np.asarray(group_values, dtype=str)
    x = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if not (len(groups) == len(x) == len(y)):
        raise ValueError("grouped Spearman inputs must be aligned")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("grouped Spearman inputs must be finite")
    codes, unique = pd.factorize(groups, sort=True)
    x_rank = pd.Series(x).groupby(codes, sort=False).rank(method="average").to_numpy()
    y_rank = pd.Series(y).groupby(codes, sort=False).rank(method="average").to_numpy()
    size = len(unique)
    count = np.bincount(codes, minlength=size).astype(np.float64)
    sum_x = np.bincount(codes, weights=x_rank, minlength=size)
    sum_y = np.bincount(codes, weights=y_rank, minlength=size)
    sum_xx = np.bincount(codes, weights=x_rank * x_rank, minlength=size)
    sum_yy = np.bincount(codes, weights=y_rank * y_rank, minlength=size)
    sum_xy = np.bincount(codes, weights=x_rank * y_rank, minlength=size)
    covariance = sum_xy - (sum_x * sum_y / count)
    variance_x = sum_xx - (sum_x * sum_x / count)
    variance_y = sum_yy - (sum_y * sum_y / count)
    denominator = np.sqrt(np.maximum(variance_x * variance_y, 0.0))
    rho = np.full(size, np.nan, dtype=np.float64)
    valid = (count >= 3) & (denominator > 0)
    rho[valid] = covariance[valid] / denominator[valid]
    return pd.Series(rho, index=np.asarray(unique, dtype=str), dtype=np.float64)


def select_with_abstention(
    challenger_minus_incumbent: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    differences = np.asarray(challenger_minus_incumbent, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    if len(differences) < 2:
        return {
            "selected": "gpt2",
            "reason": "insufficient_valid_calibration_texts",
            "mean_difference": float(differences.mean()) if len(differences) else None,
            "ci_95_low": None,
            "ci_95_high": None,
            "valid_texts": int(len(differences)),
        }
    interval = paired_bootstrap_mean_difference(
        differences,
        np.zeros(len(differences), dtype=np.float64),
        samples=samples,
        seed=seed,
    )
    selected = (
        "pythia_410m_deduped_full"
        if interval["ci_95_low"] > 0.0
        else "gpt2"
    )
    return {
        "selected": selected,
        "reason": (
            "challenger_bootstrap_lower_bound_above_zero"
            if selected != "gpt2"
            else "uncertain_or_nonpositive_default_to_incumbent"
        ),
        "mean_difference": interval["mean_difference"],
        "ci_95_low": interval["ci_95_low"],
        "ci_95_high": interval["ci_95_high"],
        "valid_texts": interval["n_pairs"],
    }


def _derived_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)]).encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def evaluate_nested_matching(
    frame: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    analysis = protocol["analysis"]
    model_keys = [item["key"] for item in protocol["frozen_backbones"]]
    incumbent, challenger = model_keys
    feature_set = analysis["model_feature_set"]
    feature_arrays = {
        model_key: frame[
            _prefixed_feature_columns(model_key, feature_set)
        ].to_numpy(dtype=np.float64)
        for model_key in model_keys
    }
    target = frame["log_duration"].to_numpy(dtype=np.float64)
    participant_values = frame["Participant_ID"].astype(str).to_numpy()
    text_values = frame["Text_ID"].astype(str).to_numpy()
    fold_values = frame["outer_fold"].to_numpy(dtype=int)
    participants = sorted(set(participant_values))
    outer_folds = int(analysis["outer_holdout"]["folds"])
    alpha = float(analysis["ridge_alpha"])
    calibration = analysis["calibration_text_selection"]
    bootstrap_samples = int(analysis["selection_rule"]["calibration_bootstrap_samples"])
    base_seed = int(analysis["outer_holdout"]["seed"])
    rows: list[dict[str, Any]] = []
    leakage: list[dict[str, int]] = []

    for participant_index, participant in enumerate(participants, start=1):
        print(
            f"[participant {participant_index}/{len(participants)}] nested fitting",
            flush=True,
        )
        for outer_fold in range(outer_folds):
            masks, split_diagnostics = build_nested_masks(
                participant_values,
                text_values,
                fold_values,
                participant=participant,
                outer_fold=outer_fold,
            )
            leakage.append(split_diagnostics)
            calibration_predictions: dict[str, np.ndarray] = {}
            evaluation_predictions: dict[str, np.ndarray] = {}
            for model_key in model_keys:
                model = fit_standardized_ridge(
                    feature_arrays[model_key][masks["fit"]],
                    target[masks["fit"]],
                    alpha=alpha,
                )
                calibration_predictions[model_key] = predict_standardized_ridge(
                    model,
                    feature_arrays[model_key][masks["calibration"]],
                )
                evaluation_predictions[model_key] = predict_standardized_ridge(
                    model,
                    feature_arrays[model_key][masks["evaluation"]],
                )

            calibration_target = target[masks["calibration"]]
            calibration_texts = text_values[masks["calibration"]]
            per_text = {
                model_key: grouped_spearman(
                    calibration_texts,
                    calibration_predictions[model_key],
                    calibration_target,
                )
                for model_key in model_keys
            }
            calibration_differences = per_text[challenger] - per_text[incumbent]
            text_order = stable_calibration_text_order(
                calibration_texts,
                participant=participant,
                outer_fold=outer_fold,
                seed=base_seed,
            )
            evaluation_target = target[masks["evaluation"]]
            evaluation_rho = {
                model_key: safe_spearman(
                    evaluation_predictions[model_key], evaluation_target
                )
                for model_key in model_keys
            }
            if not all(np.isfinite(list(evaluation_rho.values()))):
                raise RuntimeError("invalid evaluation Spearman correlation")

            for budget in calibration["budgets"]:
                selected_texts = (
                    text_order if budget == "all" else text_order[: int(budget)]
                )
                differences = calibration_differences.reindex(selected_texts).dropna()
                selection = select_with_abstention(
                    differences.to_numpy(dtype=np.float64),
                    samples=bootstrap_samples,
                    seed=_derived_seed(base_seed, participant, outer_fold, budget),
                )
                mean_difference = selection["mean_difference"]
                argmax_model = (
                    challenger
                    if mean_difference is not None and mean_difference > 0.0
                    else incumbent
                )
                oracle_model = (
                    challenger
                    if evaluation_rho[challenger] > evaluation_rho[incumbent]
                    else incumbent
                )
                rows.append(
                    {
                        "Participant_ID": participant,
                        "outer_fold": outer_fold,
                        "calibration_budget": str(budget),
                        "requested_calibration_texts": len(selected_texts),
                        "valid_calibration_texts": selection["valid_texts"],
                        "calibration_mean_pythia_minus_gpt2": mean_difference,
                        "calibration_ci_95_low": selection["ci_95_low"],
                        "calibration_ci_95_high": selection["ci_95_high"],
                        "selected_model": selection["selected"],
                        "selection_reason": selection["reason"],
                        "argmax_model": argmax_model,
                        "oracle_model": oracle_model,
                        "evaluation_gpt2_rho": evaluation_rho[incumbent],
                        "evaluation_pythia_rho": evaluation_rho[challenger],
                        "evaluation_pythia_minus_gpt2": (
                            evaluation_rho[challenger] - evaluation_rho[incumbent]
                        ),
                        "evaluation_selected_minus_gpt2": (
                            evaluation_rho[selection["selected"]]
                            - evaluation_rho[incumbent]
                        ),
                        "evaluation_argmax_minus_gpt2": (
                            evaluation_rho[argmax_model] - evaluation_rho[incumbent]
                        ),
                        "evaluation_oracle_minus_gpt2": (
                            evaluation_rho[oracle_model] - evaluation_rho[incumbent]
                        ),
                    }
                )

    leakage_summary = {
        "cells": len(leakage),
        "target_participant_rows_in_fit": int(
            sum(item["target_participant_rows_in_fit"] for item in leakage)
        ),
        "evaluation_texts_in_fit": int(
            sum(item["evaluation_texts_in_fit"] for item in leakage)
        ),
        "evaluation_texts_in_calibration": int(
            sum(item["evaluation_texts_in_calibration"] for item in leakage)
        ),
    }
    leakage_summary["passed"] = not any(
        leakage_summary[key]
        for key in (
            "target_participant_rows_in_fit",
            "evaluation_texts_in_fit",
            "evaluation_texts_in_calibration",
        )
    )
    return pd.DataFrame(rows), leakage_summary


def summarize_budgets(
    cells: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    analysis = protocol["analysis"]
    bootstrap_samples = int(
        analysis["aggregation"]["participant_bootstrap_samples"]
    )
    bootstrap_seed = int(analysis["aggregation"]["participant_bootstrap_seed"])
    summaries: dict[str, dict[str, Any]] = {}
    for budget_index, budget in enumerate(
        analysis["calibration_text_selection"]["budgets"]
    ):
        label = str(budget)
        selected = cells.loc[cells["calibration_budget"].eq(label)].copy()
        participant = selected.groupby("Participant_ID", sort=True).agg(
            selected_delta=("evaluation_selected_minus_gpt2", "mean"),
            argmax_delta=("evaluation_argmax_minus_gpt2", "mean"),
            oracle_delta=("evaluation_oracle_minus_gpt2", "mean"),
            pythia_delta=("evaluation_pythia_minus_gpt2", "mean"),
        )
        fold = selected.groupby("outer_fold", sort=True).agg(
            selected_delta=("evaluation_selected_minus_gpt2", "mean"),
            argmax_delta=("evaluation_argmax_minus_gpt2", "mean"),
            oracle_delta=("evaluation_oracle_minus_gpt2", "mean"),
        )

        def bootstrap(column: str, offset: int) -> dict[str, float]:
            values = participant[column].to_numpy(dtype=np.float64)
            return paired_bootstrap_mean_difference(
                values,
                np.zeros(len(values), dtype=np.float64),
                samples=bootstrap_samples,
                seed=bootstrap_seed + budget_index * 10 + offset,
            )

        summaries[label] = {
            "requested_calibration_texts": (
                "all" if budget == "all" else int(budget)
            ),
            "participant_fold_cells": len(selected),
            "mean_valid_calibration_texts": float(
                selected["valid_calibration_texts"].mean()
            ),
            "challenger_selection": {
                "cells": int(
                    selected["selected_model"].eq(
                        "pythia_410m_deduped_full"
                    ).sum()
                ),
                "fraction": float(
                    selected["selected_model"].eq(
                        "pythia_410m_deduped_full"
                    ).mean()
                ),
            },
            "selected_minus_gpt2_participant_bootstrap": bootstrap(
                "selected_delta", 0
            ),
            "argmax_minus_gpt2_participant_bootstrap": bootstrap(
                "argmax_delta", 1
            ),
            "oracle_minus_gpt2_participant_bootstrap": bootstrap(
                "oracle_delta", 2
            ),
            "always_pythia_minus_gpt2_participant_bootstrap": bootstrap(
                "pythia_delta", 3
            ),
            "participants_with_positive_selected_delta": int(
                participant["selected_delta"].gt(0).sum()
            ),
            "fraction_participants_with_positive_selected_delta": float(
                participant["selected_delta"].gt(0).mean()
            ),
            "positive_outer_folds": int(fold["selected_delta"].gt(0).sum()),
            "outer_fold_count": len(fold),
            "fold_mean_selected_differences": {
                str(int(index)): float(value)
                for index, value in fold["selected_delta"].items()
            },
            "mean_evaluation_rho": {
                "gpt2": float(selected["evaluation_gpt2_rho"].mean()),
                "pythia_410m_deduped_full": float(
                    selected["evaluation_pythia_rho"].mean()
                ),
            },
        }
    return summaries


def make_decision(
    budget_summaries: Mapping[str, Mapping[str, Any]],
    leakage: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    primary_budget = str(
        protocol["analysis"]["calibration_text_selection"]["primary_budget"]
    )
    primary = budget_summaries[primary_budget]
    requirements = protocol["decision_gate"]["all_required"]
    checks = {
        "participant_bootstrap_ci_low_positive": (
            primary["selected_minus_gpt2_participant_bootstrap"]["ci_95_low"]
            > float(
                requirements[
                    "participant_bootstrap_ci_95_low_selected_minus_gpt2_greater_than"
                ]
            )
        ),
        "positive_outer_folds": (
            primary["positive_outer_folds"]
            >= int(requirements["minimum_positive_outer_folds"])
        ),
        "positive_participant_fraction": (
            primary["fraction_participants_with_positive_selected_delta"]
            >= float(
                requirements[
                    "minimum_fraction_participants_with_positive_mean_delta"
                ]
            )
        ),
        "challenger_selection_fraction": (
            primary["challenger_selection"]["fraction"]
            >= float(requirements["minimum_fraction_cells_with_challenger_selected"])
        ),
        "leakage_checks": bool(leakage["passed"]),
    }
    passed = all(checks.values())
    return {
        "label": (
            "personalized_selection_feasibility_passed"
            if passed
            else "personalized_selection_feasibility_not_demonstrated"
        ),
        "primary_calibration_budget": int(primary_budget),
        "checks": checks,
        "all_gates_passed": passed,
        "conditional_model_bank_expansion_allowed": passed,
        "production_model": "gpt2",
        "production_model_changed": False,
        "new_cohort_confirmation_required": True,
    }


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    decision = summary["decision"]
    lines = [
        "# Personalized Backbone Matching Feasibility v1 — Run 001",
        "",
        f"- Completed: {summary['created_at']}",
        f"- Runtime: {summary['compute']['wall_seconds']:.2f} seconds",
        "- Device: CPU only (GPU used: false)",
        f"- Decision: **`{decision['label']}`**",
        f"- Conditional model-bank expansion allowed: **{str(decision['conditional_model_bank_expansion_allowed']).lower()}**",
        "- Production backbone: `gpt2` (unchanged)",
        "",
        "## Design integrity",
        "",
        "- Every evaluation cell holds out both the target participant and complete evaluation `Text_ID` passage units from fitting.",
        "- Model selection uses only the target participant's deterministic calibration texts; evaluation texts never enter fitting or selection.",
        "- Frozen label-free GPT-2 and Pythia caches were reused; no language model, QA set, fine-tuning, network, or GPU was used.",
        "- GECO aggregate outcomes were previously inspected, so this remains exploratory and cannot confirm production personalization.",
        "",
        "## Calibration learning curve",
        "",
        "| Texts | Pythia selected | Selected − GPT-2 | 95% CI | Positive participants | Positive folds | Argmax − GPT-2 | Oracle upper bound |",
        "|---:|---:|---:|:---:|---:|---:|---:|---:|",
    ]
    for budget, result in summary["budgets"].items():
        selected = result["selected_minus_gpt2_participant_bootstrap"]
        argmax = result["argmax_minus_gpt2_participant_bootstrap"]
        oracle = result["oracle_minus_gpt2_participant_bootstrap"]
        lines.append(
            f"| {budget} | {result['challenger_selection']['cells']}/{result['participant_fold_cells']} "
            f"({result['challenger_selection']['fraction']:.1%}) | {selected['mean_difference']:+.6f} | "
            f"[{selected['ci_95_low']:+.6f}, {selected['ci_95_high']:+.6f}] | "
            f"{result['participants_with_positive_selected_delta']}/{selected['n_pairs']} | "
            f"{result['positive_outer_folds']}/{result['outer_fold_count']} | "
            f"{argmax['mean_difference']:+.6f} | {oracle['mean_difference']:+.6f} |"
        )

    primary = summary["budgets"][str(decision["primary_calibration_budget"])]
    lines.extend(
        [
            "",
            "## Frozen primary gate",
            "",
            *[
                f"- {'PASS' if passed else 'FAIL'} — `{name}`"
                for name, passed in decision["checks"].items()
            ],
            "",
            "## Interpretation",
            "",
        ]
    )
    if decision["all_gates_passed"]:
        lines.extend(
            [
                "The ten-text uncertainty-aware selector cleared every frozen feasibility clause. Label-free smoke tests for the exact SmolLM2, OLMo, and Granite expansion bank are permitted, but this is still exploratory and production remains GPT-2 pending a new cohort.",
            ]
        )
    else:
        lines.extend(
            [
                "The ten-text uncertainty-aware selector did not clear every frozen clause. The conditional SmolLM2/OLMo/Granite extraction is therefore stopped: adding candidates now would increase selection variance without evidence that personalized model choice generalizes.",
            ]
        )
    oracle = primary["oracle_minus_gpt2_participant_bootstrap"]
    argmax = primary["argmax_minus_gpt2_participant_bootstrap"]
    lines.extend(
        [
            "",
            f"At the primary budget, the plain calibration argmax changed held-out Spearman by {argmax['mean_difference']:+.6f}; the non-deployable evaluation oracle upper bound was {oracle['mean_difference']:+.6f}. The gap between them distinguishes available reader-level heterogeneity from calibration reliability.",
            "",
            "Detailed participant-cell results are stored only in the ignored local experiment directory; the tracked summary contains aggregates only.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--geco", type=Path, default=DEFAULT_GECO_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    protocol = load_protocol(args.protocol)
    print("[load] validating frozen outcome and feature inputs", flush=True)
    frame, dataset = load_analysis_frame(protocol, geco_path=args.geco)
    print(
        f"[analysis] {dataset['participants']} participants, {dataset['texts']} texts, "
        f"{dataset['analysis_rows']} positive context-scored rows",
        flush=True,
    )
    cells, leakage = evaluate_nested_matching(frame, protocol)
    budgets = summarize_budgets(cells, protocol)
    decision = make_decision(budgets, leakage, protocol)
    wall_seconds = time.perf_counter() - started
    wall_limit = float(protocol["compute"]["maximum_feasibility_wall_minutes"]) * 60
    if wall_seconds > wall_limit:
        decision = {
            **decision,
            "label": "compute_budget_exceeded",
            "all_gates_passed": False,
            "conditional_model_bank_expansion_allowed": False,
        }

    summary = {
        "schema_version": 1,
        "experiment": "personalized_backbone_matching_feasibility_v1_run_001",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_sha256": _sha256(args.protocol),
        "runner_sha256": _sha256(Path(__file__)),
        "protocol_frozen_before_personalized_outcome_access": True,
        "study_role": protocol["claim_scope"]["study_role"],
        "dataset": dataset,
        "analysis": {
            "outcome": protocol["analysis"]["primary_outcome"]["name"],
            "outer_folds": protocol["analysis"]["outer_holdout"]["folds"],
            "participants": dataset["participants"],
            "participant_fold_cells": leakage["cells"],
            "ridge_alpha": protocol["analysis"]["ridge_alpha"],
            "selection_rule": protocol["analysis"]["selection_rule"]["primary"],
        },
        "budgets": budgets,
        "decision": decision,
        "leakage": leakage,
        "compute": {
            "device": "cpu",
            "gpu_used": False,
            "language_models_loaded": False,
            "training_or_fine_tuning": False,
            "wall_seconds": wall_seconds,
            "wall_limit_seconds": wall_limit,
            "wall_budget_exceeded": wall_seconds > wall_limit,
        },
        "guardrails": {
            "question_answer_dataset_used": False,
            "complete_text_holdout": True,
            "target_participant_holdout": True,
            "scalers_fit_on_training_only": True,
            "production_model_changed": False,
            "new_cohort_confirmation_required": True,
        },
        "pre_evaluation_incident": {
            "first_attempt_produced_personalized_result": False,
            "stage": "feature_identity_merge_before_any_model_fit",
            "error": "pandas default NA parsing converted the literal GECO word 'null' to a missing display label",
            "fix": "read cached display labels literally and convert only frozen numeric feature columns",
            "affected_feature_rows": 1,
            "methodological_change": False,
            "regression_test_added": True,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.output_dir / "participant-fold-cells.csv", index=False, lineterminator="\n")
    _write_json(args.output_dir / "summary.json", summary)
    _write_json(args.summary_json, summary)
    write_report(args.report, summary)
    print(
        f"[decision] {decision['label']} (expand={decision['conditional_model_bank_expansion_allowed']})",
        flush=True,
    )


if __name__ == "__main__":
    main()

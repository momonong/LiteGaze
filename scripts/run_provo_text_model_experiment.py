"""Run a passage-held-out GPT-2 text-model experiment on the PROVO corpus.

The experiment compares a lexical baseline (M0), causal surprisal (M1), and
predictive entropy (M2).  It never uses QA data, keeps the language model frozen,
fits all scalers on training passages only, and includes a shuffled-target
sentinel to expose evaluation leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from wordfreq import zipf_frequency

from core.cognition.generalization import (
    cross_fit_grouped_ridge,
    grouped_spearman_table,
    paired_bootstrap_mean_difference,
)
from core.cognition.pipeline import LanguageModelCalculator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROVO_PATH = (
    PROJECT_ROOT / "data/provo/raw/Provo_Corpus-Eyetracking_Data.csv"
)
DEFAULT_CACHE_PATH = (
    PROJECT_ROOT / "data/provo/text_modeling/gpt2-small-causal-features.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data/provo/text_modeling/provo-gpt2-small-run-001"
)
EXPECTED_PROVO_SHA256 = (
    "38aedcb29bc9171009916eb2bcc2375729f104a2a1005c64a563da94b611b9e7"
)
MODEL_TYPE = "gpt2"
MODEL_LANGUAGE = "en"
MODEL_NAME = "gpt2"
RANDOM_SEED = 20260805
N_FOLDS = 5
RIDGE_ALPHA = 1.0
BOOTSTRAP_SAMPLES = 10_000

SOURCE_COLUMNS = (
    "Participant_ID",
    "Text_ID",
    "IA_ID",
    "IA_LABEL",
    "IA_DWELL_TIME",
    "IA_FIRST_RUN_DWELL_TIME",
    "IA_FIRST_FIXATION_DURATION",
)
OUTCOMES = {
    "total_reading_time": "IA_DWELL_TIME",
    "gaze_duration": "IA_FIRST_RUN_DWELL_TIME",
    "first_fixation_duration": "IA_FIRST_FIXATION_DURATION",
}
M0_FEATURES = (
    "log_char_length",
    "zipf_frequency",
    "relative_position",
    "starts_upper",
    "ends_punctuation",
    "subtoken_count",
    "prev_log_char_length",
    "prev_zipf_frequency",
)
M1_FEATURES = M0_FEATURES + (
    "causal_surprisal",
    "prev_causal_surprisal",
)
M2_FEATURES = M1_FEATURES + (
    "shannon_entropy",
    "renyi_entropy",
    "prev_shannon_entropy",
    "prev_renyi_entropy",
)
FEATURE_SETS = {
    "word_length_only": ("log_char_length",),
    "m0_lexical": M0_FEATURES,
    "m1_causal_surprisal": M1_FEATURES,
    "m2_entropy": M2_FEATURES,
}
COMPARISONS = (
    ("m0_lexical", "word_length_only"),
    ("m1_causal_surprisal", "m0_lexical"),
    ("m1_causal_surprisal", "word_length_only"),
    ("m2_entropy", "m1_causal_surprisal"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _git_state() -> dict[str, Any]:
    def output(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            encoding="utf-8",
        ).strip()

    return {
        "commit": output("rev-parse", "HEAD"),
        "branch": output("branch", "--show-current"),
        "dirty": bool(output("status", "--short")),
    }


def load_provo(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load complete display identity fields and return participant/item tables."""
    source_hash = _sha256(path)
    if source_hash != EXPECTED_PROVO_SHA256:
        raise RuntimeError(
            "PROVO source hash changed; freeze a new protocol before inspecting outcomes"
        )
    raw = pd.read_csv(
        path,
        usecols=list(SOURCE_COLUMNS),
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    # EyeLink labels include display-padding whitespace.  It is not part of the
    # lexical item and would otherwise become an extra GPT-2 space token.
    raw["IA_LABEL"] = raw["IA_LABEL"].map(
        lambda value: unicodedata.normalize("NFKC", str(value)).strip()
    )
    if raw[list(SOURCE_COLUMNS[:4])].eq("").any().any():
        raise ValueError("PROVO identity/display fields must be complete")
    raw["Text_ID"] = raw["Text_ID"].astype(str)
    raw["IA_ID"] = pd.to_numeric(raw["IA_ID"], errors="raise").astype(int)
    if raw.duplicated(["Participant_ID", "Text_ID", "IA_ID"]).any():
        raise ValueError("duplicate participant/text/item rows in PROVO source")

    items = raw[["Text_ID", "IA_ID", "IA_LABEL"]].drop_duplicates()
    if items.groupby(["Text_ID", "IA_ID"])["IA_LABEL"].nunique().gt(1).any():
        raise ValueError("a PROVO display item has inconsistent labels")
    items = items.sort_values(["Text_ID", "IA_ID"]).reset_index(drop=True)

    fingerprint = {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": source_hash,
        "size_bytes": path.stat().st_size,
        "row_count": len(raw),
        "participant_count": raw["Participant_ID"].nunique(),
        "text_count": raw["Text_ID"].nunique(),
        "item_count": len(items),
    }
    if fingerprint["participant_count"] != 84 or fingerprint["text_count"] != 55:
        raise RuntimeError(f"unexpected PROVO identity: {fingerprint}")
    return raw, items, fingerprint


def _strip_edge_punctuation(value: str) -> str:
    start = 0
    end = len(value)
    while start < end and unicodedata.category(value[start]).startswith("P"):
        start += 1
    while end > start and unicodedata.category(value[end - 1]).startswith("P"):
        end -= 1
    return value[start:end]


def _lexical_features(word: str, position: int, item_count: int) -> dict[str, float]:
    normalized = unicodedata.normalize("NFKC", str(word)).strip()
    char_length = sum(character.isalnum() for character in normalized)
    lookup = _strip_edge_punctuation(normalized).casefold()
    first_cased = next(
        (
            character
            for character in normalized
            if character.lower() != character.upper()
        ),
        "",
    )
    return {
        "log_char_length": float(np.log1p(char_length)),
        "zipf_frequency": float(np.clip(zipf_frequency(lookup, "en"), 0.0, 8.0)),
        "relative_position": float(position / max(item_count - 1, 1)),
        "starts_upper": float(bool(first_cased) and first_cased.isupper()),
        "ends_punctuation": float(
            bool(normalized)
            and unicodedata.category(normalized[-1]).startswith("P")
        ),
    }


def _model_revision(calculator: LanguageModelCalculator) -> dict[str, Any]:
    return {
        "model_type": MODEL_TYPE,
        "model_name": MODEL_NAME,
        "model_commit": getattr(calculator.model.config, "_commit_hash", None),
        "tokenizer_commit": calculator.tokenizer.init_kwargs.get("_commit_hash"),
        "tokenizer_class": type(calculator.tokenizer).__name__,
        "model_class": type(calculator.model).__name__,
        "device": calculator.device,
        "dtype": str(next(calculator.model.parameters()).dtype),
        "metric_contract": calculator.metric_contract(),
    }


def extract_features(
    items: pd.DataFrame,
    *,
    device: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract frozen, label-free GPT-2 and lexical features per display item."""
    calculator = LanguageModelCalculator(
        model_type=MODEL_TYPE,
        lang=MODEL_LANGUAGE,
        device=device,
    )
    records: list[dict[str, Any]] = []
    grouped = list(items.groupby("Text_ID", sort=True))
    for group_index, (text_id, group) in enumerate(grouped, start=1):
        group = group.sort_values("IA_ID")
        words = group["IA_LABEL"].astype(str).tolist()
        metrics = calculator.compute(words)
        encoded = calculator.tokenizer(words, is_split_into_words=True)
        subtoken_counts = np.zeros(len(words), dtype=int)
        for word_id in encoded.word_ids():
            if word_id is not None:
                subtoken_counts[word_id] += 1

        for position, ((_, item), surprisal, entropy, renyi) in enumerate(
            zip(
                group.iterrows(),
                metrics["surprisals"],
                metrics["entropies"],
                metrics["renyi_entropies"],
                strict=True,
            )
        ):
            records.append(
                {
                    "Text_ID": str(text_id),
                    "IA_ID": int(item["IA_ID"]),
                    "IA_LABEL": str(item["IA_LABEL"]),
                    **_lexical_features(str(item["IA_LABEL"]), position, len(words)),
                    "subtoken_count": int(subtoken_counts[position]),
                    "causal_surprisal": float(surprisal),
                    "shannon_entropy": float(entropy),
                    "renyi_entropy": float(renyi),
                    "is_context_scored": int(position > 0),
                }
            )
        if group_index % 10 == 0 or group_index == len(grouped):
            print(f"[features] scored {group_index}/{len(grouped)} passages", flush=True)

    features = pd.DataFrame(records).sort_values(["Text_ID", "IA_ID"])
    shift_columns = (
        "log_char_length",
        "zipf_frequency",
        "causal_surprisal",
        "shannon_entropy",
        "renyi_entropy",
    )
    for column in shift_columns:
        features[f"prev_{column}"] = features.groupby("Text_ID")[column].shift(1)
    feature_columns = sorted(
        {feature for feature_set in FEATURE_SETS.values() for feature in feature_set}
    )
    scored = features["is_context_scored"].eq(1)
    if not np.isfinite(features.loc[scored, feature_columns].to_numpy(dtype=float)).all():
        raise RuntimeError("feature extraction produced non-finite scored rows")
    return features.reset_index(drop=True), _model_revision(calculator)


def load_or_extract_features(
    items: pd.DataFrame,
    fingerprint: Mapping[str, Any],
    *,
    cache_path: Path,
    device: str,
    force: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata_path = cache_path.with_suffix(".metadata.json")
    code_hashes = {
        "extractor_sha256": _sha256(Path(__file__)),
        "pipeline_sha256": _sha256(PROJECT_ROOT / "core/cognition/pipeline.py"),
    }
    if cache_path.exists() and metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        reusable = (
            metadata.get("source_sha256") == fingerprint["sha256"]
            and metadata.get("item_count") == fingerprint["item_count"]
            and metadata.get("code_hashes") == code_hashes
            and metadata.get("model", {}).get("model_name") == MODEL_NAME
        )
        if reusable:
            print(f"[features] reusing cache {cache_path}")
            return pd.read_csv(
                cache_path,
                dtype={"Text_ID": str, "IA_ID": int},
            ), metadata

    features, model = extract_features(items, device=device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(cache_path, index=False, lineterminator="\n")
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_sha256": fingerprint["sha256"],
        "item_count": len(features),
        "feature_sha256": _sha256(cache_path),
        "code_hashes": code_hashes,
        "model": model,
        "question_answer_dataset_used": False,
        "outcome_columns_read_during_extraction": False,
    }
    _write_json(metadata_path, metadata)
    return features, metadata


def _aligned_values(
    table: pd.DataFrame,
    first_model: str,
    second_model: str,
    *,
    index_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    pivot = table.pivot_table(
        index=list(index_columns),
        columns="model",
        values="spearman_rho",
        aggfunc="first",
    )
    return (
        pivot[first_model].to_numpy(dtype=float),
        pivot[second_model].to_numpy(dtype=float),
    )


def _summarize_outcome(
    predictions: pd.DataFrame,
    *,
    target_column: str,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_columns = [
        f"prediction_{name}"
        for name in (*FEATURE_SETS.keys(), "target_shuffle_sentinel")
    ]
    participant_metrics = grouped_spearman_table(
        predictions,
        target_column=target_column,
        prediction_columns=prediction_columns,
        grouping_columns=("Participant_ID",),
    )
    fold_metrics = grouped_spearman_table(
        predictions,
        target_column=target_column,
        prediction_columns=prediction_columns,
        grouping_columns=("outer_fold", "Participant_ID"),
    )

    aggregate_spec: dict[str, tuple[str, str]] = {target_column: (target_column, "mean")}
    for prediction_column in prediction_columns:
        aggregate_spec[prediction_column] = (prediction_column, "first")
    item_predictions = (
        predictions.groupby(["Text_ID", "IA_ID"], as_index=False)
        .agg(**aggregate_spec)
    )
    text_metrics = grouped_spearman_table(
        item_predictions,
        target_column=target_column,
        prediction_columns=prediction_columns,
        grouping_columns=("Text_ID",),
    )

    model_summaries: list[dict[str, Any]] = []
    for prediction_column in prediction_columns:
        model_name = prediction_column.removeprefix("prediction_")
        participant_values = participant_metrics.loc[
            participant_metrics["model"].eq(model_name), "spearman_rho"
        ]
        text_values = text_metrics.loc[
            text_metrics["model"].eq(model_name), "spearman_rho"
        ]
        fold_values = (
            fold_metrics.loc[fold_metrics["model"].eq(model_name)]
            .groupby("outer_fold")["spearman_rho"]
            .mean()
            .sort_index()
        )
        mae = float(
            np.mean(
                np.abs(
                    predictions[prediction_column].to_numpy(dtype=float)
                    - predictions[target_column].to_numpy(dtype=float)
                )
            )
        )
        model_summaries.append(
            {
                "model": model_name,
                "macro_participant_spearman": float(np.nanmean(participant_values)),
                "macro_text_spearman": float(np.nanmean(text_values)),
                "mean_absolute_error_log": mae,
                "fold_macro_participant_spearman": {
                    str(int(fold)): float(value) for fold, value in fold_values.items()
                },
            }
        )

    comparisons: list[dict[str, Any]] = []
    fold_pivot = fold_metrics.pivot_table(
        index=["outer_fold", "Participant_ID"],
        columns="model",
        values="spearman_rho",
        aggfunc="first",
    )
    for comparison_index, (first, second) in enumerate(COMPARISONS):
        participant_first, participant_second = _aligned_values(
            participant_metrics,
            first,
            second,
            index_columns=("Participant_ID",),
        )
        text_first, text_second = _aligned_values(
            text_metrics,
            first,
            second,
            index_columns=("Text_ID",),
        )
        per_fold = (
            fold_pivot[first] - fold_pivot[second]
        ).groupby("outer_fold").mean()
        comparisons.append(
            {
                "comparison": f"{first}_minus_{second}",
                "participant_bootstrap": paired_bootstrap_mean_difference(
                    participant_first,
                    participant_second,
                    samples=bootstrap_samples,
                    seed=seed + comparison_index,
                ),
                "text_bootstrap": paired_bootstrap_mean_difference(
                    text_first,
                    text_second,
                    samples=bootstrap_samples,
                    seed=seed + 100 + comparison_index,
                ),
                "positive_outer_folds": int((per_fold > 0).sum()),
                "outer_fold_count": int(len(per_fold)),
                "fold_mean_differences": {
                    str(int(fold)): float(value) for fold, value in per_fold.items()
                },
            }
        )
    return (
        {"models": model_summaries, "comparisons": comparisons},
        participant_metrics,
        text_metrics,
        fold_metrics,
    )


def evaluate_outcome(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    *,
    outcome_name: str,
    source_column: str,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = raw.merge(
        features,
        on=["Text_ID", "IA_ID", "IA_LABEL"],
        how="left",
        validate="many_to_one",
    )
    frame[source_column] = pd.to_numeric(frame[source_column], errors="coerce")
    frame = frame.loc[
        frame["is_context_scored"].eq(1)
        & frame[source_column].notna()
        & frame[source_column].gt(0)
    ].copy()
    frame["log_duration"] = np.log1p(frame[source_column].to_numpy(dtype=float))
    feature_columns = sorted(
        {feature for feature_set in FEATURE_SETS.values() for feature in feature_set}
    )
    if frame[feature_columns].isna().any().any():
        raise RuntimeError(f"{outcome_name} contains missing model features")

    predictions, diagnostics = cross_fit_grouped_ridge(
        frame,
        group_column="Text_ID",
        target_column="log_duration",
        feature_sets=FEATURE_SETS,
        n_folds=N_FOLDS,
        alpha=RIDGE_ALPHA,
        seed=RANDOM_SEED,
        shuffled_target_model="m2_entropy",
    )
    summary, participant, text, fold = _summarize_outcome(
        predictions,
        target_column="log_duration",
        bootstrap_samples=bootstrap_samples,
        seed=RANDOM_SEED + 1_000 * list(OUTCOMES).index(outcome_name),
    )
    summary.update(
        {
            "source_column": source_column,
            "conditional_on_positive_duration": True,
            "row_count": len(predictions),
            "participant_count": predictions["Participant_ID"].nunique(),
            "text_count": predictions["Text_ID"].nunique(),
            "cross_fit_diagnostics": diagnostics,
        }
    )
    for table in (participant, text, fold, predictions):
        table.insert(0, "outcome", outcome_name)
    return summary, participant, text, fold, predictions


def _comparison(summary: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return next(item for item in summary["comparisons"] if item["comparison"] == name)


def make_decision(outcomes: Mapping[str, Any]) -> dict[str, Any]:
    primary = outcomes["total_reading_time"]
    m1 = _comparison(primary, "m1_causal_surprisal_minus_m0_lexical")
    m2 = _comparison(primary, "m2_entropy_minus_m1_causal_surprisal")
    m1_pass = (
        m1["participant_bootstrap"]["ci_95_low"] > 0
        and m1["text_bootstrap"]["ci_95_low"] > 0
        and m1["positive_outer_folds"] >= 4
    )
    m2_pass = (
        m2["participant_bootstrap"]["ci_95_low"] > 0
        and m2["text_bootstrap"]["ci_95_low"] > 0
        and m2["positive_outer_folds"] >= 4
    )
    if m1_pass and m2_pass:
        label = "causal_surprisal_and_entropy_incremental_on_provo"
    elif m1_pass:
        label = "causal_surprisal_incremental_on_provo"
    else:
        label = "no_confirmed_incremental_text_model_gain_on_provo"
    return {
        "label": label,
        "m1_causal_surprisal_gate_passed": m1_pass,
        "m2_entropy_gate_passed": m2_pass,
        "final_confirmation_required": True,
        "provo_role": "known_replication_corpus_not_final_confirmation",
    }


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# PROVO GPT-2 Small Text-Model Generalization — Run 001",
        "",
        f"- Completed: {summary['created_at']}",
        f"- Runtime: {summary['duration_seconds']:.2f} seconds",
        f"- Device: `{summary['compute']['device']}` (GPU used: {str(summary['compute']['gpu_used']).lower()})",
        f"- Decision: **`{summary['decision']['label']}`**",
        "- Role: grouped replication/development evidence; PROVO is not an untouched final corpus",
        "",
        "## Design",
        "",
        "- 55 complete passages assigned to five deterministic outer folds.",
        "- Every scaler and Ridge model was fit on training passages only.",
        "- Frozen GPT-2 small; no LM fine-tuning, QA data, or hyperparameter search.",
        "- The first item of each passage was excluded because it has no left context.",
        "- A shuffled-training-target model is reported as a leakage sentinel.",
        "",
    ]
    for outcome_name, outcome in summary["outcomes"].items():
        lines.extend(
            [
                f"## {outcome_name.replace('_', ' ').title()}",
                "",
                f"Rows: {outcome['row_count']:,}; participants: {outcome['participant_count']}; texts: {outcome['text_count']}.",
                "",
                "| Model | Macro participant rho | Macro text rho | Log-duration MAE |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for model in outcome["models"]:
            lines.append(
                f"| `{model['model']}` | {model['macro_participant_spearman']:.4f} "
                f"| {model['macro_text_spearman']:.4f} | {model['mean_absolute_error_log']:.4f} |"
            )
        lines.extend(
            [
                "",
                "| Comparison | Participant delta [95% CI] | Text delta [95% CI] | Positive folds |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for comparison in outcome["comparisons"]:
            participant = comparison["participant_bootstrap"]
            text = comparison["text_bootstrap"]
            lines.append(
                f"| `{comparison['comparison']}` "
                f"| {participant['mean_difference']:+.4f} "
                f"[{participant['ci_95_low']:+.4f}, {participant['ci_95_high']:+.4f}] "
                f"| {text['mean_difference']:+.4f} "
                f"[{text['ci_95_low']:+.4f}, {text['ci_95_high']:+.4f}] "
                f"| {comparison['positive_outer_folds']}/{comparison['outer_fold_count']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Guardrails",
            "",
            "- No question-answer dataset or PROVO cloze/LSA/predictability column was used.",
            "- No gaze or duration value was used as a predictor.",
            "- Raw causal surprisal/entropy were retained; no document-relative target normalization was used.",
            "- This result may choose the next engineering ablation, but cannot serve as final confirmation because historical PROVO results were already known.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provo-path", type=Path, default=DEFAULT_PROVO_PATH)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--force-feature-extraction", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    started = time.perf_counter()
    raw, items, fingerprint = load_provo(args.provo_path.resolve())
    features, feature_metadata = load_or_extract_features(
        items,
        fingerprint,
        cache_path=args.cache_path.resolve(),
        device=args.device,
        force=args.force_feature_extraction,
    )

    outcome_summaries: dict[str, Any] = {}
    participant_tables: list[pd.DataFrame] = []
    text_tables: list[pd.DataFrame] = []
    fold_tables: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []
    for outcome_name, source_column in OUTCOMES.items():
        print(f"[benchmark] evaluating {outcome_name}", flush=True)
        outcome, participant, text, fold, predictions = evaluate_outcome(
            raw,
            features,
            outcome_name=outcome_name,
            source_column=source_column,
            bootstrap_samples=args.bootstrap_samples,
        )
        outcome_summaries[outcome_name] = outcome
        participant_tables.append(participant)
        text_tables.append(text)
        fold_tables.append(fold)
        prediction_tables.append(predictions)

    duration = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "experiment": "provo_gpt2_small_text_generalization_run_001",
        "created_at": datetime.now(UTC).isoformat(),
        "duration_seconds": duration,
        "configuration": {
            "seed": RANDOM_SEED,
            "outer_folds": N_FOLDS,
            "ridge_alpha": RIDGE_ALPHA,
            "bootstrap_samples": args.bootstrap_samples,
            "feature_sets": {key: list(value) for key, value in FEATURE_SETS.items()},
            "outcomes": OUTCOMES,
        },
        "dataset": fingerprint,
        "feature_cache": {
            "path": args.cache_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
            **feature_metadata,
        },
        "outcomes": outcome_summaries,
        "decision": make_decision(outcome_summaries),
        "compute": {
            "device": args.device,
            "gpu_used": args.device == "cuda",
            "language_model_fine_tuned": False,
        },
        "leakage_controls": {
            "question_answer_dataset_used": False,
            "provo_predictability_cloze_or_lsa_used": False,
            "gaze_feature_used_as_predictor": False,
            "complete_text_holdout": True,
            "scaler_fit_on_training_only": True,
            "target_shuffle_sentinel_included": True,
            "hyperparameter_search_used": False,
        },
        "source": _git_state(),
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    participant_path = output_dir / "participant_metrics.csv"
    text_path = output_dir / "text_metrics.csv"
    fold_path = output_dir / "fold_metrics.csv"
    predictions_path = output_dir / "cross_fitted_predictions.csv.gz"
    manifest_path = output_dir / "manifest.json"

    _write_json(summary_path, summary)
    write_report(report_path, summary)
    pd.concat(participant_tables, ignore_index=True).to_csv(
        participant_path, index=False, lineterminator="\n"
    )
    pd.concat(text_tables, ignore_index=True).to_csv(
        text_path, index=False, lineterminator="\n"
    )
    pd.concat(fold_tables, ignore_index=True).to_csv(
        fold_path, index=False, lineterminator="\n"
    )
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        predictions_path,
        index=False,
        compression="gzip",
        lineterminator="\n",
    )
    manifest = {
        "schema_version": 1,
        "experiment": summary["experiment"],
        "created_at": summary["created_at"],
        "dataset_sha256": fingerprint["sha256"],
        "feature_sha256": feature_metadata["feature_sha256"],
        "source": summary["source"],
        "artifacts": {
            path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in (
                summary_path,
                report_path,
                participant_path,
                text_path,
                fold_path,
                predictions_path,
            )
        },
    }
    _write_json(manifest_path, manifest)

    print(f"[complete] {summary['decision']['label']}")
    print(f"[complete] runtime={duration:.2f}s device={args.device}")
    print(f"[complete] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

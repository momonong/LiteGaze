"""Run the preregistered OneStop text-model confirmation on CPU.

The first source pass reads only display identity and labels. Frozen GPT-2
features are cached before a second, explicitly whitelisted outcome pass. No QA
field or corpus-provided linguistic annotation is loaded by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import unicodedata
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.cognition.generalization import (
    cross_fit_grouped_ridge,
    grouped_spearman_table,
    paired_bootstrap_mean_difference,
)
from core.cognition.pipeline import LanguageModelCalculator
from scripts import run_provo_text_model_experiment as frozen
from scripts.prepare_onestop_confirmation import (
    ANALYSIS_READ_COLUMNS,
    DEFAULT_SOURCE_PATH,
    OUTCOME_COLUMNS,
    inspect_archive,
    sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    PROJECT_ROOT
    / "docs/experiments/protocols/2026-08-05-onestop-confirmation-v1.json"
)
DEFAULT_CACHE_PATH = (
    PROJECT_ROOT / "data/onestop/text_modeling/gpt2-small-causal-features.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/onestop/text_modeling/onestop-gpt2-small-confirmation-run-001"
)
DEFAULT_TRACKED_SUMMARY = (
    PROJECT_ROOT
    / "docs/experiments/results/2026-08-05-onestop-confirmation-run-001.json"
)
DEFAULT_TRACKED_REPORT = (
    PROJECT_ROOT
    / "docs/experiments/2026-08-05-onestop-confirmation-run-001.md"
)

PROTOCOL_ID = "onestop-ordinary-advanced-confirmation-v1"
MODEL_NAME = "gpt2"
RANDOM_SEED = 20260805
N_FOLDS = 5
RIDGE_ALPHA = 1.0
BOOTSTRAP_SAMPLES = 10_000
EXPECTED_ARTICLE_COUNT = 30
CHUNK_ROWS = 250_000

IDENTITY_COLUMNS = tuple(
    column for column in ANALYSIS_READ_COLUMNS if column not in OUTCOME_COLUMNS
)
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
FEATURE_SETS = {
    "word_length_only": ("log_char_length",),
    "m0_lexical": M0_FEATURES,
    "m1_causal_surprisal": M1_FEATURES,
}
OUTCOMES = {
    "total_reading_time": "IA_DWELL_TIME",
    "gaze_duration": "IA_FIRST_RUN_DWELL_TIME",
    "first_fixation_duration": "IA_FIRST_FIXATION_DURATION",
}
COMPARISONS = (
    ("m0_lexical", "word_length_only"),
    ("m1_causal_surprisal", "m0_lexical"),
    ("m1_causal_surprisal", "word_length_only"),
)


def _load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _assert_protocol_contract() -> dict[str, Any]:
    protocol = _load_protocol()
    checks = {
        "protocol_id": protocol["protocol_id"] == PROTOCOL_ID,
        "outcome_blind_freeze": protocol["status_at_freeze"] == "outcome_blind",
        "analysis_columns": tuple(protocol["analysis_read_columns"])
        == ANALYSIS_READ_COLUMNS,
        "m0_features": tuple(protocol["models"]["m0_lexical"]) == M0_FEATURES,
        "m1_features": tuple(protocol["models"]["m1_causal_surprisal"])
        == M1_FEATURES,
        "ridge_alpha": protocol["models"]["ridge_alpha"] == RIDGE_ALPHA,
        "folds": protocol["grouping"]["outer_folds"] == N_FOLDS,
        "seed": protocol["grouping"]["seed"] == RANDOM_SEED,
        "bootstrap": protocol["decision_gate"]["bootstrap_samples"]
        == BOOTSTRAP_SAMPLES,
        "cpu_only": protocol["compute"]["device"] == "cpu"
        and protocol["compute"]["gpu_allowed"] is False,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    if failures:
        raise RuntimeError(f"runner drifted from frozen protocol: {failures}")
    return protocol


def _strict_bool(values: pd.Series, name: str) -> pd.Series:
    normalized = values.astype(str).str.strip().str.casefold()
    unexpected = sorted(set(normalized).difference({"true", "false"}))
    if unexpected:
        raise ValueError(f"{name} contains non-boolean values: {unexpected[:5]}")
    return normalized.eq("true")


def _scope_mask(chunk: pd.DataFrame) -> pd.Series:
    practice = _strict_bool(chunk["practice_trial"], "practice_trial")
    preview = _strict_bool(chunk["question_preview"], "question_preview")
    repeated = _strict_bool(
        chunk["repeated_reading_trial"], "repeated_reading_trial"
    )
    difficulty = chunk["difficulty_level"].astype(str).str.strip()
    article = pd.to_numeric(chunk["article_id"], errors="raise")
    return (~practice) & (~preview) & (~repeated) & difficulty.eq("Adv") & article.ne(0)


def _ensure_extracted_csv(path: Path, manifest: Mapping[str, Any]) -> Path:
    """Materialize the verified main CSV to avoid large ZipExtFile parser crashes."""
    member_name = str(manifest["archive"]["member_name"])
    expected_size = int(manifest["archive"]["uncompressed_size_bytes"])
    extracted = path.with_suffix("")
    metadata_path = extracted.with_suffix(extracted.suffix + ".metadata.json")

    if extracted.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        reusable = (
            extracted.stat().st_size == expected_size
            and metadata.get("archive_sha256") == manifest["source"]["sha256"]
            and metadata.get("member_name") == member_name
            and metadata.get("size_bytes") == expected_size
            and metadata.get("sha256") == sha256(extracted)
        )
        if reusable:
            return extracted
        raise RuntimeError("existing extracted OneStop CSV failed provenance checks")

    if extracted.exists() != metadata_path.exists():
        raise RuntimeError("incomplete extracted OneStop CSV provenance pair")
    temporary = extracted.with_suffix(extracted.suffix + ".part")
    digest = hashlib.sha256()
    written = 0
    try:
        with zipfile.ZipFile(path) as archive, archive.open(
            member_name, "r"
        ) as source, temporary.open("wb") as output:
            while block := source.read(1024 * 1024):
                output.write(block)
                digest.update(block)
                written += len(block)
        if written != expected_size:
            raise RuntimeError(
                f"extracted OneStop CSV size mismatch: {written} != {expected_size}"
            )
        os.replace(temporary, extracted)
    finally:
        if temporary.exists():
            temporary.unlink()
    frozen._write_json(
        metadata_path,
        {
            "archive_sha256": manifest["source"]["sha256"],
            "member_name": member_name,
            "sha256": digest.hexdigest(),
            "size_bytes": written,
        },
    )
    return extracted


def _read_selected_source(
    path: Path,
    *,
    columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read only an explicit column whitelist and apply frozen identity filters."""
    manifest = inspect_archive(path)
    csv_path = _ensure_extracted_csv(path, manifest)
    chunks: list[pd.DataFrame] = []
    input_rows = 0
    selected_rows = 0
    reader = pd.read_csv(
        csv_path,
        usecols=list(columns),
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_ROWS,
        low_memory=False,
    )
    for chunk in reader:
        input_rows += len(chunk)
        mask = _scope_mask(chunk)
        selected = chunk.loc[mask, list(columns)].copy()
        selected_rows += len(selected)
        chunks.append(selected)
    if not chunks or not selected_rows:
        raise RuntimeError("frozen OneStop scope selected no rows")
    return pd.concat(chunks, ignore_index=True), {
        "archive_input_rows": input_rows,
        "selected_rows": selected_rows,
        "columns_read": list(columns),
        "outcome_columns_read": sorted(set(columns).intersection(OUTCOME_COLUMNS)),
    }


def _normalize_labels(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: unicodedata.normalize("NFKC", str(value)).strip()
    )


def _integer_column(frame: pd.DataFrame, name: str) -> pd.Series:
    numeric = pd.to_numeric(frame[name], errors="raise")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} contains non-integer values")
    return numeric.astype(int)


def _trial_id(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["participant_id"].astype(str)
        + "|"
        + frame["list_number"].astype(str)
        + "|"
        + frame["trial_index"].astype(str)
    )


def _sequence_digest(group: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for item_id, label in group.sort_values("IA_ID")[["IA_ID", "IA_LABEL"]].itertuples(
        index=False, name=None
    ):
        digest.update(str(int(item_id)).encode("ascii"))
        digest.update(b"\x1f")
        digest.update(str(label).encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def _items_digest(items: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    ordered = items.sort_values(["Article_ID", "Text_ID", "IA_ID", "IA_LABEL"])
    for row in ordered[["Article_ID", "Text_ID", "IA_ID", "IA_LABEL"]].itertuples(
        index=False, name=None
    ):
        digest.update(json.dumps(row, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def prepare_identity(
    source: pd.DataFrame,
    *,
    enforce_expected_articles: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build participant-trial context identities without reading outcomes."""
    frame = source.copy()
    frame["IA_LABEL"] = _normalize_labels(frame["IA_LABEL"])
    if frame["IA_LABEL"].eq("").any():
        raise ValueError("OneStop contains an empty normalized IA_LABEL")
    for column in ("article_batch", "article_id", "paragraph_id", "IA_ID"):
        frame[column] = _integer_column(frame, column)
    frame["Trial_ID"] = _trial_id(frame)

    identity_columns = (
        "participant_id",
        "list_number",
        "trial_index",
        "article_batch",
        "article_id",
        "paragraph_id",
        "difficulty_level",
    )
    if frame.duplicated(["Trial_ID", "IA_ID"]).any():
        raise ValueError("duplicate OneStop trial/item rows")
    for column in identity_columns[3:]:
        if frame.groupby("Trial_ID")[column].nunique().gt(1).any():
            raise ValueError(f"OneStop trial has inconsistent {column}")

    sequence_hashes = {
        trial_id: _sequence_digest(group)
        for trial_id, group in frame.groupby("Trial_ID", sort=False)
    }
    frame["Sequence_Hash"] = frame["Trial_ID"].map(sequence_hashes)
    frame["Article_ID"] = (
        frame["article_batch"].astype(str) + ":" + frame["article_id"].astype(str)
    )
    frame["Text_ID"] = (
        frame["Article_ID"]
        + ":p"
        + frame["paragraph_id"].astype(str)
        + ":"
        + frame["difficulty_level"].astype(str)
        + ":"
        + frame["Sequence_Hash"]
    )

    trial_map = frame[
        ["Trial_ID", "participant_id", "Article_ID", "Text_ID"]
    ].drop_duplicates()
    if trial_map["Trial_ID"].duplicated().any():
        raise ValueError("OneStop trial maps to multiple display contexts")
    items = frame[["Article_ID", "Text_ID", "IA_ID", "IA_LABEL"]].drop_duplicates()
    if items.groupby(["Text_ID", "IA_ID"])["IA_LABEL"].nunique().gt(1).any():
        raise ValueError("OneStop display context has inconsistent item labels")
    items = items.sort_values(["Article_ID", "Text_ID", "IA_ID"]).reset_index(
        drop=True
    )
    article_count = items["Article_ID"].nunique()
    if enforce_expected_articles and article_count != EXPECTED_ARTICLE_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_ARTICLE_COUNT} articles, found {article_count}"
        )
    fingerprint = {
        "selected_identity_rows": len(frame),
        "participant_count": frame["participant_id"].nunique(),
        "article_count": article_count,
        "context_count": items["Text_ID"].nunique(),
        "item_count": len(items),
        "item_identity_sha256": _items_digest(items),
        "difficulty_level": "Adv",
        "question_preview": False,
        "repeated_reading": False,
        "practice": False,
        "outcome_columns_read": False,
    }
    return trial_map.reset_index(drop=True), items, fingerprint


def load_identity(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source, read_diagnostics = _read_selected_source(path, columns=IDENTITY_COLUMNS)
    trial_map, items, fingerprint = prepare_identity(source)
    fingerprint.update(read_diagnostics)
    fingerprint.update(
        {
            "path": path.resolve().relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
    )
    if fingerprint["outcome_columns_read"]:
        raise RuntimeError("identity pass unexpectedly read an outcome column")
    return trial_map, items, fingerprint


def _model_revision(calculator: LanguageModelCalculator) -> dict[str, Any]:
    return {
        "model_type": "gpt2",
        "model_name": MODEL_NAME,
        "model_commit": getattr(calculator.model.config, "_commit_hash", None),
        "tokenizer_commit": calculator.tokenizer.init_kwargs.get("_commit_hash"),
        "tokenizer_class": type(calculator.tokenizer).__name__,
        "model_class": type(calculator.model).__name__,
        "device": calculator.device,
        "dtype": str(next(calculator.model.parameters()).dtype),
        "attention_implementation": getattr(
            calculator.model.config, "_attn_implementation", None
        ),
        "metric_contract": calculator.metric_contract(),
    }


def extract_features(items: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute only the frozen M0/M1 fields from labels on explicit CPU."""
    calculator = LanguageModelCalculator(model_type="gpt2", lang="en", device="cpu")
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

        for position, ((_, item), surprisal) in enumerate(
            zip(group.iterrows(), metrics["surprisals"], strict=True)
        ):
            records.append(
                {
                    "Article_ID": str(item["Article_ID"]),
                    "Text_ID": str(text_id),
                    "IA_ID": int(item["IA_ID"]),
                    "IA_LABEL": str(item["IA_LABEL"]),
                    **frozen._lexical_features(
                        str(item["IA_LABEL"]), position, len(words)
                    ),
                    "subtoken_count": int(subtoken_counts[position]),
                    "causal_surprisal": float(surprisal),
                    "is_context_scored": int(position > 0),
                }
            )
        if group_index % 20 == 0 or group_index == len(grouped):
            print(
                f"[features] scored {group_index}/{len(grouped)} display contexts",
                flush=True,
            )

    features = pd.DataFrame(records).sort_values(["Article_ID", "Text_ID", "IA_ID"])
    for column in ("log_char_length", "zipf_frequency", "causal_surprisal"):
        features[f"prev_{column}"] = features.groupby("Text_ID")[column].shift(1)
    scored = features["is_context_scored"].eq(1)
    if not np.isfinite(features.loc[scored, list(M1_FEATURES)].to_numpy(float)).all():
        raise RuntimeError("OneStop feature extraction produced non-finite values")
    return features.reset_index(drop=True), _model_revision(calculator)


def load_or_extract_features(
    items: pd.DataFrame,
    fingerprint: Mapping[str, Any],
    *,
    cache_path: Path,
    force: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata_path = cache_path.with_suffix(".metadata.json")
    code_hashes = {
        "runner_sha256": frozen._sha256(Path(__file__)),
        "frozen_protocol_sha256": frozen._sha256(PROTOCOL_PATH),
        "pipeline_sha256": frozen._sha256(PROJECT_ROOT / "core/cognition/pipeline.py"),
    }
    if cache_path.exists() and metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        reusable = (
            metadata.get("source_sha256") == fingerprint["sha256"]
            and metadata.get("item_identity_sha256")
            == fingerprint["item_identity_sha256"]
            and metadata.get("code_hashes") == code_hashes
            and metadata.get("model", {}).get("model_name") == MODEL_NAME
        )
        if reusable:
            features = pd.read_csv(
                cache_path,
                dtype={"Article_ID": str, "Text_ID": str, "IA_ID": int},
            )
            if len(features) != fingerprint["item_count"]:
                raise RuntimeError("cached OneStop feature count changed")
            print(f"[features] reusing label-only cache {cache_path}")
            return features, metadata

    features, model = extract_features(items)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(cache_path, index=False, lineterminator="\n")
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_sha256": fingerprint["sha256"],
        "item_identity_sha256": fingerprint["item_identity_sha256"],
        "item_count": len(features),
        "feature_sha256": frozen._sha256(cache_path),
        "code_hashes": code_hashes,
        "model": model,
        "device": "cpu",
        "gpu_used": False,
        "language_model_fine_tuned": False,
        "question_answer_fields_read": False,
        "corpus_precomputed_features_read": False,
        "outcome_columns_read_during_extraction": False,
    }
    frozen._write_json(metadata_path, metadata)
    return features, metadata


def load_outcomes(path: Path, trial_map: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Perform the second pass that first admits the three frozen outcomes."""
    source, diagnostics = _read_selected_source(path, columns=ANALYSIS_READ_COLUMNS)
    source["IA_LABEL"] = _normalize_labels(source["IA_LABEL"])
    source["IA_ID"] = _integer_column(source, "IA_ID")
    source["Trial_ID"] = _trial_id(source)
    if source.duplicated(["Trial_ID", "IA_ID"]).any():
        raise ValueError("duplicate OneStop participant/trial/item outcome rows")
    frame = source.merge(
        trial_map[["Trial_ID", "Article_ID", "Text_ID"]],
        on="Trial_ID",
        how="left",
        validate="many_to_one",
    )
    if frame[["Article_ID", "Text_ID"]].isna().any().any():
        raise RuntimeError("outcome pass contains a trial absent from identity pass")
    frame = frame.rename(columns={"participant_id": "Participant_ID"})
    diagnostics.update(
        {
            "participant_count": frame["Participant_ID"].nunique(),
            "article_count": frame["Article_ID"].nunique(),
            "context_count": frame["Text_ID"].nunique(),
            "outcome_columns_read": list(OUTCOME_COLUMNS),
        }
    )
    if diagnostics["article_count"] != EXPECTED_ARTICLE_COUNT:
        raise RuntimeError("outcome pass does not contain the frozen 30 articles")
    return frame, diagnostics


def _aligned_values(
    table: pd.DataFrame,
    first: str,
    second: str,
    *,
    index_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    pivot = table.pivot_table(
        index=list(index_columns),
        columns="model",
        values="spearman_rho",
        aggfunc="first",
    )
    return pivot[first].to_numpy(float), pivot[second].to_numpy(float)


def _summarize_outcome(
    predictions: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_columns = [
        f"prediction_{name}"
        for name in (*FEATURE_SETS, "target_shuffle_sentinel")
    ]
    participant_metrics = grouped_spearman_table(
        predictions,
        target_column="log_duration",
        prediction_columns=prediction_columns,
        grouping_columns=("Participant_ID",),
    )
    fold_metrics = grouped_spearman_table(
        predictions,
        target_column="log_duration",
        prediction_columns=prediction_columns,
        grouping_columns=("outer_fold", "Participant_ID"),
    )
    aggregation: dict[str, tuple[str, str]] = {
        "log_duration": ("log_duration", "mean")
    }
    for column in prediction_columns:
        aggregation[column] = (column, "first")
    item_predictions = predictions.groupby(
        ["Article_ID", "Text_ID", "IA_ID", "IA_LABEL"], as_index=False
    ).agg(**aggregation)
    article_metrics = grouped_spearman_table(
        item_predictions,
        target_column="log_duration",
        prediction_columns=prediction_columns,
        grouping_columns=("Article_ID",),
    )

    models: list[dict[str, Any]] = []
    for prediction_column in prediction_columns:
        name = prediction_column.removeprefix("prediction_")
        participant_values = participant_metrics.loc[
            participant_metrics["model"].eq(name), "spearman_rho"
        ]
        article_values = article_metrics.loc[
            article_metrics["model"].eq(name), "spearman_rho"
        ]
        fold_values = (
            fold_metrics.loc[fold_metrics["model"].eq(name)]
            .groupby("outer_fold")["spearman_rho"]
            .mean()
            .sort_index()
        )
        models.append(
            {
                "model": name,
                "macro_participant_spearman": float(np.nanmean(participant_values)),
                "macro_article_spearman": float(np.nanmean(article_values)),
                "mean_absolute_error_log": float(
                    np.mean(
                        np.abs(
                            predictions[prediction_column].to_numpy(float)
                            - predictions["log_duration"].to_numpy(float)
                        )
                    )
                ),
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
    for index, (first, second) in enumerate(COMPARISONS):
        participant_first, participant_second = _aligned_values(
            participant_metrics,
            first,
            second,
            index_columns=("Participant_ID",),
        )
        article_first, article_second = _aligned_values(
            article_metrics,
            first,
            second,
            index_columns=("Article_ID",),
        )
        fold_difference = (fold_pivot[first] - fold_pivot[second]).groupby(
            "outer_fold"
        ).mean()
        comparisons.append(
            {
                "comparison": f"{first}_minus_{second}",
                "participant_bootstrap": paired_bootstrap_mean_difference(
                    participant_first,
                    participant_second,
                    samples=bootstrap_samples,
                    seed=seed + index,
                ),
                "article_bootstrap": paired_bootstrap_mean_difference(
                    article_first,
                    article_second,
                    samples=bootstrap_samples,
                    seed=seed + 100 + index,
                ),
                "positive_outer_folds": int((fold_difference > 0).sum()),
                "outer_fold_count": len(fold_difference),
                "fold_mean_differences": {
                    str(int(fold)): float(value)
                    for fold, value in fold_difference.items()
                },
            }
        )

    sentinel_participant = participant_metrics.loc[
        participant_metrics["model"].eq("target_shuffle_sentinel"), "spearman_rho"
    ].to_numpy(float)
    sentinel_article = article_metrics.loc[
        article_metrics["model"].eq("target_shuffle_sentinel"), "spearman_rho"
    ].to_numpy(float)
    sentinel = {
        "participant_bootstrap_vs_zero": paired_bootstrap_mean_difference(
            sentinel_participant,
            np.zeros_like(sentinel_participant),
            samples=bootstrap_samples,
            seed=seed + 500,
        ),
        "article_bootstrap_vs_zero": paired_bootstrap_mean_difference(
            sentinel_article,
            np.zeros_like(sentinel_article),
            samples=bootstrap_samples,
            seed=seed + 501,
        ),
    }
    return (
        {"models": models, "comparisons": comparisons, "shuffle_sentinel": sentinel},
        participant_metrics,
        article_metrics,
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
        on=["Article_ID", "Text_ID", "IA_ID", "IA_LABEL"],
        how="left",
        validate="many_to_one",
    )
    frame[source_column] = pd.to_numeric(frame[source_column], errors="coerce")
    frame = frame.loc[
        frame["is_context_scored"].eq(1)
        & frame[source_column].notna()
        & frame[source_column].gt(0)
    ].copy()
    if not len(frame):
        raise RuntimeError(f"{outcome_name} selected no finite positive durations")
    if frame[list(M1_FEATURES)].isna().any().any():
        raise RuntimeError(f"{outcome_name} contains missing frozen features")
    frame["log_duration"] = np.log1p(frame[source_column].to_numpy(float))
    predictions, diagnostics = cross_fit_grouped_ridge(
        frame,
        group_column="Article_ID",
        target_column="log_duration",
        feature_sets=FEATURE_SETS,
        n_folds=N_FOLDS,
        alpha=RIDGE_ALPHA,
        seed=RANDOM_SEED,
        shuffled_target_model="m1_causal_surprisal",
    )
    summary, participants, articles, folds = _summarize_outcome(
        predictions,
        bootstrap_samples=bootstrap_samples,
        seed=RANDOM_SEED + 1_000 * list(OUTCOMES).index(outcome_name),
    )
    summary.update(
        {
            "source_column": source_column,
            "conditional_on_positive_duration": True,
            "row_count": len(predictions),
            "participant_count": predictions["Participant_ID"].nunique(),
            "article_count": predictions["Article_ID"].nunique(),
            "context_count": predictions["Text_ID"].nunique(),
            "cross_fit_diagnostics": diagnostics,
        }
    )
    for table in (participants, articles, folds, predictions):
        table.insert(0, "outcome", outcome_name)
    return summary, participants, articles, folds, predictions


def _comparison(summary: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return next(item for item in summary["comparisons"] if item["comparison"] == name)


def make_decision(outcomes: Mapping[str, Any]) -> dict[str, Any]:
    primary = outcomes["total_reading_time"]
    effect = _comparison(primary, "m1_causal_surprisal_minus_m0_lexical")
    effect_gate = (
        effect["participant_bootstrap"]["ci_95_low"] > 0
        and effect["article_bootstrap"]["ci_95_low"] > 0
        and effect["positive_outer_folds"] >= 4
    )
    sentinel = primary["shuffle_sentinel"]
    sentinel_gate = (
        sentinel["participant_bootstrap_vs_zero"]["ci_95_low"] <= 0
        and sentinel["article_bootstrap_vs_zero"]["ci_95_low"] <= 0
    )
    confirmed = effect_gate and sentinel_gate
    return {
        "label": (
            "causal_surprisal_confirms_on_pristine_onestop"
            if confirmed
            else "causal_surprisal_does_not_confirm_on_pristine_onestop"
        ),
        "confirmed": confirmed,
        "m1_effect_gate_passed": effect_gate,
        "shuffle_sentinel_gate_passed": sentinel_gate,
        "protocol_id": PROTOCOL_ID,
        "protocol_frozen_before_any_outcome_access": True,
        "secondary_outcomes_used_for_decision": False,
        "unfavorable_result_triggers_tuning": False,
    }


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    decision = summary["decision"]
    dataset = summary["dataset"]
    lines = [
        "# OneStop Pristine Text-Model Confirmation - Run 001",
        "",
        f"- Completed: {summary['created_at']}",
        f"- Runtime: {summary['duration_seconds']:.2f} seconds",
        "- Device: `cpu` (GPU used: false)",
        f"- Protocol: `{PROTOCOL_ID}`",
        f"- Decision: **`{decision['label']}`**",
        "",
        "## Scope and provenance",
        "",
        f"- Source SHA-256: `{dataset['sha256']}`",
        f"- Selected participants: {dataset['participant_count']}",
        f"- Complete article groups: {dataset['article_count']}",
        f"- Display contexts: {dataset['context_count']}",
        f"- Unique displayed items: {dataset['item_count']:,}",
        "- Ordinary reading, original Advanced Guardian paragraphs, first reading only.",
        "- The protocol and source/header admission were committed before outcome access.",
        "",
        "## Fixed design",
        "",
        "- Five deterministic complete-article folds; no paragraph or row leakage.",
        "- M0 lexical controls versus M1 frozen GPT-2-small causal surprisal.",
        "- Ridge alpha 1.0 and training-fold-only standardization.",
        "- A shuffled-training-target M1 sentinel was included.",
        "- Secondary outcomes are descriptive and cannot change the decision.",
        "",
    ]
    for outcome_name, outcome in summary["outcomes"].items():
        lines.extend(
            [
                f"## {outcome_name.replace('_', ' ').title()}",
                "",
                f"Rows: {outcome['row_count']:,}; participants: "
                f"{outcome['participant_count']}; articles: {outcome['article_count']}.",
                "",
                "| Model | Macro participant rho | Macro article rho | Log-duration MAE |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for model in outcome["models"]:
            lines.append(
                f"| `{model['model']}` | "
                f"{model['macro_participant_spearman']:.4f} | "
                f"{model['macro_article_spearman']:.4f} | "
                f"{model['mean_absolute_error_log']:.4f} |"
            )
        lines.extend(
            [
                "",
                "| Comparison | Participant delta [95% CI] | Article delta [95% CI] | Positive folds |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for comparison in outcome["comparisons"]:
            participant = comparison["participant_bootstrap"]
            article = comparison["article_bootstrap"]
            lines.append(
                f"| `{comparison['comparison']}` | "
                f"{participant['mean_difference']:+.4f} "
                f"[{participant['ci_95_low']:+.4f}, {participant['ci_95_high']:+.4f}] | "
                f"{article['mean_difference']:+.4f} "
                f"[{article['ci_95_low']:+.4f}, {article['ci_95_high']:+.4f}] | "
                f"{comparison['positive_outer_folds']}/{comparison['outer_fold_count']} |"
            )
        sentinel = outcome["shuffle_sentinel"]
        participant = sentinel["participant_bootstrap_vs_zero"]
        article = sentinel["article_bootstrap_vs_zero"]
        lines.extend(
            [
                "",
                "Shuffle sentinel macro rho versus zero: "
                f"participant {participant['mean_difference']:+.4f} "
                f"[{participant['ci_95_low']:+.4f}, {participant['ci_95_high']:+.4f}]; "
                f"article {article['mean_difference']:+.4f} "
                f"[{article['ci_95_low']:+.4f}, {article['ci_95_high']:+.4f}].",
                "",
            ]
        )

    lines.extend(
        [
            "## Guardrails and interpretation boundary",
            "",
            "- No question, answer, correctness, comprehension-score, or STARC span field was loaded.",
            "- No corpus-provided surprisal, frequency, word-length, syntax, or semantic field was loaded.",
            "- No gaze coordinate or duration was used as a predictor.",
            "- No feature, model size, alpha, fold, filter, or threshold search was performed.",
            "- Participants answered a question after each paragraph, so the result applies to ordinary reading for comprehension, not unrestricted browsing.",
            "- OneStop is not used for subsequent tuning regardless of the outcome.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tracked-summary", type=Path, default=DEFAULT_TRACKED_SUMMARY)
    parser.add_argument("--tracked-report", type=Path, default=DEFAULT_TRACKED_REPORT)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--force-feature-extraction", action="store_true")
    parser.add_argument("--features-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    if not args.features_only and args.bootstrap_samples != BOOTSTRAP_SAMPLES:
        raise RuntimeError("formal confirmation requires exactly 10,000 bootstraps")
    protocol = _assert_protocol_contract()
    source_state = frozen._git_state()
    if source_state["dirty"]:
        raise RuntimeError("confirmation requires a clean committed source tree")

    started = time.perf_counter()
    source_path = args.source.resolve()
    trial_map, items, dataset = load_identity(source_path)
    features, feature_metadata = load_or_extract_features(
        items,
        dataset,
        cache_path=args.cache_path.resolve(),
        force=args.force_feature_extraction,
    )
    if args.features_only:
        print("[complete] label-only features frozen before outcome access")
        print(f"[complete] feature_sha256={feature_metadata['feature_sha256']}")
        return 0

    raw, outcome_read = load_outcomes(source_path, trial_map)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outcome_summaries: dict[str, Any] = {}
    participant_tables: list[pd.DataFrame] = []
    article_tables: list[pd.DataFrame] = []
    fold_tables: list[pd.DataFrame] = []
    prediction_paths: list[Path] = []
    for outcome_name, source_column in OUTCOMES.items():
        print(f"[benchmark] evaluating {outcome_name}", flush=True)
        outcome, participants, articles, folds, predictions = evaluate_outcome(
            raw,
            features,
            outcome_name=outcome_name,
            source_column=source_column,
            bootstrap_samples=args.bootstrap_samples,
        )
        outcome_summaries[outcome_name] = outcome
        participant_tables.append(participants)
        article_tables.append(articles)
        fold_tables.append(folds)
        prediction_path = output_dir / f"cross_fitted_{outcome_name}.csv.gz"
        keep = [
            "outcome",
            "Participant_ID",
            "Article_ID",
            "Text_ID",
            "IA_ID",
            "IA_LABEL",
            "outer_fold",
            "log_duration",
            *[column for column in predictions if column.startswith("prediction_")],
        ]
        predictions[keep].to_csv(
            prediction_path,
            index=False,
            compression="gzip",
            lineterminator="\n",
        )
        prediction_paths.append(prediction_path)

    duration = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "experiment": "onestop_gpt2_small_pristine_confirmation_run_001",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": frozen._sha256(PROTOCOL_PATH),
        "created_at": datetime.now(UTC).isoformat(),
        "duration_seconds": duration,
        "configuration": {
            "seed": RANDOM_SEED,
            "outer_folds": N_FOLDS,
            "ridge_alpha": RIDGE_ALPHA,
            "bootstrap_samples": args.bootstrap_samples,
            "feature_sets": {key: list(value) for key, value in FEATURE_SETS.items()},
            "outcomes": OUTCOMES,
            "primary_outcome": "total_reading_time",
        },
        "dataset": dataset,
        "outcome_read": outcome_read,
        "feature_cache": {
            "path": args.cache_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
            **feature_metadata,
        },
        "outcomes": outcome_summaries,
        "decision": make_decision(outcome_summaries),
        "compute": {
            "device": "cpu",
            "gpu_allowed": False,
            "gpu_used": False,
            "language_model_fine_tuned": False,
        },
        "leakage_controls": {
            "question_answer_fields_read": False,
            "corpus_precomputed_features_read": False,
            "gaze_feature_used_as_predictor": False,
            "complete_article_holdout": True,
            "scaler_fit_on_training_only": True,
            "target_shuffle_sentinel_included": True,
            "hyperparameter_search_used": False,
            "feature_cache_completed_before_outcome_pass": True,
            "protocol_frozen_before_any_outcome_access": True,
        },
        "source": source_state,
        "protocol_snapshot": protocol,
    }

    local_summary = output_dir / "summary.json"
    local_report = output_dir / "report.md"
    participant_path = output_dir / "participant_metrics.csv"
    article_path = output_dir / "article_metrics.csv"
    fold_path = output_dir / "fold_metrics.csv"
    manifest_path = output_dir / "manifest.json"
    frozen._write_json(local_summary, summary)
    frozen._write_json(args.tracked_summary.resolve(), summary)
    write_report(local_report, summary)
    write_report(args.tracked_report.resolve(), summary)
    pd.concat(participant_tables, ignore_index=True).to_csv(
        participant_path, index=False, lineterminator="\n"
    )
    pd.concat(article_tables, ignore_index=True).to_csv(
        article_path, index=False, lineterminator="\n"
    )
    pd.concat(fold_tables, ignore_index=True).to_csv(
        fold_path, index=False, lineterminator="\n"
    )
    artifact_paths = [
        local_summary,
        local_report,
        participant_path,
        article_path,
        fold_path,
        *prediction_paths,
    ]
    manifest = {
        "schema_version": 1,
        "experiment": summary["experiment"],
        "created_at": summary["created_at"],
        "dataset_sha256": dataset["sha256"],
        "feature_sha256": feature_metadata["feature_sha256"],
        "protocol_sha256": summary["protocol_sha256"],
        "source": source_state,
        "artifacts": {
            path.name: {
                "sha256": frozen._sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
        },
    }
    frozen._write_json(manifest_path, manifest)
    print(f"[complete] {summary['decision']['label']}")
    print(f"[complete] runtime={duration:.2f}s device=cpu")
    print(f"[complete] report={args.tracked_report.resolve()}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())

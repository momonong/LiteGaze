"""Run the frozen non-Chinese-source causal-LM development benchmark.

The command has a strict two-stage boundary: every candidate first completes
label-free feature extraction, and only then are Provo reading-time outcomes
loaded for grouped evaluation.  Raw features and predictions remain under the
ignored ``data/`` tree; compact audited results are promoted separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import time
import traceback
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.cognition.causal_backbone import (
    BackboneSpec,
    FrozenCausalBackbone,
    model_id_has_excluded_prefix,
)
from core.cognition.generalization import (
    cross_fit_grouped_ridge,
    grouped_spearman_table,
    paired_bootstrap_mean_difference,
)
from scripts import run_provo_text_model_experiment as legacy_provo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "docs/experiments/protocols/2026-08-08-non-cn-text-backbone-provo-dev-v1.json"
)
DEFAULT_PROVO_PATH = PROJECT_ROOT / "data/provo/raw/Provo_Corpus-Eyetracking_Data.csv"
DEFAULT_CACHE_DIR = (
    PROJECT_ROOT / "data/provo/text_modeling/non-cn-text-backbone-provo-dev-v1"
)
DEFAULT_OUTPUT_DIR = DEFAULT_CACHE_DIR / "run-001"
IDENTITY_COLUMNS = ("Participant_ID", "Text_ID", "IA_ID", "IA_LABEL")
OUTCOME_COLUMNS = {
    "total_reading_time": "IA_DWELL_TIME",
    "gaze_duration": "IA_FIRST_RUN_DWELL_TIME",
    "first_fixation_duration": "IA_FIRST_FIXATION_DURATION",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
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


def load_protocol(path: Path) -> tuple[dict[str, Any], list[BackboneSpec]]:
    """Load and enforce the exact source allowlist before any model access."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != "non-cn-text-backbone-provo-development-v1":
        raise ValueError("unexpected backbone benchmark protocol")
    policy = payload["source_policy"]
    if policy.get("mode") != "exact_allowlist":
        raise ValueError("model source policy must use an exact allowlist")
    if policy.get("trust_remote_code") is not False:
        raise ValueError("trust_remote_code must remain false")
    if policy.get("instruction_or_chat_models_allowed") is not False:
        raise ValueError("instruction/chat models are forbidden")

    excluded = tuple(policy["excluded_model_id_prefixes"])
    specs: list[BackboneSpec] = []
    keys: set[str] = set()
    identities: set[tuple[str, str]] = set()
    for item in policy["allowlist"]:
        spec = BackboneSpec(
            key=str(item["key"]),
            model_id=str(item["model_id"]),
            revision=str(item["revision"]),
            developer=str(item["developer"]),
            license=str(item["license"]),
            role=str(item["role"]),
        )
        if spec.key in keys:
            raise ValueError(f"duplicate backbone key: {spec.key}")
        if (spec.model_id, spec.revision) in identities:
            raise ValueError(f"duplicate model identity: {spec.model_id}@{spec.revision}")
        if not re.fullmatch(r"[0-9a-f]{40}", spec.revision):
            raise ValueError(f"backbone revision is not an immutable SHA: {spec.key}")
        if model_id_has_excluded_prefix(spec.model_id, excluded):
            raise ValueError(f"excluded model source reached allowlist: {spec.model_id}")
        if spec.license.casefold() not in {"mit", "apache-2.0"}:
            raise ValueError(f"unapproved model license for {spec.key}: {spec.license}")
        keys.add(spec.key)
        identities.add((spec.model_id, spec.revision))
        specs.append(spec)
    if not specs or specs[0].key != "gpt2":
        raise ValueError("gpt2 must be the first and immutable baseline")
    return payload, specs


def verify_remote_model_identities(
    specs: Sequence[BackboneSpec],
) -> dict[str, dict[str, Any]]:
    """Confirm every frozen repository remains official, public, and ungated."""
    from huggingface_hub import HfApi

    api = HfApi()
    verified: dict[str, dict[str, Any]] = {}
    for spec in specs:
        info = api.model_info(
            spec.model_id,
            revision=spec.revision,
            files_metadata=False,
        )
        if info.sha != spec.revision:
            raise RuntimeError(
                f"remote identity mismatch for {spec.key}: {info.sha!r}"
            )
        if info.private or bool(info.gated):
            raise RuntimeError(
                f"{spec.key} is no longer a public ungated repository"
            )
        verified[spec.key] = {
            "model_id": spec.model_id,
            "requested_revision": spec.revision,
            "resolved_revision": info.sha,
            "private": bool(info.private),
            "gated": bool(info.gated),
        }
    return verified


def _normalize_label(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value)).strip()


def load_label_free_provo_items(
    path: Path,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read identities only, leaving every duration column unopened."""
    expected = protocol["dataset"]
    source_hash = _sha256(path)
    if source_hash != expected["sha256"]:
        raise RuntimeError("Provo source hash does not match the frozen protocol")
    raw = pd.read_csv(
        path,
        usecols=list(IDENTITY_COLUMNS),
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    raw["IA_LABEL"] = raw["IA_LABEL"].map(_normalize_label)
    raw["Text_ID"] = raw["Text_ID"].astype(str)
    raw["IA_ID"] = pd.to_numeric(raw["IA_ID"], errors="raise").astype(int)
    if raw[list(IDENTITY_COLUMNS)].eq("").any().any():
        raise ValueError("Provo identity/display fields must be complete")
    if raw.duplicated(["Participant_ID", "Text_ID", "IA_ID"]).any():
        raise ValueError("duplicate participant/text/item row in Provo")
    items = raw[["Text_ID", "IA_ID", "IA_LABEL"]].drop_duplicates()
    if items.groupby(["Text_ID", "IA_ID"])["IA_LABEL"].nunique().gt(1).any():
        raise ValueError("inconsistent Provo display label")
    items = items.sort_values(["Text_ID", "IA_ID"]).reset_index(drop=True)
    observed = {
        "source_sha256": source_hash,
        "row_count": len(raw),
        "participant_count": raw["Participant_ID"].nunique(),
        "text_count": items["Text_ID"].nunique(),
        "item_count": len(items),
        "outcome_columns_read": False,
    }
    expected_identity = {
        "participant_count": expected["expected_participants"],
        "text_count": expected["expected_texts"],
        "item_count": expected["expected_items"],
    }
    if {key: observed[key] for key in expected_identity} != expected_identity:
        raise RuntimeError(f"unexpected Provo identity: {observed}")
    return items, observed


def _text_sort_key(value: str) -> tuple[int, int | str]:
    normalized = str(value)
    try:
        return (0, int(normalized))
    except ValueError:
        return (1, normalized)


def _score_item_groups(
    backbone: FrozenCausalBackbone,
    groups: Sequence[tuple[str, pd.DataFrame]],
    *,
    progress_offset: int,
    progress_total: int,
    maximum_peak_reserved_bytes: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for local_index, (text_id, raw_group) in enumerate(groups, start=1):
        group = raw_group.sort_values("IA_ID")
        words = group["IA_LABEL"].astype(str).tolist()
        result = backbone.score_words(words)
        if backbone.peak_cuda_reserved_bytes > maximum_peak_reserved_bytes:
            raise RuntimeError(
                f"{backbone.spec.key} exceeded peak CUDA reserved budget: "
                f"{backbone.peak_cuda_reserved_bytes} > {maximum_peak_reserved_bytes}"
            )
        for position, ((_, item), surprisal, subtoken_count) in enumerate(
            zip(
                group.iterrows(),
                result.surprisals,
                result.subtoken_counts,
                strict=True,
            )
        ):
            records.append(
                {
                    "Text_ID": str(text_id),
                    "IA_ID": int(item["IA_ID"]),
                    "IA_LABEL": str(item["IA_LABEL"]),
                    **legacy_provo._lexical_features(
                        str(item["IA_LABEL"]),
                        position,
                        len(words),
                    ),
                    "subtoken_count": int(subtoken_count),
                    "causal_surprisal": float(surprisal),
                    "is_context_scored": int(position > 0),
                }
            )
        completed = progress_offset + local_index
        if completed % 10 == 0 or completed == progress_total:
            print(
                f"[features:{backbone.spec.key}] scored "
                f"{completed}/{progress_total} passages",
                flush=True,
            )
    return pd.DataFrame(records)


def _finalize_features(features: pd.DataFrame) -> pd.DataFrame:
    result = features.sort_values(["Text_ID", "IA_ID"]).reset_index(drop=True)
    for column in ("log_char_length", "zipf_frequency", "causal_surprisal"):
        result[f"prev_{column}"] = result.groupby("Text_ID")[column].shift(1)
    scored = result["is_context_scored"].eq(1)
    required = (
        "log_char_length",
        "zipf_frequency",
        "relative_position",
        "starts_upper",
        "ends_punctuation",
        "prev_log_char_length",
        "prev_zipf_frequency",
        "subtoken_count",
        "causal_surprisal",
        "prev_causal_surprisal",
    )
    if result.loc[scored, list(required)].isna().any().any():
        raise RuntimeError("context-scored features contain missing values")
    matrix = result.loc[scored, list(required)].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise RuntimeError("context-scored features contain non-finite values")
    if result["subtoken_count"].lt(1).any():
        raise RuntimeError("at least one display item has zero tokenizer coverage")
    return result


def _cache_paths(cache_dir: Path, key: str) -> tuple[Path, Path, Path]:
    return (
        cache_dir / f"{key}.features.csv",
        cache_dir / f"{key}.metadata.json",
        cache_dir / f"{key}.failure.json",
    )


def extract_or_load_backbone(
    items: pd.DataFrame,
    dataset: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_path: Path,
    spec: BackboneSpec,
    remote_identity: Mapping[str, Any],
    *,
    cache_dir: Path,
    force: bool,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    feature_path, metadata_path, failure_path = _cache_paths(cache_dir, spec.key)
    implementation_hashes = {
        "runner_sha256": _sha256(Path(__file__)),
        "extractor_sha256": _sha256(
            PROJECT_ROOT / "core/cognition/causal_backbone.py"
        ),
        "lexical_protocol_sha256": _sha256(Path(legacy_provo.__file__)),
        "protocol_sha256": _sha256(protocol_path),
    }
    if feature_path.exists() and metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        reusable = (
            metadata.get("status") == "complete"
            and metadata.get("source_sha256") == dataset["source_sha256"]
            and metadata.get("item_count") == len(items)
            and metadata.get("model", {}).get("requested_revision") == spec.revision
            and metadata.get("implementation_hashes") == implementation_hashes
            and metadata.get("feature_sha256") == _sha256(feature_path)
        )
        if reusable:
            print(f"[features:{spec.key}] reusing {feature_path}", flush=True)
            return (
                pd.read_csv(feature_path, dtype={"Text_ID": str, "IA_ID": int}),
                {**metadata, "cache_reused": True},
            )

    smoke_count = int(protocol["smoke_gate"]["texts"])
    text_ids = sorted(items["Text_ID"].unique(), key=_text_sort_key)
    smoke_ids = text_ids[:smoke_count]
    grouped = {
        str(text_id): group
        for text_id, group in items.groupby("Text_ID", sort=False)
    }
    smoke_groups = [(text_id, grouped[text_id]) for text_id in smoke_ids]
    remaining_groups = [
        (text_id, grouped[text_id]) for text_id in text_ids if text_id not in smoke_ids
    ]
    maximum_peak_reserved_bytes = int(
        float(protocol["smoke_gate"]["maximum_peak_cuda_reserved_gib"]) * 2**30
    )
    backbone: FrozenCausalBackbone | None = None
    started = time.perf_counter()
    try:
        backbone = FrozenCausalBackbone(
            spec,
            device=protocol["compute"]["device"],
            dtype=protocol["compute"]["dtype"],
            separator_policy=protocol["feature_extraction"].get(
                "separator_only_token_policy", "reject"
            ),
        )
        smoke_features = _score_item_groups(
            backbone,
            smoke_groups,
            progress_offset=0,
            progress_total=len(text_ids),
            maximum_peak_reserved_bytes=maximum_peak_reserved_bytes,
        )
        smoke_features = _finalize_features(smoke_features)
        smoke_item_count = int(items["Text_ID"].isin(smoke_ids).sum())
        item_coverage = len(smoke_features) / smoke_item_count
        minimum_subtoken_count = int(smoke_features["subtoken_count"].min())
        if item_coverage < float(protocol["smoke_gate"]["required_item_coverage"]):
            raise RuntimeError(
                f"{spec.key} smoke item coverage {item_coverage:.6f} is below "
                "the frozen requirement"
            )
        if minimum_subtoken_count < int(
            protocol["smoke_gate"]["minimum_tokens_per_word"]
        ):
            raise RuntimeError(
                f"{spec.key} smoke tokenizer left a display word uncovered"
            )
        smoke_summary = {
            "text_ids": smoke_ids,
            "text_count": len(smoke_ids),
            "item_count": len(smoke_features),
            "item_coverage": item_coverage,
            "minimum_subtoken_count": minimum_subtoken_count,
            "finite_context_scored_features": True,
            "peak_cuda_reserved_bytes": backbone.peak_cuda_reserved_bytes,
            "passed": True,
        }
        remaining_features = _score_item_groups(
            backbone,
            remaining_groups,
            progress_offset=len(smoke_groups),
            progress_total=len(text_ids),
            maximum_peak_reserved_bytes=maximum_peak_reserved_bytes,
        )
        features = _finalize_features(
            pd.concat([smoke_features, remaining_features], ignore_index=True)
        )
        if len(features) != len(items):
            raise RuntimeError("full feature extraction did not preserve every item")
        model_metadata = backbone.metadata()
        cache_dir.mkdir(parents=True, exist_ok=True)
        features.to_csv(feature_path, index=False, lineterminator="\n")
        metadata = {
            "schema_version": 1,
            "status": "complete",
            "created_at": datetime.now(UTC).isoformat(),
            "duration_seconds": time.perf_counter() - started,
            "source_sha256": dataset["source_sha256"],
            "item_count": len(features),
            "text_count": features["Text_ID"].nunique(),
            "feature_sha256": _sha256(feature_path),
            "implementation_hashes": implementation_hashes,
            "remote_identity": dict(remote_identity),
            "model": model_metadata,
            "smoke_gate": smoke_summary,
            "question_answer_dataset_used": False,
            "outcome_columns_read_during_extraction": False,
            "language_model_fine_tuned": False,
            "cache_reused": False,
        }
        _write_json(metadata_path, metadata)
        if failure_path.exists():
            failure_path.unlink()
        return features, metadata
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "created_at": datetime.now(UTC).isoformat(),
            "duration_seconds": time.perf_counter() - started,
            "backbone": {
                "key": spec.key,
                "model_id": spec.model_id,
                "revision": spec.revision,
            },
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "outcome_columns_read": False,
        }
        _write_json(failure_path, failure)
        return None, failure
    finally:
        if backbone is not None:
            backbone.close()


def _paired_model_comparison(
    participant_metrics: pd.DataFrame,
    text_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    *,
    first: str,
    second: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    participant_pivot = participant_metrics.pivot_table(
        index="Participant_ID",
        columns="model",
        values="spearman_rho",
        aggfunc="first",
    )
    text_pivot = text_metrics.pivot_table(
        index="Text_ID",
        columns="model",
        values="spearman_rho",
        aggfunc="first",
    )
    fold_pivot = fold_metrics.pivot_table(
        index=["outer_fold", "Participant_ID"],
        columns="model",
        values="spearman_rho",
        aggfunc="first",
    )
    per_fold = (fold_pivot[first] - fold_pivot[second]).groupby("outer_fold").mean()
    return {
        "comparison": f"{first}_minus_{second}",
        "participant_bootstrap": paired_bootstrap_mean_difference(
            participant_pivot[first],
            participant_pivot[second],
            samples=bootstrap_samples,
            seed=seed,
        ),
        "text_bootstrap": paired_bootstrap_mean_difference(
            text_pivot[first],
            text_pivot[second],
            samples=bootstrap_samples,
            seed=seed + 100,
        ),
        "positive_outer_folds": int((per_fold > 0).sum()),
        "outer_fold_count": int(len(per_fold)),
        "fold_mean_differences": {
            str(int(fold)): float(value) for fold, value in per_fold.items()
        },
    }


def evaluate_backbone_outcome(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    protocol: Mapping[str, Any],
    *,
    outcome_name: str,
    source_column: str,
    seed_offset: int,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
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
    feature_sets = {
        name: tuple(columns)
        for name, columns in protocol["analysis"]["feature_sets"].items()
    }
    feature_columns = sorted({column for columns in feature_sets.values() for column in columns})
    if frame[feature_columns].isna().any().any():
        raise RuntimeError(f"{outcome_name} contains missing benchmark features")

    predictions, diagnostics = cross_fit_grouped_ridge(
        frame,
        group_column="Text_ID",
        target_column="log_duration",
        feature_sets=feature_sets,
        n_folds=int(protocol["analysis"]["outer_folds"]),
        alpha=float(protocol["analysis"]["ridge_alpha"]),
        seed=int(protocol["analysis"]["seed"]),
        shuffled_target_model=protocol["analysis"]["target_shuffle_sentinel_model"],
    )
    prediction_columns = [
        f"prediction_{name}"
        for name in (*feature_sets.keys(), "target_shuffle_sentinel")
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
    aggregate_spec: dict[str, tuple[str, str]] = {
        "log_duration": ("log_duration", "mean")
    }
    for prediction_column in prediction_columns:
        aggregate_spec[prediction_column] = (prediction_column, "first")
    item_predictions = predictions.groupby(
        ["Text_ID", "IA_ID"], as_index=False
    ).agg(**aggregate_spec)
    text_metrics = grouped_spearman_table(
        item_predictions,
        target_column="log_duration",
        prediction_columns=prediction_columns,
        grouping_columns=("Text_ID",),
    )

    models: list[dict[str, Any]] = []
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
        models.append(
            {
                "model": model_name,
                "macro_participant_spearman": float(np.nanmean(participant_values)),
                "macro_text_spearman": float(np.nanmean(text_values)),
                "mean_absolute_error_log": float(
                    np.mean(
                        np.abs(
                            predictions[prediction_column].to_numpy(dtype=float)
                            - predictions["log_duration"].to_numpy(dtype=float)
                        )
                    )
                ),
                "fold_macro_participant_spearman": {
                    str(int(fold)): float(value) for fold, value in fold_values.items()
                },
            }
        )

    bootstrap_samples = int(protocol["analysis"]["bootstrap_samples"])
    increment = _paired_model_comparison(
        participant_metrics,
        text_metrics,
        fold_metrics,
        first="m1_causal_surprisal",
        second="m0_tokenization",
        bootstrap_samples=bootstrap_samples,
        seed=int(protocol["analysis"]["seed"]) + seed_offset,
    )
    tokenization_increment = _paired_model_comparison(
        participant_metrics,
        text_metrics,
        fold_metrics,
        first="m0_tokenization",
        second="m0_lexical_shared",
        bootstrap_samples=bootstrap_samples,
        seed=int(protocol["analysis"]["seed"]) + seed_offset + 10,
    )
    gate = protocol["interpretation"]["incremental_gate"]
    incremental_gate_passed = (
        increment["participant_bootstrap"]["ci_95_low"]
        > float(gate["participant_bootstrap_ci_95_low_greater_than"])
        and increment["text_bootstrap"]["ci_95_low"]
        > float(gate["text_bootstrap_ci_95_low_greater_than"])
        and increment["positive_outer_folds"]
        >= int(gate["minimum_positive_outer_folds"])
    )
    summary = {
        "source_column": source_column,
        "conditional_on_positive_duration": True,
        "row_count": len(predictions),
        "participant_count": predictions["Participant_ID"].nunique(),
        "text_count": predictions["Text_ID"].nunique(),
        "models": models,
        "comparisons": [tokenization_increment, increment],
        "incremental_gate_passed": incremental_gate_passed,
        "cross_fit_diagnostics": diagnostics,
    }
    return summary, {
        "participant": participant_metrics,
        "text": text_metrics,
        "fold": fold_metrics,
    }


def _m1_metrics(tables: Mapping[str, pd.DataFrame], key: str) -> pd.DataFrame:
    result = tables[key]
    return result.loc[result["model"].eq("m1_causal_surprisal")].copy()


def _cross_backbone_comparison(
    tables_by_backbone: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    first: str,
    second: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    participant = pd.concat(
        [
            _m1_metrics(tables, "participant").assign(backbone=key)
            for key, tables in tables_by_backbone.items()
        ],
        ignore_index=True,
    )
    text = pd.concat(
        [
            _m1_metrics(tables, "text").assign(backbone=key)
            for key, tables in tables_by_backbone.items()
        ],
        ignore_index=True,
    )
    fold = pd.concat(
        [
            _m1_metrics(tables, "fold").assign(backbone=key)
            for key, tables in tables_by_backbone.items()
        ],
        ignore_index=True,
    )
    participant_pivot = participant.pivot_table(
        index="Participant_ID",
        columns="backbone",
        values="spearman_rho",
        aggfunc="first",
    )
    text_pivot = text.pivot_table(
        index="Text_ID",
        columns="backbone",
        values="spearman_rho",
        aggfunc="first",
    )
    fold_pivot = fold.pivot_table(
        index=["outer_fold", "Participant_ID"],
        columns="backbone",
        values="spearman_rho",
        aggfunc="first",
    )
    per_fold = (fold_pivot[first] - fold_pivot[second]).groupby("outer_fold").mean()
    return {
        "comparison": f"{first}_minus_{second}",
        "participant_bootstrap": paired_bootstrap_mean_difference(
            participant_pivot[first],
            participant_pivot[second],
            samples=bootstrap_samples,
            seed=seed,
        ),
        "text_bootstrap": paired_bootstrap_mean_difference(
            text_pivot[first],
            text_pivot[second],
            samples=bootstrap_samples,
            seed=seed + 100,
        ),
        "positive_outer_folds": int((per_fold > 0).sum()),
        "outer_fold_count": int(len(per_fold)),
        "fold_mean_differences": {
            str(int(fold_id)): float(value) for fold_id, value in per_fold.items()
        },
    }


def _find_model(summary: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return next(model for model in summary["models"] if model["model"] == name)


def _find_comparison(summary: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return next(
        comparison
        for comparison in summary["comparisons"]
        if comparison["comparison"] == name
    )


def evaluate(
    protocol: Mapping[str, Any],
    protocol_path: Path,
    specs: Sequence[BackboneSpec],
    features_by_backbone: Mapping[str, pd.DataFrame],
    metadata_by_backbone: Mapping[str, Mapping[str, Any]],
    *,
    provo_path: Path,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Mapping[str, pd.DataFrame]]]]:
    """Open outcomes only after all label-free extraction attempts finish."""
    raw, _, fingerprint = legacy_provo.load_provo(provo_path)
    if fingerprint["sha256"] != protocol["dataset"]["sha256"]:
        raise RuntimeError("evaluation source changed after label-free extraction")

    outcomes_by_backbone: dict[str, Any] = {}
    metric_tables: dict[str, dict[str, Mapping[str, pd.DataFrame]]] = {}
    for backbone_index, spec in enumerate(specs):
        if spec.key not in features_by_backbone:
            continue
        print(f"[evaluation:{spec.key}] starting grouped outcomes", flush=True)
        outcomes: dict[str, Any] = {}
        metric_tables[spec.key] = {}
        for outcome_index, (outcome_name, source_column) in enumerate(
            OUTCOME_COLUMNS.items()
        ):
            summary, tables = evaluate_backbone_outcome(
                raw,
                features_by_backbone[spec.key],
                protocol,
                outcome_name=outcome_name,
                source_column=source_column,
                seed_offset=1_000 * backbone_index + 100 * outcome_index,
            )
            outcomes[outcome_name] = summary
            metric_tables[spec.key][outcome_name] = tables
        outcomes_by_backbone[spec.key] = outcomes

    if "gpt2" not in outcomes_by_backbone:
        raise RuntimeError("gpt2 baseline is required for cross-backbone evaluation")
    primary_name = protocol["analysis"]["primary_outcome"]["name"]
    primary_tables = {
        key: outcomes[primary_name]
        for key, outcomes in metric_tables.items()
    }
    bootstrap_samples = int(protocol["analysis"]["bootstrap_samples"])
    base_seed = int(protocol["analysis"]["seed"]) + 50_000
    vs_gpt2: dict[str, Any] = {}
    for index, spec in enumerate(specs):
        if spec.key == "gpt2" or spec.key not in primary_tables:
            continue
        vs_gpt2[spec.key] = _cross_backbone_comparison(
            primary_tables,
            first=spec.key,
            second="gpt2",
            bootstrap_samples=bootstrap_samples,
            seed=base_seed + index,
        )

    planned = protocol["interpretation"]["planned_training_contrast"]
    planned_contrast = None
    if planned["first"] in primary_tables and planned["second"] in primary_tables:
        planned_contrast = _cross_backbone_comparison(
            primary_tables,
            first=planned["first"],
            second=planned["second"],
            bootstrap_samples=bootstrap_samples,
            seed=base_seed + 1_000,
        )

    shortlist_rule = protocol["interpretation"]["shortlist_rule"]
    eligible: list[tuple[float, str]] = []
    eligibility: dict[str, Any] = {}
    for spec in specs:
        if spec.key == "gpt2":
            continue
        if spec.key not in outcomes_by_backbone:
            status = metadata_by_backbone.get(spec.key, {}).get(
                "status", "not_available"
            )
            eligibility[spec.key] = {
                "incremental_gate_passed": False,
                "eligible": False,
                "reason": status,
            }
            continue
        cross = vs_gpt2[spec.key]
        participant_delta = cross["participant_bootstrap"]["mean_difference"]
        text_delta = cross["text_bootstrap"]["mean_difference"]
        incremental_pass = outcomes_by_backbone[spec.key][primary_name][
            "incremental_gate_passed"
        ]
        passes = (
            incremental_pass
            and participant_delta
            >= float(shortlist_rule["participant_point_difference_vs_gpt2_at_least"])
            and text_delta
            >= float(shortlist_rule["text_point_difference_vs_gpt2_at_least"])
        )
        ranking_score = min(participant_delta, text_delta)
        eligibility[spec.key] = {
            "incremental_gate_passed": incremental_pass,
            "participant_point_difference_vs_gpt2": participant_delta,
            "text_point_difference_vs_gpt2": text_delta,
            "ranking_score": ranking_score,
            "eligible": passes,
        }
        if passes:
            eligible.append((ranking_score, spec.key))
    eligible.sort(key=lambda item: (-item[0], item[1]))
    shortlist = [
        key
        for _, key in eligible[
            : int(protocol["interpretation"]["maximum_non_gpt2_shortlist_size"])
        ]
    ]

    total_inference_seconds = sum(
        float(metadata["model"]["inference_seconds"])
        for metadata in metadata_by_backbone.values()
        if metadata.get("status") == "complete"
    )
    summary = {
        "schema_version": 1,
        "experiment": "non_cn_text_backbone_provo_development_run_001",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "path": protocol_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _sha256(protocol_path),
            "status_at_freeze": protocol["status_at_freeze"],
        },
        "dataset": {
            **fingerprint,
            "role": protocol["dataset"]["role"],
            "question_answer_dataset_used": False,
        },
        "configuration": protocol["analysis"],
        "backbone_features": metadata_by_backbone,
        "outcomes": outcomes_by_backbone,
        "cross_backbone_primary": {
            "outcome": primary_name,
            "comparisons_vs_gpt2": vs_gpt2,
            "planned_training_contrast": planned_contrast,
        },
        "decision": {
            "label": (
                "development_challengers_shortlisted"
                if shortlist
                else "retain_gpt2_no_challenger_shortlisted"
            ),
            "shortlist": shortlist,
            "eligibility": eligibility,
            "production_model_changed": False,
            "confirmation_claim_allowed": False,
            "required_next_evidence": protocol["interpretation"][
                "required_next_evidence"
            ],
        },
        "compute": {
            "device": protocol["compute"]["device"],
            "dtype": protocol["compute"]["dtype"],
            "one_backbone_loaded_at_a_time": True,
            "language_model_fine_tuned": False,
            "total_inference_seconds": total_inference_seconds,
            "total_inference_gpu_hours": total_inference_seconds / 3600,
            "gpu_budget_hours": protocol["compute"][
                "maximum_total_inference_gpu_hours"
            ],
            "gpu_budget_exceeded": total_inference_seconds / 3600
            > float(protocol["compute"]["maximum_total_inference_gpu_hours"]),
        },
        "leakage_controls": {
            "features_extracted_before_outcomes_loaded": True,
            "question_answer_dataset_used": False,
            "complete_text_holdout": True,
            "scaler_fit_on_training_only": True,
            "target_shuffle_sentinel_included": True,
            "language_model_fine_tuned": False,
            "provo_role": "development_only",
            "onestop_accessed": False,
        },
        "source": _git_state(),
    }
    return summary, metric_tables


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    primary_name = summary["cross_backbone_primary"]["outcome"]
    lines = [
        "# Non-Chinese-Source Text Backbone Development Screen — Run 001",
        "",
        f"- Completed: {summary['created_at']}",
        f"- Decision: **`{summary['decision']['label']}`**",
        f"- Shortlist: `{summary['decision']['shortlist']}`",
        "- Scope: Provo development evidence only; no production model change",
        "- QA data: not used",
        "",
        "## Frozen backbone results",
        "",
        "| Backbone | M1 participant rho | M1 text rho | M1-M0 participant delta [95% CI] | M1-M0 text delta [95% CI] | Positive folds | Peak reserved GiB | Tokens/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, outcomes in summary["outcomes"].items():
        outcome = outcomes[primary_name]
        m1 = _find_model(outcome, "m1_causal_surprisal")
        increment = _find_comparison(
            outcome,
            "m1_causal_surprisal_minus_m0_tokenization",
        )
        participant = increment["participant_bootstrap"]
        text = increment["text_bootstrap"]
        model = summary["backbone_features"][key]["model"]
        lines.append(
            f"| `{key}` | {m1['macro_participant_spearman']:.4f} | "
            f"{m1['macro_text_spearman']:.4f} | "
            f"{participant['mean_difference']:+.4f} "
            f"[{participant['ci_95_low']:+.4f}, {participant['ci_95_high']:+.4f}] | "
            f"{text['mean_difference']:+.4f} "
            f"[{text['ci_95_low']:+.4f}, {text['ci_95_high']:+.4f}] | "
            f"{increment['positive_outer_folds']}/{increment['outer_fold_count']} | "
            f"{model['peak_cuda_reserved_bytes'] / 2**30:.3f} | "
            f"{model['tokens_per_second']:.1f} |"
        )

    failures = {
        key: metadata
        for key, metadata in summary["backbone_features"].items()
        if metadata.get("status") != "complete"
    }
    if failures:
        lines.extend(
            [
                "",
                "## Frozen technical failures",
                "",
                "| Backbone | Status | Recorded error | Outcome columns read |",
                "| --- | --- | --- | ---: |",
            ]
        )
        for key, metadata in failures.items():
            error = str(metadata.get("error", "not run")).replace("|", "\\|")
            lines.append(
                f"| `{key}` | `{metadata.get('status')}` | {error} | "
                f"{str(metadata.get('outcome_columns_read', False)).lower()} |"
            )

    lines.extend(
        [
            "",
            "## Paired M1 comparisons versus GPT-2",
            "",
            "| Challenger | Participant delta [95% CI] | Text delta [95% CI] | Positive folds | Eligible |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    comparisons = summary["cross_backbone_primary"]["comparisons_vs_gpt2"]
    for key, comparison in comparisons.items():
        participant = comparison["participant_bootstrap"]
        text = comparison["text_bootstrap"]
        eligible = summary["decision"]["eligibility"][key]["eligible"]
        lines.append(
            f"| `{key}` | {participant['mean_difference']:+.4f} "
            f"[{participant['ci_95_low']:+.4f}, {participant['ci_95_high']:+.4f}] | "
            f"{text['mean_difference']:+.4f} "
            f"[{text['ci_95_low']:+.4f}, {text['ci_95_high']:+.4f}] | "
            f"{comparison['positive_outer_folds']}/{comparison['outer_fold_count']} | "
            f"{str(eligible).lower()} |"
        )

    planned = summary["cross_backbone_primary"]["planned_training_contrast"]
    if planned is not None:
        participant = planned["participant_bootstrap"]
        text = planned["text_bootstrap"]
        lines.extend(
            [
                "",
                "## Planned Pythia training contrast",
                "",
                f"`{planned['comparison']}`: participant "
                f"{participant['mean_difference']:+.4f} "
                f"[{participant['ci_95_low']:+.4f}, {participant['ci_95_high']:+.4f}]; "
                f"text {text['mean_difference']:+.4f} "
                f"[{text['ci_95_low']:+.4f}, {text['ci_95_high']:+.4f}]; "
                f"positive folds {planned['positive_outer_folds']}/"
                f"{planned['outer_fold_count']}.",
            ]
        )

    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Exact non-Chinese-source allowlist and immutable revisions enforced before download.",
            "- All label-free features were extracted before outcome columns were opened.",
            "- No model was trained, fine-tuned, prompted, or selected on QA correctness.",
            "- Complete passages, not random rows, were held out.",
            "- Provo is historically inspected and supports development decisions only.",
            "- OneStop was not accessed and remains unavailable for repeated model selection.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("all", "extract", "evaluate"),
        default="all",
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--provo-path", type=Path, default=DEFAULT_PROVO_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backbones", nargs="*")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    provo_path = args.provo_path.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    protocol, all_specs = load_protocol(protocol_path)
    requested = set(args.backbones or [spec.key for spec in all_specs])
    unknown = requested.difference(spec.key for spec in all_specs)
    if unknown:
        raise ValueError(f"requested backbones are outside the allowlist: {sorted(unknown)}")
    specs = [spec for spec in all_specs if spec.key in requested]

    remote_identities: dict[str, dict[str, Any]] = {}
    if args.phase in {"all", "extract"}:
        remote_identities = verify_remote_model_identities(specs)

    items, dataset = load_label_free_provo_items(provo_path, protocol)
    features_by_backbone: dict[str, pd.DataFrame] = {}
    metadata_by_backbone: dict[str, Mapping[str, Any]] = {}
    if args.phase in {"all", "extract"}:
        inference_budget_seconds = float(
            protocol["compute"]["maximum_total_inference_gpu_hours"]
        ) * 3600
        cumulative_inference_seconds = 0.0
        for spec in specs:
            if cumulative_inference_seconds >= inference_budget_seconds:
                metadata_by_backbone[spec.key] = {
                    "schema_version": 1,
                    "status": "not_run_gpu_budget_reached",
                    "outcome_columns_read": False,
                }
                print(
                    f"[backbone:{spec.key}] not run: GPU time budget reached",
                    flush=True,
                )
                continue
            print(f"[backbone:{spec.key}] starting label-free extraction", flush=True)
            features, metadata = extract_or_load_backbone(
                items,
                dataset,
                protocol,
                protocol_path,
                spec,
                remote_identities[spec.key],
                cache_dir=cache_dir,
                force=args.force,
            )
            metadata_by_backbone[spec.key] = metadata
            if metadata.get("status") == "complete":
                cumulative_inference_seconds += float(
                    metadata["model"]["inference_seconds"]
                )
            if features is not None:
                features_by_backbone[spec.key] = features
            else:
                print(
                    f"[backbone:{spec.key}] failed: {metadata['error']}",
                    flush=True,
                )
        extraction_index = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "dataset": dataset,
            "protocol_sha256": _sha256(protocol_path),
            "backbones": metadata_by_backbone,
            "outcomes_read": False,
        }
        _write_json(cache_dir / "extraction-index.json", extraction_index)

    if args.phase == "evaluate":
        for spec in all_specs:
            feature_path, metadata_path, failure_path = _cache_paths(
                cache_dir, spec.key
            )
            if feature_path.exists() and metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("status") == "complete":
                    features_by_backbone[spec.key] = pd.read_csv(
                        feature_path,
                        dtype={"Text_ID": str, "IA_ID": int},
                    )
                    metadata_by_backbone[spec.key] = metadata
                    continue
            if failure_path.exists():
                metadata_by_backbone[spec.key] = json.loads(
                    failure_path.read_text(encoding="utf-8")
                )

    if args.phase == "extract":
        completed = len(features_by_backbone)
        print(f"[complete] extracted {completed}/{len(specs)} backbones", flush=True)
        return 0 if completed == len(specs) else 2

    summary, _ = evaluate(
        protocol,
        protocol_path,
        all_specs,
        features_by_backbone,
        metadata_by_backbone,
        provo_path=provo_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    _write_json(summary_path, summary)
    write_report(report_path, summary)
    manifest = {
        "schema_version": 1,
        "experiment": summary["experiment"],
        "created_at": summary["created_at"],
        "protocol_sha256": summary["protocol"]["sha256"],
        "dataset_sha256": summary["dataset"]["sha256"],
        "source": summary["source"],
        "artifacts": {
            path.name: {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (summary_path, report_path)
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(f"[complete] {summary['decision']['label']}", flush=True)
    print(f"[complete] shortlist={summary['decision']['shortlist']}", flush=True)
    print(f"[complete] report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

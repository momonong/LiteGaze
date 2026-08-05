"""Build the CPU-only, provenance-complete English text candidate.

PROVO and English GECO L2 are development corpora whose outcomes were already
inspected in earlier frozen experiments.  This builder therefore produces a
candidate artifact, not new confirmation evidence.  OneStop is never loaded;
it remains an immutable external audit corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.cognition.generalization import (
    cross_fit_grouped_ridge,
    fit_weighted_standardized_ridge,
    grouped_spearman_table,
    paired_bootstrap_mean_difference,
    predict_standardized_ridge,
)
from core.cognition.text_artifact import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERALIZATION_PATH = PROJECT_ROOT / "core/cognition/generalization.py"
RUNTIME_LOADER_PATH = PROJECT_ROOT / "core/cognition/text_artifact.py"
PROTOCOL_PATH = (
    PROJECT_ROOT
    / "docs/experiments/protocols/2026-08-06-production-text-fusion-v1.json"
)
DEFAULT_ARTIFACT_PATH = (
    PROJECT_ROOT / "core/cognition/artifacts/en_text_difficulty_m1_v1.json"
)
DEFAULT_MANIFEST_PATH = DEFAULT_ARTIFACT_PATH.with_suffix(".manifest.json")
DEFAULT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs/experiments/results/2026-08-06-production-text-artifact-run-001.json"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "docs/experiments/2026-08-06-production-text-artifact-run-001.md"
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
    "m0_lexical": M0_FEATURES,
    "m1_causal_surprisal": M1_FEATURES,
}
RANDOM_SEED = 20260805
N_FOLDS = 5
RIDGE_ALPHA = 1.0
BOOTSTRAP_SAMPLES = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["LEXIGAZE_COGNITION_DEVICE"] = "cpu"
    started = time.perf_counter()

    protocol = _read_json(PROTOCOL_PATH)
    _validate_protocol(protocol)
    frame, source_provenance = _load_development_frame()
    diagnostic, predictions = _development_diagnostic(frame)
    if not diagnostic["consistency_gate_passed"]:
        raise RuntimeError(
            "development consistency diagnostic failed; candidate was not written"
        )

    feature_matrix = frame.loc[:, list(M1_FEATURES)].to_numpy(dtype=np.float64)
    target = frame["log_total_reading_time"].to_numpy(dtype=np.float64)
    weights = frame["corpus_balanced_weight"].to_numpy(dtype=np.float64)
    model = fit_weighted_standardized_ridge(
        feature_matrix,
        target,
        weights,
        alpha=RIDGE_ALPHA,
    )
    raw_prediction = predict_standardized_ridge(model, feature_matrix)
    raw_lower = _weighted_quantile(raw_prediction, weights, 0.05)
    raw_upper = _weighted_quantile(raw_prediction, weights, 0.95)
    if raw_upper <= raw_lower:
        raise RuntimeError("candidate score calibration bounds collapsed")

    unique_items = frame.drop_duplicates(
        ["corpus", "development_group_id", "IA_ID"]
    )
    item_features = unique_items.loc[:, list(M1_FEATURES)].to_numpy(dtype=np.float64)
    feature_lower = np.quantile(item_features, 0.005, axis=0)
    feature_upper = np.quantile(item_features, 0.995, axis=0)
    created_at = datetime.now(UTC).isoformat()
    artifact = _make_artifact(
        created_at=created_at,
        frame=frame,
        source_provenance=source_provenance,
        diagnostic=diagnostic,
        model=model,
        raw_lower=raw_lower,
        raw_upper=raw_upper,
        feature_lower=feature_lower,
        feature_upper=feature_upper,
    )

    artifact_path = args.artifact_path.resolve()
    manifest_path = args.manifest_path.resolve()
    summary_path = args.summary_path.resolve()
    report_path = args.report_path.resolve()
    _write_json(artifact_path, artifact)
    manifest = {
        "schema_version": 1,
        "artifact_id": artifact["artifact_id"],
        "created_at": created_at,
        "artifact": {
            "file": artifact_path.name,
            "sha256": sha256_file(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
        },
        "protocol": {
            "path": _relative(PROTOCOL_PATH),
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "builder": {
            "path": _relative(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "implementation": {
            _relative(path): sha256_file(path)
            for path in (GENERALIZATION_PATH, RUNTIME_LOADER_PATH)
        },
    }
    _write_json(manifest_path, manifest)

    duration = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "experiment": "production_text_artifact_run_001",
        "created_at": created_at,
        "duration_seconds": duration,
        "protocol_id": protocol["protocol_id"],
        "claim_boundary": protocol["claim_boundary"],
        "training": artifact["training"],
        "feature_schema": list(M1_FEATURES),
        "development_diagnostic": diagnostic,
        "artifact": {
            "path": _relative(artifact_path),
            "sha256": manifest["artifact"]["sha256"],
            "manifest_path": _relative(manifest_path),
            "status": "candidate",
        },
        "score_calibration": artifact["score_calibration"],
        "leakage_controls": artifact["leakage_controls"],
        "compute": {
            "device": "cpu",
            "gpu_used": False,
            "language_model_inference_run": False,
            "verified_feature_caches_reused": True,
        },
        "source": _git_state(),
        "verification": {
            "cross_fitted_prediction_rows": len(predictions),
            "artifact_manifest_verified_after_write": (
                sha256_file(artifact_path) == manifest["artifact"]["sha256"]
            ),
        },
    }
    _write_json(summary_path, summary)
    _write_report(report_path, summary)

    print(f"[complete] candidate={artifact_path}")
    print(f"[complete] rows={len(frame):,} runtime={duration:.2f}s device=cpu")
    print("[complete] production default unchanged; independent fusion audit required")
    return 0


def _load_development_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    # Importing the frozen loaders also imports the language-model module, so it
    # happens only after CUDA has been hidden.  No model is instantiated here.
    from scripts import run_geco_l2_text_model_experiment as geco
    from scripts import run_provo_text_model_experiment as provo

    sources: list[tuple[str, pd.DataFrame, pd.DataFrame, Mapping[str, Any], str]] = []
    provo_raw, _, provo_fingerprint = provo.load_provo(
        provo.DEFAULT_PROVO_PATH.resolve()
    )
    provo_features, provo_metadata = _load_verified_feature_cache(
        provo.DEFAULT_CACHE_PATH,
        expected_source_sha256=provo_fingerprint["sha256"],
    )
    sources.append(
        (
            "PROVO",
            provo_raw,
            provo_features,
            {"dataset": provo_fingerprint, "feature_cache": provo_metadata},
            "IA_DWELL_TIME",
        )
    )

    geco_raw, _, geco_fingerprint = geco.load_geco_l2(
        geco.DEFAULT_GECO_L2_PATH.resolve()
    )
    geco_features, geco_metadata = _load_verified_feature_cache(
        geco.DEFAULT_CACHE_PATH,
        expected_source_sha256=geco_fingerprint["sha256"],
    )
    sources.append(
        (
            "GECO_L2_English",
            geco_raw,
            geco_features,
            {"dataset": geco_fingerprint, "feature_cache": geco_metadata},
            "WORD_TOTAL_READING_TIME",
        )
    )

    frames: list[pd.DataFrame] = []
    provenance: dict[str, Any] = {}
    for corpus, raw, features, details, outcome_column in sources:
        merged = raw.merge(
            features,
            on=["Text_ID", "IA_ID", "IA_LABEL"],
            how="left",
            validate="many_to_one",
        )
        outcome = pd.to_numeric(merged[outcome_column], errors="coerce")
        merged = merged.loc[
            merged["is_context_scored"].eq(1)
            & outcome.notna()
            & outcome.gt(0)
        ].copy()
        merged["log_total_reading_time"] = np.log1p(
            pd.to_numeric(merged[outcome_column], errors="raise").to_numpy(dtype=float)
        )
        if merged.loc[:, list(M1_FEATURES)].isna().any().any():
            raise RuntimeError(f"{corpus} contains missing candidate features")
        feature_values = merged.loc[:, list(M1_FEATURES)].to_numpy(dtype=np.float64)
        if not np.isfinite(feature_values).all():
            raise RuntimeError(f"{corpus} contains non-finite candidate features")
        merged["corpus"] = corpus
        merged["development_group_id"] = (
            corpus + "|" + merged["Text_ID"].astype(str)
        )
        merged["participant_group_id"] = (
            corpus + "|" + merged["Participant_ID"].astype(str)
        )
        frames.append(
            merged.loc[
                :,
                [
                    "corpus",
                    "Participant_ID",
                    "participant_group_id",
                    "Text_ID",
                    "development_group_id",
                    "IA_ID",
                    "log_total_reading_time",
                    *M1_FEATURES,
                ],
            ]
        )
        provenance[corpus] = details

    frame = pd.concat(frames, ignore_index=True)
    corpus_sizes = frame.groupby("corpus")["corpus"].transform("size")
    frame["corpus_balanced_weight"] = len(frame) / (
        frame["corpus"].nunique() * corpus_sizes.to_numpy(dtype=float)
    )
    return frame, provenance


def _load_verified_feature_cache(
    path: Path,
    *,
    expected_source_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache_path = path.resolve()
    metadata_path = cache_path.with_suffix(".metadata.json")
    if not cache_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"verified feature cache pair is required: {cache_path}"
        )
    metadata = _read_json(metadata_path)
    checks = {
        "source_sha256": metadata.get("source_sha256") == expected_source_sha256,
        "feature_sha256": metadata.get("feature_sha256") == sha256_file(cache_path),
        "question_answer_dataset_used": (
            metadata.get("question_answer_dataset_used") is False
        ),
        "outcomes_hidden_during_extraction": (
            metadata.get("outcome_columns_read_during_extraction") is False
        ),
        "model_name": metadata.get("model", {}).get("model_name") == "gpt2",
        "context_direction": (
            metadata.get("model", {})
            .get("metric_contract", {})
            .get("context_direction")
            == "left_only"
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"feature cache provenance failed: {checks}")
    columns = ["Text_ID", "IA_ID", "IA_LABEL", "is_context_scored", *M1_FEATURES]
    features = pd.read_csv(
        cache_path,
        usecols=columns,
        dtype={"Text_ID": str, "IA_ID": int},
    )
    if len(features) != int(metadata["item_count"]):
        raise RuntimeError("feature cache item count changed")
    return features, {
        "path": _relative(cache_path),
        "sha256": sha256_file(cache_path),
        "metadata_path": _relative(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "item_count": len(features),
        "model": metadata["model"],
        "code_hashes": metadata.get("code_hashes", {}),
        "question_answer_dataset_used": False,
        "outcome_columns_read_during_extraction": False,
    }


def _development_diagnostic(
    frame: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    predictions, diagnostics = cross_fit_grouped_ridge(
        frame,
        group_column="development_group_id",
        target_column="log_total_reading_time",
        feature_sets=FEATURE_SETS,
        n_folds=N_FOLDS,
        alpha=RIDGE_ALPHA,
        seed=RANDOM_SEED,
        shuffled_target_model="m1_causal_surprisal",
        sample_weight_column="corpus_balanced_weight",
    )
    prediction_columns = (
        "prediction_m0_lexical",
        "prediction_m1_causal_surprisal",
        "prediction_target_shuffle_sentinel",
    )
    participant_metrics = grouped_spearman_table(
        predictions,
        target_column="log_total_reading_time",
        prediction_columns=prediction_columns,
        grouping_columns=("corpus", "participant_group_id"),
    )
    item_predictions = (
        predictions.groupby(
            ["corpus", "development_group_id", "IA_ID"], as_index=False
        )
        .agg(
            log_total_reading_time=("log_total_reading_time", "mean"),
            prediction_m0_lexical=("prediction_m0_lexical", "first"),
            prediction_m1_causal_surprisal=(
                "prediction_m1_causal_surprisal",
                "first",
            ),
            prediction_target_shuffle_sentinel=(
                "prediction_target_shuffle_sentinel",
                "first",
            ),
        )
    )
    text_metrics = grouped_spearman_table(
        item_predictions,
        target_column="log_total_reading_time",
        prediction_columns=prediction_columns,
        grouping_columns=("corpus", "development_group_id"),
    )
    fold_metrics = grouped_spearman_table(
        predictions,
        target_column="log_total_reading_time",
        prediction_columns=prediction_columns,
        grouping_columns=("corpus", "outer_fold", "participant_group_id"),
    )

    corpora: dict[str, Any] = {}
    directions: list[bool] = []
    for corpus_index, corpus in enumerate(sorted(frame["corpus"].unique())):
        participant = participant_metrics.loc[participant_metrics["corpus"].eq(corpus)]
        text = text_metrics.loc[text_metrics["corpus"].eq(corpus)]
        fold = fold_metrics.loc[fold_metrics["corpus"].eq(corpus)]
        participant_comparison = _metric_comparison(
            participant,
            index_columns=("participant_group_id",),
            samples=BOOTSTRAP_SAMPLES,
            seed=RANDOM_SEED + 100 + corpus_index,
        )
        text_comparison = _metric_comparison(
            text,
            index_columns=("development_group_id",),
            samples=BOOTSTRAP_SAMPLES,
            seed=RANDOM_SEED + 200 + corpus_index,
        )
        fold_pivot = fold.pivot_table(
            index=["outer_fold", "participant_group_id"],
            columns="model",
            values="spearman_rho",
            aggfunc="first",
        )
        per_fold = (
            fold_pivot["m1_causal_surprisal"] - fold_pivot["m0_lexical"]
        ).groupby("outer_fold").mean()
        corpus_direction = (
            participant_comparison["mean_difference"] > 0
            and text_comparison["mean_difference"] > 0
            and int((per_fold > 0).sum()) >= 4
        )
        directions.append(corpus_direction)
        corpora[corpus] = {
            "rows": int(frame["corpus"].eq(corpus).sum()),
            "participants": int(
                frame.loc[
                    frame["corpus"].eq(corpus), "participant_group_id"
                ].nunique()
            ),
            "texts": int(
                frame.loc[
                    frame["corpus"].eq(corpus), "development_group_id"
                ].nunique()
            ),
            "participant_comparison": participant_comparison,
            "text_comparison": text_comparison,
            "positive_outer_folds": int((per_fold > 0).sum()),
            "outer_fold_count": len(per_fold),
            "direction_consistent": corpus_direction,
        }

    return (
        {
            "role": "implementation_consistency_only",
            "new_confirmation_claimed": False,
            "feature_sets": {
                name: list(features) for name, features in FEATURE_SETS.items()
            },
            "corpora": corpora,
            "cross_fit_diagnostics": diagnostics,
            "consistency_gate_passed": all(directions),
        },
        predictions,
    )


def _metric_comparison(
    table: pd.DataFrame,
    *,
    index_columns: Sequence[str],
    samples: int,
    seed: int,
) -> dict[str, float]:
    pivot = table.pivot_table(
        index=list(index_columns),
        columns="model",
        values="spearman_rho",
        aggfunc="first",
    )
    paired = pivot[["m1_causal_surprisal", "m0_lexical"]].dropna()
    return paired_bootstrap_mean_difference(
        paired["m1_causal_surprisal"].to_numpy(dtype=float),
        paired["m0_lexical"].to_numpy(dtype=float),
        samples=samples,
        seed=seed,
    )


def _make_artifact(
    *,
    created_at: str,
    frame: pd.DataFrame,
    source_provenance: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    model: Any,
    raw_lower: float,
    raw_upper: float,
    feature_lower: np.ndarray,
    feature_upper: np.ndarray,
) -> dict[str, Any]:
    corpus_rows = frame.groupby("corpus").size().to_dict()
    return {
        "schema_version": 1,
        "artifact_type": "standardized_ridge_text_difficulty",
        "artifact_id": "lexigaze-en-text-difficulty-m1-v1",
        "created_at": created_at,
        "language": "en",
        "feature_schema": list(M1_FEATURES),
        "feature_policy": {
            "language_model": "gpt2",
            "context_direction": "left_only",
            "surprisal_unit": "nats",
            "language_model_frozen": True,
            "language_model_fine_tuned": False,
            "excluded_features": [
                "shannon_entropy",
                "renyi_entropy",
                "prev_shannon_entropy",
                "prev_renyi_entropy",
            ],
        },
        "model": {
            "family": "standardized_ridge",
            "alpha": RIDGE_ALPHA,
            "target": "log1p_total_reading_time_ms",
            "feature_mean": model.mean.tolist(),
            "feature_scale": model.scale.tolist(),
            "coefficients": model.coefficients[1:].tolist(),
            "intercept": float(model.coefficients[0]),
        },
        "score_calibration": {
            "method": "fixed_training_prediction_quantile_clip",
            "lower_quantile": 0.05,
            "upper_quantile": 0.95,
            "raw_lower": float(raw_lower),
            "raw_upper": float(raw_upper),
            "request_local_normalization": False,
        },
        "training_distribution": {
            "feature_lower_quantile": 0.005,
            "feature_upper_quantile": 0.995,
            "feature_lower": feature_lower.tolist(),
            "feature_upper": feature_upper.tolist(),
            "out_of_distribution_behavior": (
                "retain_score_and_set_in_distribution_false"
            ),
        },
        "training": {
            "corpora": ["PROVO", "GECO_L2_English"],
            "excluded_corpora": ["OneStop_Eye_Movements"],
            "row_count": len(frame),
            "rows_by_corpus": {key: int(value) for key, value in corpus_rows.items()},
            "corpus_weighting": "equal_total_weight_per_corpus",
            "outcome": "positive_total_reading_time_ms",
            "outcome_transform": "log1p",
        },
        "provenance": {
            "protocol_path": _relative(PROTOCOL_PATH),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "builder_path": _relative(Path(__file__).resolve()),
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            "implementation_sha256": {
                _relative(path): sha256_file(path)
                for path in (GENERALIZATION_PATH, RUNTIME_LOADER_PATH)
            },
            "sources": dict(source_provenance),
            "development_diagnostic": dict(diagnostic),
        },
        "leakage_controls": {
            "question_answer_dataset_used": False,
            "onestop_used_for_training_selection_or_thresholding": False,
            "language_model_fine_tuned": False,
            "request_local_normalization_used": False,
            "verified_label_free_feature_caches_reused": True,
            "complete_text_cross_fit_diagnostic": True,
        },
        "promotion": {
            "status": "candidate",
            "production_default_changed": False,
            "independent_fusion_evaluation_required": True,
            "protocol_id": "production-text-fusion-v1",
        },
        "evidence": {
            "development_generalization": (
                "docs/experiments/2026-08-05-text-model-generalization-run-001.md"
            ),
            "immutable_external_confirmation": (
                "docs/experiments/2026-08-05-onestop-confirmation-run-001.md"
            ),
        },
        "compute": {
            "device": "cpu",
            "gpu_used": False,
            "language_model_inference_run": False,
        },
    }


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be within [0, 1]")
    order = np.argsort(values)
    ordered_values = np.asarray(values, dtype=np.float64)[order]
    ordered_weights = np.asarray(weights, dtype=np.float64)[order]
    midpoint = np.cumsum(ordered_weights) - 0.5 * ordered_weights
    midpoint /= ordered_weights.sum()
    return float(np.interp(quantile, midpoint, ordered_values))


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("protocol_id") != "production-text-fusion-v1":
        raise RuntimeError("unexpected production text/fusion protocol")
    artifact = protocol.get("text_artifact", {})
    if tuple(artifact.get("features", ())) != M1_FEATURES:
        raise RuntimeError("builder features diverged from the frozen protocol")
    compute = protocol.get("compute", {})
    if compute.get("device") != "cpu" or compute.get("gpu_allowed") is not False:
        raise RuntimeError("frozen protocol must remain CPU-only")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    diagnostic = summary["development_diagnostic"]
    rows = []
    for corpus, result in diagnostic["corpora"].items():
        participant = result["participant_comparison"]
        text = result["text_comparison"]
        rows.append(
            "| "
            + " | ".join(
                [
                    corpus,
                    str(result["rows"]),
                    f"{participant['mean_difference']:+.4f}",
                    f"[{participant['ci_95_low']:+.4f}, {participant['ci_95_high']:+.4f}]",
                    f"{text['mean_difference']:+.4f}",
                    f"[{text['ci_95_low']:+.4f}, {text['ci_95_high']:+.4f}]",
                    f"{result['positive_outer_folds']}/{result['outer_fold_count']}",
                ]
            )
            + " |"
        )
    lines = [
        "# Production Text Artifact - Run 001",
        "",
        f"- Date: {summary['created_at']}",
        "- Protocol: `production-text-fusion-v1`",
        "- Compute: CPU only; verified GPT-2 feature caches reused",
        "- Status: candidate; the production default was not changed",
        "",
        "## Outcome",
        "",
        (
            "A provenance-complete M1 text artifact was built from PROVO and English "
            "GECO L2 with equal total corpus weight. OneStop and all QA outcomes were "
            "excluded. This pooled development fit is an implementation diagnostic, "
            "not a new confirmation experiment."
        ),
        "",
        "| Corpus | Rows | Participant M1-M0 | 95% CI | Text M1-M0 | 95% CI | Positive folds |",
        "| --- | ---: | ---: | --- | ---: | --- | ---: |",
        *rows,
        "",
        "## Decision",
        "",
        (
            "The candidate remains isolated from the product default. Promotion "
            "requires a future real-data evaluation where combined fusion beats "
            "gaze-only on both capture-held-out and article-held-out groups using an "
            "independent difficulty target that is not derived from gaze, text "
            "features, or a public QA benchmark."
        ),
        "",
        "## Leakage and compute controls",
        "",
        "- OneStop was not loaded, trained on, or used for threshold selection.",
        "- Entropy features were excluded.",
        "- No language-model fine-tuning or inference was run.",
        "- No request-local normalization is present in the artifact.",
        "- GPU use was disabled with `CUDA_VISIBLE_DEVICES=-1`.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _git_state() -> dict[str, Any]:
    def output(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "commit": output("rev-parse", "HEAD"),
        "branch": output("branch", "--show-current"),
        "dirty": bool(output("status", "--short")),
    }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())

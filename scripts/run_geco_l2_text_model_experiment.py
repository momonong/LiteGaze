"""Replicate the frozen GPT-2 text-model protocol on English GECO L2.

The feature sets, Ridge configuration, folds, and decision gate are imported
from the already-frozen PROVO protocol.  This script reads only raw GECO word
identities and reading-time outcomes; legacy prepared surprisal, attention, and
cognitive-mass fields are deliberately excluded.
"""

from __future__ import annotations

import argparse
import json
import time
import unicodedata
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from scripts import run_provo_text_model_experiment as protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GECO_L2_PATH = PROJECT_ROOT / "data/geco/L2ReadingData.csv"
DEFAULT_CACHE_PATH = (
    PROJECT_ROOT
    / "data/geco/text_modeling/geco-l2-gpt2-small-causal-features.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data/geco/text_modeling/geco-l2-gpt2-small-run-001"
)
EXPECTED_GECO_L2_SHA256 = (
    "cd551640cfd122b5e360d70c12998798125e7485c6deb900bc71b8e591c59b87"
)
EXPECTED_IDENTITY = {
    "row_count": 534_154,
    "participant_count": 19,
    "text_count": 588,
    "item_count": 56_411,
}

SOURCE_COLUMNS = (
    "PP_NR",
    "GROUP",
    "LANGUAGE",
    "PART",
    "TRIAL",
    "WORD_ID_WITHIN_TRIAL",
    "WORD_ID",
    "WORD",
    "WORD_TOTAL_READING_TIME",
    "WORD_GAZE_DURATION",
    "WORD_FIRST_FIXATION_DURATION",
)
OUTCOMES = {
    "total_reading_time": "WORD_TOTAL_READING_TIME",
    "gaze_duration": "WORD_GAZE_DURATION",
    "first_fixation_duration": "WORD_FIRST_FIXATION_DURATION",
}


def _normalize_word(value: object) -> str:
    """Remove GECO display padding without changing internal punctuation."""
    return unicodedata.normalize("NFKC", str(value)).strip()


def _prepare_geco_l2(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and map GECO identities to the frozen benchmark schema."""
    missing = set(SOURCE_COLUMNS).difference(source.columns)
    if missing:
        raise ValueError(f"GECO L2 source is missing columns: {sorted(missing)}")

    raw = source.copy()
    identity_columns = (
        "PP_NR",
        "GROUP",
        "LANGUAGE",
        "PART",
        "TRIAL",
        "WORD_ID_WITHIN_TRIAL",
        "WORD_ID",
        "WORD",
    )
    if raw[list(identity_columns)].eq("").any().any():
        raise ValueError("GECO L2 identity/display fields must be complete")
    if set(raw["GROUP"].unique()) != {"bilingual"}:
        raise ValueError("GECO L2 source must contain only bilingual readers")
    if set(raw["LANGUAGE"].unique()) != {"English"}:
        raise ValueError("GECO L2 source must contain only English trials")

    raw["IA_LABEL"] = raw["WORD"].map(_normalize_word)
    if raw["IA_LABEL"].eq("").any():
        raise ValueError("GECO L2 contains an empty normalized word label")
    raw["Participant_ID"] = raw["PP_NR"].astype(str)
    raw["Text_ID"] = (
        raw["PART"].astype(str) + ":" + raw["TRIAL"].astype(str)
    )
    raw["IA_ID"] = pd.to_numeric(
        raw["WORD_ID_WITHIN_TRIAL"], errors="raise"
    ).astype(int)
    identity = ["Participant_ID", "Text_ID", "IA_ID"]
    if raw.duplicated(identity).any():
        raise ValueError("duplicate participant/text/item rows in GECO L2 source")

    items = raw[["Text_ID", "IA_ID", "IA_LABEL"]].drop_duplicates()
    inconsistent = (
        items.groupby(["Text_ID", "IA_ID"])["IA_LABEL"].nunique().gt(1)
    )
    if inconsistent.any():
        raise ValueError("a GECO L2 display item has inconsistent labels")
    items = items.sort_values(["Text_ID", "IA_ID"]).reset_index(drop=True)
    return raw, items


def load_geco_l2(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load and fingerprint the frozen English GECO L2 source."""
    source_hash = protocol._sha256(path)
    if source_hash != EXPECTED_GECO_L2_SHA256:
        raise RuntimeError(
            "GECO L2 source hash changed; freeze a new protocol before outcomes"
        )
    source = pd.read_csv(
        path,
        usecols=list(SOURCE_COLUMNS),
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    raw, items = _prepare_geco_l2(source)
    fingerprint = {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": source_hash,
        "size_bytes": path.stat().st_size,
        "row_count": len(raw),
        "participant_count": raw["Participant_ID"].nunique(),
        "text_count": raw["Text_ID"].nunique(),
        "item_count": len(items),
        "reader_group": "bilingual",
        "stimulus_language": "English",
    }
    observed_identity = {
        key: fingerprint[key] for key in EXPECTED_IDENTITY
    }
    if observed_identity != EXPECTED_IDENTITY:
        raise RuntimeError(f"unexpected GECO L2 identity: {fingerprint}")
    return raw, items, fingerprint


def load_or_extract_features(
    items: pd.DataFrame,
    fingerprint: Mapping[str, Any],
    *,
    cache_path: Path,
    device: str,
    force: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Cache label-free features with source and implementation provenance."""
    metadata_path = cache_path.with_suffix(".metadata.json")
    code_hashes = {
        "geco_loader_sha256": protocol._sha256(Path(__file__)),
        "frozen_protocol_sha256": protocol._sha256(Path(protocol.__file__)),
        "pipeline_sha256": protocol._sha256(
            PROJECT_ROOT / "core/cognition/pipeline.py"
        ),
    }
    if cache_path.exists() and metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        reusable = (
            metadata.get("source_sha256") == fingerprint["sha256"]
            and metadata.get("item_count") == fingerprint["item_count"]
            and metadata.get("code_hashes") == code_hashes
            and metadata.get("model", {}).get("model_name")
            == protocol.MODEL_NAME
        )
        if reusable:
            print(f"[features] reusing cache {cache_path}")
            return (
                pd.read_csv(cache_path, dtype={"Text_ID": str, "IA_ID": int}),
                metadata,
            )

    features, model = protocol.extract_features(items, device=device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(cache_path, index=False, lineterminator="\n")
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_sha256": fingerprint["sha256"],
        "item_count": len(features),
        "feature_sha256": protocol._sha256(cache_path),
        "code_hashes": code_hashes,
        "model": model,
        "question_answer_dataset_used": False,
        "legacy_precomputed_text_features_used": False,
        "outcome_columns_read_during_extraction": False,
    }
    protocol._write_json(metadata_path, metadata)
    return features, metadata


def make_decision(outcomes: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the gate frozen before GECO L2 outcomes were inspected."""
    primary = outcomes["total_reading_time"]
    m1 = protocol._comparison(
        primary, "m1_causal_surprisal_minus_m0_lexical"
    )
    m2 = protocol._comparison(primary, "m2_entropy_minus_m1_causal_surprisal")
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
        label = "causal_surprisal_and_entropy_replicate_on_geco_l2"
    elif m1_pass:
        label = "causal_surprisal_replicates_on_geco_l2"
    else:
        label = "causal_surprisal_does_not_replicate_on_geco_l2"
    return {
        "label": label,
        "m1_causal_surprisal_gate_passed": m1_pass,
        "m2_entropy_gate_passed": m2_pass,
        "protocol_frozen_before_geco_l2_outcomes": True,
        "final_confirmation_required": True,
        "geco_role": "external_replication_but_historically_inspected_corpus",
    }


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    """Write a compact human-auditable result report."""
    lines = [
        "# GECO L2 GPT-2 Small Text-Model Replication - Run 001",
        "",
        f"- Completed: {summary['created_at']}",
        f"- Runtime: {summary['duration_seconds']:.2f} seconds",
        f"- Device: `{summary['compute']['device']}` "
        f"(GPU used: {str(summary['compute']['gpu_used']).lower()})",
        f"- Decision: **`{summary['decision']['label']}`**",
        "- Role: frozen cross-corpus replication; GECO has historical project exposure and is not a pristine final corpus",
        "",
        "## Design",
        "",
        "- 588 complete English passages assigned to five deterministic outer folds.",
        "- The feature sets, Ridge alpha, folds, and decision gate were frozen on PROVO before GECO L2 outcomes were inspected.",
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
                f"Rows: {outcome['row_count']:,}; participants: "
                f"{outcome['participant_count']}; texts: {outcome['text_count']}.",
                "",
                "| Model | Macro participant rho | Macro text rho | Log-duration MAE |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for model in outcome["models"]:
            lines.append(
                f"| `{model['model']}` | "
                f"{model['macro_participant_spearman']:.4f} | "
                f"{model['macro_text_spearman']:.4f} | "
                f"{model['mean_absolute_error_log']:.4f} |"
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
                f"| `{comparison['comparison']}` | "
                f"{participant['mean_difference']:+.4f} "
                f"[{participant['ci_95_low']:+.4f}, "
                f"{participant['ci_95_high']:+.4f}] | "
                f"{text['mean_difference']:+.4f} "
                f"[{text['ci_95_low']:+.4f}, {text['ci_95_high']:+.4f}] | "
                f"{comparison['positive_outer_folds']}/"
                f"{comparison['outer_fold_count']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Guardrails",
            "",
            "- The raw source was verified as bilingual-reader English GECO L2 only.",
            "- No prepared surprisal, attention, or cognitive-mass field was read.",
            "- No question-answer dataset or gaze coordinate was used.",
            "- No reading-time outcome was used during feature extraction.",
            "- This historically inspected corpus is external replication evidence, not final untouched confirmation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geco-path", type=Path, default=DEFAULT_GECO_L2_PATH)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--bootstrap-samples", type=int, default=protocol.BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--force-feature-extraction", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    started = time.perf_counter()
    raw, items, fingerprint = load_geco_l2(args.geco_path.resolve())
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
        outcome, participant, text, fold, predictions = protocol.evaluate_outcome(
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
        "experiment": "geco_l2_gpt2_small_text_replication_run_001",
        "created_at": datetime.now(UTC).isoformat(),
        "duration_seconds": duration,
        "configuration": {
            "seed": protocol.RANDOM_SEED,
            "outer_folds": protocol.N_FOLDS,
            "ridge_alpha": protocol.RIDGE_ALPHA,
            "bootstrap_samples": args.bootstrap_samples,
            "feature_sets": {
                key: list(value) for key, value in protocol.FEATURE_SETS.items()
            },
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
            "legacy_precomputed_text_features_used": False,
            "gaze_feature_used_as_predictor": False,
            "complete_text_holdout": True,
            "scaler_fit_on_training_only": True,
            "target_shuffle_sentinel_included": True,
            "hyperparameter_search_used": False,
            "protocol_frozen_before_geco_l2_outcomes": True,
        },
        "source": protocol._git_state(),
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

    protocol._write_json(summary_path, summary)
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
            artifact.name: {
                "sha256": protocol._sha256(artifact),
                "size_bytes": artifact.stat().st_size,
            }
            for artifact in (
                summary_path,
                report_path,
                participant_path,
                text_path,
                fold_path,
                predictions_path,
            )
        },
    }
    protocol._write_json(manifest_path, manifest)

    print(f"[complete] {summary['decision']['label']}")
    print(f"[complete] runtime={duration:.2f}s device={args.device}")
    print(f"[complete] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

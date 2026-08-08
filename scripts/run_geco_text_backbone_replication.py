"""Run the frozen GPT-2 versus Pythia-410M GECO L2 replication."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from core.cognition.causal_backbone import BackboneSpec, model_id_has_excluded_prefix
from scripts import run_geco_l2_text_model_experiment as legacy_geco
from scripts import run_text_backbone_benchmark as benchmark


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "docs/experiments/protocols/2026-08-08-pythia-geco-l2-replication-v1.json"
)
DEFAULT_GECO_PATH = PROJECT_ROOT / "data/geco/L2ReadingData.csv"
DEFAULT_CACHE_DIR = (
    PROJECT_ROOT / "data/geco/text_modeling/pythia-geco-l2-replication-v1"
)
DEFAULT_OUTPUT_DIR = DEFAULT_CACHE_DIR / "run-001"
IDENTITY_SOURCE_COLUMNS = (
    "PP_NR",
    "GROUP",
    "LANGUAGE",
    "PART",
    "TRIAL",
    "WORD_ID_WITHIN_TRIAL",
    "WORD_ID",
    "WORD",
)
OUTCOME_COLUMNS = {
    "total_reading_time": "WORD_TOTAL_READING_TIME",
    "gaze_duration": "WORD_GAZE_DURATION",
    "first_fixation_duration": "WORD_FIRST_FIXATION_DURATION",
}


def load_protocol(path: Path) -> tuple[dict[str, Any], list[BackboneSpec]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != "pythia-full-geco-l2-replication-v1":
        raise ValueError("unexpected GECO replication protocol")
    policy = payload["source_policy"]
    if policy.get("mode") != "exact_allowlist":
        raise ValueError("GECO model policy must use an exact allowlist")
    if policy.get("trust_remote_code") is not False:
        raise ValueError("trust_remote_code must remain false")
    excluded = tuple(policy["excluded_model_id_prefixes"])
    specs: list[BackboneSpec] = []
    for item in policy["allowlist"]:
        spec = BackboneSpec(
            key=str(item["key"]),
            model_id=str(item["model_id"]),
            revision=str(item["revision"]),
            developer=str(item["developer"]),
            license=str(item["license"]),
            role=str(item["role"]),
        )
        if not re.fullmatch(r"[0-9a-f]{40}", spec.revision):
            raise ValueError(f"mutable GECO model revision: {spec.key}")
        if model_id_has_excluded_prefix(spec.model_id, excluded):
            raise ValueError(f"excluded model source in GECO allowlist: {spec.model_id}")
        specs.append(spec)
    if [spec.key for spec in specs] != ["gpt2", "pythia_410m_deduped_full"]:
        raise ValueError("GECO v1 must contain exactly GPT-2 and full Pythia")
    return payload, specs


def load_label_free_geco_items(
    path: Path,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read GECO identities only and standardize them without durations."""
    expected = protocol["dataset"]
    source_hash = benchmark._sha256(path)
    if source_hash != expected["sha256"]:
        raise RuntimeError("GECO source hash does not match the frozen protocol")
    raw = pd.read_csv(
        path,
        usecols=list(IDENTITY_SOURCE_COLUMNS),
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    if raw[list(IDENTITY_SOURCE_COLUMNS)].eq("").any().any():
        raise ValueError("GECO identity/display fields must be complete")
    if set(raw["GROUP"].unique()) != {expected["reader_group"]}:
        raise ValueError("GECO source is not the frozen bilingual-reader subset")
    if set(raw["LANGUAGE"].unique()) != {expected["stimulus_language"]}:
        raise ValueError("GECO source is not the frozen English subset")
    raw["IA_LABEL"] = raw["WORD"].map(legacy_geco._normalize_word)
    if raw["IA_LABEL"].eq("").any():
        raise ValueError("GECO contains an empty normalized label")
    if raw["IA_LABEL"].str.contains("\ufffd", regex=False).any():
        raise ValueError("GECO contains a Unicode replacement character")
    raw["Participant_ID"] = raw["PP_NR"].astype(str)
    raw["Text_ID"] = raw["PART"].astype(str) + ":" + raw["TRIAL"].astype(str)
    raw["IA_ID"] = pd.to_numeric(
        raw["WORD_ID_WITHIN_TRIAL"], errors="raise"
    ).astype(int)
    if raw.duplicated(["Participant_ID", "Text_ID", "IA_ID"]).any():
        raise ValueError("duplicate participant/text/item row in GECO")
    items = raw[["Text_ID", "IA_ID", "IA_LABEL"]].drop_duplicates()
    if items.groupby(["Text_ID", "IA_ID"])["IA_LABEL"].nunique().gt(1).any():
        raise ValueError("inconsistent GECO display label")
    items = items.sort_values(["Text_ID", "IA_ID"]).reset_index(drop=True)
    observed = {
        "source_sha256": source_hash,
        "row_count": len(raw),
        "participant_count": raw["Participant_ID"].nunique(),
        "text_count": items["Text_ID"].nunique(),
        "item_count": len(items),
        "reader_group": expected["reader_group"],
        "stimulus_language": expected["stimulus_language"],
        "outcome_columns_read": False,
    }
    frozen_identity = {
        "row_count": expected["expected_rows"],
        "participant_count": expected["expected_participants"],
        "text_count": expected["expected_texts"],
        "item_count": expected["expected_items"],
    }
    if {key: observed[key] for key in frozen_identity} != frozen_identity:
        raise RuntimeError(f"unexpected GECO identity: {observed}")
    return items, observed


def extract_or_load(
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
    """Reuse the frozen generic extractor and bind it to this GECO loader."""
    _, metadata_path, _ = benchmark._cache_paths(cache_dir, spec.key)
    runner_hash = benchmark._sha256(Path(__file__))
    local_force = force
    if metadata_path.exists() and not force:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        local_force = existing.get("geco_runner_sha256") != runner_hash
    features, metadata = benchmark.extract_or_load_backbone(
        items,
        dataset,
        protocol,
        protocol_path,
        spec,
        remote_identity,
        cache_dir=cache_dir,
        force=local_force,
    )
    if metadata.get("status") == "complete":
        metadata = {**metadata, "geco_runner_sha256": runner_hash}
        benchmark._write_json(metadata_path, metadata)
    return features, metadata


def make_decision(
    challenger_outcome: Mapping[str, Any],
    comparison: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    gate = protocol["decision_gate"]
    increment_pass = bool(challenger_outcome["incremental_gate_passed"])
    participant = comparison["participant_bootstrap"]
    text = comparison["text_bootstrap"]
    positive_folds = int(comparison["positive_outer_folds"])
    strong_spec = gate["strong_replication_vs_gpt2"]
    directional_spec = gate["directional_replication_vs_gpt2"]
    strong = (
        increment_pass
        and participant["ci_95_low"]
        > float(strong_spec["participant_bootstrap_ci_95_low_greater_than"])
        and text["ci_95_low"]
        > float(strong_spec["text_bootstrap_ci_95_low_greater_than"])
        and positive_folds >= int(strong_spec["minimum_positive_outer_folds"])
    )
    directional = (
        increment_pass
        and participant["mean_difference"]
        > float(
            directional_spec["participant_point_difference_greater_than"]
        )
        and text["mean_difference"]
        > float(directional_spec["text_point_difference_greater_than"])
        and positive_folds
        >= int(directional_spec["minimum_positive_outer_folds"])
    )
    if strong:
        label = "strong_replication_over_gpt2"
    elif directional:
        label = "directional_replication_over_gpt2"
    else:
        label = "does_not_replicate_over_gpt2"
    return {
        "label": label,
        "challenger_incremental_gate_passed": increment_pass,
        "strong_replication_gate_passed": strong,
        "directional_replication_gate_passed": directional,
        "production_model_changed": False,
        "one_stop_retest_allowed": False,
        "independent_word_review_confirmation_required": True,
    }


def evaluate(
    protocol: Mapping[str, Any],
    protocol_path: Path,
    specs: Sequence[BackboneSpec],
    features_by_backbone: Mapping[str, pd.DataFrame],
    metadata_by_backbone: Mapping[str, Mapping[str, Any]],
    *,
    geco_path: Path,
) -> dict[str, Any]:
    raw, _, fingerprint = legacy_geco.load_geco_l2(geco_path)
    if fingerprint["sha256"] != protocol["dataset"]["sha256"]:
        raise RuntimeError("GECO source changed after feature extraction")
    outcomes_by_backbone: dict[str, Any] = {}
    metric_tables: dict[str, dict[str, Mapping[str, pd.DataFrame]]] = {}
    for backbone_index, spec in enumerate(specs):
        if spec.key not in features_by_backbone:
            continue
        print(f"[evaluation:{spec.key}] starting GECO outcomes", flush=True)
        outcomes_by_backbone[spec.key] = {}
        metric_tables[spec.key] = {}
        for outcome_index, (outcome_name, source_column) in enumerate(
            OUTCOME_COLUMNS.items()
        ):
            summary, tables = benchmark.evaluate_backbone_outcome(
                raw,
                features_by_backbone[spec.key],
                protocol,
                outcome_name=outcome_name,
                source_column=source_column,
                seed_offset=1_000 * backbone_index + 100 * outcome_index,
            )
            outcomes_by_backbone[spec.key][outcome_name] = summary
            metric_tables[spec.key][outcome_name] = tables

    required = {"gpt2", "pythia_410m_deduped_full"}
    if set(outcomes_by_backbone) != required:
        raise RuntimeError("both frozen GECO backbones must be evaluable")
    primary_name = protocol["analysis"]["primary_outcome"]["name"]
    primary_tables = {
        key: tables[primary_name] for key, tables in metric_tables.items()
    }
    comparison = benchmark._cross_backbone_comparison(
        primary_tables,
        first="pythia_410m_deduped_full",
        second="gpt2",
        bootstrap_samples=int(protocol["analysis"]["bootstrap_samples"]),
        seed=int(protocol["analysis"]["seed"]) + 50_000,
    )
    decision = make_decision(
        outcomes_by_backbone["pythia_410m_deduped_full"][primary_name],
        comparison,
        protocol,
    )
    total_inference_seconds = sum(
        float(metadata["model"]["inference_seconds"])
        for metadata in metadata_by_backbone.values()
        if metadata.get("status") == "complete"
    )
    return {
        "schema_version": 1,
        "experiment": "pythia_full_geco_l2_replication_run_001",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "path": protocol_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": benchmark._sha256(protocol_path),
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
            "comparison": comparison,
        },
        "decision": decision,
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
            "throughput_is_selection_metric": False,
        },
        "leakage_controls": {
            "features_extracted_before_outcomes_loaded": True,
            "question_answer_dataset_used": False,
            "legacy_precomputed_text_features_used": False,
            "complete_text_holdout": True,
            "scaler_fit_on_training_only": True,
            "target_shuffle_sentinel_included": True,
            "language_model_fine_tuned": False,
            "candidate_search_reopened": False,
            "onestop_accessed": False,
        },
        "source": benchmark._git_state(),
    }


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    primary_name = summary["cross_backbone_primary"]["outcome"]
    comparison = summary["cross_backbone_primary"]["comparison"]
    lines = [
        "# Pythia-410M Full GECO L2 Replication — Run 001",
        "",
        f"- Completed: {summary['created_at']}",
        f"- Decision: **`{summary['decision']['label']}`**",
        "- Product model changed: false",
        "- QA data used: false",
        "",
        "## Primary total-reading-time results",
        "",
        "| Backbone | M1 participant rho | M1 text rho | M1-M0 participant delta [95% CI] | M1-M0 text delta [95% CI] | Positive folds |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, outcomes in summary["outcomes"].items():
        outcome = outcomes[primary_name]
        m1 = benchmark._find_model(outcome, "m1_causal_surprisal")
        increment = benchmark._find_comparison(
            outcome, "m1_causal_surprisal_minus_m0_tokenization"
        )
        participant = increment["participant_bootstrap"]
        text = increment["text_bootstrap"]
        lines.append(
            f"| `{key}` | {m1['macro_participant_spearman']:.4f} | "
            f"{m1['macro_text_spearman']:.4f} | "
            f"{participant['mean_difference']:+.4f} "
            f"[{participant['ci_95_low']:+.4f}, {participant['ci_95_high']:+.4f}] | "
            f"{text['mean_difference']:+.4f} "
            f"[{text['ci_95_low']:+.4f}, {text['ci_95_high']:+.4f}] | "
            f"{increment['positive_outer_folds']}/{increment['outer_fold_count']} |"
        )
    participant = comparison["participant_bootstrap"]
    text = comparison["text_bootstrap"]
    lines.extend(
        [
            "",
            "## Frozen Pythia-minus-GPT-2 contrast",
            "",
            f"- Participant delta: {participant['mean_difference']:+.4f} "
            f"[{participant['ci_95_low']:+.4f}, {participant['ci_95_high']:+.4f}]",
            f"- Text delta: {text['mean_difference']:+.4f} "
            f"[{text['ci_95_low']:+.4f}, {text['ci_95_high']:+.4f}]",
            f"- Positive outer folds: {comparison['positive_outer_folds']}/"
            f"{comparison['outer_fold_count']}",
            "",
            "## Guardrails",
            "",
            "- Only the sole Provo-shortlisted challenger and GPT-2 were evaluated.",
            "- GECO identities and text were extracted before duration columns were opened.",
            "- Prepared corpus surprisal, attention, and cognitive-mass fields were not used.",
            "- Complete texts were held out and all scaling was training-fold only.",
            "- No model training, fine-tuning, prompting, QA data, or OneStop access occurred.",
            "- GECO is replication evidence, not independent personalized confirmation.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("all", "extract", "evaluate"), default="all"
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--geco-path", type=Path, default=DEFAULT_GECO_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    geco_path = args.geco_path.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    protocol, specs = load_protocol(protocol_path)
    items, dataset = load_label_free_geco_items(geco_path, protocol)
    features_by_backbone: dict[str, pd.DataFrame] = {}
    metadata_by_backbone: dict[str, Mapping[str, Any]] = {}

    if args.phase in {"all", "extract"}:
        remote = benchmark.verify_remote_model_identities(specs)
        budget_seconds = float(
            protocol["compute"]["maximum_total_inference_gpu_hours"]
        ) * 3600
        cumulative_inference = 0.0
        for spec in specs:
            if cumulative_inference >= budget_seconds:
                raise RuntimeError("GECO inference GPU budget reached")
            print(f"[backbone:{spec.key}] starting label-free GECO extraction", flush=True)
            features, metadata = extract_or_load(
                items,
                dataset,
                protocol,
                protocol_path,
                spec,
                remote[spec.key],
                cache_dir=cache_dir,
                force=args.force,
            )
            metadata_by_backbone[spec.key] = metadata
            if features is None:
                print(f"[backbone:{spec.key}] failed: {metadata['error']}", flush=True)
                continue
            features_by_backbone[spec.key] = features
            cumulative_inference += float(metadata["model"]["inference_seconds"])
        benchmark._write_json(
            cache_dir / "extraction-index.json",
            {
                "schema_version": 1,
                "created_at": datetime.now(UTC).isoformat(),
                "dataset": dataset,
                "protocol_sha256": benchmark._sha256(protocol_path),
                "backbones": metadata_by_backbone,
                "outcomes_read": False,
            },
        )

    if args.phase == "evaluate":
        for spec in specs:
            feature_path, metadata_path, _ = benchmark._cache_paths(
                cache_dir, spec.key
            )
            if not feature_path.exists() or not metadata_path.exists():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("status") != "complete":
                continue
            features_by_backbone[spec.key] = pd.read_csv(
                feature_path, dtype={"Text_ID": str, "IA_ID": int}
            )
            metadata_by_backbone[spec.key] = metadata

    if args.phase == "extract":
        print(
            f"[complete] extracted {len(features_by_backbone)}/{len(specs)} backbones",
            flush=True,
        )
        return 0 if len(features_by_backbone) == len(specs) else 2

    summary = evaluate(
        protocol,
        protocol_path,
        specs,
        features_by_backbone,
        metadata_by_backbone,
        geco_path=geco_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    benchmark._write_json(summary_path, summary)
    write_report(report_path, summary)
    benchmark._write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "experiment": summary["experiment"],
            "created_at": summary["created_at"],
            "protocol_sha256": summary["protocol"]["sha256"],
            "dataset_sha256": summary["dataset"]["sha256"],
            "source": summary["source"],
            "artifacts": {
                path.name: {
                    "sha256": benchmark._sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in (summary_path, report_path)
            },
        },
    )
    print(f"[complete] {summary['decision']['label']}", flush=True)
    print(f"[complete] report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

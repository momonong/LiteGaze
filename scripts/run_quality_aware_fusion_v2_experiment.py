"""Run the frozen CPU-only quality-aware fusion v2 corruption benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.cognition.quality_fusion import (  # noqa: E402
    PROTOCOL_ID,
    QualityAwareFusionConfig,
    fuse_quality_aware,
)
from scripts.fusion.orchestrator import (  # noqa: E402
    aggregate_gaze_events,
    align_gaze_occurrences,
)

DEFAULT_PROTOCOL = (
    ROOT
    / "docs"
    / "experiments"
    / "protocols"
    / "2026-08-07-quality-aware-text-fusion-v2.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "experiments" / "results"
IMPLEMENTATION_PATHS = (
    ROOT / "core" / "cognition" / "quality_fusion.py",
    ROOT / "scripts" / "fusion" / "orchestrator.py",
    ROOT / "web" / "static" / "gaze_integration.js",
    ROOT / "web" / "static" / "mapping.js",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default="run-001")
    return parser.parse_args()


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected experiment protocol")
    if protocol.get("status") != "frozen_before_benchmark":
        raise ValueError("experiment protocol is not frozen")
    if protocol["compute"] != {
        "device": "cpu",
        "gpu_allowed": False,
        "torch_allowed": False,
        "network_allowed": False,
    }:
        raise ValueError("experiment compute policy changed")

    torch_before = "torch" in sys.modules
    config = QualityAwareFusionConfig.from_protocol(protocol)
    benchmark = protocol["synthetic_benchmark"]
    rng = np.random.default_rng(int(benchmark["seed"]))
    sessions = int(benchmark["sessions_per_condition"])
    token_count = int(benchmark["tokens_per_session"])
    text_sigma = float(benchmark["text_noise_sigma"])
    quality_sigma = float(benchmark["quality_observation_noise_sigma"])

    condition_results: dict[str, dict[str, float | int]] = {}
    all_target: list[np.ndarray] = []
    all_text: list[np.ndarray] = []
    all_gaze_neutral: list[np.ndarray] = []
    all_static: list[np.ndarray] = []
    all_candidate: list[np.ndarray] = []
    missing_fallback_delta = 0.0

    for condition_name, condition in benchmark["conditions"].items():
        targets: list[np.ndarray] = []
        text_predictions: list[np.ndarray] = []
        gaze_neutral_predictions: list[np.ndarray] = []
        static_predictions: list[np.ndarray] = []
        candidate_predictions: list[np.ndarray] = []
        gaze_weights: list[np.ndarray] = []
        missing_rows = 0

        for _ in range(sessions):
            target = rng.beta(2.2, 2.2, size=token_count)
            text_score = np.clip(
                target + rng.normal(0.0, text_sigma, size=token_count),
                0.0,
                1.0,
            )
            missing = rng.random(token_count) < float(
                condition["missing_probability"]
            )
            gaze_score = np.clip(
                target
                + float(condition["gaze_bias"])
                + rng.normal(
                    0.0,
                    float(condition["gaze_noise_sigma"]),
                    size=token_count,
                ),
                0.0,
                1.0,
            )
            observed_quality = {
                name: np.clip(
                    float(condition[name])
                    + rng.normal(0.0, quality_sigma, size=token_count),
                    0.0,
                    1.0,
                )
                for name in (
                    "mapping_confidence",
                    "tracking_coverage",
                    "stability",
                    "calibration_quality",
                )
            }

            candidate = np.empty(token_count, dtype=np.float64)
            weight = np.empty(token_count, dtype=np.float64)
            for index in range(token_count):
                result = fuse_quality_aware(
                    text_score=float(text_score[index]),
                    gaze_score=(None if missing[index] else float(gaze_score[index])),
                    mapping_confidence=float(
                        observed_quality["mapping_confidence"][index]
                    ),
                    tracking_coverage=float(
                        observed_quality["tracking_coverage"][index]
                    ),
                    stability=float(observed_quality["stability"][index]),
                    calibration_quality=float(
                        observed_quality["calibration_quality"][index]
                    ),
                    config=config,
                )
                candidate[index] = result.fused_score
                weight[index] = result.gaze_weight

            static = np.where(
                missing,
                text_score,
                0.6 * gaze_score + 0.4 * text_score,
            )
            gaze_neutral = np.where(missing, 0.5, gaze_score)
            if missing.any():
                missing_fallback_delta = max(
                    missing_fallback_delta,
                    float(np.max(np.abs(candidate[missing] - text_score[missing]))),
                )
            missing_rows += int(missing.sum())
            targets.append(target)
            text_predictions.append(text_score)
            gaze_neutral_predictions.append(gaze_neutral)
            static_predictions.append(static)
            candidate_predictions.append(candidate)
            gaze_weights.append(weight)

        condition_target = np.concatenate(targets)
        condition_text = np.concatenate(text_predictions)
        condition_gaze = np.concatenate(gaze_neutral_predictions)
        condition_static = np.concatenate(static_predictions)
        condition_candidate = np.concatenate(candidate_predictions)
        condition_weight = np.concatenate(gaze_weights)
        condition_results[condition_name] = {
            "rows": int(len(condition_target)),
            "missing_rows": missing_rows,
            "text_only_mae": _mae(condition_text, condition_target),
            "gaze_only_neutral_missing_mae": _mae(
                condition_gaze, condition_target
            ),
            "static_fusion_mae": _mae(condition_static, condition_target),
            "quality_aware_candidate_mae": _mae(
                condition_candidate, condition_target
            ),
            "candidate_minus_static_mae": _mae(
                condition_candidate, condition_target
            )
            - _mae(condition_static, condition_target),
            "mean_gaze_weight": float(np.mean(condition_weight)),
        }
        all_target.append(condition_target)
        all_text.append(condition_text)
        all_gaze_neutral.append(condition_gaze)
        all_static.append(condition_static)
        all_candidate.append(condition_candidate)

    target_all = np.concatenate(all_target)
    text_all = np.concatenate(all_text)
    gaze_all = np.concatenate(all_gaze_neutral)
    static_all = np.concatenate(all_static)
    candidate_all = np.concatenate(all_candidate)
    occurrence = _occurrence_collision_probe(token_count)

    gate_spec = protocol["decision_gate"]
    expected_order = gate_spec["mean_gaze_weight_order"]
    observed_weights = {
        name: float(condition_results[name]["mean_gaze_weight"])
        for name in expected_order
    }
    gates = {
        "occurrence_collision_count_equals_zero": (
            occurrence["occurrence_collision_count"]
            == int(gate_spec["occurrence_collision_count_equals"])
        ),
        "missing_fallback_within_tolerance": (
            missing_fallback_delta
            <= float(gate_spec["missing_fallback_max_absolute_delta_at_most"])
        ),
        "mean_gaze_weight_order_met": _mean_gaze_weight_order_met(
            expected_order,
            observed_weights,
        ),
        "candidate_aggregate_mae_less_than_static": (
            _mae(candidate_all, target_all) < _mae(static_all, target_all)
        ),
        "candidate_clean_within_static_margin": (
            float(condition_results["clean"]["quality_aware_candidate_mae"])
            <= float(condition_results["clean"]["static_fusion_mae"])
            + float(gate_spec["candidate_clean_mae_may_exceed_static_by_at_most"])
        ),
        "missing_candidate_equals_text_only": (
            float(condition_results["missing"]["quality_aware_candidate_mae"])
            == float(condition_results["missing"]["text_only_mae"])
        ),
    }
    for condition_name in gate_spec["candidate_mae_less_than_static_in"]:
        gates[f"candidate_{condition_name}_mae_less_than_static"] = (
            float(
                condition_results[condition_name]["quality_aware_candidate_mae"]
            )
            < float(condition_results[condition_name]["static_fusion_mae"])
        )
    passed = all(gates.values())

    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "quality-aware-text-fusion-v2-corruption-benchmark",
        "run_id": str(args.run_id),
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "protocol_commit": _read_git_head(ROOT),
        "implementation_sha256": _combined_sha256(IMPLEMENTATION_PATHS),
        "configuration": {
            "seed": int(benchmark["seed"]),
            "sessions_per_condition": sessions,
            "tokens_per_session": token_count,
            "condition_count": len(condition_results),
            "total_rows": int(len(target_all)),
            "candidate": {
                "gaze_weight_power": config.gaze_weight_power,
                "quality_aggregation": "geometric_mean",
                "mode": "shadow_only",
            },
        },
        "conditions": condition_results,
        "aggregate": {
            "text_only_mae": _mae(text_all, target_all),
            "gaze_only_neutral_missing_mae": _mae(gaze_all, target_all),
            "static_fusion_mae": _mae(static_all, target_all),
            "quality_aware_candidate_mae": _mae(candidate_all, target_all),
            "candidate_minus_static_mae": _mae(candidate_all, target_all)
            - _mae(static_all, target_all),
        },
        "occurrence_probe": occurrence,
        "missing_fallback_max_absolute_delta": missing_fallback_delta,
        "observed_mean_gaze_weight_order": observed_weights,
        "gates": {**gates, "passed": passed},
        "decision": {
            "status": (
                "retain_shadow_candidate_and_freeze_real_capture_manifest"
                if passed
                else "record_failure_without_parameter_changes"
            ),
            "production_model_changed": False,
            "real_webcam_generalization_claimed": False,
        },
        "leakage_controls": protocol["leakage_controls"],
        "compute": {
            "device": "cpu",
            "gpu_used": False,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "torch_imported_before": torch_before,
            "torch_imported_after": "torch" in sys.modules,
            "network_used": False,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"2026-08-07-quality-aware-text-fusion-v2-{args.run_id}"
    json_path = args.output_dir / f"{stem}.json"
    report_path = args.output_dir.parent / f"{stem}.md"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_markdown_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"JSON: {json_path}")
    print(f"Report: {report_path}")
    return 0


def _occurrence_collision_probe(token_count: int) -> dict[str, int]:
    vocabulary = ["the", "reader", "examined", "ordinary", "report"]
    words = [vocabulary[index % len(vocabulary)] for index in range(token_count)]
    analysis = [
        {"word": word, "position": index, "load_score": 0.5}
        for index, word in enumerate(words)
    ]
    occurrence_events = [
        {
            "occurrence_id": f"page:1:word:{index}",
            "page_num": 1,
            "word_index": index,
            "word": word,
            "dwell_count": 1 + index % 4,
            "fixation_count": 1,
            "hit_count": 1 + index % 4,
            "confidence_counts": {"high": 1 + index % 4},
        }
        for index, word in enumerate(words)
    ]
    occurrence_aggregated = aggregate_gaze_events(occurrence_events)
    aligned = align_gaze_occurrences(analysis, occurrence_aggregated)
    legacy_events = [
        {"word": event["word"], "dwell_count": event["dwell_count"]}
        for event in occurrence_events
    ]
    legacy_aggregated = aggregate_gaze_events(legacy_events)
    unique_aligned = len({record["occurrence_id"] for record in aligned})
    return {
        "input_occurrences": token_count,
        "occurrence_aggregates": len(occurrence_aggregated),
        "unique_aligned_occurrences": unique_aligned,
        "occurrence_collision_count": token_count - unique_aligned,
        "legacy_word_collision_count": token_count - len(legacy_aggregated),
    }


def _mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - target)))


def _mean_gaze_weight_order_met(
    expected_order: list[str], observed_weights: dict[str, float]
) -> bool:
    """Return whether every adjacent frozen quality condition is decreasing."""
    return all(
        observed_weights[first] > observed_weights[second]
        for first, second in pairwise(expected_order)
    )


def _combined_sha256(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _read_git_head(root: Path) -> str:
    head = (root / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref_path = root / ".git" / head.removeprefix("ref: ")
    return ref_path.read_text(encoding="utf-8").strip()


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Quality-Aware Text/Fusion v2 - Run 001",
        "",
        f"- Protocol: `{summary['protocol_id']}`",
        f"- Protocol commit: `{summary['protocol_commit']}`",
        "- Compute: CPU only; no Torch, network, participant data, or QA dataset",
        f"- Decision: **`{summary['decision']['status']}`**",
        "- Production default changed: **no**",
        "",
        "## Frozen corruption benchmark",
        "",
        "| Condition | Rows | Missing | Text MAE | Static MAE | Candidate MAE | Candidate - Static | Mean gaze weight |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in summary["conditions"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(result["rows"]),
                    str(result["missing_rows"]),
                    f"{result['text_only_mae']:.6f}",
                    f"{result['static_fusion_mae']:.6f}",
                    f"{result['quality_aware_candidate_mae']:.6f}",
                    f"{result['candidate_minus_static_mae']:+.6f}",
                    f"{result['mean_gaze_weight']:.6f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision gates",
            "",
            *[
                f"- [{'x' if passed else ' '}] `{name}`"
                for name, passed in summary["gates"].items()
                if name != "passed"
            ],
            "",
            "## Interpretation boundary",
            "",
            "This run tests deterministic software behavior under a frozen synthetic corruption model. It does not establish benefit on real webcam captures and cannot promote the production default. A later protocol must freeze participant, article, and device/source holdouts before any independent real-capture outcomes are inspected.",
            "",
            "## Integrity",
            "",
            f"- Protocol SHA-256: `{summary['protocol_sha256']}`",
            f"- Implementation SHA-256: `{summary['implementation_sha256']}`",
            f"- Occurrence collisions: `{summary['occurrence_probe']['occurrence_collision_count']}`",
            f"- Legacy word-key collisions in the same probe: `{summary['occurrence_probe']['legacy_word_collision_count']}`",
            f"- Missing fallback max delta: `{summary['missing_fallback_max_absolute_delta']}`",
            f"- Torch imported after run: `{summary['compute']['torch_imported_after']}`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

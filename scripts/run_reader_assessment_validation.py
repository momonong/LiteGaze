"""CPU-only validation harness for the experimental reader assessment v2.

The simulation checks routing and uncertainty under explicit response-model
assumptions. It does not fit item parameters and is not evidence that the pilot
items measure real English proficiency.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from core.cognitive_inspector.adaptive import (
    ITEM_BY_ID,
    MAX_ROUNDS,
    MIN_ROUNDS,
    estimate_theta,
    initial_passage,
    select_next_passage,
    should_stop,
    validate_item_bank,
)
from core.cognitive_inspector.inspector import CognitiveInspector


def _probability(theta: float, item: dict[str, Any], truth: dict[str, float]) -> float:
    exponent = -truth["a"] * (theta - truth["b"])
    logistic = 1.0 / (1.0 + math.exp(min(40.0, max(-40.0, exponent))))
    return truth["c"] + (1.0 - truth["c"]) * logistic


def _truth_parameters(
    rng: np.random.Generator, shifted: bool
) -> dict[str, dict[str, float]]:
    parameters = {}
    for question_id, item in ITEM_BY_ID.items():
        if shifted:
            parameters[question_id] = {
                "a": float(item["discrimination_a"] * rng.lognormal(0.0, 0.18)),
                "b": float(item["difficulty_b"] + rng.normal(0.0, 0.35)),
                "c": float(
                    np.clip(item["guessing_c"] + rng.normal(0.0, 0.035), 0.15, 0.35)
                ),
            }
        else:
            parameters[question_id] = {
                "a": float(item["discrimination_a"]),
                "b": float(item["difficulty_b"]),
                "c": float(item["guessing_c"]),
            }
    return parameters


def _simulate_regime(
    *,
    participants: int,
    seed: int,
    shifted_truth: bool,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    truth = _truth_parameters(rng, shifted_truth)
    true_theta = np.clip(rng.normal(0.0, 1.15, participants), -3.0, 3.0)
    estimated = np.empty(participants, dtype=float)
    lower = np.empty(participants, dtype=float)
    upper = np.empty(participants, dtype=float)
    rounds = np.empty(participants, dtype=int)
    item_counts = np.empty(participants, dtype=int)
    passage_exposure: Counter[str] = Counter()

    for participant_index, theta in enumerate(true_theta):
        assessment_id = f"sim-{seed}-{participant_index}"
        passage = initial_passage(assessment_id)
        history: list[dict[str, Any]] = []
        while True:
            item_results = []
            for item in passage["questions"]:
                probability = _probability(theta, item, truth[item["question_id"]])
                item_results.append(
                    {
                        "question_id": item["question_id"],
                        "correct": bool(rng.random() < probability),
                    }
                )
            history.append(
                {
                    "passage_id": passage["passage_id"],
                    "item_results": item_results,
                }
            )
            passage_exposure[passage["passage_id"]] += 1
            if should_stop(history):
                break
            passage = select_next_passage(history, assessment_id)
            if passage is None:
                break

        posterior = estimate_theta(history)
        estimated[participant_index] = posterior["theta"]
        lower[participant_index], upper[participant_index] = posterior[
            "credible_interval_95"
        ]
        rounds[participant_index] = len(history)
        item_counts[participant_index] = posterior["item_count"]

    error = estimated - true_theta
    rank_true = np.argsort(np.argsort(true_theta))
    rank_estimated = np.argsort(np.argsort(estimated))
    spearman = float(np.corrcoef(rank_true, rank_estimated)[0, 1])
    coverage = float(np.mean((true_theta >= lower) & (true_theta <= upper)))
    bins = [-3.1, -1.5, -0.5, 0.5, 1.5, 3.1]
    bin_rows = []
    for left, right in pairwise(bins):
        mask = (true_theta >= left) & (true_theta < right)
        if not np.any(mask):
            continue
        bin_rows.append(
            {
                "true_theta_range": [left, right],
                "n": int(np.sum(mask)),
                "mean_true_theta": round(float(np.mean(true_theta[mask])), 4),
                "mean_estimated_theta": round(float(np.mean(estimated[mask])), 4),
                "rmse": round(float(np.sqrt(np.mean(error[mask] ** 2))), 4),
            }
        )

    return {
        "participants": participants,
        "truth_regime": "shifted_item_parameters"
        if shifted_truth
        else "matched_model_assumptions",
        "mean_bias": round(float(np.mean(error)), 4),
        "rmse": round(float(np.sqrt(np.mean(error**2))), 4),
        "mae": round(float(np.mean(np.abs(error))), 4),
        "spearman_theta": round(spearman, 4),
        "credible_interval_95_coverage": round(coverage, 4),
        "mean_rounds": round(float(np.mean(rounds)), 4),
        "max_round_fraction": round(float(np.mean(rounds == MAX_ROUNDS)), 4),
        "round_distribution": {
            str(int(value)): int(count)
            for value, count in zip(*np.unique(rounds, return_counts=True))
        },
        "mean_item_count": round(float(np.mean(item_counts)), 4),
        "passage_exposure": dict(sorted(passage_exposure.items())),
        "ability_bins": bin_rows,
    }


def _gaze_metamorphic_checks() -> dict[str, Any]:
    def events(samples_per_word: int, tick_ms: float, confidence: float) -> list[dict]:
        output = []
        timestamp = 0.0
        for index in range(30):
            for _ in range(samples_per_word):
                output.append(
                    {
                        "word": f"word{index}",
                        "index": index,
                        "timestamp_ms": timestamp,
                        "confidence": confidence,
                    }
                )
                timestamp += tick_ms
        return output

    slow = CognitiveInspector(sample_rate_hz=8).analyze(events(2, 125.0, 0.9))
    fast = CognitiveInspector(sample_rate_hz=16).analyze(events(4, 62.5, 0.9))
    low_confidence = CognitiveInspector(sample_rate_hz=8).analyze(events(2, 125.0, 0.1))
    sample_rate_delta = abs(
        slow["summary"]["median_fixation_duration_ms"]
        - fast["summary"]["median_fixation_duration_ms"]
    )
    return {
        "sample_rate_invariance": {
            "passed": sample_rate_delta <= 1.0,
            "median_duration_delta_ms": sample_rate_delta,
        },
        "confidence_separation": {
            "passed": (
                slow["summary"] == low_confidence["summary"]
                and slow["data_quality"]["score"]
                > low_confidence["data_quality"]["score"]
            ),
            "high_quality_score": slow["data_quality"]["score"],
            "low_quality_score": low_confidence["data_quality"]["score"],
        },
        "unsupported_claim_abstention": {
            "passed": all(
                slow["claims"][claim]["status"] == "not_estimated"
                for claim in (
                    "cognitive_ability",
                    "english_proficiency",
                    "attention",
                    "fatigue",
                    "cognitive_load",
                )
            )
        },
    }


def run_validation(participants: int, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    bank = validate_item_bank()
    matched = _simulate_regime(
        participants=participants, seed=seed, shifted_truth=False
    )
    shifted = _simulate_regime(
        participants=participants, seed=seed + 1, shifted_truth=True
    )
    metamorphic = _gaze_metamorphic_checks()

    gates = {
        "item_bank_static_audit": bank["ok"],
        "matched_spearman_at_least_0_65": matched["spearman_theta"] >= 0.65,
        "matched_rmse_below_0_90": matched["rmse"] < 0.90,
        "matched_interval_coverage_at_least_0_85": matched[
            "credible_interval_95_coverage"
        ]
        >= 0.85,
        "shifted_spearman_at_least_0_55": shifted["spearman_theta"] >= 0.55,
        "rounds_within_protocol": all(
            MIN_ROUNDS <= int(round_count) <= MAX_ROUNDS
            for regime in (matched, shifted)
            for round_count in regime["round_distribution"]
        ),
        "gaze_metamorphic_checks": all(
            check["passed"] for check in metamorphic.values()
        ),
    }
    return {
        "experiment": "reader_assessment_v2_cpu_validation",
        "seed": seed,
        "participants_per_regime": participants,
        "total_simulated_participants": participants * 2,
        "no_parameter_fitting_performed": True,
        "qa_content_used_for_fitting": False,
        "gpu_requested": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "item_bank": bank,
        "matched_assumption_sanity_check": matched,
        "shifted_parameter_stress_test": shifted,
        "gaze_metamorphic_checks": metamorphic,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "operational_measurement_ready": False,
        "known_limitations": [
            "The item parameters are expert seeds rather than empirical calibration estimates.",
            "A high max-round fraction indicates that the six-passage bank is information-limited.",
            "The normal prior shrinks estimates at the extreme ends of the pilot scale.",
            "No real participant, item-form, subgroup fairness, test-retest, or external-criterion validation has been run.",
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "interpretation_limit": (
            "Simulation validates software behaviour under declared assumptions only. "
            "It does not establish item calibration, CEFR linkage, external validity, or fairness."
        ),
    }


def _markdown(result: dict[str, Any]) -> str:
    matched = result["matched_assumption_sanity_check"]
    shifted = result["shifted_parameter_stress_test"]
    gate_rows = "\n".join(
        f"| {name} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in result["gates"].items()
    )
    return f"""# Reader Assessment v2 CPU Validation

- Simulated participants: `{result["total_simulated_participants"]}` ({result["participants_per_regime"]} per regime)
- Seed: `{result["seed"]}`
- Runtime: `{result["elapsed_seconds"]} s`
- GPU requested: `{result["gpu_requested"]}`
- Parameter fitting: `{not result["no_parameter_fitting_performed"]}`
- QA content used for fitting: `{result["qa_content_used_for_fitting"]}`
- Overall gate: `{"PASS" if result["all_gates_passed"] else "FAIL"}`
- Operational measurement ready: `{result["operational_measurement_ready"]}`

## Gates

| Gate | Result |
| :--- | :---: |
{gate_rows}

## Simulation results

| Regime | Spearman theta | RMSE | 95% interval coverage | Mean rounds |
| :--- | ---: | ---: | ---: | ---: |
| Matched assumptions | {matched["spearman_theta"]} | {matched["rmse"]} | {matched["credible_interval_95_coverage"]} | {matched["mean_rounds"]} |
| Shifted item parameters | {shifted["spearman_theta"]} | {shifted["rmse"]} | {shifted["credible_interval_95_coverage"]} | {shifted["mean_rounds"]} |

Max-round fractions were `{matched["max_round_fraction"]}` (matched) and `{shifted["max_round_fraction"]}` (shifted). A high value means the current six-passage pilot bank is not yet an efficient variable-length CAT.

## Interpretation boundary

{result["interpretation_limit"]}

The shifted regime perturbs difficulty, discrimination, and guessing parameters that the estimator never sees. It is a robustness stress test, not a substitute for real participant/item holdout validation.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--participants", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/reader_assessment/experiments"),
    )
    parser.add_argument("--name", default="validation_v2")
    args = parser.parse_args()
    if args.participants < 100:
        parser.error("--participants must be at least 100")

    result = run_validation(args.participants, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.name}.json"
    markdown_path = args.output_dir / f"{args.name}.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "all_gates_passed": result["all_gates_passed"],
                "elapsed_seconds": result["elapsed_seconds"],
                "json": str(json_path),
                "markdown": str(markdown_path),
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

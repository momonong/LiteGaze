"""Monte Carlo sensitivity analysis for paired, quality-gated collection yield."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    ROOT
    / "docs"
    / "experiments"
    / "protocols"
    / "2026-08-08-general-collection-sample-size-v1.json"
)


def _paired_normal_power(sample_size: int, effect_size: float, alpha: float) -> float:
    if sample_size <= 1:
        return 0.0
    normal = NormalDist()
    critical = normal.inv_cdf(1.0 - alpha / 2.0)
    shift = effect_size * math.sqrt(sample_size)
    return normal.cdf(-critical - shift) + 1.0 - normal.cdf(critical - shift)


def simulate(protocol: dict[str, object]) -> dict[str, object]:
    rng = np.random.default_rng(int(protocol["seed"]))
    iterations = int(protocol["iterations"])
    targets = dict(protocol["targets"])
    paired_target = int(targets["paired_behavioral_participants"])
    gaze_target = int(targets["paired_word_gaze_candidates"])
    subgroup_target = int(targets["minority_subgroup_paired_participants"])
    subgroup_prevalence = float(targets["minority_subgroup_prevalence"])
    minimum_probability = float(targets["minimum_target_probability"])
    test = dict(protocol["paired_confirmation_test"])
    alpha = float(test["alpha_two_sided"])
    effects = [float(value) for value in test["standardized_effect_sizes"]]
    maximum_enrollment = max(int(value) for value in protocol["candidate_enrollments"])
    power_tables = {
        effect: np.array(
            [
                _paired_normal_power(sample_size, effect, alpha)
                for sample_size in range(maximum_enrollment + 1)
            ]
        )
        for effect in effects
    }
    rows: list[dict[str, object]] = []

    for scenario_name, raw_scenario in dict(protocol["scenarios"]).items():
        scenario = dict(raw_scenario)
        for enrollment in protocol["candidate_enrollments"]:
            n = int(enrollment)
            visit1 = rng.binomial(n, float(scenario["visit1_completion"]), iterations)
            visit2 = rng.binomial(
                visit1,
                float(scenario["visit2_retention_given_visit1"]),
            )
            behavioral = rng.binomial(
                visit2,
                float(scenario["behavioral_usable_given_visit2"]),
            )
            word_gaze = rng.binomial(
                behavioral,
                float(scenario["word_gaze_candidate_given_behavioral"]),
            )
            subgroup = rng.binomial(behavioral, subgroup_prevalence)
            powers = {
                f"paired_power_d_{effect:.2f}": float(
                    np.mean(power_tables[effect][behavioral])
                )
                for effect in effects
            }
            row = {
                "scenario": scenario_name,
                "enrolled": n,
                "mean_visit1_completed": float(np.mean(visit1)),
                "mean_visit2_completed": float(np.mean(visit2)),
                "mean_paired_behavioral": float(np.mean(behavioral)),
                "mean_paired_word_gaze": float(np.mean(word_gaze)),
                "mean_minority_subgroup_paired": float(np.mean(subgroup)),
                "prob_paired_behavioral_target": float(np.mean(behavioral >= paired_target)),
                "prob_word_gaze_target": float(np.mean(word_gaze >= gaze_target)),
                "prob_subgroup_target": float(np.mean(subgroup >= subgroup_target)),
                **powers,
            }
            row["all_yield_targets_probability"] = float(
                np.mean(
                    (behavioral >= paired_target)
                    & (word_gaze >= gaze_target)
                    & (subgroup >= subgroup_target)
                )
            )
            rows.append(row)

    recommendations: dict[str, object] = {}
    for scenario_name in protocol["scenarios"]:
        eligible = [
            row
            for row in rows
            if row["scenario"] == scenario_name
            and row["all_yield_targets_probability"] >= minimum_probability
            and row["paired_power_d_0.40"] >= 0.80
        ]
        recommendations[str(scenario_name)] = (
            min(eligible, key=lambda row: int(row["enrolled"]))["enrolled"]
            if eligible
            else None
        )
    return {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "status": "completed_assumption_sensitive_not_a_formal_authorization",
        "seed": protocol["seed"],
        "iterations": iterations,
        "participant_outcomes_used": False,
        "gpu_used": False,
        "targets": targets,
        "recommendation_minimum_enrollment_by_scenario": recommendations,
        "rows": rows,
        "interpretation_boundary": (
            "These results are conditional on assumed retention, usability, gaze-quality, "
            "effect-size, and subgroup rates. Replace assumptions with blinded rehearsal "
            "aggregates before formal recruitment; do not tune on confirmation outcomes."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    result = simulate(protocol)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

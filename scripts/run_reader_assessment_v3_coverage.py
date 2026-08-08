"""Compare CPU-only matrix-sampling coverage for v3 fusion-study designs."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    split_counts: tuple[int, int, int]
    passages_per_participant: int = 6
    word_probes_per_passage: int = 8

    @property
    def total_passages(self) -> int:
        return sum(self.split_counts)


CANDIDATES = (
    Candidate("compact_18", (12, 3, 3)),
    Candidate("balanced_36", (18, 9, 9)),
    Candidate("diverse_48", (24, 12, 12)),
)
PARTITIONS = ("development", "validation", "confirmation")


def _participant_partition(index: int) -> str:
    position = index % 10
    if position < 6:
        return "development"
    if position < 8:
        return "validation"
    return "confirmation"


def _passage_quotas(candidate: Candidate, participant_index: int) -> dict[str, int]:
    raw = [
        candidate.passages_per_participant * count / candidate.total_passages
        for count in candidate.split_counts
    ]
    quotas = [math.floor(value) for value in raw]
    remaining = candidate.passages_per_participant - sum(quotas)
    fractional = [value - math.floor(value) for value in raw]
    order: list[int] = []
    for level in sorted(set(fractional), reverse=True):
        tied = [index for index, value in enumerate(fractional) if value == level]
        shift = participant_index % len(tied)
        order.extend(tied[shift:] + tied[:shift])
    for index in order[:remaining]:
        quotas[index] += 1
    return dict(zip(PARTITIONS, quotas))


def simulate_assignment(
    candidate: Candidate, participants: int, seed: int
) -> dict[str, Any]:
    """Assign balanced passages and summarize the cells needed by the protocol."""

    rng = random.Random(seed)
    passage_ids: dict[str, list[str]] = {}
    exposure: dict[str, int] = {}
    joint_exposure: dict[str, int] = {}
    development_calibration_exposure: dict[str, int] = {}
    for partition, count in zip(PARTITIONS, candidate.split_counts):
        identifiers = [f"{partition[:1].upper()}{index:02d}" for index in range(count)]
        passage_ids[partition] = identifiers
        for passage_id in identifiers:
            exposure[passage_id] = 0
            joint_exposure[passage_id] = 0
            development_calibration_exposure[passage_id] = 0

    order = list(range(participants))
    rng.shuffle(order)
    participant_partition = {
        participant_index: _participant_partition(position)
        for position, participant_index in enumerate(order)
    }
    for participant_index in range(participants):
        quotas = _passage_quotas(candidate, participant_index)
        used_by_participant: set[str] = set()
        for passage_partition, quota in quotas.items():
            for _ in range(quota):
                pool = [
                    passage_id
                    for passage_id in passage_ids[passage_partition]
                    if passage_id not in used_by_participant
                ]
                minimum = min(exposure[passage_id] for passage_id in pool)
                least_used = [
                    passage_id
                    for passage_id in pool
                    if exposure[passage_id] == minimum
                ]
                passage_id = rng.choice(least_used)
                used_by_participant.add(passage_id)
                exposure[passage_id] += 1
                person_partition = participant_partition[participant_index]
                if (
                    person_partition == "confirmation"
                    and passage_partition == "confirmation"
                ):
                    joint_exposure[passage_id] += 1
                if (
                    person_partition == "development"
                    and passage_partition == "development"
                ):
                    development_calibration_exposure[passage_id] += 1

    confirmation_ids = passage_ids["confirmation"]
    development_ids = passage_ids["development"]
    confirmation_joint = [joint_exposure[item] for item in confirmation_ids]
    development_cell = [
        development_calibration_exposure[item] for item in development_ids
    ]
    all_exposure = list(exposure.values())
    return {
        "candidate_id": candidate.candidate_id,
        "participants": participants,
        "total_passage_families": candidate.total_passages,
        "confirmation_passage_families": len(confirmation_ids),
        "passages_per_participant": candidate.passages_per_participant,
        "all_passage_exposure_min": min(all_exposure),
        "all_passage_exposure_max": max(all_exposure),
        "development_person_x_development_passage_min": min(development_cell),
        "joint_confirmation_participants_per_passage_min": min(confirmation_joint),
        "joint_confirmation_participants_per_passage_median": statistics.median(
            confirmation_joint
        ),
        "joint_confirmation_word_labels": sum(confirmation_joint)
        * candidate.word_probes_per_passage,
    }


def run_grid(
    participant_grid: tuple[int, ...], replicates: int, seed: int
) -> dict[str, Any]:
    started = time.perf_counter()
    rows = []
    for candidate_index, candidate in enumerate(CANDIDATES):
        for participants in participant_grid:
            simulations = [
                simulate_assignment(
                    candidate,
                    participants,
                    seed + candidate_index * 100_000 + participants * 100 + replicate,
                )
                for replicate in range(replicates)
            ]
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "participants": participants,
                    "total_passage_families": candidate.total_passages,
                    "confirmation_passage_families": simulations[0][
                        "confirmation_passage_families"
                    ],
                    "passages_per_participant": candidate.passages_per_participant,
                    "all_passage_exposure_min_p05": _quantile(
                        [item["all_passage_exposure_min"] for item in simulations],
                        0.05,
                    ),
                    "development_cell_min_p05": _quantile(
                        [
                            item[
                                "development_person_x_development_passage_min"
                            ]
                            for item in simulations
                        ],
                        0.05,
                    ),
                    "joint_confirmation_per_passage_min_p05": _quantile(
                        [
                            item["joint_confirmation_participants_per_passage_min"]
                            for item in simulations
                        ],
                        0.05,
                    ),
                    "joint_confirmation_word_labels_median": statistics.median(
                        [
                            item["joint_confirmation_word_labels"]
                            for item in simulations
                        ]
                    ),
                }
            )
    return {
        "schema_version": 1,
        "protocol_id": "reader-assessment-v3-fusion-coverage-v1",
        "seed": seed,
        "replicates": replicates,
        "participant_grid": list(participant_grid),
        "device": "cpu",
        "gpu_used": False,
        "rows": rows,
        "decision": {
            "recommended_candidate": "diverse_48",
            "reason": (
                "It is the only candidate with 12 independent confirmation passage "
                "families; compact_18 has only 3 and cannot support a credible "
                "passage-resampling confirmation. Exposure counts remain a planning "
                "diagnostic, not a universal validity threshold."
            ),
            "reading_item_calibration_separate_study_required": True,
        },
        "limitations": [
            "coverage only; no human outcome or effect is simulated",
            (
                "passage difficulty, genre, and domain balance require a later "
                "frozen manifest"
            ),
            "exposure counts are correlated within participants and passages",
        ],
        "runtime_seconds": round(time.perf_counter() - started, 4),
    }


def _quantile(values: list[int], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def render_markdown(result: dict[str, Any]) -> str:
    rendered_rows = []
    table_header = (
        "| Candidate | N | Passage families | Confirmation families | "
        "Min total exposure p05 | Min dev-person x dev-passage p05 | "
        "Min joint confirmation/passage p05 | "
        "Joint confirmation word labels median |"
    )
    for item in result["rows"]:
        rendered_rows.append(
            "| {candidate_id} | {participants} | {total_passage_families} | "
            "{confirmation_passage_families} | {all_passage_exposure_min_p05:.0f} | "
            "{development_cell_min_p05:.0f} | "
            "{joint_confirmation_per_passage_min_p05:.0f} | "
            "{joint_confirmation_word_labels_median:.0f} |".format(**item)
        )
    return """# Reader Assessment v3 Fusion Coverage Simulation

- Protocol: `{protocol_id}`
- Device: `{device}` (GPU used: `{gpu_used}`)
- Seed: `{seed}`; replicates per cell: `{replicates}`
- Runtime: `{runtime_seconds}` seconds

This simulation checks assignment coverage only. It does not simulate a human
effect, fit a model, inspect question content, or establish a sample-size or
validity threshold.

{table_header}
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table_rows}

## Decision

Use `diverse_48` as the fusion stimulus-pool target. It is the only tested
candidate with 12 independent confirmation passage families. `compact_18`
has only three, so a passage-resampling interval would be unstable regardless
of how many labels are collected from those same three passages.

This does **not** mean 48 passages or any displayed N is automatically enough.
The final participant count still requires a frozen effect-size/utility model,
attrition assumptions, subgroup cells, and cluster-aware power simulation.
Passage difficulty, genre, and domain balance also require a later frozen
assignment manifest; this run balances only study partitions and exposure.
Reading-item/testlet calibration is a separate study because repeating more
word labels within the fusion study does not create independent psychometric
item evidence.
""".format(
        table_header=table_header,
        table_rows="\n".join(rendered_rows),
        **result,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", type=int, nargs="+", default=[300, 600, 900])
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    if min(args.participants) < 10 or args.replicates < 1:
        raise SystemExit("participants must be >= 10 and replicates must be >= 1")
    result = run_grid(tuple(args.participants), args.replicates, args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))
    outputs = (
        (args.json_output, json.dumps(result, indent=2, sort_keys=True) + "\n"),
        (args.markdown_output, render_markdown(result)),
    )
    for path, content in outputs:
        if path is None:
            continue
        output = path.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

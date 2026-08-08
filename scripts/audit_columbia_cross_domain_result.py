"""Recompute Columbia v2 aggregates from ignored row-level evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.columbia_gaze.metrics import (
    summarize_model,
    zero_gaze_summary,
)
from scripts.gaze_diversity.metrics import angular_errors_degrees, summarize_errors
from scripts.run_columbia_cross_domain_gaze import (
    _effectiveness_decision,
    _file_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = (
    ROOT
    / "docs"
    / "experiments"
    / "results"
    / "2026-08-08-columbia-cross-domain-gaze-v2-run-001.json"
)
DEFAULT_EVIDENCE = (
    ROOT
    / "output"
    / "columbia-cross-domain-gaze-v2-run-001"
    / "prediction-evidence.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "experiments"
    / "results"
    / "2026-08-08-columbia-cross-domain-gaze-v2-run-001-independent-audit.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    evidence = _load_evidence(args.evidence)
    metadata = evidence.pop("metadata")
    candidate_prediction = evidence["candidate_prediction"]
    production_prediction = evidence["production_prediction"]
    targets = evidence["targets"]
    subjects = evidence["subject_indices"]
    head_poses = evidence["head_poses"]
    vertical_gazes = evidence["vertical_gazes"]
    horizontal_gazes = evidence["horizontal_gazes"]
    fallback_mask = evidence["fallback_mask"]
    production_success = evidence["production_success"]

    zero_all, zero_subject_means = zero_gaze_summary(
        targets,
        subjects,
        head_poses=head_poses,
        vertical_gazes=vertical_gazes,
        horizontal_gazes=horizontal_gazes,
    )
    candidate = summarize_model(
        candidate_prediction,
        targets,
        subjects,
        head_poses=head_poses,
        vertical_gazes=vertical_gazes,
        horizontal_gazes=horizontal_gazes,
        zero_subject_means=zero_subject_means,
        bootstrap_resamples=20000,
        bootstrap_seed=20260813,
    )
    candidate_errors = angular_errors_degrees(candidate_prediction, targets)
    candidate["eye_corner_strata"] = {
        "official_annotation_5865": summarize_errors(candidate_errors[~fallback_mask]),
        "mediapipe_fallback_15": summarize_errors(candidate_errors[fallback_mask]),
    }

    successful = production_success.astype(bool, copy=False)
    production_targets = targets[successful]
    production_subjects = subjects[successful]
    production_heads = head_poses[successful]
    production_verticals = vertical_gazes[successful]
    production_horizontals = horizontal_gazes[successful]
    production_zero, production_zero_subject_means = zero_gaze_summary(
        production_targets,
        production_subjects,
        head_poses=production_heads,
        vertical_gazes=production_verticals,
        horizontal_gazes=production_horizontals,
    )
    production = summarize_model(
        production_prediction,
        production_targets,
        production_subjects,
        head_poses=production_heads,
        vertical_gazes=production_verticals,
        horizontal_gazes=production_horizontals,
        zero_subject_means=production_zero_subject_means,
        bootstrap_resamples=20000,
        bootstrap_seed=20260813,
    )
    recomputed_metrics = {
        "zero_gaze_all_rows": zero_all,
        "candidate": candidate,
        "production_zero_gaze_success_population": production_zero,
        "production": production,
    }
    candidate_effectiveness = _effectiveness_decision(
        candidate,
        zero_all,
        minimum_subjects=42,
    )
    production_effectiveness = _effectiveness_decision(
        production,
        production_zero,
        minimum_subjects=42,
    )
    production_effectiveness["coverage_at_least_0_95"] = (
        float(result["preprocessing"]["production"]["coverage"]) >= 0.95
    )
    production_effectiveness["passed"] = all(
        value for key, value in production_effectiveness.items() if key != "passed"
    )
    recomputed_effectiveness = {
        "candidate": candidate_effectiveness,
        "production": production_effectiveness,
    }
    expected_integrity = {
        "protocol_sha256": result["protocol"]["sha256"],
        "base_protocol_sha256": result["protocol"]["base_sha256"],
        "implementation_sha256": result["implementation_sha256"],
        "candidate_state_dict_sha256": [
            item["state_dict_sha256"] for item in result["candidate_training"]
        ],
        "production_state_dict_sha256": result["production_model"]["state_dict_sha256"],
    }
    checks = {
        "result_experiment_exact": (
            result.get("experiment") == "columbia-cross-domain-gaze-v2"
        ),
        "evidence_sha256_matches": (
            _file_sha256(args.evidence) == result["local_prediction_evidence"]["sha256"]
        ),
        "evidence_integrity_matches": metadata == expected_integrity,
        "candidate_rows_exact": candidate_prediction.shape == (5880, 2),
        "production_rows_match_success_mask": (
            production_prediction.shape == (int(successful.sum()), 2)
        ),
        "fallback_rows_exact": int(fallback_mask.sum()) == 15,
        "subjects_exact": len(np.unique(subjects)) == 56,
        "all_evidence_finite": all(
            np.isfinite(values).all()
            for values in (candidate_prediction, production_prediction, targets)
        ),
        "metrics_recompute_exact": _nested_close(
            recomputed_metrics,
            result["metrics"],
        ),
        "effectiveness_recompute_exact": (
            recomputed_effectiveness == result["effectiveness"]
        ),
        "audit_runtime_did_not_import_torch": "torch" not in sys.modules,
    }
    checks["passed"] = all(checks.values())
    audit = {
        "schema_version": 1,
        "experiment": "columbia-cross-domain-gaze-v2-independent-audit",
        "created_at": datetime.now(UTC).isoformat(),
        "result_sha256": _file_sha256(args.result),
        "evidence_sha256": _file_sha256(args.evidence),
        "checks": checks,
        "recomputed_summary": {
            "zero_macro_subject_mean_degrees": zero_all["macro_subject_mean_degrees"],
            "candidate_macro_subject_mean_degrees": candidate[
                "macro_subject_mean_degrees"
            ],
            "production_macro_subject_mean_degrees": production[
                "macro_subject_mean_degrees"
            ],
            "candidate_confirmed": candidate_effectiveness["passed"],
            "production_supported": production_effectiveness["passed"],
        },
    }
    _atomic_json(args.output, audit)
    print(
        "INDEPENDENT_AUDIT "
        f"passed={checks['passed']} "
        f"candidate={candidate['macro_subject_mean_degrees']:.4f} "
        f"production={production['macro_subject_mean_degrees']:.4f}",
        flush=True,
    )
    print(f"AUDIT_JSON={args.output}")
    return 0 if checks["passed"] else 1


def _load_evidence(path: Path) -> dict[str, Any]:
    required = {
        "metadata_json",
        "candidate_prediction",
        "production_prediction",
        "targets",
        "subject_indices",
        "head_poses",
        "vertical_gazes",
        "horizontal_gazes",
        "fallback_mask",
        "production_success",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("prediction evidence keys do not match the frozen schema")
        payload = {name: archive[name].copy() for name in required - {"metadata_json"}}
        payload["metadata"] = json.loads(str(archive["metadata_json"].item()))
    rows = len(payload["targets"])
    if rows != 5880:
        raise ValueError("prediction evidence does not contain 5,880 targets")
    one_dimensional = (
        "subject_indices",
        "head_poses",
        "vertical_gazes",
        "horizontal_gazes",
        "fallback_mask",
        "production_success",
    )
    if any(payload[name].shape != (rows,) for name in one_dimensional):
        raise ValueError("prediction evidence factor shape mismatch")
    return payload


def _nested_close(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _nested_close(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _nested_close(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())

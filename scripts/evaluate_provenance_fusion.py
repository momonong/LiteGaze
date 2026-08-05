"""Evaluate the candidate text/gaze fusion on an independent real capture.

The input manifest must prove that the difficulty target is independent of the
gaze and text predictors and that fusion parameters were frozen before outcomes
were read.  Reading-time-only or public-QA evaluations are rejected.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from core.cognition.fusion_validation import evaluate_fusion_candidate
from core.cognition.text_artifact import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FUSION_VALIDATION_PATH = PROJECT_ROOT / "core/cognition/fusion_validation.py"
GENERALIZATION_PATH = PROJECT_ROOT / "core/cognition/generalization.py"
TEXT_ARTIFACT_RUNTIME_PATH = PROJECT_ROOT / "core/cognition/text_artifact.py"
PROTOCOL_PATH = (
    PROJECT_ROOT
    / "docs/experiments/protocols/2026-08-06-production-text-fusion-v1.json"
)
TEXT_ARTIFACT_MANIFEST = (
    PROJECT_ROOT / "core/cognition/artifacts/en_text_difficulty_m1_v1.manifest.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output/provenance-fusion-evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    input_path = args.input.resolve()
    dataset_manifest_path = args.dataset_manifest.resolve()
    output_dir = args.output_dir.resolve()

    metadata = _read_json(dataset_manifest_path)
    _validate_source_manifest(metadata, input_path)
    frame = pd.read_csv(input_path)
    summary, predictions = evaluate_fusion_candidate(
        frame,
        dataset_metadata=metadata,
    )
    created_at = datetime.now(UTC).isoformat()
    summary.update(
        {
            "created_at": created_at,
            "input": {
                "path": str(input_path),
                "sha256": sha256_file(input_path),
                "dataset_manifest_path": str(dataset_manifest_path),
                "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            },
            "protocol": {
                "path": PROTOCOL_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(PROTOCOL_PATH),
            },
            "implementation": {
                path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
                for path in (
                    Path(__file__).resolve(),
                    FUSION_VALIDATION_PATH,
                    GENERALIZATION_PATH,
                    TEXT_ARTIFACT_RUNTIME_PATH,
                )
            },
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    predictions_path = output_dir / "cross_fitted_predictions.csv.gz"
    report_path = output_dir / "report.md"
    manifest_path = output_dir / "manifest.json"
    _write_json(summary_path, summary)
    predictions.to_csv(
        predictions_path,
        index=False,
        compression="gzip",
        lineterminator="\n",
    )
    _write_report(report_path, summary)
    manifest = {
        "schema_version": 1,
        "protocol_id": "production-text-fusion-v1",
        "created_at": created_at,
        "implementation": summary["implementation"],
        "artifacts": {
            path.name: {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (summary_path, predictions_path, report_path)
        },
    }
    _write_json(manifest_path, manifest)

    print(f"[complete] {summary['promotion']['status']}")
    print(f"[complete] production_model_changed={summary['promotion']['production_model_changed']}")
    print(f"[complete] report={report_path}")
    return 0


def _validate_source_manifest(metadata: Mapping[str, Any], input_path: Path) -> None:
    expected_input_hash = str(metadata.get("input_sha256", ""))
    if not expected_input_hash or expected_input_hash != sha256_file(input_path):
        raise ValueError("fusion input SHA-256 does not match its frozen manifest")
    if not TEXT_ARTIFACT_MANIFEST.is_file():
        raise FileNotFoundError("candidate text artifact manifest is missing")
    artifact_manifest = _read_json(TEXT_ARTIFACT_MANIFEST)
    expected_artifact_hash = artifact_manifest.get("artifact", {}).get("sha256")
    if metadata.get("text_artifact_sha256") != expected_artifact_hash:
        raise ValueError("fusion text scores do not match the candidate artifact")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    rows = []
    for axis, result in summary["holdouts"].items():
        comparison = result["comparisons"]["combined_minus_gaze_only"]
        rows.append(
            "| "
            + " | ".join(
                [
                    axis,
                    str(result["group_count"]),
                    f"{result['models']['gaze_only']['macro_group_spearman']:.4f}",
                    f"{result['models']['combined']['macro_group_spearman']:.4f}",
                    f"{comparison['mean_difference']:+.4f}",
                    f"[{comparison['ci_95_low']:+.4f}, {comparison['ci_95_high']:+.4f}]",
                    f"{comparison['positive_outer_folds']}/{comparison['outer_fold_count']}",
                    "pass" if result["gate"]["passed"] else "fail",
                ]
            )
            + " |"
        )
    lines = [
        "# Independent Text/Gaze Fusion Evaluation",
        "",
        f"- Created: {summary['created_at']}",
        f"- Dataset: `{summary['dataset'].get('dataset_id', 'unknown')}`",
        f"- Decision: **{summary['promotion']['status']}**",
        "- Production model changed: no",
        "",
        "| Holdout | Groups | Gaze-only rho | Combined rho | Delta | 95% CI | Positive folds | Gate |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |",
        *rows,
        "",
        (
            "The primary target was independently collected and was not derived from "
            "gaze, text-model output, reading time, or a public QA benchmark. Both "
            "the capture-group and article holdouts must pass before promotion is "
            "eligible."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    raise SystemExit(main())

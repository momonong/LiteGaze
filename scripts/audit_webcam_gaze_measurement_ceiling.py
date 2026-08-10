"""Audit fixed-target webcam gaze resolution without opening images or using GPU."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from core.gaze_core.measurement_ceiling import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_LINE_GAP_PX,
    DEFAULT_MEDIAN_WORD_WIDTH_PX,
    DEFAULT_PREFLIGHT_PROTOCOL_PATH,
    DEFAULT_TARGET_OVERLAP_TOLERANCE_SIGNED,
    build_measurement_ceiling_result,
    deterministic_json,
    render_measurement_ceiling_markdown,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant-session", type=Path, required=True)
    parser.add_argument(
        "--calibration-session-metadata",
        type=Path,
        required=True,
    )
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--line-gap-px", type=float, default=DEFAULT_LINE_GAP_PX)
    parser.add_argument(
        "--median-word-width-px",
        type=float,
        default=DEFAULT_MEDIAN_WORD_WIDTH_PX,
    )
    parser.add_argument(
        "--analysis-protocol",
        type=Path,
        default=DEFAULT_PREFLIGHT_PROTOCOL_PATH,
        help=(
            "Frozen participant five-point receipt-integrity preflight. This is "
            "not the separate 193-sample measurement-ceiling capture protocol."
        ),
    )
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument(
        "--participant-session-label",
        default="self-development participant session",
    )
    parser.add_argument(
        "--calibration-manifest-label",
        default="linked calibration manifest",
    )
    parser.add_argument(
        "--model-artifact-label",
        default="linked calibration model artifact",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    parser.add_argument(
        "--target-overlap-tolerance",
        type=float,
        default=DEFAULT_TARGET_OVERLAP_TOLERANCE_SIGNED,
        help=(
            "Minimum allowed 2D Euclidean separation in signed normalized [-1, 1] "
            "screen coordinates; distances below the threshold overlap. The default "
            "0.2 equals 0.1 in [0, 1] viewport-fraction coordinates."
        ),
    )
    return parser.parse_args(argv)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    result = build_measurement_ceiling_result(
        participant_session_path=args.participant_session,
        calibration_session_metadata_path=args.calibration_session_metadata,
        calibration_manifest_path=args.calibration_manifest,
        model_artifact_path=args.model_artifact,
        line_gap_px=args.line_gap_px,
        median_word_width_px=args.median_word_width_px,
        participant_session_label=args.participant_session_label,
        calibration_manifest_label=args.calibration_manifest_label,
        model_artifact_label=args.model_artifact_label,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        target_overlap_tolerance=args.target_overlap_tolerance,
        analysis_protocol_path=args.analysis_protocol,
    )
    result_reference = f"results/{args.json_output.name}"
    _atomic_write(args.json_output, deterministic_json(result))
    _atomic_write(
        args.markdown_output,
        render_measurement_ceiling_markdown(
            result,
            result_reference=result_reference,
        ),
    )
    print(
        "WEBCAM_GAZE_MEASUREMENT_CEILING="
        f"status={result['status']} "
        f"geometry={result['measurement_status']['geometry']} "
        f"uncertainty={result['measurement_status']['uncertainty']} "
        f"eligible_claim={result['decision']['eligible_claim']} "
        f"evidence={result['evidence_class']} "
        f"json={args.json_output} markdown={args.markdown_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

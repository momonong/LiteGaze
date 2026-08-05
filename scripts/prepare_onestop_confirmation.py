"""Acquire and inspect the frozen OneStop confirmation archive without outcomes.

The inspector verifies the immutable source identity and parses only the CSV
header. It never parses, aggregates, prints, or serializes a data-row value.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://osf.io/download/xkgfz/"
EXPECTED_FILE_NAME = "ia_Paragraph_ordinary.csv.zip"
EXPECTED_SIZE_BYTES = 177_291_322
EXPECTED_SHA256 = (
    "8883478946ee52381e7057683c9e84dc69fcea9054acc34f0c900463a6b546e9"
)
DEFAULT_SOURCE_PATH = PROJECT_ROOT / "data/onestop/raw" / EXPECTED_FILE_NAME
DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "docs/experiments/results/2026-08-05-onestop-confirmation-source.json"
)

ANALYSIS_READ_COLUMNS = (
    "participant_id",
    "list_number",
    "question_preview",
    "article_batch",
    "trial_index",
    "practice_trial",
    "article_id",
    "paragraph_id",
    "difficulty_level",
    "repeated_reading_trial",
    "IA_ID",
    "IA_LABEL",
    "IA_DWELL_TIME",
    "IA_FIRST_RUN_DWELL_TIME",
    "IA_FIRST_FIXATION_DURATION",
)
OUTCOME_COLUMNS = (
    "IA_DWELL_TIME",
    "IA_FIRST_RUN_DWELL_TIME",
    "IA_FIRST_FIXATION_DURATION",
)
FORBIDDEN_ANALYSIS_COLUMNS = (
    "question",
    "onestopqa_question_id",
    "same_critical_span",
    "selected_answer",
    "selected_answer_position",
    "is_correct",
    "correct_answer_position",
    "answers_order",
    "answer_1",
    "answer_2",
    "answer_3",
    "answer_4",
    "auxiliary_span_type",
    "critical_span_indices",
    "distractor_span_indices",
    "word_length",
    "word_length_no_punctuation",
    "subtlex_frequency",
    "wordfreq_frequency",
    "gpt2_surprisal",
    "universal_pos",
    "ptb_pos",
    "head_word_index",
    "dependency_relation",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def download_source(path: Path) -> None:
    """Stream the frozen source to disk and accept it only after hash checking."""
    path = path.resolve()
    if path.exists():
        observed_hash = sha256(path)
        if observed_hash != EXPECTED_SHA256:
            raise RuntimeError(
                f"existing OneStop source hash mismatch: {observed_hash}"
            )
        if path.stat().st_size != EXPECTED_SIZE_BYTES:
            raise RuntimeError("existing OneStop source size mismatch")
        print(f"[source] verified existing archive: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "LexiGaze-confirmation/1.0"},
    )
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open(
            "wb"
        ) as output:
            while block := response.read(1024 * 1024):
                output.write(block)
                digest.update(block)
                size += len(block)
        observed_hash = digest.hexdigest()
        if size != EXPECTED_SIZE_BYTES or observed_hash != EXPECTED_SHA256:
            raise RuntimeError(
                "downloaded OneStop source identity mismatch: "
                f"size={size}, sha256={observed_hash}"
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"[source] downloaded and verified {size} bytes: {path}")


def inspect_archive(
    path: Path,
    *,
    expected_size: int = EXPECTED_SIZE_BYTES,
    expected_sha256: str = EXPECTED_SHA256,
) -> dict[str, Any]:
    """Return source/header identity without parsing any CSV data row."""
    path = path.resolve()
    observed_size = path.stat().st_size
    observed_hash = sha256(path)
    if observed_size != expected_size:
        raise RuntimeError(
            f"OneStop source size mismatch: {observed_size} != {expected_size}"
        )
    if observed_hash != expected_sha256:
        raise RuntimeError(
            f"OneStop source hash mismatch: {observed_hash} != {expected_sha256}"
        )

    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        csv_members = [
            item
            for item in members
            if item.filename.lower().endswith(".csv")
            and "__MACOSX" not in Path(item.filename).parts
            and not Path(item.filename).name.startswith("._")
        ]
        if len(csv_members) != 1:
            raise RuntimeError(
                f"expected exactly one CSV member, found {len(csv_members)}"
            )
        member = csv_members[0]
        with archive.open(member, "r") as source:
            header_bytes = source.readline()

    try:
        header = header_bytes.decode("utf-8-sig").rstrip("\r\n")
    except UnicodeDecodeError as exc:
        raise RuntimeError("OneStop CSV header is not UTF-8") from exc
    delimiter = "\t" if "\t" in header else ","
    columns = next(csv.reader([header], delimiter=delimiter))
    if not columns or any(not column for column in columns):
        raise RuntimeError("OneStop CSV header contains an empty column")
    if len(set(columns)) != len(columns):
        raise RuntimeError("OneStop CSV header contains duplicate columns")
    missing = sorted(set(ANALYSIS_READ_COLUMNS).difference(columns))
    if missing:
        raise RuntimeError(f"OneStop CSV is missing frozen columns: {missing}")

    return {
        "schema_version": 1,
        "protocol_id": "onestop-ordinary-advanced-confirmation-v1",
        "source": {
            "download_url": SOURCE_URL,
            "file_name": path.name,
            "sha256": observed_hash,
            "size_bytes": observed_size,
        },
        "archive": {
            "member_name": member.filename,
            "compressed_size_bytes": member.compress_size,
            "uncompressed_size_bytes": member.file_size,
            "delimiter": "tab" if delimiter == "\t" else "comma",
            "column_count": len(columns),
            "columns": columns,
        },
        "guardrails": {
            "analysis_read_columns": list(ANALYSIS_READ_COLUMNS),
            "outcome_columns_present_but_values_not_read": sorted(
                set(OUTCOME_COLUMNS).intersection(columns)
            ),
            "forbidden_analysis_columns_present_but_ignored": sorted(
                set(FORBIDDEN_ANALYSIS_COLUMNS).intersection(columns)
            ),
            "csv_data_rows_parsed": 0,
            "outcome_values_exposed": False,
            "schema_matches_frozen_protocol": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    if args.download:
        download_source(source)
    if not source.exists():
        raise SystemExit(f"source not found; rerun with --download: {source}")
    manifest = inspect_archive(source)
    _atomic_write_json(args.manifest.resolve(), manifest)
    print(
        "[schema] frozen source/header verified; "
        f"columns={manifest['archive']['column_count']}; rows_parsed=0"
    )
    print(f"[schema] manifest={args.manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

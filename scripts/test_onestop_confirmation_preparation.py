"""Offline tests for outcome-blind OneStop acquisition and schema inspection."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.prepare_onestop_confirmation import (
    ANALYSIS_READ_COLUMNS,
    inspect_archive,
)


class OneStopConfirmationPreparationTests(unittest.TestCase):
    def _archive(self, root: Path, *, columns: list[str]) -> Path:
        path = root / "source.zip"
        header = "\t".join(columns).encode("utf-8") + b"\n"
        # The inspector must not parse or expose this outcome-row canary.
        body = b"SENSITIVE-OUTCOME-CANARY\xff\n"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("ia_Paragraph_ordinary.csv", header + body)
            archive.writestr("__MACOSX/._ia_Paragraph_ordinary.csv", b"sidecar")
        return path

    def test_inspector_parses_only_header_and_records_ignored_fields(self) -> None:
        columns = [*ANALYSIS_READ_COLUMNS, "question", "gpt2_surprisal"]
        with tempfile.TemporaryDirectory() as temp_name:
            path = self._archive(Path(temp_name), columns=columns)
            source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = inspect_archive(
                path,
                expected_size=path.stat().st_size,
                expected_sha256=source_hash,
            )

        self.assertEqual(manifest["guardrails"]["csv_data_rows_parsed"], 0)
        self.assertFalse(manifest["guardrails"]["outcome_values_exposed"])
        self.assertIn(
            "question",
            manifest["guardrails"]["forbidden_analysis_columns_present_but_ignored"],
        )
        self.assertNotIn("SENSITIVE-OUTCOME-CANARY", repr(manifest))

    def test_inspector_rejects_missing_frozen_column(self) -> None:
        columns = list(ANALYSIS_READ_COLUMNS[:-1])
        with tempfile.TemporaryDirectory() as temp_name:
            path = self._archive(Path(temp_name), columns=columns)
            source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError, "missing frozen columns"):
                inspect_archive(
                    path,
                    expected_size=path.stat().st_size,
                    expected_sha256=source_hash,
                )

    def test_inspector_rejects_identity_mismatch_before_header(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            path = self._archive(
                Path(temp_name), columns=list(ANALYSIS_READ_COLUMNS)
            )
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                inspect_archive(
                    path,
                    expected_size=path.stat().st_size,
                    expected_sha256="0" * 64,
                )


if __name__ == "__main__":
    unittest.main()

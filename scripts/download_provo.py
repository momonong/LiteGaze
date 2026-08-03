"""Acquire the frozen PROVO eye-tracking CSV from the official OSF project."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data/provo/raw/Provo_Corpus-Eyetracking_Data.csv"
)
PROVO_URL = "https://osf.io/download/a32be/"
PROVO_OSF_PROJECT = "https://osf.io/sjefs/"
PROVO_FILENAME = "Provo_Corpus-Eyetracking_Data.csv"
PROVO_SIZE_BYTES = 69_662_713
PROVO_SHA256 = "38aedcb29bc9171009916eb2bcc2375729f104a2a1005c64a563da94b611b9e7"
CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_provo_file(
    path: Path,
    *,
    expected_size: int = PROVO_SIZE_BYTES,
    expected_sha256: str = PROVO_SHA256,
) -> dict[str, Any]:
    """Verify the official file identity and return a stable fingerprint."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"PROVO source file does not exist: {resolved}")
    size = resolved.stat().st_size
    if size != expected_size:
        raise ValueError(
            f"PROVO size mismatch: expected {expected_size}, found {size}"
        )
    sha256 = sha256_file(resolved)
    if sha256 != expected_sha256:
        raise ValueError(
            "PROVO SHA-256 mismatch: "
            f"expected {expected_sha256}, found {sha256}"
        )
    return {
        "path": resolved.as_posix(),
        "filename": resolved.name,
        "size_bytes": size,
        "sha256": sha256,
        "source_url": PROVO_URL,
        "osf_project": PROVO_OSF_PROJECT,
    }


def download_provo(destination: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Download once, verify fully, and atomically publish the source CSV."""
    resolved = destination.resolve()
    if resolved.exists():
        return verify_provo_file(resolved)

    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=resolved.parent,
            prefix=f".{resolved.name}.",
            suffix=".part",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            with requests.get(
                PROVO_URL,
                stream=True,
                timeout=(15, 120),
                headers={"User-Agent": "LexiGaze reproducibility pipeline"},
            ) as response:
                response.raise_for_status()
                for block in response.iter_content(chunk_size=CHUNK_SIZE):
                    if block:
                        handle.write(block)

        verify_provo_file(temporary)
        os.replace(temporary, resolved)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return verify_provo_file(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify an existing file without downloading it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fingerprint = (
        verify_provo_file(args.output)
        if args.check_only
        else download_provo(args.output)
    )
    print(
        "Verified PROVO source: "
        f"{fingerprint['size_bytes']:,} bytes, SHA-256 {fingerprint['sha256']}"
    )
    print(f"Path: {fingerprint['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

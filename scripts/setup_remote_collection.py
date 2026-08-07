#!/usr/bin/env python3
"""Provision the tunnel helper used by the readiness-gated study launcher."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
BIN_DIR = SCRIPTS_DIR / "bin"
BIN_DIR.mkdir(exist_ok=True)

SYSTEM_OS = platform.system().lower()
IS_WINDOWS = SYSTEM_OS == "windows"
if IS_WINDOWS:
    NGROK_EXE = BIN_DIR / "ngrok.exe"
    NGROK_URL = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip"
elif SYSTEM_OS == "linux":
    NGROK_EXE = BIN_DIR / "ngrok"
    NGROK_URL = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz"
else:
    NGROK_EXE = BIN_DIR / "ngrok"
    NGROK_URL = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-darwin-amd64.zip"


def _safe_archive_target(name: str) -> Path:
    target = (BIN_DIR / name).resolve()
    if target != BIN_DIR.resolve() and BIN_DIR.resolve() not in target.parents:
        raise ValueError("ngrok archive contains an unsafe path")
    return target


def download_ngrok() -> bool:
    if NGROK_EXE.exists():
        return True
    archive_name = "ngrok.tgz" if NGROK_URL.endswith(".tgz") else "ngrok.zip"
    archive_path = BIN_DIR / archive_name
    print(f"Downloading ngrok for {platform.system()} from its official distribution.")
    try:
        request = urllib.request.Request(
            NGROK_URL,
            headers={"User-Agent": "LexiGaze participant-study setup"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            archive_path.write_bytes(response.read())
        if archive_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as archive:
                for member in archive.infolist():
                    _safe_archive_target(member.filename)
                archive.extractall(BIN_DIR)
        else:
            with tarfile.open(archive_path, "r:gz") as archive:
                for member in archive.getmembers():
                    _safe_archive_target(member.name)
                archive.extractall(BIN_DIR)
        archive_path.unlink(missing_ok=True)
        if not IS_WINDOWS:
            os.chmod(NGROK_EXE, 0o755)
        return NGROK_EXE.is_file()
    except Exception as exc:
        archive_path.unlink(missing_ok=True)
        print(f"Error installing ngrok: {exc}")
        return False


def get_public_url() -> str | None:
    for _ in range(10):
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:4040/api/tunnels", timeout=2
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            for tunnel in payload.get("tunnels", []):
                if tunnel.get("proto") == "https":
                    return str(tunnel.get("public_url") or "") or None
        except Exception:
            pass
        time.sleep(1)
    return None


def start_tunnel() -> tuple[str | None, subprocess.Popen | None]:
    """Start ngrok with local request-body inspection explicitly disabled."""

    options: dict[str, object] = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if IS_WINDOWS:
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        process = subprocess.Popen(
            [str(NGROK_EXE), "http", "8080", "--inspect=false"],
            **options,
        )
    except Exception as exc:
        print(f"Failed to launch ngrok: {exc}")
        return None, None
    public_url = get_public_url()
    if not public_url:
        process.terminate()
        process.wait()
        return None, None
    return public_url, process


def main() -> int:
    print(
        "Direct remote collection is disabled. Run `python -X utf8 run.py "
        "--study-tunnel`; it refuses to open until participant pilot readiness "
        "passes.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

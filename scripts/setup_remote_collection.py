#!/usr/bin/env python3
# scripts/setup_remote_collection.py
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
BIN_DIR = SCRIPTS_DIR / "bin"
BIN_DIR.mkdir(exist_ok=True)
MAX_NGROK_ARCHIVE_BYTES = 100 * 1024 * 1024

# Auto-detect OS platform
SYSTEM_OS = platform.system().lower()
IS_WINDOWS = SYSTEM_OS == "windows"
IS_LINUX = SYSTEM_OS == "linux"

if IS_WINDOWS:
    NGROK_EXE = BIN_DIR / "ngrok.exe"
    NGROK_URL = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip"
elif IS_LINUX:
    NGROK_EXE = BIN_DIR / "ngrok"
    NGROK_URL = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz"
else:  # macOS / Darwin fallback
    NGROK_EXE = BIN_DIR / "ngrok"
    NGROK_URL = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-darwin-amd64.zip"


def _copy_limited(source, destination, maximum_bytes=MAX_NGROK_ARCHIVE_BYTES):
    copied = 0
    while True:
        chunk = source.read(1024 * 1024)
        if not chunk:
            return copied
        copied += len(chunk)
        if copied > maximum_bytes:
            raise RuntimeError(f"ngrok archive exceeds {maximum_bytes} bytes")
        destination.write(chunk)


def _extract_ngrok_binary(archive_path, destination=NGROK_EXE):
    """Extract only the expected executable, never archive-provided paths."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_name = destination.name
    temporary_destination = destination.with_name(f".{destination.name}.tmp")
    temporary_destination.unlink(missing_ok=True)

    try:
        if str(archive_path).endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as archive:
                candidates = [
                    member
                    for member in archive.infolist()
                    if not member.is_dir()
                    and Path(member.filename.replace("\\", "/")).name == expected_name
                ]
                if len(candidates) != 1:
                    raise RuntimeError(
                        f"ngrok archive must contain exactly one {expected_name} executable"
                    )
                with (
                    archive.open(candidates[0], "r") as source,
                    temporary_destination.open("wb") as output,
                ):
                    shutil.copyfileobj(source, output)
        else:
            with tarfile.open(archive_path, "r:gz") as archive:
                candidates = [
                    member
                    for member in archive.getmembers()
                    if member.isfile()
                    and Path(member.name.replace("\\", "/")).name == expected_name
                ]
                if len(candidates) != 1:
                    raise RuntimeError(
                        f"ngrok archive must contain exactly one {expected_name} executable"
                    )
                source = archive.extractfile(candidates[0])
                if source is None:
                    raise RuntimeError(f"unable to read {expected_name} from ngrok archive")
                with source, temporary_destination.open("wb") as output:
                    shutil.copyfileobj(source, output)

        os.replace(temporary_destination, destination)
    finally:
        temporary_destination.unlink(missing_ok=True)


def download_ngrok():
    if NGROK_EXE.exists():
        return True
    
    archive_name = "ngrok.zip" if not NGROK_URL.endswith(".tgz") else "ngrok.tgz"
    archive_path = BIN_DIR / archive_name
    print("==================================================")
    print(f"  Downloading ngrok for {platform.system()}...")
    print("==================================================")
    print(f"URL: {NGROK_URL}")
    try:
        req = urllib.request.Request(
            NGROK_URL,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with (
            urllib.request.urlopen(req, timeout=60) as response,
            archive_path.open('wb') as out_file,
        ):
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_NGROK_ARCHIVE_BYTES:
                raise RuntimeError(
                    f"ngrok archive exceeds {MAX_NGROK_ARCHIVE_BYTES} bytes"
                )
            _copy_limited(response, out_file)
            
        print(f"Extracting {archive_name}...")
        _extract_ngrok_binary(archive_path)
        
        # Add executable permissions for Linux / macOS
        if not IS_WINDOWS:
            os.chmod(str(NGROK_EXE), 0o755)
            
        print(f"ngrok successfully installed: {NGROK_EXE}")
        return True
    except Exception as exc:  # noqa: BLE001 - convert all installer failures to fail-closed False
        print(f"Error installing ngrok: {exc}")
        return False
    finally:
        archive_path.unlink(missing_ok=True)


class _NgrokLogMonitor:
    """Drain one ngrok process's logs and capture only its HTTPS URL."""

    def __init__(self, stream, local_port):
        self.public_url = None
        self.ready = threading.Event()
        self._stream = stream
        self._local_port = local_port
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self):
        for line in self._stream:
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            address = str(entry.get("addr", "")).rstrip("/")
            url = str(entry.get("url", ""))
            targets_local_port = address == str(self._local_port) or address.endswith(
                f":{self._local_port}"
            )
            if (
                entry.get("msg") == "started tunnel"
                and targets_local_port
                and url.startswith("https://")
            ):
                self.public_url = url
                self.ready.set()

    def join(self, timeout=1):
        self._thread.join(timeout=timeout)


def get_public_url(proc, monitor, timeout=10):
    print("Waiting for this ngrok process to report its HTTPS URL...")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if monitor.ready.wait(timeout=0.1):
            return monitor.public_url
        if proc.poll() is not None:
            return None
    return None


def stop_tunnel(proc):
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    monitor = getattr(proc, "_lexigaze_log_monitor", None)
    if monitor is not None:
        monitor.join(timeout=1)


def start_tunnel(local_port=8080):
    print("==================================================")
    print(f"  Starting ngrok Tunnel on Port {local_port}...")
    print("==================================================")
    
    proc = None
    try:
        creation_flags = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
        proc = subprocess.Popen(
            [
                str(NGROK_EXE),
                "http",
                str(local_port),
                "--inspect=false",
                "--log=stdout",
                "--log-format=json",
                "--log-level=info",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creation_flags,
        )
        if proc.stdout is None:
            raise RuntimeError("ngrok log stream was not created")
        monitor = _NgrokLogMonitor(proc.stdout, local_port)
        proc._lexigaze_log_monitor = monitor
        print("ngrok process started in the background.")
    except Exception as exc:  # noqa: BLE001 - clean up a partially started child on any error
        stop_tunnel(proc)
        print(f"Failed to launch ngrok process: {exc}")
        return None, None

    public_url = get_public_url(proc, monitor)
    if not public_url:
        print("\n[!] Error: Failed to retrieve ngrok public URL.")
        print("Please verify that:")
        print(f"1. Your local Flask server is running on port {local_port}.")
        cmd_str = (
            "scripts/bin/ngrok config add-authtoken <your-token>"
            if not IS_WINDOWS
            else "scripts\\bin\\ngrok.exe config add-authtoken <your-token>"
        )
        print(f"2. You have registered your ngrok authtoken by running:\n   {cmd_str}")
        print("   (Get a free token at https://dashboard.ngrok.com)\n")
        stop_tunnel(proc)
        return None, None

    return public_url, proc


def main():
    print(
        "Standalone tunnel startup is disabled because it cannot guarantee that "
        "the Flask app has authentication and resource guardrails enabled.",
        file=sys.stderr,
    )
    print(
        "Use: uv run python -X utf8 run.py --tunnel",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

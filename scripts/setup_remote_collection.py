#!/usr/bin/env python3
# scripts/setup_remote_collection.py
import os
import sys
import json
import zipfile
import tarfile
import platform
import urllib.request
import subprocess
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
BIN_DIR = SCRIPTS_DIR / "bin"
BIN_DIR.mkdir(exist_ok=True)

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
else: # macOS / Darwin fallback
    NGROK_EXE = BIN_DIR / "ngrok"
    NGROK_URL = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-darwin-amd64.zip"

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
        with urllib.request.urlopen(req) as response, open(archive_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
            
        print(f"Extracting {archive_name}...")
        if archive_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(BIN_DIR)
        else: # .tgz / tar.gz
            with tarfile.open(archive_path, "r:gz") as tar_ref:
                tar_ref.extractall(BIN_DIR)
                
        archive_path.unlink()
        
        # Add executable permissions for Linux / macOS
        if not IS_WINDOWS:
            os.chmod(str(NGROK_EXE), 0o755)
            
        print(f"ngrok successfully installed: {NGROK_EXE}")
        return True
    except Exception as exc:
        print(f"Error installing ngrok: {exc}")
        return False

def get_public_url():
    print("Querying ngrok local API for tunnel URL...")
    for _ in range(10):
        try:
            req = urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels")
            data = json.loads(req.read().decode())
            tunnels = data.get("tunnels", [])
            for t in tunnels:
                if t.get("proto") == "https":
                    return t.get("public_url")
        except Exception:
            pass
        time.sleep(1)
    return None

def start_tunnel():
    print("==================================================")
    print("  Starting ngrok Tunnel on Port 8080...")
    print("==================================================")
    
    # Start ngrok process
    try:
        proc = subprocess.Popen(
            [str(NGROK_EXE), "http", "8080"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("ngrok process started in the background.")
    except Exception as exc:
        print(f"Failed to launch ngrok process: {exc}")
        return None, None

    public_url = get_public_url()
    if not public_url:
        print("\n[!] Error: Failed to retrieve ngrok public URL.")
        print("Please verify that:")
        print("1. Your local Flask server is running on port 8080.")
        cmd_str = f"scripts/bin/ngrok config add-authtoken <your-token>" if not IS_WINDOWS else f"scripts\\bin\\ngrok.exe config add-authtoken <your-token>"
        print(f"2. You have registered your ngrok authtoken by running:\n   {cmd_str}")
        print("   (Get a free token at https://dashboard.ngrok.com)\n")
        proc.terminate()
        return None, None

    return public_url, proc

def main():
    if not download_ngrok():
        sys.exit(1)
        
    public_url, proc = start_tunnel()
    if not public_url:
        sys.exit(1)
        
    calibration_url = f"{public_url}/gaze"
    reading_url = f"{public_url}/"
    qr_image_url = f"https://api.qrserver.com/v1/create-qr-code/?size=250x250&data={calibration_url}"
    
    print("\n" + "="*50)
    print(" 🎉 LexiGaze Remote Tunnel is ONLINE!")
    print("="*50)
    print(f" Local Port  : 8080")
    print(f" Public Web  : {public_url}")
    print(f" Calibration : {calibration_url}")
    print(f" Reading     : {reading_url}")
    print("-"*50)
    print(" 📱 Scan QR Code to open calibration on Laptop/Phone:")
    print(f" QR Code URL : {qr_image_url}")
    print("="*50)
    print(" Press Ctrl + C in this terminal to stop the tunnel.\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping ngrok tunnel...")
        proc.terminate()
        proc.wait()
        print("Tunnel offline.")

if __name__ == "__main__":
    main()

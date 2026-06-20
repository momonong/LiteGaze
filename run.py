# IMPORTANT: Start with `python -X utf8 run.py` to prevent UnicodeEncodeError
# crashes on Windows CP950 when library code prints non-Big5 characters.
import webbrowser
import sys
from pathlib import Path

import argparse

sys.dont_write_bytecode = True

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data'

# Import and build Flask application factory
from web import create_app
app = create_app()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run LexiGaze integration server")
    parser.add_argument("--tunnel", "-t", action="store_true", help="Start an ngrok HTTPS tunnel for remote webcam collection")
    args = parser.parse_args()

    ngrok_proc = None
    if args.tunnel:
        sys.path.insert(0, str(ROOT))
        try:
            from scripts.setup_remote_collection import download_ngrok, start_tunnel
            print("Configuring ngrok tunnel...")
            if download_ngrok():
                public_url, ngrok_proc = start_tunnel()
                if public_url:
                    calibration_url = f"{public_url}/gaze"
                    reading_url = f"{public_url}/"
                    qr_image_url = f"https://api.qrserver.com/v1/create-qr-code/?size=250x250&data={calibration_url}"
                    print("=" * 48)
                    print(" 🎉 LexiGaze Remote Tunnel is ONLINE!")
                    print("=" * 48)
                    print(f" Public Web  : {public_url}")
                    print(f" Calibration : {calibration_url}")
                    print(f" Reading     : {reading_url}")
                    print(f" QR Code URL : {qr_image_url}")
                    print("=" * 48)
                else:
                    print("[!] Failed to start ngrok tunnel. Running in local-only mode.")
            else:
                print("[!] Failed to download ngrok. Running in local-only mode.")
        except Exception as e:
            print(f"[!] Error setting up ngrok tunnel: {e}")

    url = 'http://localhost:8080/'
    print('=' * 48)
    print('  文件座標擷取工具  —  Flask Backend (Refactored)')
    print('=' * 48)
    print(f'  網址  : {url}')
    print(f'  資料  : {DATA_DIR}')
    print(f'  API   : http://localhost:8080/api/sessions')
    print(f'  認知負荷 : http://localhost:8080/api/cognitive/health')
    print(f'  停止  : Ctrl + C')
    print('=' * 48)
    
    if not args.tunnel:
        webbrowser.open(url)

    try:
        app.run(host='localhost', port=8080, debug=False)
    finally:
        if ngrok_proc:
            print("\nShutting down ngrok tunnel...")
            ngrok_proc.terminate()
            ngrok_proc.wait()
            print("Tunnel offline.")

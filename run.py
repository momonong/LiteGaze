# IMPORTANT: Start with `python -X utf8 run.py` to prevent UnicodeEncodeError
# crashes on Windows CP950 when library code prints non-Big5 characters.
import argparse
import os
import sys
import webbrowser
from pathlib import Path

sys.dont_write_bytecode = True

# Fix HF_HOME if it's set to a Windows drive path on a non-Windows OS

if os.name != 'nt' and os.environ.get("HF_HOME"):
    _hf_home = os.environ["HF_HOME"]
    if ":" in _hf_home or _hf_home.startswith("D:") or _hf_home.startswith("C:"):
        del os.environ["HF_HOME"]

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data'

# Import and build Flask application factory
from web import create_app

app = create_app()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run LexiGaze integration server")
    tunnel_group = parser.add_mutually_exclusive_group()
    tunnel_group.add_argument(
        "--study-tunnel",
        action="store_true",
        help="Start a readiness-gated participant-only HTTPS tunnel",
    )
    tunnel_group.add_argument(
        "--tunnel",
        "-t",
        action="store_true",
        help="Disabled legacy option that exposed researcher endpoints",
    )
    args = parser.parse_args()

    if args.tunnel:
        parser.error(
            "--tunnel is disabled because it exposes the researcher UI; "
            "use --study-tunnel after the pilot readiness audit passes"
        )

    ngrok_proc = None
    if args.study_tunnel:
        os.environ["LEXIGAZE_PUBLIC_STUDY_MODE"] = "1"
        sys.path.insert(0, str(ROOT))
        try:
            from scripts.setup_remote_collection import download_ngrok, start_tunnel
            print("Configuring ngrok tunnel...")
            if download_ngrok():
                public_url, ngrok_proc = start_tunnel()
                if public_url:
                    os.environ["LEXIGAZE_PUBLIC_BASE_URL"] = public_url
                    from scripts.audit_participant_study_readiness import audit

                    readiness = audit()
                    if not readiness["pilot_ready"]:
                        missing = ", ".join(readiness["pilot_missing_requirements"])
                        raise RuntimeError(
                            "participant tunnel refused; pilot readiness is incomplete: "
                            + missing
                        )
                    app = create_app()
                    study_url = f"{public_url}/study"
                    print("=" * 48)
                    print(" 🎉 LexiGaze Remote Tunnel is ONLINE!")
                    print("=" * 48)
                    print(f" Participant study: {study_url}")
                    print(" Researcher pages and admin APIs are blocked.")
                    print("=" * 48)
                else:
                    raise RuntimeError("failed to start the participant tunnel")
            else:
                raise RuntimeError("failed to provision ngrok")
        except Exception as e:
            if ngrok_proc:
                ngrok_proc.terminate()
                ngrok_proc.wait()
            raise SystemExit(f"[!] Participant tunnel was not opened: {e}") from e

    url = 'http://localhost:8080/'
    print('=' * 48)
    print('  文件座標擷取工具  —  Flask Backend (Refactored)')
    print('=' * 48)
    print(f'  網址  : {url}')
    print(f'  資料  : {DATA_DIR}')
    print('  API   : http://localhost:8080/api/sessions')
    print('  認知負荷 : http://localhost:8080/api/cognitive/health')
    print('  停止  : Ctrl + C')
    print('=' * 48)
    
    if not args.study_tunnel:
        webbrowser.open(url)

    try:
        if args.study_tunnel:
            from waitress import serve

            serve(app, host="127.0.0.1", port=8080, threads=4)
        else:
            app.run(host='localhost', port=8080, debug=False)
    finally:
        if ngrok_proc:
            print("\nShutting down ngrok tunnel...")
            ngrok_proc.terminate()
            ngrok_proc.wait()
            print("Tunnel offline.")

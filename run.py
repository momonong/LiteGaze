# IMPORTANT: Start with `python -X utf8 run.py` to prevent UnicodeEncodeError
# crashes on Windows CP950 when library code prints non-Big5 characters.
import argparse
import secrets
import sys
import threading
import webbrowser
from pathlib import Path

sys.dont_write_bytecode = True

# Fix HF_HOME if it's set to a Windows drive path on a non-Windows OS
import os

if os.name != 'nt' and os.environ.get("HF_HOME"):
    _hf_home = os.environ["HF_HOME"]
    if ":" in _hf_home or _hf_home.startswith(("D:", "C:")):
        del os.environ["HF_HOME"]

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data'

# Import and build Flask application factory
from web import create_app
from web.security import build_tunnel_bootstrap_url, validate_tunnel_token

# WSGI imports keep the historical local-only behavior.  The CLI creates its
# runtime app after parsing --tunnel so public mode can never reuse this app.
app = create_app() if __name__ != '__main__' else None


def _resolve_tunnel_token():
    from dotenv import load_dotenv

    load_dotenv(ROOT / '.env', override=False)
    configured = os.environ.get('LEXIGAZE_TUNNEL_TOKEN')
    if configured:
        return validate_tunnel_token(configured), False
    return secrets.token_urlsafe(32), True


def _print_runtime_banner():
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


def _create_tunnel_server(runtime_app):
    """Bind the protected app before creating any public forwarding route."""
    from werkzeug.serving import make_server

    return make_server('127.0.0.1', 8080, runtime_app, threaded=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run LexiGaze integration server")
    parser.add_argument(
        "--tunnel",
        "-t",
        action="store_true",
        help="Start an authenticated ngrok HTTPS tunnel for remote webcam collection",
    )
    args = parser.parse_args(argv)

    tunnel_token = None
    generated_token = False
    if args.tunnel:
        try:
            tunnel_token, generated_token = _resolve_tunnel_token()
        except RuntimeError as exc:
            print(f"[security] Refusing to start tunnel: {exc}", file=sys.stderr)
            return 2

    try:
        runtime_app = create_app(tunnel_mode=args.tunnel, tunnel_token=tunnel_token)
    except RuntimeError as exc:
        print(f"[security] Refusing to start server: {exc}", file=sys.stderr)
        return 2

    if not args.tunnel:
        _print_runtime_banner()
        webbrowser.open('http://localhost:8080/')
        runtime_app.run(host='127.0.0.1', port=8080, debug=False)
        return 0

    sys.path.insert(0, str(ROOT))
    from scripts.setup_remote_collection import (
        download_ngrok,
        start_tunnel,
        stop_tunnel,
    )

    try:
        tunnel_server = _create_tunnel_server(runtime_app)
    except (OSError, RuntimeError, SystemExit) as exc:
        print(
            "[security] Refusing to open tunnel because the protected server "
            "could not bind port 8080.",
            file=sys.stderr,
        )
        return int(getattr(exc, "code", 1) or 1)

    server_thread = threading.Thread(target=tunnel_server.serve_forever, daemon=True)
    try:
        server_thread.start()
    except RuntimeError as exc:
        tunnel_server.server_close()
        print(f"[security] Protected server thread failed to start: {exc}", file=sys.stderr)
        return 1
    ngrok_proc = None
    try:
        print("Configuring authenticated ngrok tunnel...")
        if not download_ngrok():
            print(
                "[tunnel] ngrok installation failed; no server was exposed.",
                file=sys.stderr,
            )
            return 1
        public_url, ngrok_proc = start_tunnel()
        if not public_url:
            print(
                "[tunnel] ngrok startup failed; no local-only fallback was started.",
                file=sys.stderr,
            )
            return 1

        calibration_url = build_tunnel_bootstrap_url(public_url, '/gaze', tunnel_token)
        reading_url = build_tunnel_bootstrap_url(public_url, '/', tunnel_token)
        print('=' * 60)
        print('  LexiGaze authenticated remote tunnel is ONLINE')
        print('=' * 60)
        print(f'  Calibration : {calibration_url}')
        print(f'  Reading     : {reading_url}')
        print(
            '  Share these links privately; each link contains the access secret '
            'in its URL fragment.'
        )
        print('  No access link is sent to an external QR-code service.')
        if generated_token:
            print('  Access secret: ephemeral (invalid after this server stops)')
        else:
            print('  Access secret: loaded from LEXIGAZE_TUNNEL_TOKEN')
        print('=' * 60)
        calibration_url = None
        reading_url = None
        tunnel_token = None
        _print_runtime_banner()

        while server_thread.is_alive():
            server_thread.join(timeout=1)
        print("[tunnel] Protected server stopped unexpectedly.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # noqa: BLE001 - startup must fail closed on every setup error
        print(f"[tunnel] Setup failed closed: {exc}", file=sys.stderr)
        return 1
    finally:
        if ngrok_proc:
            print("\nShutting down ngrok tunnel...")
            stop_tunnel(ngrok_proc)
            print("Tunnel offline.")
        if server_thread.is_alive():
            tunnel_server.shutdown()
        tunnel_server.server_close()
        server_thread.join(timeout=5)

if __name__ == '__main__':
    raise SystemExit(main())

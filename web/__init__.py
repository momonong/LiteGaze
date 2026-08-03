import json
import mimetypes
import os
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

from web.security import TunnelSecurityConfig, install_tunnel_security

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
DATA_DIR.mkdir(exist_ok=True)
SHENGWEN_STATIC = ROOT / 'archive' / 'shengwen' / 'web' / 'static'

mimetypes.add_type('text/javascript', '.js')

LOCAL_MAX_CONTENT_LENGTH = 500 * 1024 * 1024


def _bounded_env_int(name, default, minimum, maximum):
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value == "":
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}")
    return value


def _bounded_config_int(app, name, minimum, maximum):
    value = app.config[name]
    if isinstance(value, bool):
        raise RuntimeError(f"{name} must be an integer")  # noqa: TRY004
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if not minimum <= parsed <= maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def create_app(test_config=None, *, tunnel_mode=False, tunnel_token=None):
    if tunnel_mode:
        load_dotenv(ROOT / '.env', override=False)

    # Flask app initialization with static_folder pointing to 'static' and templates to 'templates'
    app = Flask(__name__, static_folder='static', template_folder='templates')
    app.config.from_mapping(
        MAX_CONTENT_LENGTH=LOCAL_MAX_CONTENT_LENGTH,
        TUNNEL_MODE=bool(tunnel_mode),
        TUNNEL_MAX_CONTENT_LENGTH=64 * 1024 * 1024,
        TUNNEL_MUTATION_LIMIT=300,
        TUNNEL_REALTIME_LIMIT=900,
        TUNNEL_EXPENSIVE_LIMIT=12,
        TUNNEL_AUTH_ATTEMPT_LIMIT=10,
        TUNNEL_RATE_WINDOW_SECONDS=60,
        TUNNEL_EXPENSIVE_CONCURRENCY=1,
        TUNNEL_SESSION_TTL_SECONDS=8 * 60 * 60,
    )
    if tunnel_mode:
        app.config.update(
            TUNNEL_MAX_CONTENT_LENGTH=(
                _bounded_env_int('LEXIGAZE_TUNNEL_MAX_UPLOAD_MB', 64, 1, 256)
                * 1024
                * 1024
            ),
            TUNNEL_MUTATION_LIMIT=_bounded_env_int(
                'LEXIGAZE_TUNNEL_MUTATIONS_PER_MINUTE', 300, 1, 10000
            ),
            TUNNEL_REALTIME_LIMIT=_bounded_env_int(
                'LEXIGAZE_TUNNEL_PREDICTIONS_PER_MINUTE', 900, 1, 3600
            ),
            TUNNEL_EXPENSIVE_LIMIT=_bounded_env_int(
                'LEXIGAZE_TUNNEL_EXPENSIVE_PER_MINUTE', 12, 1, 300
            ),
            TUNNEL_AUTH_ATTEMPT_LIMIT=_bounded_env_int(
                'LEXIGAZE_TUNNEL_AUTH_ATTEMPTS_PER_MINUTE', 10, 1, 300
            ),
            TUNNEL_EXPENSIVE_CONCURRENCY=_bounded_env_int(
                'LEXIGAZE_TUNNEL_EXPENSIVE_CONCURRENCY', 1, 1, 8
            ),
            TUNNEL_SESSION_TTL_SECONDS=_bounded_env_int(
                'LEXIGAZE_TUNNEL_SESSION_TTL_SECONDS', 8 * 60 * 60, 300, 24 * 60 * 60
            ),
        )
    if test_config:
        app.config.update(test_config)
    
    # Register blueprints from submodules
    from web.routes.cognitive import cognitive_bp
    from web.routes.demo import demo_bp
    from web.routes.fusion import fusion_bp
    from web.routes.gaze import gaze_api_bp, gaze_bp
    from web.routes.gaze_video import gaze_video_bp
    from web.routes.inspector import inspector_bp
    
    app.register_blueprint(gaze_bp)
    app.register_blueprint(gaze_api_bp)
    app.register_blueprint(cognitive_bp)
    app.register_blueprint(fusion_bp)
    app.register_blueprint(demo_bp)
    app.register_blueprint(inspector_bp)
    app.register_blueprint(gaze_video_bp)
    
    # ── Static files & Template views ─────────────────────────────────────────
    @app.route('/')
    def index():
        return render_template('word_track.html')

    @app.route('/gaze')
    def gaze_page():
        return render_template('gaze_page.html')

    @app.route('/gaze_static/<path:filename>')
    def gaze_static_file(filename):
        return send_from_directory(SHENGWEN_STATIC, filename)

    @app.route('/examples/<path:filename>')
    def examples_file(filename):
        return send_from_directory(ROOT / 'examples', filename)

    # ── API: Health check ────────────────────────────────────────────────────
    @app.route('/api/ping')
    def ping():
        return jsonify({'ok': True, 'sessions': len(list(DATA_DIR.glob('*.json')))})

    # ── API: List sessions (summary only, no items) ──────────────────────────
    @app.route('/api/sessions', methods=['GET'])
    def list_sessions():
        sessions = []
        for f in DATA_DIR.glob('*.json'):
            try:
                with open(f, encoding='utf-8') as fp:
                    d = json.load(fp)
                sessions.append({
                    'id':         d['id'],
                    'filename':   d['filename'],
                    'filetype':   d.get('filetype', ''),
                    'created_at': d['created_at'],
                    'item_count': d['item_count'],
                })
            except Exception:  # noqa: BLE001, S110 - preserve legacy corrupt-file skip
                pass
        sessions.sort(key=lambda x: x['created_at'], reverse=True)
        return jsonify(sessions)

    # ── API: Save a session ──────────────────────────────────────────────────
    @app.route('/api/sessions', methods=['POST'])
    def save_session():
        body       = request.get_json(force=True)
        session_id = str(uuid.uuid4())
        items      = body.get('items', [])
        session    = {
            'id':         session_id,
            'filename':   body.get('filename', 'unknown'),
            'filetype':   body.get('filetype', ''),
            'created_at': datetime.now().isoformat(),  # noqa: DTZ005 - preserve legacy schema
            'item_count': len(items),
            'items':      items,
        }
        path = DATA_DIR / f'{session_id}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(session, f, ensure_ascii=False, indent=2)
        return jsonify({'id': session_id, 'created_at': session['created_at']}), 201

    # ── API: Get full session data ───────────────────────────────────────────
    @app.route('/api/sessions/<session_id>', methods=['GET'])
    def get_session(session_id):
        path = DATA_DIR / f'{session_id}.json'
        if not path.exists():
            return jsonify({'error': 'Not found'}), 404
        with open(path, encoding='utf-8') as f:
            return jsonify(json.load(f))

    # ── API: Delete a session ────────────────────────────────────────────────
    @app.route('/api/sessions/<session_id>', methods=['DELETE'])
    def delete_session(session_id):
        path = DATA_DIR / f'{session_id}.json'
        if path.exists():
            path.unlink()
            return jsonify({'ok': True})
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(_error):
        return jsonify({
            'ok': False,
            'error': 'request_too_large',
            'max_bytes': app.config['MAX_CONTENT_LENGTH'],
        }), 413

    if app.config['TUNNEL_MODE']:
        app.config['MAX_CONTENT_LENGTH'] = _bounded_config_int(
            app,
            'TUNNEL_MAX_CONTENT_LENGTH',
            1,
            256 * 1024 * 1024,
        )
        security_config = TunnelSecurityConfig(
            max_content_length=app.config['MAX_CONTENT_LENGTH'],
            mutation_limit=_bounded_config_int(app, 'TUNNEL_MUTATION_LIMIT', 1, 10000),
            realtime_limit=_bounded_config_int(app, 'TUNNEL_REALTIME_LIMIT', 1, 3600),
            expensive_limit=_bounded_config_int(app, 'TUNNEL_EXPENSIVE_LIMIT', 1, 300),
            auth_attempt_limit=_bounded_config_int(app, 'TUNNEL_AUTH_ATTEMPT_LIMIT', 1, 300),
            rate_window_seconds=_bounded_config_int(app, 'TUNNEL_RATE_WINDOW_SECONDS', 1, 3600),
            expensive_concurrency=_bounded_config_int(
                app, 'TUNNEL_EXPENSIVE_CONCURRENCY', 1, 8
            ),
            session_ttl_seconds=_bounded_config_int(
                app, 'TUNNEL_SESSION_TTL_SECONDS', 300, 24 * 60 * 60
            ),
        )
        install_tunnel_security(
            app,
            tunnel_token if tunnel_token is not None else os.environ.get('LEXIGAZE_TUNNEL_TOKEN'),
            security_config,
        )

    return app

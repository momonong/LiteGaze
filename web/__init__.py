import mimetypes
from collections.abc import Mapping
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

from web.session_store import (
    DEFAULT_MAX_SESSION_BYTES,
    DEFAULT_MAX_SESSION_ITEMS,
    CorruptSession,
    InvalidSessionId,
    InvalidSessionPayload,
    SessionDeleteError,
    SessionNotFound,
    SessionStore,
    SessionTooLarge,
    SessionWriteError,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
SHENGWEN_STATIC = ROOT / 'archive' / 'shengwen' / 'web' / 'static'

DEFAULT_BLUEPRINTS = (
    'gaze',
    'cognitive',
    'fusion',
    'demo',
    'inspector',
    'gaze_video',
)

mimetypes.add_type('text/javascript', '.js')


def _register_blueprints(app: Flask) -> None:
    enabled = app.config.get('LEXIGAZE_BLUEPRINTS')
    if enabled is None:
        enabled = DEFAULT_BLUEPRINTS
    elif isinstance(enabled, str):
        enabled = (enabled,)
    else:
        enabled = tuple(enabled)

    unknown = sorted(set(enabled) - set(DEFAULT_BLUEPRINTS))
    if unknown:
        raise ValueError(f"Unknown LexiGaze blueprints: {', '.join(unknown)}")

    # Import only the requested subsystems. The default still registers the
    # complete application, while focused tests and lightweight deployments
    # avoid importing PyTorch/OpenCV routes they never use.
    if 'gaze' in enabled:
        from web.routes.gaze import gaze_api_bp, gaze_bp

        app.register_blueprint(gaze_bp)
        app.register_blueprint(gaze_api_bp)
    if 'cognitive' in enabled:
        from web.routes.cognitive import cognitive_bp

        app.register_blueprint(cognitive_bp)
    if 'fusion' in enabled:
        from web.routes.fusion import fusion_bp

        app.register_blueprint(fusion_bp)
    if 'demo' in enabled:
        from web.routes.demo import demo_bp

        app.register_blueprint(demo_bp)
    if 'inspector' in enabled:
        from web.routes.inspector import inspector_bp

        app.register_blueprint(inspector_bp)
    if 'gaze_video' in enabled:
        from web.routes.gaze_video import gaze_video_bp

        app.register_blueprint(gaze_video_bp)


def create_app(config: Mapping[str, object] | None = None):
    # Flask app initialization with static_folder pointing to 'static' and templates to 'templates'
    app = Flask(__name__, static_folder='static', template_folder='templates')
    app.config.from_mapping(
        MAX_CONTENT_LENGTH=500 * 1024 * 1024,  # 500 MB upload limit
        LEXIGAZE_BLUEPRINTS=None,
        LEXIGAZE_DATA_DIR=DATA_DIR,
        LEXIGAZE_SESSION_MAX_BYTES=DEFAULT_MAX_SESSION_BYTES,
        LEXIGAZE_SESSION_MAX_ITEMS=DEFAULT_MAX_SESSION_ITEMS,
    )
    if config:
        app.config.update(config)

    _register_blueprints(app)
    session_store = SessionStore(
        app.config['LEXIGAZE_DATA_DIR'],
        max_bytes=app.config['LEXIGAZE_SESSION_MAX_BYTES'],
        max_items=app.config['LEXIGAZE_SESSION_MAX_ITEMS'],
        logger=app.logger,
    )
    app.extensions['lexigaze_session_store'] = session_store
    
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
        scan = session_store.scan()
        return jsonify({
            'ok': True,
            'sessions': len(scan.sessions),
            'corrupt_sessions': scan.corrupt_count,
        })

    # ── API: List sessions (summary only, no items) ──────────────────────────
    @app.route('/api/sessions', methods=['GET'])
    def list_sessions():
        return jsonify(session_store.scan().sessions)

    # ── API: Save a session ──────────────────────────────────────────────────
    @app.route('/api/sessions', methods=['POST'])
    def save_session():
        if not request.is_json:
            return _json_error(
                'unsupported_media_type',
                'Content-Type must be application/json',
                415,
            )
        if (
            request.content_length is not None
            and request.content_length > session_store.max_bytes
        ):
            return _json_error(
                'session_too_large',
                f'Request cannot exceed {session_store.max_bytes} bytes',
                413,
            )
        try:
            body = request.get_json()
        except (BadRequest, RecursionError):
            return _json_error('invalid_json', 'Request body is not valid JSON', 400)

        try:
            result = session_store.create(body)
        except SessionTooLarge as exc:
            return _json_error('session_too_large', str(exc), 413)
        except InvalidSessionPayload as exc:
            return _json_error('invalid_payload', str(exc), 400)
        except SessionWriteError:
            app.logger.exception('Session write failed')
            return _json_error(
                'storage_error', 'Session could not be persisted', 500
            )
        return jsonify(result), 201

    # ── API: Get full session data ───────────────────────────────────────────
    @app.route('/api/sessions/<session_id>', methods=['GET'])
    def get_session(session_id):
        try:
            session = session_store.get(session_id)
        except InvalidSessionId as exc:
            return _json_error('invalid_session_id', str(exc), 400)
        except SessionNotFound:
            return _json_error('not_found', 'Session not found', 404)
        except CorruptSession as exc:
            app.logger.error('Stored session %s is corrupt: %s', session_id, exc)
            return _json_error('corrupt_session', 'Stored session is corrupt', 500)
        return jsonify(session)

    # ── API: Delete a session ────────────────────────────────────────────────
    @app.route('/api/sessions/<session_id>', methods=['DELETE'])
    def delete_session(session_id):
        try:
            session_store.delete(session_id)
        except InvalidSessionId as exc:
            return _json_error('invalid_session_id', str(exc), 400)
        except SessionNotFound:
            return _json_error('not_found', 'Session not found', 404)
        except SessionDeleteError:
            app.logger.exception('Session delete failed')
            return _json_error('storage_error', 'Session could not be deleted', 500)
        return jsonify({'ok': True})

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(_error):
        return _json_error(
            'request_too_large', 'Request exceeds the configured size limit', 413
        )
        
    return app


def _json_error(code: str, message: str, status: int):
    return jsonify({'error': code, 'message': message}), status

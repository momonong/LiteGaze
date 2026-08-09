import json
import mimetypes
import os
import secrets
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
SHENGWEN_STATIC = ROOT / "archive" / "shengwen" / "web" / "static"

DEFAULT_BLUEPRINTS = (
    "study",
    "gaze",
    "cognitive",
    "fusion",
    "demo",
    "inspector",
    "gaze_video",
)

mimetypes.add_type("text/javascript", ".js")


def _register_blueprints(app: Flask) -> None:
    enabled = app.config.get("LEXIGAZE_BLUEPRINTS")
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
    if "gaze" in enabled:
        from web.routes.gaze import gaze_api_bp, gaze_bp

        app.register_blueprint(gaze_bp)
        app.register_blueprint(gaze_api_bp)
    if "study" in enabled:
        from web.routes.study import study_bp

        app.register_blueprint(study_bp)
    if "cognitive" in enabled:
        from web.routes.cognitive import cognitive_bp

        app.register_blueprint(cognitive_bp)
    if "fusion" in enabled:
        from web.routes.fusion import fusion_bp

        app.register_blueprint(fusion_bp)
    if "demo" in enabled:
        from web.routes.demo import demo_bp

        app.register_blueprint(demo_bp)
    if "inspector" in enabled:
        from web.routes.inspector import inspector_bp

        app.register_blueprint(inspector_bp)
    if "gaze_video" in enabled:
        from web.routes.gaze_video import gaze_video_bp

        app.register_blueprint(gaze_video_bp)


def create_app(config: Mapping[str, object] | None = None):
    # Flask app initialization with static_folder pointing to 'static' and templates to 'templates'
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload limit
    app.config["LEXIGAZE_BLUEPRINTS"] = None
    if config:
        app.config.update(config)

    def configured(name: str, default: str = "") -> str:
        if name in app.config and app.config.get(name) is not None:
            return str(app.config.get(name) or default).strip()
        return str(os.environ.get(name, default)).strip()

    def public_study_mode() -> bool:
        return configured("LEXIGAZE_PUBLIC_STUDY_MODE").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    if public_study_mode():
        self_development_video = configured(
            "LEXIGAZE_UNENCRYPTED_SELF_DEVELOPMENT"
        ).lower() in {"1", "true", "yes", "on"}
        upload_limit = (
            80 * 1024 * 1024 if self_development_video else 16 * 1024 * 1024
        )
        app.config["MAX_CONTENT_LENGTH"] = min(
            int(app.config["MAX_CONTENT_LENGTH"]),
            upload_limit,
        )

    @app.before_request
    def restrict_public_study_surface():
        """Expose only participant endpoints when the public study mode is on."""

        if not public_study_mode():
            return None

        expected_key = configured("LEXIGAZE_RESEARCHER_API_KEY")
        supplied_key = request.headers.get("X-Lexigaze-Researcher-Key", "")
        if expected_key and secrets.compare_digest(expected_key, supplied_key):
            return None

        path = request.path
        if path == "/":
            if request.args.get("study") == "1":
                return redirect("/study/assessment?study=1")
            return redirect("/study")
        if path == "/gaze" and request.args.get("study") != "1":
            return redirect("/study")

        public_pages = {
            "/study",
            "/study/assessment",
            "/study/collection",
            "/gaze",
            "/favicon.ico",
        }
        public_api_paths = {
            "/api/gaze/models",
            "/api/gaze/session",
            "/api/gaze/sample",
            "/api/gaze/predict",
            "/api/list_models",
            "/api/session",
            "/api/sample",
            "/api/predict",
            "/api/inspector/adaptive/start",
            "/api/inspector/adaptive/score",
            "/api/inspector/adaptive/next",
            "/api/inspector/adaptive/report",
        }
        if (
            path in public_pages
            or path in public_api_paths
            or path.startswith("/static/")
            or path.startswith("/api/study/")
        ):
            return None
        if path.startswith("/api/"):
            return jsonify(
                {
                    "ok": False,
                    "error": "endpoint is disabled on the public study surface",
                }
            ), 403
        return redirect("/study")

    @app.after_request
    def secure_public_study_surface(response):
        if not public_study_mode():
            return response
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = (
            "camera=(self), microphone=(), geolocation=()"
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'none'; connect-src 'self'; "
            "font-src 'self'; form-action 'self'; frame-ancestors 'none'; "
            "img-src 'self' data:; media-src 'self' blob:; "
            "script-src 'self'; style-src 'self' 'unsafe-inline'"
        )
        return response

    _register_blueprints(app)

    # ── Static files & Template views ─────────────────────────────────────────
    @app.route("/")
    def index():
        return render_template("word_track.html")

    @app.route("/gaze")
    def gaze_page():
        return render_template("gaze_page.html")

    @app.route("/gaze_static/<path:filename>")
    def gaze_static_file(filename):
        return send_from_directory(SHENGWEN_STATIC, filename)

    @app.route("/examples/<path:filename>")
    def examples_file(filename):
        return send_from_directory(ROOT / "examples", filename)

    # ── API: Health check ────────────────────────────────────────────────────
    @app.route("/api/ping")
    def ping():
        return jsonify({"ok": True, "sessions": len(list(DATA_DIR.glob("*.json")))})

    # ── API: List sessions (summary only, no items) ──────────────────────────
    @app.route("/api/sessions", methods=["GET"])
    def list_sessions():
        sessions = []
        for f in DATA_DIR.glob("*.json"):
            try:
                with open(f, encoding="utf-8") as fp:
                    d = json.load(fp)
                sessions.append(
                    {
                        "id": d["id"],
                        "filename": d["filename"],
                        "filetype": d.get("filetype", ""),
                        "created_at": d["created_at"],
                        "item_count": d["item_count"],
                    }
                )
            except Exception:
                pass
        sessions.sort(key=lambda x: x["created_at"], reverse=True)
        return jsonify(sessions)

    # ── API: Save a session ──────────────────────────────────────────────────
    @app.route("/api/sessions", methods=["POST"])
    def save_session():
        body = request.get_json(force=True)
        session_id = str(uuid.uuid4())
        items = body.get("items", [])
        session = {
            "id": session_id,
            "filename": body.get("filename", "unknown"),
            "filetype": body.get("filetype", ""),
            "created_at": datetime.now().isoformat(),
            "item_count": len(items),
            "items": items,
        }
        path = DATA_DIR / f"{session_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session, f, ensure_ascii=False, indent=2)
        return jsonify({"id": session_id, "created_at": session["created_at"]}), 201

    # ── API: Get full session data ───────────────────────────────────────────
    @app.route("/api/sessions/<session_id>", methods=["GET"])
    def get_session(session_id):
        path = DATA_DIR / f"{session_id}.json"
        if not path.exists():
            return jsonify({"error": "Not found"}), 404
        with open(path, encoding="utf-8") as f:
            return jsonify(json.load(f))

    # ── API: Delete a session ────────────────────────────────────────────────
    @app.route("/api/sessions/<session_id>", methods=["DELETE"])
    def delete_session(session_id):
        path = DATA_DIR / f"{session_id}.json"
        if path.exists():
            path.unlink()
            return jsonify({"ok": True})
        return jsonify({"error": "Not found"}), 404

    return app

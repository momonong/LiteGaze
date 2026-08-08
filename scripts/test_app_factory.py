from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from web import create_app


class FocusedAppFactoryTests(unittest.TestCase):
    def test_registers_only_requested_blueprints(self):
        app = create_app({
            "TESTING": True,
            "LEXIGAZE_BLUEPRINTS": ("inspector",),
        })
        routes = {rule.rule for rule in app.url_map.iter_rules()}

        self.assertIn("/api/inspector/analyze", routes)
        self.assertNotIn("/api/gaze/health", routes)
        self.assertNotIn("torch", sys.modules)

    def test_rejects_unknown_blueprint(self):
        with self.assertRaisesRegex(ValueError, "unknown-subsystem"):
            create_app({"LEXIGAZE_BLUEPRINTS": ("unknown-subsystem",)})

    def test_gaze_motion_audit_route_is_lightweight_and_handles_missing_session(self):
        app = create_app({
            "TESTING": True,
            "LEXIGAZE_BLUEPRINTS": ("gaze",),
        })

        response = app.test_client().get(
            "/api/gaze/datasets/does-not-exist-for-test/motion-audit"
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json()["error"], "session not found")
        self.assertNotIn("torch", sys.modules)

    def test_gaze_routes_honor_configured_storage_root(self):
        with tempfile.TemporaryDirectory(prefix="lexigaze-gaze-root-") as name:
            root = Path(name)
            app = create_app({
                "TESTING": True,
                "LEXIGAZE_BLUEPRINTS": ("gaze",),
                "LEXIGAZE_GAZE_ROOT": str(root),
            })
            response = app.test_client().get("/api/gaze/health")
            self.assertEqual(response.status_code, 200)
            self.assertTrue((root / "examples" / "models").is_dir())


if __name__ == "__main__":
    unittest.main()

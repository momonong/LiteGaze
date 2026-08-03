from __future__ import annotations

import sys
import unittest

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


if __name__ == "__main__":
    unittest.main()

"""Runs the JavaScript tests in tests/js (announcements, overlay geometry, scene panel)
under Node, if it is installed."""
import os
import shutil
import subprocess
import unittest

JS_TESTS = os.path.join(os.path.dirname(__file__), "js")


@unittest.skipUnless(shutil.which("node"), "Node.js is not installed")
class FrontendJavaScriptTests(unittest.TestCase):
    def run_node(self, name):
        result = subprocess.run(["node", "--test", os.path.join(JS_TESTS, name)],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_announcement_manager_and_speaker(self):
        self.run_node("announcer.test.js")

    def test_automatic_medicine_identification(self):
        self.run_node("medicine-identification.test.js")

    def test_medicine_candidates_and_capture(self):
        self.run_node("medicine.test.js")

    def test_product_research(self):
        self.run_node("product-research.test.js")

    def test_ask_ai(self):
        self.run_node("ask-ai.test.js")

    def test_overlay_geometry_and_scene_panel(self):
        self.run_node("view.test.js")

    def test_interface_state_and_sheets(self):
        self.run_node("ui.test.js")


if __name__ == "__main__":
    unittest.main()

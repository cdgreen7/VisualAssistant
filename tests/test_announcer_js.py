"""Runs the JavaScript announcement-manager tests (tests/js) under Node, if it is installed."""
import os
import shutil
import subprocess
import unittest

JS_TESTS = os.path.join(os.path.dirname(__file__), "js", "announcer.test.js")


@unittest.skipUnless(shutil.which("node"), "Node.js is not installed")
class AnnouncerJavaScriptTests(unittest.TestCase):
    def test_announcement_manager_and_speaker(self):
        result = subprocess.run(["node", "--test", JS_TESTS], capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

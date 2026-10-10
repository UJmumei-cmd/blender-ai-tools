"""Blender の中で、拡張機能の manifest が通り、登録と解除ができるかを確かめる"""

import subprocess
import tempfile
import unittest
from pathlib import Path

import bpy

import blender_ai_tools

PACKAGE_DIR = Path(__file__).resolve().parents[2] / "blender_ai_tools"


def run_extension_command(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([bpy.app.binary_path, "--command", "extension", *args],
                          capture_output=True, text=True, timeout=120)


class ExtensionTest(unittest.TestCase):
    def test_manifest_validates(self):
        result = run_extension_command("validate", str(PACKAGE_DIR))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_builds_zip(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result = run_extension_command("build", "--source-dir", str(PACKAGE_DIR), "--output-dir", output_dir)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(len(list(Path(output_dir).glob("blender_ai_tools-*.zip"))), 1)

    def test_register_and_unregister(self):
        blender_ai_tools.register()
        blender_ai_tools.unregister()


if __name__ == "__main__":
    unittest.main()

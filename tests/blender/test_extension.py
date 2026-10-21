"""Blender の中で、拡張機能の manifest が通り、登録と解除ができるかを確かめる"""

import os
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

    def test_installed_zip_enables_at_startup(self):
        """作った zip を、別の Blender の設定のフォルダに入れて有効にする（起動の途中の登録では bpy.data がまだ読めない）。
        ユーザーが入れるのと同じ道を通す。設定のフォルダは一時フォルダなので、この Blender の設定は変えない"""
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "dist"
            output.mkdir()
            built = run_extension_command("build", "--source-dir", str(PACKAGE_DIR), "--output-dir", str(output))
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            archive = next(output.glob("blender_ai_tools-*.zip"))
            env = {**os.environ, "BLENDER_USER_RESOURCES": str(Path(folder) / "user")}
            installed = subprocess.run([bpy.app.binary_path, "--background", "--factory-startup", "--command", "extension",
                                        "install-file", "-r", "user_default", "-e", str(archive)],
                                       capture_output=True, text=True, timeout=120, env=env)
            self.assertEqual(installed.returncode, 0, installed.stdout + installed.stderr)
            # 入れたあと、ふつうに起動する（拡張機能は起動の途中で登録される）
            started = subprocess.run([bpy.app.binary_path, "--background", "--python-exit-code", "1", "--python-expr",
                                      "import bpy, sys; sys.exit(0 if hasattr(bpy.ops.ai_tools, 'save_version') else 3)"],
                                     capture_output=True, text=True, timeout=120, env=env)
            self.assertEqual(started.returncode, 0, started.stdout + started.stderr)
            self.assertNotIn("Exception in module register", started.stdout + started.stderr)

    def test_register_and_unregister(self):
        blender_ai_tools.register()
        blender_ai_tools.unregister()


if __name__ == "__main__":
    unittest.main()

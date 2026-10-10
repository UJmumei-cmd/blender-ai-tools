"""拡張機能のパッケージが、bpy なしで読み込めて、manifest が決まりどおりかを確かめる"""

import importlib
import sys
import tomllib
import unittest
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[2] / "blender_ai_tools"


class PackageTest(unittest.TestCase):
    def test_imports_without_bpy(self):
        """__init__.py は bpy を import しない（Blender の外でもテストできるようにするため）"""
        had_bpy = "bpy" in sys.modules
        package = importlib.import_module("blender_ai_tools")
        self.assertTrue(callable(package.register))
        self.assertTrue(callable(package.unregister))
        if not had_bpy:
            self.assertNotIn("bpy", sys.modules)

    def test_manifest(self):
        manifest = tomllib.loads((PACKAGE_DIR / "blender_manifest.toml").read_text(encoding="utf-8"))
        self.assertEqual(manifest["id"], "blender_ai_tools")
        self.assertEqual(manifest["type"], "add-on")
        self.assertEqual(manifest["blender_version_min"], "5.2.0")


if __name__ == "__main__":
    unittest.main()

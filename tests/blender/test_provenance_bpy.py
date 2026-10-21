"""A15：Blender の中でだけ分かる環境（Blender と mmd_tools の版）を確かめる。

クラウドの Blender には mmd_tools が入っていない。入っているかどうかは決めつけず、形と型を確かめる。
mmd_tools を探す所は、にせのアドオンの一覧で、入っているときの動きも確かめる。
"""

import re
import types
import unittest

import bpy

from blender_ai_tools.provenance import RunRecord, compare_results, python_environment
from blender_ai_tools.provenance.environment_bpy import (_find_mmd_tools, _is_mmd_tools_name, _version_text,
                                                         blender_environment)

VERSION_PATTERN = re.compile(r"\d+(\.\d+)*([-+.].*)?")


def fake_module(name: str, version=None) -> types.SimpleNamespace:
    """addon_utils.modules() が返すような、にせのモジュール"""
    info = {} if version is None else {"version": version}
    return types.SimpleNamespace(__name__=name, bl_info=info)


def bl_info_of(module) -> dict:
    return module.bl_info


class BlenderEnvironmentTest(unittest.TestCase):
    def test_shape_and_types(self):
        environment = blender_environment()
        self.assertEqual(set(environment), {"blender_version", "blender_build_hash", "mmd_tools_module",
                                            "mmd_tools_version", "mmd_tools_enabled"})
        self.assertEqual(environment["blender_version"], bpy.app.version_string)
        self.assertIsInstance(environment["blender_build_hash"], str)
        self.assertTrue(environment["blender_build_hash"])
        build_hash = bpy.app.build_hash
        if isinstance(build_hash, bytes):
            build_hash = build_hash.decode("ascii")
        self.assertEqual(environment["blender_build_hash"], build_hash)
        self.assertIs(type(environment["mmd_tools_enabled"]), bool)
        module = environment["mmd_tools_module"]
        version = environment["mmd_tools_version"]
        if module is None:
            # 入っていないときは、版が無く、有効でもない
            self.assertIsNone(version)
            self.assertFalse(environment["mmd_tools_enabled"])
        else:
            self.assertTrue(_is_mmd_tools_name(module))
            if version is not None:
                self.assertRegex(version, VERSION_PATTERN)

    def test_can_be_recorded(self):
        """python_environment() と合わせて記録でき、JSON にして読み戻しても同じ"""
        environment = {**python_environment(), **blender_environment()}
        record = RunRecord(tool_id="measure.test", tool_version="1", started_at="2026-10-10T00:00:00Z",
                           finished_at="2026-10-10T00:00:01Z", addon_version="0.1.0", environment=environment,
                           inputs=(), settings={}, seed=None, results={})
        loaded = RunRecord.from_json(record.to_json())
        self.assertEqual(loaded.environment, environment)
        self.assertEqual(compare_results(loaded.environment, environment), [])


class FindMmdToolsTest(unittest.TestCase):
    def test_module_names(self):
        self.assertTrue(_is_mmd_tools_name("mmd_tools"))
        self.assertTrue(_is_mmd_tools_name("bl_ext.blender_org.mmd_tools"))
        self.assertTrue(_is_mmd_tools_name("bl_ext.user_default.mmd_tools"))
        for name in ("mmd_tools_local", "my_mmd_tools", "bl_ext.blender_org.mmd_tools_extra", "mmd", ""):
            with self.subTest(name=name):
                self.assertFalse(_is_mmd_tools_name(name))

    def test_not_installed(self):
        modules = [fake_module("io_scene_fbx", (5, 15, 0)), fake_module("bl_ext.blender_org.mmd_tools_extra", (1,))]
        self.assertEqual(_find_mmd_tools(modules, bl_info_of, lambda name: True), (None, None, False))
        self.assertEqual(_find_mmd_tools([], bl_info_of, lambda name: True), (None, None, False))

    def test_extension(self):
        modules = [fake_module("io_scene_fbx", (5, 15, 0)), fake_module("bl_ext.blender_org.mmd_tools", (4, 3, 2))]
        self.assertEqual(_find_mmd_tools(modules, bl_info_of, lambda name: True),
                         ("bl_ext.blender_org.mmd_tools", "4.3.2", True))
        self.assertEqual(_find_mmd_tools(modules, bl_info_of, lambda name: False),
                         ("bl_ext.blender_org.mmd_tools", "4.3.2", False))

    def test_legacy_add_on(self):
        modules = [fake_module("mmd_tools", (2, 10, 1))]
        self.assertEqual(_find_mmd_tools(modules, bl_info_of, lambda name: True), ("mmd_tools", "2.10.1", True))

    def test_prefers_the_enabled_one(self):
        modules = [fake_module("bl_ext.blender_org.mmd_tools", (4, 3, 2)), fake_module("mmd_tools", (2, 10, 1))]
        self.assertEqual(_find_mmd_tools(modules, bl_info_of, lambda name: name == "mmd_tools"),
                         ("mmd_tools", "2.10.1", True))
        # どちらも有効でないときは、名前の順で最初の物
        self.assertEqual(_find_mmd_tools(modules, bl_info_of, lambda name: False),
                         ("bl_ext.blender_org.mmd_tools", "4.3.2", False))

    def test_unknown_version(self):
        self.assertEqual(_find_mmd_tools([fake_module("mmd_tools")], bl_info_of, lambda name: True),
                         ("mmd_tools", None, True))

    def test_version_text(self):
        self.assertEqual(_version_text((4, 3, 2)), "4.3.2")
        self.assertEqual(_version_text([1, 0]), "1.0")
        self.assertEqual(_version_text("4.3.2"), "4.3.2")
        self.assertIsNone(_version_text(()))
        self.assertIsNone(_version_text(""))
        self.assertIsNone(_version_text(None))


if __name__ == "__main__":
    unittest.main()

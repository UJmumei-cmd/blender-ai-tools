"""目録に載せた道具を、LLM の返事（JSON の文）から呼ぶ流れを、最初から最後まで確かめる"""

import json
import unittest

from blender_ai_tools.catalog import (ConfirmationRequiredError, InvalidParamsError, ToolCallError,
                                      UnknownToolError, parse_tool_call)
from blender_ai_tools.registry import build_catalog
from blender_ai_tools.thresholds import MAX_BONE_NAMES


from blender_ai_tools.catalog import Effect
from blender_ai_tools.motion.course import CHECKS
from blender_ai_tools.motion.poses import POSES
from blender_ai_tools.registry import POSE_NAMES, allowed_from_params
from blender_ai_tools.versions.diff import ALL_CATEGORIES, CATEGORIES


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self.catalog = build_catalog()

    def test_listing(self):
        self.assertIn("bones.map_names", self.catalog)
        self.assertEqual(self.catalog.categories(), ("bones", "export", "motion", "versions"))
        self.assertEqual([spec.id for spec in self.catalog.specs()],
                         ["bones.map_names", "export.convert_and_check", "motion.run_course", "versions.compare_with_previous",
                          "versions.list", "versions.save"])
        # 同じ道具なら、作り直しても指紋は同じ（A15 の記録で比べられる）
        self.assertEqual(self.catalog.fingerprint(), build_catalog().fingerprint())
        json.dumps(self.catalog.listing(), ensure_ascii=False)

    def test_call_from_llm_reply(self):
        reply = json.dumps({"tool": "bones.map_names",
                            "params": {"names": ["J_Bip_C_Hips", "J_Bip_L_UpperArm", "左腕", "スカート_L", "謎の骨"]}},
                           ensure_ascii=False)
        call = parse_tool_call(reply)
        self.assertFalse(self.catalog.needs_confirmation(call.tool))  # 見るだけの道具は確認なしで動く
        result = self.catalog.call(call.tool, call.params)
        json.dumps(result, ensure_ascii=False)  # 結果は JSON にできる
        self.assertEqual(result["mapping"]["J_Bip_C_Hips"], "センター")
        self.assertEqual(result["unknown"], ["謎の骨"])
        # 揺れ物の骨は、人の形ではない骨として分かる（ユーザーに聞かない）。報告の「揺れ物の骨 N 本」のために、別にも出す
        self.assertIn("スカート_L", result["known_extras"])
        self.assertEqual(result["sway_extras"], ["スカート_L"])
        self.assertFalse(result["semi_standard_compliant"])
        # 左腕 と J_Bip_L_UpperArm が同じ骨を取り合うときは、名前がそのものの 左腕 が勝つ
        self.assertEqual(result["mapping"]["左腕"], "左腕")
        self.assertEqual(result["shadowed"], {"J_Bip_L_UpperArm": "左腕"})

    def test_result_shape_and_version(self):
        """結果の項目は決まった形。項目を足したら道具の版を上げる（A15 の記録で、前の結果と比べられる）"""
        self.assertEqual(self.catalog.get("bones.map_names").version, "2")
        result = self.catalog.call("bones.map_names", {"names": ["髪1", "謎の骨", "左腕", "操作中心"]})
        self.assertEqual(set(result), {"mapping", "unknown", "known_extras", "sway_extras", "conflicts", "ambiguous",
                                       "shadowed", "missing_standard", "missing_semi_standard", "semi_standard_compliant"})
        # 揺れ物の骨は known_extras の中の印：両方に入る。揺れ物でない知っている骨（操作中心）は known_extras だけ
        self.assertEqual(result["known_extras"], ["髪1", "操作中心"])
        self.assertEqual(result["sway_extras"], ["髪1"])
        self.assertEqual(result["unknown"], ["謎の骨"])
        self.assertEqual(result["mapping"], {"左腕": "左腕"})

    def test_rejects_bad_calls(self):
        with self.assertRaises(UnknownToolError):
            self.catalog.call("bones.delete_all", {})
        with self.assertRaises(InvalidParamsError):
            self.catalog.call("bones.map_names", {"names": "左腕"})
        with self.assertRaises(InvalidParamsError):
            self.catalog.call("bones.map_names", {"names": ["左腕"], "extra": 1})
        with self.assertRaises(InvalidParamsError):
            self.catalog.call("bones.map_names", {"names": ["x"] * (MAX_BONE_NAMES + 1)})
        with self.assertRaises(ToolCallError):
            parse_tool_call('{"tool": "bones.map_names", "params": {"names": []}, "code": "import os"}')

    def test_read_tool_never_needs_confirmation(self):
        self.assertFalse(self.catalog.needs_confirmation("bones.map_names", plan_approved=False))
        try:
            self.catalog.call("bones.map_names", {"names": []})
        except ConfirmationRequiredError:
            self.fail("見るだけの道具で確認を求めた")


class HeavyToolsTest(unittest.TestCase):
    """bpy が要る道具（E1＋E2・A6・A5）の、スキーマと重さの印。中身は tests/blender/test_registry_bpy.py で確かめる"""

    def setUp(self):
        self.catalog = build_catalog()

    def test_heavy_flags_and_effects(self):
        # 重い物（別の Blender で動かす）と、軽い物。どれも見るだけ（モデルは変えない。版の保存は新しいファイルを作るだけ）
        heavy = {spec.id: spec.heavy for spec in self.catalog.specs()}
        self.assertEqual(heavy, {"bones.map_names": False, "export.convert_and_check": True, "motion.run_course": True,
                                 "versions.compare_with_previous": False, "versions.list": False, "versions.save": False})
        self.assertTrue(all(spec.effect == Effect.READ for spec in self.catalog.specs()))
        for spec in self.catalog.specs():
            self.assertFalse(self.catalog.needs_confirmation(spec.id))

    def test_schemas_refuse_bad_params_before_running(self):
        """値のまちがいは、bpy の部品を読み込む前（スキーマ）で断る"""
        bad = [
            ("export.convert_and_check", {"input_blend": "a.blend", "output_dir": "out"}),  # koumakan_repo が無い
            ("export.convert_and_check", {"input_blend": "a.blend", "output_dir": "out", "koumakan_repo": "r", "x": 1}),
            ("export.convert_and_check", {"input_blend": 1, "output_dir": "out", "koumakan_repo": "r"}),
            ("versions.save", {"base_path": 1}),
            ("versions.list", {"base_path": "x" * 2000}),
            ("versions.compare_with_previous", {"allowed": [{"object": "Body", "categories": ["colour"]}]}),
            ("versions.compare_with_previous", {"allowed": [{"object": "Body", "categories": []}]}),
            ("versions.compare_with_previous", {"allowed": [{"categories": ["vertices"]}]}),
            ("versions.compare_with_previous", {"allowed": [{"object": "Body", "all": "yes"}]}),
            ("versions.compare_with_previous", {"allowed": {"Body": "*"}}),
            ("motion.run_course", {"poses": ["no_such_pose"]}),
            ("motion.run_course", {"poses": ["arms_up", "arms_up"]}),
            ("motion.run_course", {"checks": ["dance"]}),
            ("motion.run_course", {"visibility": "yes"}),
            ("motion.run_course", {"app_motions": "nod"}),
        ]
        for tool, params in bad:
            with self.subTest(tool=tool, params=params), self.assertRaises(InvalidParamsError):
                self.catalog.call(tool, params)

    def test_pose_and_check_choices_follow_the_source(self):
        schema = self.catalog.get("motion.run_course").input_schema
        self.assertEqual(schema["properties"]["poses"]["items"]["enum"], [pose.name for pose in POSES])
        self.assertEqual(list(POSE_NAMES), [pose.name for pose in POSES])
        self.assertEqual(schema["properties"]["checks"]["items"]["enum"], list(CHECKS))
        self.assertEqual(self.catalog.get("versions.compare_with_previous").input_schema["properties"]["allowed"]["items"]
                         ["properties"]["categories"]["items"]["enum"], list(CATEGORIES))


class AllowedFromParamsTest(unittest.TestCase):
    def test_categories_and_all(self):
        allowed = allowed_from_params([{"object": "Skirt", "categories": ["vertices", "weights"]},
                                       {"object": "Hair", "all": True},
                                       {"object": "Ribbon", "all": True, "categories": ["bones"]}])
        self.assertEqual(allowed.objects["Skirt"], frozenset({"vertices", "weights"}))
        self.assertEqual(allowed.objects["Hair"], ALL_CATEGORIES)
        self.assertEqual(allowed.objects["Ribbon"], ALL_CATEGORIES)  # all が勝つ
        self.assertTrue(allowed.allows("Skirt", "weights"))
        self.assertFalse(allowed.allows("Skirt", "bones"))
        self.assertFalse(allowed.allows("Body", "vertices"))
        self.assertEqual(allowed_from_params(None).objects, {})
        self.assertEqual(allowed_from_params([]).objects, {})

    def test_refuses_empty_and_duplicate_items(self):
        with self.assertRaises(ValueError):
            allowed_from_params([{"object": "Body"}])
        with self.assertRaises(ValueError):
            allowed_from_params([{"object": "Body", "all": False}])
        with self.assertRaises(ValueError):
            allowed_from_params([{"object": "Body", "all": True}, {"object": "Body", "categories": ["vertices"]}])


if __name__ == "__main__":
    unittest.main()

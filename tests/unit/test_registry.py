"""目録に載せた道具を、LLM の返事（JSON の文）から呼ぶ流れを、最初から最後まで確かめる"""

import json
import unittest

from blender_ai_tools.catalog import (ConfirmationRequiredError, InvalidParamsError, ToolCallError,
                                      UnknownToolError, parse_tool_call)
from blender_ai_tools.registry import build_catalog
from blender_ai_tools.thresholds import MAX_BONE_NAMES


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self.catalog = build_catalog()

    def test_listing(self):
        self.assertIn("bones.map_names", self.catalog)
        self.assertEqual(self.catalog.categories(), ("bones",))
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


if __name__ == "__main__":
    unittest.main()

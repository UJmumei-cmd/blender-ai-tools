"""道具の結果から報告を作る所（report/sources.py）を確かめる（bpy なし）"""

import json
import unittest

from blender_ai_tools import texts
from blender_ai_tools.export.roundtrip import REST_POSE, TEST_POSES
from blender_ai_tools.measure.results import Measurement, Signal
from blender_ai_tools.report import sources
from blender_ai_tools.report.model import _ITEM_ID, Report, count_signals


def deviation(pose, value, signal=Signal.GREEN):
    return Measurement("roundtrip_deviation", value, "m", signal, "ずれ", {"pose": pose}).to_data()


def export_result(*, ok=True, signal="green", problems=(), warnings=(), roundtrip=None):
    if roundtrip is None:
        roundtrip = [Measurement("roundtrip_coverage", 1.0, "ratio", Signal.GREEN, "全部").to_data(),
                     deviation(REST_POSE, 0.0001), deviation("arms_up", 0.0002),
                     deviation("knees", 0.0007, Signal.RED),
                     Measurement("influence_limit_effect", 0.00046, "m", Signal.YELLOW, "少し").to_data()]
    return {"convert": {"ok": ok, "signal": signal, "problems": list(problems), "warnings": list(warnings),
                        "summary": {}, "output_dir": "/tmp/out", "usdz": None, "rig_path": None,
                        "input_unchanged": True, "timed_out": False, "returncode": 0, "record": {}},
            "roundtrip": roundtrip}


def change(category, name, key, count, outside, ja="変わった", en="changed"):
    return {"category": category, "object": name, "count": count, "outside_allowed": outside, "key": key,
            "texts": {"ja": ja, "en": en}}


class SourcesTest(unittest.TestCase):
    def tearDown(self):
        texts.set_language("ja")

    def assert_ids_ascii(self, report):
        for item in report.items:
            self.assertRegex(item.item_id, _ITEM_ID)
            self.assertTrue(item.item_id.isascii(), item.item_id)
            self.assertEqual(set(item.titles), {"ja", "en"})
        # 報告は JSON にできて、読み戻せる
        self.assertEqual(Report.from_json(report.to_json()), report)

    def test_kinds_and_unknown_tools(self):
        self.assertEqual(dict(sources.REPORT_KINDS), {"export.convert_and_check": "export",
                                                      "motion.run_course": "pose_course",
                                                      "versions.compare_with_previous": "versions"})
        self.assertTrue(sources.makes_report("motion.run_course"))
        self.assertFalse(sources.makes_report("versions.save"))
        with self.assertRaises(ValueError):
            sources.report_from_tool_result("versions.save", {})
        with self.assertRaises(ValueError):
            sources.report_from_tool_result("motion.run_course", [])
        for kind in sources.REPORT_KINDS.values():
            self.assertNotEqual(sources.kind_label(kind, language="ja"), kind)
            self.assertNotEqual(sources.kind_label(kind, language="en"), kind)
        self.assertEqual(sources.kind_label("brand_new"), "brand_new")

    def test_export_report(self):
        report = sources.report_from_tool_result("export.convert_and_check", export_result(),
                                                 source={"job_id": "job_1", "version": "model_v001.blend"})
        self.assertEqual(report.kind, "export")
        self.assertEqual(report.source, {"job_id": "job_1", "version": "model_v001.blend",
                                         "tool_id": "export.convert_and_check"})
        self.assertEqual([item.item_id for item in report.items],
                         ["export/convert", "export/roundtrip_coverage", "export/roundtrip_deviation/rest",
                          "export/roundtrip_deviation/arms_up", "export/roundtrip_deviation/knees",
                          "export/influence_limit_effect"])
        convert = report.item("export/convert")
        self.assertEqual((convert.signal, convert.measurement["reason"]), (Signal.GREEN, "書き出せました"))
        self.assertEqual(convert.titles, {"ja": "書き出し（convert_blend.py）", "en": "Export (convert_blend.py)"})
        knees = report.item("export/roundtrip_deviation/knees")
        self.assertEqual(knees.titles, {"ja": "読み戻した形のずれ：ひざを曲げる",
                                        "en": "Round-trip shape deviation: Knees bent"})
        self.assertEqual(knees.signal, Signal.RED)
        self.assertEqual(report.item("export/influence_limit_effect").titles["ja"], "4本に絞った影響")
        self.assertEqual(count_signals(report.items)["red"], 1)
        self.assert_ids_ascii(report)

    def test_every_export_pose_has_labels(self):
        """往復の確認のポーズ（休みの姿勢と TEST_POSES）には、どれも日本語と英語の名前がある"""
        self.assertEqual(sources.EXPORT_POSES, (REST_POSE, *TEST_POSES))
        for pose in sources.EXPORT_POSES:
            with self.subTest(pose=pose):
                self.assertIn(f"report.export_pose_{pose}", texts.TEXTS)
        report = sources.report_from_tool_result(
            "export.convert_and_check", export_result(roundtrip=[deviation(pose, 0.0) for pose in sources.EXPORT_POSES]))
        for item in report.items[1:]:
            self.assertNotIn("_", item.titles["ja"])

    def test_export_failure(self):
        """書き出せなかったとき：問題の文が理由に入る。ok でないのに緑と書いてあっても赤にする"""
        gray = Measurement.unmeasured("roundtrip_deviation", "m", "書き出しに失敗したので比べられません").to_data()
        report = sources.report_from_tool_result(
            "export.convert_and_check",
            export_result(ok=False, signal="green", problems=["出力がありません"], warnings=["知らないお知らせ"],
                          roundtrip=[gray]))
        self.assertEqual([item.item_id for item in report.items], ["export/convert", "export/roundtrip_deviation"])
        convert = report.item("export/convert")
        self.assertEqual(convert.signal, Signal.RED)
        self.assertEqual(convert.measurement["reason"], "出力がありません；知らないお知らせ")
        self.assertEqual(convert.measurement["value"], 1.0)
        self.assertEqual(report.item("export/roundtrip_deviation").signal, Signal.GRAY)
        no_detail = sources.report_from_tool_result("export.convert_and_check",
                                                    export_result(ok=False, signal="red", roundtrip=[gray]))
        self.assertIn("くわしい理由はありません", no_detail.item("export/convert").measurement["reason"])
        for bad in ({"convert": {}}, {"roundtrip": []}, export_result(signal="purple")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sources.report_from_tool_result("export.convert_and_check", bad)

    def test_course_report(self):
        events = [{"pose": "arms_up", "check": "floor_sink",
                   "measurement": Measurement("floor_sink", 0.0, "m", Signal.GREEN, "沈まない").to_data()},
                  {"pose": "arms_up", "check": "cloth_penetration",
                   "measurement": Measurement("penetration_depth", 0.003, "m", Signal.RED, "深い").to_data()},
                  {"pose": "app_motions", "check": "app_motion",
                   "measurement": Measurement.unmeasured("app_motion", "count", "まだ").to_data()}]
        report = sources.report_from_tool_result("motion.run_course", {"events": events, "summary": {}, "lines": []})
        self.assertEqual(report.kind, "pose_course")
        self.assertEqual([item.item_id for item in report.items],
                         ["pose_course/arms_up/floor_sink", "pose_course/arms_up/cloth_penetration",
                          "pose_course/app_motions/app_motion"])
        self.assertEqual(count_signals(report.items), {"red": 1, "yellow": 0, "gray": 1, "green": 1, "accepted": 0})
        self.assert_ids_ascii(report)
        with self.assertRaises(ValueError):
            sources.report_from_tool_result("motion.run_course", {"events": [{"pose": "arms_up"}]})
        with self.assertRaises(ValueError):
            sources.report_from_tool_result("motion.run_course", {})

    def test_versions_report(self):
        changes = [change("vertices", "体", "versions.vertices_moved", 3, True, "体：頂点が 3 個動きました",
                          "体: 3 vertices moved"),
                   change("weights", "Skirt", "versions.weights_changed", 5, False),
                   change("shape_keys", "体", "versions.shape_key_changed", 2, True),
                   change("shape_keys", "体", "versions.shape_key_changed", 4, True)]
        summary = Measurement("unexpected_changes", 3.0, "count", Signal.RED, "外の変化").to_data(include_details=False)
        result = {"measurement": summary, "changes": changes, "lines": []}
        report = sources.report_from_tool_result("versions.compare_with_previous", result)
        self.assertEqual(report.kind, "versions")
        body = sources.short_hash("体")
        self.assertEqual(len(body), 12)
        self.assertEqual([item.item_id for item in report.items],
                         [f"versions/vertices/{body}/vertices_moved",
                          f"versions/weights/{sources.short_hash('Skirt')}/weights_changed",
                          f"versions/shape_keys/{body}/shape_key_changed",
                          f"versions/shape_keys/{body}/shape_key_changed/2"])
        first = report.items[0]
        self.assertEqual(first.titles, {"ja": "体：頂点が 3 個動きました", "en": "体: 3 vertices moved"})
        self.assertEqual((first.signal, first.measurement["value"]), (Signal.RED, 3.0))
        self.assertEqual(report.items[1].signal, Signal.GREEN)
        self.assert_ids_ascii(report)
        # 同じ結果からは、同じ id（次の版でも答えを使える）
        again = sources.report_from_tool_result("versions.compare_with_previous", json.loads(json.dumps(result)))
        self.assertEqual([i.item_id for i in again.items], [i.item_id for i in report.items])

    def test_versions_without_changes_uses_the_summary(self):
        for summary in (Measurement("unexpected_changes", 0.0, "count", Signal.GREEN, "変化なし"),
                        Measurement.unmeasured("unexpected_changes", "count", "前の版がありません")):
            with self.subTest(signal=summary.signal):
                report = sources.report_from_tool_result(
                    "versions.compare_with_previous",
                    {"measurement": summary.to_data(include_details=False), "changes": [], "lines": []})
                self.assertEqual([(i.item_id, i.signal) for i in report.items],
                                 [("versions/unexpected_changes", summary.signal)])

    def test_versions_needs_both_languages(self):
        old = {"category": "vertices", "object": "体", "count": 1, "outside_allowed": True,
               "key": "versions.vertices_moved", "text": "体：頂点が動きました"}
        for bad in (old, {**old, "texts": {"ja": "x"}}, {**change("vertices", "体", "vertices_moved", 1, True)},
                    {**change("vertices", "体", "versions.vertices_moved", -1, True)}, "not a change"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sources.report_from_tool_result("versions.compare_with_previous",
                                                {"measurement": {}, "changes": [bad], "lines": []})


if __name__ == "__main__":
    unittest.main()

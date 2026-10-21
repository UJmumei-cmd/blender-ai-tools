"""信号の報告（A11）の形・一言・HTML・答えの記録を確かめる"""

import json
import tempfile
import unittest
from pathlib import Path

from blender_ai_tools import texts
from blender_ai_tools.measure.results import Measurement, Signal
from blender_ai_tools.motion.course import Event
from blender_ai_tools.report import html as report_html
from blender_ai_tools.report import model
from blender_ai_tools.report.decisions import DecisionStore

TIME = "2026-10-21T07:00:00Z"


def item(name, signal, value=1.0, *, group="pose_course", decision=None, reason="理由", details=None):
    measurement = (Measurement.unmeasured(name, "m", reason or "測れない", details) if signal == Signal.GRAY
                   else Measurement(name, value, "m", signal, reason, details or {}))
    return model.item_from_measurement(group, measurement).__class__(**{**model.item_from_measurement(group, measurement).to_data(),
                                                                       "decision": decision})


class ItemTest(unittest.TestCase):
    def tearDown(self):
        texts.set_language("ja")

    def test_from_measurement_titles_and_id(self):
        measurement = Measurement("penetration_depth", 0.0012, "m", Signal.RED, "深い", {"count": 3})
        made = model.item_from_measurement("course", measurement)
        self.assertEqual(made.item_id, "course/penetration_depth")
        self.assertEqual(made.titles, {"ja": "めり込みの深さ", "en": "Penetration depth"})
        self.assertEqual(made.signal, Signal.RED)
        self.assertEqual(made.measurement["details"], {"count": 3})
        self.assertEqual(model.item_from_measurement("x", measurement, include_details=False).measurement["details"], {})
        unknown = model.item_from_measurement("x", Measurement("brand_new_name", 1.0, "count", Signal.GREEN))
        self.assertEqual(unknown.titles, {"ja": "brand_new_name", "en": "brand_new_name"})

    def test_round_trip_and_validation(self):
        made = item("asymmetry", Signal.YELLOW, decision="later")
        self.assertEqual(model.Item.from_data(made.to_data()), made)
        bad = {**made.to_data(), "decision": "ignore"}
        with self.assertRaises(ValueError):
            model.Item.from_data(bad)
        with self.assertRaises(ValueError):
            model.Item.from_data({**made.to_data(), "titles": {"ja": "x"}})
        with self.assertRaises(ValueError):
            model.Item.from_data({**made.to_data(), "extra": 1})
        with self.assertRaises(ValueError):
            model.Item(item_id="a b", group="g", titles={"ja": "x", "en": "x"}, measurement=made.measurement)

    def test_items_from_course_events(self):
        events = [Event("arms_up", "cloth_penetration", Measurement("penetration_depth", 0.002, "m", Signal.RED, "深い")),
                  Event("app:nod", "app_motion", Measurement.unmeasured("app_motion", "count", "まだ"))]
        items = model.items_from_course_events(events)
        self.assertEqual([i.item_id for i in items], ["pose_course/arms_up/cloth_penetration", "pose_course/app:nod/app_motion"])
        self.assertIn("服", items[0].titles["ja"])
        self.assertIn("nod", items[1].titles["ja"])
        self.assertNotEqual(items[0].titles["ja"], items[0].titles["en"])

    def test_duplicate_ids_rejected(self):
        m = Measurement("x", 1.0, "m", Signal.GREEN)
        with self.assertRaises(ValueError):
            model.items_from_measurements("g", [m, m])


class CountsAndHeadlineTest(unittest.TestCase):
    def tearDown(self):
        texts.set_language("ja")

    def test_counts_exclude_accepted(self):
        items = [item("a", Signal.RED), item("b", Signal.RED, decision="accept"), item("c", Signal.YELLOW),
                 item("d", Signal.GRAY), item("e", Signal.GREEN)]
        self.assertEqual(model.count_signals(items), {"red": 1, "yellow": 1, "gray": 1, "green": 1, "accepted": 1})
        self.assertEqual([i.item_id for i in model.sort_items(items)],
                         ["pose_course/a", "pose_course/d", "pose_course/c", "pose_course/e", "pose_course/b"])

    def test_headline_variants(self):
        self.assertEqual(model.headline([]), "点検の結果がありません")
        self.assertEqual(model.headline([item("e", Signal.GREEN)]), "大丈夫です")
        self.assertEqual(model.headline([item("c", Signal.YELLOW), item("c2", Signal.YELLOW)]), "大丈夫です（黄 2 件）")
        self.assertEqual(model.headline([item("d", Signal.GRAY), item("c", Signal.YELLOW)]),
                         "赤はありません。測れなかった物が 1 件あります（黄 1 件）")
        line = model.headline([item("a", Signal.RED), item("b", Signal.RED, decision="accept"), item("d", Signal.GRAY)])
        self.assertTrue(line.startswith("赤が 1 件あります（黄 0 件、測れなかった物 1 件）"))
        self.assertIn("「このままでよい」とした物 1 件", line)
        self.assertEqual(model.headline([item("e", Signal.GREEN)], language="en"), "OK")
        # 「このままでよい」とした赤だけなら、赤は 0 で「大丈夫」になる（一覧には残る）
        self.assertTrue(model.headline([item("a", Signal.RED, decision="accept")]).startswith("大丈夫です"))

    def test_value_text(self):
        self.assertEqual(model.value_text({"value": 0.00123, "unit": "m"}), "1.23 mm")
        self.assertEqual(model.value_text({"value": 12.345, "unit": "deg"}), "12.3 度")
        self.assertEqual(model.value_text({"value": 3.0, "unit": "count"}), "3 件")
        self.assertEqual(model.value_text({"value": 0.5, "unit": "ratio"}), "0.5 ratio")
        self.assertEqual(model.value_text({"value": None, "unit": "m"}), "値なし")
        self.assertEqual(model.value_text({"value": 3.0, "unit": "count"}, language="en"), "3")

    def test_compare_red_count(self):
        items = [item("a", Signal.RED), item("b", Signal.RED, decision="accept")]
        first = model.compare_red_count(items, None)
        self.assertEqual((first.signal, first.details["current"]), (Signal.GRAY, 1))
        self.assertEqual(model.compare_red_count(items, 1).signal, Signal.GREEN)
        self.assertEqual(model.compare_red_count(items, 2).signal, Signal.GREEN)
        increased = model.compare_red_count(items, 0)
        self.assertEqual((increased.signal, increased.value), (Signal.RED, 1.0))
        self.assertIn("0 件から 1 件", increased.reason)
        with self.assertRaises(ValueError):
            model.compare_red_count(items, -1)
        with self.assertRaises(ValueError):
            model.compare_red_count(items, True)


class ReportTest(unittest.TestCase):
    def test_round_trip_and_decisions(self):
        report = model.new_report("pose_course", [item("a", Signal.RED), item("b", Signal.YELLOW)],
                                  source={"tool_id": "motion.course", "version": "model_v003"})
        self.assertEqual(model.Report.from_json(report.to_json()), report)
        self.assertTrue(report.report_id.endswith("_pose_course"))
        decided = report.with_decisions({"pose_course/a": {"decision": "accept", "note": "飾りの帯", "decided_at": TIME}})
        self.assertEqual(decided.item("pose_course/a").decision, "accept")
        self.assertEqual(decided.item("pose_course/a").decision_note, "飾りの帯")
        self.assertIsNone(decided.item("pose_course/b").decision)
        with self.assertRaises(KeyError):
            decided.item("nothing")
        with self.assertRaises(ValueError):
            model.Report(report_id="r", kind="Pose", created_at=TIME, items=())
        with self.assertRaises(ValueError):
            model.Report(report_id="r", kind="pose", created_at=TIME, items=(item("a", Signal.RED), item("a", Signal.RED)))
        with self.assertRaises(ValueError):
            model.Report.from_data({**report.to_data(), "extra": 1})


class HtmlTest(unittest.TestCase):
    def tearDown(self):
        texts.set_language("ja")

    def test_html_contents_and_escaping(self):
        nasty = "<script>alert(1)</script> & 材質"
        items = [item("a", Signal.RED, reason=nasty, details={"worst": [1, 2]}),
                 item("b", Signal.GREEN, decision="accept")]
        report = model.new_report("pose_course", items)
        page = report_html.to_html(report)
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertIn("赤が 1 件あります", page)
        self.assertIn('class="headline red"', page)
        self.assertIn("このままでよい", page)
        self.assertIn("&quot;worst&quot;", page)
        self.assertIn('id="pose_course/a"', page)
        self.assertTrue(page.startswith("<!DOCTYPE html>"))
        self.assertNotIn("http://", page)  # 外の物は読まない
        self.assertNotIn("https://", page)
        english = report_html.to_html(report, language="en")
        self.assertIn("1 red", english)
        self.assertIn('lang="en"', english)
        big = model.new_report("x", [item("a", Signal.RED, details={"many": list(range(5000))})])
        self.assertNotIn("<details>", report_html.to_html(big))  # 長い details は入れない

    def test_write_html(self):
        with tempfile.TemporaryDirectory() as folder:
            path = report_html.write_html(model.new_report("x", [item("e", Signal.GREEN)]), Path(folder) / "out" / "r.html")
            self.assertTrue(path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>"))


class DecisionStoreTest(unittest.TestCase):
    def test_store_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "decisions.json"
            store = DecisionStore(path)
            self.assertEqual(store.decisions, {})
            store.decide("pose_course/a", "accept", note="飾りの帯", version="model_v003")
            with self.assertRaises(ValueError):
                store.decide("pose_course/b", "ignore")
            items = [item("a", Signal.RED, decision="accept"), item("b", Signal.RED), item("c", Signal.YELLOW)]
            self.assertIsNone(store.baseline_red("pose_course"))
            self.assertEqual(store.record_baseline("pose_course", items, version="model_v003"), 1)
            store.save()
            again = DecisionStore(path)
            self.assertEqual(again.decision_of("pose_course/a")["note"], "飾りの帯")
            self.assertEqual(again.baseline_red("pose_course"), 1)
            report = model.new_report("pose_course", items).with_decisions(again.decisions)
            self.assertEqual(report.item("pose_course/a").decision, "accept")
            again.clear("pose_course/a")
            self.assertIsNone(again.decision_of("pose_course/a"))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["schema_version"], 1)

    def test_store_rejects_bad_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "decisions.json"
            for text in ('{"schema_version": 1, "decisions": {"a": {"decision": "maybe", "decided_at": "%s"}}, "baselines": {}}' % TIME,
                         '{"schema_version": 1, "decisions": {}, "baselines": {"x": {"red": -1, "updated_at": "%s"}}}' % TIME,
                         '{"schema_version": 2, "decisions": {}, "baselines": {}}', "{"):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text[:40]), self.assertRaises(ValueError):
                    DecisionStore(path)


class TextsTest(unittest.TestCase):
    def test_report_texts_complete(self):
        self.assertEqual(texts.problems(), [])
        self.assertIn("report.headline_red", texts.TEXTS)

    def test_every_measure_name_has_a_title(self):
        """measure/ の測った結果の名前（*_NAME の定数）には、どれも報告の題 report.item_<名前> がある"""
        import ast
        from pathlib import Path

        folder = Path(__file__).resolve().parents[2] / "blender_ai_tools" / "measure"
        names = {}
        for path in sorted(folder.glob("*.py")):
            for node in ast.parse(path.read_text(encoding="utf-8")).body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                        and isinstance(node.value.value, str):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.endswith("_NAME"):
                            names[f"{path.name}:{target.id}"] = node.value.value
        self.assertGreaterEqual(len(names), 20)  # 見張りそのものが、定数を見つけている
        for where, name in sorted(names.items()):
            with self.subTest(constant=where, name=name):
                self.assertIn(f"report.item_{name}", texts.TEXTS)


if __name__ == "__main__":
    unittest.main()

"""目録（A8）に載せた道具を、Blender の中で本物のモデルで動かす（A6 版の管理、E1＋E2 書き出して確かめる、A10 の裏の係から）"""

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import bpy
import numpy as np

from blender_ai_tools import provenance
from blender_ai_tools.catalog import InvalidParamsError
from blender_ai_tools.jobs import launcher_bpy, worker
from blender_ai_tools.motion.course import CHECKS
from blender_ai_tools.registry import POSE_NAMES, build_catalog
from blender_ai_tools.report.sources import report_from_tool_result
from tests.fixtures import mmd_model

KOUMAKAN = os.environ.get("BLENDER_AI_TOOLS_KOUMAKAN_REPO", "")


def move_vertex(obj, index: int, step: tuple[float, float, float]) -> None:
    """頂点を動かす（メッシュの頂点と、全部のシェイプキーを同じだけ。編集モードで基本の形を動かしたときと同じ）"""
    delta = np.array(step)
    obj.data.vertices[index].co = tuple(np.array(obj.data.vertices[index].co) + delta)
    if obj.data.shape_keys is not None:
        for block in obj.data.shape_keys.key_blocks:
            block.data[index].co = tuple(np.array(block.data[index].co) + delta)
    obj.data.update()


def open_fresh_model(folder: Path) -> Path:
    """試し用のモデルを作って保存し、そのファイルを開く（開いているファイルの場所が要る道具のため）"""
    bpy.ops.wm.read_homefile(use_empty=True)
    mmd_model.build_model()
    path = folder / "model.blend"
    mmd_model.save_model(path)
    bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False)
    return path


class VersionToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.blend = open_fresh_model(self.folder)
        self.catalog = build_catalog()

    def tearDown(self):
        bpy.ops.wm.read_homefile(use_empty=True)
        self.temp.cleanup()

    def test_save_list_and_compare(self):
        saved = self.catalog.call("versions.save", {})
        self.assertEqual(saved["name"], "model_v001.blend")
        self.assertTrue(Path(saved["version"]).is_file())
        self.assertTrue(Path(saved["snapshot_dir"]).is_dir())
        self.assertEqual(bpy.data.filepath, str(self.blend))  # 開いているファイルは変わらない
        listed = self.catalog.call("versions.list", {})
        self.assertEqual([v["number"] for v in listed["versions"]], [1])
        self.assertEqual(len(listed["lines"]), 1)
        same = self.catalog.call("versions.compare_with_previous", {})
        self.assertEqual(same["measurement"]["signal"], "green")
        self.assertEqual(same["changes"], [])
        # 体の点を 1cm 動かすと、変えてよい所の外の変化（赤）。Body を変えてよい所にすると緑
        move_vertex(bpy.data.objects[mmd_model.BODY_NAME], 0, (0.01, 0.0, 0.0))
        changed = self.catalog.call("versions.compare_with_previous", {})
        self.assertEqual(changed["measurement"]["signal"], "red")
        self.assertEqual([(c["object"], c["category"], c["outside_allowed"]) for c in changed["changes"]],
                         [(mmd_model.BODY_NAME, "vertices", True)])
        self.assertEqual(len(changed["lines"]), 1)
        allowed = self.catalog.call("versions.compare_with_previous",
                                    {"allowed": [{"object": mmd_model.BODY_NAME, "categories": ["vertices"]}]})
        self.assertEqual(allowed["measurement"]["signal"], "green")
        everything = self.catalog.call("versions.compare_with_previous", {"allowed": [{"object": mmd_model.BODY_NAME, "all": True}]})
        self.assertEqual(everything["measurement"]["signal"], "green")
        second = self.catalog.call("versions.save", {})
        self.assertEqual(second["name"], "model_v002.blend")
        self.assertEqual([v["number"] for v in self.catalog.call("versions.list", {})["versions"]], [1, 2])

    def test_bad_params_are_refused_before_running(self):
        with self.assertRaises(InvalidParamsError):
            self.catalog.call("versions.compare_with_previous", {"allowed": [{"object": "Body", "categories": ["colour"]}]})
        with self.assertRaises(InvalidParamsError):
            self.catalog.call("versions.save", {"base_path": 1})
        with self.assertRaises(ValueError):
            self.catalog.call("versions.compare_with_previous", {"allowed": [{"object": "Body"}]})  # 種類も all も無い

    def test_unsaved_file_is_refused(self):
        bpy.ops.wm.read_homefile(use_empty=True)
        mmd_model.build_model()
        with self.assertRaises(ValueError):
            self.catalog.call("versions.save", {})
        with self.assertRaises(ValueError):
            self.catalog.call("versions.list", {})


class CourseToolThroughWorkerTest(unittest.TestCase):
    """確かめの動き（motion.run_course）を、本物の裏の Blender で動かす（ポーズ 2 つ、見え方なし）"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.blend = open_fresh_model(self.folder)

    def tearDown(self):
        bpy.ops.wm.read_homefile(use_empty=True)
        self.temp.cleanup()

    def test_run_course_in_background_blender(self):
        poses = list(POSE_NAMES[:2])
        params = {"poses": poses, "visibility": False}
        self.assertEqual(build_catalog().validate_call("motion.run_course", params), [])
        digest = provenance.file_digest(self.blend).sha256
        queue = launcher_bpy.make_queue(self.folder / "jobs")
        job = launcher_bpy.make_job("motion.run_course", params, input_blend=self.blend)
        queue.enqueue(job)
        self.assertEqual(queue.start_next().job_id, job.job_id)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            final = queue.poll()
            if queue.running_job is None:
                break
            time.sleep(0.5)
        else:
            self.fail("裏の Blender が時間内に終わりませんでした")
        self.assertEqual((final.status, final.error, final.error_key), ("done", None, None), queue.log_text(job.job_id))
        self.assertEqual(final.total, len(poses))  # ポーズごとに進み具合を書く
        data = worker.read_result(queue.job_dir(job.job_id))
        self.assertEqual(data["tool_id"], "motion.run_course")
        events = data["result"]["events"]
        self.assertEqual([e["pose"] for e in events if e["check"] != "app_motion"],
                         [pose for pose in poses for _ in CHECKS])
        self.assertEqual([e["check"] for e in events if e["check"] == "app_motion"], ["app_motion"])
        self.assertEqual(provenance.file_digest(self.blend).sha256, digest)  # 入力は変わらない
        # 結果から報告を作れる（画面の「結果を見る」と同じ）
        report = report_from_tool_result("motion.run_course", data["result"], source={"job_id": job.job_id})
        self.assertEqual(report.kind, "pose_course")
        self.assertEqual(len(report.items), len(events))
        self.assertEqual(report.items[0].item_id, f"pose_course/{poses[0]}/{CHECKS[0]}")


@unittest.skipUnless(KOUMAKAN, "環境変数 BLENDER_AI_TOOLS_KOUMAKAN_REPO（紅魔館のリポジトリの場所）がありません")
class ExportToolTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.blend = open_fresh_model(self.folder)
        self.catalog = build_catalog()
        self.version = Path(self.catalog.call("versions.save", {})["version"])

    def tearDown(self):
        bpy.ops.wm.read_homefile(use_empty=True)
        self.temp.cleanup()

    def params(self, output: str) -> dict:
        return {"input_blend": str(self.version), "output_dir": str(self.folder / output), "koumakan_repo": KOUMAKAN,
                "blender": ""}

    def test_convert_and_check_directly(self):
        digest = provenance.file_digest(self.version).sha256
        result = self.catalog.call("export.convert_and_check", self.params("out"))
        self.assertTrue(result["convert"]["ok"], result["convert"]["problems"])
        self.assertEqual(result["convert"]["summary"]["vertices_limited"], 4)
        self.assertTrue((self.folder / "out" / "character.usdz").is_file())
        names = [m["name"] for m in result["roundtrip"]]
        self.assertIn("roundtrip_deviation", names)
        self.assertIn("influence_limit_effect", names)
        effect = next(m for m in result["roundtrip"] if m["name"] == "influence_limit_effect")
        self.assertEqual(effect["signal"], "yellow")  # 4本に絞った影響：試し用モデルでは約 0.46 mm
        self.assertEqual(provenance.file_digest(self.version).sha256, digest)  # 入力は変わらない
        self.assertEqual(bpy.data.filepath, str(self.blend))

    def test_convert_and_check_through_background_worker(self):
        queue = launcher_bpy.make_queue(self.folder / "jobs")
        job = launcher_bpy.make_job("export.convert_and_check", self.params("out_bg"), input_blend=self.version)
        queue.enqueue(job)
        self.assertEqual(queue.start_next().job_id, job.job_id)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            final = queue.poll()
            if queue.running_job is None:
                break
            time.sleep(0.5)
        else:
            self.fail("裏の Blender が時間内に終わりませんでした")
        self.assertEqual((final.status, final.error, final.error_key), ("done", None, None), queue.log_text(job.job_id))
        self.assertEqual((final.step, final.total), (2, 2))
        data = worker.read_result(queue.job_dir(job.job_id))
        self.assertTrue(data["result"]["convert"]["ok"])
        self.assertIn("influence_limit_effect", [m["name"] for m in data["result"]["roundtrip"]])

    def test_refuses_app_model_folder(self):
        app_model = Path(KOUMAKAN) / "KoumakanApp" / "KoumakanApp" / "Model"
        params = self.params("x")
        params["output_dir"] = str(app_model)
        with self.assertRaises(ValueError):
            self.catalog.call("export.convert_and_check", params)


if __name__ == "__main__":
    unittest.main()

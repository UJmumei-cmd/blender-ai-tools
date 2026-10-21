"""裏で動かす係（A10）を、本物の別の Blender で確かめる"""

import contextlib
import io
import os
import stat
import tempfile
import time
import unittest
from pathlib import Path

import bpy

from blender_ai_tools import provenance
from blender_ai_tools.jobs import launcher_bpy, progress, worker
from tests.fixtures import mmd_model


def wait_for(queue, *, limit=300.0, interval=0.5):
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        result = queue.poll()
        if queue.running_job is None:
            return result
        time.sleep(interval)
    raise AssertionError("裏の Blender が時間内に終わりませんでした")


class LauncherTest(unittest.TestCase):
    def test_worker_script_and_blender(self):
        self.assertTrue(launcher_bpy.worker_script().is_file())
        self.assertEqual(launcher_bpy.blender_binary(), bpy.app.binary_path)
        self.assertEqual(launcher_bpy.blender_binary(""), bpy.app.binary_path)
        with self.assertRaises(ValueError):
            launcher_bpy.blender_binary("/nowhere/blender")

    def test_blender_binary_accepts_mac_app_bundle(self):
        """設定に Blender.app（フォルダ）が書いてあれば、その中の実行ファイルを使う"""
        with tempfile.TemporaryDirectory() as folder:
            bundle = Path(folder) / "Blender.app"
            binary = bundle / "Contents" / "MacOS" / "Blender"
            binary.parent.mkdir(parents=True)
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
            self.assertEqual(launcher_bpy.blender_binary(bundle), str(binary))
            self.assertEqual(launcher_bpy.blender_binary(binary), str(binary))
            with self.assertRaises(ValueError):
                launcher_bpy.blender_binary(Path(folder) / "Other.app")

    def test_default_jobs_root(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(launcher_bpy.default_jobs_root(folder), Path(folder) / "jobs")

    def test_make_job_follows_autoexec_preference(self):
        job = launcher_bpy.make_job("bones.map_names", {"names": ["左腕"]}, input_blend=None)
        self.assertEqual(job.autoexec, bool(bpy.context.preferences.filepaths.use_scripts_auto_execute))
        self.assertEqual(job.tool_id, "bones.map_names")
        self.assertIsNone(job.input_blend)

    def test_make_job_makes_input_absolute(self):
        """裏の Blender は別の cwd で動くので、入力の場所は絶対にして渡す（「//」の相対の場所も）"""
        job = launcher_bpy.make_job("bones.map_names", {"names": ["左腕"]}, input_blend="model_v001.blend")
        self.assertTrue(os.path.isabs(job.input_blend))
        self.assertEqual(job.input_blend, os.path.abspath("model_v001.blend"))
        job = launcher_bpy.make_job("bones.map_names", {"names": ["左腕"]}, input_blend=Path("/tmp") / "x.blend")
        self.assertEqual(job.input_blend, os.path.abspath(os.path.join("/tmp", "x.blend")))
        relative = launcher_bpy.make_job("bones.map_names", {"names": ["左腕"]}, input_blend="//x.blend")
        self.assertTrue(os.path.isabs(relative.input_blend))
        self.assertNotIn("//x", relative.input_blend.replace(os.sep, "/")[1:])  # 「//」が解かれている

    def test_tick_survives_exceptions(self):
        """タイマーの1回で例外が起きても、次の間隔を返す（例外を上げると Blender がタイマーを外し、順番待ちが止まる）"""
        class BrokenQueue:
            def __init__(self):
                self.polls = 0

            def poll(self):
                self.polls += 1
                raise OSError("読めません")

            def start_next(self):
                raise AssertionError("poll が失敗したら、この回は start_next を呼ばない")

        updates = []
        broken = BrokenQueue()
        tick = launcher_bpy.make_tick(broken, interval=0.25, on_update=lambda: updates.append(1))
        with contextlib.redirect_stderr(io.StringIO()) as captured:
            self.assertEqual(tick(), 0.25)
            self.assertEqual(tick(), 0.25)
        self.assertEqual(broken.polls, 2)
        self.assertIn("読めません", captured.getvalue())
        self.assertEqual(updates, [])

        class QuietQueue:
            def poll(self):
                return None

            def start_next(self):
                return None

        tick = launcher_bpy.make_tick(QuietQueue(), interval=0.5, on_update=lambda: updates.append(1))
        self.assertEqual(tick(), 0.5)
        self.assertEqual(updates, [1])

    def test_start_and_stop_polling(self):
        with tempfile.TemporaryDirectory() as folder:
            queue = launcher_bpy.make_queue(Path(folder) / "jobs")
            stop = launcher_bpy.start_polling(queue)
            stop()
            stop()  # 2回呼んでも大丈夫


class RealWorkerTest(unittest.TestCase):
    """本物の別の Blender で、目録の道具を動かす"""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        bpy.ops.wm.read_homefile(use_empty=True)
        mmd_model.build_model()
        cls.blend = Path(cls.temp.name) / "model.blend"
        mmd_model.save_model(cls.blend)
        cls.digest = provenance.file_digest(cls.blend).sha256

    def test_tool_runs_in_background_blender(self):
        queue = launcher_bpy.make_queue(Path(self.temp.name) / "jobs")
        job = launcher_bpy.make_job("bones.map_names", {"names": ["J_Bip_C_Hips", "左腕", "謎の骨"]},
                                    input_blend=self.blend)
        folder = queue.enqueue(job)
        self.assertEqual(queue.start_next().job_id, job.job_id)
        final = wait_for(queue)
        self.assertEqual((final.status, final.error, final.error_key), ("done", None, None), queue.log_text(job.job_id))
        data = worker.read_result(folder)
        self.assertEqual(data["result"]["mapping"]["J_Bip_C_Hips"], "センター")
        self.assertEqual(data["result"]["unknown"], ["謎の骨"])
        self.assertEqual(data["environment"]["blender_version"], bpy.app.version_string)
        self.assertEqual(data["input"]["sha256"], self.digest)
        self.assertEqual(provenance.file_digest(self.blend).sha256, self.digest)  # 入力は変わらない
        self.assertIn("Blender", queue.log_text(job.job_id))

    def test_missing_input_fails_quickly(self):
        queue = launcher_bpy.make_queue(Path(self.temp.name) / "jobs2")
        job = launcher_bpy.make_job("bones.map_names", {"names": ["左腕"]}, input_blend=Path(self.temp.name) / "none.blend")
        queue.enqueue(job)
        queue.start_next()
        final = wait_for(queue)
        self.assertEqual((final.status, final.error_key), ("failed", "jobs.error_input_missing"))
        self.assertIsNone(worker.read_result(queue.job_dir(job.job_id)))

    def test_unknown_tool_fails_in_worker(self):
        queue = launcher_bpy.make_queue(Path(self.temp.name) / "jobs3")
        job = launcher_bpy.make_job("bones.nothing", {}, input_blend=None)
        queue.enqueue(job)
        queue.start_next()
        final = wait_for(queue)
        self.assertEqual(final.status, "failed")
        self.assertIn("bones.nothing", final.error)


if __name__ == "__main__":
    unittest.main()

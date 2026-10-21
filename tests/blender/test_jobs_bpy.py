"""裏で動かす係（A10）を、本物の別の Blender で確かめる"""

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

    def test_default_jobs_root(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(launcher_bpy.default_jobs_root(folder), Path(folder) / "jobs")

    def test_make_job_follows_autoexec_preference(self):
        job = launcher_bpy.make_job("bones.map_names", {"names": ["左腕"]}, input_blend=None)
        self.assertEqual(job.autoexec, bool(bpy.context.preferences.filepaths.use_scripts_auto_execute))
        self.assertEqual(job.tool_id, "bones.map_names")

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

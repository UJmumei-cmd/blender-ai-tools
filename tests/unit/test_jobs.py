"""裏で動かす係（A10）：仕事の形、進み具合、裏の係の中身、順番待ちを確かめる（Blender なし）"""

import contextlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from blender_ai_tools import provenance, thresholds
from blender_ai_tools.jobs import progress as prog
from blender_ai_tools.jobs import spec, worker
from blender_ai_tools.jobs.queue import JobQueue, WorkerProcess, build_worker_command
from blender_ai_tools.measure.results import Signal
from tests.fixtures.fake_worker import build_fake_catalog

FAKE_WORKER = Path(__file__).resolve().parents[1] / "fixtures" / "fake_worker.py"
TIME = "2026-10-20T19:19:00Z"


def make_job(tool_id="fake.echo", params=None, *, job_id="job_a", input_blend=None, created_at=TIME, **kwargs):
    return spec.JobSpec(job_id=job_id, tool_id=tool_id, params={"value": "こんにちは"} if params is None else params,
                        input_blend=input_blend, created_at=created_at, **kwargs)


# MARK: - 仕事の形


class JobSpecTest(unittest.TestCase):
    def test_round_trip(self):
        job = make_job(input_blend="/tmp/x.blend", confirmed=True, autoexec=True)
        text = job.to_json()
        self.assertEqual(spec.JobSpec.from_json(text), job)
        self.assertEqual(json.loads(text)["schema_version"], spec.SCHEMA_VERSION)
        self.assertTrue(text.endswith("\n"))

    def test_new_job_id(self):
        job_id = spec.new_job_id("2026-10-20T19:19:00Z")
        self.assertRegex(job_id, r"^20261020T191900Z_[0-9a-f]{6}$")
        self.assertNotEqual(spec.new_job_id(), spec.new_job_id())
        with self.assertRaises(ValueError):
            spec.new_job_id("きのう")

    def test_rejects_bad_values(self):
        cases = {
            "id": dict(job_id="a b"), "id_long": dict(job_id="x" * 65), "tool": dict(tool_id="MapNames"),
            "tool_no_dot": dict(tool_id="bones"), "nan": dict(params={"x": math.nan}),
            "param_key": dict(params={1: 2}), "param_type": dict(params={"x": {1, 2}}), "params": dict(params=[1]),
            "time": dict(created_at="2026-10-20 19:19"), "input": dict(input_blend=""),
            "confirmed": dict(confirmed=1), "schema": dict(schema_version=2),
        }
        for label, kwargs in cases.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                make_job(**kwargs)

    def test_from_json_is_strict(self):
        good = make_job().to_json()
        with self.assertRaises(ValueError):
            spec.JobSpec.from_json(good.replace('"job_id"', '"job_id": "b", "job_id"'))  # 同じキーが2回
        with self.assertRaises(ValueError):
            spec.JobSpec.from_data({**json.loads(good), "code": "import os"})  # 知らない項目
        with self.assertRaises(ValueError):
            spec.JobSpec.from_data({k: v for k, v in json.loads(good).items() if k != "tool_id"})
        with self.assertRaises(ValueError):
            spec.strict_json_loads('{"x": NaN}')

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as folder:
            job = make_job()
            spec.save_job(job, folder)
            self.assertEqual(spec.load_job(folder), job)
            with self.assertRaises(ValueError):
                spec.save_job(job, folder)  # 書き換えない
            with self.assertRaises(ValueError):
                spec.load_job(Path(folder) / "nothing")


# MARK: - 進み具合


class ProgressTest(unittest.TestCase):
    def test_round_trip_and_finished(self):
        running = prog.Progress(status="running", updated_at=TIME, step=2, total=5, message_key="jobs.step_of",
                                message_values={"step": 2, "total": 5}, fraction=0.4, started_at=TIME, pid=12)
        self.assertEqual(prog.Progress.from_json(running.to_json()), running)
        self.assertFalse(running.is_finished)
        done = prog.Progress(status="done", updated_at=TIME, finished_at=TIME, fraction=1.0)
        self.assertTrue(done.is_finished)

    def test_rejects_bad_values(self):
        cases = {
            "status": dict(status="paused", updated_at=TIME),
            "fraction": dict(status="running", updated_at=TIME, fraction=1.5),
            "fraction_bool": dict(status="running", updated_at=TIME, fraction=True),
            "no_finished_at": dict(status="done", updated_at=TIME),
            "step": dict(status="running", updated_at=TIME, step=-1),
            "time": dict(status="queued", updated_at="now"),
            "pid": dict(status="running", updated_at=TIME, pid=-1),
            "values": dict(status="running", updated_at=TIME, message_values={"x": math.inf}),
        }
        for label, kwargs in cases.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                prog.Progress(**kwargs)
        with self.assertRaises(ValueError):
            prog.Progress.from_data({"status": "queued", "updated_at": TIME, "extra": 1})

    def test_write_read_cancel(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(prog.read_progress(folder))
            queued = prog.Progress(status="queued", updated_at=TIME)
            prog.write_progress(folder, queued)
            self.assertEqual(prog.read_progress(folder), queued)
            self.assertEqual([p.name for p in Path(folder).iterdir()], [prog.PROGRESS_FILE])  # 一時ファイルが残らない
            self.assertFalse(prog.cancel_requested(folder))
            prog.request_cancel(folder)
            prog.request_cancel(folder)  # 2回でも同じ
            self.assertTrue(prog.cancel_requested(folder))
            (Path(folder) / prog.PROGRESS_FILE).write_text("{broken", encoding="utf-8")
            with self.assertRaises(ValueError):
                prog.read_progress(folder)

    def test_reporter(self):
        with tempfile.TemporaryDirectory() as folder:
            reporter = prog.Reporter(folder, pid=7)
            started = reporter.start()
            self.assertEqual((started.status, started.pid), ("running", 7))
            self.assertEqual(reporter.report(1, 4).fraction, 0.25)
            self.assertIsNone(reporter.report(1, 0).fraction)  # 全部の段の数が分からない
            self.assertEqual(reporter.report(9, 4).fraction, 1.0)  # 超えても 1 まで
            with self.assertRaises(ValueError):
                reporter.report(-1, 4)
            finished = reporter.finish("failed", error="x", error_key="jobs.error_cancelled")
            self.assertEqual((finished.status, finished.pid, finished.error), ("failed", None, "x"))
            self.assertIsNotNone(finished.finished_at)
            self.assertEqual(prog.read_progress(folder), finished)
            with self.assertRaises(ValueError):
                reporter.finish("running")
            prog.request_cancel(folder)
            with self.assertRaises(prog.JobCancelled):
                reporter.report(2, 4)

    def test_module_report_only_inside_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            prog.report(1, 2)  # 画面の中では何もしない
            prog.check_cancel()
            self.assertFalse(prog.is_running_in_worker())
            self.assertIsNone(prog.read_progress(folder))
            reporter = prog.Reporter(folder)
            with prog.current_reporter(reporter):
                self.assertTrue(prog.is_running_in_worker())
                prog.report(1, 2, "jobs.step_of", {"step": 1, "total": 2})
                self.assertEqual(prog.read_progress(folder).message_values, {"step": 1, "total": 2})
                prog.request_cancel(folder)
                with self.assertRaises(prog.JobCancelled):
                    prog.check_cancel()
            self.assertFalse(prog.is_running_in_worker())


# MARK: - 裏の係の中身（同じプロセスで）


class WorkerTest(unittest.TestCase):
    def setUp(self):
        self.catalog = build_fake_catalog()
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.input = self.folder / "model.blend"
        self.input.write_bytes(b"BLENDER-fake")

    def tearDown(self):
        self.temp.cleanup()

    def run_job(self, job, **kwargs):
        job_dir = self.folder / job.job_id
        with contextlib.redirect_stdout(io.StringIO()):  # 例外の詳しい文（本物では log.txt に行く）を、テストの出力に混ぜない
            return worker.run_job(job, self.catalog, job_dir, **kwargs), job_dir

    def test_done_writes_result(self):
        opened = []
        job = make_job(input_blend=str(self.input))
        result, job_dir = self.run_job(job, open_input=opened.append, environment={"blender_version": "5.2.2"})
        self.assertEqual(result.status, "done")
        self.assertEqual(result.fraction, 1.0)
        self.assertEqual(opened, [self.input])
        data = worker.read_result(job_dir)
        self.assertEqual(data["tool_id"], "fake.echo")
        self.assertEqual(data["tool_version"], "1")
        self.assertEqual(data["result"]["value"], "こんにちは")
        self.assertEqual(data["result"]["measurement"]["signal"], "green")  # Measurement は to_data の形
        self.assertEqual(data["input"]["sha256"], provenance.file_digest(self.input).sha256)
        self.assertEqual(data["environment"]["blender_version"], "5.2.2")
        self.assertIn("python_version", data["environment"])
        self.assertEqual(data["catalog_fingerprint"], self.catalog.fingerprint())
        self.assertEqual(prog.read_progress(job_dir), result)
        self.assertEqual(self.input.read_bytes(), b"BLENDER-fake")  # 入力は変わらない

    def test_no_input(self):
        result, job_dir = self.run_job(make_job())
        self.assertEqual(result.status, "done")
        self.assertIsNone(worker.read_result(job_dir)["input"])

    def test_failures(self):
        cases = {
            "raises": (make_job("fake.fail", {}), "ValueError"),
            "unknown": (make_job("fake.nothing", {}), "fake.nothing"),
            "bad_params": (make_job("fake.echo", {"value": 1}), "InvalidParamsError"),
            "needs_confirmation": (make_job("fake.modify", {}), "ConfirmationRequiredError"),
        }
        for label, (job, expected) in cases.items():
            with self.subTest(label=label):
                result, job_dir = self.run_job(job)
                self.assertEqual(result.status, "failed")
                self.assertIn(expected, result.error)
                self.assertIsNone(worker.read_result(job_dir))  # 失敗のときは結果を書かない

    def test_confirmed_modify_runs(self):
        result, _ = self.run_job(make_job("fake.modify", {}, confirmed=True))
        self.assertEqual(result.status, "done")

    def test_missing_and_changed_input(self):
        result, _ = self.run_job(make_job(input_blend=str(self.folder / "none.blend")))
        self.assertEqual((result.status, result.error_key), ("failed", "jobs.error_input_missing"))
        job = make_job("fake.touch_input", {"path": str(self.input)}, input_blend=str(self.input))
        result, job_dir = self.run_job(job)
        self.assertEqual((result.status, result.error_key), ("failed", "jobs.error_input_changed"))
        self.assertEqual(result.error_values["path"], str(self.input))
        self.assertIsNone(worker.read_result(job_dir))

    def test_cancelled_before_start(self):
        job = make_job("fake.sleep", {"steps": 3, "seconds": 0.0})
        job_dir = self.folder / job.job_id
        prog.request_cancel(job_dir)
        result = worker.run_job(job, self.catalog, job_dir)
        self.assertEqual((result.status, result.error_key), ("cancelled", "jobs.error_cancelled"))

    def test_jsonable(self):
        from dataclasses import dataclass

        @dataclass
        class Pair:
            a: int
            b: tuple

        value = worker.jsonable({"p": Pair(1, (2, 3)), "m": [Signal.RED]})
        self.assertEqual(value, {"p": {"a": 1, "b": [2, 3]}, "m": [Signal.RED]})


# MARK: - 順番待ち（本物の別プロセス：テスト用の裏の係）


def python_runner(job, job_dir):
    log = open(job_dir / prog.LOG_FILE, "ab")
    process = subprocess.Popen([sys.executable, "-I", str(FAKE_WORKER), str(job_dir)], stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, cwd=job_dir)
    return WorkerProcess(process=process, log=log)


class JobQueueTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "jobs"

    def tearDown(self):
        self.temp.cleanup()

    def wait(self, queue, *, limit=60.0, interval=0.05):
        """動いている仕事が終わるまで poll を繰り返す。終わりの進み具合を返す"""
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            progress = queue.poll()
            if queue.running_job is None:  # プロセスが片付いた（「終わり」を書いただけでは、まだ動いている）
                return progress
            time.sleep(interval)
        self.fail("仕事が時間内に終わりませんでした")

    def wait_for_step(self, queue, step, *, limit=30.0):
        """動いている仕事が step 段まで進むのを待つ（先に終わったら失敗）"""
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            progress = queue.poll()
            self.assertIsNotNone(progress, "仕事が動いていません")
            self.assertFalse(progress.is_finished, f"段 {step} の前に終わりました：{progress}")
            if progress.step >= step:
                return progress
            time.sleep(0.02)
        self.fail(f"段 {step} まで進みませんでした")

    def test_build_worker_command(self):
        command = build_worker_command("/app/blender", "/x/worker_main_bpy.py", "/jobs/a", autoexec=False)
        self.assertEqual(command, ["/app/blender", "--background", "--factory-startup", "--disable-autoexec",
                                   "--python-exit-code", "1", "--python", "/x/worker_main_bpy.py", "--", "/jobs/a"])
        self.assertIn("--enable-autoexec", build_worker_command("b", "w", "d", autoexec=True))
        self.assertLess(command.index("--python-exit-code"), command.index("--python"))

    def test_run_one_job_end_to_end(self):
        queue = JobQueue(self.root, python_runner)
        job = make_job("fake.sleep", {"steps": 3, "seconds": 0.05}, job_id="one")
        folder = queue.enqueue(job)
        self.assertEqual(prog.read_progress(folder).status, "queued")
        with self.assertRaises(ValueError):
            queue.enqueue(job)  # 同じ id
        self.assertEqual(queue.start_next().job_id, "one")
        self.assertIsNone(queue.start_next())  # 同時に動かすのは1つ
        final = self.wait(queue)
        self.assertEqual((final.status, final.step, final.total, final.fraction), ("done", 3, 3, 1.0))
        self.assertIsNone(queue.running_job)
        self.assertEqual(worker.read_result(folder)["result"], {"steps": 3})
        self.assertTrue((folder / prog.LOG_FILE).exists())
        self.assertIsNone(queue.poll())

    def test_jobs_run_one_at_a_time_in_order(self):
        queue = JobQueue(self.root, python_runner)
        queue.enqueue(make_job("fake.echo", {"value": "b"}, job_id="b", created_at="2026-10-20T19:19:02Z"))
        queue.enqueue(make_job("fake.echo", {"value": "a"}, job_id="a", created_at="2026-10-20T19:19:01Z"))
        self.assertEqual([j.job_id for j, _ in queue.jobs()], ["a", "b"])
        self.assertEqual(queue.start_next().job_id, "a")
        self.assertEqual(self.wait(queue).status, "done")
        self.assertEqual(queue.start_next().job_id, "b")
        self.assertEqual(self.wait(queue).status, "done")
        self.assertIsNone(queue.start_next())
        self.assertEqual([p.status for _, p in queue.jobs()], ["done", "done"])

    def test_cancel_queued_and_running(self):
        queue = JobQueue(self.root, python_runner, grace=30.0)
        queue.enqueue(make_job("fake.sleep", {"steps": 100, "seconds": 0.05}, job_id="slow"))
        queue.enqueue(make_job("fake.echo", {"value": "x"}, job_id="later", created_at="2026-10-20T19:19:01Z"))
        cancelled = queue.cancel("later")
        self.assertEqual((cancelled.status, cancelled.error_key), ("cancelled", "jobs.error_cancelled"))
        self.assertEqual(queue.start_next().job_id, "slow")
        self.wait_for_step(queue, 2)
        queue.cancel("slow")
        self.assertTrue(queue.is_cancel_requested("slow"))
        started = time.monotonic()
        final = self.wait(queue)
        self.assertEqual((final.status, final.error_key), ("cancelled", "jobs.error_cancelled"))
        self.assertLess(time.monotonic() - started, 10.0)  # 裏の係が自分で止まった（terminate を待たない）
        self.assertIsNone(worker.read_result(queue.job_dir("slow")))
        self.assertEqual(queue.start_next(), None)  # later はもう中止ずみ
        with self.assertRaises(ValueError):
            queue.cancel("nothing")

    def test_process_death_is_failed(self):
        queue = JobQueue(self.root, python_runner)
        queue.enqueue(make_job("fake.crash", {"code": 3}, job_id="crash"))
        queue.start_next()
        final = self.wait(queue)
        self.assertEqual((final.status, final.error_key, final.error_values), ("failed", "jobs.error_process_died", {"code": 3}))
        self.assertIsNone(final.pid)

    def test_timeout_kills_long_job(self):
        clock = [0.0]
        queue = JobQueue(self.root, python_runner, clock=lambda: clock[0], timeout=5.0, grace=1.0)
        queue.enqueue(make_job("fake.sleep", {"steps": 1000, "seconds": 0.05}, job_id="long"))
        queue.start_next()
        self.wait_for_step(queue, 1)
        clock[0] = 6.0  # 時間の上限を過ぎた
        final = self.wait(queue)
        self.assertEqual((final.status, final.error_key), ("failed", "jobs.error_timeout"))
        self.assertEqual(final.error_values, {"seconds": 5})

    def test_terminate_when_worker_ignores_cancel(self):
        """中止を無視する係（進み具合を書かない）は、猶予のあとに terminate される"""
        def stubborn_runner(job, job_dir):
            process = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(60)"],
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return WorkerProcess(process=process)

        clock = [0.0]
        queue = JobQueue(self.root, stubborn_runner, clock=lambda: clock[0], timeout=100.0, grace=1.0)
        queue.enqueue(make_job("fake.sleep", {"steps": 1, "seconds": 0}, job_id="stubborn"))
        queue.start_next()
        queue.cancel("stubborn")
        clock[0] = 2.0  # 猶予を過ぎた → terminate
        final = self.wait(queue, limit=20.0)
        self.assertEqual((final.status, final.error_key), ("cancelled", "jobs.error_cancelled"))

    def test_start_failure_is_recorded(self):
        def broken_runner(job, job_dir):
            raise OSError("blender が無い")

        queue = JobQueue(self.root, broken_runner)
        queue.enqueue(make_job(job_id="x"))
        self.assertIsNone(queue.start_next())
        progress = queue.progress_of("x")
        self.assertEqual((progress.status, progress.error_key), ("failed", "jobs.error_start_failed"))
        self.assertIn("blender が無い", progress.error_values["reason"])

    def test_remove_and_unreadable(self):
        queue = JobQueue(self.root, python_runner)
        queue.enqueue(make_job(job_id="done_job"))
        queue.enqueue(make_job(job_id="queued_job", created_at="2026-10-20T19:19:01Z"))
        with self.assertRaises(ValueError):
            queue.remove("queued_job")  # 終わっていない
        queue.start_next()
        with self.assertRaises(ValueError):
            queue.remove("done_job")  # 動いている
        self.wait(queue)
        queue.remove("done_job")
        self.assertFalse(queue.job_dir("done_job").exists())
        broken = self.root / "broken"
        broken.mkdir()
        (broken / spec.JOB_FILE).write_text("{", encoding="utf-8")
        self.assertEqual([j.job_id for j, _ in queue.jobs()], ["queued_job"])
        self.assertIn("broken", queue.unreadable)
        with self.assertRaises(ValueError):
            queue.remove("nothing")

    def test_recover_marks_orphans(self):
        queue = JobQueue(self.root, python_runner)
        queue.enqueue(make_job(job_id="orphan"))
        prog.write_progress(queue.job_dir("orphan"), prog.Progress(status="running", updated_at=TIME, pid=99999))
        self.assertEqual(queue.recover(), ["orphan"])
        progress = queue.progress_of("orphan")
        self.assertEqual((progress.status, progress.error_key, progress.pid), ("failed", "jobs.error_orphaned", None))
        self.assertEqual(queue.recover(), [])

    def test_log_text(self):
        queue = JobQueue(self.root, python_runner)
        queue.enqueue(make_job("fake.fail", {}, job_id="f"))
        queue.start_next()
        final = self.wait(queue)
        self.assertEqual(final.status, "failed")
        self.assertIn("わざと失敗しました", queue.log_text("f"))  # 例外の詳しい文は log.txt に残る
        self.assertEqual(queue.log_text("nothing"), "")


class ThresholdsTest(unittest.TestCase):
    def test_defaults_come_from_thresholds(self):
        queue = JobQueue(Path(tempfile.mkdtemp()) / "q", python_runner)
        self.assertEqual(queue.timeout, thresholds.JOB_TIMEOUT_SECONDS)
        self.assertEqual(queue.grace, thresholds.JOB_CANCEL_GRACE_SECONDS)


if __name__ == "__main__":
    unittest.main()

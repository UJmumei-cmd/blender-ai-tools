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
from unittest import mock

import numpy as np

from blender_ai_tools import provenance, texts, thresholds
from blender_ai_tools.jobs import progress as prog
from blender_ai_tools.jobs import spec, worker
from blender_ai_tools.jobs.queue import JobQueue, WorkerProcess, build_worker_command, pid_alive
from blender_ai_tools.measure.results import Signal
from tests.fixtures.fake_worker import build_fake_catalog

FAKE_WORKER = Path(__file__).resolve().parents[1] / "fixtures" / "fake_worker.py"
JOBS_DIR = Path(__file__).resolve().parents[2] / "blender_ai_tools" / "jobs"
TIME = "2026-10-20T19:19:00Z"


def rendered(progress):
    """失敗や中止の理由を、texts の文にする（キーが無い・値が合わないと KeyError・ValueError）"""
    return texts.text(progress.error_key, **progress.error_values)


def dead_pid():
    """もう終わったプロセスの番号（生きている番号と間違えないため、本当に動かして終わらせた物を使う）"""
    process = subprocess.Popen([sys.executable, "-I", "-c", "pass"], stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    process.wait()
    return process.pid


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
        self.assertRegex(job_id, r"^20261020T191900Z_[0-9]{6}_[0-9a-f]{6}$")
        self.assertTrue(spec.is_job_id(job_id))
        self.assertNotEqual(spec.new_job_id(), spec.new_job_id())
        with self.assertRaises(ValueError):
            spec.new_job_id("きのう")

    def test_new_job_ids_sort_in_creation_order(self):
        """同じ秒に作った仕事の id は、作った順に並ぶ（順番待ちは created_at と job_id で並べるため。乱数だけだと順が狂う）"""
        ids = [spec.new_job_id(TIME) for _ in range(50)]
        self.assertEqual(sorted(ids), ids)
        self.assertEqual(len(set(ids)), 50)

    def test_is_job_id(self):
        for good in ("a", "job_1", "20261020T191900Z_000001_a1b2c3", "x" * 64):
            self.assertTrue(spec.is_job_id(good), good)
        for bad in ("", "a b", "../x", "a/b", ".", "..", "x" * 65, 1, None, "-a", "日本"):
            self.assertFalse(spec.is_job_id(bad), repr(bad))

    def test_rejects_bad_values(self):
        cases = {
            "id": dict(job_id="a b"), "id_long": dict(job_id="x" * 65), "tool": dict(tool_id="MapNames"),
            "tool_no_dot": dict(tool_id="bones"), "nan": dict(params={"x": math.nan}),
            "param_key": dict(params={1: 2}), "param_type": dict(params={"x": {1, 2}}), "params": dict(params=[1]),
            "time": dict(created_at="2026-10-20 19:19"), "input": dict(input_blend=""),
            "input_relative": dict(input_blend="model.blend"), "input_relative_dir": dict(input_blend="versions/m_v001.blend"),
            "confirmed": dict(confirmed=1), "schema": dict(schema_version=2),
        }
        for label, kwargs in cases.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                make_job(**kwargs)

    def test_input_blend_must_be_absolute(self):
        """裏の Blender は別の cwd（仕事のフォルダ）で動くので、相対の場所は別の物を指してしまう。絶対の場所だけを受け取る"""
        absolute = make_job(input_blend=str(Path(tempfile.gettempdir()) / "x.blend"))
        self.assertTrue(os.path.isabs(absolute.input_blend))
        with self.assertRaises(ValueError):
            make_job(input_blend="x.blend")
        with self.assertRaises(ValueError):
            spec.JobSpec.from_json(absolute.to_json().replace(absolute.input_blend, "x.blend"))

    def test_strict_json_rejects_huge_numbers_and_deep_nesting(self):
        with self.assertRaises(ValueError):
            spec.strict_json_loads('{"x": 1e999}')  # float にすると無限大になる数
        with self.assertRaises(ValueError):
            spec.strict_json_loads('{"x": -1e999}')
        with self.assertRaises(ValueError):
            spec.strict_json_loads("[" * 100000 + "]" * 100000)  # 入れ子が深すぎる（RecursionError を ValueError にする）
        self.assertEqual(spec.strict_json_loads('{"x": 1e300}'), {"x": 1e300})

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

    def test_atomic_write_retries_replace(self):
        """Windows では、読む側が開いている間 os.replace が PermissionError になる。少し待って何回か試す"""
        real_replace = os.replace
        failures = [2]

        def flaky_replace(src, dst):
            if failures[0] > 0:
                failures[0] -= 1
                raise PermissionError("使用中")
            real_replace(src, dst)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "progress.json"
            with mock.patch("blender_ai_tools.jobs.spec.os.replace", side_effect=flaky_replace), \
                    mock.patch("blender_ai_tools.jobs.spec.time.sleep"):
                spec.write_text_atomic(path, "ok")
            self.assertEqual(path.read_text(encoding="utf-8"), "ok")
            self.assertEqual([p.name for p in Path(folder).iterdir()], ["progress.json"])
            # 何回試しても置き換えられないときは、一時ファイルを消して PermissionError を上げる
            with mock.patch("blender_ai_tools.jobs.spec.os.replace", side_effect=PermissionError("使用中")), \
                    mock.patch("blender_ai_tools.jobs.spec.time.sleep"), self.assertRaises(PermissionError):
                spec.write_text_atomic(path, "again")
            self.assertEqual(path.read_text(encoding="utf-8"), "ok")
            self.assertEqual([p.name for p in Path(folder).iterdir()], ["progress.json"])


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

    def test_report_accepts_numpy_numbers(self):
        """道具は numpy の数で段を数えることがある。report() がそれで落ちると、仕事が丸ごと失敗になる"""
        with tempfile.TemporaryDirectory() as folder:
            reporter = prog.Reporter(folder)
            written = reporter.report(np.int64(2), np.int32(4), "jobs.step_of", {"step": np.int64(2), "total": np.float32(4)})
            self.assertEqual((written.step, written.total, written.fraction), (2, 4, 0.5))
            self.assertEqual(written.message_values, {"step": 2, "total": 4.0})
            self.assertEqual(prog.read_progress(folder), written)
            for bad in ((True, 4), (1.0, 4), (1, "4")):
                with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                    reporter.report(*bad)

    def test_cancel_is_not_an_ordinary_exception(self):
        """道具の中の except Exception で中止が握りつぶされないように、JobCancelled は Exception の子ではない"""
        self.assertFalse(issubclass(prog.JobCancelled, Exception))
        self.assertTrue(issubclass(prog.JobCancelled, BaseException))

    def test_module_report_only_inside_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            prog.report(1, 2)  # 画面の中では何もしない
            prog.check_cancel()
            self.assertFalse(prog.is_running_in_worker())
            self.assertIsNone(prog.current_job_dir())
            self.assertIsNone(prog.read_progress(folder))
            reporter = prog.Reporter(folder)
            with prog.current_reporter(reporter):
                self.assertTrue(prog.is_running_in_worker())
                self.assertEqual(prog.current_job_dir(), Path(folder))
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
        job = make_job("fake.sleep", {"steps": 3, "seconds": 0.0}, input_blend=str(self.input))
        job_dir = self.folder / job.job_id
        prog.request_cancel(job_dir)
        opened = []
        result = worker.run_job(job, self.catalog, job_dir, open_input=opened.append)
        self.assertEqual((result.status, result.error_key), ("cancelled", "jobs.error_cancelled"))
        self.assertEqual(rendered(result), "中止の頼みを受けて止めました")
        self.assertEqual(opened, [])  # 中止ずみなら、重い .blend を開かない

    def test_cancel_during_tool_is_not_swallowed(self):
        """except Exception で何でも握りつぶす道具でも、中止は通る"""
        job = make_job("fake.swallow", {"steps": 6, "cancel_at_step": 2})
        result, job_dir = self.run_job(job)
        self.assertEqual((result.status, result.error_key), ("cancelled", "jobs.error_cancelled"))
        self.assertLessEqual(result.step, 2)
        self.assertIsNone(worker.read_result(job_dir))

    def test_error_texts_render(self):
        """裏の係が書く理由のキーと値が、texts の文に入る"""
        result, _ = self.run_job(make_job(input_blend=str(self.folder / "none.blend")))
        self.assertIn("none.blend", rendered(result))
        result, _ = self.run_job(make_job("fake.touch_input", {"path": str(self.input)}, input_blend=str(self.input)))
        self.assertIn(str(self.input), rendered(result))

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

    def test_same_second_jobs_run_in_enqueue_order(self):
        """同じ秒に入れた仕事（書き出し → 確認 など）は、入れた順に動く"""
        queue = JobQueue(self.root, python_runner)
        ids = []
        for index in range(5):
            job = make_job("fake.echo", {"value": str(index)}, job_id=spec.new_job_id(TIME))
            queue.enqueue(job)
            ids.append(job.job_id)
        self.assertEqual([j.job_id for j, _ in queue.jobs()], ids)
        self.assertEqual(queue.start_next().job_id, ids[0])
        self.assertEqual(self.wait(queue).status, "done")

    @unittest.skipIf(sys.platform == "win32", "SIGTERM を無視する子は Windows では作れない")
    def test_kill_when_worker_ignores_terminate(self):
        """terminate（SIGTERM）も無視する係は、もう一度の猶予のあとに kill される"""
        ready = self.root.parent / "ready"
        processes = []

        def deaf_runner(job, job_dir):
            code = ("import pathlib, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    f"pathlib.Path({str(ready)!r}).write_text('ok'); time.sleep(60)")
            process = subprocess.Popen([sys.executable, "-I", "-c", code], stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processes.append(process)
            return WorkerProcess(process=process)

        clock = [0.0]
        queue = JobQueue(self.root, deaf_runner, clock=lambda: clock[0], timeout=100.0, grace=1.0)
        queue.enqueue(make_job(job_id="deaf"))
        queue.start_next()
        deadline = time.monotonic() + 30.0
        while not ready.exists() and time.monotonic() < deadline:  # 子が SIGTERM を無視する用意をするまで
            time.sleep(0.02)
        self.assertTrue(ready.exists(), "子が始まりませんでした")
        queue.cancel("deaf")
        clock[0] = 2.0
        queue.poll()  # terminate（無視される）
        time.sleep(0.3)
        self.assertIsNone(processes[0].poll(), "SIGTERM を無視する子が、terminate で消えてしまいました")
        clock[0] = 4.0
        queue.poll()  # kill
        final = self.wait(queue, limit=20.0)
        self.assertEqual((final.status, final.error_key), ("cancelled", "jobs.error_cancelled"))
        self.assertEqual(processes[0].returncode, -9)

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
        self.assertEqual(rendered(final), "裏の Blender が途中で止まりました（終了コード 3）")

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
        self.assertIn("blender が無い", rendered(progress))

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

    def test_log_text(self):
        queue = JobQueue(self.root, python_runner)
        queue.enqueue(make_job("fake.fail", {}, job_id="f"))
        queue.start_next()
        final = self.wait(queue)
        self.assertEqual(final.status, "failed")
        self.assertIn("わざと失敗しました", queue.log_text("f"))  # 例外の詳しい文は log.txt に残る
        self.assertEqual(queue.log_text("nothing"), "")


class PidAliveTest(unittest.TestCase):
    def test_pid_alive(self):
        self.assertTrue(pid_alive(os.getpid()))
        self.assertFalse(pid_alive(dead_pid()))
        for bad in (None, 0, -1, "12", True):
            self.assertFalse(pid_alive(bad), repr(bad))


class FakeProcess:
    """Popen の代わり。poll() で見るだけ（wait・communicate を呼ぶと画面が固まるので、呼んだら失敗にする）"""

    def __init__(self, *, ignore_terminate=False):
        self.pid = 4242
        self.returncode = None
        self.calls = []
        self.ignore_terminate = ignore_terminate

    def poll(self):
        return self.returncode

    def terminate(self):
        self.calls.append("terminate")
        if not self.ignore_terminate:
            self.returncode = -15

    def kill(self):
        self.calls.append("kill")
        self.returncode = -9

    def wait(self, timeout=None):
        raise AssertionError("poll() の中で wait() を呼んではいけない（画面が固まる）")

    communicate = wait


class FakeProcessQueueTest(unittest.TestCase):
    """時計と偽のプロセスで、中止 → terminate → kill の順と時間を正確に確かめる"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "jobs"
        self.clock = [0.0]
        self.fake = None

    def tearDown(self):
        self.temp.cleanup()

    def make_queue(self, *, ignore_terminate=False, timeout=100.0, grace=10.0):
        def runner(job, job_dir):
            self.fake = FakeProcess(ignore_terminate=ignore_terminate)
            return WorkerProcess(process=self.fake)

        return JobQueue(self.root, runner, clock=lambda: self.clock[0], timeout=timeout, grace=grace)

    def worker_writes(self, queue, job_id, **kwargs):
        prog.write_progress(queue.job_dir(job_id), prog.Progress(status="running", updated_at=TIME, started_at=TIME,
                                                                 pid=self.fake.pid, **kwargs))

    def test_cancel_terminate_kill_sequence(self):
        queue = self.make_queue(ignore_terminate=True)
        queue.enqueue(make_job(job_id="seq"))
        queue.start_next()
        self.worker_writes(queue, "seq", step=1, total=3)
        self.clock[0] = 1.0
        queue.cancel("seq")  # 中止を頼んだ時刻 1.0
        self.assertTrue(queue.is_cancel_requested("seq"))
        self.clock[0] = 10.9
        self.assertEqual(queue.poll().status, "running")
        self.assertEqual(self.fake.calls, [])  # 猶予（10 秒）の前は何もしない
        self.clock[0] = 11.0
        queue.poll()
        self.assertEqual(self.fake.calls, ["terminate"])
        self.clock[0] = 20.9
        queue.poll()
        self.assertEqual(self.fake.calls, ["terminate"])  # terminate を無視されても、もう一度の猶予までは待つ
        self.clock[0] = 21.0
        queue.poll()
        self.assertEqual(self.fake.calls, ["terminate", "kill"])
        self.assertIsNotNone(queue.running_job)
        final = queue.poll()  # 消えたのを見て片付ける
        self.assertIsNone(queue.running_job)
        self.assertEqual((final.status, final.error_key, final.step), ("cancelled", "jobs.error_cancelled", 1))
        self.assertEqual(final.pid, None)
        self.assertEqual(final, queue.progress_of("seq"))
        self.assertEqual(self.fake.calls, ["terminate", "kill"])  # 死んだあとに kill を重ねない

    def test_timeout_sequence(self):
        queue = self.make_queue(ignore_terminate=True, timeout=5.0, grace=2.0)
        queue.enqueue(make_job(job_id="slow"))
        queue.start_next()
        self.worker_writes(queue, "slow")
        self.clock[0] = 4.9
        queue.poll()
        self.assertFalse(queue.is_cancel_requested("slow"))
        self.clock[0] = 5.0
        queue.poll()
        self.assertTrue(queue.is_cancel_requested("slow"))  # 上限で、まず中止を頼む
        self.assertEqual(self.fake.calls, [])
        self.clock[0] = 7.0
        queue.poll()
        self.assertEqual(self.fake.calls, ["terminate"])
        self.clock[0] = 9.0
        queue.poll()
        self.assertEqual(self.fake.calls, ["terminate", "kill"])
        final = queue.poll()
        self.assertEqual((final.status, final.error_key, final.error_values), ("failed", "jobs.error_timeout", {"seconds": 5}))
        self.assertEqual(rendered(final), "時間の上限（5 秒）を過ぎたので止めました")

    def test_worker_finished_but_process_alive(self):
        """「終わり」を書いたあと、プロセスが消えるまでの間：中止を頼まず、次の仕事も始めない"""
        queue = self.make_queue()
        queue.enqueue(make_job(job_id="done_soon"))
        queue.enqueue(make_job(job_id="next", created_at="2026-10-20T19:19:01Z"))
        queue.start_next()
        now = provenance.now_iso()
        prog.write_progress(queue.job_dir("done_soon"), prog.Progress(status="done", updated_at=now, started_at=TIME,
                                                                      finished_at=now, fraction=1.0))
        self.assertEqual(queue.poll().status, "done")
        self.assertEqual(queue.running_job.job_id, "done_soon")
        self.assertIsNone(queue.start_next())  # プロセスが消えるまでは、次を始めない
        self.assertEqual(queue.cancel("done_soon").status, "done")
        self.assertFalse(queue.is_cancel_requested("done_soon"))  # 終わった仕事に中止の印を付けない
        self.clock[0] = 1000.0
        queue.poll()
        self.assertEqual(self.fake.calls, [])  # 終わった仕事を terminate しない
        with self.assertRaises(ValueError):
            queue.remove("done_soon")
        self.fake.returncode = 0
        self.assertEqual(queue.poll().status, "done")
        self.assertIsNone(queue.running_job)
        self.assertEqual(queue.start_next().job_id, "next")

    def test_timeout_terminates_process_lingering_after_done(self):
        """「終わり」を書いたのに消えないプロセスも、時間の上限と猶予のあとには止める（結果はもう書かれているので、done のまま）"""
        queue = self.make_queue(timeout=100.0, grace=10.0)
        queue.enqueue(make_job(job_id="linger"))
        queue.start_next()
        now = provenance.now_iso()
        prog.write_progress(queue.job_dir("linger"), prog.Progress(status="done", updated_at=now, started_at=TIME,
                                                                   finished_at=now, fraction=1.0))
        self.clock[0] = 100.0
        self.assertEqual(queue.poll().status, "done")
        self.assertFalse(queue.is_cancel_requested("linger"))  # 終わった仕事に中止の印は付けない
        self.assertEqual(self.fake.calls, [])
        self.clock[0] = 110.0
        queue.poll()
        self.assertEqual(self.fake.calls, ["terminate"])
        final = queue.poll()
        self.assertEqual((final.status, final.error_key), ("done", None))
        self.assertIsNone(queue.running_job)
        self.assertEqual(queue.progress_of("linger").status, "done")

    def test_poll_survives_unreadable_progress(self):
        """progress.json が一時的に読めなくても（Windows の共有違反など）、poll() は落ちずに次の回に続ける"""
        queue = self.make_queue()
        queue.enqueue(make_job(job_id="x"))
        queue.start_next()
        self.worker_writes(queue, "x", step=2, total=4)
        with mock.patch("blender_ai_tools.jobs.queue.read_progress", side_effect=OSError("使用中")):
            progress = queue.poll()
        self.assertEqual(progress.status, "running")
        self.assertEqual(queue.running_job.job_id, "x")
        self.assertEqual(queue.poll().step, 2)
        (queue.job_dir("x") / prog.PROGRESS_FILE).write_text("{broken", encoding="utf-8")
        self.assertEqual(queue.poll().status, "running")
        self.fake.returncode = 7
        final = queue.poll()
        self.assertEqual((final.status, final.error_key, final.error_values), ("failed", "jobs.error_process_died", {"code": 7}))
        self.assertEqual(rendered(final), "裏の Blender が途中で止まりました（終了コード 7）")

    def test_job_ids_are_checked(self):
        """仕事の id は英数字だけ。「../」のような id で、root の外のフォルダを見たり消したりしない"""
        queue = self.make_queue()
        outside = self.root.parent / "outside"
        outside.mkdir()
        spec.save_job(make_job(job_id="victim"), outside)
        now = provenance.now_iso()
        prog.write_progress(outside, prog.Progress(status="done", updated_at=now, finished_at=now))
        for bad in ("../outside", "a/b", "", ".", "..", "x" * 65):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    queue.job_dir(bad)
                with self.assertRaises(ValueError):
                    queue.remove(bad)
                with self.assertRaises(ValueError):
                    queue.cancel(bad)
                with self.assertRaises(ValueError):
                    queue.progress_of(bad)
        self.assertTrue((outside / spec.JOB_FILE).is_file())

    def test_enqueue_recreates_root(self):
        queue = self.make_queue()
        import shutil
        shutil.rmtree(self.root)
        self.assertEqual(queue.jobs(), [])  # root が消されていても落ちない
        self.assertIsNone(queue.start_next())
        self.assertEqual(queue.recover(), [])
        folder = queue.enqueue(make_job(job_id="a"))
        self.assertTrue((folder / spec.JOB_FILE).is_file())


class RecoverTest(unittest.TestCase):
    """画面を開き直したあと：前の回の裏の Blender が生きていれば待ち、死んでいれば失敗にする"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "jobs"
        self.sleeper = None

    def tearDown(self):
        if self.sleeper is not None and self.sleeper.poll() is None:
            self.sleeper.kill()
            self.sleeper.wait()
        self.temp.cleanup()

    def start_sleeper(self):
        self.sleeper = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(120)"],
                                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return self.sleeper.pid

    def running(self, pid):
        return prog.Progress(status="running", updated_at=TIME, started_at=TIME, step=1, total=4, pid=pid)

    def test_dead_orphan_is_failed(self):
        queue = JobQueue(self.root, python_runner)
        queue.enqueue(make_job(job_id="orphan"))
        prog.write_progress(queue.job_dir("orphan"), self.running(dead_pid()))
        queue.enqueue(make_job(job_id="no_pid"))
        prog.write_progress(queue.job_dir("no_pid"), prog.Progress(status="running", updated_at=TIME))
        self.assertEqual(queue.recover(), ["no_pid", "orphan"])
        for job_id in ("orphan", "no_pid"):
            progress = queue.progress_of(job_id)
            self.assertEqual((progress.status, progress.error_key, progress.pid), ("failed", "jobs.error_orphaned", None))
            self.assertIsNotNone(progress.finished_at)
            self.assertIn("結果はありません", rendered(progress))
        self.assertEqual(queue.recover(), [])

    def test_live_orphan_is_left_running_and_blocks_the_queue(self):
        """前の回の裏の Blender がまだ動いている：失敗にせず（あとで結果を書く）、終わるまで次の仕事を始めない（2つ同時に動かさない）"""
        pid = self.start_sleeper()
        queue = JobQueue(self.root, python_runner)
        queue.enqueue(make_job(job_id="orphan"))
        prog.write_progress(queue.job_dir("orphan"), self.running(pid))
        queue.enqueue(make_job("fake.echo", {"value": "x"}, job_id="waiting", created_at="2026-10-20T19:19:01Z"))
        self.assertEqual(queue.recover(), [])
        self.assertEqual(queue.progress_of("orphan").status, "running")
        self.assertIsNone(queue.start_next())  # 生きている間は始めない
        self.assertIsNone(queue.running_job)
        self.assertEqual(queue.progress_of("waiting").status, "queued")
        # 中止は頼める（cancel ファイルを置くだけ。裏の係が自分で止まる）
        self.assertEqual(queue.cancel("orphan").status, "running")
        self.assertTrue(queue.is_cancel_requested("orphan"))
        self.sleeper.kill()
        self.sleeper.wait()
        # 消えた（「終わり」を書かずに）：失敗にして、次の仕事を始める
        self.assertEqual(queue.start_next().job_id, "waiting")
        orphan = queue.progress_of("orphan")
        self.assertEqual((orphan.status, orphan.error_key), ("failed", "jobs.error_orphaned"))
        deadline = time.monotonic() + 60.0
        while queue.running_job is not None and time.monotonic() < deadline:
            queue.poll()
            time.sleep(0.05)
        self.assertEqual(queue.progress_of("waiting").status, "done")

    def test_recover_does_not_touch_own_running_job(self):
        queue = JobQueue(self.root, python_runner)
        queue.enqueue(make_job("fake.sleep", {"steps": 50, "seconds": 0.05}, job_id="mine"))
        queue.start_next()
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and queue.poll().status != "running":
            time.sleep(0.02)
        self.assertEqual(queue.recover(), [])
        self.assertEqual(queue.progress_of("mine").status, "running")
        queue.cancel("mine")
        while queue.running_job is not None and time.monotonic() < deadline:
            queue.poll()
            time.sleep(0.05)
        self.assertEqual(queue.progress_of("mine").status, "cancelled")


class TextsKeysTest(unittest.TestCase):
    def test_keys_used_in_jobs_exist_in_texts(self):
        """jobs/ のコードが使う文のキー（"jobs.…"）は、すべて texts/jobs.py にある"""
        used = set()
        for path in sorted(JOBS_DIR.glob("*.py")):
            used.update(re.findall(r'"(jobs\.[a-z_]+)"', path.read_text(encoding="utf-8")))
        self.assertTrue(used)
        self.assertEqual(sorted(used - set(texts.TEXTS)), [])


class ThresholdsTest(unittest.TestCase):
    def test_defaults_come_from_thresholds(self):
        queue = JobQueue(Path(tempfile.mkdtemp()) / "q", python_runner)
        self.assertEqual(queue.timeout, thresholds.JOB_TIMEOUT_SECONDS)
        self.assertEqual(queue.grace, thresholds.JOB_CANCEL_GRACE_SECONDS)


if __name__ == "__main__":
    unittest.main()

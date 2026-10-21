"""画面の側の順番待ち（bpy なし）：仕事を1つずつ別の Blender で始め、進み具合を読み、中止し、片付ける。

仕事のフォルダ：<root>/<job_id>/ に job.json・progress.json・result.json・log.txt（裏の Blender の出力）・cancel。
同時に動かすのは1つ（CPU とメモリを使い切らないため）。画面の側はタイマーで poll() を呼ぶだけで、待たない。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO, Any

from .. import provenance, thresholds
from .progress import (CANCEL_FILE, LOG_FILE, Progress, cancel_requested, read_progress, request_cancel,
                       write_progress)
from .spec import JOB_FILE, JobSpec, load_job, save_job


@dataclass
class WorkerProcess:
    """動いている裏の Blender。log は、出力を書いているファイル（終わったら閉じる）"""

    process: subprocess.Popen[bytes]
    log: IO[bytes] | None = None

    def close_log(self) -> None:
        if self.log is not None:
            try:
                self.log.close()
            finally:
                self.log = None


Runner = Callable[[JobSpec, Path], WorkerProcess]


def build_worker_command(blender: str | os.PathLike[str], worker_script: str | os.PathLike[str],
                         job_dir: str | os.PathLike[str], *, autoexec: bool) -> list[str]:
    """裏の Blender を始めるコマンド（shell を通さず、このリストのまま動かす）。

    --python-exit-code は --python より前に置く（Blender は引数を前から順に処理する）。
    autoexec が False なら --disable-autoexec（開く .blend の中の Python を動かさない）。
    """
    flag = "--enable-autoexec" if autoexec else "--disable-autoexec"
    return [os.fspath(blender), "--background", "--factory-startup", flag, "--python-exit-code", "1",
            "--python", os.fspath(worker_script), "--", os.fspath(job_dir)]


def blender_runner(blender: str | os.PathLike[str], worker_script: str | os.PathLike[str]) -> Runner:
    """Blender で裏の係を始める runner。出力は仕事のフォルダの log.txt に書く"""
    script = Path(worker_script)
    if not script.is_file():
        raise ValueError(f"裏の係の入口のファイルがありません：{script}")

    def run(job: JobSpec, job_dir: Path) -> WorkerProcess:
        command = build_worker_command(blender, script, job_dir, autoexec=job.autoexec)
        log = open(job_dir / LOG_FILE, "ab")
        try:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       cwd=job_dir)
        except Exception:
            log.close()
            raise
        return WorkerProcess(process=process, log=log)

    return run


@dataclass
class _Running:
    job: JobSpec
    job_dir: Path
    worker: WorkerProcess
    started_at: float
    cancel_at: float | None = None
    terminated_at: float | None = None
    killed: bool = False
    timed_out: bool = False


class JobQueue:
    """順番待ち。root の下の、job.json があるフォルダを仕事として扱う"""

    def __init__(self, root: str | os.PathLike[str], runner: Runner, *, clock: Callable[[], float] = time.monotonic,
                 timeout: float = thresholds.JOB_TIMEOUT_SECONDS,
                 grace: float = thresholds.JOB_CANCEL_GRACE_SECONDS) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._runner = runner
        self._clock = clock
        self.timeout = float(timeout)
        self.grace = float(grace)
        self._running: _Running | None = None
        # 読めなかった仕事のフォルダ（名前 → 理由）。壊れたフォルダがあっても、ほかの仕事は止めない
        self.unreadable: dict[str, str] = {}

    # MARK: - 仕事の一覧

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def enqueue(self, job: JobSpec) -> Path:
        """仕事を順番待ちに入れる（フォルダを作り、job.json と「順番待ち」の progress.json を書く）"""
        folder = self.job_dir(job.job_id)
        try:
            folder.mkdir(parents=False, exist_ok=False)
        except FileExistsError as error:
            raise ValueError(f"同じ id の仕事がもうあります：{job.job_id}") from error
        save_job(job, folder)
        write_progress(folder, Progress(status="queued", updated_at=provenance.now_iso()))
        return folder

    def jobs(self) -> list[tuple[JobSpec, Progress]]:
        """すべての仕事（作った時刻の順）。読めないフォルダは unreadable に入れて飛ばす"""
        found: list[tuple[JobSpec, Progress]] = []
        self.unreadable = {}
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir() or not (folder / JOB_FILE).is_file():
                continue
            try:
                job = load_job(folder)
                progress = read_progress(folder) or Progress(status="queued", updated_at=job.created_at)
            except ValueError as error:
                self.unreadable[folder.name] = str(error)
                continue
            found.append((job, progress))
        found.sort(key=lambda item: (item[0].created_at, item[0].job_id))
        return found

    def progress_of(self, job_id: str) -> Progress | None:
        return read_progress(self.job_dir(job_id))

    @property
    def running_job(self) -> JobSpec | None:
        return None if self._running is None else self._running.job

    # MARK: - 始める・見る・止める

    def recover(self) -> list[str]:
        """画面を開き直したあとに呼ぶ。前の回に「動いています」のまま残った仕事を、失敗（結果なし）にする。
        戻り値は、そうした仕事の id"""
        orphaned = []
        for job, progress in self.jobs():
            if progress.status == "running" and (self._running is None or self._running.job.job_id != job.job_id):
                write_progress(self.job_dir(job.job_id), replace(
                    progress, status="failed", updated_at=provenance.now_iso(), finished_at=provenance.now_iso(),
                    error_key="jobs.error_orphaned", error_values={}, pid=None))
                orphaned.append(job.job_id)
        return orphaned

    def start_next(self) -> JobSpec | None:
        """何も動いていなければ、いちばん古い順番待ちの仕事を始める。始めた仕事を返す（無ければ None）"""
        if self._running is not None:
            return None
        for job, progress in self.jobs():
            if progress.status != "queued":
                continue
            folder = self.job_dir(job.job_id)
            try:
                worker = self._runner(job, folder)
            except (OSError, ValueError) as error:
                now = provenance.now_iso()
                write_progress(folder, Progress(status="failed", updated_at=now, finished_at=now,
                                                error_key="jobs.error_start_failed", error_values={"reason": str(error)}))
                continue
            self._running = _Running(job=job, job_dir=folder, worker=worker, started_at=self._clock())
            return job
        return None

    def poll(self) -> Progress | None:
        """動いている仕事の進み具合を読み、終わっていれば片付ける。何も動いていなければ None。

        裏の係が「終わり」を書いても、プロセスが消えるまでは running_job に残る（片付いたかは running_job で見る）。

        - 裏の Blender が「終わり」を書かずに消えたら failed（終了コード付き）にする。中止を頼んでいたなら cancelled。
        - 時間の上限を過ぎたら中止を頼み、猶予のあとに terminate、さらに猶予のあとに kill する。
        """
        running = self._running
        if running is None:
            return None
        folder = running.job_dir
        try:
            progress = read_progress(folder)
        except ValueError:
            progress = None
        if progress is None:
            progress = Progress(status="running", updated_at=provenance.now_iso())
        code = running.worker.process.poll()
        now = self._clock()
        if code is None:
            if running.cancel_at is None and now - running.started_at >= self.timeout:
                running.timed_out = True
                self.cancel(running.job.job_id)
            if running.cancel_at is not None:
                if running.terminated_at is None and now - running.cancel_at >= self.grace:
                    running.worker.process.terminate()
                    running.terminated_at = now
                elif running.terminated_at is not None and not running.killed and now - running.terminated_at >= self.grace:
                    running.worker.process.kill()
                    running.killed = True
            return progress
        running.worker.close_log()
        if not progress.is_finished:
            stamp = provenance.now_iso()
            if running.timed_out:
                progress = replace(progress, status="failed", updated_at=stamp, finished_at=stamp, pid=None,
                                   error_key="jobs.error_timeout", error_values={"seconds": int(self.timeout)})
            elif running.cancel_at is not None:
                progress = replace(progress, status="cancelled", updated_at=stamp, finished_at=stamp, pid=None,
                                   error_key="jobs.error_cancelled", error_values={})
            else:
                progress = replace(progress, status="failed", updated_at=stamp, finished_at=stamp, pid=None,
                                   error_key="jobs.error_process_died", error_values={"code": code})
            write_progress(folder, progress)
        elif running.timed_out and progress.status == "cancelled":
            stamp = provenance.now_iso()
            progress = replace(progress, status="failed", updated_at=stamp, error_key="jobs.error_timeout",
                               error_values={"seconds": int(self.timeout)})
            write_progress(folder, progress)
        self._running = None
        return progress

    def cancel(self, job_id: str) -> Progress:
        """中止する。順番待ちならすぐ cancelled にする。動いていれば中止を頼む（裏の係が自分で止まるのを待つ）"""
        folder = self.job_dir(job_id)
        if not (folder / JOB_FILE).is_file():
            raise ValueError(f"その仕事はありません：{job_id}")
        progress = read_progress(folder)
        running = self._running is not None and self._running.job.job_id == job_id
        if running:
            # 動いている（進み具合がまだ「順番待ち」のままでも）：中止を頼み、裏の係が自分で止まるのを待つ
            request_cancel(folder)
            if self._running.cancel_at is None:
                self._running.cancel_at = self._clock()
            return progress if progress is not None else Progress(status="running", updated_at=provenance.now_iso())
        if progress is None or progress.status == "queued":
            now = provenance.now_iso()
            finished = Progress(status="cancelled", updated_at=now, finished_at=now, error_key="jobs.error_cancelled")
            write_progress(folder, finished)
            return finished
        if progress.is_finished:
            return progress
        # 「動いています」のまま残っているが、この画面が始めた仕事ではない（前の回の物）：頼みだけ置く
        request_cancel(folder)
        return progress

    def remove(self, job_id: str) -> None:
        """終わった仕事のフォルダを消す（動いている・順番待ちの仕事は消さない）"""
        folder = self.job_dir(job_id)
        if not (folder / JOB_FILE).is_file():
            raise ValueError(f"その仕事はありません：{job_id}")
        if self._running is not None and self._running.job.job_id == job_id:
            raise ValueError(f"動いている仕事は消せません：{job_id}（先に中止してください）")
        progress = read_progress(folder)
        if progress is not None and not progress.is_finished:
            raise ValueError(f"終わっていない仕事は消せません：{job_id}（先に中止してください）")
        shutil.rmtree(folder)

    def is_cancel_requested(self, job_id: str) -> bool:
        return cancel_requested(self.job_dir(job_id))

    def log_text(self, job_id: str, *, max_bytes: int = 65536) -> str:
        """裏の Blender の出力（終わりの max_bytes だけ）"""
        path = self.job_dir(job_id) / LOG_FILE
        if not path.is_file():
            return ""
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")


__all__ = ["JobQueue", "Runner", "WorkerProcess", "blender_runner", "build_worker_command", "CANCEL_FILE"]

"""画面の側の bpy の部分：Blender の場所、設定、仕事を作る、タイマーで進み具合を読む。

画面のパネル（ui/）は、ここの make_queue()・make_job()・start_polling() を使う。重い道具のボタンは、
その道具を直接呼ばず、仕事を順番待ちに入れる。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import bpy

from .. import provenance, thresholds
from .queue import JobQueue, blender_runner
from .spec import JobSpec, new_job_id


def worker_script() -> Path:
    """裏の Blender の入口のファイル（このパッケージの中）"""
    return Path(__file__).with_name("worker_main_bpy.py")


def blender_binary(preferred: str | os.PathLike[str] | None = None) -> str:
    """裏の Blender の場所。設定に書いてあればそれ、無ければいま動いている Blender と同じ物"""
    text = "" if preferred is None else os.fspath(preferred).strip()
    path = os.path.abspath(bpy.path.abspath(text)) if text else bpy.app.binary_path
    if not os.path.isfile(path) or not os.access(path, os.X_OK):
        raise ValueError(f"Blender が見つからないか、動かせるファイルではありません：{path}")
    return path


def autoexec_allowed() -> bool:
    """画面の Blender が、.blend の中の Python（ドライバーの式など）を動かす設定か"""
    return bool(bpy.context.preferences.filepaths.use_scripts_auto_execute)


def default_jobs_root(work_dir: str | os.PathLike[str] | None = None) -> Path:
    """仕事のフォルダの置き場所。設定の作業フォルダがあればその下の jobs/。無ければ .blend の隣の <名前>_ai_tools/jobs/"""
    text = "" if work_dir is None else os.fspath(work_dir).strip()
    if text:
        return Path(os.path.abspath(bpy.path.abspath(text))) / "jobs"
    if not bpy.data.filepath:
        raise ValueError("ファイルを保存してから使ってください（仕事のフォルダを .blend の隣に作るため）")
    blend = Path(bpy.data.filepath)
    return blend.with_name(f"{blend.stem}_ai_tools") / "jobs"


def make_queue(root: str | os.PathLike[str], *, blender: str | os.PathLike[str] | None = None) -> JobQueue:
    return JobQueue(root, blender_runner(blender_binary(blender), worker_script()))


def make_job(tool_id: str, params: dict[str, Any], *, input_blend: str | os.PathLike[str] | None,
             confirmed: bool = False, plan_approved: bool = False, job_id: str | None = None) -> JobSpec:
    """仕事を作る。autoexec は画面の Blender の設定に合わせる（結果が変わりうるので、仕事の形に残る）"""
    return JobSpec(job_id=job_id or new_job_id(), tool_id=tool_id, params=params,
                   input_blend=None if input_blend is None else os.fspath(input_blend),
                   created_at=provenance.now_iso(), confirmed=confirmed, plan_approved=plan_approved,
                   autoexec=autoexec_allowed())


def start_polling(queue: JobQueue, *, interval: float = thresholds.JOB_POLL_INTERVAL_SECONDS,
                  on_update: Callable[[], None] | None = None) -> Callable[[], None]:
    """タイマーで、進み具合を読み、次の仕事を始める。止める関数を返す（register の解除のときに呼ぶ）。
    画面を描き直すのは on_update（パネルの側が渡す）"""

    def tick() -> float | None:
        if stopped[0]:
            return None
        queue.poll()
        queue.start_next()
        if on_update is not None:
            on_update()
        return interval

    stopped = [False]
    bpy.app.timers.register(tick, first_interval=interval, persistent=True)

    def stop() -> None:
        stopped[0] = True
        if bpy.app.timers.is_registered(tick):
            bpy.app.timers.unregister(tick)

    return stop

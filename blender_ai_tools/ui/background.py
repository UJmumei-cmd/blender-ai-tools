"""画面の側の、裏の係（A10）の順番待ち。いまのファイルの仕事のフォルダ（作業フォルダの jobs/）に、JobQueue を1つだけ持つ。

- JobQueue を作るとフォルダができるので、作るのは、仕事のフォルダがもうあるとき（register・別のファイルを開いたとき・
  保存したときで、まだ順番待ちが無いとき）か、ユーザーが仕事を出したとき（ensure_queue）だけ。
- 作ったら、前の回の残りを片付けて（queue.recover()）から、タイマーで進み具合を読む（launcher_bpy.start_polling）。
- 別のファイルを開いたら、前のファイルのタイマーを止め、新しいファイルの仕事のフォルダにつなぎ直す。
- unregister でタイマーを止める（裏の Blender は止めない。終わりと結果は、自分で書く）。
- 仕事の一覧（仕事, 進み具合）は、タイマーの1回ごとと操作のあとに読み直して覚える。パネルは覚えた物を読むだけ。
- 動いていた仕事が「終わりました」になり、その道具が報告を作る物なら、結果（result.json）から報告を作って見せる。

CLAUDE.md 7章：1つの仕事のフォルダに JobQueue は1つだけ。
"""

import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bpy

from ..jobs import launcher_bpy, worker
from ..jobs.progress import Progress
from ..jobs.queue import JobQueue
from ..jobs.spec import JobSpec
from ..report import sources
from ..report.model import Report
from ..versions import naming
from . import preferences, state

# 作業フォルダの中の、仕事のフォルダの名前
JOBS_DIR = "jobs"


@dataclass(frozen=True)
class JobRow:
    """パネルに出す仕事1つ。

    cleaning: 裏の係は終わり（done・failed・cancelled）を書いたが、プロセスがまだ消えていない（片付け中）
    """

    job: JobSpec
    progress: Progress
    cleaning: bool = False

    @property
    def running(self) -> bool:
        return not self.progress.is_finished or self.cleaning


_queue: JobQueue | None = None
_stop: Callable[[], None] | None = None
_rows: tuple[JobRow, ...] = ()
_unreadable: dict[str, str] = {}
# この回の画面で、まだ終わっていなかったのを見た仕事（終わったら、結果から報告を作る）
_watched: set[str] = set()


# MARK: - 場所と順番待ち


def jobs_root() -> Path | None:
    """いまのファイルの仕事のフォルダ。作業フォルダが決まらない（ファイルを保存していない）ときは None"""
    folder = preferences.work_dir_path(preferences.get_preferences())
    return None if folder is None else folder / JOBS_DIR


def queue() -> JobQueue | None:
    """いまの順番待ち（まだ作っていなければ None）"""
    return _queue


def _make(root: Path) -> JobQueue:
    global _queue, _stop
    made = launcher_bpy.make_queue(root, blender=preferences.get_preferences().blender_path.strip() or None)
    made.recover()
    _queue = made
    _stop = launcher_bpy.start_polling(made, on_update=after_tick)
    refresh()
    return made


def detach() -> None:
    """タイマーを止めて、順番待ちを手放す（裏の Blender は止めない）"""
    global _queue, _stop, _rows, _unreadable
    if _stop is not None:
        try:
            _stop()
        except Exception:  # noqa: BLE001  止めるときのまちがいで、解除を止めない
            traceback.print_exc()
    _queue = None
    _stop = None
    _rows = ()
    _unreadable = {}
    _watched.clear()


def attach() -> JobQueue | None:
    """いまのファイルの仕事のフォルダがもうあれば、そこにつなぐ（無ければ作らない）。つないだ順番待ちを返す"""
    root = jobs_root()
    if _queue is not None and root is not None and _queue.root == root:
        return _queue
    if _queue is not None:
        detach()
    if root is None or not root.is_dir():
        return None
    try:
        return _make(root)
    except (ValueError, OSError):
        # 設定の Blender が見つからないなど。仕事を出すとき（ensure_queue）に、理由を知らせる
        traceback.print_exc()
        return None


def ensure_queue() -> JobQueue:
    """仕事を出すための順番待ち（いまのファイルの仕事のフォルダ。無ければ作る）。

    ファイルを保存していないとき・Blender が見つからないときは ValueError。
    作業フォルダの設定が変わっていたら、新しい場所につなぎ直す。
    """
    root = jobs_root()
    if root is None:
        raise ValueError("ファイルを保存してから使ってください（仕事のフォルダを作業フォルダに作るため）")
    if _queue is not None and _queue.root == root:
        return _queue
    detach()
    return _make(root)


# MARK: - 仕事の一覧


def refresh() -> None:
    """仕事の一覧を読み直して覚える。動いていた仕事が終わっていたら、結果から報告を作る"""
    global _rows, _unreadable
    if _queue is None:
        _rows, _unreadable = (), {}
        return
    running = _queue.running_job
    running_id = None if running is None else running.job_id
    rows = []
    finished_now = []
    for job, progress in _queue.jobs():
        rows.append(JobRow(job=job, progress=progress, cleaning=progress.is_finished and job.job_id == running_id))
        if not progress.is_finished:
            _watched.add(job.job_id)
        elif job.job_id in _watched:
            _watched.discard(job.job_id)
            if progress.status == "done":
                finished_now.append(job)
    _rows = tuple(rows)
    _unreadable = dict(_queue.unreadable)
    for job in finished_now:
        if sources.makes_report(job.tool_id):
            try:
                state.show_report(load_report(job.job_id))
            except (ValueError, OSError):
                traceback.print_exc()


def rows() -> tuple[JobRow, ...]:
    """覚えている仕事の一覧（作った順）。読み直さない"""
    return _rows


def unreadable() -> dict[str, str]:
    """読めなかった仕事のフォルダ（名前 → 理由）"""
    return dict(_unreadable)


def load_report(job_id: str) -> Report:
    """終わった仕事の結果（result.json）から報告を作る。無い・読めない・報告を作らない道具なら ValueError"""
    if _queue is None:
        raise ValueError("仕事の順番待ちがありません")
    data = worker.read_result(_queue.job_dir(job_id))
    if data is None:
        raise ValueError(f"仕事 {job_id} の結果がまだありません")
    source: dict[str, Any] = {"job_id": job_id, "tool_version": data.get("tool_version")}
    input_data = data.get("input")
    if isinstance(input_data, dict) and isinstance(input_data.get("path"), str):
        input_path = Path(input_data["path"])
        source["input_sha256"] = input_data.get("sha256")
        if naming.parse_version(input_path) is not None:
            source["version"] = input_path.name
    return sources.report_from_tool_result(data.get("tool_id"), data.get("result"), source=source)


# MARK: - タイマーの1回分


def _redraw() -> None:
    """3D ビューを描き直す（画面の無いバックグラウンドの Blender では、窓が無いので何もしない）"""
    manager = getattr(bpy.context, "window_manager", None)
    for window in getattr(manager, "windows", ()) or ():
        screen = getattr(window, "screen", None)
        for area in getattr(screen, "areas", ()) or ():
            if area.type == "VIEW_3D":
                area.tag_redraw()


def after_tick() -> None:
    """タイマーの1回ごと（進み具合を読み、次の仕事を始めたあと）：一覧を読み直して、描き直す"""
    refresh()
    _redraw()


def step() -> None:
    """タイマーの1回分を、いま動かす（テストと、操作のすぐあとに使う）：進み具合を読み、次の仕事を始め、一覧を読み直す"""
    if _queue is None:
        return
    _queue.poll()
    _queue.start_next()
    after_tick()


# MARK: - 登録とファイル


def _data_ready() -> bool:
    """bpy.data を読めるか。Blender の起動の途中で拡張機能が登録されるときは、まだ読めない（_RestrictData）。
    そのときは、起動のファイルを読んだあとの on_load_post でつなぐ"""
    return hasattr(bpy.data, "filepath")


def register() -> None:
    if not _data_ready():
        return
    attach()
    state.refresh_versions()


def unregister() -> None:
    detach()


@bpy.app.handlers.persistent
def on_load_post(*_args: Any) -> None:
    """別のファイルを開いたら：前のファイルのタイマーを止め、新しいファイルの仕事のフォルダにつなぐ"""
    detach()
    state.clear()
    attach()
    state.refresh_versions()


@bpy.app.handlers.persistent
def on_save_post(*_args: Any) -> None:
    """保存したら：まだ順番待ちが無ければ、仕事のフォルダにつなぐ（あれば）。版の一覧を読み直す"""
    if _queue is None:
        attach()
    state.refresh_versions()

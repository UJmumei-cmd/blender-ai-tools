"""進み具合（progress.json）、中止の頼み（cancel ファイル）、道具の中から呼ぶ report()。

裏の Blender が progress.json を書き、画面の Blender が読む。書くときは一時ファイルから置き換えるので、
読む側が書きかけの文を読むことはない。中止は、画面の側が仕事のフォルダに cancel という名前のファイルを置き、
裏の側が report() のたびにそれを見て、JobCancelled を上げて止まる（道具が長い計算の途中で report() を呼ぶほど、早く止まる）。
"""

from __future__ import annotations

import contextlib
import contextvars
import math
import operator
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import provenance
from .spec import SCHEMA_VERSION, check_json_value, is_time, strict_json_dumps, strict_json_loads, write_text_atomic

# 仕事のフォルダの中のファイルの名前
PROGRESS_FILE = "progress.json"
RESULT_FILE = "result.json"
LOG_FILE = "log.txt"
CANCEL_FILE = "cancel"

STATUSES = ("queued", "running", "done", "failed", "cancelled")
FINISHED_STATUSES = ("done", "failed", "cancelled")


class JobCancelled(BaseException):
    """中止の頼みを受けた（道具の中から上がる。裏の係が受け止めて「中止」と書く）。

    Exception ではなく BaseException の子にする（KeyboardInterrupt と同じ）。道具の中の `except Exception:` で
    握りつぶされず、裏の係まで届くようにするため"""


# MARK: - 進み具合の形


@dataclass(frozen=True)
class Progress:
    """1つの仕事の進み具合。

    status: queued（順番待ち）・running・done・failed・cancelled
    step・total: いまの段と、全部の段の数（分からないときは 0）
    message_key・message_values: いまの段の説明（texts のキーと、入れる値）。空なら無し
    fraction: 進んだ割合（0〜1）。分からないときは None
    started_at・updated_at・finished_at: 時刻（provenance.now_iso() の形）
    error・error_key・error_values: 失敗や中止の理由。error は自由な文（例外の文）、error_key は決まった理由（texts のキー）
    pid: 裏の Blender のプロセス番号（動いているときだけ）
    """

    status: str
    updated_at: str
    step: int = 0
    total: int = 0
    message_key: str = ""
    message_values: dict[str, Any] = field(default_factory=dict)
    fraction: float | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    error_key: str | None = None
    error_values: dict[str, Any] = field(default_factory=dict)
    pid: int | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        problems: list[str] = []
        if self.status not in STATUSES:
            problems.append(f"status は {STATUSES} のどれかにしてください（いまは {self.status!r}）")
        for name in ("step", "total"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                problems.append(f"{name} は 0 以上の整数にしてください（いまは {value!r}）")
        if self.fraction is not None and (isinstance(self.fraction, bool) or not isinstance(self.fraction, (int, float))
                                          or not math.isfinite(self.fraction) or not 0.0 <= self.fraction <= 1.0):
            problems.append(f"fraction は 0〜1 の数か None にしてください（いまは {self.fraction!r}）")
        if not is_time(self.updated_at):
            problems.append(f"updated_at は時刻の形にしてください（いまは {self.updated_at!r}）")
        for name in ("started_at", "finished_at"):
            value = getattr(self, name)
            if value is not None and not is_time(value):
                problems.append(f"{name} は時刻の形か None にしてください（いまは {value!r}）")
        for name in ("message_key", ):
            if not isinstance(getattr(self, name), str):
                problems.append(f"{name} は文字にしてください")
        for name in ("error", "error_key"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                problems.append(f"{name} は文字か None にしてください")
        for name in ("message_values", "error_values"):
            value = getattr(self, name)
            if isinstance(value, dict):
                check_json_value(value, name, problems)
            else:
                problems.append(f"{name} は辞書にしてください")
        if self.pid is not None and (type(self.pid) is not int or self.pid < 0):
            problems.append(f"pid は 0 以上の整数か None にしてください（いまは {self.pid!r}）")
        if self.status in FINISHED_STATUSES and self.finished_at is None:
            problems.append(f"{self.status} のときは finished_at を書いてください")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            problems.append(f"schema_version は {SCHEMA_VERSION} だけを扱えます")
        if problems:
            raise ValueError("進み具合の形が違います：" + "；".join(problems))
        if self.fraction is not None:
            object.__setattr__(self, "fraction", float(self.fraction))

    @property
    def is_finished(self) -> bool:
        return self.status in FINISHED_STATUSES

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "status": self.status, "step": self.step, "total": self.total,
            "message_key": self.message_key, "message_values": self.message_values, "fraction": self.fraction,
            "started_at": self.started_at, "updated_at": self.updated_at, "finished_at": self.finished_at,
            "error": self.error, "error_key": self.error_key, "error_values": self.error_values, "pid": self.pid,
        }

    def to_json(self) -> str:
        return strict_json_dumps(self.to_data())

    @classmethod
    def from_data(cls, data: Any) -> Progress:
        if not isinstance(data, dict):
            raise ValueError(f"進み具合の形は辞書にしてください（いまは {type(data).__name__}）")
        names = {"schema_version", "status", "step", "total", "message_key", "message_values", "fraction",
                 "started_at", "updated_at", "finished_at", "error", "error_key", "error_values", "pid"}
        unknown = set(data) - names
        if unknown:
            raise ValueError(f"進み具合の形に知らない項目があります: {sorted(unknown)}")
        if "status" not in data or "updated_at" not in data:
            raise ValueError("進み具合の形に status か updated_at がありません")
        return cls(**{key: value for key, value in data.items()})

    @classmethod
    def from_json(cls, text: str) -> Progress:
        return cls.from_data(strict_json_loads(text, "進み具合の JSON"))


def write_progress(job_dir: str | os.PathLike[str], progress: Progress) -> Path:
    path = Path(job_dir) / PROGRESS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, progress.to_json())
    return path


def read_progress(job_dir: str | os.PathLike[str]) -> Progress | None:
    """progress.json を読む。まだ無ければ None。形が違えば ValueError"""
    path = Path(job_dir) / PROGRESS_FILE
    if not path.is_file():
        return None
    return Progress.from_json(path.read_text(encoding="utf-8"))


def request_cancel(job_dir: str | os.PathLike[str]) -> None:
    """中止を頼む（cancel ファイルを置く。もうあれば何もしない）"""
    path = Path(job_dir) / CANCEL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "x", encoding="utf-8") as handle:
            handle.write(provenance.now_iso() + "\n")
    except FileExistsError:
        pass


def cancel_requested(job_dir: str | os.PathLike[str]) -> bool:
    return (Path(job_dir) / CANCEL_FILE).exists()


# MARK: - 裏の係の中で使う報告


class Reporter:
    """裏の係の中で、進み具合を書き、中止の頼みを見る。1つの仕事に1つ"""

    def __init__(self, job_dir: str | os.PathLike[str], *, pid: int | None = None) -> None:
        self.job_dir = Path(job_dir)
        self.pid = pid
        self.started_at: str | None = None
        self.last: Progress | None = None

    def start(self) -> Progress:
        """「動いています」と書く（始めの時刻を覚える）"""
        self.started_at = provenance.now_iso()
        return self._write(Progress(status="running", updated_at=self.started_at, started_at=self.started_at, pid=self.pid))

    def check_cancel(self) -> None:
        """中止が頼まれていれば JobCancelled を上げる"""
        if cancel_requested(self.job_dir):
            raise JobCancelled("中止の頼みを受けました")

    def report(self, step: int, total: int, message_key: str = "", values: dict[str, Any] | None = None) -> Progress:
        """いまの段を書く（step は 0 から total まで。total が 0 なら割合は分からない）。先に中止の頼みを見る。
        values は message_key の文に入れる値（辞書）。step・total と values には numpy の数を渡してもよい
        （Python の数にして書く。bool と小数は ValueError）"""
        self.check_cancel()
        step, total = _as_count("step", step), _as_count("total", total)
        if not isinstance(values, dict) and values is not None:
            raise ValueError(f"values は辞書か None にしてください（いまは {type(values).__name__}）")
        fraction = None if total == 0 else min(1.0, step / total)
        now = provenance.now_iso()
        return self._write(Progress(status="running", updated_at=now, step=step, total=total, message_key=message_key,
                                    message_values=provenance.to_plain(dict(values or {}), "values"), fraction=fraction,
                                    started_at=self.started_at or now, pid=self.pid))

    def finish(self, status: str, *, error: str | None = None, error_key: str | None = None,
               **error_values: Any) -> Progress:
        """done・failed・cancelled のどれかで終える。done のときは割合を 1 にする"""
        if status not in FINISHED_STATUSES:
            raise ValueError(f"終わりの status は {FINISHED_STATUSES} のどれかにしてください（いまは {status!r}）")
        now = provenance.now_iso()
        last = self.last
        return self._write(Progress(
            status=status, updated_at=now, step=last.step if last else 0, total=last.total if last else 0,
            message_key=last.message_key if last else "", message_values=dict(last.message_values) if last else {},
            fraction=1.0 if status == "done" else (last.fraction if last else None),
            started_at=self.started_at, finished_at=now, error=error, error_key=error_key,
            error_values=dict(error_values), pid=None))

    def _write(self, progress: Progress) -> Progress:
        write_progress(self.job_dir, progress)
        self.last = progress
        return progress


def _as_count(name: str, value: Any) -> int:
    """0 以上の整数にする（numpy の整数も受け取る。bool・小数・文字は ValueError）"""
    if isinstance(value, bool):
        raise ValueError(f"{name} は 0 以上の整数にしてください（いまは {value!r}）")
    try:
        count = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} は 0 以上の整数にしてください（いまは {value!r}）") from error
    if count < 0:
        raise ValueError(f"{name} は 0 以上の整数にしてください（いまは {value!r}）")
    return count


_current: contextvars.ContextVar[Reporter | None] = contextvars.ContextVar("blender_ai_tools_job_reporter", default=None)


@contextlib.contextmanager
def current_reporter(reporter: Reporter | None) -> Iterator[None]:
    """この中で動く道具の report() が、この reporter に書く"""
    token = _current.set(reporter)
    try:
        yield
    finally:
        _current.reset(token)


def report(step: int, total: int, message_key: str = "", values: dict[str, Any] | None = None) -> None:
    """道具の中から呼ぶ。裏の係の中なら進み具合を書き（中止が頼まれていれば JobCancelled）、画面の中なら何もしない"""
    reporter = _current.get()
    if reporter is not None:
        reporter.report(step, total, message_key, values)


def check_cancel() -> None:
    """道具の中から呼ぶ。裏の係の中で中止が頼まれていれば JobCancelled。画面の中なら何もしない"""
    reporter = _current.get()
    if reporter is not None:
        reporter.check_cancel()


def is_running_in_worker() -> bool:
    return _current.get() is not None


def current_job_dir() -> Path | None:
    """道具の中から呼ぶ。裏の係の中なら、いまの仕事のフォルダ（道具が報告の HTML などを置いてよい所）。画面の中なら None"""
    reporter = _current.get()
    return None if reporter is None else reporter.job_dir

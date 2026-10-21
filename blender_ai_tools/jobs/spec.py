"""裏の係（A10）に渡す仕事の形（JobSpec）と、そのファイル（job.json）。

仕事＝目録（A8）の道具の id と値、開く .blend（この道具が保存した版の写し）、ユーザーの確認の印。
裏の Blender は、このファイルを読んで catalog.call() を呼ぶ。道具の名前と値以外は渡さない（コードは渡さない）。
"""

from __future__ import annotations

import datetime
import itertools
import json
import math
import os
import re
import secrets
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .. import provenance

SCHEMA_VERSION = 1
# 仕事のフォルダの中の、仕事のファイルの名前
JOB_FILE = "job.json"

# 仕事の id の形（フォルダの名前にもなるので、英数字・_・- だけ。「..」や「/」は入らない）
_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
# 目録の道具の id の形（catalog.ToolSpec と同じ）
_TOOL_ID = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+")
_TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
# 同じプロセスで作った仕事の通し番号（同じ秒に作った仕事が、作った順に並ぶように id に入れる）
_SEQUENCE = itertools.count(1)
# 置き換え（os.replace）を試す回数と、間に待つ秒。Windows では、読む側が開いている間は置き換えられない
_REPLACE_ATTEMPTS = 5
_REPLACE_WAIT_SECONDS = 0.05


# MARK: - JSON を厳しく読む


class _Rejected(Exception):
    pass


def _reject_constant(name: str) -> Any:
    raise _Rejected(f"JSON に {name} は書けません")


def _parse_finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise _Rejected(f"数 {text[:40]} は大きすぎます（無限大になります）")
    return value


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _Rejected(f"同じキーが2回あります: {key}")
        result[key] = value
    return result


def strict_json_loads(text: str, what: str = "JSON") -> Any:
    """JSON を読む。読めない・NaN や Infinity（無限大になる大きな数も）が書いてある・同じキーが2回ある・
    入れ子が深すぎるときは ValueError"""
    if not isinstance(text, str):
        raise ValueError(f"{what} は文字で渡してください（いまは {type(text).__name__}）")
    try:
        return json.loads(text, parse_constant=_reject_constant, parse_float=_parse_finite_float,
                          object_pairs_hook=_object_without_duplicates)
    except _Rejected as error:
        raise ValueError(f"{what} が読めません：{error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{what} が読めません：{error.msg}（{error.lineno} 行 {error.colno} 文字）") from error
    except RecursionError as error:
        raise ValueError(f"{what} が読めません：入れ子が深すぎます") from error


def strict_json_dumps(data: Any) -> str:
    """決まった形の JSON（キーの順をそろえ、字下げ 2、日本語はそのまま、NaN は断る）"""
    return json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def is_time(value: Any) -> bool:
    """provenance.now_iso() の形（2026-10-20T19:19:00Z）か"""
    if not isinstance(value, str) or not _TIME.fullmatch(value):
        return False
    try:
        datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return True


def check_json_value(value: Any, where: str, problems: list[str]) -> None:
    """JSON の値（dict・list・str・int・float・bool・None。NaN と無限大は無し）だけか"""
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            problems.append(f"{where} に NaN か無限大があります")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                problems.append(f"{where} の辞書のキーが文字ではありません（{key!r}）")
            else:
                check_json_value(item, f"{where}.{key}", problems)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            check_json_value(item, f"{where}[{index}]", problems)
        return
    problems.append(f"{where} は JSON の値ではありません（{type(value).__name__}）")


# MARK: - 仕事の形


def is_job_id(value: Any) -> bool:
    """仕事の id の形か（英数字・_・- の 1〜64 文字。フォルダの名前に使えて、root の外に出ない）"""
    return isinstance(value, str) and _JOB_ID.fullmatch(value) is not None


def new_job_id(now: str | None = None) -> str:
    """新しい仕事の id（例 20261020T191900Z_000001_a1b2c3）。時刻、このプロセスでの通し番号、6文字の乱数。

    順番待ちは created_at（秒まで）と job_id で並べるので、同じ秒に作った仕事が作った順に並ぶように、
    乱数の前に通し番号を入れる。測った数字には関わらない"""
    stamp = now or provenance.now_iso()
    if not is_time(stamp):
        raise ValueError(f"時刻の形が違います（いまは {stamp!r}）")
    return f"{re.sub(r'[^0-9TZ]', '', stamp)}_{next(_SEQUENCE):06d}_{secrets.token_hex(3)}"


def write_text_atomic(path: Path, text: str) -> None:
    """同じフォルダの一時ファイルに書いてから置き換える（読む側が、書きかけの文を読まないように）。

    Windows では、読む側がファイルを開いている間は置き換えられない（PermissionError）ので、少し待って何回か試す。
    それでも置き換えられなければ、一時ファイルを消して PermissionError を上げる（元のファイルはそのまま残る）。
    """
    temp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with open(temp, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(temp, path)
                break
            except PermissionError:
                if attempt == _REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(_REPLACE_WAIT_SECONDS)
    finally:
        if temp.exists():
            temp.unlink(missing_ok=True)


@dataclass(frozen=True)
class JobSpec:
    """1つの仕事。

    job_id: 仕事の id（英数字・_・-。フォルダの名前にもなる）
    tool_id: 目録の道具の id（例 "export.convert_and_check"）
    params: 道具に渡す値（JSON の値だけ。裏の係が、目録のスキーマで確かめ直す）
    input_blend: 裏の Blender が開く .blend（この道具が保存した版の写し）の絶対の場所。None なら何も開かない。
                 裏の Blender は別の cwd（仕事のフォルダ）で動くので、相対の場所は受け取らない
    created_at: 作った時刻（provenance.now_iso() の形）
    confirmed・plan_approved: ユーザーの確認の印（catalog.call に渡す。画面の側で、確認を取ったあとに立てる）
    autoexec: 開く .blend の中の Python（ドライバーの式など）を動かしてよいか。画面の Blender の設定に合わせる。
              False が安全な側（動かさない）。結果が変わりうるので、仕事の形に残す
    """

    job_id: str
    tool_id: str
    params: dict[str, Any]
    input_blend: str | None
    created_at: str
    confirmed: bool = False
    plan_approved: bool = False
    autoexec: bool = False
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        problems: list[str] = []
        if not isinstance(self.job_id, str) or not _JOB_ID.fullmatch(self.job_id):
            problems.append(f"job_id は英数字・_・- の 1〜64 文字にしてください（いまは {self.job_id!r}）")
        if not isinstance(self.tool_id, str) or not _TOOL_ID.fullmatch(self.tool_id):
            problems.append(f"tool_id は目録の道具の id の形にしてください（例 export.convert。いまは {self.tool_id!r}）")
        if isinstance(self.params, dict):
            check_json_value(self.params, "params", problems)
            object.__setattr__(self, "params", json.loads(json.dumps(self.params, allow_nan=False)) if not problems else self.params)
        else:
            problems.append(f"params は辞書にしてください（いまは {type(self.params).__name__}）")
        if self.input_blend is not None and (not isinstance(self.input_blend, str) or not self.input_blend):
            problems.append(f"input_blend は .blend の場所の文字か None にしてください（いまは {self.input_blend!r}）")
        elif self.input_blend is not None and not os.path.isabs(self.input_blend):
            problems.append(f"input_blend は絶対の場所にしてください（裏の Blender の cwd は別の所。いまは {self.input_blend!r}）")
        if not is_time(self.created_at):
            problems.append(f"created_at は 2026-10-20T19:19:00Z の形にしてください（いまは {self.created_at!r}）")
        for name in ("confirmed", "plan_approved", "autoexec"):
            if type(getattr(self, name)) is not bool:
                problems.append(f"{name} は True か False にしてください")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            problems.append(f"schema_version は {SCHEMA_VERSION} だけを扱えます（いまは {self.schema_version!r}）")
        if problems:
            raise ValueError("仕事の形が違います：" + "；".join(problems))

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "tool_id": self.tool_id,
            "params": self.params,
            "input_blend": self.input_blend,
            "created_at": self.created_at,
            "confirmed": self.confirmed,
            "plan_approved": self.plan_approved,
            "autoexec": self.autoexec,
        }

    def to_json(self) -> str:
        return strict_json_dumps(self.to_data())

    @classmethod
    def from_data(cls, data: Any) -> JobSpec:
        if not isinstance(data, dict):
            raise ValueError(f"仕事の形は辞書にしてください（いまは {type(data).__name__}）")
        fields = {"schema_version", "job_id", "tool_id", "params", "input_blend", "created_at", "confirmed",
                  "plan_approved", "autoexec"}
        unknown = set(data) - fields
        if unknown:
            raise ValueError(f"仕事の形に知らない項目があります: {sorted(unknown)}")
        missing = {"job_id", "tool_id", "params", "created_at"} - set(data)
        if missing:
            raise ValueError(f"仕事の形に足りない項目があります: {sorted(missing)}")
        return cls(job_id=data["job_id"], tool_id=data["tool_id"], params=data["params"],
                   input_blend=data.get("input_blend"), created_at=data["created_at"],
                   confirmed=data.get("confirmed", False), plan_approved=data.get("plan_approved", False),
                   autoexec=data.get("autoexec", False), schema_version=data.get("schema_version", SCHEMA_VERSION))

    @classmethod
    def from_json(cls, text: str) -> JobSpec:
        return cls.from_data(strict_json_loads(text, "仕事の JSON"))

    def with_input(self, input_blend: str | os.PathLike[str] | None) -> JobSpec:
        return replace(self, input_blend=None if input_blend is None else os.fspath(input_blend))


def save_job(job: JobSpec, job_dir: str | os.PathLike[str]) -> Path:
    """仕事のフォルダに job.json を書く（あとから書き換えない。すでにあれば ValueError）"""
    path = Path(job_dir) / JOB_FILE
    if path.exists():
        raise ValueError(f"仕事のファイルがもうあります（書き換えません）：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, job.to_json())
    return path


def load_job(job_dir: str | os.PathLike[str]) -> JobSpec:
    """仕事のフォルダの job.json を読む。無い・形が違うときは ValueError"""
    path = Path(job_dir) / JOB_FILE
    if not path.is_file():
        raise ValueError(f"仕事のファイルがありません：{path}")
    return JobSpec.from_json(path.read_text(encoding="utf-8"))

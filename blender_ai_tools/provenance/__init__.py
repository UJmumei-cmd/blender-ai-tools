"""A15 道具の版と再現性の記録。

道具を1回動かすたびに、あとで同じ結果を出し直すのに要る物を、すべて記録する（CLAUDE.md 7章「再現性」）。

- 道具の id と版、拡張機能の版、Python・numpy の版（Blender と mmd_tools の版は environment_bpy.py）
- 入力ファイルの sha256 と大きさ
- 設定と乱数の種
- 紅魔館の convert_blend.py のコミットと、コミットしていない変更があるか
- 道具の目録（A8）の指紋（文字列で受け取る）

2回の記録の結果を、再現性の幅の中で比べる（compare_results）。幅より小さい差は、変化として数えない。
幅は thresholds.py の数字を使う：

    長さ（m）     thresholds.REPRO_TOLERANCE_LENGTH
    角度（度）    thresholds.REPRO_TOLERANCE_ANGLE_DEG
    割合         thresholds.REPRO_TOLERANCE_RATIO

このモジュールは bpy を使わない。目録（catalog/）も読み込まない（部品どうしを別々に保つため）。
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import platform
import re
import secrets
import subprocess
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

import numpy as np

from .. import thresholds
from ..thresholds import GIT_TIMEOUT_SECONDS

__all__ = [
    "DIFFERENCE_KINDS",
    "GIT_TIMEOUT_SECONDS",
    "SCHEMA_VERSION",
    "Difference",
    "FileDigest",
    "GitState",
    "RunRecord",
    "addon_version",
    "compare_results",
    "file_digest",
    "git_state",
    "is_reproducible",
    "load_record",
    "make_rng",
    "now_iso",
    "python_environment",
    "save_record",
]

# MARK: - 決まり

# 記録の形の版。項目の意味を変えたら上げる（古い記録を読めるかは、そのときに決める）
SCHEMA_VERSION = 1

# git の答えを待つ時間の上限（秒）は thresholds.GIT_TIMEOUT_SECONDS（仮）。上で読み込む

# ファイルを読むときの1回の大きさ（1 MiB）。大きなファイルでもメモリを使いすぎない
_CHUNK_SIZE = 1024 * 1024
# 同じ名前の記録があるとき、_2・_3 … と試す数の上限
_MAX_SAME_NAME = 10000
# ファイルの名前の部分の長さの上限（文字）
_MAX_NAME_PART = 100
# ファイルの名前に使ってよい文字。ほかの文字は "-" にする
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")
# sha256 の16進（小文字 64 文字）
_SHA256 = re.compile(r"[0-9a-f]{64}")
# git のコミット（sha1 の 40 文字か、sha256 の 64 文字）
_COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
# 拡張機能の manifest（版はここが正）
_MANIFEST = Path(__file__).resolve().parents[1] / "blender_manifest.toml"
# git に別のリポジトリを読ませてしまう環境変数。git_state では消してから git を動かす
_GIT_REDIRECT_VARIABLES = frozenset({
    "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_NAMESPACE", "GIT_PREFIX",
})


def _describe(value: Any) -> str:
    """エラーの文に入れる、値の短い説明"""
    text = repr(value)
    if len(text) > 60:
        text = text[:57] + "..."
    return f"{text}（{type(value).__name__}）"


def _is_number(value: Any) -> bool:
    """数か（int・float・numpy の数。bool は数に入れない）"""
    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def _is_integer(value: Any) -> bool:
    """整数か（bool は入れない）"""
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _join_key(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


# JSON にできる numpy の配列の種類（dtype.kind）：True/False・整数・正の整数・小数・文字・object（中を1つずつ調べる）。
# 複素数・時刻・時間・bytes・構造体は、tolist で別の物（ns の整数・名前の無いリストなど）に変わるので入れない
_JSON_ARRAY_KINDS = "biufUO"


def _check_array_kind(value: np.ndarray, where: str) -> None:
    """JSON にできる種類の配列か。できないときは TypeError"""
    if value.dtype.kind not in _JSON_ARRAY_KINDS:
        raise TypeError(f"{where or '値'}：この種類の numpy の配列（{value.dtype}）は JSON にできません。"
                        "数・文字・True/False の配列にしてください")


def _plain_text(value: str | PurePath) -> str:
    """ただの文字にする。str を受け継いだ型（str と Enum を合わせた物など）は、str() だと "Side.LEFT" のような
    名前になることがあるので、中身の文字を取る（json.dumps が書く物と同じ）"""
    if isinstance(value, str):
        return str.__str__(value)
    return str(value)


# MARK: - 入力ファイルの指紋


@dataclass(frozen=True)
class FileDigest:
    """入力ファイル1つの指紋。

    name: ファイルの名前（フォルダを除いた物）
    path: 測ったときの場所（絶対パス）。別の機械では違ってよい。同じファイルかどうかは sha256 で決める
    sha256: 中身の sha256（16進、小文字 64 文字）
    size: 大きさ（バイト）
    """

    name: str
    path: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        problems: list[str] = []
        for field_name in ("name", "path"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                problems.append(f"{field_name} は空でない文字にしてください（いまは {_describe(value)}）")
        if not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256):
            problems.append(f"sha256 は16進の小文字 64 文字にしてください（いまは {_describe(self.sha256)}）")
        if not _is_integer(self.size) or self.size < 0:
            problems.append(f"size は 0 以上の整数にしてください（いまは {_describe(self.size)}）")
        if problems:
            raise ValueError("入力ファイルの指紋の形が違います：" + "；".join(problems))


def file_digest(path: str | os.PathLike[str]) -> FileDigest:
    """ファイルの sha256 と大きさを測る。1 MiB ずつ読むので、大きなファイルでもメモリを使いすぎない。

    ファイルが無いとき・フォルダのとき・ふつうのファイルでないとき（FIFO など）は ValueError。
    """
    file_path = Path(path)
    if file_path.is_dir():
        raise ValueError(f"フォルダのハッシュは取れません。ファイルを指定してください：{file_path}")
    if not file_path.exists():
        raise ValueError(f"ファイルがありません：{file_path}")
    if not file_path.is_file():
        raise ValueError(f"ふつうのファイルではありません：{file_path}")
    digest = hashlib.sha256()
    size = 0
    try:
        with file_path.open("rb") as stream:
            while chunk := stream.read(_CHUNK_SIZE):
                digest.update(chunk)
                size += len(chunk)
    except (FileNotFoundError, IsADirectoryError) as error:
        # 調べたあとに消された・置きかわったとき
        raise ValueError(f"ファイルが読めません：{file_path}") from error
    return FileDigest(name=file_path.name, path=os.path.abspath(file_path), sha256=digest.hexdigest(), size=size)


# MARK: - git の状態


@dataclass(frozen=True)
class GitState:
    """git のリポジトリの状態（紅魔館の convert_blend.py の版を記録するのに使う）。

    commit: HEAD のコミット（16進）。分からないときは None
    dirty: コミットしていない変更（まだ git に入れていないファイルを含む）があるか。分からないときは None
    """

    commit: str | None
    dirty: bool | None

    def __post_init__(self) -> None:
        problems: list[str] = []
        if self.commit is not None and (not isinstance(self.commit, str) or not _COMMIT.fullmatch(self.commit)):
            problems.append(f"commit は16進の小文字 40 文字か 64 文字、または None にしてください（いまは {_describe(self.commit)}）")
        if self.dirty is not None and type(self.dirty) is not bool:
            problems.append(f"dirty は True・False・None のどれかにしてください（いまは {_describe(self.dirty)}）")
        if problems:
            raise ValueError("git の状態の形が違います：" + "；".join(problems))


def _run_git(directory: str, *args: str) -> subprocess.CompletedProcess[bytes] | None:
    """git を別のプロセスで動かす（shell は使わない）。git が無い・時間切れのときは None"""
    env = {key: value for key, value in os.environ.items() if key not in _GIT_REDIRECT_VARIABLES}
    # 見るだけなので、git に index のロックを取らせない
    env["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        return subprocess.run(["git", "-C", directory, *args], capture_output=True, stdin=subprocess.DEVNULL,
                              timeout=GIT_TIMEOUT_SECONDS, env=env, check=False)
    except (OSError, subprocess.SubprocessError):
        return None


def git_state(repo_dir: str | os.PathLike[str]) -> GitState:
    """リポジトリの中のフォルダ（またはファイル）の、HEAD のコミットと、コミットしていない変更があるかを調べる。

    - ファイルを渡したときは、そのファイルのあるフォルダで調べる（convert_blend.py の場所をそのまま渡してよい）。
    - git が無い・フォルダが無い・リポジトリではないときは GitState(None, None)。
    - まだコミットが1つも無いリポジトリでは、commit が None になる。
    - dirty は、リポジトリ全体で見る（まだ git に入れていないファイルも変更に数える。git の設定
      status.showUntrackedFiles=no があっても数える）。
    - 空の文字は ValueError（"." と同じになり、いまの作業フォルダの、別のリポジトリを調べてしまうため）。
    """
    if not os.fspath(repo_dir):
        raise ValueError("git の状態を調べるフォルダが空の文字です。紅魔館の convert_blend.py の場所を指定してください")
    directory = Path(repo_dir)
    if directory.is_file():
        directory = directory.parent
    if not directory.is_dir():
        return GitState(None, None)
    status = _run_git(str(directory), "status", "--porcelain", "--untracked-files=normal")
    if status is None or status.returncode != 0:
        return GitState(None, None)
    dirty = bool(status.stdout.strip())
    commit: str | None = None
    head = _run_git(str(directory), "rev-parse", "HEAD")
    if head is not None and head.returncode == 0:
        text = head.stdout.decode("ascii", errors="replace").strip()
        if _COMMIT.fullmatch(text):
            commit = text
    return GitState(commit=commit, dirty=dirty)


# MARK: - 版と環境


def addon_version() -> str:
    """拡張機能（AI の道具）の版。blender_manifest.toml の version を読む"""
    with _MANIFEST.open("rb") as stream:
        manifest = tomllib.load(stream)
    version = manifest.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError(f"blender_manifest.toml に version がありません：{_MANIFEST}")
    return version


def python_environment() -> dict[str, str]:
    """Python・numpy の版と、OS・CPU の名前（bpy なしで分かる物）。

    Blender と mmd_tools の版は environment_bpy.blender_environment() で取り、この辞書に足して記録する。
    """
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
    }


# MARK: - 乱数と時刻


def make_rng(seed: int = thresholds.DEFAULT_SEED) -> np.random.Generator:
    """種を決めた乱数の生成器。種は、呼んだ側が記録する（RunRecord.seed）。

    numpy の default_rng の方式が将来変わっても数が変わらないように、PCG64 をはっきり指定する。
    ただし、同じ種でも numpy の版が変わると、正規分布などの数が変わることがある（numpy の決まり）。
    そのため numpy の版も記録する（python_environment）。

    種は 0 以上の整数。bool・小数・負の数は ValueError。
    """
    if not _is_integer(seed):
        raise ValueError(f"乱数の種は 0 以上の整数にしてください（いまは {_describe(seed)}）")
    if seed < 0:
        raise ValueError(f"乱数の種は 0 以上にしてください（いまは {int(seed)}）")
    return np.random.Generator(np.random.PCG64(int(seed)))


def now_iso() -> str:
    """いまの時刻（UTC）。例 "2026-10-10T02:30:00Z"（秒まで）"""
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _check_time(name: str, value: Any, problems: list[str]) -> None:
    """ISO-8601 の時刻で、時差（Z や +09:00）が付いているか"""
    if not isinstance(value, str):
        problems.append(f"{name} は ISO-8601 の時刻の文字にしてください（いまは {_describe(value)}）")
        return
    try:
        moment = datetime.datetime.fromisoformat(value)
    except ValueError:
        problems.append(f"{name} が ISO-8601 の時刻ではありません（いまは {_describe(value)}）")
        return
    if moment.tzinfo is None:
        problems.append(f"{name} に時差（Z など）が付いていません（いまは {value!r}）。now_iso() を使ってください")


# MARK: - 記録


@dataclass(frozen=True)
class RunRecord:
    """道具を1回動かした記録。

    tool_id・tool_version: 道具の id と版（目録 A8 の物）
    started_at・finished_at: 始めと終わりの時刻（now_iso() の形。時差が要る）
    addon_version: 拡張機能の版（addon_version()）
    environment: Python・numpy・Blender・mmd_tools の版など（python_environment() と blender_environment()）
    inputs: 入力ファイルの指紋（file_digest()）。順番も記録する
    settings: 道具に渡した設定
    seed: 乱数の種（乱数を使わない道具は None）
    results: 結果。数・文字・True/False・None・リスト・辞書・numpy の数と配列・Path を入れてよい
    catalog_fingerprint: 道具の目録の指紋（Catalog.fingerprint()）
    convert_blend: 紅魔館の convert_blend.py の git の状態（git_state()）
    notes: メモ
    schema_version: 記録の形の版（SCHEMA_VERSION）

    JSON にすると、NaN と ±inf は null（測れなかった値）になる。
    """

    tool_id: str
    tool_version: str
    started_at: str
    finished_at: str
    addon_version: str
    environment: dict[str, Any]
    inputs: tuple[FileDigest, ...]
    settings: dict[str, Any]
    seed: int | None
    results: dict[str, Any]
    catalog_fingerprint: str | None = None
    convert_blend: GitState | None = None
    notes: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        problems: list[str] = []
        for name in ("tool_id", "tool_version", "addon_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                problems.append(f"{name} は空でない文字にしてください（いまは {_describe(value)}）")
        _check_time("started_at", self.started_at, problems)
        _check_time("finished_at", self.finished_at, problems)
        for name in ("environment", "settings", "results"):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                object.__setattr__(self, name, dict(value))
            else:
                problems.append(f"{name} は辞書にしてください（いまは {_describe(value)}）")
        if isinstance(self.inputs, (list, tuple)) and all(isinstance(item, FileDigest) for item in self.inputs):
            object.__setattr__(self, "inputs", tuple(self.inputs))
        else:
            problems.append(f"inputs は FileDigest のタプルにしてください（いまは {_describe(self.inputs)}）")
        if self.seed is not None:
            if not _is_integer(self.seed) or self.seed < 0:
                problems.append(f"seed は 0 以上の整数か None にしてください（いまは {_describe(self.seed)}）")
            else:
                object.__setattr__(self, "seed", int(self.seed))
        if self.catalog_fingerprint is not None and (
                not isinstance(self.catalog_fingerprint, str) or not self.catalog_fingerprint):
            problems.append(f"catalog_fingerprint は空でない文字か None にしてください（いまは {_describe(self.catalog_fingerprint)}）")
        if self.convert_blend is not None and not isinstance(self.convert_blend, GitState):
            problems.append(f"convert_blend は GitState か None にしてください（いまは {_describe(self.convert_blend)}）")
        if not isinstance(self.notes, str):
            problems.append(f"notes は文字にしてください（いまは {_describe(self.notes)}）")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            problems.append(f"schema_version は {SCHEMA_VERSION} だけを扱えます（いまは {_describe(self.schema_version)}）")
        if problems:
            raise ValueError("記録の形が違います：" + "；".join(problems))

    def to_json(self) -> str:
        """決まった形の JSON（キーの順をそろえ、字下げ 2、日本語はそのまま）。同じ記録からは、いつも同じ文字になる。

        numpy の数は Python の数、配列は入れ子のリスト、タプルはリスト、Path は文字、NaN と ±inf は null にする。
        str・int を受け継いだ型（Signal などの Enum）は、中身の文字・数にする（Signal.RED は 3）。
        ほかの型（numpy 以外の物・set・bytes・複素数・Measurement など）は TypeError（勝手に文字にしない）。
        """
        data = {
            "schema_version": self.schema_version,
            "tool_id": self.tool_id,
            "tool_version": self.tool_version,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "addon_version": self.addon_version,
            "environment": self.environment,
            "inputs": [{"name": item.name, "path": item.path, "sha256": item.sha256, "size": item.size}
                       for item in self.inputs],
            "settings": self.settings,
            "seed": self.seed,
            "results": self.results,
            "catalog_fingerprint": self.catalog_fingerprint,
            "convert_blend": None if self.convert_blend is None else {
                "commit": self.convert_blend.commit, "dirty": self.convert_blend.dirty},
            "notes": self.notes,
        }
        return json.dumps(_to_plain(data, ""), sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> RunRecord:
        """to_json() の文字から記録を作る。to_json(from_json(x)) == x になる。

        JSON が読めない・同じキーが2回ある・NaN が書いてある・知らない項目がある・形が違うときは ValueError。
        """
        if not isinstance(text, str):
            raise ValueError(f"記録の JSON は文字で渡してください（いまは {_describe(text)}）")
        try:
            data = json.loads(text, object_pairs_hook=_object_without_duplicates, parse_constant=_reject_constant)
        except json.JSONDecodeError as error:
            raise ValueError(f"記録の JSON が読めません：{error}") from error
        if not isinstance(data, dict):
            raise ValueError("記録の JSON は、いちばん外が {} の形にしてください")
        missing = sorted(_REQUIRED_KEYS - data.keys())
        unknown = sorted(data.keys() - _REQUIRED_KEYS - _OPTIONAL_KEYS)
        if missing or unknown:
            problems = []
            if missing:
                problems.append("足りない項目：" + "、".join(missing))
            if unknown:
                problems.append("知らない項目：" + "、".join(unknown))
            raise ValueError("記録の項目が違います（" + "；".join(problems) + "）")
        if data["schema_version"] != SCHEMA_VERSION or type(data["schema_version"]) is not int:
            raise ValueError(f"この道具が読める記録の形は schema_version {SCHEMA_VERSION} だけです"
                             f"（いまは {_describe(data['schema_version'])}）")
        return cls(
            tool_id=data["tool_id"],
            tool_version=data["tool_version"],
            started_at=data["started_at"],
            finished_at=data["finished_at"],
            addon_version=data["addon_version"],
            environment=_require_dict("environment", data["environment"]),
            inputs=tuple(_file_digest_from_data(index, item)
                         for index, item in enumerate(_require_list(data["inputs"]))),
            settings=_require_dict("settings", data["settings"]),
            seed=data["seed"],
            results=_require_dict("results", data["results"]),
            catalog_fingerprint=data.get("catalog_fingerprint"),
            convert_blend=_git_state_from_data(data.get("convert_blend")),
            notes=data.get("notes", ""),
            schema_version=data["schema_version"],
        )


_REQUIRED_KEYS = frozenset({
    "schema_version", "tool_id", "tool_version", "started_at", "finished_at", "addon_version", "environment",
    "inputs", "settings", "seed", "results",
})
_OPTIONAL_KEYS = frozenset({"catalog_fingerprint", "convert_blend", "notes"})


# MARK: - JSON にする・読む


def to_plain(value: Any, where: str = "") -> Any:
    """JSON にそのまま書ける物（dict・list・str・int・float・bool・None）にする。RunRecord.to_json と同じ決まり。

    numpy の数は Python の数、配列は入れ子のリスト、タプルはリスト、Path は文字、NaN と ±inf は None。
    できない型は TypeError（where は、まちがいの場所を伝えるための名前）。裏の係（A10）が結果を書くときにも使う。
    """
    return _to_plain(value, where)


def _to_plain(value: Any, where: str) -> Any:
    """JSON にできる物（dict・list・str・int・float・bool・None）にする。できない型は TypeError"""
    if value is None:
        return None
    if isinstance(value, str):
        return _plain_text(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, PurePath):
        return _plain_text(value)
    if isinstance(value, np.ndarray):
        _check_array_kind(value, where)
        return _to_plain(value.tolist(), where)
    if isinstance(value, Mapping):
        plain: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{where or '値'}：辞書のキーは文字にしてください（いまは {_describe(key)}）")
            plain[_plain_text(key)] = _to_plain(item, _join_key(where, _plain_text(key)))
        return plain
    if isinstance(value, (list, tuple)):
        return [_to_plain(item, f"{where}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{where or '値'}：JSON にできない型です（{type(value).__name__}）。"
                    "数・文字・True/False・None・リスト・辞書・numpy の配列にしてから記録してください")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"記録の JSON に同じキー {key!r} が2回あります")
        result[key] = value
    return result


def _reject_constant(name: str) -> Any:
    raise ValueError(f"記録の JSON に {name} は書けません（測れなかった値は null にする）")


def _require_dict(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"記録の {name} は {{}} の形にしてください（いまは {_describe(value)}）")
    return value


def _require_list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"記録の inputs は [] の形にしてください（いまは {_describe(value)}）")
    return value


def _file_digest_from_data(index: int, item: Any) -> FileDigest:
    keys = {"name", "path", "sha256", "size"}
    if not isinstance(item, dict) or item.keys() != keys:
        raise ValueError(f"記録の inputs[{index}] は name・path・sha256・size の4つの項目にしてください")
    return FileDigest(name=item["name"], path=item["path"], sha256=item["sha256"], size=item["size"])


def _git_state_from_data(item: Any) -> GitState | None:
    if item is None:
        return None
    if not isinstance(item, dict) or item.keys() != {"commit", "dirty"}:
        raise ValueError("記録の convert_blend は null か、commit・dirty の2つの項目にしてください")
    return GitState(commit=item["commit"], dirty=item["dirty"])


# MARK: - 保存と読み込み


def _safe_name_part(text: str) -> str:
    """ファイルの名前に使えない文字（: / \\ や空白など）を "-" にする"""
    return _UNSAFE_NAME_CHARS.sub("-", text)[:_MAX_NAME_PART]


def _open_new(path: Path) -> int:
    """まだ無いファイルを作って開く。あるときは FileExistsError（上書きしない。リンクもたどらない）"""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    return os.open(path, flags, 0o666)


def save_record(record: RunRecord, directory: str | os.PathLike[str]) -> Path:
    """記録を JSON のファイルにして保存し、その場所を返す。

    - フォルダが無ければ作る。
    - 名前は「<始めの時刻の : などを - にした物>_<道具の id>.json」。
    - 同じ名前のファイルがあるときは上書きせず、_2・_3 … を付ける。
    - 同じフォルダに一時ファイルを書いてから os.replace で置くので、書きかけのファイルが残らない。
    """
    if not isinstance(record, RunRecord):
        raise ValueError(f"保存できるのは RunRecord だけです（いまは {_describe(record)}）")
    # 先に JSON にする（JSON にできないときは、ファイルを1つも作らない）
    data = (record.to_json() + "\n").encode("utf-8")
    folder = Path(directory)
    if folder.exists() and not folder.is_dir():
        raise ValueError(f"記録を置く場所がフォルダではありません：{folder}")
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{_safe_name_part(record.started_at)}_{_safe_name_part(record.tool_id)}"

    temp_path = folder / f".{stem}.{secrets.token_hex(8)}.tmp"
    descriptor = _open_new(temp_path)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        target = _reserve_name(folder, stem)
        try:
            os.replace(temp_path, target)
        except BaseException:
            # 取っておいた空のファイル（自分で作った物）を消す
            target.unlink(missing_ok=True)
            raise
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return target


def _reserve_name(folder: Path, stem: str) -> Path:
    """まだ使われていない名前を、空のファイルを作って取っておく（ほかのプロセスと同時でも、同じ名前を取らない）"""
    for number in range(1, _MAX_SAME_NAME + 1):
        name = f"{stem}.json" if number == 1 else f"{stem}_{number}.json"
        path = folder / name
        try:
            os.close(_open_new(path))
        except FileExistsError:
            continue
        return path
    raise FileExistsError(f"同じ名前の記録が多すぎます（{_MAX_SAME_NAME} 個）：{folder / stem}.json")


def load_record(path: str | os.PathLike[str]) -> RunRecord:
    """save_record で保存した記録を読む。ファイルが無い・読めない・形が違うときは ValueError"""
    file_path = Path(path)
    if not file_path.is_file():
        raise ValueError(f"記録のファイルがありません：{file_path}")
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"記録のファイルが UTF-8 ではありません：{file_path}") from error
    return RunRecord.from_json(text)


# MARK: - 結果の比べ

# 違いの種類
#   value: 数の差が幅より大きい（片方だけ NaN のときも。そのとき delta は None）
#   missing: 片方にだけあるキー
#   type: 値の種類が違う（数と文字、数と True/False、配列と数 など）
#   shape: 配列の形が違う
#   other: 数でない値（文字・True/False・数でないリスト）が違う。数でないリストは、長さ・中の辞書のキー・
#          中の数でない値が違うとき（中の数は幅を使って比べ、幅を超えたら "value"）
DIFFERENCE_KINDS = ("value", "missing", "type", "shape", "other")


@dataclass(frozen=True)
class Difference:
    """2つの結果の違い1つ。

    key: 入れ子の辞書のキーを . でつないだ物（例 "pose.knee_L"）
    a・b: それぞれの値（片方に無いときは None。kind が "missing" になる）
    delta: 数の差の大きさ（配列は要素ごとの差のいちばん大きい物）。数で表せないときは None
    tolerance: 使った幅（数を比べたときだけ。ほかは None）
    kind: 違いの種類（DIFFERENCE_KINDS のどれか）
    """

    key: str
    a: object
    b: object
    delta: float | None
    tolerance: float | None
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in DIFFERENCE_KINDS:
            raise ValueError(f"違いの種類は {'・'.join(DIFFERENCE_KINDS)} のどれかにしてください（いまは {self.kind!r}）")


# 比べるときの値の種類
_UNMEASURED = "unmeasured"  # None・NaN・±inf（JSON では null）
_BOOL = "bool"
_NUMBER = "number"
_TEXT = "text"
_ARRAY = "array"  # 数（と測れなかった値）だけの、四角い配列
_SEQUENCE = "sequence"  # それ以外のリスト（辞書のリスト、長さのそろわないリストなど）
_EMPTY_MAPPING = "empty_mapping"


def _check_tolerance(name: str, value: Any) -> float:
    if not _is_number(value):
        raise ValueError(f"{name} は 0 以上の数にしてください（いまは {_describe(value)}）")
    number = float(value)
    if math.isnan(number) or number < 0:
        raise ValueError(f"{name} は 0 以上の数にしてください（いまは {number}）")
    return number


def _flatten(mapping: Mapping[Any, Any], prefix: str, out: dict[str, Any], side: str) -> None:
    """入れ子の辞書を、. でつないだキーの平らな辞書にする。空の辞書は、そのまま1つの値として残す"""
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise TypeError(f"{side} の辞書のキーは文字にしてください（いまは {_describe(key)}）")
        dotted = _join_key(prefix, _plain_text(key))
        if isinstance(value, Mapping) and len(value) > 0:
            _flatten(value, dotted, out, side)
            continue
        if dotted in out:
            raise ValueError(f"{side} のキー {dotted!r} が2回出てきます（. を含むキーと、入れ子の辞書が重なっています）")
        out[dotted] = value


def _json_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """配列を JSON（入れ子のリスト）にしたあとの形。(0, 3) は [] になるので (0,) になる"""
    for index, length in enumerate(shape):
        if length == 0:
            return shape[: index + 1]
    return shape


def _numeric_leaves_only(value: Any) -> bool:
    """リストの中が、数と測れなかった値（None）だけか"""
    if isinstance(value, (list, tuple)):
        return all(_numeric_leaves_only(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.dtype.kind in "iuf"
    return value is None or _is_number(value)


def _plain_for_equality(value: Any) -> Any:
    """数でない値を == で比べるための形（JSON にしたときと同じ形）。True と 1 を同じにしないため、bool には印を付ける"""
    if isinstance(value, (bool, np.bool_)):
        return (_BOOL, bool(value))
    if isinstance(value, np.ndarray):
        _check_array_kind(value, "")
        return _plain_for_equality(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_plain_for_equality(item) for item in value]
    if isinstance(value, Mapping):
        plain = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"辞書のキーは文字にしてください（いまは {_describe(key)}）")
            plain[_plain_text(key)] = _plain_for_equality(item)
        return plain
    if value is None:
        return None
    if _is_integer(value):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (str, PurePath)):
        return _plain_text(value)
    raise TypeError(f"比べられない型です（{type(value).__name__}）")


def _kind_of(value: Any) -> tuple[str, Any]:
    """値の種類と、比べるための形"""
    if value is None:
        return _UNMEASURED, None
    if isinstance(value, (bool, np.bool_)):
        return _BOOL, bool(value)
    if _is_integer(value):
        return _NUMBER, int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return (_NUMBER, number) if math.isfinite(number) else (_UNMEASURED, None)
    if isinstance(value, (str, PurePath)):
        return _TEXT, _plain_text(value)
    if isinstance(value, Mapping):
        return _EMPTY_MAPPING, {}
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _kind_of(value[()])
        _check_array_kind(value, "")
        if value.dtype.kind in "iuf":
            return _ARRAY, np.asarray(value, dtype=np.float64).reshape(_json_shape(value.shape))
        # bool・文字・object の配列は、JSON にしたときと同じリストにして比べる
        return _kind_of(value.tolist())
    if isinstance(value, (list, tuple)):
        if _numeric_leaves_only(value):
            try:
                array = np.array(value, dtype=np.float64)
            except (ValueError, OverflowError):
                pass  # 四角くない・大きすぎる整数：数でないリストとして、中を1つずつ比べる
            else:
                # [np.zeros((0, 3))] は JSON で [[]] になるので、形をそろえる
                return _ARRAY, array.reshape(_json_shape(array.shape))
        return _SEQUENCE, _plain_for_equality(value)
    raise TypeError(f"比べられない型です（{type(value).__name__}）。数・文字・True/False・None・リスト・辞書・numpy の配列にしてください")


def _compare_numbers(x: int | float, y: int | float, tolerance: float) -> float | None:
    """2つの数の差。幅より大きいときだけ差を返す（幅の中なら None）"""
    if isinstance(x, int) and isinstance(y, int):
        exact = abs(x - y)  # 大きな整数でも、ずれずに比べる
        if exact <= tolerance:
            return None
        try:
            return float(exact)
        except OverflowError:
            return math.inf
    try:
        delta = abs(float(x) - float(y))
    except OverflowError:
        delta = math.inf  # float にできないほど大きな整数と小数：差はとても大きい
    return delta if delta > tolerance else None


@dataclass
class _NestedComparison:
    """数でないリストの中を比べた結果"""

    other: bool = False  # 数でない所（長さ・辞書のキー・文字・True/False・値の種類）が違う
    unmeasured: bool = False  # 片方だけ測れなかった数がある
    delta: float | None = None  # 幅を超えた差のうち、いちばん大きい物


def _compare_nested(x: Any, y: Any, tolerance: float, result: _NestedComparison) -> None:
    """_plain_for_equality の形の2つの値を、中まで比べる。数は幅を使い、ほかは == で比べる"""
    if result.other:
        return  # 数でない違いが見つかったら、それで決まる
    if isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            result.other = True
            return
        for item_x, item_y in zip(x, y):
            _compare_nested(item_x, item_y, tolerance, result)
        return
    if isinstance(x, dict) and isinstance(y, dict):
        if x.keys() != y.keys():
            result.other = True
            return
        for name in x:
            _compare_nested(x[name], y[name], tolerance, result)
        return
    # True/False は (_BOOL, 値) の形になっているので、ここでは数に入らない
    number_x = _is_number(x)
    number_y = _is_number(y)
    if number_x and number_y:
        delta = _compare_numbers(x, y, tolerance)
        if delta is not None:
            result.delta = delta if result.delta is None else max(result.delta, delta)
        return
    if (number_x and y is None) or (x is None and number_y):
        result.unmeasured = True
        return
    if x != y:
        result.other = True


def _compare_leaf(key: str, a: Any, b: Any, tolerance: float) -> Difference | None:
    kind_a, x = _kind_of(a)
    kind_b, y = _kind_of(b)
    if kind_a == _UNMEASURED and kind_b == _UNMEASURED:
        return None  # どちらも測れなかった
    if {kind_a, kind_b} == {_UNMEASURED, _NUMBER}:
        return Difference(key, a, b, None, tolerance, "value")
    if kind_a != kind_b:
        return Difference(key, a, b, None, None, "type")
    if kind_a == _NUMBER:
        delta = _compare_numbers(x, y, tolerance)
        return None if delta is None else Difference(key, a, b, delta, tolerance, "value")
    if kind_a == _ARRAY:
        if x.shape != y.shape:
            return Difference(key, a, b, None, None, "shape")
        missing_x = ~np.isfinite(x)
        missing_y = ~np.isfinite(y)
        if np.any(missing_x != missing_y):
            return Difference(key, a, b, None, tolerance, "value")
        measured = ~missing_x
        if not np.any(measured):
            return None
        with np.errstate(over="ignore", invalid="ignore"):
            delta = float(np.max(np.abs(x[measured] - y[measured])))
        return Difference(key, a, b, delta, tolerance, "value") if delta > tolerance else None
    if kind_a == _SEQUENCE:
        nested = _NestedComparison()
        _compare_nested(x, y, tolerance, nested)
        if nested.other:
            return Difference(key, a, b, None, None, "other")
        if nested.unmeasured:
            return Difference(key, a, b, None, tolerance, "value")
        if nested.delta is not None:
            return Difference(key, a, b, nested.delta, tolerance, "value")
        return None
    # True/False・文字・空の辞書は == で比べる
    return None if x == y else Difference(key, a, b, None, None, "other")


def compare_results(a: Mapping[str, Any], b: Mapping[str, Any], tolerances: Mapping[str, float] | None = None,
                    default_tolerance: float = 0.0) -> list[Difference]:
    """2回の結果を比べ、再現性の幅を超える違いを、キーの順に並べて返す。

    - 入れ子の辞書は、キーを . でつないで比べる（{"pose": {"knee_L": 1}} は "pose.knee_L"）。
    - 数（int・float・numpy の数。bool は入れない）は、差の大きさが幅より大きいときだけ違いにする（幅と同じなら違いにしない）。
    - numpy の配列と数のリストは、要素ごとに比べる。delta は差のいちばん大きい物。形が違えば "shape"。
    - NaN・±inf・None は「測れなかった値」として同じに扱う（JSON にすると、どれも null になるため）。
      どちらも測れなかったときは同じ。片方だけのときは違い（"value"、delta は None）。
    - True/False・文字は == で比べる。True と 1 は種類が違う（"type"）。
    - 数でないリスト（出来事の辞書のリスト、長さのそろわない数のリストなど）は、中まで比べる。中の数は幅を使って比べ
      （"value"、delta は差のいちばん大きい物）、長さ・辞書のキー・文字・True/False が違えば "other" にする。
    - 幅は tolerances[キー] を使い、無ければ default_tolerance。負の幅は ValueError。

    例（同じ入力を2回測ったとき）：
        compare_results(first.results, second.results, tolerances={
            "leg_length": thresholds.REPRO_TOLERANCE_LENGTH,
            "pose.knee_L": thresholds.REPRO_TOLERANCE_ANGLE_DEG,
            "volume_ratio": thresholds.REPRO_TOLERANCE_RATIO,
        })
    """
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        raise TypeError(f"比べる結果は辞書にしてください（いまは {type(a).__name__} と {type(b).__name__}）")
    default = _check_tolerance("default_tolerance", default_tolerance)
    per_key: dict[str, float] = {}
    if tolerances is not None:
        if not isinstance(tolerances, Mapping):
            raise ValueError(f"tolerances は、キーから幅への辞書にしてください（いまは {_describe(tolerances)}）")
        for key, value in tolerances.items():
            if not isinstance(key, str):
                raise ValueError(f"tolerances のキーは文字にしてください（いまは {_describe(key)}）")
            per_key[_plain_text(key)] = _check_tolerance(f"tolerances[{key!r}]", value)
    flat_a: dict[str, Any] = {}
    flat_b: dict[str, Any] = {}
    _flatten(a, "", flat_a, "a")
    _flatten(b, "", flat_b, "b")

    differences: list[Difference] = []
    for key in sorted(flat_a.keys() | flat_b.keys()):
        if key not in flat_b:
            differences.append(Difference(key, flat_a[key], None, None, None, "missing"))
        elif key not in flat_a:
            differences.append(Difference(key, None, flat_b[key], None, None, "missing"))
        else:
            difference = _compare_leaf(key, flat_a[key], flat_b[key], per_key.get(key, default))
            if difference is not None:
                differences.append(difference)
    return differences


def is_reproducible(a: Mapping[str, Any], b: Mapping[str, Any], tolerances: Mapping[str, float] | None = None,
                    default_tolerance: float = 0.0) -> bool:
    """2回の結果が、再現性の幅の中で同じか（compare_results の違いが0か）"""
    return not compare_results(a, b, tolerances, default_tolerance)

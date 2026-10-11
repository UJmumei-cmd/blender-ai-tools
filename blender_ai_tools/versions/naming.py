"""A6 版のファイルの名前（bpy なし）。

版のファイルは、元のファイルと同じフォルダに「<元の名前>_v001.blend」「_v002.blend」… の名前で置く
（番号は3桁。999 を超えたら 1000、1001 … と桁が増える）。版ごとの記録（形・骨・ウェイトなどの控え。snapshot.py）は、
そのとなりの「<元の名前>_v001.snapshot」フォルダに置く。

決まり：
- 元のファイルも、ある版も上書きしない（CLAUDE.md 2章4）。次の番号は、いちばん大きい番号の次。
  あいだの空いた番号（_v002 を消した など）は使わない（あとから見て、並びが分からなくならないように）。
- 次の番号は、空のファイルを「無いときだけ作る」（O_EXCL）で取っておく。ほかのプロセスと同時に保存しても、
  同じ名前を取らない。取ったファイルは、保存する所（save_bpy.py）が中身の入ったファイルで置き換える。
- いま開いているファイルが版のファイル（model_v003.blend）でも、元の名前（model）で数える（base_stem）。
  元の名前のファイルが、ほんとうにあるかどうかは問わない。
- 名前の終わりの「_v」と数字は、この道具が作る形だけを版とみなす：小文字の v、3桁（001〜999）か、
  0 で始まらない4桁以上（1000〜）。_v000・_v1・_v0001・_V003 は版ではない（元の名前の一部）。
- 拡張子は大文字・小文字を区別しない（.blend も .BLEND も受け付ける）。作る版の拡張子は、いつも .blend。
  .blend1 などの控えのファイルは、版に数えない。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

# MARK: - 決まり

BLEND_SUFFIX = ".blend"
SNAPSHOT_SUFFIX = ".snapshot"
# 版の番号の桁（これより少ない桁は 0 で埋める）
VERSION_DIGITS = 3
# 名前の終わりの「_v」と番号。3桁か、0 で始まらない4桁以上（version_path が作る形だけ）
_VERSION_STEM = re.compile(r"^(?P<stem>.+)_v(?P<number>[0-9]{3}|[1-9][0-9]{3,})$")
# 次の番号を取るとき、ほかのプロセスに先に取られていたら、次の番号を試す。その回数の上限
MAX_RESERVE_TRIES = 1000


# MARK: - 入力の確かめ

def _as_path(path: Any, what: str) -> Path:
    if isinstance(path, (bytes, bytearray)) or not isinstance(path, (str, os.PathLike)):
        raise ValueError(f"{what} はファイルの場所（文字か Path）にしてください（いまの値：{path!r}）")
    result = Path(path)
    if str(result) in ("", "."):
        raise ValueError(f"{what} が空です")
    return result


def _check_blend(path: Any, what: str = "base_blend") -> Path:
    """.blend のファイルの場所にする。拡張子が .blend でない（大文字・小文字は問わない）、名前が無いときは ValueError"""
    result = _as_path(path, what)
    if result.suffix.lower() != BLEND_SUFFIX or not result.stem:
        raise ValueError(f"{what} は .blend のファイルにしてください（いまの値：{str(result)!r}）")
    return result


def _as_number(n: Any) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"版の番号は 1 以上の整数にしてください（いまの値：{n!r}）")
    return n


# MARK: - 名前を読む

def _split_stem(stem: str) -> tuple[str, int] | None:
    """「<元の名前>_vNNN」なら (元の名前, NNN)。版の形でなければ None"""
    match = _VERSION_STEM.match(stem)
    if match is None:
        return None
    number = int(match.group("number"))
    if number < 1:
        return None
    return match.group("stem"), number


def parse_version(path: str | os.PathLike[str]) -> tuple[str, int] | None:
    """版のファイル（<元の名前>_vNNN.blend）なら (元の名前, 番号)。.blend でない・版の形でないときは None"""
    result = _as_path(path, "path")
    if result.suffix.lower() != BLEND_SUFFIX:
        return None
    return _split_stem(result.stem)


def version_number(path: str | os.PathLike[str]) -> int | None:
    """版のファイルの番号。版のファイルでなければ None"""
    parsed = parse_version(path)
    return None if parsed is None else parsed[1]


def base_stem(path: str | os.PathLike[str]) -> str:
    """元の名前：拡張子を外し、終わりの「_vNNN」を1つだけ外した物（model_v003.blend → model）。
    .blend でないときは ValueError"""
    stem = _check_blend(path, "path").stem
    parsed = _split_stem(stem)
    return stem if parsed is None else parsed[0]


def base_path(path: str | os.PathLike[str]) -> Path:
    """元のファイルの場所（<フォルダ>/<元の名前>.blend）。そのファイルがあるかどうかは問わない"""
    checked = _check_blend(path, "path")
    return checked.parent / f"{base_stem(checked)}{BLEND_SUFFIX}"


def version_path(base_blend: str | os.PathLike[str], n: int) -> Path:
    """n 番目の版の場所：<フォルダ>/<元の名前>_v{n:03d}.blend。base_blend が版のファイルでも、元の名前で作る"""
    checked = _check_blend(base_blend)
    number = _as_number(n)
    return checked.parent / f"{base_stem(checked)}_v{number:0{VERSION_DIGITS}d}{BLEND_SUFFIX}"


def snapshot_dir(version_blend: str | os.PathLike[str]) -> Path:
    """版の記録を置くフォルダ：<フォルダ>/<元の名前>_vNNN.snapshot。版のファイルでないときは ValueError"""
    checked = _check_blend(version_blend, "version_blend")
    parsed = _split_stem(checked.stem)
    if parsed is None:
        raise ValueError(f"版のファイル（<名前>_v001.blend の形）ではありません：{checked.name}")
    stem, number = parsed
    return checked.parent / f"{stem}_v{number:0{VERSION_DIGITS}d}{SNAPSHOT_SUFFIX}"


# MARK: - フォルダの中を見る

def _entries(folder: Path) -> list[os.DirEntry[str]]:
    try:
        with os.scandir(folder) as found:
            return list(found)
    except FileNotFoundError:
        return []


def _entry_number(name: str, stem: str, suffix: str) -> int | None:
    """name が「<stem>_vNNN<suffix>」（拡張子は大文字・小文字を問わない）なら NNN"""
    if len(name) <= len(suffix) or name[-len(suffix):].lower() != suffix:
        return None
    parsed = _split_stem(name[:-len(suffix)])
    if parsed is None or parsed[0] != stem:
        return None
    return parsed[1]


def list_versions(base_blend: str | os.PathLike[str]) -> list[tuple[int, Path]]:
    """ディスクにある版のファイル：[(番号, 場所), ...]（番号の小さい順。同じ番号なら名前の順）。

    同じフォルダの「<元の名前>_vNNN.blend」のファイルだけ（拡張子の大文字・小文字は問わない）。フォルダ・.blend1・
    ほかの名前のファイル・下のフォルダの中の物は入れない。フォルダが無ければ空。
    中身が空のファイル（保存の途中か、保存に失敗した物）も入る（番号は使われているため）。開ける版は latest_version を使う。
    """
    checked = _check_blend(base_blend)
    stem = base_stem(checked)
    found = []
    for entry in _entries(checked.parent):
        number = _entry_number(entry.name, stem, BLEND_SUFFIX)
        if number is not None and entry.is_file():
            found.append((number, checked.parent / entry.name))
    return sorted(found, key=lambda item: (item[0], item[1].name))


def latest_version(base_blend: str | os.PathLike[str]) -> Path | None:
    """いちばん新しい版（番号のいちばん大きい、中身のある版のファイル）。無ければ None"""
    for _, path in reversed(list_versions(base_blend)):
        try:
            if path.stat().st_size > 0:
                return path
        except OSError:
            continue
    return None


def _taken_numbers(folder: Path, stem: str) -> set[int]:
    """使われている番号：版のファイル（.blend）か、版の記録（.snapshot）の名前がある番号（ファイルでもフォルダでも）"""
    taken = set()
    for entry in _entries(folder):
        for suffix in (BLEND_SUFFIX, SNAPSHOT_SUFFIX):
            number = _entry_number(entry.name, stem, suffix)
            if number is not None:
                taken.add(number)
    return taken


def _open_new(path: Path) -> int:
    """まだ無いファイルを作って開く。あるときは FileExistsError（上書きしない。リンクもたどらない）"""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    return os.open(path, flags, 0o666)


def next_version_path(base_blend: str | os.PathLike[str], *, reserve: bool = True) -> Path:
    """次の版の場所：いちばん大きい番号（版のファイルか版の記録がある番号）の次。

    reserve が True（ふつう）なら、その名前の空のファイルを「無いときだけ作る」で作って取っておく。
    ほかのプロセスが先に作っていたら、次の番号を試す（同時に保存しても、同じ名前を取らない）。
    使わなかったときは release_reservation で消す。
    reserve が False なら、ファイルを作らずに、いまの時点で空いている次の場所を返すだけ（表示のため）。
    フォルダが無いときは ValueError。
    """
    if not isinstance(reserve, bool):
        raise ValueError(f"reserve は True か False にしてください（いまの値：{reserve!r}）")
    checked = _check_blend(base_blend)
    folder = checked.parent
    if not folder.is_dir():
        raise ValueError(f"版を置くフォルダがありません：{folder}")
    start = max(_taken_numbers(folder, base_stem(checked)), default=0) + 1
    for number in range(start, start + MAX_RESERVE_TRIES):
        path = version_path(checked, number)
        if os.path.lexists(snapshot_dir(path)):
            continue
        if not reserve:
            if not os.path.lexists(path):
                return path
            continue
        try:
            os.close(_open_new(path))
        except FileExistsError:
            continue
        return path
    raise FileExistsError(f"版の番号を取れませんでした（{start} から {MAX_RESERVE_TRIES} 個、どれも使われていました）："
                          f"{folder / base_stem(checked)}")


def release_reservation(path: str | os.PathLike[str]) -> bool:
    """next_version_path で取っておいた版の場所を返す：中身が空のファイルなら消して True。
    中身があるファイル（保存できた版）・無いファイルは、そのままにして False"""
    checked = _check_blend(path, "path")
    try:
        if checked.is_file() and not checked.is_symlink() and checked.stat().st_size == 0:
            checked.unlink()
            return True
    except FileNotFoundError:
        pass
    return False

"""A6 版を保存する・前の版と比べる（Blender の中）。

- save_version：いまの Blender の中身を、新しい版のファイル（<元の名前>_v001.blend …）に写しとして保存し、
  となりに版の記録（<元の名前>_v001.snapshot/）を置く。
- previous_version：いちばん新しい版のファイル。
- compare_with_previous：いまの Blender の中身を、いちばん新しい版（か、渡した版）の記録と比べる。

決まり（CLAUDE.md 2章4「元のファイルを上書きしない」）：
- 元のファイルも、ある版も上書きしない。番号は naming.next_version_path で、空のファイルを「無いときだけ作る」で
  取っておく（ほかのプロセスと同時でも同じ名前にならない）。記録のフォルダも「無いときだけ作る」。どちらかが
  すでにあれば、次の番号にする。
- 保存は bpy.ops.wm.save_as_mainfile(copy=True)。いま開いているファイルの場所（bpy.data.filepath）も、
  「保存していない変更がある」の印（bpy.data.is_dirty）も変わらない。
- Blender は、ある名前のファイルに保存するとき、前のファイルを .blend1 に残す。取っておいた空のファイルの .blend1 を
  作らないように、同じフォルダの一時ファイルに保存してから、取っておいた名前に置き換える（os.replace）。
  置き換えるのは、自分で作った空のファイルだけ（中身が入っていたら、置き換えずに止める）。
- 途中で失敗したら、自分で作った物（一時ファイル・記録・取っておいた空のファイル）を消してから止まる。
- 新しい版を開くかどうか（版を戻す）は、ユーザーが決めて、ユーザーが開く。ここでは開かない。
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Any

import bpy

from .. import thresholds
from ..measure.results import Measurement
from . import naming
from .diff import AllowedChanges, Change
from .history import compare_snapshot_with_version
from .snapshot import ARRAYS_NAME, JSON_NAME, save_snapshot
from .snapshot_bpy import take_snapshot

# 次の番号を取ったあと、記録のフォルダがすでにあったとき、番号を取り直す回数の上限
MAX_SAVE_ATTEMPTS = 100


# MARK: - 元のファイル

def current_base(base_path: str | os.PathLike[str] | None = None) -> Path | None:
    """版の名前のもとにする元のファイル（<フォルダ>/<元の名前>.blend）。

    base_path を渡せばそれ、渡さなければ、いま開いているファイル（bpy.data.filepath）から作る
    （いまのファイルが版のファイル model_v003.blend なら model.blend）。まだ保存していないファイルで、
    base_path も無いときは None。.blend でないときは ValueError。
    """
    if base_path is None:
        current = bpy.data.filepath
        if not current:
            return None
        base_path = current
    return naming.base_path(base_path)


def _require_base(base_path: str | os.PathLike[str] | None) -> Path:
    base = current_base(base_path)
    if base is None:
        raise ValueError("このファイルは、まだ保存されていません。版を保存する前に、ふつうに保存するか、"
                         "base_path（元のファイルの場所）を渡してください")
    return base


# MARK: - 保存

def _reserve(base: Path) -> tuple[Path, Path]:
    """次の版の名前（空のファイル）と、記録のフォルダを取っておく"""
    for _ in range(MAX_SAVE_ATTEMPTS):
        target = naming.next_version_path(base)
        folder = naming.snapshot_dir(target)
        try:
            folder.mkdir()
        except FileExistsError:
            naming.release_reservation(target)
            continue
        except BaseException:
            naming.release_reservation(target)
            raise
        return target, folder
    raise FileExistsError(f"版の番号を取れませんでした（{MAX_SAVE_ATTEMPTS} 回、記録のフォルダがすでにありました）：{base}")


def _remove_record(folder: Path) -> None:
    """自分で作った記録のフォルダを消す（中の物は、この関数で作った2つのファイルと一時ファイルだけのはず）"""
    for name in (JSON_NAME, ARRAYS_NAME):
        (folder / name).unlink(missing_ok=True)
    for leftover in folder.glob(".snapshot.*.tmp"):
        leftover.unlink(missing_ok=True)
    try:
        folder.rmdir()
    except OSError:
        pass  # ほかの物が入っていたら残す（消さない）


def _save_copy(target: Path) -> None:
    """いまの中身を、写しとして target（取っておいた空のファイル）に保存する"""
    temp = target.parent / f".{target.stem}.{secrets.token_hex(8)}.saving{naming.BLEND_SUFFIX}"
    try:
        result = bpy.ops.wm.save_as_mainfile(filepath=str(temp), copy=True, check_existing=False)
        if "FINISHED" not in result or not temp.is_file() or temp.stat().st_size == 0:
            raise RuntimeError(f"版を保存できませんでした：{target.name}（{result}）")
        if not target.is_file() or target.is_symlink() or target.stat().st_size != 0:
            raise FileExistsError(f"取っておいた版の名前に、ほかの物が入っていました（上書きしません）：{target}")
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
        Path(f"{temp}@").unlink(missing_ok=True)  # Blender が書きかけに使う名前


def save_version(*, base_path: str | os.PathLike[str] | None = None, scene: Any = None) -> tuple[Path, Path]:
    """いまの中身を新しい版として保存する：(版のファイルの場所, 記録のフォルダの場所)。

    base_path：元のファイル（版の名前のもと）。None なら、いま開いているファイルから決める（current_base）。
        まだ保存していないファイルで base_path も無いときは ValueError。
    scene：記録を作るシーン（None なら、いまのシーン）。ファイルは、全部のシーンを入れて保存する。
    いま開いているファイルの場所と中身は変えない。元のファイル・ある版は上書きしない。
    """
    base = _require_base(base_path)
    if not base.parent.is_dir():
        raise ValueError(f"版を置くフォルダがありません：{base.parent}")
    snapshot = take_snapshot(scene)  # 先に読む（読めないときは、ファイルを1つも作らない）
    target, folder = _reserve(base)
    try:
        save_snapshot(snapshot, folder)
        _save_copy(target)
    except BaseException:
        _remove_record(folder)
        naming.release_reservation(target)
        raise
    return target, folder


# MARK: - 前の版と比べる

def previous_version(base_path: str | os.PathLike[str] | None = None) -> Path | None:
    """いちばん新しい版のファイル（中身のある物）。版が無い・まだ保存していないファイルのときは None"""
    base = current_base(base_path)
    if base is None:
        return None
    return naming.latest_version(base)


def compare_with_previous(*, base_path: str | os.PathLike[str] | None = None,
                          version_path: str | os.PathLike[str] | None = None, scene: Any = None,
                          allowed: AllowedChanges | dict[str, Any] | None = None,
                          tolerance: float = thresholds.REPRO_TOLERANCE_LENGTH,
                          language: str | None = None) -> tuple[list[Change], Measurement]:
    """いまの Blender の中身（scene の記録）を、前の版の記録と比べる：(変わった所, まとめの信号)。

    前の版は version_path（渡したとき）か、いちばん新しい版（previous_version）。
    前の版が無い・記録が読めないときは ([], 灰)。allowed は「変えてよい所」（None なら、どこも変えてはいけない）。
    """
    previous = Path(version_path) if version_path is not None else previous_version(base_path)
    snapshot = take_snapshot(scene)
    return compare_snapshot_with_version(snapshot, previous, allowed=allowed, tolerance=tolerance, language=language)

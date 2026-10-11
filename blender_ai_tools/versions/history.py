"""A6 保存した版どうしを比べる・版の一覧を出す（bpy なし）。

版のファイルのとなりの記録（<元の名前>_vNNN.snapshot。naming.snapshot_dir）を読んで比べる。
.blend のファイルは開かない（開くのは、版を戻すときのユーザーの操作）。

記録が無い・読めないときは、比べられないので灰にする（測れなかった物は緑にしない。CLAUDE.md 2章5）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .. import texts, thresholds
from ..measure.deformation import CHANGES_NAME
from ..measure.results import Measurement
from .diff import AllowedChanges, Change, diff_snapshots, summarize_changes
from .naming import list_versions, snapshot_dir, version_number
from .snapshot import Snapshot, load_snapshot


def _gray(reason: str, details: dict[str, Any]) -> Measurement:
    return Measurement.unmeasured(CHANGES_NAME, "count", reason, details)


def load_version_snapshot(version_blend: str | os.PathLike[str]) -> Snapshot:
    """版のファイルの記録を読む。版のファイルの名前でない・記録が無い・読めないときは ValueError"""
    return load_snapshot(snapshot_dir(version_blend))


def _try_load(version_blend: Path, language: str | None) -> tuple[Snapshot | None, Measurement | None]:
    try:
        return load_version_snapshot(version_blend), None
    except (ValueError, OSError) as error:
        reason = texts.text("versions.snapshot_missing", language=language, name=version_blend.name, error=str(error))
        return None, _gray(reason, {"version": str(version_blend)})


def compare_snapshot_with_version(snapshot: Snapshot, version_blend: str | os.PathLike[str] | None, *,
                                  allowed: AllowedChanges | dict[str, Any] | None = None,
                                  tolerance: float = thresholds.REPRO_TOLERANCE_LENGTH,
                                  language: str | None = None) -> tuple[list[Change], Measurement]:
    """記録（ふつうは、いまの Blender の中の物）を、版 version_blend の記録と比べる：(変わった所, まとめの信号)。

    version_blend が None（前の版が無い）・記録が読めないときは、([], 灰)。
    """
    if not isinstance(snapshot, Snapshot):
        raise ValueError(f"snapshot は Snapshot にしてください（いまの型：{type(snapshot).__name__}）")
    if version_blend is None:
        return [], _gray(texts.text("versions.no_previous", language=language), {})
    path = Path(version_blend)
    previous, gray = _try_load(path, language)
    if previous is None:
        return [], gray
    changes = diff_snapshots(previous, snapshot, allowed=allowed, tolerance=tolerance)
    measurement = summarize_changes(changes, language=language)
    measurement.details["previous_version"] = str(path)
    return changes, measurement


def compare_versions(old_version: str | os.PathLike[str], new_version: str | os.PathLike[str], *,
                     allowed: AllowedChanges | dict[str, Any] | None = None,
                     tolerance: float = thresholds.REPRO_TOLERANCE_LENGTH,
                     language: str | None = None) -> tuple[list[Change], Measurement]:
    """保存した2つの版の記録を比べる：(変わった所, まとめの信号)。どちらかの記録が読めないときは ([], 灰)"""
    old_path, new_path = Path(old_version), Path(new_version)
    old, gray = _try_load(old_path, language)
    if old is None:
        return [], gray
    new, gray = _try_load(new_path, language)
    if new is None:
        return [], gray
    changes = diff_snapshots(old, new, allowed=allowed, tolerance=tolerance)
    measurement = summarize_changes(changes, language=language)
    measurement.details.update({"previous_version": str(old_path), "version": str(new_path)})
    return changes, measurement


def saved_versions(base_blend: str | os.PathLike[str]) -> list[tuple[int, Path]]:
    """開ける版：[(番号, 場所), ...]（番号の小さい順）。中身が空のファイル（保存の途中・失敗）は入れない。
    版を戻すときは、ユーザーがこの中から選んで開く"""
    found = []
    for number, path in list_versions(base_blend):
        try:
            if path.stat().st_size > 0:
                found.append((number, path))
        except OSError:
            continue
    return found


def compare_latest(base_blend: str | os.PathLike[str], *, allowed: AllowedChanges | dict[str, Any] | None = None,
                   tolerance: float = thresholds.REPRO_TOLERANCE_LENGTH,
                   language: str | None = None) -> tuple[list[Change], Measurement]:
    """いちばん新しい版と、その1つ前の版を比べる（新しい版で何が変わったか）。版が2つ無いときは ([], 灰)"""
    versions = saved_versions(base_blend)
    if len(versions) < 2:
        return [], _gray(texts.text("versions.no_previous", language=language), {})
    return compare_versions(versions[-2][1], versions[-1][1], allowed=allowed, tolerance=tolerance,
                            language=language)


def version_lines(base_blend: str | os.PathLike[str], language: str | None = None) -> list[str]:
    """版の一覧の文（「第 1 版：model_v001.blend」など。番号の小さい順）"""
    return [texts.text("versions.version_entry", language=language, number=number, name=path.name)
            for number, path in saved_versions(base_blend)]


def is_version_file(path: str | os.PathLike[str]) -> bool:
    """版のファイルの名前（<元の名前>_vNNN.blend）か"""
    return version_number(path) is not None

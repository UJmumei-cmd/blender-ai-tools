"""画面の、いまの作業のあいだだけの覚え（.blend には保存しない）。

- 登録カードを読んだ物（カードのテキストが同じあいだは読み直さない。パネルは何度も描き直すため）
- 「確かめる」「登録する」の結果（理由と知らせ）
- 「骨の名前を調べる」の結果
- いまの報告（A11）：もとの報告（答えなし）と、decisions.json の答えを写した物。答えが変わるたびに写し直す
- 版の一覧（A6。パネルは描くたびに読まないので、ここに覚える）

別のファイルを開いたら、全部消す（on_load_post）。
パネルを描くときは、ここを読むだけにする（描くときに何も変えない）。
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bpy

from .. import registry
from ..bones import map_bones
from ..catalog import CatalogError
from ..model import card_bpy
from ..model.card import ModelCard, answerable_bones
from ..report.decisions import DECISIONS_FILE, DecisionStore
from ..report.model import Report
from . import preferences

# MARK: - 登録カードを読んだ物


@dataclass(frozen=True)
class BoneQuestion:
    """ユーザーに聞く骨1本（辞書に無い・1つに決まらない・取り合いになった骨）。

    name: 骨の名前（モデルの名前のまま）
    answer: 登録カードに入っている答え（共通の名前か "extra"）。まだ答えていないときは None
    candidates: 1つに決まらなかった骨の、候補の共通の名前
    conflict: 取り合いになった共通の名前（取り合いでなければ None）
    """

    name: str
    answer: str | None
    candidates: tuple[str, ...] = ()
    conflict: str | None = None


@dataclass(frozen=True)
class CardView:
    """パネルに出す、登録カードを読んだ物。

    exists: カードのテキストがあるか
    card: 読めたカード（無い・読めないときは None）
    error: 読めなかった理由（読めたときは None）
    identity: カードが「どのモデルを、いつ測った物か」の印（種類や答えを変えても変わらない）
    questions: ユーザーに聞く骨（まだ答えていない骨が先。どちらもモデルの骨の順）
    """

    exists: bool
    card: ModelCard | None = None
    error: str | None = None
    identity: str | None = None
    questions: tuple[BoneQuestion, ...] = ()


_NO_CARD = CardView(exists=False)


def card_identity(card: ModelCard) -> str:
    """カードの印：アーマチュアの名前・作った時刻・指紋。部品の種類・骨の答えを変えても同じになる"""
    return json.dumps([card.armature_name, card.created_at, card.fingerprints], sort_keys=True, ensure_ascii=False)


def _questions(card: ModelCard) -> tuple[BoneQuestion, ...]:
    if card.skeleton is None:
        return ()
    names = card.skeleton["names"]
    raw = map_bones(names)
    conflict_of = {name: canonical for canonical, claimants in raw.conflicts.items() for name in claimants}
    # 候補のうち、ほかの骨がもう使っている共通の名前は出さない（答えても card.set_bone_answer が断るため）
    taken = set(card.bone_mapping["mapping"].values())
    questions = [BoneQuestion(name=name, answer=card.bone_answers.get(name),
                              candidates=tuple(c for c in raw.ambiguous.get(name, ()) if c not in taken),
                              conflict=conflict_of.get(name))
                 for name in answerable_bones(names)]
    return tuple([q for q in questions if q.answer is None] + [q for q in questions if q.answer is not None])


# 前に読んだ (カードのテキスト, 読んだ物)
_view_cache: tuple[str, CardView] | None = None


def card_view() -> CardView:
    """いまのファイルの登録カードを読む（テキストが前と同じなら、前に読んだ物を使う）。読めないときも例外にしない"""
    global _view_cache
    holder = bpy.data.texts.get(card_bpy.TEXT_NAME)
    if holder is None:
        return _NO_CARD
    content = holder.as_string()
    if _view_cache is not None and _view_cache[0] == content:
        return _view_cache[1]
    try:
        card = ModelCard.from_json(content)
    except ValueError as error:
        view = CardView(exists=True, error=str(error))
    else:
        view = CardView(exists=True, card=card, identity=card_identity(card),
                        questions=() if card.is_legacy else _questions(card))
    _view_cache = (content, view)
    return view


# MARK: - 操作の結果


@dataclass(frozen=True)
class CardStatus:
    """「確かめる」「登録する」の結果。

    identity: そのときのカードの印（card_identity。カードが無い・読めないときは None）
    checked: True は「確かめる」、False は「登録する」の結果
    reasons: 測り直しが要る理由（その時の言語の文）
    notes: 知らせ（その時の言語の文）
    """

    identity: str | None
    checked: bool
    reasons: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class BoneCheck:
    """「骨の名前を調べる」の結果。

    armature: 調べたアーマチュアの名前
    names: 調べた骨の名前（同じ名前は1つ。モデルの骨の順）
    result: 道具 bones.map_names の結果（JSON の形）
    """

    armature: str
    names: tuple[str, ...]
    result: dict[str, Any]


_card_status: CardStatus | None = None
_bone_check: BoneCheck | None = None


def set_card_status(status: CardStatus | None) -> None:
    global _card_status
    _card_status = status


def card_status(identity: str | None) -> CardStatus | None:
    """いまのカード（印 identity）についての、前の結果。別のカードの結果なら None"""
    if _card_status is None or _card_status.identity != identity:
        return None
    return _card_status


def set_bone_check(check: BoneCheck | None) -> None:
    global _bone_check
    _bone_check = check


def bone_check() -> BoneCheck | None:
    return _bone_check


# MARK: - 報告（A11）


@dataclass(frozen=True)
class ReportView:
    """パネルに出す報告。

    report: 答え（decisions.json）を写した報告
    baseline_red: 同じ点検の種類の、記録した赤の数の基準（無ければ None）
    store_error: decisions.json が読めなかった理由（読めたとき・ファイルを保存していないときは None）
    html_path: 書いた HTML の場所（まだ書いていなければ None）
    """

    report: Report
    baseline_red: int | None = None
    store_error: str | None = None
    html_path: Path | None = None


_report_base: Report | None = None
_report_view: ReportView | None = None


def decisions_path() -> Path | None:
    """答えの記録（decisions.json）の場所：作業フォルダの中。作業フォルダが決まらない（保存していない）ときは None"""
    folder = preferences.work_dir_path(preferences.get_preferences())
    return None if folder is None else folder / DECISIONS_FILE


def load_store() -> DecisionStore | None:
    """答えの記録を読む。作業フォルダが決まらないときは None。読めないときは ValueError"""
    path = decisions_path()
    return None if path is None else DecisionStore(path)


def show_report(report: Report | None) -> None:
    """いまの報告を置きかえる（答えは decisions.json から写す。None なら報告を消す）"""
    global _report_base, _report_view
    _report_base = report
    _report_view = None
    if report is not None:
        apply_decisions()


def apply_decisions(store: DecisionStore | None = None) -> None:
    """いまの報告に、答えの記録を写し直す。store を渡さなければ、decisions.json を読む（読めなくても例外にしない）"""
    global _report_view
    if _report_base is None:
        _report_view = None
        return
    html_path = _report_view.html_path if _report_view is not None else None
    error = None
    if store is None:
        try:
            store = load_store()
        except (ValueError, OSError) as failure:
            error = str(failure)
    if store is None:
        _report_view = ReportView(report=_report_base, store_error=error, html_path=html_path)
        return
    _report_view = ReportView(report=_report_base.with_decisions(store.decisions),
                              baseline_red=store.baseline_red(_report_base.kind), html_path=html_path)


def set_report_html(path: Path) -> None:
    """いまの報告の HTML を書いた場所を覚える"""
    global _report_view
    if _report_view is not None:
        _report_view = ReportView(report=_report_view.report, baseline_red=_report_view.baseline_red,
                                  store_error=_report_view.store_error, html_path=path)


def report_view() -> ReportView | None:
    return _report_view


# MARK: - 版の一覧（A6）


@dataclass(frozen=True)
class VersionEntry:
    """保存した版1つ：番号・場所・ファイルの名前"""

    number: int
    path: str
    name: str


# 版の一覧（番号の小さい順）。ファイルを保存していない・読めなかったときは None
_versions: tuple[VersionEntry, ...] | None = None


def refresh_versions() -> tuple[VersionEntry, ...] | None:
    """版の一覧を、目録の道具 versions.list で読み直して覚える（ファイルを保存していないときは None）"""
    global _versions
    if not bpy.data.filepath:
        _versions = None
        return None
    try:
        result = registry.build_catalog().call("versions.list", {})
    except (ValueError, OSError, CatalogError):
        _versions = None
        return None
    _versions = tuple(VersionEntry(number=entry["number"], path=entry["path"], name=entry["name"])
                      for entry in result["versions"])
    return _versions


def versions() -> tuple[VersionEntry, ...] | None:
    """覚えている版の一覧（読み直さない）"""
    return _versions


def clear() -> None:
    """覚えた物を全部消す"""
    global _view_cache, _card_status, _bone_check, _report_base, _report_view, _versions
    _view_cache = None
    _card_status = None
    _bone_check = None
    _report_base = None
    _report_view = None
    _versions = None


@bpy.app.handlers.persistent
def on_load_post(*_args: Any) -> None:
    """別のファイルを開いたら、前のファイルの結果を消す"""
    clear()

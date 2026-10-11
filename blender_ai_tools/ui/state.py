"""画面の、いまの作業のあいだだけの覚え（.blend には保存しない）。

- 登録カードを読んだ物（カードのテキストが同じあいだは読み直さない。パネルは何度も描き直すため）
- 「確かめる」「登録する」の結果（理由と知らせ）
- 「骨の名前を調べる」の結果

別のファイルを開いたら、全部消す（on_load_post）。
"""

import json
from dataclasses import dataclass
from typing import Any

import bpy

from ..bones import map_bones
from ..model import card_bpy
from ..model.card import ModelCard, answerable_bones

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


def clear() -> None:
    """覚えた物を全部消す"""
    global _view_cache, _card_status, _bone_check
    _view_cache = None
    _card_status = None
    _bone_check = None


@bpy.app.handlers.persistent
def on_load_post(*_args: Any) -> None:
    """別のファイルを開いたら、前のファイルの結果を消す"""
    clear()

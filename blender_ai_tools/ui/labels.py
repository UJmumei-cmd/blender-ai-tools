"""画面に出す名前（部品の種類・服の種類・骨の答え・信号）と、一覧（EnumProperty の items）を作る所。

文はどれも texts の screen.* から、いまの言語で作る。数字の文字は、ここ（プログラム）で作る。
一覧の items を関数で渡すときは、Blender が文字を読み終わるまで、Python が文字を持っていないといけない
（持っていないと、文字がこわれることがある）。そのため、作った一覧は言語ごとに _ITEMS に残す。
"""

import unicodedata
from typing import Any

from .. import texts
from ..bones import canonical_names
from ..measure.results import Signal
from ..model.card import CLOTH_TYPE_IDS, EXTRA_ANSWER, PART_KINDS
from ..report.model import DECISIONS, decision_label
from ..texts import text

# MARK: - 名前


def kind_label(kind: str) -> str:
    """部品の種類の名前（体・服 など）。文の無い種類は、そのままの名前"""
    key = f"screen.kind_{kind}"
    return text(key) if key in texts.TEXTS else kind


def cloth_label(cloth_type: str | None) -> str:
    """服の種類の名前。まだ選んでいない（None）ときは「服の種類を選ぶ」"""
    if cloth_type is None:
        return text("screen.cloth_type_none")
    key = f"screen.cloth_{cloth_type}"
    return text(key) if key in texts.TEXTS else cloth_type


def chosen_kind_label(kind: str, kind_source: str) -> str:
    """一覧に出す、いまの種類。名前から推測しただけの種類は「（推測）」を付ける"""
    label = kind_label(kind)
    return text("screen.kind_guess", kind=label) if kind_source != "user" else label


def answer_label(answer: str) -> str:
    """骨の答えの名前（共通の名前はそのまま、"extra" は「揺れ物・人の形ではない」）"""
    return text("screen.answer_extra") if answer == EXTRA_ANSWER else answer


def tool_label(tool_id: str) -> str:
    """目録の道具の名前（書き出して確かめる など）。文の無い道具は、道具の id のまま"""
    key = f"screen.tool_{tool_id.replace('.', '_')}"
    return text(key) if key in texts.TEXTS else tool_id


def job_status_label(status: str) -> str:
    """仕事のようすの名前（順番待ち・動いています など。jobs.status_*）。知らないようすは、そのままの名前"""
    key = f"jobs.status_{status}"
    return text(key) if key in texts.TEXTS else status


def job_text(key: str | None, values: Any, fallback: str | None = None) -> str | None:
    """仕事の進み具合・理由の文（texts のキーと値）。キーが無い・値が合わないときは fallback（無ければキーのまま）"""
    if not key:
        return fallback
    if key in texts.TEXTS and isinstance(values, dict):
        try:
            return text(key, **values)
        except (KeyError, ValueError, IndexError):
            pass
    return fallback if fallback else key


# MARK: - 信号

# 信号の色の四角（Blender のストリップの色：01 赤、03 黄、04 緑、09 灰）
SIGNAL_ICONS = {
    Signal.GREEN: "STRIP_COLOR_04",
    Signal.YELLOW: "STRIP_COLOR_03",
    Signal.GRAY: "STRIP_COLOR_09",
    Signal.RED: "STRIP_COLOR_01",
}


def signal_label(signal: Signal) -> str:
    """信号の名前（緑・黄・灰・赤）"""
    return text(f"screen.signal_{signal.name.lower()}")


def signal_row(signal: Signal, line: str) -> str:
    """信号の名前を前に付けた1行（色だけに頼らないため）"""
    return text("screen.signal_row", signal=signal_label(signal), text=line)


def count_text(value: float | None) -> str:
    """数（個・本）の文字。値が無いときは「測れていません」"""
    return text("screen.not_measured") if value is None else str(int(round(value)))


def millimeters_text(meters: float) -> str:
    """メートルの値を、mm の文字（小数1けた）にする。報告は mm（CLAUDE.md 3章）"""
    return f"{meters * 1000.0:.1f}"


# MARK: - 一覧（EnumProperty の items）

# 言語ごとに作った一覧（Blender が文字を使い終わるまで、ここで持つ）
_ITEMS: dict[tuple[str, str], list[tuple[str, str, str]]] = {}


def _cached(name: str, build: Any) -> list[tuple[str, str, str]]:
    key = (name, texts.current_language())
    items = _ITEMS.get(key)
    if items is None:
        items = _ITEMS[key] = build()
    return items


def kind_items(self: Any = None, context: Any = None) -> list[tuple[str, str, str]]:
    """部品の種類の一覧（model.card.PART_KINDS の順）"""
    return _cached("kind", lambda: [(kind, kind_label(kind), "") for kind in PART_KINDS])


def cloth_items(self: Any = None, context: Any = None) -> list[tuple[str, str, str]]:
    """服の種類の一覧（model.card.CLOTH_TYPE_IDS の順）"""
    return _cached("cloth", lambda: [(cloth, cloth_label(cloth), "") for cloth in CLOTH_TYPE_IDS])


def answer_items(self: Any = None, context: Any = None) -> list[tuple[str, str, str]]:
    """骨の答えの一覧：「揺れ物・人の形ではない」と、共通の名前（MMD 準標準。辞書の順）"""
    return _cached("answer", lambda: [(EXTRA_ANSWER, text("screen.answer_extra"), text("screen.answer_extra_tip"))]
                   + [(name, name, "") for name in canonical_names()])


def decision_items(self: Any = None, context: Any = None) -> list[tuple[str, str, str]]:
    """報告の1件への答えの一覧（report.model.DECISIONS の順）"""
    return _cached("decision", lambda: [(decision, decision_label(decision), "") for decision in DECISIONS])


def clear_cache() -> None:
    _ITEMS.clear()


# MARK: - 長い文を折り返す


def text_width(line: str) -> int:
    """文の幅の目安（半角を 1、全角を 2 と数える）"""
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in line)


def wrap_text(line: str, width: int) -> list[str]:
    """文を、幅 width（text_width の数え方）ごとに折り返す。英語は空白で切り、日本語は字ごとに切る。
    width が小さすぎても、1行に1字は入れる"""
    if width < 2:
        raise ValueError(f"折り返しの幅は 2 以上にしてください（いまは {width}）")
    lines: list[str] = []
    current = ""
    current_width = 0
    for word in _words(line):
        word_width = text_width(word)
        # 語の後ろの空白は、行の終わりなら要らないので、入るかどうかは空白を除いて決める
        if current and current_width + text_width(word.rstrip()) > width:
            lines.append(current.rstrip())
            current, current_width = "", 0
            word = word.lstrip()
            word_width = text_width(word)
        while word_width > width:
            # 1語が幅より長いとき（長い英語の語・場所の文字など）は、字ごとに切る（ここに来るとき current は空）
            head, word = _split_at_width(word, width)
            lines.append(head.rstrip())
            word_width = text_width(word)
        current += word
        current_width += word_width
    if current.strip() or not lines:
        lines.append(current.rstrip())
    return lines


def _words(line: str) -> list[str]:
    """英語の語（後ろの空白を付けた物）と、日本語の字に分ける"""
    words: list[str] = []
    current = ""
    for char in line:
        if text_width(char) == 2:
            if current:
                words.append(current)
                current = ""
            words.append(char)
        elif char == " ":
            current += char
            words.append(current)
            current = ""
        else:
            current += char
    if current:
        words.append(current)
    return words


def _split_at_width(word: str, width: int) -> tuple[str, str]:
    """word を、幅 width までの頭と残りに分ける（頭には少なくとも1字入れる）"""
    used = 0
    for index, char in enumerate(word):
        used += text_width(char)
        if used > width:
            return (word[:index], word[index:]) if index else (word[:1], word[1:])
    return word, ""

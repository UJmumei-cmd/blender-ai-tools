"""画面の文と報告の文（日本語が元。英語にも切り替えられる）。

文の形は、この texts パッケージだけに置く（CLAUDE.md 7章「文の形は1か所にまとめます」）。
分野ごとに1つのファイル（screen.py：画面・versions.py・motion.py など）に TEXTS = {キー: {"ja": ..., "en": ...}} を書き、
ここでまとめる。ファイルを足したら、下の _MODULES にも足す（文字列からファイルを探すことはしない）。

- キーは「分野.名前」の形（例 "screen.register_card"・"versions.saved"）。分野の名前は、ファイルの名前と同じにする。
- 数字などは {name} の形で入れる（例 "体の穴が {count} 個あります"）。日本語と英語で、入れる名前をそろえる。
- 数字の文字は、呼ぶ側（プログラム）が作って渡す。LLM・Jeff に書かせない（CLAUDE.md 2章3）。
"""

from __future__ import annotations

import string
from types import MappingProxyType
from typing import Any

from . import card, export, jobs, motion, report, screen, versions

_MODULES = (screen, versions, motion, jobs, report, card, export)

LANGUAGES = ("ja", "en")
# 文が無い言語のときに使う言語（日本語が元）
FALLBACK_LANGUAGE = "ja"

_language = FALLBACK_LANGUAGE


def _merge() -> MappingProxyType:
    merged: dict[str, dict[str, str]] = {}
    for module in _MODULES:
        area = module.__name__.rsplit(".", 1)[-1]
        for key, entry in module.TEXTS.items():
            if key in merged:
                raise ValueError(f"文のキーが重なっています: {key}")
            if not key.startswith(area + "."):
                raise ValueError(f"文のキー {key} は「{area}.」で始めてください（{area}.py の中の文）")
            merged[key] = MappingProxyType(dict(entry))
    return MappingProxyType(merged)


TEXTS = _merge()


def placeholders(template: str) -> set[str]:
    """文の形の中の {name} の名前"""
    names = set()
    for _, field, _, _ in string.Formatter().parse(template):
        if field is not None:
            names.add(field)
    return names


def problems() -> list[str]:
    """文の形のまちがい（日本語が無い、英語が無い、入れる名前が日英で違う、名前の無い {} がある）。無ければ空"""
    found = []
    for key, entry in TEXTS.items():
        if not entry.get("ja"):
            found.append(f"{key}: 日本語がありません")
        if not entry.get("en"):
            found.append(f"{key}: 英語がありません")
        unknown = set(entry) - set(LANGUAGES)
        if unknown:
            found.append(f"{key}: 知らない言語 {sorted(unknown)}")
        try:
            names = {lang: placeholders(entry[lang]) for lang in LANGUAGES if entry.get(lang)}
        except ValueError as error:
            found.append(f"{key}: 文の形が読めません（{error}）")
            continue
        if len({frozenset(n) for n in names.values()}) > 1:
            found.append(f"{key}: 入れる名前が日本語と英語で違います {names}")
        if any("" in n or any(part.isdigit() for part in n) for n in names.values()):
            found.append(f"{key}: 名前の無い {{}} や番号の {{0}} は使わず、{{name}} の形にしてください")
    return found


def set_language(language: str) -> None:
    """画面と報告の言語を決める（"ja" か "en"）"""
    global _language
    if language not in LANGUAGES:
        raise ValueError(f"言語は {LANGUAGES} のどれかにしてください（いまは {language!r}）")
    _language = language


def current_language() -> str:
    return _language


def text(key: str, *, language: str | None = None, **values: Any) -> str:
    """キーの文を、いまの言語（language を渡せばその言語）で返す。{name} には values を入れる。

    その言語の文が無いときは日本語を使う。キーが無い、入れる値が足りない・多いときは KeyError・ValueError。
    """
    entry = TEXTS[key]
    lang = language or _language
    if lang not in LANGUAGES:
        raise ValueError(f"言語は {LANGUAGES} のどれかにしてください（いまは {lang!r}）")
    template = entry.get(lang) or entry[FALLBACK_LANGUAGE]
    needed = placeholders(template)
    given = set(values)
    if needed != given:
        raise ValueError(f"{key}: 入れる値が合いません（いる：{sorted(needed)}、渡した：{sorted(given)}）")
    return template.format(**values)

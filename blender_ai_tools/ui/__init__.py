"""画面（段階1）：拡張機能の設定・3D ビューの横のパネル・ボタンの操作。bpy を使うのは ui/ の下と *_bpy.py だけ。

- register：設定のクラスを登録し、設定の言語で画面の文の言語を決めて（texts.set_language）、
  パネルと操作を、その言語の名前で登録する。別のファイルを開いたときに、覚えた結果を消す係も付ける。
- unregister：全部を外し、言語を登録の前に戻す（テストなど、同じ Blender の中のほかの物に影響を残さない）。
- apply_language：言語を変える。パネルと操作の名前（タブの名前も）を、新しい言語で作り直す。
  設定の言語を変えたときに、設定のクラスから呼ばれる。
"""

import bpy

from .. import texts
from . import labels, operators, panel, preferences, state

# 登録する順（設定のクラスは別に登録する）。パネルは親を先にする
CLASSES = (*operators.CLASSES, *panel.CLASSES)

# 登録したクラス（登録した順）
_registered: list[type] = []
# 登録する前の言語（unregister で戻す）
_previous_language: str | None = None


def apply_language(choice: str | None = None) -> str:
    """画面の文の言語を決めて、登録してあるパネルと操作の名前を作り直す。決めた言語（"ja"・"en"）を返す。

    choice：設定の言語の選び方（"auto"・"ja"・"en"）。None なら、いまの設定を使う
    （拡張機能を設定から有効にしていないときは、初めの値 "auto"）。
    """
    if choice is None:
        choice = preferences.get_preferences().language
    language = preferences.resolve_language(choice, preferences.blender_locale())
    texts.set_language(language)
    _refresh_classes()
    return language


def _register_class(cls: type) -> None:
    cls.localize()
    bpy.utils.register_class(cls)
    _registered.append(cls)


def _unregister_classes() -> list[type]:
    """登録したクラスを、後ろから外す。外したクラスを、登録した順で返す"""
    classes = list(_registered)
    while _registered:
        bpy.utils.unregister_class(_registered.pop())
    return classes


def _refresh_classes() -> None:
    """登録してあるパネルと操作を、いまの言語の名前で登録し直す（Blender は登録したあとの名前の変更を読まないため）"""
    for cls in _unregister_classes():
        _register_class(cls)


def is_registered() -> bool:
    return bool(_registered) or preferences.preferences_registered()


def register() -> None:
    global _previous_language
    if is_registered():
        unregister()
    _previous_language = texts.current_language()
    try:
        preferences.register_preferences()
        language = apply_language()
        if preferences.preferences_language() != language:
            # 設定の項目の名前と説明を、決まった言語で作り直す（保存した設定の値は、そのまま残る）
            preferences.register_preferences()
        for cls in CLASSES:
            _register_class(cls)
        if state.on_load_post not in bpy.app.handlers.load_post:
            bpy.app.handlers.load_post.append(state.on_load_post)
    except Exception:
        unregister()
        raise


def unregister() -> None:
    global _previous_language
    if state.on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(state.on_load_post)
    _unregister_classes()
    preferences.unregister_preferences()
    state.clear()
    labels.clear_cache()
    if _previous_language is not None:
        texts.set_language(_previous_language)
        _previous_language = None

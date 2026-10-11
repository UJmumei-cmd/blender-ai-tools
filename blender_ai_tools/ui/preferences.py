"""拡張機能の設定（紅魔館の場所・書き出しに使う Blender・作業フォルダ・画面の言語）。

- 設定のクラスの bl_idname は、拡張機能のいちばん上のパッケージの名前（入れた拡張機能なら
  "bl_ext.user_default.blender_ai_tools" のような名前）。文字で書かず、__package__ から作る。
- get_preferences は、拡張機能を設定から有効にしていないとき（テストでパッケージを直接読み込んだときなど）も、
  初めの値を持つ物を返す。
- 項目の名前と説明（ツールチップ）は、登録するときの言語で作る（register_preferences）。
  設定の画面の文（draw）は、いまの言語で毎回作る。

このファイルでは from __future__ import annotations を使わない（bpy.props の注釈を、Blender が読めるようにするため）。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bpy

from .. import texts
from ..export.convert import koumakan_paths
from ..texts import text

# MARK: - 決まり

# 拡張機能のいちばん上のパッケージの名前（このファイルは <いちばん上>.ui.preferences）
ADDON_ID = __package__.rpartition(".")[0]

# 画面の言語の選び方："auto" は Blender の言語に合わせる
LANGUAGE_AUTO = "auto"
LANGUAGE_CHOICES = (LANGUAGE_AUTO, *texts.LANGUAGES)

# 作業フォルダを書いていないときに、.blend の横に作るフォルダの名前の後ろ（「<ファイルの名前>_ai_tools」）
WORK_DIR_SUFFIX = "_ai_tools"

# Blender の画面の文の翻訳と重ならないようにする、文の分け方の名前（この名前の翻訳は無いので、文はそのまま出る）
TRANSLATION_CONTEXT = "ai_tools"


# MARK: - 言語


def resolve_language(choice: str, locale: str) -> str:
    """設定の言語（"auto"・"ja"・"en"）と Blender の言語（bpy.app.translations.locale）から、使う言語を決める。

    "auto"：Blender の言語が日本語（"ja" で始まる）なら "ja"、ほかは "en"。
    知らない選び方は ValueError。
    """
    if choice not in LANGUAGE_CHOICES:
        raise ValueError(f"言語の選び方は {LANGUAGE_CHOICES} のどれかにしてください（いまは {choice!r}）")
    if choice != LANGUAGE_AUTO:
        return choice
    return "ja" if isinstance(locale, str) and locale.lower().startswith("ja") else "en"


def blender_locale() -> str:
    """Blender のいまの言語（"ja_JP"・"en_US" など。分からないときは ""）"""
    return bpy.app.translations.locale or ""


# MARK: - 設定の値


@dataclass(frozen=True)
class PreferenceValues:
    """拡張機能を有効にしていないときの設定（どれも初めの値）。設定のクラスと同じ名前の項目を持つ"""

    koumakan_repo: str = ""
    blender_path: str = ""
    work_dir: str = ""
    language: str = LANGUAGE_AUTO


DEFAULTS = PreferenceValues()


def get_preferences(context: Any = None) -> Any:
    """いまの設定。拡張機能を設定から有効にしていないときは、初めの値の PreferenceValues を返す"""
    context = bpy.context if context is None else context
    try:
        addon = context.preferences.addons.get(ADDON_ID)
    except AttributeError:
        return DEFAULTS
    preferences = getattr(addon, "preferences", None) if addon is not None else None
    return DEFAULTS if preferences is None else preferences


def _absolute(value: str) -> str:
    """Blender の「//」で始まる場所（.blend からの場所）を、ふつうの場所にする"""
    return bpy.path.abspath(value)


def koumakan_repo_path(preferences: Any) -> Path | None:
    """紅魔館のリポジトリの場所。書いていないときは None"""
    value = preferences.koumakan_repo.strip()
    return Path(_absolute(value)) if value else None


def blender_executable(preferences: Any) -> str:
    """書き出しと裏の作業に使う Blender のファイル。書いていないときは、いま動いている Blender"""
    value = preferences.blender_path.strip()
    return _absolute(value) if value else bpy.app.binary_path


def work_dir_path(preferences: Any, blend_path: str | None = None) -> Path | None:
    """作業フォルダ。書いていないときは、.blend の横の「<ファイルの名前>_ai_tools」。

    blend_path を渡さなければ、いま開いている .blend を使う。.blend を保存していない（場所が無い）ときは None。
    フォルダは作らない（作るのは、使う道具）。
    """
    value = preferences.work_dir.strip()
    if value:
        return Path(_absolute(value))
    blend = bpy.data.filepath if blend_path is None else blend_path
    if not blend:
        return None
    path = Path(blend)
    return path.parent / f"{path.stem}{WORK_DIR_SUFFIX}"


# MARK: - 設定のクラス


def _language_changed(self: Any, context: Any) -> None:
    """言語を変えたら、すぐに画面の文の言語を変える（パネルとボタンの名前も作り直す）"""
    from . import apply_language

    apply_language()


def _language_items() -> list[tuple[str, str, str]]:
    return [(choice, text(f"screen.pref_language_{choice}"), "") for choice in LANGUAGE_CHOICES]


class AIToolsPreferences(bpy.types.AddonPreferences):
    """拡張機能の設定。項目は register_preferences が、いまの言語で作る"""

    bl_idname = ADDON_ID

    def draw(self, context: Any) -> None:
        draw_preferences(self, self.layout, context)


def _annotations() -> dict[str, Any]:
    """設定の項目（名前と説明は、いまの言語）"""
    return {
        "language": bpy.props.EnumProperty(
            name=text("screen.pref_language"), description=text("screen.pref_language_tip"),
            items=_language_items(), default=LANGUAGE_AUTO, update=_language_changed,
            translation_context=TRANSLATION_CONTEXT),
        "koumakan_repo": bpy.props.StringProperty(
            name=text("screen.pref_koumakan_repo"), description=text("screen.pref_koumakan_repo_tip"),
            default="", subtype="DIR_PATH", translation_context=TRANSLATION_CONTEXT),
        "blender_path": bpy.props.StringProperty(
            name=text("screen.pref_blender_path"), description=text("screen.pref_blender_path_tip"),
            default="", subtype="FILE_PATH", translation_context=TRANSLATION_CONTEXT),
        "work_dir": bpy.props.StringProperty(
            name=text("screen.pref_work_dir"), description=text("screen.pref_work_dir_tip"),
            default="", subtype="DIR_PATH", translation_context=TRANSLATION_CONTEXT),
    }


# 設定のクラスを、どの言語で作ったか（登録していないときは None）
_built_language: str | None = None


def preferences_registered() -> bool:
    return _built_language is not None


def preferences_language() -> str | None:
    """設定のクラスの項目の名前を作った言語（登録していないときは None）"""
    return _built_language


def register_preferences() -> None:
    """設定のクラスを、いまの言語の名前と説明で登録する。登録してあれば、作り直す（保存した値はそのまま残る）"""
    global _built_language
    unregister_preferences()
    AIToolsPreferences.__annotations__ = _annotations()
    bpy.utils.register_class(AIToolsPreferences)
    _built_language = texts.current_language()


def unregister_preferences() -> None:
    global _built_language
    if _built_language is None:
        return
    _built_language = None
    bpy.utils.unregister_class(AIToolsPreferences)


# MARK: - 設定の画面


def koumakan_status(preferences: Any) -> tuple[str, str]:
    """紅魔館の場所の、いまのようす（文, アイコン）"""
    repo = koumakan_repo_path(preferences)
    if repo is None:
        return text("screen.pref_koumakan_empty"), "INFO"
    try:
        koumakan_paths(repo)
    except ValueError as error:
        return text("screen.pref_koumakan_missing", error=str(error)), "ERROR"
    return text("screen.pref_koumakan_found"), "CHECKMARK"


def draw_preferences(preferences: Any, layout: Any, context: Any) -> None:
    """設定の画面。文はいまの言語で作り、Blender の翻訳を通さない（translate=False）"""
    layout.prop(preferences, "language", text=text("screen.pref_language"), translate=False)

    box = layout.box()
    box.prop(preferences, "koumakan_repo", text=text("screen.pref_koumakan_repo"), translate=False)
    message, icon = koumakan_status(preferences)
    box.label(text=message, icon=icon, translate=False)

    box = layout.box()
    box.prop(preferences, "blender_path", text=text("screen.pref_blender_path"), translate=False)
    box.label(text=text("screen.pref_blender_default", path=bpy.app.binary_path), icon="INFO", translate=False)

    box = layout.box()
    box.prop(preferences, "work_dir", text=text("screen.pref_work_dir"), translate=False)
    box.label(text=text("screen.pref_work_dir_default", suffix=WORK_DIR_SUFFIX), icon="INFO", translate=False)
    folder = work_dir_path(preferences)
    if folder is None:
        box.label(text=text("screen.pref_work_dir_unsaved"), translate=False)
    else:
        box.label(text=text("screen.pref_work_dir_now", path=str(folder)), translate=False)

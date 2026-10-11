"""パネルのボタンが動かす操作（オペレーター）。どれも bl_idname は "ai_tools.*"。

- モデルの形は変えない。.blend のファイルは保存しない（保存するのはユーザー。CLAUDE.md 2章4）。
  変えるのは、.blend の中の登録カードのテキスト（ai_tools_card.json）だけ。
- 骨の名前を調べるときは、道具の目録（A8）の bones.map_names を目録から呼ぶ（CLAUDE.md 2章1）。
- まちがいは self.report({"ERROR"}, 文) で知らせ、画面に例外を出さない。文は texts の screen.*（いまの言語）。
  道具（card.py など）の日本語の説明は、その文の中に入れて見せる。
- 名前と説明（ツールチップ）は、いまの言語で作る（localize。言語を変えたら作り直す）。

このファイルでは from __future__ import annotations を使わない（bpy.props の注釈を、Blender が読めるようにするため）。
"""

import traceback
from collections.abc import Callable
from typing import Any

import bpy

from .. import registry
from ..catalog import CatalogError
from ..model import card_bpy, extract_bpy
from ..model.card import (
    CLOTH_TYPE_IDS, EXTRA_ANSWER, PART_KINDS, ModelCard, set_bone_answer, set_material_kind, set_part_kind,
)
from ..texts import text
from . import labels, state
from .preferences import TRANSLATION_CONTEXT

# 骨の名前の対応を調べる、目録の道具の id
MAP_NAMES_TOOL = "bones.map_names"
# 読めなかった前の登録カードを残すテキストの名前（同じ名前があれば、Blender が .001 などを付ける）
UNREADABLE_CARD_BACKUP = "ai_tools_card_unreadable.json"


# MARK: - まちがいの知らせ方


class Refused(Exception):
    """操作を断る。message は、いまの言語の文（そのまま見せる）"""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def run_guarded(operator: Any, failure_key: str, action: Callable[[], set[str]]) -> set[str]:
    """action を動かす。断った・まちがいがあったときは ERROR で知らせて {"CANCELLED"} を返す（例外を画面に出さない）。

    failure_key：ValueError（道具の日本語の説明）を入れる文のキー（{error} を1つ持つ文）
    """
    try:
        return action()
    except Refused as refused:
        operator.report({"ERROR"}, refused.message)
    except (ValueError, CatalogError) as error:
        operator.report({"ERROR"}, text(failure_key, error=str(error)))
    except Exception as error:  # 思いがけないエラーも、画面に例外を出さない（くわしくはコンソールに出す）
        traceback.print_exc()
        operator.report({"ERROR"}, text("screen.unexpected_error", error=f"{type(error).__name__}: {error}"))
    return {"CANCELLED"}


# MARK: - 登録カードを変える（bpy を使わない部分）


def editable_card() -> ModelCard:
    """いまのファイルの登録カード（変えてよい物）。無い・古い形なら Refused、読めないなら ValueError"""
    card = card_bpy.load_card()
    if card is None:
        raise Refused(text("screen.no_card_error"))
    if card.is_legacy:
        raise Refused(text("screen.legacy_card_error"))
    return card


def _current_choice(card: ModelCard, object_name: str, material_name: str) -> Any:
    """いまの部品（material_name が空）か材質。無い名前は ValueError（card.py の日本語の説明）"""
    part = card.part(object_name)
    return part.material(material_name) if material_name else part


def with_kind(card: ModelCard, object_name: str, material_name: str, kind: str) -> ModelCard:
    """部品（material_name が空）か材質の種類を、ユーザーが選んだ新しいカード。

    kind は model.card.PART_KINDS のどれか（ほかは ValueError）。服のまま種類を選び直したときは、服の種類を残す。
    服から別の種類にしたときは、服の種類を外す（服でない物は服の種類を持たない）。
    """
    if kind not in PART_KINDS:
        raise ValueError(f"部品の種類は {'・'.join(PART_KINDS)} のどれかにしてください（いまは {kind!r}）")
    current = _current_choice(card, object_name, material_name)
    cloth_type = current.cloth_type if kind == "cloth" and current.kind == "cloth" else None
    if material_name:
        return set_material_kind(card, object_name, material_name, kind, cloth_type)
    return set_part_kind(card, object_name, kind, cloth_type)


def with_cloth_type(card: ModelCard, object_name: str, material_name: str, cloth_type: str) -> ModelCard:
    """服の部品（material_name が空）か材質の、服の種類をユーザーが選んだ新しいカード（種類は「服」と決まる）。

    cloth_type は model.card.CLOTH_TYPE_IDS のどれか（ほかは ValueError）。種類が服でない物は Refused。
    """
    if cloth_type not in CLOTH_TYPE_IDS:
        raise ValueError(f"服の種類は {'・'.join(CLOTH_TYPE_IDS)} のどれかにしてください（いまは {cloth_type!r}）")
    current = _current_choice(card, object_name, material_name)
    if current.kind != "cloth":
        name = f"{object_name} / {material_name}" if material_name else object_name
        raise Refused(text("screen.cloth_type_needs_cloth", name=name))
    if material_name:
        return set_material_kind(card, object_name, material_name, "cloth", cloth_type)
    return set_part_kind(card, object_name, "cloth", cloth_type)


def bone_check_counts(result: dict[str, Any], total: int) -> dict[str, int]:
    """bones.map_names の結果を、数にまとめる（報告の文に入れる数）"""
    return {
        "total": total,
        "mapped": len(result["mapping"]),
        "extras": len(result["known_extras"]),
        "unknown": len(result["unknown"]),
        "conflicts": len(result["conflicts"]),
        "missing": len(result["missing_standard"]) + len(result["missing_semi_standard"]),
    }


# MARK: - 名前と説明


class _Localized:
    """名前（bl_label）と説明（description）を、texts のキーから、いまの言語で作る操作のもと"""

    bl_translation_context = TRANSLATION_CONTEXT
    LABEL_KEY = ""
    TIP_KEY = ""

    @classmethod
    def localize(cls) -> None:
        """名前と項目を、いまの言語で作る（登録する前に呼ぶ）"""
        cls.bl_label = text(cls.LABEL_KEY)
        cls.__annotations__ = cls.properties()

    @classmethod
    def properties(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def description(cls, context: Any, properties: Any) -> str:
        return text(cls.TIP_KEY)


def _name_property() -> Any:
    """部品・材質・骨の名前（パネルが入れる。やり直しの画面には出さない）"""
    return bpy.props.StringProperty(options={"HIDDEN", "SKIP_SAVE"})


# MARK: - モデル（A1 登録カード）


class AI_TOOLS_OT_register_card(_Localized, bpy.types.Operator):
    """モデルを測って登録カードを作り、ファイルの中に置く。前のカードの、ユーザーが決めた物は写す"""

    bl_idname = "ai_tools.register_card"
    bl_label = ""
    bl_options = {"REGISTER", "UNDO"}
    LABEL_KEY = "screen.op_register_card"
    TIP_KEY = "screen.register_card_tip"

    def execute(self, context: Any) -> set[str]:
        return run_guarded(self, "screen.register_failed", self._register)

    def _register(self) -> set[str]:
        notes: list[str] = []
        unreadable_text = None
        try:
            previous = card_bpy.load_card()
        except ValueError:
            # 読めないカードは写せない。中身は消さずに、別の名前のテキストに残す（作り直したカードを置く前に）
            previous = None
            unreadable_text = bpy.data.texts[card_bpy.TEXT_NAME].as_string()
        card, carried = card_bpy.build_card_with_notes(previous=previous)
        if unreadable_text is not None:
            backup = bpy.data.texts.new(UNREADABLE_CARD_BACKUP)
            backup.write(unreadable_text)
            backup.use_fake_user = True
            notes.append(text("screen.previous_card_unreadable", backup=backup.name))
        notes.extend(carried)
        card_bpy.save_card(card)
        state.set_card_status(state.CardStatus(identity=state.card_identity(card), checked=False, notes=tuple(notes)))
        for note in notes:
            self.report({"WARNING"}, note)
        bones = len(card.skeleton["names"]) if card.skeleton is not None else 0
        self.report({"INFO"}, text("screen.card_registered", parts=len(card.parts), bones=bones))
        return {"FINISHED"}


class AI_TOOLS_OT_check_card(_Localized, bpy.types.Operator):
    """登録カードが、いまのモデルにまだ合うかを確かめる（何も変えない）"""

    bl_idname = "ai_tools.check_card"
    bl_label = ""
    bl_options = {"REGISTER"}
    LABEL_KEY = "screen.op_check_card"
    TIP_KEY = "screen.check_card_tip"

    def execute(self, context: Any) -> set[str]:
        return run_guarded(self, "screen.check_failed", self._check)

    def _check(self) -> set[str]:
        result = card_bpy.check_card()
        identity = state.card_identity(result.card) if result.card is not None else None
        state.set_card_status(state.CardStatus(identity=identity, checked=True, reasons=tuple(result.reasons),
                                               notes=tuple(result.notes)))
        if result.card is None:
            # カードが無い・読めない：理由は1つ（card_bpy の文）
            self.report({"WARNING"}, result.reasons[0] if result.reasons else text("screen.no_card_error"))
            return {"CANCELLED"}
        for note in result.notes:
            self.report({"INFO"}, note)
        if result.reasons:
            for reason in result.reasons:
                self.report({"WARNING"}, reason)
            self.report({"WARNING"}, text("screen.check_stale", count=len(result.reasons)))
        else:
            self.report({"INFO"}, text("screen.check_ok"))
        return {"FINISHED"}


class AI_TOOLS_OT_set_part_kind(_Localized, bpy.types.Operator):
    """部品（material_name を入れたときは材質）の種類を、ユーザーが一覧から選ぶ"""

    bl_idname = "ai_tools.set_part_kind"
    bl_label = ""
    bl_options = {"REGISTER", "UNDO"}
    LABEL_KEY = "screen.op_set_part_kind"
    TIP_KEY = "screen.set_part_kind_tip"

    @classmethod
    def properties(cls) -> dict[str, Any]:
        return {
            "object_name": _name_property(),
            "material_name": _name_property(),
            "kind": bpy.props.EnumProperty(name=text("screen.prop_kind"), items=labels.kind_items,
                                           translation_context=TRANSLATION_CONTEXT),
        }

    def execute(self, context: Any) -> set[str]:
        return run_guarded(self, "screen.card_change_failed", self._apply)

    def _apply(self) -> set[str]:
        card = with_kind(editable_card(), self.object_name, self.material_name, self.kind)
        card_bpy.save_card(card)
        kind = labels.kind_label(self.kind)
        if self.material_name:
            message = text("screen.material_kind_set", name=self.object_name, material=self.material_name, kind=kind)
        else:
            message = text("screen.kind_set", name=self.object_name, kind=kind)
        self.report({"INFO"}, message)
        return {"FINISHED"}


class AI_TOOLS_OT_set_cloth_type(_Localized, bpy.types.Operator):
    """服の部品（material_name を入れたときは材質）の、服の種類をユーザーが一覧から選ぶ"""

    bl_idname = "ai_tools.set_cloth_type"
    bl_label = ""
    bl_options = {"REGISTER", "UNDO"}
    LABEL_KEY = "screen.op_set_cloth_type"
    TIP_KEY = "screen.set_cloth_type_tip"

    @classmethod
    def properties(cls) -> dict[str, Any]:
        return {
            "object_name": _name_property(),
            "material_name": _name_property(),
            "cloth_type": bpy.props.EnumProperty(name=text("screen.prop_cloth_type"), items=labels.cloth_items,
                                                 translation_context=TRANSLATION_CONTEXT),
        }

    def execute(self, context: Any) -> set[str]:
        return run_guarded(self, "screen.card_change_failed", self._apply)

    def _apply(self) -> set[str]:
        card = with_cloth_type(editable_card(), self.object_name, self.material_name, self.cloth_type)
        card_bpy.save_card(card)
        cloth = labels.cloth_label(self.cloth_type)
        if self.material_name:
            message = text("screen.material_cloth_type_set", name=self.object_name, material=self.material_name,
                           cloth=cloth)
        else:
            message = text("screen.cloth_type_set", name=self.object_name, cloth=cloth)
        self.report({"INFO"}, message)
        return {"FINISHED"}


# MARK: - 骨（A3 骨の名前の対応）


class AI_TOOLS_OT_check_bones(_Localized, bpy.types.Operator):
    """モデルの骨の名前を、道具の目録の bones.map_names で MMD 準標準の名前に対応させる（何も変えない）"""

    bl_idname = "ai_tools.check_bones"
    bl_label = ""
    bl_options = {"REGISTER"}
    LABEL_KEY = "screen.op_check_bones"
    TIP_KEY = "screen.check_bones_tip"

    def execute(self, context: Any) -> set[str]:
        return run_guarded(self, "screen.bones_check_failed", self._check)

    def _check(self) -> set[str]:
        armature = extract_bpy.find_model().armature
        names = tuple(dict.fromkeys(extract_bpy.bone_names(armature)))
        # 目録に載った道具だけを、目録から動かす（値はスキーマで確かめられる）
        result = registry.build_catalog().call(MAP_NAMES_TOOL, {"names": list(names)})
        state.set_bone_check(state.BoneCheck(armature=armature.name, names=names, result=result))
        self.report({"INFO"}, text("screen.bones_checked", **bone_check_counts(result, len(names))))
        return {"FINISHED"}


class AI_TOOLS_OT_answer_bone(_Localized, bpy.types.Operator):
    """辞書に無い骨に、共通の名前（MMD 準標準）か「揺れ物・人の形ではない」を答える。答えは登録カードに入る。

    ボタンから押すと、共通の名前の一覧（文字で探せる）が出る。answer を入れて動かすと、すぐに答える。
    """

    bl_idname = "ai_tools.answer_bone"
    bl_label = ""
    bl_options = {"REGISTER", "UNDO"}
    bl_property = "answer"
    LABEL_KEY = "screen.op_answer_bone"
    TIP_KEY = "screen.answer_choose_tip"

    @classmethod
    def properties(cls) -> dict[str, Any]:
        return {
            "bone_name": _name_property(),
            "answer": bpy.props.EnumProperty(name=text("screen.prop_answer"), items=labels.answer_items,
                                             options={"SKIP_SAVE"}, translation_context=TRANSLATION_CONTEXT),
        }

    @classmethod
    def description(cls, context: Any, properties: Any) -> str:
        if properties is not None and properties.is_property_set("answer") and properties.answer == EXTRA_ANSWER:
            return text("screen.answer_extra_tip")
        return text(cls.TIP_KEY)

    def invoke(self, context: Any, event: Any) -> set[str]:
        context.window_manager.invoke_search_popup(self)
        return {"RUNNING_MODAL"}

    def execute(self, context: Any) -> set[str]:
        return run_guarded(self, "screen.answer_failed", self._apply)

    def _apply(self) -> set[str]:
        card = set_bone_answer(editable_card(), self.bone_name, self.answer)
        card_bpy.save_card(card)
        stored = card.bone_answers[self.bone_name]
        self.report({"INFO"}, text("screen.answer_set", bone=self.bone_name, answer=labels.answer_label(stored)))
        return {"FINISHED"}


class AI_TOOLS_OT_clear_bone_answer(_Localized, bpy.types.Operator):
    """骨の答えを消す（その骨は、また「分からない骨」になる）"""

    bl_idname = "ai_tools.clear_bone_answer"
    bl_label = ""
    bl_options = {"REGISTER", "UNDO"}
    LABEL_KEY = "screen.op_clear_bone_answer"
    TIP_KEY = "screen.answer_clear_tip"

    @classmethod
    def properties(cls) -> dict[str, Any]:
        return {"bone_name": _name_property()}

    def execute(self, context: Any) -> set[str]:
        return run_guarded(self, "screen.answer_failed", self._apply)

    def _apply(self) -> set[str]:
        card = set_bone_answer(editable_card(), self.bone_name, None)
        card_bpy.save_card(card)
        self.report({"INFO"}, text("screen.answer_cleared", bone=self.bone_name))
        return {"FINISHED"}


# MARK: - 次の段階（A10 から）

# 次の段階のボタン：(名前, 文のキー, 説明のキー, アイコン)。裏で動かす係（A10）ができるまで押せない
NEXT_FEATURES = (
    ("export", "screen.next_export", "screen.next_export_tip", "EXPORT"),
    ("motion_check", "screen.next_motion_check", "screen.next_motion_check_tip", "ARMATURE_DATA"),
    ("versions", "screen.next_versions", "screen.next_versions_tip", "FILE_TICK"),
)
_NEXT_TIPS = {name: tip for name, _, tip, _ in NEXT_FEATURES}


class AI_TOOLS_OT_coming_soon(_Localized, bpy.types.Operator):
    """まだ使えない道具のボタン（見えるが押せない。poll がいつも False）"""

    bl_idname = "ai_tools.coming_soon"
    bl_label = ""
    bl_options = {"INTERNAL"}
    LABEL_KEY = "screen.op_coming_soon"
    TIP_KEY = "screen.coming_soon"

    @classmethod
    def properties(cls) -> dict[str, Any]:
        return {"feature": _name_property()}

    @classmethod
    def poll(cls, context: Any) -> bool:
        return False

    @classmethod
    def description(cls, context: Any, properties: Any) -> str:
        tip = _NEXT_TIPS.get(properties.feature) if properties is not None else None
        return text(tip) if tip is not None else text(cls.TIP_KEY)

    def execute(self, context: Any) -> set[str]:
        self.report({"WARNING"}, text("screen.coming_soon"))
        return {"CANCELLED"}


CLASSES = (
    AI_TOOLS_OT_register_card,
    AI_TOOLS_OT_check_card,
    AI_TOOLS_OT_set_part_kind,
    AI_TOOLS_OT_set_cloth_type,
    AI_TOOLS_OT_check_bones,
    AI_TOOLS_OT_answer_bone,
    AI_TOOLS_OT_clear_bone_answer,
    AI_TOOLS_OT_coming_soon,
)

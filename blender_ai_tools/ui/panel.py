"""3D ビューの横のパネル（タブ「AI の道具」）。段階1は、道具ごとの日本語のボタンが並ぶ簡単なパネル（CLAUDE.md 10章）。

- モデル：登録する・確かめる、カードのまとめ（背の高さ・信号）、部品と材質の種類を選ぶ一覧
- 骨：骨の名前を調べる、聞きたい骨と答え方（共通の名前の一覧から選ぶ・揺れ物など）、足りない骨
- 書き出し・確かめの動き・版：見えるが、まだ押せない（裏で動かす係 A10 のあとで動く）

文はどれも texts の screen.* から作り、Blender の翻訳を通さない（translate=False）。
描くときは何も変えない（読むだけ）。数字の文字は labels（プログラム）が作る。
"""

import traceback
from collections.abc import Callable, Sequence
from typing import Any

import bpy

from ..measure.results import Signal
from ..model import extract_bpy
from ..model.card import (
    CARD_CURRENT_NAME, EXTRA_ANSWER, HEIGHT, SEMI_STANDARD_NAME, ModelCard, summarize_card,
)
from ..texts import text
from . import labels, operators, state
from .preferences import TRANSLATION_CONTEXT

# 一覧（部品と材質・聞きたい骨・調べた骨）に1回に出す行の数。多いモデルでも描き直しが重くならないようにする
# （画面の決まり。基準の数字ではない）
MAX_LISTED_ROWS = 40
# 文の幅の目安：半角1字の幅（ピクセル。画面の大きさ 1.0 のとき）と、パネルの左右とアイコンの分の幅
_CHAR_PIXELS = 6.5
_MARGIN_PIXELS = 56
# パネルの幅が分からないとき（テストなど）の、1行の幅（半角の字の数）
_DEFAULT_WIDTH = 40
_MIN_WIDTH = 16


# MARK: - 描くための小さな道具


def line_width(context: Any) -> int:
    """パネルの1行に入る幅（半角の字の数の目安）"""
    region = getattr(context, "region", None)
    width = getattr(region, "width", None)
    if not width:
        return _DEFAULT_WIDTH
    try:
        scale = float(context.preferences.system.ui_scale) or 1.0
    except AttributeError:
        scale = 1.0
    return max(_MIN_WIDTH, int((width - _MARGIN_PIXELS * scale) / (_CHAR_PIXELS * scale)))


def wrapped_label(layout: Any, context: Any, line: str, icon: str = "NONE") -> None:
    """長い文を折り返して、何行かのラベルにする（2行目からは、アイコンの分を空ける）"""
    column = layout.column(align=True)
    for index, part in enumerate(labels.wrap_text(line, line_width(context))):
        if index == 0:
            column.label(text=part, icon=icon, translate=False)
        else:
            column.label(text=part, icon="BLANK1" if icon != "NONE" else "NONE", translate=False)


def _more_rows(layout: Any, context: Any, count: int) -> None:
    if count > 0:
        wrapped_label(layout, context, text("screen.more_rows", count=count), icon="THREE_DOTS")


def draw_safely(layout: Any, context: Any, draw: Callable[[Any, Any], None]) -> None:
    """描く。思いがけないエラーは、パネルに1行で出し、くわしくはコンソールに出す（画面を止めない）"""
    try:
        draw(layout, context)
    except Exception as error:
        traceback.print_exc()
        layout.label(text=text("screen.unexpected_error", error=f"{type(error).__name__}: {error}"), icon="ERROR",
                     translate=False)


# MARK: - いちばん上


def draw_main(layout: Any, context: Any) -> None:
    """どのモデルを扱うか（書き出しと同じ決まりで選んだ物）"""
    try:
        model = extract_bpy.find_model()
    except ValueError as error:
        wrapped_label(layout, context, text("screen.model_not_found", error=str(error)), icon="ERROR")
    else:
        layout.label(text=text("screen.model_found", armature=model.armature.name, meshes=len(model.meshes)),
                     icon="OUTLINER_OB_ARMATURE", translate=False)
    wrapped_label(layout, context, text("screen.stage_note"), icon="INFO")


# MARK: - モデル（A1 登録カード）


def draw_model(layout: Any, context: Any) -> None:
    row = layout.row(align=True)
    row.operator(operators.AI_TOOLS_OT_register_card.bl_idname, text=text("screen.register_card"), icon="ADD",
                 translate=False)
    row.operator(operators.AI_TOOLS_OT_check_card.bl_idname, text=text("screen.check_card"), icon="VIEWZOOM",
                 translate=False)
    view = state.card_view()
    if not view.exists:
        wrapped_label(layout, context, text("screen.no_card"), icon="INFO")
        return
    if view.card is None:
        column = layout.column()
        column.alert = True
        wrapped_label(column, context, text("screen.card_unreadable", error=view.error or ""), icon="ERROR")
        return
    card = view.card
    if card.is_legacy:
        column = layout.column()
        column.alert = True
        wrapped_label(column, context, text("screen.card_legacy"), icon="ERROR")
    status = state.card_status(view.identity)
    draw_summary(layout, context, card, status)
    if status is not None:
        draw_status(layout, context, status)
    draw_parts(layout, context, card, editable=not card.is_legacy)


def _summary_line(name: str, value: float | None, details: dict[str, Any]) -> str:
    """まとめの1行（数はプログラムが入れる）"""
    count = labels.count_text(value)
    if name == SEMI_STANDARD_NAME:
        if value == 0:
            return text("screen.compliance_ok")
        missing = len(details.get("missing_standard", ())) + len(details.get("missing_semi_standard", ()))
        return text("screen.compliance_missing", missing=str(missing), conflicts=str(len(details.get("conflicts", {}))))
    key = f"screen.summary_{name}"
    if key in _SUMMARY_KEYS:
        return text(key, count=count)
    return text("screen.summary_generic", name=name, count=count)


# まとめの行の文のキー（summarize_card の結果の名前ごと）
_SUMMARY_KEYS = frozenset({
    "screen.summary_unknown_bones", "screen.summary_unconfirmed_parts", "screen.summary_body_measurements",
    "screen.summary_source_terms", "screen.summary_card_current",
})


def draw_summary(layout: Any, context: Any, card: ModelCard, status: state.CardStatus | None) -> None:
    """カードのまとめ：背の高さ（mm）と、カードの信号（model.card.summarize_card）"""
    box = layout.box()
    box.label(text=text("screen.summary_title"), icon="INFO", translate=False)
    height = card.value(HEIGHT)
    if height is None:
        reason = card.measurements["values"][HEIGHT]["reason"]
        wrapped_label(box, context, text("screen.height_unmeasured", reason=reason),
                      icon=labels.SIGNAL_ICONS[Signal.GRAY])
    else:
        box.label(text=text("screen.height", value=labels.millimeters_text(height)), icon="EMPTY_SINGLE_ARROW",
                  translate=False)
    reasons = list(status.reasons) if status is not None and status.checked else None
    for measurement in summarize_card(card, reasons):
        line = _summary_line(measurement.name, measurement.value, measurement.details)
        wrapped_label(box, context, labels.signal_row(measurement.signal, line),
                      icon=labels.SIGNAL_ICONS[measurement.signal])
        # 灰と赤は理由も出す（測り直しの理由は、下の「確かめた結果」に出す）
        if measurement.signal in (Signal.GRAY, Signal.RED) and measurement.reason and \
                measurement.name != CARD_CURRENT_NAME:
            wrapped_label(box, context, measurement.reason, icon="BLANK1")


def draw_status(layout: Any, context: Any, status: state.CardStatus) -> None:
    """「確かめる」「登録する」の結果（理由と知らせ）"""
    if not status.checked and not status.notes:
        return
    box = layout.box()
    box.label(text=text("screen.check_title" if status.checked else "screen.register_title"), translate=False)
    if status.checked:
        if status.reasons:
            column = box.column()
            column.alert = True
            wrapped_label(column, context, text("screen.check_stale", count=len(status.reasons)), icon="ERROR")
            for reason in status.reasons:
                wrapped_label(column, context, reason, icon="DOT")
        else:
            wrapped_label(box, context, text("screen.check_ok"), icon="CHECKMARK")
    for note in status.notes:
        wrapped_label(box, context, note, icon="INFO")


def _draw_choice(row: Any, object_name: str, material_name: str, choice: Any, editable: bool) -> None:
    """種類（と、服なら服の種類）を選ぶ一覧。決めていない物は赤い字にする"""
    row.alert = not choice.is_confirmed
    kind_text = labels.chosen_kind_label(choice.kind, choice.kind_source)
    if not editable:
        row.label(text=kind_text, translate=False)
        if choice.kind == "cloth":
            row.label(text=labels.cloth_label(choice.cloth_type), translate=False)
        return
    props = row.operator_menu_enum(operators.AI_TOOLS_OT_set_part_kind.bl_idname, "kind", text=kind_text,
                                   translate=False)
    props.object_name = object_name
    props.material_name = material_name
    if choice.kind == "cloth":
        props = row.operator_menu_enum(operators.AI_TOOLS_OT_set_cloth_type.bl_idname, "cloth_type",
                                       text=labels.cloth_label(choice.cloth_type), translate=False)
        props.object_name = object_name
        props.material_name = material_name


def draw_parts(layout: Any, context: Any, card: ModelCard, *, editable: bool) -> None:
    """部品（メッシュ）と、材質が2つ以上ある部品の材質ごとの、種類の一覧"""
    box = layout.box()
    box.label(text=text("screen.parts_title", count=len(card.parts)), icon="MESH_DATA", translate=False)
    total = sum(1 + len(part.materials) for part in card.parts)
    shown = 0
    for part in card.parts:
        if shown >= MAX_LISTED_ROWS:
            break
        column = box.column(align=True)
        column.label(text=part.object_name, icon="OBJECT_DATA", translate=False)
        _draw_choice(column.row(align=True), part.object_name, "", part, editable)
        shown += 1
        for material in part.materials:
            if shown >= MAX_LISTED_ROWS:
                break
            row = column.row(align=True)
            row.separator(factor=2.0)
            row.label(text=material.name, icon="MATERIAL", translate=False)
            row = column.row(align=True)
            row.separator(factor=2.0)
            _draw_choice(row, part.object_name, material.name, material, editable)
            shown += 1
    _more_rows(box, context, total - shown)


# MARK: - 骨（A3 骨の名前の対応）


def draw_bones(layout: Any, context: Any) -> None:
    layout.operator(operators.AI_TOOLS_OT_check_bones.bl_idname, text=text("screen.check_bones"), icon="BONE_DATA",
                    translate=False)
    view = state.card_view()
    card = view.card if view.card is not None and not view.card.is_legacy else None
    check = state.bone_check()
    if check is not None:
        draw_bone_check(layout, context, check, card)
    if card is not None:
        draw_questions(layout, context, view.questions)
        draw_missing(layout, context, card.bone_mapping)
        return
    if view.card is not None:
        wrapped_label(layout, context, text("screen.card_legacy"), icon="ERROR")
    elif view.exists:
        wrapped_label(layout, context, text("screen.card_unreadable", error=view.error or ""), icon="ERROR")
    if check is not None:
        _draw_unknown_without_card(layout, context, check)


def draw_bone_check(layout: Any, context: Any, check: state.BoneCheck, card: ModelCard | None) -> None:
    """「骨の名前を調べる」の結果の数"""
    box = layout.box()
    box.label(text=text("screen.bones_check_title", armature=check.armature), icon="ARMATURE_DATA", translate=False)
    counts = operators.bone_check_counts(check.result, len(check.names))
    column = box.column(align=True)
    column.label(text=text("screen.bones_total", count=counts["total"]), translate=False)
    column.label(text=text("screen.bones_mapped", count=counts["mapped"]), translate=False)
    column.label(text=text("screen.bones_extras", count=counts["extras"]), translate=False)
    column.label(text=text("screen.bones_unknown", count=counts["unknown"]), translate=False)
    column.label(text=text("screen.bones_conflicts", count=counts["conflicts"]), translate=False)
    column.label(text=text("screen.bones_missing", count=counts["missing"]), translate=False)
    if card is not None and card.armature_name == check.armature:
        asked = set(check.result["unknown"]).union(*check.result["conflicts"].values())
        column.label(text=text("screen.bones_answered", count=len(asked & set(card.bone_answers))), translate=False)


def _draw_unknown_without_card(layout: Any, context: Any, check: state.BoneCheck) -> None:
    """カードが無いとき：分からない骨を見せるだけ（答えはカードに入れるので、先に登録してもらう）"""
    unknown = list(check.result["unknown"])
    claimed = [name for names in check.result["conflicts"].values() for name in names]
    names = unknown + [name for name in claimed if name not in unknown]
    if not names:
        return
    box = layout.box()
    box.label(text=text("screen.questions_title", count=len(names)), icon="QUESTION", translate=False)
    ambiguous = check.result["ambiguous"]
    for name in names[:MAX_LISTED_ROWS]:
        row = box.row(align=True)
        row.label(text=name, icon="BONE_DATA", translate=False)
        if name in ambiguous:
            row.label(text=text("screen.candidates_list", names=labels_join(ambiguous[name])), translate=False)
    _more_rows(box, context, len(names) - MAX_LISTED_ROWS)
    wrapped_label(box, context, text("screen.answers_need_card"), icon="INFO")


def labels_join(names: Sequence[str]) -> str:
    """名前を、いまの言語の区切りでつなぐ"""
    return text("screen.list_separator").join(names)


def draw_questions(layout: Any, context: Any, questions: Sequence[state.BoneQuestion]) -> None:
    """聞きたい骨と、答え方（一覧から選ぶ・揺れ物など・候補・答えを消す）"""
    box = layout.box()
    box.label(text=text("screen.questions_title", count=len(questions)), icon="QUESTION", translate=False)
    if not questions:
        box.label(text=text("screen.no_questions"), icon="CHECKMARK", translate=False)
        return
    for question in questions[:MAX_LISTED_ROWS]:
        column = box.column(align=True)
        head = column.row(align=True)
        head.alert = question.answer is None
        head.label(text=question.name, icon="BONE_DATA", translate=False)
        if question.answer is not None:
            head.label(text=text("screen.answered", answer=labels.answer_label(question.answer)), translate=False)
            props = head.operator(operators.AI_TOOLS_OT_clear_bone_answer.bl_idname, text="", icon="X",
                                  translate=False)
            props.bone_name = question.name
        elif question.conflict is not None:
            head.label(text=text("screen.conflict_with", canonical=question.conflict), translate=False)
        buttons = column.row(align=True)
        props = buttons.operator(operators.AI_TOOLS_OT_answer_bone.bl_idname, text=text("screen.answer_choose"),
                                 icon="VIEWZOOM", translate=False)
        props.bone_name = question.name
        direct = buttons.row(align=True)
        direct.operator_context = "EXEC_DEFAULT"
        props = direct.operator(operators.AI_TOOLS_OT_answer_bone.bl_idname, text=text("screen.answer_extra"),
                                translate=False)
        props.bone_name = question.name
        props.answer = EXTRA_ANSWER
        if question.candidates and question.answer is None:
            candidates = column.row(align=True)
            candidates.operator_context = "EXEC_DEFAULT"
            candidates.label(text=text("screen.candidates"), translate=False)
            for candidate in question.candidates:
                props = candidates.operator(operators.AI_TOOLS_OT_answer_bone.bl_idname, text=candidate,
                                            translate=False)
                props.bone_name = question.name
                props.answer = candidate
    _more_rows(box, context, len(questions) - MAX_LISTED_ROWS)


def draw_missing(layout: Any, context: Any, bone_mapping: dict[str, Any]) -> None:
    """足りない準標準の骨（段階1では骨を足さない。その骨が要る確かめは灰になる。CLAUDE.md 3章）"""
    missing = list(bone_mapping["missing_standard"]) + list(bone_mapping["missing_semi_standard"])
    if not missing:
        return
    box = layout.box()
    wrapped_label(box, context, text("screen.missing_title", count=len(missing)),
                  icon=labels.SIGNAL_ICONS[Signal.YELLOW])
    wrapped_label(box, context, labels_join(missing))


# MARK: - 次の段階


def draw_next(layout: Any, context: Any) -> None:
    """書き出し・確かめの動き・版のボタン（見えるが押せない）"""
    column = layout.column(align=True)
    column.enabled = False
    for feature, label_key, _tip_key, icon in operators.NEXT_FEATURES:
        props = column.operator(operators.AI_TOOLS_OT_coming_soon.bl_idname, text=text(label_key), icon=icon,
                                translate=False)
        props.feature = feature
    wrapped_label(layout, context, text("screen.next_note"), icon="INFO")


# MARK: - パネル


class _Panel:
    """3D ビューの横のパネルのもと。名前とタブの名前は、いまの言語で作る（localize）"""

    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_translation_context = TRANSLATION_CONTEXT
    LABEL_KEY = ""

    @classmethod
    def localize(cls) -> None:
        cls.bl_label = text(cls.LABEL_KEY)
        cls.bl_category = text("screen.tab")


class VIEW3D_PT_ai_tools(_Panel, bpy.types.Panel):
    bl_idname = "VIEW3D_PT_ai_tools"
    bl_label = ""
    bl_category = ""
    LABEL_KEY = "screen.panel_title"

    def draw(self, context: Any) -> None:
        draw_safely(self.layout, context, draw_main)


class VIEW3D_PT_ai_tools_model(_Panel, bpy.types.Panel):
    bl_idname = "VIEW3D_PT_ai_tools_model"
    bl_parent_id = "VIEW3D_PT_ai_tools"
    bl_label = ""
    bl_category = ""
    LABEL_KEY = "screen.section_model"

    def draw(self, context: Any) -> None:
        draw_safely(self.layout, context, draw_model)


class VIEW3D_PT_ai_tools_bones(_Panel, bpy.types.Panel):
    bl_idname = "VIEW3D_PT_ai_tools_bones"
    bl_parent_id = "VIEW3D_PT_ai_tools"
    bl_label = ""
    bl_category = ""
    LABEL_KEY = "screen.section_bones"

    def draw(self, context: Any) -> None:
        draw_safely(self.layout, context, draw_bones)


class VIEW3D_PT_ai_tools_next(_Panel, bpy.types.Panel):
    bl_idname = "VIEW3D_PT_ai_tools_next"
    bl_parent_id = "VIEW3D_PT_ai_tools"
    bl_label = ""
    bl_category = ""
    bl_options = {"DEFAULT_CLOSED"}
    LABEL_KEY = "screen.section_next"

    def draw(self, context: Any) -> None:
        draw_safely(self.layout, context, draw_next)


# 親のパネルを先にする（登録は前から、解除は後ろから）
CLASSES = (VIEW3D_PT_ai_tools, VIEW3D_PT_ai_tools_model, VIEW3D_PT_ai_tools_bones, VIEW3D_PT_ai_tools_next)

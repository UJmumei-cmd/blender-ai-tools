"""3D ビューの横のパネル（タブ「AI の道具」）。段階1は、道具ごとの日本語のボタンが並ぶ簡単なパネル（CLAUDE.md 10章）。

- モデル：登録する・確かめる、カードのまとめ（背の高さ・信号）、部品と材質の種類を選ぶ一覧
- 骨：骨の名前を調べる、聞きたい骨と答え方（共通の名前の一覧から選ぶ・揺れ物など）、足りない骨
- 確かめと書き出し：版の保存と前の版との比べ、保存した版の一覧、書き出して確かめる（E1＋E2）、確かめの動き（A5）
- 裏の仕事（A10）：順番待ち・動いている・終わった仕事、進み具合、中止・結果を見る・一覧から消す
- 報告（A11）：いちばん上の一言、赤の数の基準との比べ、1件ずつの信号と答え（直す・このままでよい・あとで）、HTML

文はどれも texts の screen.* から作り、Blender の翻訳を通さない（translate=False）。
描くときは何も変えない（読むだけ）。仕事の一覧・版の一覧・報告は、覚えた物（state・background）を読む。
数字の文字は labels・report（プログラム）が作る。
"""

import traceback
from collections.abc import Callable, Sequence
from typing import Any

import bpy

from ..measure.results import Measurement, Signal
from ..model import extract_bpy
from ..model.card import (
    CARD_CURRENT_NAME, EXTRA_ANSWER, HEIGHT, SEMI_STANDARD_NAME, ModelCard, summarize_card,
)
from ..report import sources as report_sources
from ..report.model import DECISIONS, compare_red_count, decision_label, headline, sort_items, value_text
from ..texts import current_language, text
from . import background, labels, operators, preferences, state
from .preferences import TRANSLATION_CONTEXT

# 一覧（部品と材質・聞きたい骨・調べた骨）に1回に出す行の数。多いモデルでも描き直しが重くならないようにする
# （画面の決まり。基準の数字ではない）
MAX_LISTED_ROWS = 40
# 裏の仕事の一覧と、保存した版の一覧に出す行の数（新しい物から）
MAX_JOB_ROWS = 10
MAX_VERSION_ROWS = 10
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


def _listed_parts(parts: Sequence[Any]) -> list[tuple[Any, list[Any]]]:
    """一覧に出す順：(部品, その材質の並び)。

    全部が MAX_LISTED_ROWS に入るときはカードの順。入らないときは、まだ決めていない部品と材質を先にする
    （上から決めると、隠れていた物が出てくる。決めた物を後ろに回さないと、後ろの材質の種類を決められない）。
    """
    if sum(1 + len(part.materials) for part in parts) <= MAX_LISTED_ROWS:
        return [(part, list(part.materials)) for part in parts]
    # sorted は並びを保つので、同じ側の中はカードの順のまま（False＝まだ決めていない物が先）
    ordered = sorted(parts, key=lambda part: part.is_confirmed and not part.unconfirmed_materials)
    return [(part, sorted(part.materials, key=lambda material: material.is_confirmed)) for part in ordered]


def draw_parts(layout: Any, context: Any, card: ModelCard, *, editable: bool) -> None:
    """部品（メッシュ）と、材質が2つ以上ある部品の材質ごとの、種類の一覧"""
    box = layout.box()
    box.label(text=text("screen.parts_title", count=len(card.parts)), icon="MESH_DATA", translate=False)
    total = sum(1 + len(part.materials) for part in card.parts)
    shown = 0
    for part, materials in _listed_parts(card.parts):
        if shown >= MAX_LISTED_ROWS:
            break
        column = box.column(align=True)
        column.label(text=part.object_name, icon="OBJECT_DATA", translate=False)
        _draw_choice(column.row(align=True), part.object_name, "", part, editable)
        shown += 1
        for material in materials:
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
    if "sway_extras" in check.result:
        # 揺れ物の骨の数（道具の版 2 から。無い結果でも描ける）
        column.label(text=text("screen.bones_sway", count=len(check.result["sway_extras"])), translate=False)
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


# MARK: - 確かめと書き出し（A6 版・E1＋E2・A5）


def draw_checks(layout: Any, context: Any) -> None:
    """版の保存と比べ、保存した版の一覧、書き出して確かめる、確かめの動き"""
    row = layout.row(align=True)
    row.operator(operators.AI_TOOLS_OT_save_version.bl_idname, text=text("screen.save_version"), icon="FILE_TICK",
                 translate=False)
    row.operator(operators.AI_TOOLS_OT_compare_versions.bl_idname, text=text("screen.compare_versions"),
                 icon="ARROW_LEFTRIGHT", translate=False)
    saved = state.versions()
    draw_versions(layout, context, saved)

    box = layout.box()
    box.operator(operators.AI_TOOLS_OT_export_and_check.bl_idname, text=text("screen.export_and_check"),
                 icon="EXPORT", translate=False)
    if saved:
        newest = max(saved, key=lambda entry: entry.number)
        wrapped_label(box, context, text("screen.export_input", name=newest.name), icon="FILE_BLEND")
    else:
        wrapped_label(box, context, text("screen.export_needs_version"), icon="INFO")
    message, icon = preferences.koumakan_status(preferences.get_preferences())
    if icon != "CHECKMARK":
        wrapped_label(box, context, message, icon=icon)

    box = layout.box()
    box.operator(operators.AI_TOOLS_OT_run_course.bl_idname, text=text("screen.run_course"), icon="ARMATURE_DATA",
                 translate=False)
    wrapped_label(box, context, text("screen.course_note"), icon="INFO")


def draw_versions(layout: Any, context: Any, saved: Sequence[state.VersionEntry] | None) -> None:
    """保存した版の一覧（新しい版が上）"""
    if saved is None:
        wrapped_label(layout, context, text("screen.versions_unsaved"), icon="INFO")
        return
    if not saved:
        wrapped_label(layout, context, text("screen.versions_none"), icon="INFO")
        return
    box = layout.box()
    box.label(text=text("screen.versions_title", count=len(saved)), icon="FILE_BLEND", translate=False)
    ordered = sorted(saved, key=lambda entry: entry.number, reverse=True)
    column = box.column(align=True)
    for entry in ordered[:MAX_VERSION_ROWS]:
        column.label(text=text("versions.version_entry", number=entry.number, name=entry.name), icon="DOT",
                     translate=False)
    if len(ordered) > MAX_VERSION_ROWS:
        wrapped_label(box, context, text("screen.versions_more", count=len(ordered) - MAX_VERSION_ROWS),
                      icon="THREE_DOTS")


# MARK: - 裏の仕事（A10）

# 仕事のようすごとのアイコン
JOB_ICONS = {"queued": "SORTTIME", "running": "PLAY", "done": "CHECKMARK", "failed": "ERROR",
             "cancelled": "CANCEL"}


def draw_jobs(layout: Any, context: Any) -> None:
    """仕事の一覧（新しい物が上）：ようす・進み具合・理由と、中止・結果を見る・一覧から消す"""
    rows = list(reversed(background.rows()))
    broken = background.unreadable()
    if not rows and not broken:
        wrapped_label(layout, context, text("screen.jobs_none"), icon="INFO")
        return
    for row in rows[:MAX_JOB_ROWS]:
        draw_job(layout, context, row)
    if len(rows) > MAX_JOB_ROWS:
        wrapped_label(layout, context, text("screen.jobs_more", count=len(rows) - MAX_JOB_ROWS), icon="THREE_DOTS")
    if broken:
        wrapped_label(layout, context, text("screen.jobs_unreadable", count=len(broken)), icon="ERROR")


def draw_job(layout: Any, context: Any, row: background.JobRow) -> None:
    job, progress = row.job, row.progress
    box = layout.box()
    status = text("screen.job_cleaning") if row.cleaning else labels.job_status_label(progress.status)
    icon = "TIME" if row.cleaning else JOB_ICONS.get(progress.status, "QUESTION")
    box.label(text=text("screen.job_title", tool=labels.tool_label(job.tool_id), status=status), icon=icon,
              translate=False)
    box.label(text=text("screen.job_created", time=job.created_at), icon="BLANK1", translate=False)
    if not progress.is_finished:
        message = labels.job_text(progress.message_key, progress.message_values, "")
        if progress.total:
            line = text("screen.job_step", step=progress.step, total=progress.total, message=message) if message \
                else text("jobs.step_of", step=progress.step, total=progress.total)
            wrapped_label(box, context, line, icon="BLANK1")
        elif message:
            wrapped_label(box, context, message, icon="BLANK1")
    reason = labels.job_text(progress.error_key, progress.error_values, progress.error)
    if reason:
        column = box.column()
        column.alert = progress.status == "failed"
        wrapped_label(column, context, reason, icon="BLANK1")
    buttons = box.row(align=True)
    if not progress.is_finished:
        props = buttons.operator(operators.AI_TOOLS_OT_cancel_job.bl_idname, text=text("screen.cancel_job"),
                                 icon="CANCEL", translate=False)
        props.job_id = job.job_id
        return
    if row.cleaning:
        return
    if progress.status == "done" and report_sources.makes_report(job.tool_id):
        props = buttons.operator(operators.AI_TOOLS_OT_show_job_result.bl_idname, text=text("screen.show_job_result"),
                                 icon="VIEWZOOM", translate=False)
        props.job_id = job.job_id
    props = buttons.operator(operators.AI_TOOLS_OT_remove_job.bl_idname, text=text("screen.remove_job"),
                             icon="TRASH", translate=False)
    props.job_id = job.job_id


# MARK: - 報告（A11）


def _worst_signal(items: Sequence[Any]) -> Signal:
    counted = [item.counted_signal for item in items if item.counted_signal is not None]
    return max(counted) if counted else Signal.GREEN


def baseline_line(items: Sequence[Any], baseline: int | None) -> tuple[str, Signal]:
    """赤の数を、記録した基準と比べた1行（数はプログラムが入れる）と、その信号"""
    measurement = compare_red_count(items, baseline)
    current = str(measurement.details["current"])
    if baseline is None:
        return text("report.baseline_first", current=current), measurement.signal
    key = "report.baseline_increase" if measurement.signal == Signal.RED else "report.baseline_ok"
    return text(key, current=current, previous=str(baseline)), measurement.signal


def draw_report(layout: Any, context: Any) -> None:
    """いまの報告：いちばん上の一言、前の版の基準との比べ、HTML・基準のボタン、1件ずつの信号と答え"""
    view = state.report_view()
    if view is None:
        wrapped_label(layout, context, text("screen.report_none"), icon="INFO")
        return
    report = view.report
    layout.label(text=text("screen.report_header", kind=report_sources.kind_label(report.kind), time=report.created_at),
                 icon="DOCUMENTS", translate=False)
    worst = _worst_signal(report.items)
    wrapped_label(layout, context, labels.signal_row(worst, headline(report.items)), icon=labels.SIGNAL_ICONS[worst])
    line, signal = baseline_line(report.items, view.baseline_red)
    wrapped_label(layout, context, line, icon=labels.SIGNAL_ICONS[signal])
    if view.store_error is not None:
        column = layout.column()
        column.alert = True
        wrapped_label(column, context, text("screen.decisions_unreadable", error=view.store_error), icon="ERROR")
    buttons = layout.row(align=True)
    buttons.operator(operators.AI_TOOLS_OT_write_report_html.bl_idname, text=text("screen.write_report_html"),
                     icon="FILE_TEXT", translate=False)
    buttons.operator(operators.AI_TOOLS_OT_open_report_html.bl_idname, text=text("screen.open_report_html"),
                     icon="URL", translate=False)
    buttons.operator(operators.AI_TOOLS_OT_record_baseline.bl_idname, text=text("screen.record_baseline"),
                     icon="BOOKMARKS", translate=False)
    if view.html_path is not None:
        wrapped_label(layout, context, text("screen.report_html_at", path=str(view.html_path)), icon="FILE_TEXT")
    items = sort_items(report.items)
    for item in items[:MAX_LISTED_ROWS]:
        draw_report_item(layout, context, item)
    _more_rows(layout, context, len(items) - MAX_LISTED_ROWS)


def draw_report_item(layout: Any, context: Any, item: Any) -> None:
    measurement = Measurement.from_data(item.measurement)
    box = layout.box()
    title = item.titles.get(current_language()) or item.titles["ja"]
    wrapped_label(box, context, labels.signal_row(measurement.signal, title), icon=labels.SIGNAL_ICONS[measurement.signal])
    box.label(text=text("screen.item_value", value=value_text(item.measurement)), icon="BLANK1", translate=False)
    if measurement.signal == Signal.GREEN:
        return
    if measurement.reason:
        wrapped_label(box, context, measurement.reason, icon="BLANK1")
    if item.decision is not None:
        row = box.row(align=True)
        row.label(text=text("screen.decision_current", decision=decision_label(item.decision)), icon="CHECKMARK",
                  translate=False)
        props = row.operator(operators.AI_TOOLS_OT_clear_decision.bl_idname, text="", icon="X", translate=False)
        props.item_id = item.item_id
    buttons = box.row(align=True)
    buttons.operator_context = "EXEC_DEFAULT"
    for decision in DECISIONS:
        props = buttons.operator(operators.AI_TOOLS_OT_decide.bl_idname, text=decision_label(decision),
                                 translate=False, depress=item.decision == decision)
        props.item_id = item.item_id
        props.decision = decision


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


class VIEW3D_PT_ai_tools_checks(_Panel, bpy.types.Panel):
    bl_idname = "VIEW3D_PT_ai_tools_checks"
    bl_parent_id = "VIEW3D_PT_ai_tools"
    bl_label = ""
    bl_category = ""
    LABEL_KEY = "screen.section_checks"

    def draw(self, context: Any) -> None:
        draw_safely(self.layout, context, draw_checks)


class VIEW3D_PT_ai_tools_jobs(_Panel, bpy.types.Panel):
    bl_idname = "VIEW3D_PT_ai_tools_jobs"
    bl_parent_id = "VIEW3D_PT_ai_tools"
    bl_label = ""
    bl_category = ""
    LABEL_KEY = "screen.section_jobs"

    def draw(self, context: Any) -> None:
        draw_safely(self.layout, context, draw_jobs)


class VIEW3D_PT_ai_tools_report(_Panel, bpy.types.Panel):
    bl_idname = "VIEW3D_PT_ai_tools_report"
    bl_parent_id = "VIEW3D_PT_ai_tools"
    bl_label = ""
    bl_category = ""
    LABEL_KEY = "screen.section_report"

    def draw(self, context: Any) -> None:
        draw_safely(self.layout, context, draw_report)


# 親のパネルを先にする（登録は前から、解除は後ろから）
CLASSES = (VIEW3D_PT_ai_tools, VIEW3D_PT_ai_tools_model, VIEW3D_PT_ai_tools_bones, VIEW3D_PT_ai_tools_checks,
           VIEW3D_PT_ai_tools_jobs, VIEW3D_PT_ai_tools_report)

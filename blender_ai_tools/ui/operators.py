"""パネルのボタンが動かす操作（オペレーター）。どれも bl_idname は "ai_tools.*"。

- モデルの形は変えない。開いている .blend のファイルは保存しない（保存するのはユーザー。CLAUDE.md 2章4）。
  変えるのは、.blend の中の登録カードのテキスト（ai_tools_card.json）だけ。
  新しく作るファイルは、版（_v001 など。versions.save）・写し（確かめの動きの入力。仕事のフォルダの中）・
  作業フォルダの中の物（仕事・書き出し exports/・報告 reports/・答えの記録 decisions.json）だけ。
- 道具は、目録（A8）から呼ぶ（CLAUDE.md 2章1）。骨の名前は bones.map_names、版は versions.*。
  重い道具（export.convert_and_check・motion.run_course）は、直接呼ばずに裏の係（A10）に仕事として出す。
- アプリの Model フォルダには書かない。書き出しは、いちばん新しい保存した版を、作業フォルダの exports/ の新しいフォルダへ。
- まちがいは self.report({"ERROR"}, 文) で知らせ、画面に例外を出さない。文は texts の screen.*（いまの言語）。
  道具（card.py など）の日本語の説明は、その文の中に入れて見せる。
- 名前と説明（ツールチップ）は、いまの言語で作る（localize。言語を変えたら作り直す）。

このファイルでは from __future__ import annotations を使わない（bpy.props の注釈を、Blender が読めるようにするため）。
"""

import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import bpy

from .. import registry
from ..catalog import CatalogError
from ..export.convert import check_output_dir, koumakan_paths
from ..jobs import launcher_bpy
from ..jobs.spec import JOB_FILE, new_job_id
from ..model import card_bpy, extract_bpy
from ..model.card import (
    CLOTH_TYPE_IDS, EXTRA_ANSWER, PART_KINDS, ModelCard, set_bone_answer, set_material_kind, set_part_kind,
)
from ..report import html as report_html
from ..report import sources as report_sources
from ..report.decisions import DecisionStore
from ..report.model import DECISIONS, decision_label, headline
from ..texts import current_language, text
from . import background, labels, preferences, state
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


# MARK: - 確かめと書き出し（A6 版・E1＋E2 書き出し・A5 確かめの動き）

# 目録の道具の id（どれも目録 A8 から呼ぶ。重い道具は、裏の係 A10 に仕事として出す）
SAVE_VERSION_TOOL = "versions.save"
COMPARE_TOOL = "versions.compare_with_previous"
LIST_VERSIONS_TOOL = "versions.list"
EXPORT_TOOL = "export.convert_and_check"
COURSE_TOOL = "motion.run_course"
# 「確かめの動き」のボタンの値：アプリのカメラ（まわり全体）から見える物だけを数える（段階1の基準。裏で動かすので重くてよい）
COURSE_PARAMS = {"visibility": True}
# 作業フォルダの中の置き場所
EXPORTS_DIR = "exports"
REPORTS_DIR = "reports"
# 「確かめの動き」の入力（いまの形の写し）を、仕事のフォルダに置くときの名前
COURSE_INPUT_NAME = "input.blend"


def require_saved() -> None:
    """ファイルを保存していなければ Refused（版・仕事・記録は、ファイルの横の作業フォルダに置くため）"""
    if not bpy.data.filepath:
        raise Refused(text("screen.needs_saved_file"))


def work_dir() -> Path:
    """いまのファイルの作業フォルダ（保存していなければ Refused）"""
    require_saved()
    folder = preferences.work_dir_path(preferences.get_preferences())
    if folder is None:
        raise Refused(text("screen.needs_saved_file"))
    return folder


def koumakan_repo() -> Path:
    """設定の紅魔館の場所。書いていない・convert_blend.py が無いときは Refused"""
    repo = preferences.koumakan_repo_path(preferences.get_preferences())
    if repo is None:
        raise Refused(text("screen.export_needs_koumakan"))
    try:
        koumakan_paths(repo)
    except ValueError as error:
        raise Refused(text("screen.export_koumakan_invalid", error=str(error))) from error
    return repo


def newest_version(catalog: Any) -> dict[str, Any]:
    """いちばん新しい保存した版（目録の versions.list）。版が無ければ Refused"""
    listed = catalog.call(LIST_VERSIONS_TOOL, {})["versions"]
    if not listed:
        raise Refused(text("screen.export_needs_version"))
    return max(listed, key=lambda entry: entry["number"])


def checked_params(catalog: Any, tool_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """値を、道具のスキーマで先に確かめる（合わなければ ValueError。仕事を出す前に断る）"""
    errors = catalog.validate_call(tool_id, params)
    if errors:
        raise ValueError("；".join(errors))
    return params


def save_copy(path: Path) -> None:
    """いまの中身の写しを path に保存する（開いているファイルの場所と中身は変えない）"""
    result = bpy.ops.wm.save_as_mainfile(filepath=str(path), copy=True, check_existing=False)
    if "FINISHED" not in result or not path.is_file():
        raise RuntimeError(f"写しを保存できませんでした：{path}")


class AI_TOOLS_OT_save_version(_Localized, bpy.types.Operator):
    """いまの中身を、新しい版（_v001 など）として保存する（目録の versions.save。開いているファイルは上書きしない）"""

    bl_idname = "ai_tools.save_version"
    bl_label = ""
    bl_options = {"REGISTER"}
    LABEL_KEY = "screen.op_save_version"
    TIP_KEY = "screen.save_version_tip"

    def execute(self, context: Any) -> set[str]:
        return run_guarded(self, "screen.save_version_failed", self._save)

    def _save(self) -> set[str]:
        require_saved()
        result = registry.build_catalog().call(SAVE_VERSION_TOOL, {})
        state.refresh_versions()
        self.report({"INFO"}, text("versions.saved", name=result["name"]))
        return {"FINISHED"}


class AI_TOOLS_OT_compare_versions(_Localized, bpy.types.Operator):
    """いまの中身を、前の版と比べて報告にする（目録の versions.compare_with_previous。何も変えない）"""

    bl_idname = "ai_tools.compare_versions"
    bl_label = ""
    bl_options = {"REGISTER"}
    LABEL_KEY = "screen.op_compare_versions"
    TIP_KEY = "screen.compare_versions_tip"

    def execute(self, context: Any) -> set[str]:
        return run_guarded(self, "screen.compare_failed", self._compare)

    def _compare(self) -> set[str]:
        require_saved()
        result = registry.build_catalog().call(COMPARE_TOOL, {})
        report = report_sources.report_from_tool_result(COMPARE_TOOL, result)
        state.show_report(report)
        self.report({"INFO"}, text("screen.report_shown", headline=headline(report.items)))
        return {"FINISHED"}


class AI_TOOLS_OT_export_and_check(_Localized, bpy.types.Operator):
    """いちばん新しい保存した版を、裏の Blender で書き出して確かめる（E1＋E2。作業フォルダの exports/ に書く）"""

    bl_idname = "ai_tools.export_and_check"
    bl_label = ""
    bl_options = {"REGISTER"}
    LABEL_KEY = "screen.op_export_and_check"
    TIP_KEY = "screen.export_and_check_tip"

    def execute(self, context: Any) -> set[str]:
        return run_guarded(self, "screen.export_failed", self._enqueue)

    def _enqueue(self) -> set[str]:
        folder = work_dir()
        repo = koumakan_repo()
        catalog = registry.build_catalog()
        version = newest_version(catalog)
        queue = background.ensure_queue()
        job_id = new_job_id()
        # 新しいフォルダに書く（E1 は空でないフォルダを断る。Mac の .DS_Store とまちがえないように、毎回作り直す）
        output = folder / EXPORTS_DIR / f"{Path(version['path']).stem}_{job_id}"
        if output.exists():
            raise Refused(text("screen.export_output_exists", path=str(output)))
        check_output_dir(output, repo)  # アプリの Model フォルダ・convert_blend.py のフォルダには書かない
        params = checked_params(catalog, EXPORT_TOOL, {"input_blend": version["path"], "output_dir": str(output),
                                                       "koumakan_repo": str(repo), "blender": ""})
        queue.enqueue(launcher_bpy.make_job(EXPORT_TOOL, params, input_blend=version["path"], job_id=job_id))
        background.refresh()
        self.report({"INFO"}, text("screen.job_queued", tool=labels.tool_label(EXPORT_TOOL)))
        return {"FINISHED"}


class AI_TOOLS_OT_run_course(_Localized, bpy.types.Operator):
    """いまの形の写しを、裏の Blender で確かめの動き（A5）にかける（写しは仕事のフォルダに置く）"""

    bl_idname = "ai_tools.run_course"
    bl_label = ""
    bl_options = {"REGISTER"}
    LABEL_KEY = "screen.op_run_course"
    TIP_KEY = "screen.run_course_tip"

    def execute(self, context: Any) -> set[str]:
        return run_guarded(self, "screen.course_failed", self._enqueue)

    def _enqueue(self) -> set[str]:
        work_dir()
        extract_bpy.find_model()  # モデルが無ければ、仕事を出す前に断る（ValueError の日本語の説明）
        params = checked_params(registry.build_catalog(), COURSE_TOOL, dict(COURSE_PARAMS))
        queue = background.ensure_queue()
        job_id = new_job_id()
        copy_path = queue.job_dir(job_id) / COURSE_INPUT_NAME
        queue.enqueue(launcher_bpy.make_job(COURSE_TOOL, params, input_blend=copy_path, job_id=job_id))
        # 入れたすぐあと（タイマーが仕事を始める前）に、いまの形の写しを置く。置けなければ仕事を取り消す
        try:
            save_copy(copy_path)
        except Exception as error:  # noqa: BLE001  写しが置けないときは、どんな理由でも仕事を残さない
            queue.cancel(job_id)
            queue.remove(job_id)
            background.refresh()
            raise Refused(text("screen.course_copy_failed", error=f"{type(error).__name__}: {error}")) from error
        background.refresh()
        self.report({"INFO"}, text("screen.job_queued", tool=labels.tool_label(COURSE_TOOL)))
        return {"FINISHED"}


# MARK: - 裏の仕事（A10）


def _job_property() -> Any:
    return bpy.props.StringProperty(options={"HIDDEN", "SKIP_SAVE"})


def current_queue(job_id: str) -> Any:
    """仕事 job_id がある順番待ち。無ければ Refused"""
    queue = background.queue()
    if queue is None or not any(row.job.job_id == job_id for row in background.rows()):
        background.refresh()
        queue = background.queue()
    if queue is None or not (queue.job_dir(job_id) / JOB_FILE).is_file():
        raise Refused(text("screen.job_missing", job_id=job_id))
    return queue


class AI_TOOLS_OT_cancel_job(_Localized, bpy.types.Operator):
    """仕事を中止する（順番待ちならすぐ。動いていれば、裏の Blender に中止を頼む）"""

    bl_idname = "ai_tools.cancel_job"
    bl_label = ""
    bl_options = {"REGISTER"}
    LABEL_KEY = "screen.op_cancel_job"
    TIP_KEY = "screen.cancel_job_tip"

    @classmethod
    def properties(cls) -> dict[str, Any]:
        return {"job_id": _job_property()}

    def execute(self, context: Any) -> set[str]:
        return run_guarded(self, "screen.job_failed", self._cancel)

    def _cancel(self) -> set[str]:
        current_queue(self.job_id).cancel(self.job_id)
        background.refresh()
        self.report({"INFO"}, text("screen.job_cancel_requested", job_id=self.job_id))
        return {"FINISHED"}


class AI_TOOLS_OT_remove_job(_Localized, bpy.types.Operator):
    """終わった仕事を一覧から消す（仕事のフォルダを消す。書き出したファイルは exports/ に残る）"""

    bl_idname = "ai_tools.remove_job"
    bl_label = ""
    bl_options = {"REGISTER"}
    LABEL_KEY = "screen.op_remove_job"
    TIP_KEY = "screen.remove_job_tip"

    @classmethod
    def properties(cls) -> dict[str, Any]:
        return {"job_id": _job_property()}

    def execute(self, context: Any) -> set[str]:
        return run_guarded(self, "screen.job_failed", self._remove)

    def _remove(self) -> set[str]:
        current_queue(self.job_id).remove(self.job_id)
        background.refresh()
        self.report({"INFO"}, text("screen.job_removed", job_id=self.job_id))
        return {"FINISHED"}


class AI_TOOLS_OT_show_job_result(_Localized, bpy.types.Operator):
    """終わった仕事の結果を、報告に出す"""

    bl_idname = "ai_tools.show_job_result"
    bl_label = ""
    bl_options = {"REGISTER"}
    LABEL_KEY = "screen.op_show_job_result"
    TIP_KEY = "screen.show_job_result_tip"

    @classmethod
    def properties(cls) -> dict[str, Any]:
        return {"job_id": _job_property()}

    def execute(self, context: Any) -> set[str]:
        return run_guarded(self, "screen.job_failed", self._show)

    def _show(self) -> set[str]:
        current_queue(self.job_id)
        report = background.load_report(self.job_id)
        state.show_report(report)
        self.report({"INFO"}, text("screen.report_shown", headline=headline(report.items)))
        return {"FINISHED"}


# MARK: - 報告（A11）


def current_report() -> Any:
    """いまの報告（答えを写した物）。無ければ Refused"""
    view = state.report_view()
    if view is None:
        raise Refused(text("screen.no_report"))
    return view.report


def decision_store() -> DecisionStore:
    """答えの記録（作業フォルダの decisions.json）。保存していないときは Refused、読めないときは ValueError"""
    work_dir()
    store = state.load_store()
    if store is None:
        raise Refused(text("screen.needs_saved_file"))
    return store


def report_version(report: Any) -> str | None:
    """報告のもとの版の名前（書き出しのときだけ分かる）"""
    version = report.source.get("version")
    return version if isinstance(version, str) else None


def _item_title(report: Any, item_id: str) -> str:
    try:
        item = report.item(item_id)
    except KeyError:
        raise Refused(text("screen.item_missing", item=item_id)) from None
    return item.titles.get(current_language()) or item.titles["ja"]


class AI_TOOLS_OT_decide(_Localized, bpy.types.Operator):
    """報告の1件に、直す・このままでよい・あとで を答える（作業フォルダの decisions.json に記録する）"""

    bl_idname = "ai_tools.decide"
    bl_label = ""
    bl_options = {"REGISTER"}
    LABEL_KEY = "screen.op_decide"
    TIP_KEY = "screen.decide_tip"

    @classmethod
    def properties(cls) -> dict[str, Any]:
        return {
            "item_id": _name_property(),
            "decision": bpy.props.EnumProperty(name=text("screen.prop_decision"), items=labels.decision_items,
                                               translation_context=TRANSLATION_CONTEXT),
        }

    def execute(self, context: Any) -> set[str]:
        return run_guarded(self, "screen.decision_failed", self._decide)

    def _decide(self) -> set[str]:
        report = current_report()
        title = _item_title(report, self.item_id)
        if self.decision not in DECISIONS:
            raise ValueError(f"答えは {DECISIONS} のどれかにしてください（いまは {self.decision!r}）")
        store = decision_store()
        store.decide(self.item_id, self.decision, version=report_version(report))
        store.save()
        state.apply_decisions(store)
        self.report({"INFO"}, text("screen.decided", title=title, decision=decision_label(self.decision)))
        return {"FINISHED"}


class AI_TOOLS_OT_clear_decision(_Localized, bpy.types.Operator):
    """報告の1件の答えを消す（まだ決めていない、に戻す）"""

    bl_idname = "ai_tools.clear_decision"
    bl_label = ""
    bl_options = {"REGISTER"}
    LABEL_KEY = "screen.op_clear_decision"
    TIP_KEY = "screen.clear_decision_tip"

    @classmethod
    def properties(cls) -> dict[str, Any]:
        return {"item_id": _name_property()}

    def execute(self, context: Any) -> set[str]:
        return run_guarded(self, "screen.decision_failed", self._clear)

    def _clear(self) -> set[str]:
        title = _item_title(current_report(), self.item_id)
        store = decision_store()
        store.clear(self.item_id)
        store.save()
        state.apply_decisions(store)
        self.report({"INFO"}, text("screen.decision_cleared", title=title))
        return {"FINISHED"}


def write_current_html() -> Path:
    """いまの報告（答えを写した物）を、作業フォルダの reports/<報告の id>.html に書く"""
    folder = work_dir()
    report = current_report()
    path = report_html.write_html(report, folder / REPORTS_DIR / f"{report.report_id}.html")
    state.set_report_html(path)
    return path


def open_path(path: Path) -> None:
    """ファイルを、ふだん使うアプリ（HTML ならブラウザ）で開く（テストでは差しかえる）"""
    bpy.ops.wm.path_open(filepath=str(path))


class AI_TOOLS_OT_write_report_html(_Localized, bpy.types.Operator):
    """いまの報告を HTML のファイルにする（作業フォルダの reports/）"""

    bl_idname = "ai_tools.write_report_html"
    bl_label = ""
    bl_options = {"REGISTER"}
    LABEL_KEY = "screen.op_write_report_html"
    TIP_KEY = "screen.write_report_html_tip"

    def execute(self, context: Any) -> set[str]:
        return run_guarded(self, "screen.report_html_failed", self._write)

    def _write(self) -> set[str]:
        path = write_current_html()
        self.report({"INFO"}, text("screen.report_html_written", path=str(path)))
        return {"FINISHED"}


class AI_TOOLS_OT_open_report_html(_Localized, bpy.types.Operator):
    """いまの報告を HTML に書き直して、ブラウザで開く"""

    bl_idname = "ai_tools.open_report_html"
    bl_label = ""
    bl_options = {"REGISTER"}
    LABEL_KEY = "screen.op_open_report_html"
    TIP_KEY = "screen.open_report_html_tip"

    def execute(self, context: Any) -> set[str]:
        return run_guarded(self, "screen.report_html_failed", self._open)

    def _open(self) -> set[str]:
        path = write_current_html()
        open_path(path)
        self.report({"INFO"}, text("screen.report_html_written", path=str(path)))
        return {"FINISHED"}


class AI_TOOLS_OT_record_baseline(_Localized, bpy.types.Operator):
    """いまの報告の赤の数（「このままでよい」は除く）を、次の版と比べる基準として記録する"""

    bl_idname = "ai_tools.record_baseline"
    bl_label = ""
    bl_options = {"REGISTER"}
    LABEL_KEY = "screen.op_record_baseline"
    TIP_KEY = "screen.record_baseline_tip"

    def execute(self, context: Any) -> set[str]:
        return run_guarded(self, "screen.decision_failed", self._record)

    def _record(self) -> set[str]:
        report = current_report()
        store = decision_store()
        red = store.record_baseline(report.kind, report.items, version=report_version(report))
        store.save()
        state.apply_decisions(store)
        self.report({"INFO"}, text("screen.baseline_recorded", count=red))
        return {"FINISHED"}


CLASSES = (
    AI_TOOLS_OT_register_card,
    AI_TOOLS_OT_check_card,
    AI_TOOLS_OT_set_part_kind,
    AI_TOOLS_OT_set_cloth_type,
    AI_TOOLS_OT_check_bones,
    AI_TOOLS_OT_answer_bone,
    AI_TOOLS_OT_clear_bone_answer,
    AI_TOOLS_OT_save_version,
    AI_TOOLS_OT_compare_versions,
    AI_TOOLS_OT_export_and_check,
    AI_TOOLS_OT_run_course,
    AI_TOOLS_OT_cancel_job,
    AI_TOOLS_OT_remove_job,
    AI_TOOLS_OT_show_job_result,
    AI_TOOLS_OT_decide,
    AI_TOOLS_OT_clear_decision,
    AI_TOOLS_OT_write_report_html,
    AI_TOOLS_OT_open_report_html,
    AI_TOOLS_OT_record_baseline,
)

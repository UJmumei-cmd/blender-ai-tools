"""画面（ui/）を Blender の中で確かめる：登録と解除・言語・設定・ボタンの操作・パネルの描き方。

テスト用のモデル（tests/fixtures/mmd_model.py）を空のシーンに作ってから調べる。
パネルは、バックグラウンドの Blender には画面が無いので、呼ばれた物を書き留める「にせの layout」で描く。
ファイルは一時フォルダの中だけに作る。
"""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import bpy
import numpy as np

import blender_ai_tools
from blender_ai_tools import registry, texts, ui
from blender_ai_tools.catalog import Catalog
from blender_ai_tools.jobs import launcher_bpy
from blender_ai_tools.jobs import progress as jobs_progress
from blender_ai_tools.jobs import worker as jobs_worker
from blender_ai_tools.jobs.queue import JobQueue, WorkerProcess
from blender_ai_tools.measure.results import Measurement, Signal
from blender_ai_tools.motion.course import Event
from blender_ai_tools.report import sources as report_sources
from blender_ai_tools.model import card_bpy
from blender_ai_tools.model.card import HEIGHT, card_text
from blender_ai_tools.texts import text
from blender_ai_tools.ui import background, labels, operators, panel, preferences, state
from tests.fixtures import mmd_model

ICONS = frozenset(bpy.types.UILayout.bl_rna.functions["label"].parameters["icon"].enum_items.keys())


# MARK: - にせの layout


class FakeProps:
    """layout.operator が返す、操作の値を入れる物（入れた値を values に残す）"""

    def __init__(self) -> None:
        object.__setattr__(self, "values", {})

    def __setattr__(self, name, value) -> None:
        self.values[name] = value

    def __getattr__(self, name):
        try:
            return self.values[name]
        except KeyError:
            raise AttributeError(name) from None


class FakeLayout:
    """Blender の UILayout の代わり。呼ばれた物を log に書く（効いている enabled・alert・operator_context も）"""

    def __init__(self, log: list, parent: "FakeLayout | None" = None) -> None:
        self.log = log
        self.parent = parent
        self.enabled = True
        self.active = True
        self.alert = False
        self.operator_context = None
        self.scale_y = 1.0
        self.use_property_split = False

    def _chain(self):
        node = self
        while node is not None:
            yield node
            node = node.parent

    def _state(self) -> dict:
        context = next((node.operator_context for node in self._chain() if node.operator_context), "INVOKE_DEFAULT")
        return {"enabled": all(node.enabled for node in self._chain()),
                "alert": any(node.alert for node in self._chain()), "context": context}

    def _child(self, *args, **kwargs) -> "FakeLayout":
        return FakeLayout(self.log, self)

    row = column = box = split = _child

    def label(self, *, text="", icon="NONE", translate=True, **kwargs) -> None:
        self.log.append({"kind": "label", "text": text, "icon": icon, "translate": translate, **self._state()})

    def operator(self, operator, *, text="", icon="NONE", translate=True, **kwargs) -> FakeProps:
        props = FakeProps()
        self.log.append({"kind": "operator", "idname": operator, "text": text, "icon": icon, "translate": translate,
                         "props": props, **self._state()})
        return props

    def operator_menu_enum(self, operator, property, *, text="", icon="NONE", translate=True, **kwargs) -> FakeProps:
        props = FakeProps()
        self.log.append({"kind": "menu_enum", "idname": operator, "property": property, "text": text, "icon": icon,
                         "translate": translate, "props": props, **self._state()})
        return props

    def prop(self, data, property, *, text="", icon="NONE", translate=True, **kwargs) -> None:
        self.log.append({"kind": "prop", "property": property, "text": text, "icon": icon, "translate": translate,
                         **self._state()})

    def separator(self, **kwargs) -> None:
        pass


def wide_context():
    """とても幅の広いパネルの context（文を折り返さないので、1つの文が1つのラベルになる）"""
    return SimpleNamespace(region=SimpleNamespace(width=100000), preferences=bpy.context.preferences)


def draw(panel_class, context=None) -> list:
    """パネルの draw を、にせの layout で動かし、呼ばれた物を返す（context を渡さなければ、幅の広いパネル）"""
    log: list = []
    panel_class.draw(SimpleNamespace(layout=FakeLayout(log)), wide_context() if context is None else context)
    return log


def label_texts(log: list) -> list[str]:
    return [call["text"] for call in log if call["kind"] == "label"]


def calls_of(log: list, idname: str, kind: str = "operator") -> list[dict]:
    return [call for call in log if call["kind"] == kind and call["idname"] == idname]


# MARK: - もと


class UiCase(unittest.TestCase):
    """空のシーンにモデルを作り、拡張機能を登録して、言語を日本語にする"""

    with_model = True

    def setUp(self):
        bpy.ops.wm.read_homefile(use_empty=True)
        if self.with_model:
            self.info = mmd_model.build_model()
            self.armature = bpy.data.objects[mmd_model.ARMATURE_NAME]
        self.language_before = texts.current_language()
        blender_ai_tools.register()
        self.addCleanup(self._restore)
        ui.apply_language("ja")

    def _restore(self):
        blender_ai_tools.unregister()
        texts.set_language(self.language_before)

    def card(self):
        return card_bpy.load_card()

    def card_json(self):
        holder = bpy.data.texts.get(card_bpy.TEXT_NAME)
        return None if holder is None else holder.as_string()

    def replace_card_json(self, content: str):
        """カードのテキストの中身を置きかえる（読めない物・古い形の物を作るため）"""
        holder = bpy.data.texts[card_bpy.TEXT_NAME]
        holder.clear()
        holder.write(content)

    def make_card_legacy(self):
        """置いてあるカード（版 2）を、版 1（世界の座標。読めるだけで変えられない）の形に書きかえる"""
        data = json.loads(self.card_json())
        for key in ("placement", "skeleton", "bone_answers"):
            data.pop(key)
        data["schema_version"] = 1
        for part in data["parts"]:
            part.pop("materials")
        measurements = data["measurements"]
        measurements["space"] = "world"
        measurements["mesh_z_range"] = measurements.pop("mesh_up_range")
        measurements.pop("material_up_range")
        measurements.pop("excluded_materials")
        self.replace_card_json(json.dumps(data, ensure_ascii=False))
        self.assertTrue(self.card().is_legacy)

    def edit_armature(self, change):
        """エディットモードで骨を変えて、オブジェクトモードに戻す"""
        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            change(self.armature.data.edit_bones)
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.context.view_layer.update()

    def add_unknown_bones(self):
        """辞書に無い骨「謎の骨」と、1つに決まらない骨 leftThumbProximal（左親指０ の名前を変えた物）を作る"""
        def change(bones):
            bones["左親指０"].name = "leftThumbProximal"
            bone = bones.new("謎の骨")
            bone.head = (0.0, 0.0, 1.50)
            bone.tail = (0.0, 0.0, 1.55)
            bone.parent = bones["頭"]
        self.edit_armature(change)

    def move_skirt_vertex(self):
        """スカートの頂点を1つ動かす（Body はシェイプキーの Basis の形で測るので、Body の頂点を動かしても形は変わらない。スカートを使う）"""
        skirt = bpy.data.objects[mmd_model.SKIRT_NAME]
        co = skirt.data.vertices[0].co
        skirt.data.vertices[0].co = (co.x, co.y, co.z + 0.01)
        skirt.data.update()
        bpy.context.view_layer.update()

    def add_skirt_materials(self):
        """スカートに材質を2つ付け、面を半分ずつにする（材質ごとの種類を持つ部品にする）"""
        skirt = bpy.data.objects[mmd_model.SKIRT_NAME]
        for name in ("SkirtCloth", "SkirtLace"):
            skirt.data.materials.append(bpy.data.materials.new(name))
        for polygon in skirt.data.polygons:
            polygon.material_index = polygon.index % 2
        bpy.context.view_layer.update()


# MARK: - 登録と解除・言語・設定


class RegistrationTest(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_homefile(use_empty=True)
        self.language_before = texts.current_language()
        self.addCleanup(texts.set_language, self.language_before)

    def assert_registered(self, registered: bool):
        for cls in ui.CLASSES:
            with self.subTest(cls=cls.__name__, registered=registered):
                if registered:
                    self.assertIs(getattr(bpy.types, cls.__name__, None), cls)
                else:
                    self.assertIsNone(getattr(bpy.types, cls.__name__, None))
        for cls in operators.CLASSES:
            category, name = cls.bl_idname.split(".")
            operator = getattr(getattr(bpy.ops, category), name)
            if registered:
                operator.get_rna_type()
            else:
                with self.assertRaises(KeyError):
                    operator.get_rna_type()
        self.assertEqual(preferences.AIToolsPreferences.is_registered, registered)
        self.assertEqual(preferences.preferences_registered(), registered)
        self.assertEqual(bpy.app.handlers.load_post.count(state.on_load_post), 1 if registered else 0)
        self.assertEqual(bpy.app.handlers.load_post.count(background.on_load_post), 1 if registered else 0)
        self.assertEqual(bpy.app.handlers.save_post.count(background.on_save_post), 1 if registered else 0)
        if not registered:
            self.assertIsNone(background.queue())

    def test_register_and_unregister_twice(self):
        self.assert_registered(False)
        for _ in range(2):
            blender_ai_tools.register()
            try:
                self.assert_registered(True)
            finally:
                blender_ai_tools.unregister()
            self.assert_registered(False)

    def test_register_again_without_unregister(self):
        """登録を2回続けても、二重にならない（前の登録を外してから登録する）"""
        blender_ai_tools.register()
        try:
            blender_ai_tools.register()
            self.assert_registered(True)
        finally:
            blender_ai_tools.unregister()
        self.assert_registered(False)

    def test_classes(self):
        self.assertEqual({cls.bl_idname.split(".")[0] for cls in operators.CLASSES}, {"ai_tools"})
        self.assertEqual(panel.CLASSES[0].bl_idname, "VIEW3D_PT_ai_tools")
        for cls in panel.CLASSES:
            self.assertEqual((cls.bl_space_type, cls.bl_region_type), ("VIEW_3D", "UI"))
        for cls in panel.CLASSES[1:]:
            self.assertEqual(cls.bl_parent_id, "VIEW3D_PT_ai_tools")
        self.assertEqual(preferences.ADDON_ID, blender_ai_tools.__name__)
        self.assertEqual(preferences.AIToolsPreferences.bl_idname, blender_ai_tools.__name__)

    def test_register_sets_language_and_unregister_restores_it(self):
        """初めの言語は日本語（CLAUDE.md：画面は日本語）。Blender が英語でも、設定を変えていなければ日本語になる"""
        self.assertEqual(preferences.LANGUAGE_DEFAULT, "ja")
        self.assertEqual(preferences.DEFAULTS.language, "ja")
        self.assertEqual(preferences.LANGUAGE_CHOICES, ("auto", "ja", "en"))
        with mock.patch.object(preferences, "blender_locale", return_value="en_US"):
            for before in ("ja", "en"):
                with self.subTest(before=before):
                    texts.set_language(before)
                    blender_ai_tools.register()
                    try:
                        self.assertEqual(texts.current_language(), "ja")
                        self.assertEqual(bpy.types.VIEW3D_PT_ai_tools.bl_category, "AI の道具")
                        self.assertEqual(preferences.AIToolsPreferences.bl_rna.properties["language"].default, "ja")
                    finally:
                        blender_ai_tools.unregister()
                    self.assertEqual(texts.current_language(), before)

    def test_register_failure_is_undone(self):
        """登録の途中で失敗したら、登録した物を全部外し、言語も戻してから、エラーをそのまま出す"""
        texts.set_language("en")
        with mock.patch.object(panel.VIEW3D_PT_ai_tools_report, "localize", side_effect=RuntimeError("こわれた")):
            with self.assertRaisesRegex(RuntimeError, "こわれた"):
                blender_ai_tools.register()
        self.assert_registered(False)
        self.assertFalse(ui.is_registered())
        self.assertEqual(texts.current_language(), "en")
        # 直したあとは、ふつうに登録できる
        blender_ai_tools.register()
        try:
            self.assert_registered(True)
        finally:
            blender_ai_tools.unregister()

    def test_unregister_without_register_is_harmless(self):
        blender_ai_tools.unregister()
        blender_ai_tools.unregister()
        self.assert_registered(False)

    def test_language_switch_renames_panels_and_operators(self):
        blender_ai_tools.register()
        self.addCleanup(blender_ai_tools.unregister)
        self.assertEqual(ui.apply_language("en"), "en")
        self.assertEqual(texts.current_language(), "en")
        self.assertEqual(bpy.types.VIEW3D_PT_ai_tools.bl_category, "AI Tools")
        self.assertEqual(bpy.types.VIEW3D_PT_ai_tools_model.bl_label, "Model")
        self.assertEqual(bpy.types.VIEW3D_PT_ai_tools_bones.bl_category, "AI Tools")
        self.assertEqual(bpy.types.AI_TOOLS_OT_register_card.bl_label, "Register the model")
        self.assertEqual(ui.apply_language("ja"), "ja")
        self.assertEqual(bpy.types.VIEW3D_PT_ai_tools.bl_category, "AI の道具")
        self.assertEqual(bpy.types.VIEW3D_PT_ai_tools.bl_label, "AI の道具")
        self.assertEqual(bpy.types.VIEW3D_PT_ai_tools_model.bl_label, "モデル")
        self.assertEqual(bpy.types.VIEW3D_PT_ai_tools_checks.bl_label, "確かめと書き出し")
        self.assertEqual(bpy.types.VIEW3D_PT_ai_tools_jobs.bl_label, "裏の仕事")
        self.assertEqual(bpy.types.VIEW3D_PT_ai_tools_report.bl_label, "報告")
        self.assertEqual(bpy.types.AI_TOOLS_OT_register_card.bl_label, "モデルを登録する")
        # 作り直しても、どのクラスも1回ずつ登録されている
        self.assertEqual(len(ui._registered), len(ui.CLASSES))
        for cls in ui.CLASSES:
            self.assertIs(getattr(bpy.types, cls.__name__), cls)
        # 一覧の名前も、いまの言語
        self.assertEqual(dict((k, v) for k, v, _ in labels.kind_items())["cloth"], "服")
        ui.apply_language("en")
        self.assertEqual(dict((k, v) for k, v, _ in labels.kind_items())["cloth"], "Clothes")
        self.assertEqual(operators.AI_TOOLS_OT_check_card.description(bpy.context, None),
                         "Check whether the registration card still matches the current model (nothing is changed)")

    def test_resolve_language(self):
        cases = [("auto", "ja_JP", "ja"), ("auto", "ja", "ja"), ("auto", "en_US", "en"), ("auto", "", "en"),
                 ("auto", "zh_HANS", "en"), ("ja", "en_US", "ja"), ("en", "ja_JP", "en")]
        for choice, locale, expected in cases:
            with self.subTest(choice=choice, locale=locale):
                self.assertEqual(preferences.resolve_language(choice, locale), expected)
        with self.assertRaises(ValueError):
            preferences.resolve_language("fr", "fr_FR")

    def test_preferences_when_not_enabled(self):
        """拡張機能を設定から有効にしていないとき（テストで直接読み込んだとき）も、初めの値で動く"""
        blender_ai_tools.register()
        self.addCleanup(blender_ai_tools.unregister)
        values = preferences.get_preferences()
        self.assertIsInstance(values, preferences.PreferenceValues)
        self.assertEqual((values.koumakan_repo, values.blender_path, values.work_dir, values.language),
                         ("", "", "", "ja"))
        self.assertIsNone(preferences.koumakan_repo_path(values))
        self.assertEqual(preferences.blender_executable(values), bpy.app.binary_path)
        self.assertEqual(preferences.work_dir_path(values, "/models/miku/model.blend"),
                         Path("/models/miku/model_ai_tools"))
        self.assertIsNone(preferences.work_dir_path(values, ""))
        self.assertIsNone(preferences.work_dir_path(values))  # 保存していないファイル
        chosen = preferences.PreferenceValues(koumakan_repo="/repo/koumakan", blender_path="/apps/blender",
                                              work_dir="/work/here")
        self.assertEqual(preferences.koumakan_repo_path(chosen), Path("/repo/koumakan"))
        self.assertEqual(preferences.blender_executable(chosen), "/apps/blender")
        self.assertEqual(preferences.work_dir_path(chosen, "/models/model.blend"), Path("/work/here"))

    def test_blender_executable_accepts_a_mac_app_bundle(self):
        """Mac で Blender.app（フォルダ）を選んだときは、その中の動かすファイルにする。ほかはそのまま"""
        with tempfile.TemporaryDirectory() as folder:
            app = Path(folder, "Blender.app")
            binary = app / "Contents" / "MacOS" / "Blender"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"")
            self.assertEqual(preferences.blender_executable(preferences.PreferenceValues(blender_path=str(app))),
                             str(binary))
            self.assertEqual(preferences.blender_executable(preferences.PreferenceValues(blender_path=str(binary))),
                             str(binary))
            # 中身の無い .app はそのまま（あとで「見つからない」と言われる）
            empty = Path(folder, "Other.app")
            empty.mkdir()
            self.assertEqual(preferences.blender_executable(preferences.PreferenceValues(blender_path=str(empty))),
                             str(empty))

    def test_preferences_when_enabled(self):
        """設定から有効にしたとき：言語を変えるとすぐに文が変わり、登録し直しても設定の値は残る"""
        addon = bpy.context.preferences.addons.new()
        addon.module = preferences.ADDON_ID
        self.addCleanup(bpy.context.preferences.addons.remove, addon)
        blender_ai_tools.register()
        self.addCleanup(blender_ai_tools.unregister)
        prefs = preferences.get_preferences()
        self.assertIsInstance(prefs, preferences.AIToolsPreferences)
        self.assertEqual(prefs.language, "ja")
        self.assertEqual(texts.current_language(), "ja")
        self.assertEqual(bpy.types.VIEW3D_PT_ai_tools.bl_category, "AI の道具")
        with mock.patch.object(preferences, "blender_locale", return_value="en_US"):
            prefs.language = "auto"  # Blender に合わせる：Blender が英語なら英語
            self.assertEqual(texts.current_language(), "en")
            self.assertEqual(bpy.types.VIEW3D_PT_ai_tools.bl_category, "AI Tools")
        with mock.patch.object(preferences, "blender_locale", return_value="ja_JP"):
            prefs.language = "ja"
            self.assertEqual(texts.current_language(), "ja")
            self.assertEqual(bpy.types.VIEW3D_PT_ai_tools.bl_category, "AI の道具")
            prefs.language = "en"  # 選んだ言語は、Blender の言語より強い
        self.assertEqual(texts.current_language(), "en")
        self.assertEqual(bpy.types.VIEW3D_PT_ai_tools.bl_category, "AI Tools")
        prefs.koumakan_repo = "/repo/koumakan"
        blender_ai_tools.register()  # Blender を開き直したときと同じ：保存した設定の値を読んで、言語を決める
        prefs = preferences.get_preferences()
        self.assertEqual((prefs.language, prefs.koumakan_repo), ("en", "/repo/koumakan"))
        self.assertEqual(texts.current_language(), "en")
        properties = preferences.AIToolsPreferences.bl_rna.properties
        self.assertEqual(properties["koumakan_repo"].name, "Koumakan repository")
        self.assertEqual(preferences.preferences_language(), "en")

    def test_koumakan_status(self):
        with tempfile.TemporaryDirectory() as folder:
            values = preferences.PreferenceValues(koumakan_repo=folder)
            message, icon = preferences.koumakan_status(values)
            self.assertEqual(icon, "ERROR")
            self.assertTrue(message.startswith("見つかりません："), message)
            converter = Path(folder, "tools", "model-converter")
            converter.mkdir(parents=True)
            for name in ("convert_blend.py", "convert_pmx.py"):
                (converter / name).write_text("", encoding="utf-8")
            self.assertEqual(preferences.koumakan_status(values), ("convert_blend.py が見つかりました", "CHECKMARK"))
        self.assertEqual(preferences.koumakan_status(preferences.DEFAULTS),
                         ("まだ書いていません（アプリへの書き出しに要ります）", "INFO"))

    def test_preferences_draw(self):
        log: list = []
        blender_ai_tools.register()
        self.addCleanup(blender_ai_tools.unregister)
        ui.apply_language("ja")
        preferences.draw_preferences(preferences.DEFAULTS, FakeLayout(log), bpy.context)
        self.assertEqual([call["property"] for call in log if call["kind"] == "prop"],
                         ["language", "koumakan_repo", "blender_path", "work_dir"])
        self.assertIn(f"空なら、いま動いている Blender を使います：{bpy.app.binary_path}", label_texts(log))
        self.assertIn("空なら、.blend の横の「（ファイルの名前）_ai_tools」を使います", label_texts(log))
        self.assertIn("いまのファイルは、まだ保存していません（保存すると決まります）", label_texts(log))
        self.assertTrue(all(call["translate"] is False for call in log))


# MARK: - モデル（A1 登録カード）の操作


class CardOperatorTest(UiCase):
    def test_register_card_creates_text(self):
        self.assertIsNone(bpy.data.texts.get(card_bpy.TEXT_NAME))
        self.assertEqual(bpy.ops.ai_tools.register_card(), {"FINISHED"})
        holder = bpy.data.texts.get(card_bpy.TEXT_NAME)
        self.assertIsNotNone(holder)
        self.assertEqual(holder.name, "ai_tools_card.json")
        card = self.card()
        self.assertEqual(card.armature_name, mmd_model.ARMATURE_NAME)
        self.assertEqual([part.object_name for part in card.parts], [mmd_model.BODY_NAME, mmd_model.SKIRT_NAME])
        self.assertEqual(len(card.skeleton["names"]), self.info["bone_count"])
        status = state.card_status(state.card_identity(card))
        self.assertEqual((status.checked, status.reasons, status.notes), (False, (), ()))

    def test_register_without_model_reports_error(self):
        bpy.ops.wm.read_homefile(use_empty=True)
        with self.assertRaisesRegex(RuntimeError, "登録できませんでした：アーマチュアがありません"):
            bpy.ops.ai_tools.register_card()
        self.assertIsNone(bpy.data.texts.get(card_bpy.TEXT_NAME))

    def test_register_keeps_user_choices(self):
        bpy.ops.ai_tools.register_card()
        bpy.ops.ai_tools.set_cloth_type(object_name=mmd_model.SKIRT_NAME, cloth_type="skirt")
        bpy.ops.ai_tools.set_part_kind(object_name=mmd_model.BODY_NAME, kind="body")
        bpy.ops.ai_tools.register_card()
        card = self.card()
        skirt = card.part(mmd_model.SKIRT_NAME)
        self.assertEqual((skirt.kind, skirt.kind_source, skirt.cloth_type), ("cloth", "user", "skirt"))
        self.assertEqual(card.part(mmd_model.BODY_NAME).kind_source, "user")

    def test_register_keeps_unreadable_card(self):
        """読めない前のカードは、消さずに別のテキストに残し、そう知らせる"""
        broken = bpy.data.texts.new(card_bpy.TEXT_NAME)
        broken.write("{not json")
        self.assertEqual(bpy.ops.ai_tools.register_card(), {"FINISHED"})
        backup = bpy.data.texts.get(operators.UNREADABLE_CARD_BACKUP)
        self.assertIsNotNone(backup)
        self.assertEqual(backup.as_string(), "{not json")
        self.assertTrue(backup.use_fake_user)
        card = self.card()
        self.assertEqual(card.armature_name, mmd_model.ARMATURE_NAME)
        status = state.card_status(state.card_identity(card))
        self.assertEqual(status.notes,
                         (text("screen.previous_card_unreadable", backup=operators.UNREADABLE_CARD_BACKUP),))

    def test_check_finds_no_reasons_right_after_registering(self):
        bpy.ops.ai_tools.register_card()
        self.assertEqual(bpy.ops.ai_tools.check_card(), {"FINISHED"})
        status = state.card_status(state.card_identity(self.card()))
        self.assertEqual((status.checked, status.reasons, status.notes), (True, (), ()))

    def test_check_finds_a_changed_shape(self):
        bpy.ops.ai_tools.register_card()
        self.move_skirt_vertex()
        self.assertEqual(bpy.ops.ai_tools.check_card(), {"FINISHED"})
        status = state.card_status(state.card_identity(self.card()))
        self.assertEqual(status.reasons, (card_text("card.shape_changed", name=mmd_model.SKIRT_NAME),))

    def test_check_without_card(self):
        self.assertEqual(bpy.ops.ai_tools.check_card(), {"CANCELLED"})
        status = state.card_status(None)
        self.assertEqual(status.reasons, (card_text("card.no_card"),))

    def test_set_part_kind(self):
        bpy.ops.ai_tools.register_card()
        skirt_name = mmd_model.SKIRT_NAME
        self.assertEqual(bpy.ops.ai_tools.set_part_kind(object_name=skirt_name, kind="cloth"), {"FINISHED"})
        part = self.card().part(skirt_name)
        self.assertEqual((part.kind, part.kind_source, part.cloth_type, part.is_confirmed),
                         ("cloth", "user", None, False))
        bpy.ops.ai_tools.set_cloth_type(object_name=skirt_name, cloth_type="skirt")
        part = self.card().part(skirt_name)
        self.assertEqual((part.kind, part.cloth_type, part.is_confirmed), ("cloth", "skirt", True))
        # 服のまま選び直しても、服の種類は残る
        bpy.ops.ai_tools.set_part_kind(object_name=skirt_name, kind="cloth")
        self.assertEqual(self.card().part(skirt_name).cloth_type, "skirt")
        # 服でなくすると、服の種類は外れる
        bpy.ops.ai_tools.set_part_kind(object_name=skirt_name, kind="accessory")
        part = self.card().part(skirt_name)
        self.assertEqual((part.kind, part.cloth_type, part.is_confirmed), ("accessory", None, True))

    def test_invalid_choices_are_rejected(self):
        bpy.ops.ai_tools.register_card()
        before = self.card_json()
        with self.assertRaises(TypeError):
            bpy.ops.ai_tools.set_part_kind(object_name=mmd_model.BODY_NAME, kind="bogus")
        with self.assertRaises(TypeError):
            bpy.ops.ai_tools.set_cloth_type(object_name=mmd_model.SKIRT_NAME, cloth_type="cape")
        with self.assertRaisesRegex(RuntimeError, "登録カードを変えられませんでした：部品 'Nope' はカードにありません"):
            bpy.ops.ai_tools.set_part_kind(object_name="Nope", kind="body")
        # 服でない部品には、服の種類を選べない
        with self.assertRaisesRegex(RuntimeError, "服の種類は、種類が「服」の部品と材質だけに選べます（Body）"):
            bpy.ops.ai_tools.set_cloth_type(object_name=mmd_model.BODY_NAME, cloth_type="skirt")
        self.assertEqual(self.card_json(), before)
        # 中の確かめ（PART_KINDS・CLOTH_TYPE_IDS に無い物）
        card = self.card()
        with self.assertRaises(ValueError):
            operators.with_kind(card, mmd_model.BODY_NAME, "", "bogus")
        with self.assertRaises(ValueError):
            operators.with_cloth_type(card, mmd_model.SKIRT_NAME, "", "cape")

    def test_set_material_kind(self):
        self.add_skirt_materials()
        bpy.ops.ai_tools.register_card()
        skirt_name = mmd_model.SKIRT_NAME
        self.assertEqual([m.name for m in self.card().part(skirt_name).materials], ["SkirtCloth", "SkirtLace"])
        bpy.ops.ai_tools.set_part_kind(object_name=skirt_name, material_name="SkirtLace", kind="accessory")
        bpy.ops.ai_tools.set_part_kind(object_name=skirt_name, material_name="SkirtCloth", kind="cloth")
        bpy.ops.ai_tools.set_cloth_type(object_name=skirt_name, material_name="SkirtCloth", cloth_type="skirt")
        part = self.card().part(skirt_name)
        lace, cloth = part.material("SkirtLace"), part.material("SkirtCloth")
        self.assertEqual((lace.kind, lace.kind_source, lace.cloth_type), ("accessory", "user", None))
        self.assertEqual((cloth.kind, cloth.kind_source, cloth.cloth_type), ("cloth", "user", "skirt"))
        self.assertEqual(part.unconfirmed_materials, ())
        # 部品の種類は、そのまま（推測のまま）
        self.assertEqual(part.kind_source, "guess")
        with self.assertRaisesRegex(RuntimeError, "SkirtLace"):
            bpy.ops.ai_tools.set_cloth_type(object_name=skirt_name, material_name="SkirtLace", cloth_type="skirt")

    def test_changes_need_a_card(self):
        message = "登録カードがありません。先に「登録する」を押してください"
        with self.assertRaisesRegex(RuntimeError, message):
            bpy.ops.ai_tools.set_part_kind(object_name=mmd_model.BODY_NAME, kind="body")
        with self.assertRaisesRegex(RuntimeError, message):
            bpy.ops.ai_tools.answer_bone(bone_name="謎の骨", answer="extra")
        self.assertIsNone(bpy.data.texts.get(card_bpy.TEXT_NAME))

    def test_operators_do_not_save_or_change_the_model(self):
        """操作は、モデルの形・選択・モード・フレームを変えず、.blend のファイルを保存しない"""
        self.add_unknown_bones()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "model.blend")
            bpy.ops.wm.save_as_mainfile(filepath=str(path))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            listing = sorted(os.listdir(folder))
            meshes = [bpy.data.objects[name] for name in self.info["meshes"]]
            shapes = [self._coordinates(obj) for obj in meshes]
            skirt = bpy.data.objects[mmd_model.SKIRT_NAME]
            skirt.select_set(True)
            bpy.context.view_layer.objects.active = skirt
            frame = bpy.context.scene.frame_current
            selected = sorted(obj.name for obj in bpy.context.selected_objects)

            bpy.ops.ai_tools.register_card()
            bpy.ops.ai_tools.check_card()
            bpy.ops.ai_tools.check_bones()
            bpy.ops.ai_tools.set_part_kind(object_name=mmd_model.SKIRT_NAME, kind="cloth")
            bpy.ops.ai_tools.set_cloth_type(object_name=mmd_model.SKIRT_NAME, cloth_type="skirt")
            bpy.ops.ai_tools.answer_bone(bone_name="謎の骨", answer="extra")
            bpy.ops.ai_tools.clear_bone_answer(bone_name="謎の骨")
            for log in (draw(cls) for cls in panel.CLASSES):
                self.assertTrue(log)

            self.assertEqual(bpy.data.filepath, str(path))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            self.assertEqual(sorted(os.listdir(folder)), listing)
            for obj, shape in zip(meshes, shapes):
                np.testing.assert_array_equal(self._coordinates(obj), shape)
            self.assertIs(bpy.context.view_layer.objects.active, skirt)
            self.assertEqual(sorted(obj.name for obj in bpy.context.selected_objects), selected)
            self.assertEqual(bpy.context.scene.frame_current, frame)
            self.assertEqual(bpy.context.object.mode, "OBJECT")
            self.assertTrue(self.card().part(mmd_model.SKIRT_NAME).is_confirmed)

    @staticmethod
    def _coordinates(obj) -> np.ndarray:
        values = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
        obj.data.vertices.foreach_get("co", values)
        return values


# MARK: - 骨（A3）の操作


class BoneOperatorTest(UiCase):
    def bone_names(self):
        return [bone.name for bone in self.armature.data.bones]

    def test_check_bones_goes_through_the_catalog(self):
        calls = []
        original = registry.build_catalog

        def recording_catalog():
            catalog = original()
            real_call = catalog.call

            def call(tool_id, params, **kwargs):
                calls.append((tool_id, params, kwargs))
                return real_call(tool_id, params, **kwargs)

            catalog.call = call
            return catalog

        with mock.patch.object(registry, "build_catalog", recording_catalog):
            self.assertEqual(bpy.ops.ai_tools.check_bones(), {"FINISHED"})
        names = self.bone_names()
        self.assertEqual(calls, [("bones.map_names", {"names": names}, {})])
        check = state.bone_check()
        self.assertEqual((check.armature, check.names), (mmd_model.ARMATURE_NAME, tuple(names)))
        self.assertEqual(check.result, registry.map_bone_names({"names": names}))
        self.assertEqual(operators.bone_check_counts(check.result, len(names)),
                         {"total": 96, "mapped": 96, "extras": 0, "unknown": 0, "conflicts": 0, "missing": 0})

    def test_tool_not_in_catalog_is_refused(self):
        """目録に無い道具は動かない（A8）。ボタンも同じ決まりで断られる"""
        with mock.patch.object(registry, "build_catalog", Catalog):
            with self.assertRaisesRegex(RuntimeError, "目録にない道具です: bones.map_names"):
                bpy.ops.ai_tools.check_bones()
        self.assertIsNone(state.bone_check())

    def test_check_bones_counts_unknown_bones(self):
        self.add_unknown_bones()
        bpy.ops.ai_tools.check_bones()
        check = state.bone_check()
        self.assertEqual(check.result["unknown"], ["謎の骨", "leftThumbProximal"])
        self.assertEqual(check.result["ambiguous"], {"leftThumbProximal": ["左親指０", "左親指１"]})
        self.assertEqual(operators.bone_check_counts(check.result, len(check.names)),
                         {"total": 97, "mapped": 95, "extras": 0, "unknown": 2, "conflicts": 0, "missing": 1})

    def test_answer_bone_updates_the_card(self):
        self.add_unknown_bones()
        bpy.ops.ai_tools.register_card()
        card = self.card()
        self.assertEqual(card.bone_mapping["unknown"], ["謎の骨", "leftThumbProximal"])
        self.assertEqual(card.bone_mapping["missing_semi_standard"], ["左親指０"])
        self.assertEqual([q.name for q in state.card_view().questions], ["謎の骨", "leftThumbProximal"])
        # 1つに決まらない骨の候補のうち、もう使われている 左親指１ は出さない
        self.assertEqual(state.card_view().questions[1].candidates, ("左親指０",))

        self.assertEqual(bpy.ops.ai_tools.answer_bone(bone_name="謎の骨", answer="extra"), {"FINISHED"})
        self.assertEqual(bpy.ops.ai_tools.answer_bone(bone_name="leftThumbProximal", answer="左親指０"), {"FINISHED"})
        card = self.card()
        self.assertEqual(card.bone_answers, {"leftThumbProximal": "左親指０", "謎の骨": "extra"})
        self.assertEqual(card.bone_mapping["mapping"]["leftThumbProximal"], "左親指０")
        self.assertIn("謎の骨", card.bone_mapping["known_extras"])
        self.assertEqual((card.bone_mapping["unknown"], card.bone_mapping["missing_semi_standard"]), ([], []))
        self.assertTrue(card.bone_mapping["semi_standard_compliant"])
        self.assertEqual([(q.name, q.answer) for q in state.card_view().questions],
                         [("謎の骨", "extra"), ("leftThumbProximal", "左親指０")])

        self.assertEqual(bpy.ops.ai_tools.clear_bone_answer(bone_name="謎の骨"), {"FINISHED"})
        card = self.card()
        self.assertEqual(card.bone_answers, {"leftThumbProximal": "左親指０"})
        self.assertEqual(card.bone_mapping["unknown"], ["謎の骨"])
        # まだ答えていない骨が、一覧の先に来る
        self.assertEqual([(q.name, q.answer) for q in state.card_view().questions],
                         [("謎の骨", None), ("leftThumbProximal", "左親指０")])

    def test_answer_tooltips(self):
        cls = operators.AI_TOOLS_OT_answer_bone
        self.assertEqual(cls.description(bpy.context, None), text("screen.answer_choose_tip"))
        extra = SimpleNamespace(answer="extra", is_property_set=lambda name: name == "answer")
        self.assertEqual(cls.description(bpy.context, extra), text("screen.answer_extra_tip"))
        chosen = SimpleNamespace(answer="左腕", is_property_set=lambda name: name == "answer")
        self.assertEqual(cls.description(bpy.context, chosen), text("screen.answer_choose_tip"))

    def test_bad_answers_are_rejected(self):
        self.add_unknown_bones()
        bpy.ops.ai_tools.register_card()
        before = self.card_json()
        with self.assertRaisesRegex(RuntimeError, "答えを入れられませんでした：左腕 には、もう骨 '左腕' が対応しています"):
            bpy.ops.ai_tools.answer_bone(bone_name="謎の骨", answer="左腕")
        with self.assertRaisesRegex(RuntimeError, "辞書で決まる骨なので、答えは要りません"):
            bpy.ops.ai_tools.answer_bone(bone_name="左腕", answer="extra")
        with self.assertRaises(TypeError):
            bpy.ops.ai_tools.answer_bone(bone_name="謎の骨", answer="not a bone")
        self.assertEqual(self.card_json(), before)

    def test_answer_resolves_a_conflict(self):
        """2つの骨が同じ共通の名前を取り合ったときも、答えで決められる"""
        def change(bones):
            bones["左腕"].name = "J_Bip_L_UpperArm"
            bone = bones.new("mixamorig:LeftArm")
            bone.head = (0.3, 0.0, 1.2)
            bone.tail = (0.35, 0.0, 1.2)
        self.edit_armature(change)
        bpy.ops.ai_tools.register_card()
        questions = state.card_view().questions
        self.assertEqual([(q.name, q.conflict) for q in questions],
                         [("J_Bip_L_UpperArm", "左腕"), ("mixamorig:LeftArm", "左腕")])
        bpy.ops.ai_tools.answer_bone(bone_name="J_Bip_L_UpperArm", answer="左腕")
        mapping = self.card().bone_mapping
        self.assertEqual(mapping["mapping"]["J_Bip_L_UpperArm"], "左腕")
        self.assertEqual(mapping["conflicts"], {})
        self.assertEqual(mapping["shadowed"], {"mixamorig:LeftArm": "左腕"})


# MARK: - パネル


class PanelDrawTest(UiCase):
    def draw_all(self) -> dict:
        logs = {cls.__name__: draw(cls) for cls in panel.CLASSES}
        for name, log in logs.items():
            for call in log:
                with self.subTest(panel=name, call=call.get("text")):
                    self.assertIs(call["translate"], False, "文は Blender の翻訳を通さない")
                    self.assertIn(call["icon"], ICONS)
                    if call["kind"] == "label":
                        self.assertFalse(call["text"].startswith("思いがけないエラーです"), call["text"])
        return logs

    def test_without_card(self):
        logs = self.draw_all()
        main = label_texts(logs["VIEW3D_PT_ai_tools"])
        self.assertEqual(main[0], "モデル：TestArmature（部品 2 個）")
        model = logs["VIEW3D_PT_ai_tools_model"]
        self.assertEqual([call["text"] for call in model if call["kind"] == "operator"], ["登録する", "確かめる"])
        self.assertIn("まだ登録していません。「登録する」を押してください", label_texts(model))
        self.assertEqual([call for call in model if call["kind"] == "menu_enum"], [])
        bones = logs["VIEW3D_PT_ai_tools_bones"]
        self.assertEqual([call["idname"] for call in bones if call["kind"] == "operator"], ["ai_tools.check_bones"])

    def test_checks_panel_without_a_saved_file(self):
        """保存していないファイル：ボタンは見えて押せる（押すと日本語で断る）。版・仕事・報告は「まだ」と出る"""
        logs = self.draw_all()
        checks = logs["VIEW3D_PT_ai_tools_checks"]
        self.assertEqual([(call["idname"], call["text"], call["enabled"]) for call in checks if call["kind"] == "operator"],
                         [("ai_tools.save_version", "版を保存", True), ("ai_tools.compare_versions", "前の版と比べる", True),
                          ("ai_tools.export_and_check", "書き出して確かめる", True),
                          ("ai_tools.run_course", "確かめの動き", True)])
        lines = label_texts(checks)
        self.assertIn("ファイルを保存すると、版を保存できます", lines)
        self.assertIn(text("screen.export_needs_version"), lines)
        self.assertIn("まだ書いていません（アプリへの書き出しに要ります）", lines)
        self.assertIn(text("screen.course_note"), lines)
        self.assertEqual(label_texts(logs["VIEW3D_PT_ai_tools_jobs"]), ["仕事はありません"])
        self.assertEqual(label_texts(logs["VIEW3D_PT_ai_tools_report"]), [text("screen.report_none")])
        for idname in ("save_version", "compare_versions", "export_and_check", "run_course"):
            with self.subTest(idname=idname), self.assertRaisesRegex(RuntimeError, "先にファイルを保存してください"):
                getattr(bpy.ops.ai_tools, idname)()
        self.assertIsNone(background.queue())
        self.assertEqual(bpy.data.filepath, "")

    def test_card_summary_and_parts(self):
        bpy.ops.ai_tools.register_card()
        log = self.draw_all()["VIEW3D_PT_ai_tools_model"]
        lines = label_texts(log)
        height = self.card().value(HEIGHT)
        self.assertEqual(labels.millimeters_text(height), "1450.0")
        self.assertIn("背の高さ：1450.0 mm", lines)
        self.assertIn("【緑】MMD 準標準の骨：そろっています", lines)
        self.assertIn("【緑】辞書に無い・決まらない骨：0 本", lines)
        self.assertIn("【黄】種類を決めていない部品・材質：2 個", lines)
        self.assertIn("【緑】測れなかった値：0 個", lines)
        self.assertIn("【黄】書いていない素材の出所・利用条件：2 項目", lines)
        self.assertIn("部品（2 個）", lines)
        kinds = calls_of(log, "ai_tools.set_part_kind", "menu_enum")
        self.assertEqual([(c["text"], c["props"].object_name, c["props"].material_name, c["property"], c["alert"])
                          for c in kinds],
                         [("体（推測）", "Body", "", "kind", True), ("服（推測）", "Skirt", "", "kind", True)])
        cloth = calls_of(log, "ai_tools.set_cloth_type", "menu_enum")
        self.assertEqual([(c["text"], c["props"].object_name, c["property"]) for c in cloth],
                         [("服の種類を選ぶ", "Skirt", "cloth_type")])

        bpy.ops.ai_tools.set_part_kind(object_name="Body", kind="body")
        bpy.ops.ai_tools.set_cloth_type(object_name="Skirt", cloth_type="skirt")
        bpy.ops.ai_tools.check_card()
        log = draw(panel.VIEW3D_PT_ai_tools_model)
        lines = label_texts(log)
        self.assertIn("【緑】種類を決めていない部品・材質：0 個", lines)
        self.assertIn("【緑】測り直しが要る理由：0 個", lines)
        self.assertIn("確かめた結果", lines)
        self.assertIn("登録カードは、いまのモデルに合っています", lines)
        kinds = calls_of(log, "ai_tools.set_part_kind", "menu_enum")
        self.assertEqual([(c["text"], c["alert"]) for c in kinds], [("体", False), ("服", False)])
        self.assertEqual([c["text"] for c in calls_of(log, "ai_tools.set_cloth_type", "menu_enum")], ["スカート"])

    def test_material_rows(self):
        self.add_skirt_materials()
        bpy.ops.ai_tools.register_card()
        log = draw(panel.VIEW3D_PT_ai_tools_model)
        kinds = calls_of(log, "ai_tools.set_part_kind", "menu_enum")
        self.assertEqual([(c["props"].object_name, c["props"].material_name) for c in kinds],
                         [("Body", ""), ("Skirt", ""), ("Skirt", "SkirtCloth"), ("Skirt", "SkirtLace")])
        self.assertIn("部品（2 個）", label_texts(log))
        self.assertIn("SkirtLace", label_texts(log))

    def test_stale_card_is_shown(self):
        bpy.ops.ai_tools.register_card()
        self.move_skirt_vertex()
        bpy.ops.ai_tools.check_card()
        lines = label_texts(draw(panel.VIEW3D_PT_ai_tools_model))
        self.assertIn("【灰】測り直しが要る理由：1 個", lines)
        self.assertIn("測り直しが必要です（理由 1 個）。「登録する」で、もう一度測ってください", lines)
        self.assertIn("Skirt の形が変わりました", lines)

    def test_bone_questions(self):
        self.add_unknown_bones()
        bpy.ops.ai_tools.register_card()
        log = self.draw_all()["VIEW3D_PT_ai_tools_bones"]
        lines = label_texts(log)
        self.assertIn("聞きたい骨（2 本）", lines)
        self.assertIn("leftThumbProximal", lines)
        self.assertIn("謎の骨", lines)
        answers = calls_of(log, "ai_tools.answer_bone")
        self.assertEqual([(c["text"], c["props"].values, c["context"]) for c in answers], [
            ("選ぶ…", {"bone_name": "謎の骨"}, "INVOKE_DEFAULT"),
            ("揺れ物・人の形ではない", {"bone_name": "謎の骨", "answer": "extra"}, "EXEC_DEFAULT"),
            ("選ぶ…", {"bone_name": "leftThumbProximal"}, "INVOKE_DEFAULT"),
            ("揺れ物・人の形ではない", {"bone_name": "leftThumbProximal", "answer": "extra"}, "EXEC_DEFAULT"),
            ("左親指０", {"bone_name": "leftThumbProximal", "answer": "左親指０"}, "EXEC_DEFAULT"),
        ])
        self.assertIn("足りない準標準の骨（1 本。その骨が要る確かめは灰になります）", lines)
        self.assertIn("左親指０", lines)
        self.assertEqual(calls_of(log, "ai_tools.clear_bone_answer"), [])

        bpy.ops.ai_tools.answer_bone(bone_name="謎の骨", answer="extra")
        bpy.ops.ai_tools.check_bones()
        log = draw(panel.VIEW3D_PT_ai_tools_bones)
        lines = label_texts(log)
        self.assertIn("答え：揺れ物・人の形ではない", lines)
        self.assertEqual([c["props"].bone_name for c in calls_of(log, "ai_tools.clear_bone_answer")], ["謎の骨"])
        self.assertIn("調べた結果（TestArmature）", lines)
        self.assertIn("分からない骨：2 本", lines)
        self.assertIn("そのうち、登録カードで答えた骨：1 本", lines)

    def test_bones_without_card(self):
        self.add_unknown_bones()
        bpy.ops.ai_tools.check_bones()
        log = draw(panel.VIEW3D_PT_ai_tools_bones)
        lines = label_texts(log)
        self.assertIn("骨：97 本", lines)
        self.assertIn("候補：左親指０、左親指１", lines)
        self.assertIn("答えを覚えるには、先にモデルを登録してください", lines)
        self.assertEqual(calls_of(log, "ai_tools.answer_bone"), [])

    def test_long_sentences_wrap_in_a_narrow_panel(self):
        log = draw(panel.VIEW3D_PT_ai_tools_model, bpy.context)
        lines = label_texts(log)
        sentence = "まだ登録していません。「登録する」を押してください"
        start = lines.index("まだ登録していません。「登録する」を押し")
        self.assertEqual(lines[start:start + 2], ["まだ登録していません。「登録する」を押し", "てください"])
        self.assertEqual("".join(lines[start:start + 2]), sentence)
        self.assertTrue(all(labels.text_width(line) <= panel._DEFAULT_WIDTH for line in lines))

    def test_english(self):
        ui.apply_language("en")
        log = draw(panel.VIEW3D_PT_ai_tools_model)
        self.assertEqual([call["text"] for call in log if call["kind"] == "operator"], ["Register", "Check"])
        self.assertIn("Not registered yet. Press “Register”", label_texts(log))
        bpy.ops.ai_tools.register_card()
        lines = label_texts(draw(panel.VIEW3D_PT_ai_tools_model))
        self.assertIn("Height: 1450.0 mm", lines)
        self.assertIn("[Yellow] Parts or materials without a chosen kind: 2", lines)

    def test_draw_error_is_shown_not_raised(self):
        log: list = []

        def broken(layout, context):
            raise RuntimeError("こわれた")

        with mock.patch("traceback.print_exc"):
            panel.draw_safely(FakeLayout(log), bpy.context, broken)
        self.assertEqual(label_texts(log), ["思いがけないエラーです（RuntimeError: こわれた）。くわしくは、コンソールを見てください"])

    def test_long_lists_are_cut(self):
        def change(bones):
            for index in range(panel.MAX_LISTED_ROWS + 5):
                bone = bones.new(f"謎{index:03d}")
                bone.head = (0.0, 0.0, 1.5)
                bone.tail = (0.0, 0.0, 1.6)
        self.edit_armature(change)
        bpy.ops.ai_tools.register_card()
        log = draw(panel.VIEW3D_PT_ai_tools_bones)
        self.assertEqual(len(calls_of(log, "ai_tools.answer_bone")), panel.MAX_LISTED_ROWS * 2)
        self.assertIn("ほか 5 件（上から決めると、ここに出てきます）", label_texts(log))

    def test_long_parts_list_moves_decided_rows_down(self):
        """部品と材質が一覧の上限より多いとき：決めた物は後ろに回り、まだ決めていない物が上から出てくる
        （出てこない行があると、その材質の種類を決められない）"""
        skirt = bpy.data.objects[mmd_model.SKIRT_NAME]
        names = [f"M{index:02d}" for index in range(panel.MAX_LISTED_ROWS + 5)]
        for name in names:
            skirt.data.materials.append(bpy.data.materials.new(name))
        for polygon in skirt.data.polygons:
            polygon.material_index = polygon.index % len(names)
        bpy.context.view_layer.update()
        bpy.ops.ai_tools.register_card()
        total = 2 + len(names)  # Body・Skirt と、Skirt の材質

        def shown_rows():
            log = draw(panel.VIEW3D_PT_ai_tools_model)
            rows = [(c["props"].object_name, c["props"].material_name)
                    for c in calls_of(log, "ai_tools.set_part_kind", "menu_enum")]
            self.assertEqual(len(rows), panel.MAX_LISTED_ROWS)
            self.assertIn(f"ほか {total - panel.MAX_LISTED_ROWS} 件（上から決めると、ここに出てきます）", label_texts(log))
            return rows

        rows = shown_rows()
        self.assertEqual(rows[:2], [("Body", ""), ("Skirt", "")])
        self.assertEqual(rows[2:], [("Skirt", name) for name in names[:panel.MAX_LISTED_ROWS - 2]])
        decided = names[:10]
        for name in decided:
            bpy.ops.ai_tools.set_part_kind(object_name=mmd_model.SKIRT_NAME, material_name=name, kind="accessory")
        rows = shown_rows()
        undecided = [name for name in names if name not in decided]
        # まだ決めていない材質は全部出る。隠れるのは、決めた材質だけ
        self.assertEqual(rows[:2 + len(undecided)], [("Body", ""), ("Skirt", "")] + [("Skirt", n) for n in undecided])
        self.assertTrue(all(material in decided for _, material in rows[2 + len(undecided):]))
        # 全部決めたら、カードの順に戻る（決めた物を後ろに回すのは、入らないときだけ）
        bpy.ops.ai_tools.set_part_kind(object_name=mmd_model.BODY_NAME, kind="body")
        bpy.ops.ai_tools.set_cloth_type(object_name=mmd_model.SKIRT_NAME, cloth_type="skirt")
        for name in undecided:
            bpy.ops.ai_tools.set_part_kind(object_name=mmd_model.SKIRT_NAME, material_name=name, kind="cloth")
            bpy.ops.ai_tools.set_cloth_type(object_name=mmd_model.SKIRT_NAME, material_name=name, cloth_type="skirt")
        rows = shown_rows()
        self.assertEqual(rows, [("Body", ""), ("Skirt", "")] + [("Skirt", n) for n in names[:panel.MAX_LISTED_ROWS - 2]])

    def test_legacy_card_is_shown_read_only(self):
        """版 1 の古いカード：読めるだけ。種類の一覧も答えのボタンも出さず、「登録する」をすすめる"""
        bpy.ops.ai_tools.register_card()
        self.make_card_legacy()
        logs = self.draw_all()
        model = logs["VIEW3D_PT_ai_tools_model"]
        lines = label_texts(model)
        self.assertIn("古い形の登録カードです。「登録する」で、もう一度登録してください", lines)
        self.assertIn("背の高さ：1450.0 mm", lines)
        self.assertIn("体（推測）", lines)
        self.assertEqual([c for c in model if c["kind"] == "menu_enum"], [])
        bones = logs["VIEW3D_PT_ai_tools_bones"]
        self.assertIn("古い形の登録カードです。「登録する」で、もう一度登録してください", label_texts(bones))
        self.assertEqual(calls_of(bones, "ai_tools.answer_bone"), [])
        self.assertEqual(state.card_view().questions, ())
        # 確かめると「古い形なので測り直す」の理由が出る。変える操作は断る
        self.assertEqual(bpy.ops.ai_tools.check_card(), {"FINISHED"})
        lines = label_texts(draw(panel.VIEW3D_PT_ai_tools_model))
        self.assertIn(card_text("card.legacy_card", version="1"), lines)
        self.assertIn("【灰】測り直しが要る理由：1 個", lines)
        before = self.card_json()
        message = "古い形の登録カードは変えられません。先に「登録する」で、もう一度登録してください"
        with self.assertRaisesRegex(RuntimeError, message):
            bpy.ops.ai_tools.set_part_kind(object_name=mmd_model.BODY_NAME, kind="body")
        with self.assertRaisesRegex(RuntimeError, message):
            bpy.ops.ai_tools.answer_bone(bone_name="左腕", answer="extra")
        self.assertEqual(self.card_json(), before)
        # 「登録する」で、いまの形のカードになる
        self.assertEqual(bpy.ops.ai_tools.register_card(), {"FINISHED"})
        self.assertFalse(self.card().is_legacy)

    def test_unreadable_card_is_shown(self):
        bpy.ops.ai_tools.register_card()
        self.replace_card_json("{not json")
        logs = self.draw_all()
        for name in ("VIEW3D_PT_ai_tools_model", "VIEW3D_PT_ai_tools_bones"):
            with self.subTest(panel=name):
                lines = label_texts(logs[name])
                self.assertTrue(any(line.startswith("登録カードが読めません：") for line in lines), lines)
        self.assertEqual([c for c in logs["VIEW3D_PT_ai_tools_model"] if c["kind"] == "menu_enum"], [])
        self.assertEqual(bpy.ops.ai_tools.check_card(), {"CANCELLED"})
        with self.assertRaisesRegex(RuntimeError, "登録カードを変えられませんでした：カードの JSON が読めません"):
            bpy.ops.ai_tools.set_part_kind(object_name=mmd_model.BODY_NAME, kind="body")
        self.assertEqual(self.card_json(), "{not json")

    def test_unmeasured_height_is_gray(self):
        bpy.ops.ai_tools.register_card()
        data = json.loads(self.card_json())
        data["measurements"]["values"][HEIGHT].update(value=None, reason="テストの理由")
        self.replace_card_json(json.dumps(data, ensure_ascii=False))
        log = self.draw_all()["VIEW3D_PT_ai_tools_model"]
        gray = labels.SIGNAL_ICONS[Signal.GRAY]
        self.assertIn(("背の高さ：測れませんでした（テストの理由）", gray),
                      [(c["text"], c["icon"]) for c in log if c["kind"] == "label"])
        self.assertIn("【灰】測れなかった値：1 個", label_texts(log))

    def test_bone_check_result_keys_are_optional(self):
        """目録の道具の結果：sway_extras（揺れ物の骨）があれば数を出し、無くても（古い版の結果でも）描ける。
        知らない項目が増えても、数え方は変わらない"""
        original = registry.map_bone_names

        def with_sway(params):
            return {**original(params), "sway_extras": ["髪1", "髪2"], "something_new": 1}

        def without_sway(params):
            return {key: value for key, value in original(params).items() if key != "sway_extras"}

        with mock.patch.object(registry, "map_bone_names", with_sway):
            self.assertEqual(bpy.ops.ai_tools.check_bones(), {"FINISHED"})
        check = state.bone_check()
        self.assertEqual(operators.bone_check_counts(check.result, len(check.names)),
                         {"total": 96, "mapped": 96, "extras": 0, "unknown": 0, "conflicts": 0, "missing": 0})
        lines = label_texts(self.draw_all()["VIEW3D_PT_ai_tools_bones"])
        self.assertIn("骨：96 本", lines)
        self.assertIn("そのうち、揺れ物の骨：2 本", lines)
        with mock.patch.object(registry, "map_bone_names", without_sway):
            self.assertEqual(bpy.ops.ai_tools.check_bones(), {"FINISHED"})
        self.assertNotIn("sway_extras", state.bone_check().result)
        lines = label_texts(self.draw_all()["VIEW3D_PT_ai_tools_bones"])
        self.assertIn("骨：96 本", lines)
        self.assertFalse(any(line.startswith("そのうち、揺れ物の骨") for line in lines), lines)


class PanelWithoutModelTest(UiCase):
    """モデルの無いファイル：どのパネルも描けて、ボタンは日本語で断る"""

    with_model = False

    def test_panels_draw_without_a_model(self):
        logs = {cls.__name__: draw(cls) for cls in panel.CLASSES}
        for name, log in logs.items():
            for call in log:
                with self.subTest(panel=name, call=call.get("text")):
                    self.assertIs(call["translate"], False)
                    self.assertIn(call["icon"], ICONS)
                    if call["kind"] == "label":
                        self.assertFalse(call["text"].startswith("思いがけないエラーです"), call["text"])
        self.assertEqual(label_texts(logs["VIEW3D_PT_ai_tools"])[0], "モデルが見つかりません：アーマチュアがありません")
        self.assertIn("まだ登録していません。「登録する」を押してください", label_texts(logs["VIEW3D_PT_ai_tools_model"]))
        self.assertEqual([c["idname"] for c in logs["VIEW3D_PT_ai_tools_bones"] if c["kind"] == "operator"],
                         ["ai_tools.check_bones"])

    def test_operators_refuse_in_japanese_without_a_model(self):
        with self.assertRaisesRegex(RuntimeError, "登録できませんでした：アーマチュアがありません"):
            bpy.ops.ai_tools.register_card()
        with self.assertRaisesRegex(RuntimeError, "骨の名前を調べられませんでした：アーマチュアがありません"):
            bpy.ops.ai_tools.check_bones()
        self.assertEqual(bpy.ops.ai_tools.check_card(), {"CANCELLED"})
        self.assertIsNone(bpy.data.texts.get(card_bpy.TEXT_NAME))
        self.assertIsNone(state.bone_check())


# MARK: - 確かめと書き出し・裏の仕事・報告（A6・E1＋E2・A5・A10・A11）

TIME = "2026-10-21T07:00:00Z"


class DoneProcess:
    """終わった裏の Blender の代わり（poll() はすぐに終了コードを返す）"""

    pid = 4242

    def __init__(self, code=0):
        self.returncode = code

    def poll(self):
        return self.returncode

    def terminate(self):
        pass

    def kill(self):
        pass


class PendingProcess(DoneProcess):
    """動き続ける裏の Blender の代わり（returncode を入れると終わる）"""

    def __init__(self):
        super().__init__(None)
        self.calls = []

    def terminate(self):
        self.calls.append("terminate")

    def kill(self):
        self.calls.append("kill")


def fake_export_result(params):
    """export.convert_and_check の代わり（書き出さずに、決まった結果を返す）"""
    deviation = Measurement("roundtrip_deviation", 0.0007, "m", Signal.RED, "ずれ", {"pose": "knees"})
    return {"convert": {"ok": True, "signal": "green", "problems": [], "warnings": [], "summary": {},
                        "output_dir": params["output_dir"], "usdz": None, "rig_path": None, "input_unchanged": True,
                        "timed_out": False, "returncode": 0, "record": {}},
            "roundtrip": [deviation.to_data(),
                          Measurement("influence_limit_effect", 0.0004, "m", Signal.YELLOW, "少し").to_data()]}


class StageOneCase(UiCase):
    """保存したモデルのファイルを開き、裏の係の順番待ちを、にせの裏の Blender（runner）で作るようにする。

    mode="inline"：仕事を始めると、同じプロセスの中で、にせの目録の道具を jobs.worker.run_job で動かす（すぐ終わる）。
    mode="pending"：始めても終わらない（テストが progress.json を書き、プロセスを終わらせる）。
    """

    def setUp(self):
        super().setUp()
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.blend = self.folder / "model.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(self.blend))
        self.digest = self.file_digest()
        self.work = self.folder / "model_ai_tools"
        self.repo = self.folder / "koumakan"
        self.prefs = preferences.PreferenceValues()
        patcher = mock.patch.object(preferences, "get_preferences", lambda context=None: self.prefs)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.mode = "inline"
        self.calls = []
        self.processes = []
        patcher = mock.patch.object(launcher_bpy, "make_queue", self.make_queue)
        patcher.start()
        self.addCleanup(patcher.stop)
        # 後片付け（unregister）は UiCase の _restore。そのあとで、開いたファイルが変わっていないかを確かめる
        self.addCleanup(self.assert_open_file_untouched)

    def file_digest(self) -> str:
        return hashlib.sha256(self.blend.read_bytes()).hexdigest()

    def assert_open_file_untouched(self):
        """操作は、開いている .blend の場所を変えず、そのファイルを書き換えない"""
        self.assertEqual(self.file_digest(), self.digest, "開いている .blend が書き換えられました")

    def make_queue(self, root, *, blender=None):
        return JobQueue(root, self.runner, timeout=1000.0, grace=100.0)

    def fake_catalog(self):
        catalog = Catalog()

        def course(params):
            self.calls.append(("motion.run_course", params))
            events = [Event("arms_up", "floor_sink", Measurement("floor_sink", 0.0, "m", Signal.GREEN, "沈まない")),
                      Event("arms_up", "cloth_penetration",
                            Measurement("penetration_depth", 0.003, "m", Signal.RED, "深い"))]
            return {"events": [{"pose": e.pose, "check": e.check, "measurement": e.measurement.to_data()}
                               for e in events], "summary": {}, "lines": []}

        def export(params):
            self.calls.append(("export.convert_and_check", params))
            return fake_export_result(params)

        catalog.register(registry.RUN_COURSE, course)
        catalog.register(registry.CONVERT_AND_CHECK, export)
        return catalog

    def runner(self, job, job_dir):
        if self.mode == "inline":
            final = jobs_worker.run_job(job, self.fake_catalog(), job_dir)
            process = DoneProcess(0 if final.status == "done" else 1)
        else:
            process = PendingProcess()
            jobs_progress.write_progress(job_dir, jobs_progress.Progress(status="running", updated_at=TIME,
                                                                         started_at=TIME, step=1, total=4,
                                                                         message_key="export.step_convert"))
        self.processes.append(process)
        return WorkerProcess(process=process)

    def make_koumakan(self):
        converter = self.repo / "tools" / "model-converter"
        converter.mkdir(parents=True)
        for name in ("convert_blend.py", "convert_pmx.py"):
            (converter / name).write_text("# にせの紅魔館\n", encoding="utf-8")
        self.prefs = preferences.PreferenceValues(koumakan_repo=str(self.repo))

    def jobs(self):
        return [row.job for row in background.rows()]

    def draw_checked(self, panel_class) -> list:
        log = draw(panel_class)
        for call in log:
            with self.subTest(call=call.get("text")):
                self.assertIs(call["translate"], False)
                self.assertIn(call["icon"], ICONS)
                if call["kind"] == "label":
                    self.assertFalse(call["text"].startswith("思いがけないエラーです"), call["text"])
        return log


class VersionButtonTest(StageOneCase):
    def test_save_version_and_list(self):
        self.assertEqual(state.versions(), ())
        self.assertIn("まだ版を保存していません", label_texts(self.draw_checked(panel.VIEW3D_PT_ai_tools_checks)))
        self.assertEqual(bpy.ops.ai_tools.save_version(), {"FINISHED"})
        self.assertEqual(bpy.ops.ai_tools.save_version(), {"FINISHED"})
        self.assertEqual([(entry.number, entry.name) for entry in state.versions()],
                         [(1, "model_v001.blend"), (2, "model_v002.blend")])
        self.assertTrue((self.folder / "model_v002.blend").is_file())
        self.assertEqual(bpy.data.filepath, str(self.blend))
        lines = label_texts(self.draw_checked(panel.VIEW3D_PT_ai_tools_checks))
        self.assertIn("保存した版（2 個）", lines)
        self.assertLess(lines.index("第 2 版：model_v002.blend"), lines.index("第 1 版：model_v001.blend"))
        self.assertIn("書き出す版：model_v002.blend", lines)
        self.assertIsNone(background.queue())  # 軽い道具では、仕事のフォルダを作らない
        self.assertFalse(self.work.exists())

    def test_versions_list_follows_files(self):
        """別のファイルを開くと、覚えた物を消して読み直す（保存した版の一覧と報告）"""
        bpy.ops.ai_tools.save_version()
        bpy.ops.ai_tools.compare_versions()
        self.assertIsNotNone(state.report_view())
        other = self.folder / "other.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(other), copy=True)
        bpy.ops.wm.open_mainfile(filepath=str(other))
        self.assertEqual(state.versions(), ())
        self.assertIsNone(state.report_view())
        bpy.ops.wm.open_mainfile(filepath=str(self.blend))
        self.assertEqual([entry.name for entry in state.versions()], ["model_v001.blend"])

    def test_compare_report_decisions_html_and_baseline(self):
        bpy.ops.ai_tools.save_version()
        self.assertEqual(bpy.ops.ai_tools.compare_versions(), {"FINISHED"})
        view = state.report_view()
        self.assertEqual((view.report.kind, [item.item_id for item in view.report.items]),
                         ("versions", ["versions/unexpected_changes"]))
        self.move_skirt_vertex()
        bpy.ops.ai_tools.compare_versions()
        report = state.report_view().report
        skirt = report_sources.short_hash(mmd_model.SKIRT_NAME)
        item_id = f"versions/vertices/{skirt}/vertices_moved"
        self.assertEqual([(item.item_id, item.signal) for item in report.items], [(item_id, Signal.RED)])
        self.assertTrue(report.item(item_id).titles["en"].startswith(f"{mmd_model.SKIRT_NAME}: 1 vertices moved"))
        log = self.draw_checked(panel.VIEW3D_PT_ai_tools_report)
        lines = label_texts(log)
        self.assertEqual(lines[0], f"前の版との比べ（{report.created_at}）")
        self.assertEqual(lines[1], "【赤】赤が 1 件あります（黄 0 件、測れなかった物 0 件）")
        self.assertIn("前の版の赤の数の記録がありません（今回の 1 件を基準として記録します）", lines)
        self.assertIn("値：1 件", lines)
        self.assertIn("変えてよい所の外の変化です（変える前に、変えてよい所を決めてください）", lines)
        decide = calls_of(log, "ai_tools.decide")
        self.assertEqual([(c["text"], c["props"].item_id, c["props"].decision, c["context"]) for c in decide],
                         [("直す（おすすめ）", item_id, "fix", "EXEC_DEFAULT"),
                          ("このままでよい", item_id, "accept", "EXEC_DEFAULT"),
                          ("あとで", item_id, "later", "EXEC_DEFAULT")])
        self.assertEqual(calls_of(log, "ai_tools.clear_decision"), [])

        # 答えは作業フォルダの decisions.json に記録し、報告に写す
        self.assertEqual(bpy.ops.ai_tools.decide(item_id=item_id, decision="accept"), {"FINISHED"})
        stored = json.loads((self.work / "decisions.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["decisions"][item_id]["decision"], "accept")
        self.assertEqual(state.report_view().report.item(item_id).decision, "accept")
        log = self.draw_checked(panel.VIEW3D_PT_ai_tools_report)
        lines = label_texts(log)
        self.assertEqual(lines[1], "【緑】大丈夫です、「このままでよい」とした物 1 件")
        self.assertIn("答え：このままでよい", lines)
        self.assertEqual([c["props"].item_id for c in calls_of(log, "ai_tools.clear_decision")], [item_id])
        # 比べ直しても、答えはそのまま（id が同じ）
        bpy.ops.ai_tools.compare_versions()
        self.assertEqual(state.report_view().report.item(item_id).decision, "accept")
        self.assertEqual(bpy.ops.ai_tools.clear_decision(item_id=item_id), {"FINISHED"})
        self.assertEqual(json.loads((self.work / "decisions.json").read_text(encoding="utf-8"))["decisions"], {})
        self.assertIsNone(state.report_view().report.item(item_id).decision)

        # 赤の数の基準
        self.assertEqual(bpy.ops.ai_tools.record_baseline(), {"FINISHED"})
        stored = json.loads((self.work / "decisions.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["baselines"]["versions"]["red"], 1)
        self.assertIn("赤は 1 件（前の版は 1 件）", label_texts(self.draw_checked(panel.VIEW3D_PT_ai_tools_report)))

        # HTML：作業フォルダの reports/<報告の id>.html。開くのは差しかえた関数で確かめる
        self.assertEqual(bpy.ops.ai_tools.write_report_html(), {"FINISHED"})
        html_path = self.work / "reports" / f"{state.report_view().report.report_id}.html"
        self.assertTrue(html_path.is_file())
        self.assertIn("点検の報告", html_path.read_text(encoding="utf-8"))
        with mock.patch.object(operators, "open_path") as opened:
            self.assertEqual(bpy.ops.ai_tools.open_report_html(), {"FINISHED"})
        opened.assert_called_once_with(html_path)
        self.assertIn(f"HTML：{html_path}", label_texts(self.draw_checked(panel.VIEW3D_PT_ai_tools_report)))

        # まちがった答え・無い項目は断る
        with self.assertRaisesRegex(RuntimeError, "報告に、その項目がありません"):
            bpy.ops.ai_tools.decide(item_id="versions/nope", decision="fix")
        with self.assertRaises(TypeError):
            bpy.ops.ai_tools.decide(item_id=item_id, decision="ignore")
        self.assertEqual(bpy.data.filepath, str(self.blend))

    def test_unreadable_decisions_are_shown_and_not_overwritten(self):
        bpy.ops.ai_tools.save_version()
        self.move_skirt_vertex()
        self.work.mkdir()
        (self.work / "decisions.json").write_text("{broken", encoding="utf-8")
        bpy.ops.ai_tools.compare_versions()
        lines = label_texts(self.draw_checked(panel.VIEW3D_PT_ai_tools_report))
        self.assertTrue(any(line.startswith("答えの記録（decisions.json）が読めません：") for line in lines), lines)
        item_id = state.report_view().report.items[0].item_id
        with self.assertRaisesRegex(RuntimeError, "答えを記録できませんでした"):
            bpy.ops.ai_tools.decide(item_id=item_id, decision="fix")
        self.assertEqual((self.work / "decisions.json").read_text(encoding="utf-8"), "{broken")

    def test_report_buttons_need_a_report(self):
        for idname in ("decide", "clear_decision", "write_report_html", "open_report_html", "record_baseline"):
            with self.subTest(idname=idname), self.assertRaisesRegex(RuntimeError, "報告がありません"):
                kwargs = {"item_id": "x"} if idname in ("decide", "clear_decision") else {}
                getattr(bpy.ops.ai_tools, idname)(**kwargs)
        self.assertFalse(self.work.exists())


class ExportButtonTest(StageOneCase):
    def test_refused_without_koumakan_or_version(self):
        with self.assertRaisesRegex(RuntimeError, "紅魔館の場所を、拡張機能の設定に書いてください"):
            bpy.ops.ai_tools.export_and_check()
        self.prefs = preferences.PreferenceValues(koumakan_repo=str(self.folder / "nowhere"))
        with self.assertRaisesRegex(RuntimeError, "紅魔館の場所が正しくありません"):
            bpy.ops.ai_tools.export_and_check()
        self.make_koumakan()
        with self.assertRaisesRegex(RuntimeError, "書き出す版がありません。先に「版を保存」を押してください"):
            bpy.ops.ai_tools.export_and_check()
        self.assertIsNone(background.queue())
        self.assertFalse(self.work.exists(), "断ったときは、作業フォルダを作らない")

    def test_refuses_app_model_folder_as_work_folder(self):
        """作業フォルダがアプリの Model フォルダの中でも、そこには書き出さない"""
        self.make_koumakan()
        bpy.ops.ai_tools.save_version()
        app_model = self.repo / "KoumakanApp" / "KoumakanApp" / "Model"
        self.prefs = preferences.PreferenceValues(koumakan_repo=str(self.repo), work_dir=str(app_model))
        with self.assertRaisesRegex(RuntimeError, "アプリの Model フォルダ"):
            bpy.ops.ai_tools.export_and_check()
        self.assertFalse((app_model / "exports").exists())
        self.assertEqual(self.calls, [])

    def test_exports_the_newest_version_in_the_background(self):
        self.make_koumakan()
        bpy.ops.ai_tools.save_version()
        bpy.ops.ai_tools.save_version()
        self.assertEqual(bpy.ops.ai_tools.export_and_check(), {"FINISHED"})
        queue = background.queue()
        self.assertEqual(queue.root, self.work / "jobs")
        [job] = self.jobs()
        newest = str(self.folder / "model_v002.blend")
        output = self.work / "exports" / f"model_v002_{job.job_id}"
        self.assertEqual(job.tool_id, "export.convert_and_check")
        self.assertEqual(job.input_blend, newest)
        self.assertEqual(job.params, {"input_blend": newest, "output_dir": str(output),
                                      "koumakan_repo": str(self.repo), "blender": ""})
        self.assertFalse(output.exists())
        self.assertEqual(self.calls, [])  # まだ始めていない（タイマーが始める）
        log = self.draw_checked(panel.VIEW3D_PT_ai_tools_jobs)
        self.assertIn("書き出して確かめる：順番待ち", label_texts(log))
        self.assertEqual([c["props"].job_id for c in calls_of(log, "ai_tools.cancel_job")], [job.job_id])

        background.step()  # 始めて（同じプロセスの中で）終わる。プロセスはまだ片付いていない
        self.assertEqual(self.calls, [("export.convert_and_check", job.params)])
        [row] = background.rows()
        self.assertTrue(row.cleaning)
        log = self.draw_checked(panel.VIEW3D_PT_ai_tools_jobs)
        self.assertIn("書き出して確かめる：片付け中", label_texts(log))
        self.assertEqual([c for c in log if c["kind"] == "operator"], [])
        # 終わった仕事の結果は、報告に出る
        report = state.report_view().report
        self.assertEqual(report.kind, "export")
        self.assertEqual(report.source["version"], "model_v002.blend")
        self.assertEqual(report.source["job_id"], job.job_id)
        self.assertIn("export/roundtrip_deviation/knees", [item.item_id for item in report.items])
        lines = label_texts(self.draw_checked(panel.VIEW3D_PT_ai_tools_report))
        self.assertIn("【赤】読み戻した形のずれ：ひざを曲げる", lines)
        self.assertIn("値：0.70 mm", lines)

        background.step()
        [row] = background.rows()
        self.assertFalse(row.cleaning)
        log = self.draw_checked(panel.VIEW3D_PT_ai_tools_jobs)
        self.assertIn("書き出して確かめる：終わりました", label_texts(log))
        self.assertEqual([c["idname"] for c in log if c["kind"] == "operator"],
                         ["ai_tools.show_job_result", "ai_tools.remove_job"])
        # ほかの報告を出したあとでも、結果を見直せる
        state.show_report(None)
        self.assertEqual(bpy.ops.ai_tools.show_job_result(job_id=job.job_id), {"FINISHED"})
        self.assertEqual(state.report_view().report.kind, "export")
        self.assertEqual(bpy.ops.ai_tools.remove_job(job_id=job.job_id), {"FINISHED"})
        self.assertEqual(background.rows(), ())
        self.assertFalse(queue.job_dir(job.job_id).exists())
        with self.assertRaisesRegex(RuntimeError, "その仕事はありません"):
            bpy.ops.ai_tools.show_job_result(job_id=job.job_id)
        self.assertEqual(bpy.data.filepath, str(self.blend))

    def test_bad_params_are_refused_before_enqueue(self):
        self.make_koumakan()
        bpy.ops.ai_tools.save_version()
        with mock.patch.object(Catalog, "validate_call", return_value=["値が合いません"]):
            with self.assertRaisesRegex(RuntimeError, "書き出しを始められませんでした：値が合いません"):
                bpy.ops.ai_tools.export_and_check()
        self.assertEqual(self.jobs(), [])


class CourseButtonTest(StageOneCase):
    def test_course_runs_on_a_copy_in_the_job_folder(self):
        self.assertEqual(bpy.ops.ai_tools.run_course(), {"FINISHED"})
        [job] = self.jobs()
        copy = background.queue().job_dir(job.job_id) / "input.blend"
        self.assertEqual((job.tool_id, job.params, job.input_blend), ("motion.run_course", {"visibility": True}, str(copy)))
        self.assertTrue(copy.is_file())
        self.assertEqual(bpy.data.filepath, str(self.blend))  # 写しを保存しても、開いているファイルは変わらない
        background.step()
        background.step()
        self.assertEqual(self.calls, [("motion.run_course", {"visibility": True})])
        self.assertEqual([row.progress.status for row in background.rows()], ["done"])
        report = state.report_view().report
        self.assertEqual((report.kind, [item.item_id for item in report.items]),
                         ("pose_course", ["pose_course/arms_up/floor_sink", "pose_course/arms_up/cloth_penetration"]))
        self.assertNotIn("version", report.source)  # 写しは版ではない
        self.assertIn("確かめの動き：終わりました", label_texts(self.draw_checked(panel.VIEW3D_PT_ai_tools_jobs)))

    def test_copy_failure_withdraws_the_job(self):
        with mock.patch.object(operators, "save_copy", side_effect=RuntimeError("ディスクがいっぱい")):
            with self.assertRaisesRegex(RuntimeError, "写しを保存できなかったので、仕事を取り消しました"):
                bpy.ops.ai_tools.run_course()
        self.assertEqual(self.jobs(), [])
        self.assertEqual([path.name for path in (self.work / "jobs").iterdir()], [])
        background.step()
        self.assertEqual(self.calls, [])

    def test_cancel_running_job_and_errors(self):
        self.mode = "pending"
        bpy.ops.ai_tools.run_course()
        [job] = self.jobs()
        background.step()
        [process] = self.processes
        log = self.draw_checked(panel.VIEW3D_PT_ai_tools_jobs)
        lines = label_texts(log)
        self.assertIn("確かめの動き：動いています", lines)
        self.assertIn("1 / 4：convert_blend.py で書き出しています", lines)
        self.assertEqual([c["props"].job_id for c in calls_of(log, "ai_tools.cancel_job")], [job.job_id])
        self.assertEqual(bpy.ops.ai_tools.cancel_job(job_id=job.job_id), {"FINISHED"})
        folder = background.queue().job_dir(job.job_id)
        self.assertTrue((folder / jobs_progress.CANCEL_FILE).is_file())
        # 裏の係が「中止しました」を書いたが、プロセスはまだ残っている：片付け中（ボタンは出さない）
        jobs_progress.write_progress(folder, jobs_progress.Progress(
            status="cancelled", updated_at=TIME, finished_at=TIME, error_key="jobs.error_cancelled"))
        background.step()
        log = self.draw_checked(panel.VIEW3D_PT_ai_tools_jobs)
        self.assertIn("確かめの動き：片付け中", label_texts(log))
        self.assertEqual([c for c in log if c["kind"] == "operator"], [])
        with self.assertRaisesRegex(RuntimeError, "仕事を扱えませんでした"):
            bpy.ops.ai_tools.remove_job(job_id=job.job_id)
        process.returncode = 0
        background.step()
        log = self.draw_checked(panel.VIEW3D_PT_ai_tools_jobs)
        lines = label_texts(log)
        self.assertIn("確かめの動き：中止しました", lines)
        self.assertIn("中止の頼みを受けて止めました", lines)
        self.assertEqual([c["idname"] for c in log if c["kind"] == "operator"], ["ai_tools.remove_job"])
        self.assertIsNone(state.report_view())  # 中止した仕事は報告にしない

        # 失敗した仕事：決まった理由（値を入れた文）と、自由な文
        bpy.ops.ai_tools.run_course()
        bpy.ops.ai_tools.run_course()
        first, second = [row.job for row in background.rows()][1:]
        background.step()
        folder = background.queue().job_dir(first.job_id)
        jobs_progress.write_progress(folder, jobs_progress.Progress(
            status="failed", updated_at=TIME, finished_at=TIME, error_key="jobs.error_input_changed",
            error_values={"path": "/x/input.blend"}))
        self.processes[-1].returncode = 1
        background.step()  # 1つ目を片付け、2つ目を始める
        jobs_progress.write_progress(background.queue().job_dir(second.job_id), jobs_progress.Progress(
            status="failed", updated_at=TIME, finished_at=TIME, error="ValueError: こわれた"))
        self.processes[-1].returncode = 1
        background.step()
        lines = label_texts(self.draw_checked(panel.VIEW3D_PT_ai_tools_jobs))
        self.assertIn("入力のファイルが途中で変わりました。結果は使えません（/x/input.blend）", lines)
        self.assertIn("ValueError: こわれた", lines)
        self.assertEqual(lines.count("確かめの動き：失敗しました"), 2)
        # 新しい仕事が上
        titles = [line for line in lines if line.startswith("確かめの動き：")]
        self.assertEqual(titles, ["確かめの動き：失敗しました", "確かめの動き：失敗しました", "確かめの動き：中止しました"])

    def test_refused_without_a_model(self):
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj)
        with self.assertRaisesRegex(RuntimeError, "確かめの動きを始められませんでした：アーマチュアがありません"):
            bpy.ops.ai_tools.run_course()
        self.assertIsNone(background.queue())
        self.assertFalse(self.work.exists())

    def test_queued_job_can_be_cancelled_and_removed(self):
        bpy.ops.ai_tools.run_course()
        [job] = self.jobs()
        self.assertEqual(bpy.ops.ai_tools.cancel_job(job_id=job.job_id), {"FINISHED"})
        self.assertEqual([row.progress.status for row in background.rows()], ["cancelled"])
        background.step()
        self.assertEqual(self.calls, [])  # 中止した仕事は始めない
        self.assertEqual(bpy.ops.ai_tools.remove_job(job_id=job.job_id), {"FINISHED"})
        self.assertEqual(background.rows(), ())
        with self.assertRaisesRegex(RuntimeError, "その仕事はありません"):
            bpy.ops.ai_tools.cancel_job(job_id=job.job_id)
        with self.assertRaisesRegex(RuntimeError, "仕事を扱えませんでした"):
            bpy.ops.ai_tools.cancel_job(job_id="../outside")

    def test_many_jobs_are_cut(self):
        for _ in range(panel.MAX_JOB_ROWS + 2):
            bpy.ops.ai_tools.run_course()
        lines = label_texts(self.draw_checked(panel.VIEW3D_PT_ai_tools_jobs))
        self.assertEqual(sum(line.startswith("確かめの動き：") for line in lines), panel.MAX_JOB_ROWS)
        self.assertIn("ほか 2 件（古い仕事）", lines)


class BackgroundQueueTest(StageOneCase):
    def test_queue_is_made_only_when_the_jobs_folder_exists(self):
        """register・ファイルを開いたときは、仕事のフォルダがあるときだけ順番待ちを作る（フォルダを勝手に作らない）"""
        blender_ai_tools.unregister()
        blender_ai_tools.register()
        self.assertIsNone(background.queue())
        self.assertFalse(self.work.exists())
        (self.work / "jobs").mkdir(parents=True)
        stops = []
        real_start = launcher_bpy.start_polling

        def recording_start(queue, **kwargs):
            stop = real_start(queue, **kwargs)

            def recorded_stop():
                stops.append(queue.root)
                stop()

            return recorded_stop

        with mock.patch.object(launcher_bpy, "start_polling", recording_start):
            blender_ai_tools.register()
            first = background.queue()
            self.assertEqual(first.root, self.work / "jobs")
            blender_ai_tools.register()  # 登録し直しても、1つだけ（前のタイマーは止める）
            self.assertEqual(stops, [self.work / "jobs"])
            self.assertIsNot(background.queue(), first)
            blender_ai_tools.unregister()
            self.assertEqual(stops, [self.work / "jobs"] * 2)
            self.assertIsNone(background.queue())
        blender_ai_tools.register()

    def test_reopening_recovers_orphaned_jobs(self):
        """前の回の裏の Blender が消えていたら、開き直したときに失敗（結果なし）にする"""
        self.mode = "pending"
        bpy.ops.ai_tools.run_course()
        [job] = self.jobs()
        background.step()
        other = self.folder / "other.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(other), copy=True)
        bpy.ops.wm.open_mainfile(filepath=str(other))
        self.assertIsNone(background.queue())  # other_ai_tools/jobs は無い
        self.assertEqual(background.rows(), ())
        bpy.ops.wm.open_mainfile(filepath=str(self.blend))
        self.assertEqual(background.queue().root, self.work / "jobs")
        [row] = background.rows()
        self.assertEqual((row.job.job_id, row.progress.status, row.progress.error_key),
                         (job.job_id, "failed", "jobs.error_orphaned"))

    def test_load_post_stops_the_old_timer(self):
        bpy.ops.ai_tools.run_course()
        stopped = []
        with mock.patch.object(background, "_stop", lambda: stopped.append(True)):
            background.on_load_post()
        self.assertEqual(stopped, [True])
        self.assertEqual(background.queue().root, self.work / "jobs")  # 同じファイルなので、つなぎ直す

    def test_draw_does_not_change_state(self):
        bpy.ops.ai_tools.run_course()
        before = (background.rows(), state.report_view(), state.versions())
        listing = sorted(path.name for path in self.work.rglob("*"))
        for panel_class in panel.CLASSES:
            self.draw_checked(panel_class)
        self.assertEqual((background.rows(), state.report_view(), state.versions()), before)
        self.assertEqual(sorted(path.name for path in self.work.rglob("*")), listing)
        self.assertEqual(self.calls, [])


# MARK: - 小さな道具


class LabelsTest(unittest.TestCase):
    def tearDown(self):
        texts.set_language("ja")

    def test_wrap_text(self):
        self.assertEqual(labels.wrap_text("あいうえお", 4), ["あい", "うえ", "お"])
        self.assertEqual(labels.wrap_text("hello world foo", 11), ["hello world", "foo"])
        self.assertEqual(labels.wrap_text("abcdefghij", 4), ["abcd", "efgh", "ij"])
        self.assertEqual(labels.wrap_text("abc あいう", 6), ["abc あ", "いう"])
        self.assertEqual(labels.wrap_text("", 10), [""])
        self.assertEqual(labels.wrap_text("短い", 10), ["短い"])
        with self.assertRaises(ValueError):
            labels.wrap_text("x", 1)
        self.assertEqual(labels.text_width("aあ"), 3)

    def test_line_width(self):
        def context(width, scale=1.0):
            return SimpleNamespace(region=SimpleNamespace(width=width),
                                   preferences=SimpleNamespace(system=SimpleNamespace(ui_scale=scale)))
        self.assertEqual(panel.line_width(context(300)), 37)
        self.assertEqual(panel.line_width(context(600, 2.0)), 37)
        self.assertEqual(panel.line_width(context(60)), 16)
        self.assertEqual(panel.line_width(SimpleNamespace(region=None)), 40)

    def test_numbers_and_names(self):
        texts.set_language("ja")
        self.assertEqual(labels.millimeters_text(1.4500000476837158), "1450.0")
        self.assertEqual(labels.millimeters_text(0.00004), "0.0")
        self.assertEqual(labels.count_text(3.0), "3")
        self.assertEqual(labels.count_text(None), "測れていません")
        self.assertEqual(labels.chosen_kind_label("cloth", "guess"), "服（推測）")
        self.assertEqual(labels.chosen_kind_label("cloth", "user"), "服")
        self.assertEqual(labels.cloth_label(None), "服の種類を選ぶ")
        self.assertEqual(labels.answer_label("extra"), "揺れ物・人の形ではない")
        self.assertEqual(labels.answer_label("左腕"), "左腕")
        items = labels.answer_items()
        self.assertEqual(items[0][:2], ("extra", "揺れ物・人の形ではない"))
        self.assertIn(("左腕", "左腕", ""), items)
        # 同じ言語なら、同じ一覧（Blender が文字を読む間、残っている物）を返す
        self.assertIs(labels.answer_items(), items)


if __name__ == "__main__":
    unittest.main()

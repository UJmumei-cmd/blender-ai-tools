"""画面（ui/）を Blender の中で確かめる：登録と解除・言語・設定・ボタンの操作・パネルの描き方。

テスト用のモデル（tests/fixtures/mmd_model.py）を空のシーンに作ってから調べる。
パネルは、バックグラウンドの Blender には画面が無いので、呼ばれた物を書き留める「にせの layout」で描く。
ファイルは一時フォルダの中だけに作る。
"""

import hashlib
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
from blender_ai_tools.model import card_bpy
from blender_ai_tools.model.card import HEIGHT, card_text
from blender_ai_tools.texts import text
from blender_ai_tools.ui import labels, operators, panel, preferences, state
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
        expected = preferences.resolve_language("auto", preferences.blender_locale())
        for before in ("ja", "en"):
            with self.subTest(before=before):
                texts.set_language(before)
                blender_ai_tools.register()
                try:
                    self.assertEqual(texts.current_language(), expected)
                finally:
                    blender_ai_tools.unregister()
                self.assertEqual(texts.current_language(), before)

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
        self.assertEqual(bpy.types.VIEW3D_PT_ai_tools_next.bl_label, "書き出し・確かめの動き・版")
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
                         ("", "", "", "auto"))
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

    def test_preferences_when_enabled(self):
        """設定から有効にしたとき：言語を変えるとすぐに文が変わり、登録し直しても設定の値は残る"""
        addon = bpy.context.preferences.addons.new()
        addon.module = preferences.ADDON_ID
        self.addCleanup(bpy.context.preferences.addons.remove, addon)
        blender_ai_tools.register()
        self.addCleanup(blender_ai_tools.unregister)
        prefs = preferences.get_preferences()
        self.assertIsInstance(prefs, preferences.AIToolsPreferences)
        self.assertEqual(prefs.language, "auto")
        prefs.language = "ja"
        self.assertEqual(texts.current_language(), "ja")
        self.assertEqual(bpy.types.VIEW3D_PT_ai_tools.bl_category, "AI の道具")
        prefs.language = "en"
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

    def test_next_steps_are_visible_but_disabled(self):
        log = draw(panel.VIEW3D_PT_ai_tools_next)
        buttons = calls_of(log, "ai_tools.coming_soon")
        self.assertEqual([(call["text"], call["props"].feature, call["enabled"]) for call in buttons],
                         [("書き出す", "export", False), ("確かめの動き", "motion_check", False),
                          ("版を保存する", "versions", False)])
        self.assertIn("これらは、裏で動かす係（A10）ができたら、別の Blender で動きます", label_texts(log))
        self.assertFalse(bpy.ops.ai_tools.coming_soon.poll())
        with self.assertRaises(RuntimeError):
            bpy.ops.ai_tools.coming_soon(feature="export")
        props = SimpleNamespace(feature="export")
        self.assertEqual(operators.AI_TOOLS_OT_coming_soon.description(bpy.context, props),
                         text("screen.next_export_tip"))

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

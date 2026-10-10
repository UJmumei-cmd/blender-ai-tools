"""A1 モデルの登録カードを Blender の中で作る・置く・確かめる所（model/card_bpy.py）を確かめる。

テスト用のモデル（tests/fixtures/mmd_model.py）を空のシーンに作ってから調べる。答えは、モデルを作るときの
決まった数字（背の高さ 1.45・腕の下がり角 37.5 度・関節の位置）から、手で計算した値。
ファイルは一時フォルダの中だけに作る。
"""

import math
import tempfile
import unittest
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

from blender_ai_tools import provenance, thresholds
from blender_ai_tools.model import card_bpy
from blender_ai_tools.model.card import (
    ARM_REST_DOWN, HEAD_RATIO, HEIGHT, LEG_LENGTH, SHOULDER_WIDTH, ModelCard, set_part_kind, set_source,
    summarize_card,
)
from blender_ai_tools.model.card_bpy import TEXT_NAME, build_card, check_card, load_card, save_card
from blender_ai_tools.model.extract_bpy import bone_rest, set_pose
from tests.fixtures import mmd_model

BONES_CHANGED = "骨（名前・並び・親子・位置・ねじれ・形を動かすかどうか）が変わりました"


def leg_length():
    """左脚の長さ：|足 − ひざ| + |ひざ − 足首|（モデルを作るときの関節の位置から）"""
    hip, knee, ankle = (np.array(p) for p in (mmd_model.HIP_JOINT, mmd_model.KNEE_JOINT, mmd_model.ANKLE_JOINT))
    return float(np.linalg.norm(knee - hip) + np.linalg.norm(ankle - knee))


class CardCase(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_homefile(use_empty=True)
        self.info = mmd_model.build_model()
        self.armature = bpy.data.objects[mmd_model.ARMATURE_NAME]
        self.body = bpy.data.objects[mmd_model.BODY_NAME]
        self.skirt = bpy.data.objects[mmd_model.SKIRT_NAME]

    def add_mesh(self, name, z_min, z_max, material=None):
        """アーマチュアの子の、小さな四面体のメッシュ（z_min〜z_max）"""
        data = bpy.data.meshes.new(name)
        data.from_pydata([(0.0, -0.05, z_min), (0.05, 0.05, z_min), (-0.05, 0.05, z_min), (0.0, 0.0, z_max)], [],
                         [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)])
        obj = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(obj)
        obj.parent = self.armature
        if material is not None:
            data.materials.append(bpy.data.materials.new(material))
        bpy.context.view_layer.update()
        return obj

    def add_bone_child(self, name, bone_name):
        """骨を親にした（parent_type BONE）小さな四面体のメッシュ。髪飾りなどを骨に付けたモデルに似せる。
        親の逆行列と、オブジェクト自身の位置・回転・大きさも、わざと単位行列でなくする"""
        data = bpy.data.meshes.new(name)
        data.from_pydata([(0.0, -0.05, 0.0), (0.05, 0.05, 0.0), (-0.05, 0.05, 0.0), (0.0, 0.0, 0.1)], [],
                         [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)])
        obj = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(obj)
        obj.parent = self.armature
        obj.parent_type = "BONE"
        obj.parent_bone = bone_name
        obj.matrix_parent_inverse = Matrix.Rotation(0.3, 4, "X") @ Matrix.Translation((0.01, 0.02, 0.03))
        obj.location = (0.02, 0.0, 0.05)
        obj.rotation_euler = (0.1, 0.2, 0.3)
        obj.scale = (1.0, 1.2, 0.9)
        bpy.context.view_layer.update()
        return obj

    def edit_bone(self, name, change):
        """エディットモードで骨を変えて、オブジェクトモードに戻す"""
        previous = bpy.context.view_layer.objects.active
        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            change(self.armature.data.edit_bones[name])
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")
            bpy.context.view_layer.objects.active = previous


# MARK: - 作る


class BuildCardTest(CardCase):
    def test_values(self):
        card = build_card()
        self.assertIsInstance(card, ModelCard)
        self.assertEqual(card.armature_name, mmd_model.ARMATURE_NAME)
        self.assertEqual(card.blend_name, "")  # 保存していないファイル
        self.assertEqual(card.tool_version, provenance.addon_version())
        self.assertEqual(card.blender_version, bpy.app.version_string)
        self.assertAlmostEqual(card.value(HEIGHT), mmd_model.REFERENCE_HEIGHT, delta=1e-6)
        self.assertEqual(card.value(ARM_REST_DOWN["L"]), mmd_model.ARM_DOWN_DEG)
        self.assertEqual(card.value(ARM_REST_DOWN["R"]), mmd_model.ARM_DOWN_DEG)
        self.assertAlmostEqual(card.value(LEG_LENGTH["L"]), leg_length(), delta=1e-6)
        self.assertAlmostEqual(card.value(LEG_LENGTH["R"]), leg_length(), delta=1e-6)
        self.assertAlmostEqual(card.value(SHOULDER_WIDTH), 2 * mmd_model.SHOULDER_JOINT[0], delta=1e-6)
        # 頭身の目安：1.45 ÷ (1.45 − 頭 の根元 1.25)
        self.assertAlmostEqual(card.value(HEAD_RATIO), 1.45 / (1.45 - 1.25), delta=1e-4)
        self.assertEqual(card.measurements["body_meshes"], ["Body"])
        self.assertEqual(card.measurements["body_meshes_source"], "kind")
        # 骨の名前はすべて準標準
        mapping = card.bone_mapping
        self.assertTrue(mapping["semi_standard_compliant"])
        self.assertEqual(mapping["unknown"], [])
        self.assertEqual(len(mapping["mapping"]), self.info["bone_count"])
        self.assertEqual(card.measurements["missing_joints"], [])

    def test_parts(self):
        card = build_card()
        self.assertEqual([part.object_name for part in card.parts], ["Body", "Skirt"])
        body, skirt = card.parts
        self.assertEqual((body.kind, body.kind_source, body.cloth_type, body.matched_keyword),
                         ("body", "guess", None, "body"))
        self.assertEqual((skirt.kind, skirt.kind_source, skirt.cloth_type, skirt.matched_keyword),
                         ("cloth", "guess", None, "skirt"))
        self.assertEqual(body.vertex_count, self.info["vertex_counts"]["Body"])
        self.assertEqual(skirt.vertex_count, self.info["vertex_counts"]["Skirt"])
        self.assertEqual(body.triangle_count, len(self.body.data.loop_triangles))
        self.assertEqual(skirt.triangle_count, len(self.skirt.data.loop_triangles))
        # 服の種類は、ユーザーが選ぶまで None
        chosen = set_part_kind(card, "Skirt", "cloth", "skirt")
        self.assertEqual(chosen.part("Skirt").cloth_type, "skirt")
        self.assertTrue(chosen.part("Skirt").is_confirmed)

    def test_joints_are_world_positions(self):
        card = build_card()
        joints = card.measurements["joints"]
        np.testing.assert_allclose(joints["左腕"], mmd_model.SHOULDER_JOINT, atol=1e-6)
        np.testing.assert_allclose(joints["右腕"], (-mmd_model.SHOULDER_JOINT[0], *mmd_model.SHOULDER_JOINT[1:]),
                                   atol=1e-6)
        np.testing.assert_allclose(joints["左ひざ"], mmd_model.KNEE_JOINT, atol=1e-6)
        rest = bone_rest(self.armature, space="world")
        for name in ("頭", "左足D", "右足首", "左手首"):
            np.testing.assert_allclose(joints[name], rest.heads[rest.index(name)], atol=0.0, err_msg=name)

    def test_moved_armature(self):
        """アーマチュアを動かす（Z まわりに 90 度・(1, 2, 3) へ）：関節は世界の座標、背の高さ・角度・長さは同じ"""
        before = build_card()
        self.armature.location = (1.0, 2.0, 3.0)
        self.armature.rotation_mode = "XYZ"
        self.armature.rotation_euler = (0.0, 0.0, math.radians(90.0))
        bpy.context.view_layer.update()
        card = build_card()
        x, y, z = mmd_model.SHOULDER_JOINT
        np.testing.assert_allclose(card.measurements["joints"]["左腕"], (1.0 - y, 2.0 + x, 3.0 + z), atol=1e-6)
        for name in (HEIGHT, LEG_LENGTH["L"], SHOULDER_WIDTH, HEAD_RATIO):
            self.assertAlmostEqual(card.value(name), before.value(name), delta=1e-5, msg=name)
        self.assertEqual(card.value(ARM_REST_DOWN["L"]), before.value(ARM_REST_DOWN["L"]))
        self.assertAlmostEqual(card.measurements["mesh_z_range"]["Body"][0], 3.0, delta=1e-6)

    def test_material_guess_and_body_only_height(self):
        self.add_mesh("Mesh.001", 1.0, 1.2, material="髪")
        self.add_mesh("Hat", 1.40, 1.60)
        card = build_card()
        self.assertEqual([part.object_name for part in card.parts], ["Body", "Hat", "Mesh.001", "Skirt"])
        self.assertEqual(card.part("Mesh.001").kind, "hair")
        self.assertEqual(card.part("Mesh.001").material_names, ("髪",))
        self.assertEqual(card.part("Hat").kind, "cloth")
        # 帽子は体ではないので、背の高さに入らない
        self.assertAlmostEqual(card.value(HEIGHT), 1.45, delta=1e-6)
        # ユーザーが帽子を体にすると、入る
        self.assertAlmostEqual(set_part_kind(card, "Hat", "body").value(HEIGHT), 1.60, delta=1e-6)

    def test_blend_name(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.blend"
            bpy.ops.wm.save_as_mainfile(filepath=str(path))
            self.assertEqual(build_card().blend_name, "model.blend")

    def test_previous_choices_are_kept(self):
        first = set_source(set_part_kind(build_card(), "Skirt", "cloth", "skirt"), origin="自作", terms="自由")
        second = build_card(previous=first)
        self.assertEqual(second.part("Skirt").cloth_type, "skirt")
        self.assertEqual(second.part("Skirt").kind_source, "user")
        self.assertEqual(second.source["terms"], "自由")

    def test_state_is_unchanged(self):
        """読むだけ：フレーム・モード・アクティブ・選択・ポーズを変えない。測るのは休みの姿勢"""
        bpy.context.scene.frame_set(7)
        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.mode_set(mode="POSE")
        try:
            half = math.radians(30.0)
            set_pose(self.armature, {"左腕": (math.cos(half), 0.0, math.sin(half), 0.0)})
            self.skirt.select_set(True)
            self.body.select_set(False)
            pose_before = tuple(self.armature.pose.bones["左腕"].rotation_quaternion)
            card = build_card()
            self.assertEqual(bpy.context.scene.frame_current, 7)
            self.assertEqual(self.armature.mode, "POSE")
            self.assertIs(bpy.context.view_layer.objects.active, self.armature)
            self.assertTrue(self.skirt.select_get())
            self.assertFalse(self.body.select_get())
            self.assertEqual(tuple(self.armature.pose.bones["左腕"].rotation_quaternion), pose_before)
            self.assertEqual(card.value(ARM_REST_DOWN["L"]), mmd_model.ARM_DOWN_DEG)
            self.assertNotIn(TEXT_NAME, bpy.data.texts)  # 作るだけ。置かない
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")

    def test_other_scene_and_errors(self):
        other = bpy.data.scenes.new("Other")
        with self.assertRaises(ValueError):
            build_card(other)
        build_card(bpy.context.scene)  # いまのシーンはよい
        bpy.ops.wm.read_homefile(use_empty=True)
        with self.assertRaisesRegex(ValueError, "アーマチュアがありません"):
            build_card()


class BoneParentedMeshTest(CardCase):
    """骨を親にしたメッシュ（parent_type BONE。髪飾りなど）は、ポーズで世界の位置が動く。
    カードは休みの姿勢で測るので、ポーズを付けても値と指紋は変わらない（測り直しにならない）"""

    def world_vertices(self, obj):
        """Blender が計算した、いまの世界の座標の頂点（モディファイアなし）"""
        matrix = np.array(obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world, dtype=np.float64)
        local = np.array([vertex.co for vertex in obj.data.vertices], dtype=np.float64)
        return local @ matrix[:3, :3].T + matrix[:3, 3]

    def check_rest_and_posed(self, relative):
        self.armature.data.bones["左ひじ"].use_relative_parent = relative
        ribbon = self.add_bone_child("Ribbon", "左ひじ")
        self.armature.location = (0.3, -0.2, 0.1)
        self.armature.rotation_euler = (0.0, 0.0, 0.4)
        bpy.context.view_layer.update()
        rest_world = self.world_vertices(ribbon)
        card = build_card()
        save_card(card)
        # 休みの姿勢では、Blender が計算した位置と同じ（float32 の誤差の中）
        np.testing.assert_allclose(card.measurements["mesh_z_range"]["Ribbon"],
                                   [rest_world[:, 2].min(), rest_world[:, 2].max()], atol=1e-6)
        half = math.radians(25.0)
        set_pose(self.armature, {"左腕": (math.cos(half), 0.0, math.sin(half), 0.0),
                                 "左ひじ": (math.cos(half), math.sin(half), 0.0, 0.0)})
        bpy.context.view_layer.update()
        self.assertGreater(np.abs(self.world_vertices(ribbon) - rest_world).max(), 0.05)  # ほんとうに動いた
        pose_before = tuple(self.armature.pose.bones["左ひじ"].rotation_quaternion)
        self.assertEqual(check_card()[1], [])
        posed = build_card()
        self.assertEqual(posed.measurements["mesh_z_range"], card.measurements["mesh_z_range"])
        self.assertEqual(posed.fingerprints, card.fingerprints)
        self.assertEqual(tuple(self.armature.pose.bones["左ひじ"].rotation_quaternion), pose_before)
        # ポーズを付けたままでも、手で直した物は見つける
        ribbon.location.z += 0.001
        bpy.context.view_layer.update()
        self.assertEqual(check_card()[1], ["Ribbon の形が変わりました"])

    def test_bone_parent(self):
        self.check_rest_and_posed(relative=False)

    def test_relative_bone_parent(self):
        self.check_rest_and_posed(relative=True)

    def test_other_parent_bone_is_a_change(self):
        ribbon = self.add_bone_child("Ribbon", "左ひじ")
        save_card(build_card())
        ribbon.parent_bone = "右ひじ"
        bpy.context.view_layer.update()
        self.assertEqual(check_card()[1], ["Ribbon の形が変わりました"])


# MARK: - 置く・読む


class SaveLoadTest(CardCase):
    def test_round_trip(self):
        self.assertIsNone(load_card())
        card = build_card()
        name = save_card(card)
        self.assertEqual(name, TEXT_NAME)
        text = bpy.data.texts[TEXT_NAME]
        self.assertTrue(text.use_fake_user)
        self.assertEqual(text.as_string(), card.to_json() + "\n")
        self.assertEqual(load_card(), card)
        # もう一度置くと、中身を置きかえる（テキストは1つのまま）
        changed = set_part_kind(card, "Skirt", "cloth", "skirt")
        save_card(changed)
        self.assertEqual(len([t for t in bpy.data.texts if t.name.startswith("ai_tools_card")]), 1)
        self.assertEqual(load_card(), changed)
        self.assertNotEqual(load_card(), card)

    def test_does_not_save_the_blend_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.blend"
            bpy.ops.wm.save_as_mainfile(filepath=str(path))
            before = path.read_bytes()
            save_card(build_card())
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()), ["model.blend"])
            self.assertEqual(Path(bpy.data.filepath), path)

    def test_card_is_kept_in_the_file(self):
        card = set_source(build_card(), terms="改変可・再配布不可")
        save_card(card)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "copy.blend"
            bpy.ops.wm.save_as_mainfile(filepath=str(path), copy=True)
            bpy.ops.wm.open_mainfile(filepath=str(path))
            self.assertEqual(load_card(), card)
            # 開き直したファイルでも、何も変えていなければ測り直しは要らない
            loaded, reasons = check_card()
            self.assertEqual(reasons, [])
            self.assertEqual(loaded, card)

    def test_corrupt_text(self):
        bpy.data.texts.new(TEXT_NAME).write("{ not json")
        with self.assertRaises(ValueError):
            load_card()
        card, reasons = check_card()
        self.assertIsNone(card)
        self.assertEqual(len(reasons), 1)
        self.assertTrue(reasons[0].startswith("登録カードが読めません"))
        save_card(build_card())  # 壊れたカードは、置き直せる
        self.assertIsNotNone(load_card())

    def test_save_rejects_other_objects(self):
        with self.assertRaises(ValueError):
            save_card(build_card().to_json())
        self.assertNotIn(TEXT_NAME, bpy.data.texts)

    def test_deeply_nested_text(self):
        """だれかが書いた深い入れ子のテキストでも、止まらずに「読めません」を返す"""
        bpy.data.texts.new(TEXT_NAME).write("[" * 100000)
        with self.assertRaises(ValueError):
            load_card()
        card, reasons = check_card()
        self.assertIsNone(card)
        self.assertEqual(len(reasons), 1)
        self.assertTrue(reasons[0].startswith("登録カードが読めません"))

    def test_save_refuses_a_card_that_cannot_be_read_back(self):
        """作った後に中身を書きかえたカード（同じ部品が2つ）は置かない。置いてあったカードはそのまま"""
        card = build_card()
        save_card(card)
        broken = build_card()
        broken.parts.append(broken.parts[0])
        with self.assertRaises(ValueError):
            save_card(broken)
        self.assertEqual(load_card(), card)


# MARK: - 確かめる


class CheckCardTest(CardCase):
    def setUp(self):
        super().setUp()
        self.card = build_card()
        save_card(self.card)

    def test_no_card(self):
        bpy.data.texts.remove(bpy.data.texts[TEXT_NAME])
        self.assertEqual(check_card(), (None, ["登録カードがありません。先にモデルを登録してください"]))

    def test_unchanged(self):
        card, reasons = check_card()
        self.assertEqual(card, self.card)
        self.assertEqual(reasons, [])
        signals = {m.name: m for m in summarize_card(card, reasons)}
        self.assertEqual(signals["card_current"].value, 0.0)

    def test_moved_vertex_in_basis(self):
        """シェイプキーのあるメッシュ：基本の形（Basis）の頂点を 1 mm 動かす"""
        mesh = self.body.data
        basis = mesh.shape_keys.reference_key
        co = basis.data[0].co
        basis.data[0].co = (co[0], co[1], co[2] + 0.001)
        mesh.vertices[0].co = basis.data[0].co
        mesh.update()
        _, reasons = check_card()
        self.assertEqual(reasons, ["Body の形が変わりました"])

    def test_moved_vertex_without_shape_keys(self):
        bpy.ops.wm.read_homefile(use_empty=True)
        mmd_model.build_model(with_shape_keys=False)
        save_card(build_card())
        skirt = bpy.data.objects["Skirt"]
        co = skirt.data.vertices[5].co
        skirt.data.vertices[5].co = (co[0] + 0.001, co[1], co[2])
        skirt.data.update()
        _, reasons = check_card()
        self.assertEqual(reasons, ["Skirt の形が変わりました"])

    def test_moved_vertex_in_edit_mode(self):
        """エディットモードで動かした物も（データに写してから）見つける。モードは変えない"""
        import bmesh

        bpy.context.view_layer.objects.active = self.skirt
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            edit = bmesh.from_edit_mesh(self.skirt.data)
            edit.verts.ensure_lookup_table()
            edit.verts[3].co.z += 0.002
            bmesh.update_edit_mesh(self.skirt.data)
            _, reasons = check_card()
            self.assertEqual(reasons, ["Skirt の形が変わりました"])
            self.assertEqual(self.skirt.mode, "EDIT")
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")

    def test_expression_key_is_not_a_shape_change(self):
        """表情のシェイプキー（あ）を変えても、休みの形は変わらない"""
        key = self.body.data.shape_keys.key_blocks["あ"]
        co = key.data[0].co
        key.data[0].co = (co[0], co[1], co[2] + 0.01)
        key.value = 0.5
        self.assertEqual(check_card()[1], [])

    def test_tiny_noise_is_not_a_change(self):
        """再現性の幅（0.01 mm）よりずっと小さい差は変化にしない。

        Blender の座標は float32 なので、いちばん小さい差（1 つ隣の float32。z が 1 m 前後で約 1e-7 m）で動かす。
        そろえる境い目の近くの点は、小さな差でも変わることがある（安全な向き）ので、そういう点は動かさない。
        """
        quantum = thresholds.REPRO_TOLERANCE_LENGTH
        basis = self.body.data.shape_keys.reference_key
        moved = 0
        for index in range(0, len(basis.data), 25):
            x, y, z = basis.data[index].co
            bumped = float(np.nextafter(np.float32(z), np.float32(np.inf)))
            if any(abs((c / quantum) % 1.0 - 0.5) < 0.1 for c in (x, y, z, bumped)):
                continue
            basis.data[index].co = (x, y, bumped)
            self.assertNotEqual(basis.data[index].co[2], z)  # ほんとうに動いた
            self.assertLess(abs(basis.data[index].co[2] - z), quantum / 10.0)
            moved += 1
        self.body.data.update()
        self.assertGreater(moved, 10)
        self.assertEqual(check_card()[1], [])

    def test_bone_changes(self):
        cases = {
            "head": lambda bone: setattr(bone, "head", (bone.head[0], bone.head[1], bone.head[2] + 0.001)),
            "roll": lambda bone: setattr(bone, "roll", bone.roll + 0.3),
        }
        for label, change in cases.items():
            with self.subTest(change=label):
                save_card(build_card())
                self.edit_bone("左ひじ", change)
                _, reasons = check_card()
                self.assertEqual(reasons[0], BONES_CHANGED)
                # 骨の根元を動かしても、メッシュの元の形は変わらない
                self.assertEqual(reasons, [BONES_CHANGED])

    def test_deform_flag_and_name(self):
        self.armature.data.bones["左腕捩1"].use_deform = True
        self.assertEqual(check_card()[1], [BONES_CHANGED])
        save_card(build_card())
        self.armature.data.bones["左腕捩1"].name = "左腕捩X"
        self.assertEqual(check_card()[1], [BONES_CHANGED])

    def test_moved_armature_needs_remeasure(self):
        """カードの位置は世界の座標なので、モデルを動かしただけでも測り直す"""
        self.armature.location = (0.0, 0.5, 0.0)
        bpy.context.view_layer.update()
        self.assertEqual(check_card()[1], [BONES_CHANGED, "Body の形が変わりました", "Skirt の形が変わりました"])

    def test_meshes_added_removed_and_armature_renamed(self):
        self.add_mesh("Ribbon", 1.3, 1.35)
        bpy.data.objects.remove(self.skirt)
        self.armature.name = "Armature2"
        _, reasons = check_card()
        self.assertEqual(reasons, ["アーマチュアが変わりました（TestArmature → Armature2）", "Skirt がなくなりました",
                                   "Ribbon が増えました"])

    def test_unreadable_model(self):
        self.body.hide_viewport = True  # 計算に入らないので、形を読めない
        bpy.context.view_layer.update()
        card, reasons = check_card()
        self.assertEqual(card, self.card)
        self.assertEqual(len(reasons), 1)
        self.assertTrue(reasons[0].startswith("いまのモデルを読めません"))

    def test_summary_of_stale_card(self):
        self.armature.data.bones["左腕捩1"].use_deform = True
        card, reasons = check_card()
        signals = {m.name: m for m in summarize_card(card, reasons)}
        self.assertEqual(signals["card_current"].signal.label, "灰")
        self.assertIn(BONES_CHANGED, signals["card_current"].reason)


if __name__ == "__main__":
    unittest.main()

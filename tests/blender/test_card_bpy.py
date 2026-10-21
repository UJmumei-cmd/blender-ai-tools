"""A1 モデルの登録カードを Blender の中で作る・置く・確かめる所（model/card_bpy.py）を確かめる。

テスト用のモデル（tests/fixtures/mmd_model.py）を空のシーンに作ってから調べる。答えは、モデルを作るときの
決まった数字（背の高さ 1.45・腕の下がり角 37.5 度・関節の位置）から、手で計算した値。
ファイルは一時フォルダの中だけに作る。本物の書き出し（convert_blend.py）と比べるテストは、環境変数
BLENDER_AI_TOOLS_KOUMAKAN_REPO（紅魔館のリポジトリの場所）が無いと、理由を出して飛ばす。
"""

import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

from blender_ai_tools import provenance, thresholds
from blender_ai_tools.export import convert
from blender_ai_tools.model import card_bpy, extract_bpy
from blender_ai_tools.model.card import (
    ARM_REST_DOWN, EXTRA_ANSWER, HEAD_RATIO, HEIGHT, LEG_LENGTH, SHOULDER_WIDTH, ModelCard, bones_fingerprint,
    card_text, set_bone_answer, set_material_kind, set_part_kind, set_source, shape_fingerprint, summarize_card,
)
from blender_ai_tools.model.card_bpy import (
    TEXT_NAME, CardCheck, build_card, build_card_with_notes, check_card, current_fingerprints, load_card, save_card,
)
from blender_ai_tools.model.extract_bpy import bone_rest, set_pose
from tests.fixtures import mmd_model

BONES_CHANGED = "骨（名前・並び・親子・位置・ねじれ・形を動かすかどうか）が変わりました"
MOVED_NOTE = "置き場所が変わりました（形はそのまま）"
RESIZED = "モデルの大きさか傾き（アーマチュアの置き方）が変わりました。背の高さと長さを測り直してください"
KOUMAKAN_ENV = "BLENDER_AI_TOOLS_KOUMAKAN_REPO"
UNKNOWN_BONE = "謎の骨"


def leg_length():
    """左脚の長さ：|足 − ひざ| + |ひざ − 足首|（モデルを作るときの関節の位置から）"""
    hip, knee, ankle = (np.array(p) for p in (mmd_model.HIP_JOINT, mmd_model.KNEE_JOINT, mmd_model.ANKLE_JOINT))
    return float(np.linalg.norm(knee - hip) + np.linalg.norm(ankle - knee))


def converter_angle(armature):
    """convert_pmx.rest_arm_angle と同じ書き方（mathutils の Vector で引き算）で、左腕 → 左ひじ の下がり角"""
    bones = armature.data.bones
    direction = bones["左ひじ"].head_local - bones["左腕"].head_local
    horizontal = (direction.x ** 2 + direction.y ** 2) ** 0.5
    return round(float(np.degrees(np.arctan2(-direction.z, horizontal))), 1)


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

    def edit_bones(self, change):
        """エディットモードで骨を変えて（change(edit_bones)）、オブジェクトモードに戻す"""
        previous = bpy.context.view_layer.objects.active
        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            change(self.armature.data.edit_bones)
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")
            bpy.context.view_layer.objects.active = previous

    def edit_bone(self, name, change):
        self.edit_bones(lambda bones: change(bones[name]))

    def add_unknown_bone(self, name=UNKNOWN_BONE, head=(0.0, 0.05, 0.9)):
        """辞書に無い骨（上半身の子）を足す"""
        def add(bones):
            bone = bones.new(name)
            bone.head = head
            bone.tail = (head[0], head[1], head[2] + 0.05)
            bone.parent = bones["上半身"]
        self.edit_bones(add)

    def move_armature(self, location=(1.0, 2.0, 3.0), rotation=(0.0, 0.0, math.radians(90.0)), scale=(1.0, 1.0, 1.0)):
        self.armature.location = location
        self.armature.rotation_mode = "XYZ"
        self.armature.rotation_euler = rotation
        self.armature.scale = scale
        bpy.context.view_layer.update()

    def give_body_hair_material(self):
        """Body に材質 肌（0）と 髪（1）を付け、頭の部品の面を 髪 にする。体の材質の三角形の高さの範囲を返す"""
        data = self.body.data
        data.materials.append(bpy.data.materials.new("肌"))
        data.materials.append(bpy.data.materials.new("髪"))
        start, end = self.info["parts"]["Body"]["head"]
        for polygon in data.polygons:
            if all(start <= index < end for index in polygon.vertices):
                polygon.material_index = 1
        data.update()
        bpy.context.view_layer.update()
        heights = []
        for triangle in data.loop_triangles:
            if triangle.material_index == 0:
                heights += [float(np.float32(data.vertices[index].co[2])) for index in triangle.vertices]
        return min(heights), max(heights)


# MARK: - 作る


class BuildCardTest(CardCase):
    def test_values(self):
        card = build_card()
        self.assertIsInstance(card, ModelCard)
        self.assertEqual(card.schema_version, 2)
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
        # 置き方・骨の位置
        np.testing.assert_array_equal(card.placement_matrix(), np.array(self.armature.matrix_world))
        rest = bone_rest(self.armature, space="armature")
        self.assertEqual(card.skeleton, {"names": list(rest.names), "heads": rest.heads.tolist()})

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
        self.assertEqual((body.material_names, body.materials), ((), ()))
        # 服の種類は、ユーザーが選ぶまで None
        chosen = set_part_kind(card, "Skirt", "cloth", "skirt")
        self.assertEqual(chosen.part("Skirt").cloth_type, "skirt")
        self.assertTrue(chosen.part("Skirt").is_confirmed)

    def test_joints_are_armature_positions(self):
        card = build_card()
        joints = card.measurements["joints"]
        self.assertEqual(card.measurements["space"], "armature")
        np.testing.assert_allclose(joints["左腕"], mmd_model.SHOULDER_JOINT, atol=1e-6)
        np.testing.assert_allclose(joints["右腕"], (-mmd_model.SHOULDER_JOINT[0], *mmd_model.SHOULDER_JOINT[1:]),
                                   atol=1e-6)
        np.testing.assert_allclose(joints["左ひざ"], mmd_model.KNEE_JOINT, atol=1e-6)
        rest = bone_rest(self.armature, space="armature")
        for name in ("頭", "左足D", "右足首", "左手首"):
            np.testing.assert_allclose(joints[name], rest.heads[rest.index(name)], atol=0.0, err_msg=name)

    def test_moved_armature(self):
        """アーマチュアを動かす（Z まわりに 90 度・(1, 2, 3) へ）：関節・指紋・値は同じ。置き方だけが変わる"""
        before = build_card()
        self.move_armature()
        card = build_card()
        self.assertEqual(card.measurements["joints"], before.measurements["joints"])
        self.assertEqual(card.fingerprints, before.fingerprints)
        for name in (HEIGHT, LEG_LENGTH["L"], SHOULDER_WIDTH, HEAD_RATIO, ARM_REST_DOWN["L"]):
            self.assertAlmostEqual(card.value(name), before.value(name), delta=1e-6, msg=name)
        self.assertAlmostEqual(card.measurements["mesh_up_range"]["Body"][0], 0.0, delta=1e-6)
        np.testing.assert_allclose(card.placement_matrix()[:3, 3], (1.0, 2.0, 3.0), atol=1e-6)
        np.testing.assert_array_equal(card.placement_matrix(), np.array(self.armature.matrix_world))

    def test_rotated_fbx_style_armature(self):
        """Y が上の向きで骨とメッシュを持ち、X まわりに 90 度回して立てたアーマチュア（FBX の読み込みに似せる）。
        背の高さは世界の上向きで測るので同じ。腕の下がり角は、convert_pmx と同じくアーマチュアの中で測る"""
        before = build_card()
        to_y_up = Matrix.Rotation(math.radians(-90.0), 4, "X")  # 世界の (x, y, z) → (x, z, -y)
        for obj in (self.body, self.skirt):
            obj.data.transform(to_y_up)
            if obj.data.shape_keys is not None:
                for key in obj.data.shape_keys.key_blocks:
                    for point in key.data:
                        point.co = to_y_up @ point.co
        self.armature.data.transform(to_y_up)
        self.move_armature(location=(0.0, 0.0, 0.0), rotation=(math.radians(90.0), 0.0, 0.0))
        card = build_card()
        self.assertAlmostEqual(card.value(HEIGHT), before.value(HEIGHT), delta=1e-6)
        self.assertAlmostEqual(card.value(HEAD_RATIO), before.value(HEAD_RATIO), delta=1e-4)
        self.assertAlmostEqual(card.value(LEG_LENGTH["L"]), before.value(LEG_LENGTH["L"]), delta=1e-6)
        self.assertEqual(card.value(ARM_REST_DOWN["L"]), converter_angle(self.armature))
        self.assertNotEqual(card.value(ARM_REST_DOWN["L"]), mmd_model.ARM_DOWN_DEG)  # 世界の向きで測っていない

    def test_scaled_armature_gives_world_lengths(self):
        """アーマチュアの大きさを 2 倍にすると、世界の長さも 2 倍。腕の下がり角は変わらない"""
        before = build_card()
        self.move_armature(location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=(2.0, 2.0, 2.0))
        card = build_card()
        for name in (HEIGHT, LEG_LENGTH["L"], SHOULDER_WIDTH):
            self.assertAlmostEqual(card.value(name), 2.0 * before.value(name), delta=1e-6, msg=name)
        self.assertAlmostEqual(card.value(HEAD_RATIO), before.value(HEAD_RATIO), delta=1e-6)
        self.assertEqual(card.value(ARM_REST_DOWN["L"]), before.value(ARM_REST_DOWN["L"]))
        self.assertEqual(card.fingerprints, before.fingerprints)

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

    def test_per_material_height_excludes_hair(self):
        """1つのメッシュに 肌 と 髪 の材質（MMD のモデルに似せる）：髪の三角形を外して背の高さを測る"""
        low, high = self.give_body_hair_material()
        self.assertLess(high, 1.45 - 0.05)  # 頭を外すと、ほんとうに低くなる
        card = build_card()
        body = card.part("Body")
        self.assertEqual(body.material_names, ("肌", "髪"))
        self.assertEqual([(m.name, m.kind) for m in body.materials], [("肌", "body"), ("髪", "hair")])
        self.assertEqual(card.measurements["excluded_materials"], {"Body": ["髪"]})
        self.assertAlmostEqual(card.value(HEIGHT), high - low, delta=1e-6)
        self.assertAlmostEqual(card.measurements["material_up_range"]["Body"]["髪"][1], 1.45, delta=1e-6)
        # 髪も体にすると、頭のてっぺんまで
        both = set_material_kind(card, "Body", "髪", "body")
        self.assertAlmostEqual(both.value(HEIGHT), 1.45, delta=1e-6)
        # 材質の数え方：部品 2 つ（推測のまま）と、材質 2 つ
        unconfirmed = {m.name: m for m in summarize_card(card)}["unconfirmed_parts"]
        self.assertEqual(unconfirmed.value, 4.0)

    def test_empty_material_slot(self):
        """空の材質の枠と、枠の外の番号の面は「材質の無い三角形」（外さない）"""
        data = self.body.data
        data.materials.append(bpy.data.materials.new("肌"))
        data.materials.append(None)
        data.materials.append(bpy.data.materials.new("髪"))
        start, end = self.info["parts"]["Body"]["head"]
        for polygon in data.polygons:
            if all(start <= index < end for index in polygon.vertices):
                polygon.material_index = 2
        data.polygons[0].material_index = 1
        data.update()
        card = build_card()
        body = card.part("Body")
        self.assertEqual(body.material_names, ("肌", "髪"))
        self.assertIn("", card.measurements["material_up_range"]["Body"])
        self.assertEqual(card.measurements["excluded_materials"], {"Body": ["髪"]})

    def test_blend_name(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.blend"
            bpy.ops.wm.save_as_mainfile(filepath=str(path))
            self.assertEqual(build_card().blend_name, "model.blend")

    def test_previous_choices_are_kept(self):
        first = set_source(set_part_kind(build_card(), "Skirt", "cloth", "skirt"), origin="自作", terms="自由")
        second, notes = build_card_with_notes(previous=first)
        self.assertEqual(notes, [])
        self.assertEqual(second.part("Skirt").cloth_type, "skirt")
        self.assertEqual(second.part("Skirt").kind_source, "user")
        self.assertEqual(second.source["terms"], "自由")
        self.assertEqual(build_card(previous=first), second)

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
            check_card()
            self.assertEqual((bpy.context.scene.frame_current, self.armature.mode), (7, "POSE"))
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


class ArmAngleTest(CardCase):
    """腕の下がり角は、convert_pmx.rest_arm_angle（アーマチュアの中の head_local・mathutils の float32）と同じ値"""

    def tilt_and_bend(self):
        """左ひじの根元を少し動かし（37.5 度でない角度にする）、アーマチュアを傾けて置く"""
        self.edit_bone("左ひじ", lambda bone: setattr(bone, "head", (bone.head[0], bone.head[1] - 0.0123,
                                                                    bone.head[2] - 0.0171)))
        self.move_armature(location=(0.4, -0.3, 0.2), rotation=(math.radians(15.0), math.radians(20.0), 0.3))

    def test_equals_converter_formula(self):
        self.tilt_and_bend()
        card = build_card()
        expected = converter_angle(self.armature)
        self.assertNotEqual(expected, mmd_model.ARM_DOWN_DEG)
        self.assertEqual(card.value(ARM_REST_DOWN["L"]), expected)
        # 世界の向きで測ると違う角度になる（傾けて置いたので）
        world = bone_rest(self.armature, space="world")
        direction = world.heads[world.index("左ひじ")] - world.heads[world.index("左腕")]
        world_angle = round(math.degrees(math.atan2(-direction[2], math.hypot(direction[0], direction[1]))), 1)
        self.assertNotEqual(world_angle, expected)

    def test_equals_converter_output(self):
        """本物の convert_blend.py の character_rig.json の restArmDownDegrees と同じ"""
        repo = os.environ.get(KOUMAKAN_ENV)
        if not repo:
            self.skipTest(f"環境変数 {KOUMAKAN_ENV}（紅魔館のリポジトリの場所）が無いため")
        self.tilt_and_bend()
        card = build_card()
        with tempfile.TemporaryDirectory() as folder:
            source = mmd_model.save_model(Path(folder) / "model.blend")
            result = convert.run_convert(source, Path(folder) / "out", blender=bpy.app.binary_path, repo=repo)
            self.assertTrue(result.ok, result.problems + [result.stdout, result.stderr])
        self.assertEqual(card.value(ARM_REST_DOWN["L"]), result.rig["restArmDownDegrees"])
        self.assertEqual(result.rig["restArmDownDegrees"], converter_angle(self.armature))


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
        # 休みの姿勢では、Blender が計算した位置と同じ（世界の上向きの高さ。アーマチュアの原点の高さ 0.1 を引く）
        np.testing.assert_allclose(card.measurements["mesh_up_range"]["Ribbon"],
                                   [rest_world[:, 2].min() - 0.1, rest_world[:, 2].max() - 0.1], atol=1e-6)
        half = math.radians(25.0)
        set_pose(self.armature, {"左腕": (math.cos(half), 0.0, math.sin(half), 0.0),
                                 "左ひじ": (math.cos(half), math.sin(half), 0.0, 0.0)})
        bpy.context.view_layer.update()
        self.assertGreater(np.abs(self.world_vertices(ribbon) - rest_world).max(), 0.05)  # ほんとうに動いた
        pose_before = tuple(self.armature.pose.bones["左ひじ"].rotation_quaternion)
        self.assertEqual(check_card()[1], [])
        posed = build_card()
        self.assertEqual(posed.measurements["mesh_up_range"], card.measurements["mesh_up_range"])
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
        check = check_card()
        self.assertIsInstance(check, CardCheck)
        card, reasons = check
        self.assertEqual(card, self.card)
        self.assertEqual(reasons, [])
        self.assertEqual((check.card, check.reasons, check.notes), (self.card, [], []))
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

    def test_moved_armature_gives_only_a_note(self):
        """モデル全体を動かす・Z まわりに回す：測り直しは要らず、置き場所が変わった知らせだけ"""
        self.move_armature(location=(0.0, 0.5, 0.0), rotation=(0.0, 0.0, math.radians(90.0)))
        check = check_card()
        self.assertEqual(check.reasons, [])
        self.assertEqual(check.notes, [MOVED_NOTE])
        self.assertEqual(check.notes, [card_text("card.placement_moved")])
        name, shapes, bones = current_fingerprints()
        self.assertEqual((name, shapes, bones),
                         (self.card.armature_name, self.card.fingerprints["shapes"], self.card.fingerprints["bones"]))
        # 元の置き場所に戻すと、知らせも消える
        self.move_armature(location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0))
        self.assertEqual(check_card().notes, [])

    def test_vertex_edit_still_needs_remeasure_after_moving(self):
        self.move_armature(location=(0.3, 0.0, 0.0), rotation=(0.0, 0.0, 0.7))
        co = self.skirt.data.vertices[5].co
        self.skirt.data.vertices[5].co = (co[0] + 0.001, co[1], co[2])
        self.skirt.data.update()
        check = check_card()
        self.assertEqual((check.reasons, check.notes), (["Skirt の形が変わりました"], [MOVED_NOTE]))

    def test_tilted_or_scaled_armature_needs_remeasure(self):
        # 縦横だけの大きさ（上向きの行はそのまま）も、世界の長さが変わるので測り直し
        for label, options in (("tilt", {"rotation": (math.radians(30.0), 0.0, 0.0)}),
                               ("scale", {"rotation": (0.0, 0.0, 0.0), "scale": (1.1, 1.1, 1.1)}),
                               ("x scale", {"rotation": (0.0, 0.0, 0.0), "scale": (1.5, 1.0, 1.0)})):
            with self.subTest(change=label):
                self.move_armature(location=(0.0, 0.0, 0.0), **options)
                check = check_card()
                self.assertEqual((check.reasons, check.notes), ([RESIZED], []))

    def test_moved_mesh_is_a_shape_change(self):
        """メッシュだけをアーマチュアに対して動かすのは、形の変化（アーマチュアの中の位置が変わる）"""
        self.skirt.location = (0.0, 0.0, 0.01)
        bpy.context.view_layer.update()
        check = check_card()
        self.assertEqual((check.reasons, check.notes), (["Skirt の形が変わりました"], []))

    def test_root_empty_parent(self):
        """mmd_tools の読み込みのように、アーマチュアが Empty（root）の子：root を動かす・Z まわりに回すのは知らせだけ。
        root を傾ける・大きさを変えるのは測り直し（世界の高さと長さが変わる）。指紋はどれも変わらない"""
        root = bpy.data.objects.new("root", None)
        bpy.context.scene.collection.objects.link(root)
        self.armature.parent = root
        bpy.context.view_layer.update()
        self.assertEqual((check_card().reasons, check_card().notes), ([], []))
        root.location = (2.0, -1.0, 0.5)
        root.rotation_euler = (0.0, 0.0, math.radians(135.0))
        bpy.context.view_layer.update()
        check = check_card()
        self.assertEqual((check.reasons, check.notes), ([], [MOVED_NOTE]))
        moved = build_card()
        self.assertEqual(moved.fingerprints, self.card.fingerprints)
        self.assertEqual(moved.measurements["joints"], self.card.measurements["joints"])
        np.testing.assert_array_equal(moved.placement_matrix(), np.array(self.armature.matrix_world))
        self.assertAlmostEqual(moved.value(HEIGHT), self.card.value(HEIGHT), delta=1e-6)
        self.assertAlmostEqual(moved.value(LEG_LENGTH["L"]), self.card.value(LEG_LENGTH["L"]), delta=1e-6)
        root.rotation_euler = (math.radians(20.0), 0.0, 0.0)
        bpy.context.view_layer.update()
        self.assertEqual((check_card().reasons, check_card().notes), ([RESIZED], []))
        root.rotation_euler = (0.0, 0.0, 0.0)
        root.scale = (0.08, 0.08, 0.08)  # MMD の大きさ（1 単位 ≒ 8 cm）を root で付けた置き方
        bpy.context.view_layer.update()
        self.assertEqual(check_card().reasons, [RESIZED])
        small = build_card()
        self.assertEqual(small.fingerprints, self.card.fingerprints)
        self.assertAlmostEqual(small.value(HEIGHT), 0.08 * self.card.value(HEIGHT), delta=1e-6)
        self.assertAlmostEqual(small.value(LEG_LENGTH["L"]), 0.08 * self.card.value(LEG_LENGTH["L"]), delta=1e-6)
        self.assertEqual(small.value(ARM_REST_DOWN["L"]), self.card.value(ARM_REST_DOWN["L"]))

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


# MARK: - 骨の答えと、版 1 のカード


class BoneAnswerTest(CardCase):
    def test_answers_survive_re_registration_of_the_same_model(self):
        self.add_unknown_bone()
        card = build_card()
        self.assertEqual(card.bone_mapping["unknown"], [UNKNOWN_BONE])
        answered = set_bone_answer(card, UNKNOWN_BONE, EXTRA_ANSWER)
        save_card(answered)
        again, notes = build_card_with_notes(previous=load_card())
        self.assertEqual(notes, [])
        self.assertEqual(again.bone_answers, {UNKNOWN_BONE: EXTRA_ANSWER})
        self.assertEqual(again.bone_mapping["unknown"], [])
        self.assertIn(UNKNOWN_BONE, again.bone_mapping["known_extras"])

    def test_answers_are_not_copied_to_another_model(self):
        self.add_unknown_bone()
        answered = set_source(set_bone_answer(build_card(), UNKNOWN_BONE, EXTRA_ANSWER), terms="個人利用のみ")
        note = card_text("card.choices_not_carried", armature=mmd_model.ARMATURE_NAME)
        # 骨を 1 mm 動かした（骨の指紋が違う）
        self.edit_bone("左ひじ", lambda bone: setattr(bone, "head", (bone.head[0], bone.head[1], bone.head[2] + 0.001)))
        other, notes = build_card_with_notes(previous=answered)
        self.assertEqual(notes, [note])
        self.assertEqual((other.bone_answers, other.source["terms"]), ({}, ""))
        self.assertEqual(other.bone_mapping["unknown"], [UNKNOWN_BONE])
        # アーマチュアの名前が違う
        self.armature.name = "OtherArmature"
        other, notes = build_card_with_notes(previous=answered)
        self.assertEqual(notes, [note])
        self.assertEqual(other.bone_answers, {})


def legacy_text_from_scene(card):
    """いまのシーンから、版 1 のカード（世界の座標の指紋と位置）の JSON を作る（前の版の card_bpy と同じ読み方）"""
    model = extract_bpy.find_model()
    rest = bone_rest(model.armature, space="world")
    shapes = {}
    z_range = {}
    for obj in model.meshes:
        vertices, triangles = extract_bpy.mesh_rest(obj, space="world")
        shapes[obj.name] = shape_fingerprint(obj.name, vertices, triangles)
        z_range[obj.name] = [float(vertices[:, 2].min()), float(vertices[:, 2].max())]
    data = json.loads(card.to_json())
    for key in ("placement", "skeleton", "bone_answers"):
        del data[key]
    for part in data["parts"]:
        del part["materials"]
    data["parts"][0].update(kind="body", kind_source="user")
    measurements = data["measurements"]
    data["measurements"] = {
        "space": "world",
        "joints": {name: [float(c) for c in rest.heads[rest.index(name)]] for name in measurements["joints"]},
        "missing_joints": measurements["missing_joints"],
        "mesh_z_range": z_range,
        "body_meshes": measurements["body_meshes"],
        "body_meshes_source": measurements["body_meshes_source"],
        "values": measurements["values"],
    }
    data["fingerprints"] = {"shapes": shapes, "bones": bones_fingerprint(rest.names, rest.parents, rest.heads,
                                                                         rest.tails, rest.deform, rest.matrices)}
    data["source"]["terms"] = "個人利用のみ"
    data["schema_version"] = 1
    return json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2)


class LegacyCardTest(CardCase):
    def setUp(self):
        super().setUp()
        self.move_armature(location=(0.2, 0.0, 0.0), rotation=(0.0, 0.0, 0.5))
        bpy.data.texts.new(TEXT_NAME).write(legacy_text_from_scene(build_card()))

    def test_check_asks_for_one_remeasure(self):
        card = load_card()
        self.assertTrue(card.is_legacy)
        check = check_card()
        self.assertEqual(check.card, card)
        self.assertEqual(check.reasons, [card_text("card.legacy_card", version="1")])
        self.assertEqual(check.notes, [])

    def test_re_registration_carries_choices_of_the_same_model(self):
        fresh, notes = build_card_with_notes(previous=load_card())
        self.assertEqual(notes, [])
        self.assertFalse(fresh.is_legacy)
        self.assertEqual((fresh.part("Body").kind, fresh.part("Body").kind_source), ("body", "user"))
        self.assertEqual(fresh.source["terms"], "個人利用のみ")
        save_card(fresh)
        self.assertEqual(check_card(), (fresh, []))

    def test_moved_model_cannot_be_matched_to_a_legacy_card(self):
        """版 1 の指紋は世界の座標なので、動かした後は同じモデルか確かめられない：写さずに知らせる"""
        self.move_armature(location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0))
        fresh, notes = build_card_with_notes(previous=load_card())
        self.assertEqual(notes, [card_text("card.choices_not_carried", armature=mmd_model.ARMATURE_NAME)])
        self.assertEqual(fresh.source["terms"], "")
        self.assertEqual(fresh.part("Body").kind_source, "guess")


if __name__ == "__main__":
    unittest.main()

"""テスト用のモデル（tests/fixtures/mmd_model.py）が、決まりどおりにできているかを確かめる。

ほかの多くのテストが、このモデルを土台にする。ここでは extract_bpy を使わず、bpy から直に読んで確かめる
（取り出す所のまちがいで、見本のまちがいが隠れないように）。
"""

import hashlib
import json
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import bpy
import numpy as np
from mathutils import Quaternion

from blender_ai_tools import bones
from blender_ai_tools.measure import geometry, mesh_check
from blender_ai_tools.measure import weights as weight_tools
from blender_ai_tools.measure.results import Signal
from tests.fixtures import mmd_model
from tests.shapes import signed_volume

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_SCRIPT = REPO_ROOT / "tests" / "fixtures" / "mmd_model.py"
KOUMAKAN_ENV = "BLENDER_AI_TOOLS_KOUMAKAN_REPO"

# 準標準の決まりどおりの親（体の中心と左。右は左の名前の 左 を 右 にした物）
EXPECTED_PARENTS = {
    "全ての親": None, "センター": "全ての親", "グルーブ": "センター", "腰": "グルーブ",
    "上半身": "腰", "上半身2": "上半身", "首": "上半身2", "頭": "首", "両目": "頭", "下半身": "腰",
    "左目": "頭",
    "左肩P": "上半身2", "左肩": "左肩P", "左肩C": "左肩", "左腕": "左肩",
    "左腕捩": "左腕", "左腕捩1": "左腕", "左腕捩2": "左腕", "左腕捩3": "左腕",
    "左ひじ": "左腕捩", "左手捩": "左ひじ", "左手捩1": "左ひじ", "左手捩2": "左ひじ", "左手捩3": "左ひじ",
    "左手首": "左手捩", "左ダミー": "左手首",
    "左親指０": "左手首", "左親指１": "左親指０", "左親指２": "左親指１",
    "左人指１": "左手首", "左人指２": "左人指１", "左人指３": "左人指２",
    "左中指１": "左手首", "左中指２": "左中指１", "左中指３": "左中指２",
    "左薬指１": "左手首", "左薬指２": "左薬指１", "左薬指３": "左薬指２",
    "左小指１": "左手首", "左小指２": "左小指１", "左小指３": "左小指２",
    "腰キャンセル左": "下半身", "左足": "腰キャンセル左", "左ひざ": "左足", "左足首": "左ひざ", "左つま先": "左足首",
    "左足D": "腰キャンセル左", "左ひざD": "左足D", "左足首D": "左ひざD", "左足先EX": "左足首D",
    "左足IK親": "全ての親", "左足ＩＫ": "左足IK親", "左つま先ＩＫ": "左足ＩＫ",
}
# 形を動かす骨（中心と左）
EXPECTED_DEFORM = {
    "上半身", "上半身2", "首", "頭", "下半身", "左目", "左肩", "左腕", "左腕捩", "左ひじ", "左手捩", "左手首",
    "左親指０", "左親指１", "左親指２", "左人指１", "左人指２", "左人指３", "左中指１", "左中指２", "左中指３",
    "左薬指１", "左薬指２", "左薬指３", "左小指１", "左小指２", "左小指３",
    "左足D", "左ひざD", "左足首D", "左足先EX",
}


def mirror(name):
    return name.replace("左", "右")


def expected_parents():
    result = dict(EXPECTED_PARENTS)
    for name, parent in EXPECTED_PARENTS.items():
        if "左" in name:
            result[mirror(name)] = None if parent is None else mirror(parent)
    return result


def expected_deform():
    return EXPECTED_DEFORM | {mirror(name) for name in EXPECTED_DEFORM if "左" in name}


def build(**kwargs):
    bpy.ops.wm.read_homefile(use_empty=True)
    return mmd_model.build_model(**kwargs)


def coordinates(mesh):
    flat = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", flat)
    return flat.reshape(-1, 3).astype(np.float64)


def triangles(mesh):
    flat = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", flat)
    return flat.reshape(-1, 3).astype(np.int64)


def group_weights(obj):
    """頂点グループの名前 → 頂点ごとの重み (N,)（付いていない頂点は 0）"""
    names = {group.index: group.name for group in obj.vertex_groups}
    result = {name: np.zeros(len(obj.data.vertices)) for name in names.values()}
    for vertex in obj.data.vertices:
        for element in vertex.groups:
            result[names[element.group]][vertex.index] = element.weight
    return result


def influence_counts(obj):
    return np.array([sum(1 for g in v.groups if g.weight > 0.0) for v in obj.data.vertices])


def evaluated_coordinates(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return coordinates(mesh)
    finally:
        evaluated.to_mesh_clear()


def evaluated_head(armature, name):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return np.array(armature.evaluated_get(depsgraph).pose.bones[name].head, dtype=np.float64)


# MARK: - 骨


class FixtureBonesTest(unittest.TestCase):
    def setUp(self):
        self.info = build()
        self.armature = bpy.data.objects[mmd_model.ARMATURE_NAME]

    def test_has_all_canonical_bones_with_exact_names(self):
        names = [bone.name for bone in self.armature.data.bones]
        canonical = bones.canonical_names()
        self.assertEqual(len(canonical), 96)
        self.assertEqual(len(names), 96)
        self.assertEqual(self.info["bone_count"], 96)
        self.assertEqual(set(names), set(canonical))
        self.assertEqual(self.armature.data.name, mmd_model.ARMATURE_NAME)

    def test_semi_standard_compliant(self):
        result = bones.map_bones([bone.name for bone in self.armature.data.bones])
        self.assertTrue(result.is_semi_standard_compliant)
        self.assertEqual(result.unknown, ())
        self.assertEqual(result.known_extras, ())
        self.assertEqual(result.conflicts, {})
        self.assertEqual(result.missing_standard, ())
        self.assertEqual(result.missing_semi_standard, ())
        self.assertEqual(result.mapping, {name: name for name in result.mapping})

    def test_hierarchy(self):
        expected = expected_parents()
        self.assertEqual(len(expected), 96)
        actual = {bone.name: (bone.parent.name if bone.parent else None) for bone in self.armature.data.bones}
        self.assertEqual(actual, expected)
        self.assertEqual(mmd_model.bone_parents(), expected)

    def test_deform_flags(self):
        actual = {bone.name for bone in self.armature.data.bones if bone.use_deform}
        self.assertEqual(actual, expected_deform())
        self.assertEqual(mmd_model.deform_bone_names(), expected_deform())

    def test_d_bones_overlap_visible_leg_bones(self):
        data = self.armature.data.bones
        for side in ("左", "右"):
            for visible, deform in (("足", "足D"), ("ひざ", "ひざD"), ("足首", "足首D")):
                a, b = data[side + visible], data[side + deform]
                np.testing.assert_allclose(b.head_local, a.head_local, atol=1e-7)
                np.testing.assert_allclose(b.tail_local, a.tail_local, atol=1e-7)

    def test_a_pose_arm_angle(self):
        data = self.armature.data.bones
        for side in ("左", "右"):
            direction = np.array(data[side + "ひじ"].head_local) - np.array(data[side + "腕"].head_local)
            angle = math.degrees(math.atan2(-direction[2], math.hypot(direction[0], direction[1])))
            self.assertAlmostEqual(angle, mmd_model.ARM_DOWN_DEG, places=4)
            self.assertTrue(35.0 <= angle <= 40.0)

    def test_left_is_plus_x(self):
        data = self.armature.data.bones
        for name in [bone.name for bone in data if "左" in bone.name]:
            with self.subTest(name=name):
                self.assertGreater(data[name].head_local.x, 0.0)
                self.assertLess(data[mirror(name)].head_local.x, 0.0)
                self.assertAlmostEqual(data[mirror(name)].head_local.x, -data[name].head_local.x, places=6)
        self.assertGreater(data["左手首"].head_local.x, 0.4)

    def test_eyes_and_toes_face_minus_y(self):
        data = self.armature.data.bones
        self.assertLess(data["左目"].head_local.y, -0.05)
        self.assertLess(data["左つま先"].head_local.y, data["左足首"].head_local.y - 0.05)

    def test_d_bones_follow_visible_leg(self):
        """足 を回すと 足D が、ひざ を回すと ひざD が同じだけ回る（MMD の回転付与）"""
        pose = self.armature.pose.bones
        pose["左足"].rotation_quaternion = Quaternion((1.0, 0.0, 0.0), math.radians(30.0))
        pose["左ひざ"].rotation_quaternion = Quaternion((1.0, 0.0, 0.0), math.radians(50.0))
        pose["右足首"].rotation_quaternion = Quaternion((0.0, 0.0, 1.0), math.radians(20.0))
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = self.armature.evaluated_get(depsgraph).pose.bones
        for visible, deform in (("左足", "左足D"), ("左ひざ", "左ひざD"), ("左足首", "左足首D"), ("右足首", "右足首D")):
            np.testing.assert_allclose(np.array(evaluated[deform].matrix), np.array(evaluated[visible].matrix),
                                       atol=1e-6, err_msg=deform)
        # 足首 の位置は大きく動いた（脚が曲がった）
        rest = np.array(self.armature.data.bones["左足首D"].head_local)
        self.assertGreater(np.linalg.norm(np.array(evaluated["左足首D"].head) - rest), 0.1)


# MARK: - 形


class FixtureShapeTest(unittest.TestCase):
    def setUp(self):
        self.info = build()
        self.body = bpy.data.objects[mmd_model.BODY_NAME]
        self.skirt = bpy.data.objects[mmd_model.SKIRT_NAME]
        self.armature = bpy.data.objects[mmd_model.ARMATURE_NAME]

    def test_info(self):
        self.assertEqual(self.info["armature"], "TestArmature")
        self.assertEqual(self.info["meshes"], ["Body", "Skirt"])
        self.assertEqual(self.info["height"], 1.45)
        self.assertEqual(self.info["vertex_counts"], {"Body": len(self.body.data.vertices),
                                                      "Skirt": len(self.skirt.data.vertices)})
        self.assertEqual(tuple(self.info["parts"]["Body"]), mmd_model.BODY_PARTS)
        self.assertEqual(self.info["shape_keys"], list(mmd_model.SHAPE_KEY_NAMES))

    def test_objects_at_origin_and_parented(self):
        for obj in (self.body, self.skirt):
            self.assertIs(obj.parent, self.armature)
            np.testing.assert_array_equal(np.array(obj.matrix_world), np.eye(4))
            modifiers = [m for m in obj.modifiers if m.type == "ARMATURE"]
            self.assertEqual(len(modifiers), 1)
            self.assertIs(modifiers[0].object, self.armature)
            self.assertTrue(modifiers[0].use_vertex_groups)
            self.assertFalse(modifiers[0].use_bone_envelopes)
        np.testing.assert_array_equal(np.array(self.armature.matrix_world), np.eye(4))

    def test_feet_on_floor_and_height(self):
        points = coordinates(self.body.data)
        self.assertEqual(float(points[:, 2].min()), 0.0)
        self.assertAlmostEqual(float(points[:, 2].max()), 1.45, delta=1e-6)
        # 足の裏の頂点（左右で 4 つずつ × 輪 4 つの半分）が z=0 にある
        self.assertEqual(int((points[:, 2] == 0.0).sum()), 16)
        # スカートは床から離れている
        self.assertGreater(float(coordinates(self.skirt.data)[:, 2].min()), 0.5)

    def test_other_height_scales_everything(self):
        info = build(height=1.6)
        body = bpy.data.objects["Body"]
        armature = bpy.data.objects["TestArmature"]
        points = coordinates(body.data)
        self.assertEqual(info["height"], 1.6)
        self.assertEqual(float(points[:, 2].min()), 0.0)
        self.assertAlmostEqual(float(points[:, 2].max()), 1.6, delta=1e-6)
        self.assertAlmostEqual(armature.data.bones["頭"].head_local.z, 1.25 * 1.6 / 1.45, delta=1e-6)
        self.assertAlmostEqual(armature.data.bones["左手首"].head_local.x,
                               self.armature_x_at_reference("左手首") * 1.6 / 1.45, delta=1e-6)

    @staticmethod
    def armature_x_at_reference(name):
        return next(bone.head[0] for bone in mmd_model.bone_specs() if bone.name == name)

    def test_faces_minus_y(self):
        points = coordinates(self.body.data)
        head = points[points[:, 2] > 0.85 * 1.45]
        nose = head[np.argmin(head[:, 1])]
        self.assertAlmostEqual(float(nose[0]), 0.0, delta=1e-7)
        self.assertLess(float(nose[1]), -0.1)
        self.assertGreater(float(head[:, 1].max()), 0.0)
        self.assertLess(float(head[:, 1].max()), -float(nose[1]))  # 後ろより前に出ている
        start, stop = self.info["parts"]["Body"]["foot_L"]
        foot = points[start:stop]
        self.assertLess(float(foot[:, 1].min()), -0.14)  # つま先は前（-Y）
        self.assertLess(float(foot[:, 1].max()), 0.06)  # かかとは少し後ろ
        self.assertTrue((foot[:, 0] > 0.0).all())

    def test_left_arm_on_plus_x(self):
        points = coordinates(self.body.data)
        groups = group_weights(self.body)
        for name in ("左ひじ", "左手首", "左手捩", "左人指３", "左足D"):
            with self.subTest(name=name):
                self.assertTrue((points[groups[name] > 0.0, 0] > 0.0).all())
                self.assertTrue((points[groups[mirror(name)] > 0.0, 0] < 0.0).all())
        start, stop = self.info["parts"]["Body"]["hand_L"]
        self.assertGreater(float(points[start:stop, 0].min()), 0.45)

    def test_body_parts_closed_and_outward(self):
        points = coordinates(self.body.data)
        tris = triangles(self.body.data)
        labels, count = mesh_check.face_components(tris)
        self.assertEqual(count, len(mmd_model.BODY_PARTS))
        parts = self.info["parts"]["Body"]
        covered = np.zeros(len(points), dtype=bool)
        for component in range(count):
            part_tris = tris[labels == component]
            used = np.unique(part_tris)
            name = next(name for name, (start, stop) in parts.items() if start <= used[0] < stop)
            start, stop = parts[name]
            with self.subTest(part=name):
                np.testing.assert_array_equal(used, np.arange(start, stop))
                self.assertEqual(len(mesh_check.boundary_edges(part_tris)), 0)
                self.assertEqual(len(mesh_check.non_manifold_edges(part_tris)), 0)
                self.assertEqual(len(mesh_check.inconsistent_edges(part_tris)), 0)
                self.assertGreater(signed_volume(points, part_tris), 0.0)
                self.assertEqual(len(mesh_check.degenerate_triangles(points, part_tris)), 0)
                covered[start:stop] = True
        self.assertTrue(covered.all())
        holes = mesh_check.summarize_holes(tris)
        self.assertEqual((holes.value, holes.signal), (0.0, Signal.GREEN))

    def test_skirt_open_outward_and_outside_body(self):
        points = coordinates(self.skirt.data)
        tris = triangles(self.skirt.data)
        holes = mesh_check.summarize_holes(tris)
        self.assertEqual(holes.value, 2.0)  # 上と下の縁
        self.assertEqual(len(mesh_check.inconsistent_edges(tris)), 0)
        normals = geometry.triangle_normals(points, tris)
        centers = points[tris].mean(axis=1)
        radial = centers * np.array([1.0, 1.0, 0.0])
        self.assertTrue((np.einsum("ij,ij->i", normals, radial) > 0.0).all())
        # 胴ともものどれにも入っていない（どの点も、それぞれの閉じた形の外に 5mm 以上）
        body_points = coordinates(self.body.data)
        body_tris = triangles(self.body.data)
        for part in ("torso", "thigh_L", "thigh_R"):
            start, stop = self.info["parts"]["Body"][part]
            part_tris = body_tris[(body_tris >= start).all(axis=1) & (body_tris < stop).all(axis=1)]
            distance = geometry.signed_distance(points, body_points, part_tris)
            with self.subTest(part=part):
                self.assertGreater(float(np.nanmin(distance)), 0.005)
                self.assertFalse(np.isnan(distance).any())

    def test_no_cloth(self):
        info = build(with_cloth=False)
        self.assertEqual(info["meshes"], ["Body"])
        self.assertIsNone(bpy.data.objects.get("Skirt"))
        self.assertNotIn("Skirt", info["parts"])

    def test_deterministic(self):
        first = (coordinates(self.body.data), triangles(self.body.data), coordinates(self.skirt.data),
                 group_weights(self.body))
        build()
        body = bpy.data.objects["Body"]
        second = (coordinates(body.data), triangles(body.data), coordinates(bpy.data.objects["Skirt"].data),
                  group_weights(body))
        for a, b in zip(first[:3], second[:3]):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(first[3].keys(), second[3].keys())
        for name in first[3]:
            np.testing.assert_array_equal(first[3][name], second[3][name])

    def test_rejects_bad_height_and_second_build(self):
        for height in (0.0, -1.0, float("nan"), float("inf"), True, "1.45"):
            with self.subTest(height=height), self.assertRaises(ValueError):
                mmd_model.build_model(height=height)
        with self.assertRaises(ValueError):
            mmd_model.build_model()  # 同じシーンに2回は作らない


# MARK: - ウェイト


class FixtureWeightsTest(unittest.TestCase):
    def setUp(self):
        self.info = build()
        self.body = bpy.data.objects["Body"]
        self.skirt = bpy.data.objects["Skirt"]
        self.armature = bpy.data.objects["TestArmature"]
        self.groups = group_weights(self.body)

    def test_groups_are_bones_and_only_deform_bones_carry_weight(self):
        bone_names = {bone.name for bone in self.armature.data.bones}
        deform = expected_deform()
        weighted = set()
        for obj in (self.body, self.skirt):
            for name, values in group_weights(obj).items():
                with self.subTest(mesh=obj.name, group=name):
                    self.assertIn(name, bone_names)
                    self.assertIn(name, deform)
                    self.assertGreater(float(values.max()), 0.0)
                weighted.add(name)
        # 逆も：形を動かす骨は、どれもウェイトを持つ（use_deform はウェイトのある骨だけ）
        self.assertEqual(weighted, deform)
        self.assertEqual(weighted, {bone.name for bone in self.armature.data.bones if bone.use_deform})

    def test_group_order_differs_from_bone_order(self):
        """頂点グループの番号と骨の番号が違う（番号のまちがいを、取り出す所のテストで見つけられるように）"""
        bone_index = {bone.name: i for i, bone in enumerate(self.armature.data.bones)}
        differs = [g.name for g in self.body.vertex_groups if g.index != bone_index[g.name]]
        self.assertGreater(len(differs), 10)

    def test_leg_weights_on_d_bones(self):
        for side in ("左", "右"):
            for name in ("足", "ひざ", "足首", "つま先"):
                self.assertNotIn(side + name, self.groups)
            for name in ("足D", "ひざD", "足首D", "足先EX"):
                self.assertGreater(int((self.groups[side + name] > 0.0).sum()), 0, side + name)

    def test_blends_at_elbows_and_knees(self):
        for side in ("左", "右"):
            for upper, lower in (("腕", "ひじ"), ("足D", "ひざD")):
                a, b = self.groups[side + upper], self.groups[side + lower]
                blended = (a > 0.0) & (a < 1.0) & (b > 0.0) & (b < 1.0)
                with self.subTest(side=side, joint=lower):
                    self.assertGreaterEqual(int(blended.sum()), 8)

    def test_twist_bones_carry_weight_on_their_segment(self):
        """捩りの骨のウェイトは、その骨のある区間（腕捩は上腕、手捩は前腕）に乗る（5本の骨を付けた頂点は除く）"""
        points = coordinates(self.body.data)
        data = self.armature.data.bones
        ordinary = np.ones(len(points), dtype=bool)
        ordinary[self.info["five_influence_vertices"]] = False
        for side in ("左", "右"):
            for twist, start, end in (("腕捩", "腕", "ひじ"), ("手捩", "ひじ", "手首")):
                a, b = np.array(data[side + start].head_local), np.array(data[side + end].head_local)
                carried = points[(self.groups[side + twist] > 0.0) & ordinary]
                t = (carried - a) @ (b - a) / float((b - a) @ (b - a))
                with self.subTest(side=side, twist=twist):
                    self.assertGreaterEqual(len(carried), 8)
                    self.assertTrue(((t > 0.0) & (t < 1.0 + 1e-6)).all())

    def test_weights_sum_to_one(self):
        for obj in (self.body, self.skirt):
            total = sum(group_weights(obj).values())
            np.testing.assert_allclose(total, 1.0, atol=1e-6, err_msg=obj.name)

    def test_five_influence_vertices(self):
        counts = influence_counts(self.body)
        five = self.info["five_influence_vertices"]
        self.assertEqual(len(five), 4)
        self.assertEqual(sorted(np.flatnonzero(counts == 5).tolist()), sorted(five))
        self.assertEqual(int(counts.max()), 5)
        start, stop = self.info["parts"]["Body"]["forearm_L"]
        self.assertTrue(all(start <= index < stop for index in five[:2]))
        for index in five:
            values = sorted(g.weight for g in self.body.data.vertices[index].groups)
            self.assertAlmostEqual(values[0], mmd_model.SMALL_FIFTH_WEIGHT, delta=1e-6)
            self.assertAlmostEqual(sum(values), 1.0, delta=1e-6)

    def test_four_limit_changes_only_five_influence_vertices(self):
        bone_index = {bone.name: i for i, bone in enumerate(self.armature.data.bones)}
        rows = [[(bone_index[self.body.vertex_groups[g.group].name], g.weight) for g in v.groups]
                for v in self.body.data.vertices]
        table = weight_tools.WeightTable.from_lists(rows)
        deform = np.array([bone.use_deform for bone in self.armature.data.bones])
        limited, changed = weight_tools.normalize_and_limit(table, deform=deform)
        self.assertEqual(sorted(np.flatnonzero(changed).tolist()), sorted(self.info["five_influence_vertices"]))
        difference = np.abs(limited.to_dense(len(deform)) - table.to_dense(len(deform))).max(axis=1)
        self.assertAlmostEqual(float(difference.max()), mmd_model.SMALL_FIFTH_WEIGHT, delta=1e-6)

    def test_without_five_influences(self):
        info = build(five_influences=False)
        body = bpy.data.objects["Body"]
        self.assertEqual(info["five_influence_vertices"], [])
        self.assertEqual(int(influence_counts(body).max()), 2)  # どの頂点も、混ぜても2本まで

    def test_skirt_weights(self):
        groups = group_weights(self.skirt)
        self.assertEqual(set(groups), {"下半身", "左足D", "右足D"})
        points = coordinates(self.skirt.data)
        top = points[:, 2] > 0.86 - 1e-6
        np.testing.assert_allclose(groups["下半身"][top], 1.0, atol=1e-7)
        bottom = points[:, 2] < 0.58 + 1e-6
        np.testing.assert_allclose(groups["下半身"][bottom], 0.4, atol=1e-6)
        left = bottom & (points[:, 0] > 1e-6)
        self.assertTrue((groups["左足D"][left] > groups["右足D"][left]).all())
        np.testing.assert_allclose(groups["左足D"][bottom] + groups["右足D"][bottom], 0.6, atol=1e-6)


# MARK: - シェイプキー


class FixtureShapeKeysTest(unittest.TestCase):
    def test_keys_move_a_few_head_vertices_down(self):
        build()
        body = bpy.data.objects["Body"]
        keys = body.data.shape_keys.key_blocks
        self.assertEqual([key.name for key in keys], list(mmd_model.SHAPE_KEY_NAMES))
        basis = np.array([tuple(point.co) for point in keys[0].data])
        np.testing.assert_array_equal(basis, coordinates(body.data))
        groups = group_weights(body)
        face = (groups["頭"] > 0.0) | (groups["左目"] > 0.0) | (groups["右目"] > 0.0)
        for key in list(keys)[1:]:
            moved = np.array([tuple(point.co) for point in key.data]) - basis
            changed = np.flatnonzero(np.abs(moved).max(axis=1) > 0.0)
            with self.subTest(key=key.name):
                self.assertTrue(2 <= len(changed) <= 4)
                self.assertTrue(face[changed].all())
                self.assertTrue((moved[changed, 2] < -0.005).all())
                np.testing.assert_array_equal(moved[changed, :2], 0.0)
                self.assertEqual(key.value, 0.0)

    def test_without_shape_keys(self):
        info = build(with_shape_keys=False)
        self.assertIsNone(bpy.data.objects["Body"].data.shape_keys)
        self.assertEqual(info["shape_keys"], [])


# MARK: - 保存と読み込み


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def folder_state(folder):
    """フォルダの中の全部のファイル：場所 → (大きさ, 変えた時刻, ハッシュ)。無いフォルダは None。
    名前だけでは、同じ名前のファイルの上書き（アプリの character.usdz など）を見落とすので、中身まで比べる"""
    folder = Path(folder)
    if not folder.exists():
        return None
    return {str(path.relative_to(folder)): (path.stat().st_size, path.stat().st_mtime_ns, file_sha256(path))
            for path in sorted(folder.rglob("*")) if path.is_file()}


class FolderStateTest(unittest.TestCase):
    """folder_state が、名前の変わらない上書きも見つける（アプリのフォルダを守る確かめの、確かめ）"""

    def test_detects_overwrite_with_same_name(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "sub" / "character.usdz"
            target.parent.mkdir()
            target.write_bytes(b"old")
            before = folder_state(folder)
            self.assertEqual(list(before), [str(Path("sub") / "character.usdz")])
            target.write_bytes(b"new")
            self.assertNotEqual(folder_state(folder), before)
        self.assertIsNone(folder_state(Path(folder) / "missing"))


class FixtureFileTest(unittest.TestCase):
    def test_save_and_load(self):
        info = build()
        self.assertEqual(bpy.data.filepath, "")
        with tempfile.TemporaryDirectory() as folder:
            path = mmd_model.save_model(Path(folder) / "sub" / "model.blend")
            self.assertTrue(path.exists())
            self.assertEqual(bpy.data.filepath, "")  # 写しとして保存した（いまのファイルの場所は変わらない）
            bpy.ops.wm.open_mainfile(filepath=str(path))
            armature = bpy.data.objects["TestArmature"]
            self.assertEqual(len(armature.data.bones), info["bone_count"])
            for name, count in info["vertex_counts"].items():
                self.assertEqual(len(bpy.data.objects[name].data.vertices), count)
            self.assertEqual(len(armature.pose.bones["左足D"].constraints), 1)
        bpy.ops.wm.read_homefile(use_empty=True)

    def test_command_line(self):
        expected = build(with_cloth=False)
        expected_points = coordinates(bpy.data.objects["Body"].data)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "model.blend"
            result = subprocess.run([bpy.app.binary_path, "--background", "--factory-startup", "--python",
                                     str(FIXTURE_SCRIPT), "--", str(output), "--no-cloth"],
                                    capture_output=True, text=True, timeout=300)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(output.exists())
            bpy.ops.wm.open_mainfile(filepath=str(output))
            self.assertIsNone(bpy.data.objects.get("Skirt"))
            self.assertEqual(len(bpy.data.objects["TestArmature"].data.bones), expected["bone_count"])
            # 別の Blender で作っても、まったく同じ形になる
            np.testing.assert_array_equal(coordinates(bpy.data.objects["Body"].data), expected_points)
        bpy.ops.wm.read_homefile(use_empty=True)


class ConverterCompatibilityTest(unittest.TestCase):
    """本物の convert_blend.py が、この見本を書き出せる（紅魔館の場所が無いときは飛ばす）"""

    def setUp(self):
        repo = os.environ.get(KOUMAKAN_ENV)
        if not repo:
            self.skipTest(f"環境変数 {KOUMAKAN_ENV}（紅魔館のリポジトリの場所）がありません")
        self.converter = Path(repo) / "tools" / "model-converter" / "convert_blend.py"
        if not self.converter.exists():
            self.skipTest(f"convert_blend.py がありません：{self.converter}")
        self.app_model = Path(repo) / "KoumakanApp" / "KoumakanApp" / "Model"

    def test_converts(self):
        app_before = folder_state(self.app_model)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "model.blend"
            output = Path(folder) / "out"
            build()
            mmd_model.save_model(source)
            digest = file_sha256(source)
            result = subprocess.run([bpy.app.binary_path, "--background", "--factory-startup", "--python",
                                     str(self.converter), "--", str(source), str(output)],
                                    capture_output=True, text=True, timeout=600)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((output / "character.usdz").exists())
            rig = json.loads((output / "character_rig.json").read_text(encoding="utf-8"))
            self.assertEqual(file_sha256(source), digest)  # 元のファイルは変わらない
        self.assertEqual(rig["restArmDownDegrees"], mmd_model.ARM_DOWN_DEG)
        self.assertEqual(len(rig["bones"]), 21)
        self.assertEqual(rig["originalNames"]["bones"]["hips"], "センター")
        self.assertEqual(rig["originalNames"]["bones"]["leftUpperLeg"], "左足D")
        self.assertEqual(set(rig["originalNames"]["bones"].values()), set(bones.canonical_names()))
        self.assertIn("骨の影響を4本までに絞った頂点 4 個", result.stdout)
        # アプリの Model フォルダは、作られず・増えず・上書きもされない（出力フォルダを必ず指定した）
        self.assertEqual(folder_state(self.app_model), app_before)


if __name__ == "__main__":
    unittest.main()

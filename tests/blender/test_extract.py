"""A1・A4：Blender から形・骨・ウェイト・ポーズを取り出す所（model/extract_bpy.py）を確かめる。

テスト用のモデル（tests/fixtures/mmd_model.py）を、空のシーンに作ってから調べる。
答えは、bpy から直に読んだ値と、手で計算した値（軸・空間・行列の向き・骨の並びのまちがいで外れる値）で確かめる。
"""

import math
import unittest

import bmesh
import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion

from blender_ai_tools.measure import kinematics
from blender_ai_tools.measure.weights import WeightTable, linear_blend_skinning, normalize_and_limit
from blender_ai_tools.model import extract_bpy
from blender_ai_tools.model.extract_bpy import (BoneRest, ModelObjects, bone_names, bone_rest, bone_weights,
                                                find_model, joint_positions, mesh_evaluated, mesh_rest,
                                                non_bone_groups, pose_matrices, pose_rotations,
                                                quaternions_from_rotations, reset_pose, set_pose,
                                                skinning_matrices)
from tests.fixtures import mmd_model

# アーマチュアを動かすときの、決まった動かし方：Z まわりに 90 度回し、(1, 2, 3) へ動かす
MOVE = (1.0, 2.0, 3.0)


def moved(points):
    """MOVE で動かした位置を、手で計算する：Z まわりの 90 度は (x, y, z) → (-y, x, z)"""
    p = np.asarray(points, dtype=np.float64)
    return np.stack([-p[..., 1], p[..., 0], p[..., 2]], axis=-1) + np.array(MOVE)


def quaternion_about(axis, degrees):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = math.radians(degrees) / 2.0
    return np.concatenate([[math.cos(half)], math.sin(half) * axis])


def assert_same_rotation(test, actual, expected, atol=1e-6):
    """q と -q は同じ回転なので、どちらかに近ければよい"""
    actual, expected = np.asarray(actual), np.asarray(expected)
    test.assertTrue(np.allclose(actual, expected, atol=atol) or np.allclose(actual, -expected, atol=atol),
                    f"{actual} != {expected}")


def raw_coordinates(mesh):
    flat = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", flat)
    return flat.reshape(-1, 3).astype(np.float64)


def normalized_deform_table(table, deform):
    """Blender のアーマチュアと同じ重み：形を動かす骨だけを使い、合計で割る"""
    dense = table.to_dense(len(deform))
    dense[:, ~deform] = 0.0
    total = dense.sum(axis=1, keepdims=True)
    dense = np.divide(dense, total, out=np.zeros_like(dense), where=total > 0.0)
    return WeightTable.from_dense(dense, epsilon=0.0)


class FixtureCase(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_homefile(use_empty=True)
        self.info = mmd_model.build_model()
        self.armature = bpy.data.objects["TestArmature"]
        self.body = bpy.data.objects["Body"]
        self.skirt = bpy.data.objects["Skirt"]

    def move_armature(self):
        self.armature.location = MOVE
        self.armature.rotation_mode = "XYZ"
        self.armature.rotation_euler = (0.0, 0.0, math.radians(90.0))
        bpy.context.view_layer.update()

    def make_armature(self, name, bone_count, scene=None):
        """骨が bone_count 本のアーマチュアを作る（scene を渡すと、そのシーンへ移す）"""
        data = bpy.data.armatures.new(name)
        obj = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(obj)
        previous = bpy.context.view_layer.objects.active
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        for i in range(bone_count):
            bone = data.edit_bones.new(f"b{i}")
            bone.head = (0.0, 0.0, 0.1 * i)
            bone.tail = (0.0, 0.0, 0.1 * i + 0.05)
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.context.view_layer.objects.active = previous
        if scene is not None:
            bpy.context.scene.collection.objects.unlink(obj)
            scene.collection.objects.link(obj)
        return obj

    def make_mesh(self, name, *, parent=None, modifier_target=None):
        data = bpy.data.meshes.new(name)
        data.from_pydata([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [], [(0, 1, 2)])
        obj = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(obj)
        obj.parent = parent
        if modifier_target is not None:
            obj.modifiers.new("Armature", "ARMATURE").object = modifier_target
        return obj


# MARK: - モデルを探す


class FindModelTest(FixtureCase):
    def test_finds_fixture(self):
        model = find_model()
        self.assertIsInstance(model, ModelObjects)
        self.assertIs(model.armature, self.armature)
        self.assertEqual(model.meshes, (self.body, self.skirt))

    def test_prefers_armature_with_most_bones(self):
        small = self.make_armature("AAA_Small", 2)
        self.make_mesh("AAA_SmallMesh", parent=small, modifier_target=small)
        model = find_model()
        self.assertIs(model.armature, self.armature)
        self.assertEqual([obj.name for obj in model.meshes], ["Body", "Skirt"])

    def test_tie_goes_to_first_in_object_order(self):
        """骨の数が同じなら、bpy.data.objects の並び（名前の順）で先の物（convert_blend.py の max と同じ）"""
        first = self.make_armature("AAA_Same", 96)
        self.make_mesh("AAA_Mesh", parent=first)
        self.assertIs(find_model().armature, first)

    def test_hide_render_meshes_are_ignored(self):
        self.skirt.hide_render = True
        self.body.hide_viewport = True  # 画面で隠しただけの物は、書き出しに入る
        self.assertEqual(find_model().meshes, (self.body,))

    def test_parent_or_modifier(self):
        cloth = self.make_mesh("Cloth", modifier_target=self.armature)  # 子ではないが、モディファイアがある
        hat = self.make_mesh("Hat", parent=self.armature)  # 子だが、モディファイアが無い
        self.make_mesh("Prop")  # どちらも無い
        small = self.make_armature("Small", 1)
        self.make_mesh("Other", modifier_target=small)  # ほかのアーマチュアを使う
        self.assertEqual(find_model().meshes, (self.body, cloth, hat, self.skirt))

    def test_scene_limits_the_search(self):
        other = bpy.data.scenes.new("Other")
        self.make_armature("ZZZ_Big", 120, scene=other)
        # ファイル全体で探すと、ほかのシーンの大きなアーマチュアが選ばれ、それに付いたメッシュは無い
        with self.assertRaisesRegex(ValueError, "メッシュがありません"):
            find_model()
        model = find_model(bpy.context.scene)
        self.assertIs(model.armature, self.armature)
        self.assertEqual(model.meshes, (self.body, self.skirt))

    def test_errors(self):
        bpy.ops.wm.read_homefile(use_empty=True)
        with self.assertRaisesRegex(ValueError, "アーマチュアがありません"):
            find_model()
        self.make_armature("Lonely", 3)
        self.make_mesh("Prop")
        with self.assertRaisesRegex(ValueError, "メッシュがありません"):
            find_model()

    def test_armature_of(self):
        self.assertIs(extract_bpy.armature_of(self.body), self.armature)
        hat = self.make_mesh("Hat", parent=self.armature)
        self.assertIs(extract_bpy.armature_of(hat), self.armature)
        with self.assertRaises(ValueError):
            extract_bpy.armature_of(self.make_mesh("Prop"))


# MARK: - 骨の休みの形


class BoneRestTest(FixtureCase):
    def test_order_parents_and_flags(self):
        rest = bone_rest(self.armature)
        self.assertIsInstance(rest, BoneRest)
        data = list(self.armature.data.bones)
        self.assertEqual(rest.names, tuple(bone.name for bone in data))
        self.assertEqual(rest.bone_count, 96)
        self.assertEqual(bone_names(self.armature), rest.names)
        self.assertEqual(rest.parents.dtype, np.int64)
        for i, bone in enumerate(data):
            if bone.parent is None:
                self.assertEqual(rest.parents[i], -1)
            else:
                self.assertEqual(rest.names[rest.parents[i]], bone.parent.name)
                self.assertLess(rest.parents[i], i)  # 親から子への並び
        self.assertEqual(rest.names[0], "全ての親")
        np.testing.assert_array_equal(rest.deform, [bone.use_deform for bone in data])
        self.assertEqual(rest.deform.dtype, bool)
        specs = {spec.name: spec for spec in mmd_model.bone_specs()}
        for i, name in enumerate(rest.names):
            np.testing.assert_allclose(rest.heads[i], specs[name].head, atol=1e-6, err_msg=name)
            np.testing.assert_allclose(rest.tails[i], specs[name].tail, atol=1e-6, err_msg=name)

    def test_index_and_indices(self):
        rest = bone_rest(self.armature, space="armature")
        self.assertEqual(rest.names[rest.index("左ひじ")], "左ひじ")
        np.testing.assert_array_equal(rest.indices(["頭", "左足D"]), [rest.index("頭"), rest.index("左足D")])
        with self.assertRaises(ValueError):
            rest.index("Head")
        with self.assertRaisesRegex(ValueError, "Head"):
            rest.indices(["頭", "Head"])

    def test_matrices_are_column_vector_bone_frames(self):
        rest = bone_rest(self.armature, space="armature")
        self.assertEqual(rest.matrices.shape, (96, 4, 4))
        np.testing.assert_allclose(rest.matrices[:, 3, :], np.tile([0.0, 0.0, 0.0, 1.0], (96, 1)), atol=1e-7)
        np.testing.assert_allclose(rest.matrices[:, :3, 3], rest.heads, atol=1e-7)
        direction = rest.tails - rest.heads
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        np.testing.assert_allclose(rest.matrices[:, :3, 1], direction, atol=1e-6)  # Y 軸が骨の向き
        for i, bone in enumerate(self.armature.data.bones):
            np.testing.assert_allclose(rest.matrices[i], np.array(bone.matrix_local), atol=1e-7)

    def test_world_and_armature_space(self):
        before = bone_rest(self.armature, space="armature")
        self.move_armature()
        world = bone_rest(self.armature)
        armature = bone_rest(self.armature, space="armature")
        self.assertEqual((world.space, armature.space), ("world", "armature"))
        np.testing.assert_allclose(armature.heads, before.heads, atol=0.0)
        np.testing.assert_allclose(world.heads, moved(before.heads), atol=1e-6)
        np.testing.assert_allclose(world.tails, moved(before.tails), atol=1e-6)
        # 骨の軸も回る：骨の Y 軸（列 1）を Z まわりに 90 度回した物
        np.testing.assert_allclose(world.matrices[:, :3, 1], moved(before.matrices[:, :3, 1]) - np.array(MOVE),
                                   atol=1e-6)
        np.testing.assert_allclose(world.matrices[:, :3, 3], world.heads, atol=1e-6)

    def test_scaled_armature(self):
        self.armature.scale = (2.0, 2.0, 2.0)
        bpy.context.view_layer.update()
        np.testing.assert_allclose(bone_rest(self.armature).heads,
                                   2.0 * bone_rest(self.armature, space="armature").heads, atol=1e-6)

    def test_arrays_are_read_only(self):
        rest = bone_rest(self.armature)
        with self.assertRaises(ValueError):
            rest.heads[0, 0] = 1.0

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            bone_rest(self.armature, space="local")
        with self.assertRaises(ValueError):
            bone_rest(self.body)


# MARK: - メッシュ


class MeshTest(FixtureCase):
    def test_rest_counts_and_types(self):
        vertices, triangles = mesh_rest(self.body)
        self.assertEqual(vertices.shape, (self.info["vertex_counts"]["Body"], 3))
        self.assertEqual(vertices.dtype, np.float64)
        self.assertEqual(triangles.dtype, np.int64)
        self.assertEqual(triangles.shape, (len(self.body.data.loop_triangles), 3))
        self.assertEqual(int(triangles.min()), 0)
        self.assertEqual(int(triangles.max()), len(vertices) - 1)
        np.testing.assert_array_equal(vertices, raw_coordinates(self.body.data))

    def test_rest_world_and_armature_space(self):
        local = raw_coordinates(self.body.data)
        self.move_armature()  # Body はアーマチュアの子なので、一緒に動く
        world, _ = mesh_rest(self.body)
        in_armature, _ = mesh_rest(self.body, space="armature")
        np.testing.assert_allclose(world, moved(local), atol=1e-6)
        np.testing.assert_allclose(in_armature, local, atol=1e-6)

    def test_object_offset_inside_armature(self):
        self.move_armature()
        self.skirt.location = (0.0, 0.0, 0.1)  # アーマチュアの中で 10cm 上げる
        bpy.context.view_layer.update()
        local = raw_coordinates(self.skirt.data)
        in_armature, _ = mesh_rest(self.skirt, space="armature")
        np.testing.assert_allclose(in_armature, local + np.array([0.0, 0.0, 0.1]), atol=1e-6)
        np.testing.assert_allclose(mesh_rest(self.skirt)[0], moved(local + np.array([0.0, 0.0, 0.1])), atol=1e-6)

    def test_rest_of_bone_parented_mesh_ignores_pose(self):
        """骨を親にしたメッシュ（parent_type BONE。髪飾りなど）は、ポーズを付けても、親の骨が休みの姿勢にあるときの
        位置で読む（A1 の登録カードの指紋も、この読み方）。休みの姿勢では Blender が計算した位置と同じ"""
        data = bpy.data.meshes.new("Ribbon")
        data.from_pydata([(0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.0, 0.05, 0.0)], [], [(0, 1, 2)])
        ribbon = bpy.data.objects.new("Ribbon", data)
        bpy.context.scene.collection.objects.link(ribbon)
        ribbon.parent = self.armature
        ribbon.parent_type = "BONE"
        ribbon.parent_bone = "左ひじ"
        ribbon.location = (0.01, 0.02, 0.03)
        self.move_armature()
        local = raw_coordinates(data)

        def blender_world():
            matrix = np.array(ribbon.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world)
            return local @ matrix[:3, :3].T + matrix[:3, 3]

        at_rest = blender_world()
        rest = {space: mesh_rest(ribbon, space=space, armature=self.armature)[0] for space in extract_bpy.SPACES}
        np.testing.assert_allclose(rest["world"], at_rest, atol=1e-6)
        np.testing.assert_allclose(rest["world"], moved(rest["armature"]), atol=1e-6)
        set_pose(self.armature, {"左腕": quaternion_about((1.0, 0.0, 0.0), 30.0),
                                 "左ひじ": quaternion_about((0.0, 0.0, 1.0), 40.0)})
        self.assertGreater(np.abs(blender_world() - at_rest).max(), 0.05)  # ほんとうに動いた
        for space in extract_bpy.SPACES:
            with self.subTest(space=space):
                np.testing.assert_array_equal(mesh_rest(ribbon, space=space, armature=self.armature)[0], rest[space])

    def test_evaluated_equals_rest_at_rest_pose(self):
        self.move_armature()
        for obj in (self.body, self.skirt):
            for space in ("world", "armature"):
                with self.subTest(mesh=obj.name, space=space):
                    rest, _ = mesh_rest(obj, space=space)
                    np.testing.assert_allclose(mesh_evaluated(obj, space=space), rest, atol=1e-6)

    def test_rotating_upper_arm_moves_only_arm_chain(self):
        rest = bone_rest(self.armature, space="armature")
        table, names = bone_weights(self.body, self.armature)
        chain = np.zeros(rest.bone_count, dtype=bool)
        for i in range(rest.bone_count):  # 左腕 と、その子孫
            j = i
            while j >= 0 and not chain[i]:
                chain[i] = rest.names[j] == "左腕"
                j = rest.parents[j]
        dense = table.to_dense(rest.bone_count)
        on_chain = dense[:, chain].sum(axis=1) > 0.0
        before = mesh_evaluated(self.body, space="armature")
        set_pose(self.armature, {"左腕": quaternion_about((1.0, 0.0, 0.0), 30.0)})
        after = mesh_evaluated(self.body, space="armature")
        moved_distance = np.linalg.norm(after - before, axis=1)
        self.assertLess(float(moved_distance[~on_chain].max()), 1e-6)
        start, stop = self.info["parts"]["Body"]["hand_L"]
        self.assertGreater(float(moved_distance[start:stop].min()), 0.1)
        start, stop = self.info["parts"]["Body"]["hand_R"]
        self.assertLess(float(moved_distance[start:stop].max()), 1e-6)

    def test_mask_modifier_is_rejected(self):
        group = self.body.vertex_groups.new(name="mask")
        group.add(list(range(100)), 1.0, "REPLACE")
        modifier = self.body.modifiers.new("Mask", "MASK")
        modifier.vertex_group = "mask"
        with self.assertRaisesRegex(ValueError, "頂点の数が変わります"):
            mesh_evaluated(self.body)
        self.assertEqual(mesh_rest(self.body)[0].shape[0], self.info["vertex_counts"]["Body"])

    def test_edit_mode_reads_current_edit_and_keeps_mode(self):
        bpy.context.view_layer.objects.active = self.body
        self.body.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            edit = bmesh.from_edit_mesh(self.body.data)
            edit.verts.ensure_lookup_table()
            edit.verts[0].co.z += 0.25
            bmesh.update_edit_mesh(self.body.data)
            expected = float(edit.verts[0].co.z)
            vertices, _ = mesh_rest(self.body, space="armature")
            self.assertAlmostEqual(float(vertices[0, 2]), expected, delta=1e-6)
            self.assertEqual(self.body.mode, "EDIT")
            self.assertIs(bpy.context.view_layer.objects.active, self.body)
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            mesh_rest(self.armature)
        with self.assertRaises(ValueError):
            mesh_rest(self.body, space="local")
        with self.assertRaises(ValueError):
            mesh_evaluated(self.body, space="armature", armature=self.body)


# MARK: - ウェイト


class WeightsTest(FixtureCase):
    def independent_dense(self, obj):
        """bpy から直に読んだ (N, B) の重み（骨の名前で引く。骨でないグループは使わない）"""
        names = [bone.name for bone in self.armature.data.bones]
        index = {name: i for i, name in enumerate(names)}
        dense = np.zeros((len(obj.data.vertices), len(names)))
        for vertex in obj.data.vertices:
            for element in vertex.groups:
                name = obj.vertex_groups[element.group].name
                if name in index:
                    dense[vertex.index, index[name]] += element.weight
        return dense

    def test_matches_vertex_groups_by_name(self):
        for obj in (self.body, self.skirt):
            table, names = bone_weights(obj, self.armature)
            self.assertEqual(names, bone_rest(self.armature).names)
            self.assertEqual(table.vertex_count, len(obj.data.vertices))
            np.testing.assert_array_equal(table.to_dense(len(names)), self.independent_dense(obj))

    def test_group_index_is_not_bone_index(self):
        """頂点グループの番号をそのまま骨の番号にすると、この見本では合わない（番号の対応を確かめる前提）"""
        names = bone_names(self.armature)
        self.assertTrue(any(names.index(group.name) != group.index for group in self.body.vertex_groups))

    def test_width_and_slot_order(self):
        table, _ = bone_weights(self.body, self.armature)
        self.assertEqual(table.width, 5)
        used = table.bone_ids >= 0
        counts = used.sum(axis=1)
        five = self.info["five_influence_vertices"]
        self.assertEqual(sorted(np.flatnonzero(counts == 5).tolist()), sorted(five))
        # 使った枠は前に詰まり、骨の番号の小さい順
        for row, count in zip(table.bone_ids, counts):
            self.assertTrue((row[:count] >= 0).all() and (row[count:] == -1).all())
            self.assertTrue((np.diff(row[:count]) > 0).all())
        self.assertTrue((table.weights[~used] == 0.0).all())
        self.assertTrue((table.weights[used] > 0.0).all())
        skirt_table, _ = bone_weights(self.skirt, self.armature)
        self.assertEqual(skirt_table.width, 3)

    def test_width_without_five_influences(self):
        bpy.ops.wm.read_homefile(use_empty=True)
        mmd_model.build_model(five_influences=False)
        table, _ = bone_weights(bpy.data.objects["Body"], bpy.data.objects["TestArmature"])
        self.assertEqual(table.width, 2)

    def test_leg_weights_are_on_deform_d_bones(self):
        table, names = bone_weights(self.body, self.armature)
        rest = bone_rest(self.armature)
        used = np.unique(table.bone_ids[table.bone_ids >= 0])
        self.assertTrue(rest.deform[used].all())
        used_names = {names[i] for i in used}
        self.assertTrue({"左足D", "左ひざD", "左足首D", "左足先EX"} <= used_names)
        self.assertFalse({"左足", "左ひざ", "左足首"} & used_names)

    def test_non_bone_groups_and_zero_weights_are_ignored(self):
        before, _ = bone_weights(self.body, self.armature)
        extra = self.body.vertex_groups.new(name="まゆげ")
        extra.add(list(range(10)), 0.7, "REPLACE")
        self.body.vertex_groups.new(name="mmd_edge_scale").add([0, 1, 2], 1.0, "REPLACE")
        self.assertIsNone(self.body.vertex_groups.get("左足"))
        self.body.vertex_groups.new(name="左足").add([0, 1], 0.0, "REPLACE")  # 骨の名前だが、重みが 0
        after, _ = bone_weights(self.body, self.armature)
        np.testing.assert_array_equal(after.bone_ids, before.bone_ids)
        np.testing.assert_array_equal(after.weights, before.weights)
        self.assertEqual(non_bone_groups(self.body, self.armature), ("まゆげ", "mmd_edge_scale"))
        self.assertEqual(non_bone_groups(self.skirt, self.armature), ())

    def test_non_deform_bone_weights_are_kept(self):
        """形を動かさない骨のグループも入れる（分けるのは呼ぶ側。deform を使う）"""
        self.body.vertex_groups.new(name="左足").add([5], 0.5, "REPLACE")
        table, names = bone_weights(self.body, self.armature)
        row = dict(zip(table.bone_ids[5].tolist(), table.weights[5].tolist()))
        self.assertAlmostEqual(row[names.index("左足")], 0.5, places=6)


# MARK: - 形の計算と合うか（線形ブレンドスキニング）

# いくつかの骨を回したポーズ（骨のローカルの回転）
TEST_POSE = {
    "左腕": quaternion_about((1.0, 0.0, 0.0), 30.0),
    "右ひじ": quaternion_about((0.0, 0.0, 1.0), 45.0),
    "左手捩": quaternion_about((0.0, 1.0, 0.0), 60.0),
    "左手首": quaternion_about((1.0, 0.0, 1.0), 50.0),
    "左足": quaternion_about((1.0, 0.0, 0.0), 40.0),
    "右ひざ": quaternion_about((1.0, 0.0, 0.0), 70.0),
    "上半身": quaternion_about((0.0, 1.0, 0.0), 20.0),
    "下半身": quaternion_about((0.0, 0.0, 1.0), -15.0),
    "頭": quaternion_about((1.0, 0.0, 0.0), 15.0),
}


class SkinningTest(FixtureCase):
    def lbs(self, obj, space, table=None):
        rest = bone_rest(self.armature, space=space)
        if table is None:
            table, _ = bone_weights(obj, self.armature)
        vertices, _ = mesh_rest(obj, space=space)
        matrices = skinning_matrices(self.armature, space=space)[0]
        return linear_blend_skinning(vertices, normalized_deform_table(table, rest.deform), matrices,
                                     deform=rest.deform)

    def test_skinning_matrices_are_pose_times_inverse_rest(self):
        set_pose(self.armature, TEST_POSE)
        for space in ("armature", "world"):
            rest = bone_rest(self.armature, space=space)
            posed = pose_matrices(self.armature, space=space)[0]
            expected = np.array([p @ np.linalg.inv(r) for p, r in zip(posed, rest.matrices)])
            np.testing.assert_allclose(skinning_matrices(self.armature, space=space)[0], expected, atol=1e-9)
        reset_pose(self.armature)
        np.testing.assert_allclose(skinning_matrices(self.armature)[0], np.tile(np.eye(4), (96, 1, 1)), atol=1e-6)

    def test_lbs_reproduces_blender_in_armature_space(self):
        set_pose(self.armature, TEST_POSE)
        for obj in (self.body, self.skirt):
            with self.subTest(mesh=obj.name):
                evaluated = mesh_evaluated(obj, space="armature")
                rest, _ = mesh_rest(obj, space="armature")
                self.assertGreater(float(np.abs(evaluated - rest).max()), 0.1)  # ポーズで大きく動いた
                np.testing.assert_allclose(self.lbs(obj, "armature"), evaluated, atol=1e-5)

    def test_lbs_reproduces_blender_in_world_space(self):
        self.move_armature()
        set_pose(self.armature, TEST_POSE)
        for obj in (self.body, self.skirt):
            with self.subTest(mesh=obj.name):
                np.testing.assert_allclose(self.lbs(obj, "world"), mesh_evaluated(obj), atol=1e-5)
                # アーマチュアの中で計算して、あとで動かしても同じ
                np.testing.assert_allclose(moved(self.lbs(obj, "armature")), mesh_evaluated(obj), atol=1e-5)

    def test_wrong_conventions_do_not_match(self):
        """まちがった行列（休みの逆を掛けない・転置した）では合わない（このテストが意味を持つことの確かめ）"""
        set_pose(self.armature, TEST_POSE)
        rest = bone_rest(self.armature, space="armature")
        table, _ = bone_weights(self.body, self.armature)
        normalized = normalized_deform_table(table, rest.deform)
        vertices, _ = mesh_rest(self.body, space="armature")
        evaluated = mesh_evaluated(self.body, space="armature")
        posed = pose_matrices(self.armature, space="armature")[0]
        without_inverse = linear_blend_skinning(vertices, normalized, posed, deform=rest.deform)
        self.assertGreater(float(np.abs(without_inverse - evaluated).max()), 0.1)
        transposed = np.swapaxes(skinning_matrices(self.armature, space="armature")[0], 1, 2)
        with self.assertRaises(ValueError):
            linear_blend_skinning(vertices, normalized, transposed, deform=rest.deform)

    def test_four_influence_limit_has_small_effect(self):
        set_pose(self.armature, TEST_POSE)
        rest = bone_rest(self.armature, space="armature")
        table, _ = bone_weights(self.body, self.armature)
        limited, changed = normalize_and_limit(table, deform=rest.deform)
        five = sorted(self.info["five_influence_vertices"])
        self.assertEqual(np.flatnonzero(changed).tolist(), five)
        difference = np.linalg.norm(self.lbs(self.body, "armature", limited)
                                    - mesh_evaluated(self.body, space="armature"), axis=1)
        others = np.ones(len(difference), dtype=bool)
        others[five] = False
        self.assertLess(float(difference[others].max()), 1e-5)
        left = [i for i in five if i < self.info["parts"]["Body"]["forearm_R"][0]]
        self.assertTrue((difference[left] > 1e-4).all())  # 左の手首を回したので、左の頂点が動く
        self.assertTrue((difference[five] < 0.01).all())  # 小さい差


# MARK: - ポーズ


class PoseTest(FixtureCase):
    def test_rest_pose_matrices_equal_rest(self):
        for space in ("armature", "world"):
            posed = pose_matrices(self.armature, space=space)
            self.assertEqual(posed.shape, (1, 96, 4, 4))
            np.testing.assert_allclose(posed[0], bone_rest(self.armature, space=space).matrices, atol=1e-6)

    def test_rotations_identity_at_rest(self):
        self.move_armature()
        rotations = pose_rotations(self.armature)
        self.assertEqual(rotations.shape, (1, 96, 4))
        np.testing.assert_allclose(rotations[0], np.tile([1.0, 0.0, 0.0, 0.0], (96, 1)), atol=1e-6)

    def check_known_rotation(self, axis):
        """左腕 をローカルの X 軸まわりに 30 度回す。axis は、その軸の世界の向き（呼ぶ側が手で計算した物）"""
        rest = bone_rest(self.armature)
        set_pose(self.armature, {"左腕": quaternion_about((1.0, 0.0, 0.0), 30.0)})
        rotations = pose_rotations(self.armature)[0]
        np.testing.assert_allclose(rest.matrices[rest.index("左腕"), :3, 0], axis, atol=1e-6)
        expected = quaternion_about(axis, 30.0)
        for name in ("左腕", "左ひじ", "左手首", "左人指３"):  # 子は、親と同じだけ回る
            assert_same_rotation(self, rotations[rest.index(name)], expected)
        for name in ("右腕", "頭", "左肩"):
            assert_same_rotation(self, rotations[rest.index(name)], [1.0, 0.0, 0.0, 0.0])
        # 回転で、休みの骨の向きがポーズの骨の向きになる
        posed = pose_matrices(self.armature)[0]
        for name in ("左ひじ", "左手首"):
            i = rest.index(name)
            rest_dir = rest.matrices[i, :3, 1]
            np.testing.assert_allclose(kinematics.quat_rotate(rotations[i], rest_dir), posed[i, :3, 1], atol=1e-6)

    def test_known_rotation(self):
        self.check_known_rotation(np.array(self.armature.data.bones["左腕"].matrix_local)[:3, 0])

    def test_known_rotation_with_moved_armature(self):
        local_x = np.array(self.armature.data.bones["左腕"].matrix_local)[:3, 0]
        self.move_armature()
        self.check_known_rotation(moved(local_x) - np.array(MOVE))  # Z まわりに 90 度回した向き

    def test_d_bones_follow(self):
        set_pose(self.armature, {"左足": quaternion_about((1.0, 0.0, 0.0), 30.0),
                                 "右ひざ": quaternion_about((1.0, 0.0, 0.0), 45.0)})
        rest = bone_rest(self.armature)
        rotations = pose_rotations(self.armature)[0]
        for visible, deform in (("左足", "左足D"), ("左ひざ", "左ひざD"), ("右ひざ", "右ひざD"), ("右足首", "右足首D")):
            np.testing.assert_allclose(rotations[rest.index(deform)], rotations[rest.index(visible)], atol=1e-6)
        positions = joint_positions(self.armature, ["左ひざ", "左ひざD", "右足首", "右足首D"])[0]
        np.testing.assert_allclose(positions[1], positions[0], atol=1e-6)
        np.testing.assert_allclose(positions[3], positions[2], atol=1e-6)

    def test_knee_bend_from_thigh_contract(self):
        """pose_rotations は knee_bend_from_thigh の thigh_rotation として使える（ねじった太ももでも、曲がりが正しい）"""
        rest = bone_rest(self.armature)
        legs = ["左足D", "左ひざD", "左足首D"]
        hip, knee, _ = rest.heads[rest.indices(legs)]
        axis = kinematics.knee_rest_axis(hip, knee, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(axis, [1.0, 0.0, 0.0], atol=1e-9)

        def bend():
            points = joint_positions(self.armature, legs)[0]
            thigh = pose_rotations(self.armature)[0, rest.index("左足D")]
            return float(kinematics.knee_bend_from_thigh(points[0], points[1], points[2], rest_axis=axis,
                                                         thigh_rotation=thigh))

        at_rest = bend()
        self.assertTrue(0.0 < at_rest < 10.0)  # ひざが少し前に出ている（ふつうの向きに少し曲がっている）
        set_pose(self.armature, {"左ひざ": quaternion_about((1.0, 0.0, 0.0), 60.0)})
        self.assertAlmostEqual(bend(), at_rest + 60.0, delta=1e-3)
        set_pose(self.armature, {"左足": quaternion_about((0.0, 1.0, 0.0), 90.0)})  # 太ももを 90 度ねじる
        self.assertAlmostEqual(bend(), at_rest + 60.0, delta=1e-3)
        set_pose(self.armature, {"左ひざ": quaternion_about((1.0, 0.0, 0.0), -30.0)})  # 逆曲がり
        self.assertAlmostEqual(bend(), at_rest - 30.0, delta=1e-3)

    def test_joint_positions(self):
        self.move_armature()
        names = ["頭", "左手首", "右足首D"]
        rest = bone_rest(self.armature)
        positions = joint_positions(self.armature, names)
        self.assertEqual(positions.shape, (1, 3, 3))
        np.testing.assert_allclose(positions[0], rest.heads[rest.indices(names)], atol=1e-6)
        in_armature = joint_positions(self.armature, names, space="armature")
        np.testing.assert_allclose(moved(in_armature[0]), positions[0], atol=1e-6)
        with self.assertRaisesRegex(ValueError, "Head"):
            joint_positions(self.armature, ["頭", "Head"])
        with self.assertRaises(ValueError):
            joint_positions(self.armature, "頭")

    def test_quaternion_conversion_matches_mathutils(self):
        rotations = [Matrix.Identity(3)]
        for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, -2, 3), (-3, 1, 1)):
            for degrees in (10.0, 90.0, 179.0, 180.0, 250.0):
                rotations.append(Quaternion(axis, math.radians(degrees)).to_matrix())
        rotations.append(Euler((0.3, -1.2, 2.5), "ZXY").to_matrix())
        array = np.array([np.array(r) for r in rotations])
        converted = quaternions_from_rotations(array)
        self.assertTrue((converted[:, 0] >= 0.0).all())
        for matrix, q in zip(rotations, converted):
            assert_same_rotation(self, q, np.array(matrix.to_quaternion()), atol=1e-6)
            np.testing.assert_allclose(np.array(Quaternion(q).to_matrix()), np.array(matrix), atol=1e-6)


# MARK: - フレーム・ポーズ・モードを戻す


class FrameTest(FixtureCase):
    def setUp(self):
        super().setUp()
        bone = self.armature.pose.bones["左腕"]
        bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        bone.keyframe_insert("rotation_quaternion", frame=1)
        bone.rotation_quaternion = Quaternion((1.0, 0.0, 0.0), math.radians(60.0))
        bone.keyframe_insert("rotation_quaternion", frame=11)
        bpy.context.scene.frame_set(5)

    def test_frames(self):
        rest = bone_rest(self.armature)
        arm = rest.index("左腕")
        rotations = pose_rotations(self.armature, [1, 11])
        self.assertEqual(rotations.shape, (2, 96, 4))
        assert_same_rotation(self, rotations[0, arm], [1.0, 0.0, 0.0, 0.0])
        assert_same_rotation(self, rotations[1, arm], quaternion_about(rest.matrices[arm, :3, 0], 60.0))
        matrices = pose_matrices(self.armature, np.array([1, 11]))
        self.assertEqual(matrices.shape, (2, 96, 4, 4))
        np.testing.assert_allclose(matrices[0], rest.matrices, atol=1e-6)
        positions = joint_positions(self.armature, ["左手首"], frames=[1.0, 11.0])
        np.testing.assert_allclose(positions[:, 0], matrices[:, rest.index("左手首"), :3, 3], atol=1e-9)

    def test_frame_and_unkeyed_pose_are_restored(self):
        scene = bpy.context.scene
        bone = self.armature.pose.bones["左腕"]
        manual = Quaternion((0.0, 0.0, 1.0), math.radians(25.0))
        bone.rotation_quaternion = manual  # キーを打っていない変更
        self.armature.pose.bones["右腕"].location = (0.0, 0.01, 0.0)
        bpy.context.view_layer.update()
        before = pose_matrices(self.armature)
        pose_matrices(self.armature, [1, 11])
        self.assertEqual((scene.frame_current, scene.frame_subframe), (5, 0.0))
        np.testing.assert_allclose(tuple(bone.rotation_quaternion), tuple(manual), atol=1e-7)
        np.testing.assert_allclose(tuple(self.armature.pose.bones["右腕"].location), (0.0, 0.01, 0.0), atol=1e-9)
        np.testing.assert_allclose(pose_matrices(self.armature), before, atol=1e-7)

    def test_subframe_is_restored(self):
        scene = bpy.context.scene
        scene.frame_set(7, subframe=0.25)
        pose_matrices(self.armature, [2.5])
        self.assertEqual(scene.frame_current, 7)
        self.assertAlmostEqual(scene.frame_subframe, 0.25, places=6)

    def test_mode_active_and_selection_are_kept(self):
        view_layer = bpy.context.view_layer
        view_layer.objects.active = self.armature
        self.armature.select_set(True)
        self.body.select_set(False)
        bpy.ops.object.mode_set(mode="POSE")
        try:
            pose_matrices(self.armature, [1, 11])
            pose_rotations(self.armature, [3])
            joint_positions(self.armature, ["頭"], [4])
            bone_weights(self.body, self.armature)
            mesh_evaluated(self.body)
            self.assertEqual(self.armature.mode, "POSE")
            self.assertIs(view_layer.objects.active, self.armature)
            self.assertTrue(self.armature.select_get())
            self.assertFalse(self.body.select_get())
            self.assertEqual(bpy.context.scene.frame_current, 5)
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")

    def test_empty_and_bad_frames(self):
        self.assertEqual(pose_matrices(self.armature, []).shape, (0, 96, 4, 4))
        self.assertEqual(bpy.context.scene.frame_current, 5)
        for frames in ([float("nan")], [1, float("inf")], "12", [True], 3, [None], np.array(5), np.array([[1, 2]])):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                pose_matrices(self.armature, frames)
        self.assertEqual(bpy.context.scene.frame_current, 5)

    def test_data_is_not_changed(self):
        before = (raw_coordinates(self.body.data), [tuple(g.name for g in self.body.vertex_groups)],
                  len(self.body.modifiers))
        mesh_rest(self.body)
        mesh_evaluated(self.body)
        bone_weights(self.body, self.armature)
        pose_matrices(self.armature, [1, 11])
        after = (raw_coordinates(self.body.data), [tuple(g.name for g in self.body.vertex_groups)],
                 len(self.body.modifiers))
        np.testing.assert_array_equal(after[0], before[0])
        self.assertEqual(after[1:], before[1:])


# MARK: - 古い行列・計算されていない物


class EvaluationStateTest(FixtureCase):
    """depsgraph をまだ計算していない変更や、計算に入らない物（画面で無効など）を、古い値のまま黙って返さない"""

    def move_without_update(self):
        """アーマチュアを MOVE で動かすが、depsgraph は計算しない（元の物の matrix_world は古いまま）"""
        self.armature.location = MOVE
        self.armature.rotation_mode = "XYZ"
        self.armature.rotation_euler = (0.0, 0.0, math.radians(90.0))
        # テストの前提：matrix_world はまだ動いていない
        np.testing.assert_array_equal(np.array(self.armature.matrix_world), np.eye(4))

    def test_bone_rest_world_after_move_without_update(self):
        local = bone_rest(self.armature, space="armature")
        self.move_without_update()
        world = bone_rest(self.armature)
        np.testing.assert_allclose(world.heads, moved(local.heads), atol=1e-6)
        np.testing.assert_allclose(world.matrices[:, :3, 3], moved(local.heads), atol=1e-6)

    def test_pose_rotations_identity_after_move_without_update(self):
        self.move_without_update()
        np.testing.assert_allclose(pose_rotations(self.armature)[0], np.tile([1.0, 0.0, 0.0, 0.0], (96, 1)),
                                   atol=1e-6)

    def test_skinning_world_identity_after_move_without_update(self):
        self.move_without_update()
        np.testing.assert_allclose(skinning_matrices(self.armature, space="world")[0],
                                   np.tile(np.eye(4), (96, 1, 1)), atol=1e-6)

    def test_mesh_rest_after_move_without_update(self):
        local = raw_coordinates(self.body.data)
        self.move_without_update()
        np.testing.assert_allclose(mesh_rest(self.body)[0], moved(local), atol=1e-6)
        np.testing.assert_allclose(mesh_rest(self.body, space="armature")[0], local, atol=1e-6)
        np.testing.assert_allclose(mesh_evaluated(self.body), moved(local), atol=1e-6)

    def test_mesh_disabled_in_viewport_is_rejected(self):
        """画面で無効（hide_viewport）のメッシュは計算されない。書き出しには入る（find_model）が、形は読めない"""
        set_pose(self.armature, {"左腕": quaternion_about((1.0, 0.0, 0.0), 30.0)})
        self.body.hide_viewport = True
        bpy.context.view_layer.update()
        self.assertIn(self.body, find_model().meshes)
        with self.assertRaisesRegex(ValueError, "計算されていません"):
            mesh_evaluated(self.body)
        with self.assertRaisesRegex(ValueError, "計算されていません"):
            mesh_rest(self.body)
        # 計算されている物は、そのまま読める
        self.assertEqual(mesh_evaluated(self.skirt).shape, (self.info["vertex_counts"]["Skirt"], 3))

    def test_mesh_in_excluded_collection_is_rejected(self):
        collection = bpy.data.collections.new("Excluded")
        bpy.context.scene.collection.children.link(collection)
        bpy.context.scene.collection.objects.unlink(self.body)
        collection.objects.link(self.body)
        bpy.context.view_layer.layer_collection.children["Excluded"].exclude = True
        set_pose(self.armature, {"左腕": quaternion_about((1.0, 0.0, 0.0), 30.0)})
        with self.assertRaisesRegex(ValueError, "計算されていません"):
            mesh_evaluated(self.body)

    def test_armature_not_evaluated_is_rejected(self):
        """アーマチュアも、それを使う物も全部、画面で無効だと、ポーズは計算されない（古いポーズを返さない）"""
        for obj in (self.armature, self.body, self.skirt):
            obj.hide_viewport = True
        bpy.context.view_layer.update()
        set_pose(self.armature, {"右腕": quaternion_about((1.0, 0.0, 0.0), 30.0)})
        for read in (lambda: pose_matrices(self.armature), lambda: pose_matrices(self.armature, [1, 2]),
                     lambda: pose_rotations(self.armature), lambda: joint_positions(self.armature, ["右手首"]),
                     lambda: skinning_matrices(self.armature), lambda: bone_rest(self.armature)):
            with self.assertRaisesRegex(ValueError, "計算されていません"):
                read()
        # アーマチュアの中の休みの形は、行列を使わないので読める
        self.assertEqual(bone_rest(self.armature, space="armature").bone_count, 96)

    def test_hidden_armature_used_by_visible_mesh_is_evaluated(self):
        """アーマチュアだけを画面で無効にしても、見えるメッシュが使っていれば計算される（読める）"""
        self.armature.hide_viewport = True
        bpy.context.view_layer.update()
        set_pose(self.armature, {"右腕": quaternion_about((1.0, 0.0, 0.0), 30.0)})
        rest = bone_rest(self.armature)
        rotation = pose_rotations(self.armature)[0, rest.index("右腕")]
        assert_same_rotation(self, rotation, quaternion_about(rest.matrices[rest.index("右腕"), :3, 0], 30.0))

    def test_rest_position_is_rejected_for_poses(self):
        """アーマチュアが「レストポジション」だと、ポーズもアニメーションも使われない。ポーズとして読ませない"""
        set_pose(self.armature, {"左ひざ": quaternion_about((1.0, 0.0, 0.0), -40.0)})
        self.armature.data.pose_position = "REST"
        bpy.context.view_layer.update()
        for read in (lambda: pose_matrices(self.armature), lambda: pose_rotations(self.armature, [1]),
                     lambda: joint_positions(self.armature, ["左足首"]), lambda: skinning_matrices(self.armature)):
            with self.assertRaisesRegex(ValueError, "レストポジション"):
                read()
        self.armature.data.pose_position = "POSE"
        bpy.context.view_layer.update()
        self.assertGreater(float(np.abs(pose_rotations(self.armature)[0] - [1.0, 0.0, 0.0, 0.0]).max()), 0.1)

    def test_rest_shape_is_the_basis_shape_key(self):
        """シェイプキーがあると、Blender は基本のキー（Basis）の形から計算する。休みの形もそれにする"""
        reference = self.body.data.shape_keys.reference_key
        expected = np.array([tuple(point.co) for point in reference.data], dtype=np.float64)
        self.body.data.vertices[0].co.z += 0.1  # スクリプトで頂点だけを動かす（基本のキーとずれる）
        bpy.context.view_layer.update()
        vertices, _ = mesh_rest(self.body, space="armature")
        np.testing.assert_allclose(vertices, expected, atol=1e-7)
        np.testing.assert_allclose(mesh_evaluated(self.body, space="armature"), vertices, atol=1e-6)


# MARK: - ポーズを決める


class SetPoseTest(FixtureCase):
    def test_set_and_reset(self):
        set_pose(self.armature, {"頭": [2.0, 0.0, 0.0, 0.0], "左腕": quaternion_about((0.0, 0.0, 1.0), 20.0)})
        np.testing.assert_allclose(tuple(self.armature.pose.bones["頭"].rotation_quaternion), (1.0, 0.0, 0.0, 0.0))
        np.testing.assert_allclose(tuple(self.armature.pose.bones["左腕"].rotation_quaternion),
                                   quaternion_about((0.0, 0.0, 1.0), 20.0), atol=1e-7)
        self.armature.pose.bones["右足"].location = (0.0, 0.0, 0.1)
        self.armature.pose.bones["右足"].scale = (1.0, 2.0, 1.0)
        reset_pose(self.armature)
        np.testing.assert_allclose(pose_rotations(self.armature)[0], np.tile([1.0, 0.0, 0.0, 0.0], (96, 1)),
                                   atol=1e-6)
        np.testing.assert_allclose(pose_matrices(self.armature, space="armature")[0],
                                   bone_rest(self.armature, space="armature").matrices, atol=1e-6)

    def test_euler_and_axis_angle_bones(self):
        expected_q = quaternion_about((1.0, 2.0, 0.5), 40.0)
        set_pose(self.armature, {"左腕": expected_q})
        expected = pose_rotations(self.armature)[0]
        for mode in ("XYZ", "ZXY", "AXIS_ANGLE"):
            with self.subTest(mode=mode):
                reset_pose(self.armature)
                self.armature.pose.bones["左腕"].rotation_mode = mode
                set_pose(self.armature, {"左腕": expected_q})
                np.testing.assert_allclose(pose_rotations(self.armature)[0], expected, atol=1e-6)

    def test_rejects_bad_input_without_changing_anything(self):
        good = quaternion_about((1.0, 0.0, 0.0), 30.0)
        for rotations in ({"左腕": good, "Head": good}, {"左腕": good, "頭": [0.0, 0.0, 0.0, 0.0]},
                          {"左腕": good, "頭": [1.0, 0.0, 0.0]}, {"左腕": good, "頭": [float("nan"), 0, 0, 1]},
                          {"左腕": good, "頭": "q"}):
            with self.subTest(rotations=list(rotations)), self.assertRaises(ValueError):
                set_pose(self.armature, rotations)
            np.testing.assert_allclose(tuple(self.armature.pose.bones["左腕"].rotation_quaternion),
                                       (1.0, 0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            set_pose(self.armature, [("左腕", good)])
        with self.assertRaises(ValueError):
            set_pose(self.body, {"左腕": good})

    def test_pose_kept_restores_unkeyed_pose(self):
        """pose_kept：中で付けたポーズ（キーの無い変更）と、アーマチュアのオブジェクトの位置を、元に戻す。例外のときも"""
        arm = self.armature.pose.bones["左腕"]
        neck = self.armature.pose.bones["首"]
        neck.rotation_mode = "XZY"
        neck.rotation_euler = (0.1, -0.2, 0.3)
        arm.location = (0.01, 0.0, 0.0)
        bpy.context.view_layer.update()
        before = {bone.name: (tuple(bone.location), tuple(bone.rotation_quaternion), tuple(bone.rotation_euler),
                              tuple(bone.scale)) for bone in self.armature.pose.bones}
        joints_before = joint_positions(self.armature, ["左手首"])
        with self.assertRaises(RuntimeError):
            with extract_bpy.pose_kept(self.armature):
                reset_pose(self.armature)
                set_pose(self.armature, {"左腕": quaternion_about((0.0, 1.0, 0.0), 50.0),
                                         "首": quaternion_about((0.0, 0.0, 1.0), 20.0)})
                self.armature.location = (0.5, 0.0, 0.0)
                bpy.context.view_layer.update()
                self.assertGreater(np.abs(joint_positions(self.armature, ["左手首"]) - joints_before).max(), 0.05)
                raise RuntimeError("途中で止まった")
        after = {bone.name: (tuple(bone.location), tuple(bone.rotation_quaternion), tuple(bone.rotation_euler),
                             tuple(bone.scale)) for bone in self.armature.pose.bones}
        self.assertEqual(after, before)
        self.assertEqual(tuple(self.armature.location), (0.0, 0.0, 0.0))
        self.assertEqual(neck.rotation_mode, "XZY")
        np.testing.assert_allclose(joint_positions(self.armature, ["左手首"]), joints_before, atol=1e-7)
        with self.assertRaises(ValueError):
            with extract_bpy.pose_kept(self.body):
                pass


if __name__ == "__main__":
    unittest.main()

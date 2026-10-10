"""E2 往復の確認の計算（blender_ai_tools/export/roundtrip.py）を、手で作った並びで確かめる（bpy なし）。

答えは手で計算した値。行列の向き（行ベクトルと列ベクトル）・空間（geomBind を使うか）・骨の並び（skel:joints）・
USD の骨の親の決まりのまちがいで外れる値を選ぶ。
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from blender_ai_tools import thresholds
from blender_ai_tools.export import roundtrip
from blender_ai_tools.export.roundtrip import (ExportedBlendShape, ExportedMesh, ExportedModel, PointMatch,
                                               check_coverage, check_influences, check_names, check_no_physics,
                                               check_skeleton, compare_pose, compare_rest, compose_rest,
                                               exported_weight_table, fan_triangulate, gf_to_column, joint_parents,
                                               load_rig, local_rotation, match_meshes, match_points,
                                               original_names, path_gaps, posed_joint_matrices, resolve_pose,
                                               rest_points, skin_exported)
from blender_ai_tools.measure.results import Signal

LIMIT = thresholds.ROUNDTRIP_MAX_DEVIATION


def translation(x, y, z):
    matrix = np.eye(4)
    matrix[:3, 3] = (x, y, z)
    return matrix


def rotation(axis, degrees, about=(0.0, 0.0, 0.0)):
    """点 about を中心に、軸 axis のまわりに degrees 度回す行列（列ベクトルの形。右手の向き）"""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    t = math.radians(degrees)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    r = np.eye(3) + math.sin(t) * k + (1.0 - math.cos(t)) * (k @ k)
    matrix = np.eye(4)
    matrix[:3, :3] = r
    c = np.asarray(about, dtype=np.float64)
    matrix[:3, 3] = c - r @ c
    return matrix


# MARK: - 2本の骨の見本

# 骨格：root（原点）と arm（root の子、根元は (0, 0, 1)）
JOINT_PATHS = ("root", "root/arm")
BIND = np.stack([np.eye(4), translation(0.0, 0.0, 1.0)])
# arm を、根元 (0, 0, 1) を中心に X のまわりに 90 度回したポーズ：(x, y, z) → (x, -z', y')（根元から見て）
POSED = np.stack([np.eye(4), translation(0.0, 0.0, 1.0) @ rotation((1, 0, 0), 90.0)])
# 形の位置：骨格の空間で (0.5, 0, 0) にある（geomBind）。点はメッシュの中の座標
GEOM_BIND = translation(0.5, 0.0, 0.0)
POINTS_LOCAL = np.array([[-0.5, 0.0, 2.0], [-0.5, 0.0, 2.0], [-0.5, 0.0, 0.5]])
# 骨格の空間の休みの点
REST_SKEL = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.5]])
# 点0 は arm に 1.0、点1 は root と arm に半分ずつ、点2 は root に 1.0（2つ目の枠は「番号 1・重み 0」の空き）
JOINT_INDICES = np.array([[1, 0], [0, 1], [0, 1]])
JOINT_WEIGHTS = np.array([[1.0, 0.0], [0.5, 0.5], [1.0, 0.0]])
# 手で計算した、ポーズの点：
#   点0：根元から (0, 0, 1) → X のまわりに 90 度で (0, -1, 0) → 根元を足して (0, -1, 1)
#   点1：0.5 × (0, 0, 2) + 0.5 × (0, -1, 1) = (0, -0.5, 1.5)
#   点2：root は動かないので (0, 0, 0.5)
POSED_SKEL = np.array([[0.0, -1.0, 1.0], [0.0, -0.5, 1.5], [0.0, 0.0, 0.5]])
# 三角形1つ（点の並びだけを使う）
TRIANGLES = np.array([[0, 1, 2]])


def make_mesh(**changes):
    values = dict(path="/root/Arm/Body/Body", name="Body", points=POINTS_LOCAL, triangles=TRIANGLES,
                  geom_bind=GEOM_BIND, joint_indices=JOINT_INDICES, joint_weights=JOINT_WEIGHTS,
                  joint_names=("root", "arm"), source_name="Body")
    values.update(changes)
    return ExportedMesh(**values)


def make_model(meshes=None, **changes):
    values = dict(joint_paths=JOINT_PATHS, bind=BIND, rest_local=np.stack([np.eye(4), translation(0, 0, 1)]),
                  meshes=(make_mesh(),) if meshes is None else tuple(meshes))
    values.update(changes)
    return ExportedModel(**values)


RIG = {
    "version": 1,
    "bones": {"hips": "root"},
    "expressions": {"a": [{"shapeKey": "a", "weight": 1.0}]},
    "originalNames": {"bones": {"root": "センター", "arm": "左腕"}, "shapeKeys": {"a": "あ"}, "materials": {}},
}


# MARK: - 行列の向き


class MatrixConventionTest(unittest.TestCase):
    def test_translation_in_last_row_moves_to_last_column(self):
        rows = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 2, 3, 1]]  # Gf：平行移動は下の行
        matrix = gf_to_column(rows)
        np.testing.assert_array_equal(matrix[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(matrix[3], [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_equal(matrix @ [0.0, 0.0, 0.0, 1.0], [1.0, 2.0, 3.0, 1.0])

    def test_rotation_rows_are_images_of_axes(self):
        # Gf の行 i は、軸 i が行く先。Z のまわりに 90 度：X → Y
        rows = [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        matrix = gf_to_column(rows)
        np.testing.assert_allclose(matrix @ [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0])
        np.testing.assert_allclose(matrix, rotation((0, 0, 1), 90.0), atol=1e-15)

    def test_stack_of_matrices(self):
        rows = np.stack([np.eye(4), translation(4, 5, 6).T])
        np.testing.assert_array_equal(gf_to_column(rows)[1], translation(4, 5, 6))

    def test_untransposed_matrix_is_rejected(self):
        with self.assertRaises(ValueError):
            make_model(bind=np.stack([np.eye(4), translation(0, 0, 1).T]))
        with self.assertRaises(ValueError):
            skin_exported(make_mesh(), BIND, np.stack([np.eye(4), POSED[1].T]))

    def test_bad_shape(self):
        with self.assertRaises(ValueError):
            gf_to_column(np.eye(3))


# MARK: - 骨格の並び


class TopologyTest(unittest.TestCase):
    def test_nearest_listed_ancestor_is_parent(self):
        paths = ["a", "a/b", "a/b/c/d", "x", "x/y"]
        np.testing.assert_array_equal(joint_parents(paths), [-1, 0, 1, -1, 3])

    def test_path_gaps(self):
        self.assertEqual(path_gaps(["a", "a/b", "a/b/c/d"]), {"a/b/c/d": ("c",)})
        self.assertEqual(path_gaps(["p/q/r"]), {"p/q/r": ("p", "q")})
        self.assertEqual(path_gaps(["a", "a/b"]), {})

    def test_bad_paths(self):
        with self.assertRaises(ValueError):
            joint_parents(["a", "a"])
        with self.assertRaises(ValueError):
            joint_parents(["a//b"])
        with self.assertRaises(ValueError):
            joint_parents("a/b")

    def test_compose_rest_multiplies_parent_first(self):
        local = np.stack([translation(0, 0, 1), translation(1, 0, 0) @ rotation((0, 0, 1), 90.0)])
        world = compose_rest(local, np.array([-1, 0]))
        np.testing.assert_allclose(world[0], translation(0, 0, 1))
        expected = translation(1, 0, 1) @ rotation((0, 0, 1), 90.0)
        np.testing.assert_allclose(world[1], expected, atol=1e-15)
        # 子の X 軸は Y を向く（親の後に子を掛けたとき）
        np.testing.assert_allclose(world[1][:3, 0], [0.0, 1.0, 0.0], atol=1e-15)

    def test_compose_rest_rejects_parent_after_child(self):
        with self.assertRaises(ValueError):
            compose_rest(np.stack([np.eye(4), np.eye(4)]), np.array([1, -1]))
        with self.assertRaises(ValueError):
            compose_rest(np.stack([np.eye(4)]), np.array([0]))


# MARK: - 面を三角形にする


class TriangulateTest(unittest.TestCase):
    def test_fan(self):
        triangles = fan_triangulate([3, 4, 5], list(range(12)), 12)
        np.testing.assert_array_equal(triangles, [[0, 1, 2], [3, 4, 5], [3, 5, 6], [7, 8, 9], [7, 9, 10],
                                                  [7, 10, 11]])
        self.assertEqual(triangles.dtype, np.int64)

    def test_left_handed_reverses_winding(self):
        triangles = fan_triangulate([3, 4], [0, 1, 2, 3, 4, 5, 6], 7, left_handed=True)
        np.testing.assert_array_equal(triangles, [[0, 2, 1], [3, 5, 4], [3, 6, 5]])

    def test_degenerate_faces_skipped(self):
        np.testing.assert_array_equal(fan_triangulate([2, 3], [0, 1, 0, 1, 2], 3), [[0, 1, 2]])

    def test_errors(self):
        with self.assertRaises(ValueError):
            fan_triangulate([3, 3], [0, 1, 2], 3)
        with self.assertRaises(ValueError):
            fan_triangulate([3], [0, 1, 5], 3)
        self.assertEqual(fan_triangulate([], [], 0).shape, (0, 3))


# MARK: - 骨で動かす


class SkinTest(unittest.TestCase):
    def test_rest_points_use_geom_bind(self):
        np.testing.assert_allclose(rest_points(make_mesh()), REST_SKEL)
        np.testing.assert_allclose(skin_exported(make_mesh(), BIND, BIND), REST_SKEL)

    def test_posed_points_exact(self):
        np.testing.assert_allclose(skin_exported(make_mesh(), BIND, POSED), POSED_SKEL, atol=1e-12)

    def test_wrong_conventions_would_differ(self):
        """行列を転置したまま（逆の回り方）や、geomBind を使わないと、答えが変わる（この見本で見分けられる）"""
        transposed_rotation = np.stack([np.eye(4), translation(0, 0, 1) @ rotation((1, 0, 0), -90.0)])
        wrong = skin_exported(make_mesh(), BIND, transposed_rotation)
        self.assertGreater(np.abs(wrong - POSED_SKEL).max(), 1.0)
        no_bind = skin_exported(make_mesh(geom_bind=np.eye(4)), BIND, POSED)
        self.assertGreater(np.abs(no_bind - POSED_SKEL).max(), 0.4)

    def test_mesh_joint_order(self):
        """skel:joints があるときは、その並びの番号を骨格の番号に読みかえる"""
        mesh = make_mesh(joint_indices=1 - JOINT_INDICES, joint_names=("arm", "root"),
                         joint_map=np.array([1, 0]))
        np.testing.assert_allclose(skin_exported(mesh, BIND, POSED), POSED_SKEL, atol=1e-12)

    def test_zero_weight_slots_are_ignored(self):
        table = exported_weight_table(make_mesh(), 2)
        np.testing.assert_array_equal(table.bone_ids, [[1, -1], [0, 1], [0, -1]])
        np.testing.assert_array_equal(table.weights, [[1.0, 0.0], [0.5, 0.5], [1.0, 0.0]])

    def test_weights_used_as_written(self):
        """重みの合計で割らない（合計 0.5 なら、動いた点も半分）"""
        mesh = make_mesh(joint_weights=np.array([[0.5, 0.0], [0.5, 0.5], [1.0, 0.0]]))
        np.testing.assert_allclose(skin_exported(mesh, BIND, POSED)[0], 0.5 * POSED_SKEL[0], atol=1e-12)

    def test_unskinned_mesh_does_not_move(self):
        mesh = make_mesh(skinned=False, joint_indices=None, joint_weights=None, joint_names=(),
                         local_to_skeleton=translation(1, 0, 0))
        np.testing.assert_allclose(skin_exported(mesh, BIND, POSED), POINTS_LOCAL + [1.0, 0.0, 0.0])

    def test_weight_on_joint_missing_from_skeleton(self):
        mesh = make_mesh(joint_names=("root", "other"), joint_map=np.array([0, -1]))
        with self.assertRaises(ValueError):
            skin_exported(mesh, BIND, POSED)

    def test_posed_count_must_match(self):
        with self.assertRaises(ValueError):
            skin_exported(make_mesh(), BIND, POSED[:1])


# MARK: - 点の対応


GRID = np.array([[x, y, z] for x in (0.0, 0.1) for y in (0.0, 0.2) for z in (0.0, 0.3)])


class MatchPointsTest(unittest.TestCase):
    def test_same_order(self):
        match = match_points(GRID, GRID)
        self.assertEqual(match.method, "order")
        np.testing.assert_array_equal(match.index, np.arange(8))
        self.assertEqual(len(match.unmatched_source), 0)
        np.testing.assert_array_equal(match.distance, np.zeros(8))

    def test_masked_vertex_with_candidates(self):
        hidden = np.zeros(8, dtype=bool)
        hidden[2] = True
        match = match_points(GRID, GRID[~hidden], candidates=~hidden)
        self.assertEqual(match.method, "order")
        np.testing.assert_array_equal(match.index, [0, 1, 3, 4, 5, 6, 7])
        np.testing.assert_array_equal(match.unmatched_source, [2])

    def test_deleted_vertex_without_candidates_uses_nearest(self):
        keep = [0, 1, 3, 4, 5, 6, 7]
        match = match_points(GRID, GRID[keep])
        self.assertEqual(match.method, "nearest")
        np.testing.assert_array_equal(match.index, keep)
        np.testing.assert_array_equal(match.unmatched_source, [2])

    def test_scrambled_order_uses_nearest(self):
        order = [7, 3, 5, 1, 6, 2, 4, 0]
        match = match_points(GRID, GRID[order])
        self.assertEqual(match.method, "nearest")
        np.testing.assert_array_equal(match.index, order)

    def test_small_shift_stays_matched_and_is_measured(self):
        moved = GRID.copy()
        moved[3, 0] += 0.0003
        match = match_points(GRID, moved)
        self.assertEqual(match.method, "order")
        self.assertEqual(match.index[3], 3)
        self.assertAlmostEqual(match.distance[3], 0.0003, places=12)

    def test_whole_shape_shift_is_kept_in_order(self):
        """全体がずれた形は「対応なし」にせず、並びのまま対応させて、ずれとして測る"""
        match = match_points(GRID, GRID + [0.01, 0.0, 0.0])
        self.assertEqual(match.method, "order")
        np.testing.assert_allclose(match.distance, np.full(8, 0.01))

    def test_far_point_in_nearest_mode_is_unmatched(self):
        exported = np.vstack([GRID[:7], [[5.0, 5.0, 5.0]], GRID[7:]])
        match = match_points(GRID, exported)
        self.assertEqual(match.method, "nearest")
        self.assertEqual(match.index[7], -1)
        np.testing.assert_array_equal(match.index[[0, 6, 8]], [0, 6, 7])
        self.assertFalse(match.matched[7])

    def test_coincident_source_vertices_keep_order(self):
        source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        match = match_points(source, source)
        np.testing.assert_array_equal(match.index, [0, 1, 2, 3])

    def test_coincident_source_vertices_in_nearest_mode(self):
        """UV の切れ目で分かれた頂点（同じ座標）は、いちばん近い点で対応させても、1つずつ別の頂点に割り振る"""
        source = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        exported = source[[3, 0, 2, 1]]  # 並びが違い、頂点 4 が無い
        match = match_points(source, exported)
        self.assertEqual(match.method, "nearest")
        np.testing.assert_array_equal(match.index, [3, 0, 2, 1])
        np.testing.assert_array_equal(match.unmatched_source, [4])
        # 同じ座標の頂点より多く対応したときは、いちばん小さい番号のまま（2つが同じ頂点に対応する）
        crowded = match_points(source[:3], np.array([[1.0, 0.0, 0.0]] * 3 + [[0.0, 0.0, 0.0]]))
        np.testing.assert_array_equal(crowded.index, [0, 2, 0, 1])

    def test_empty(self):
        match = match_points(GRID, np.zeros((0, 3)))
        self.assertEqual(len(match.index), 0)
        np.testing.assert_array_equal(match.unmatched_source, np.arange(8))

    def test_bad_candidates(self):
        with self.assertRaises(ValueError):
            match_points(GRID, GRID, candidates=np.ones(8))


# MARK: - 形のずれ


def rest_setup(mesh=None):
    model = make_model(meshes=None if mesh is None else [mesh])
    source_rest = {"Body": REST_SKEL.copy()}
    matches = match_meshes(model, source_rest)
    return model, source_rest, matches


class CompareTest(unittest.TestCase):
    def test_rest_is_green(self):
        model, source_rest, matches = rest_setup()
        result = compare_rest(model, source_rest, BIND, matches)
        self.assertEqual(result.signal, Signal.GREEN)
        self.assertEqual(result.value, 0.0)
        self.assertEqual(result.details["pose"], "rest")
        self.assertEqual(result.details["measured_count"], 3)

    def test_exact_pose_is_green(self):
        model, _, matches = rest_setup()
        result = compare_pose(model, {"Body": POSED_SKEL}, POSED, matches, pose="bend", rest=BIND)
        self.assertEqual(result.signal, Signal.GREEN)
        self.assertLess(result.value, 1e-12)
        self.assertEqual(result.details["moved_joint_count"], 1)

    def test_deviation_value_and_worst(self):
        model, _, matches = rest_setup()
        blender = POSED_SKEL.copy()
        blender[1, 0] += 0.0003
        result = compare_pose(model, {"Body": blender}, POSED, matches, pose="bend")
        self.assertEqual(result.signal, Signal.GREEN)
        self.assertAlmostEqual(result.value, 0.0003, places=12)
        mesh = result.details["meshes"]["/root/Arm/Body/Body"]
        self.assertEqual(int(mesh["worst_exported"][0]), 1)
        self.assertEqual(int(mesh["worst_source"][0]), 1)
        self.assertAlmostEqual(mesh["deviations"][1], 0.0003, places=12)

        blender[1, 0] += 0.0003  # 0.6mm
        result = compare_pose(model, {"Body": blender}, POSED, matches, pose="bend")
        self.assertEqual(result.signal, Signal.RED)
        self.assertAlmostEqual(result.value, 0.0006, places=12)
        self.assertEqual(result.details["over_limit_count"], 1)
        self.assertEqual(result.details["worst"][0]["source_index"], 1)

    def test_unknown_joint_makes_points_unmeasured(self):
        model, _, matches = rest_setup()
        posed = POSED.copy()
        posed[1] = np.nan
        result = compare_pose(model, {"Body": POSED_SKEL}, posed, matches, pose="bend")
        self.assertEqual(result.signal, Signal.GRAY)
        self.assertEqual(result.details["unmeasured_count"], 2)
        self.assertEqual(result.details["unknown_joints"], ("arm",))
        deviations = result.details["meshes"]["/root/Arm/Body/Body"]["deviations"]
        self.assertTrue(np.isnan(deviations[:2]).all())
        self.assertEqual(deviations[2], 0.0)
        self.assertTrue(result.reason)

    def test_red_beats_unmeasured(self):
        model, _, matches = rest_setup()
        posed = POSED.copy()
        posed[1] = np.nan
        blender = POSED_SKEL.copy()
        blender[2, 2] += 0.001
        self.assertEqual(compare_pose(model, {"Body": blender}, posed, matches, pose="bend").signal, Signal.RED)

    def test_nothing_measured_is_gray(self):
        model, _, matches = rest_setup()
        result = compare_pose(model, {"Body": POSED_SKEL}, np.full((2, 4, 4), np.nan), matches, pose="bend")
        self.assertEqual(result.signal, Signal.GRAY)
        self.assertIsNone(result.value)

    def test_pose_that_moved_nothing_is_gray(self):
        model, _, matches = rest_setup()
        result = compare_pose(model, {"Body": REST_SKEL}, BIND, matches, pose="bend", rest=BIND)
        self.assertEqual(result.signal, Signal.GRAY)
        self.assertEqual(result.details["moved_joint_count"], 0)

    def test_red_beats_pose_that_moved_nothing(self):
        """骨が動かなかったポーズでも、測れた点が基準を超えていれば赤（赤 > 灰。赤を灰で隠さない）"""
        model, _, matches = rest_setup()
        blender = REST_SKEL.copy()
        blender[2, 2] += 0.0006
        result = compare_pose(model, {"Body": blender}, BIND, matches, pose="bend", rest=BIND)
        self.assertEqual(result.signal, Signal.RED)
        self.assertAlmostEqual(result.value, 0.0006, places=12)
        self.assertEqual(result.details["moved_joint_count"], 0)
        self.assertIn("動いた骨がありません", result.reason)

    def test_missing_source_mesh_is_unmeasured(self):
        model, _, matches = rest_setup()
        result = compare_pose(model, {}, POSED, matches, pose="bend")
        self.assertEqual(result.signal, Signal.GRAY)

    def test_wrong_bind_shows_at_rest(self):
        """bindTransforms が元の休みの行列と違えば、休みの姿勢でずれる"""
        model, source_rest, matches = rest_setup()
        blender_rest = np.stack([np.eye(4), translation(0.0, 0.0, 1.001)])
        result = compare_rest(model, source_rest, blender_rest, matches)
        self.assertEqual(result.signal, Signal.RED)
        self.assertAlmostEqual(result.value, 0.001, places=12)

    def test_source_count_must_match(self):
        model, _, matches = rest_setup()
        with self.assertRaises(ValueError):
            compare_pose(model, {"Body": POSED_SKEL[:2]}, POSED, matches, pose="bend")


# MARK: - 長さはアプリのメートル

# Z が上から Y が上へ（X のまわりに -90 度）。convert_blend.py の /root と同じ
Z_UP_TO_Y_UP = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def scaled(x, y, z):
    matrix = np.eye(4)
    matrix[0, 0], matrix[1, 1], matrix[2, 2] = x, y, z
    return matrix


class MetersTest(unittest.TestCase):
    """ずれは、骨格の空間の長さではなく、アプリで見える長さ（メートル）で測る。

    アーマチュアのオブジェクトに大きさ（スケール）があると、骨格の空間の 1 は 1m ではない。
    アプリは骨格の空間を skeleton_world（と metersPerUnit）で大きくして見せる
    """

    def scaled_setup(self, skeleton_world, meters_per_unit=1.0):
        model = make_model(skeleton_world=skeleton_world, meters_per_unit=meters_per_unit)
        source_rest = {"Body": REST_SKEL.copy()}
        return model, source_rest, match_meshes(model, source_rest)

    def test_uniform_scale_doubles_deviation(self):
        # アーマチュアの大きさ 2（Y が上への回しも入れる。回しは長さを変えない）
        model, _, matches = self.scaled_setup(Z_UP_TO_Y_UP @ translation(1, 2, 3) @ scaled(2, 2, 2))
        blender = POSED_SKEL.copy()
        blender[1, 0] += 0.0003  # 骨格の空間で 0.3（アプリでは 0.6mm）
        result = compare_pose(model, {"Body": blender}, POSED, matches, pose="bend")
        self.assertAlmostEqual(result.value, 0.0006, places=12)
        self.assertEqual(result.signal, Signal.RED)
        mesh = result.details["meshes"]["/root/Arm/Body/Body"]
        self.assertAlmostEqual(mesh["deviations"][1], 0.0006, places=12)
        self.assertAlmostEqual(mesh["max"], 0.0006, places=12)

    def test_meters_per_unit(self):
        # 骨格の空間の 1 が 100 × 0.01 = 1m
        model, _, matches = self.scaled_setup(scaled(100, 100, 100), meters_per_unit=0.01)
        blender = POSED_SKEL.copy()
        blender[1, 0] += 0.0003
        result = compare_pose(model, {"Body": blender}, POSED, matches, pose="bend")
        self.assertAlmostEqual(result.value, 0.0003, places=12)
        self.assertEqual(result.signal, Signal.GREEN)

    def test_non_uniform_scale_uses_the_direction(self):
        model, _, matches = self.scaled_setup(scaled(1, 1, 3))
        blender = POSED_SKEL.copy()
        blender[1, 2] += 0.0001  # Z の向きは 3 倍
        blender[2, 0] += 0.0002  # X の向きは 1 倍
        result = compare_pose(model, {"Body": blender}, POSED, matches, pose="bend")
        deviations = result.details["meshes"]["/root/Arm/Body/Body"]["deviations"]
        np.testing.assert_allclose(deviations, [0.0, 0.0003, 0.0002], atol=1e-12)

    def test_rest_and_match_distances_in_meters(self):
        model, source_rest, matches = self.scaled_setup(scaled(2, 2, 2))
        shifted = {"Body": REST_SKEL + [0.0, 0.0, 0.0003]}  # 骨格の空間で 0.3（アプリでは 0.6mm）
        matches = match_meshes(model, shifted)
        np.testing.assert_allclose(matches["/root/Arm/Body/Body"].distance, np.full(3, 0.0006), atol=1e-12)
        result = compare_rest(model, shifted, BIND, matches)
        self.assertAlmostEqual(result.value, 0.0006, places=12)
        self.assertEqual(result.signal, Signal.RED)
        # 対応の距離もメートル：骨格の空間で 0.006mm の差は、アプリでは 0.012mm で、対応の距離（0.01mm）を超える
        distinct = ExportedModel(joint_paths=JOINT_PATHS, bind=BIND, rest_local=None,
                                 meshes=distinct_model().meshes, skeleton_world=scaled(2, 2, 2))
        source = {"Body": np.vstack([DISTINCT, [[5.0, 5.0, 5.0]]])}  # 4 つ目は書き出されていない頂点
        source["Body"][0, 0] += 0.000006
        far = match_meshes(distinct, source)["/root/Arm/Body/Body"]
        self.assertEqual(far.method, "nearest")
        np.testing.assert_array_equal(far.index, [-1, 1, 2])
        self.assertAlmostEqual(far.distance[0], 0.000012, places=12)
        # 大きさ 1 なら 0.006mm で、対応する
        near = match_meshes(distinct_model(), source)["/root/Arm/Body/Body"]
        np.testing.assert_array_equal(near.index, [0, 1, 2])

    def test_skeleton_check_in_meters(self):
        model = gap_model(np.eye(4))
        model = ExportedModel(joint_paths=model.joint_paths, bind=model.bind, rest_local=model.rest_local,
                              meshes=model.meshes, skeleton_world=scaled(2, 2, 2))
        result = check_skeleton(model)
        self.assertAlmostEqual(result.value, 2.0 * math.sqrt(0.1 ** 2 + 0.5 ** 2), places=12)

    def test_bad_units_are_rejected(self):
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(meters_per_unit=value), self.assertRaises(ValueError):
                make_model(meters_per_unit=value)
        with self.assertRaises(ValueError):
            make_model(skeleton_world=scaled(1, 1, 0))  # つぶれた行列では長さを測れない


# MARK: - 抜けと余り

# 重ならない3つの点（骨格の空間。geomBind は単位行列）
DISTINCT = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0]])


def distinct_model(points=DISTINCT):
    count = len(points)
    mesh = ExportedMesh(path="/root/Arm/Body/Body", name="Body", points=points,
                        triangles=np.zeros((0, 3), dtype=np.int64), joint_indices=np.zeros((count, 1), dtype=np.int64),
                        joint_weights=np.ones((count, 1)), source_name="Body")
    return make_model(meshes=[mesh])


class CoverageTest(unittest.TestCase):
    def test_all_present_is_green(self):
        model, source_rest, matches = rest_setup()
        result = check_coverage(model, {"Body": 3}, matches)
        self.assertEqual(result.signal, Signal.GREEN)
        self.assertEqual(result.value, 0.0)
        self.assertEqual(result.details["match_methods"], {"/root/Arm/Body/Body": "order"})

    def test_lost_vertex_is_red(self):
        model = distinct_model()
        source = {"Body": np.vstack([DISTINCT[:1], [[3.0, 3.0, 3.0]], DISTINCT[1:]])}
        matches = match_meshes(model, source)
        self.assertEqual(matches["/root/Arm/Body/Body"].method, "nearest")
        result = check_coverage(model, {"Body": 4}, matches)
        self.assertEqual(result.signal, Signal.RED)
        self.assertEqual(result.value, 1.0)
        np.testing.assert_array_equal(result.details["lost_vertices"]["Body"], [1])

    def test_masked_vertex_is_expected(self):
        model = make_model()
        source = {"Body": np.vstack([REST_SKEL[:1], [[3.0, 3.0, 3.0]], REST_SKEL[1:]])}
        hidden = {"Body": np.array([False, True, False, False])}
        matches = match_meshes(model, source, expected_missing=hidden)
        self.assertEqual(matches["/root/Arm/Body/Body"].method, "order")
        np.testing.assert_array_equal(matches["/root/Arm/Body/Body"].index, [0, 2, 3])
        result = check_coverage(model, {"Body": 4}, matches, expected_missing=hidden)
        self.assertEqual(result.signal, Signal.GREEN)
        self.assertEqual(result.details["expected_hidden_counts"], {"Body": 1})

    def test_hidden_but_exported_is_yellow(self):
        model, source_rest, _ = rest_setup()
        hidden = {"Body": np.array([False, False, True])}
        matches = match_meshes(model, source_rest)  # 消えるはずの点も、書き出されている
        result = check_coverage(model, {"Body": 3}, matches, expected_missing=hidden)
        self.assertEqual(result.signal, Signal.YELLOW)
        np.testing.assert_array_equal(result.details["hidden_exported"]["Body"], [2])

    def test_missing_and_extra_meshes_are_red(self):
        model, source_rest, matches = rest_setup()
        result = check_coverage(model, {"Body": 3, "Skirt": 10}, matches)
        self.assertEqual(result.signal, Signal.RED)
        self.assertEqual(result.details["missing_meshes"], ("Skirt",))
        extra = make_mesh(path="/root/Arm/Prop/Prop", name="Prop", source_name="Prop")
        model = make_model(meshes=[make_mesh(), extra])
        result = check_coverage(model, {"Body": 3}, match_meshes(model, source_rest))
        self.assertEqual(result.signal, Signal.RED)
        self.assertEqual(result.details["extra_meshes"], ("/root/Arm/Prop/Prop",))

    def test_unmeasured_mesh_is_gray(self):
        model = make_model()
        result = check_coverage(model, {"Body": 3}, {}, unmeasured={"Body": "頂点の数が変わります"})
        self.assertEqual(result.signal, Signal.GRAY)

    def test_unmatched_exported_point_is_gray(self):
        model = distinct_model(np.vstack([DISTINCT, [[9.0, 9.0, 9.0]]]))
        matches = match_meshes(model, {"Body": DISTINCT})
        result = check_coverage(model, {"Body": 3}, matches)
        self.assertEqual(result.signal, Signal.GRAY)
        self.assertEqual(result.details["unmatched_exported"], {"/root/Arm/Body/Body": 1})
        np.testing.assert_array_equal(matches["/root/Arm/Body/Body"].index, [0, 1, 2, -1])

    def test_nothing_to_check_is_gray(self):
        """元のメッシュも書き出した形も無いときは、「抜けが 0」の緑にしない（調べる物が無いので灰）"""
        result = check_coverage(make_model(meshes=[]), {}, {})
        self.assertEqual(result.signal, Signal.GRAY)
        self.assertIsNone(result.value)
        self.assertTrue(result.reason)
        # 片方だけ無いときは、抜け・余りとして赤のまま
        self.assertEqual(check_coverage(make_model(meshes=[]), {"Body": 3}, {}).signal, Signal.RED)


# MARK: - 骨格の休みの姿勢


def gap_model(rest_leg):
    """convert_blend.py の書き出しで見つかった形：骨の場所の途中（missing）が骨の一覧に無い。

    root の根元 (0, 0, 1)、抜けた骨 missing と leg の根元は (0.1, 0, 0.5)。Blender は leg の restTransforms を
    missing から見た形（単位行列）で書くが、USD は root を親とみなして組み立てる
    """
    bind = np.stack([translation(0, 0, 1), translation(0.1, 0, 0.5)])
    mesh = ExportedMesh(path="/root/A/Leg/Leg", name="Leg", points=np.array([[0.1, 0.0, 0.3], [0.0, 0.0, 1.2]]),
                        triangles=np.zeros((0, 3), dtype=np.int64), joint_indices=np.array([[1], [0]]),
                        joint_weights=np.array([[1.0], [1.0]]))
    return ExportedModel(joint_paths=("root", "root/missing/leg"), bind=bind,
                         rest_local=np.stack([translation(0, 0, 1), rest_leg]), meshes=(mesh,))


class SkeletonTest(unittest.TestCase):
    def test_consistent_rest_is_green(self):
        result = check_skeleton(make_model())
        self.assertEqual(result.signal, Signal.GREEN)
        self.assertEqual(result.value, 0.0)
        self.assertEqual(result.details["mismatched_joints"], ())
        self.assertEqual(result.details["path_gaps"], {})

    def test_rest_relative_to_missing_parent_is_red(self):
        result = check_skeleton(gap_model(np.eye(4)), {"root": "下半身", "leg": "左足D", "missing": "腰キャンセル左"})
        self.assertEqual(result.signal, Signal.RED)
        # アプリの休みの姿勢：leg = root の行列 = 平行移動 (0, 0, 1)。点 (0.1, 0, 0.3) は (0, 0, 0.8) へ動く
        self.assertAlmostEqual(result.value, math.sqrt(0.1 ** 2 + 0.5 ** 2), places=12)
        self.assertEqual(result.details["mismatched_joints"], ("leg",))
        self.assertEqual(result.details["mismatched_original"], ("左足D",))
        self.assertEqual(result.details["path_gaps"], {"leg": ("missing",)})
        self.assertEqual(result.details["path_gaps_original"], {"左足D": ("腰キャンセル左",)})
        self.assertEqual(result.details["meshes"]["/root/A/Leg/Leg"]["affected_count"], 1)

    def test_rest_relative_to_listed_parent_is_green(self):
        """抜けがあっても、restTransforms が USD の親（root）から見た形で書かれていれば合う"""
        result = check_skeleton(gap_model(translation(0.1, 0, -0.5)))
        self.assertEqual(result.signal, Signal.GREEN)
        self.assertLess(result.value, 1e-12)
        self.assertEqual(result.details["path_gaps"], {"leg": ("missing",)})

    def test_mismatch_without_weights_is_yellow(self):
        model = gap_model(np.eye(4))
        mesh = model.meshes[0]
        only_root = ExportedMesh(path=mesh.path, name=mesh.name, points=mesh.points, triangles=mesh.triangles,
                                 joint_indices=np.array([[0], [0]]), joint_weights=np.array([[1.0], [1.0]]))
        model = ExportedModel(joint_paths=model.joint_paths, bind=model.bind, rest_local=model.rest_local,
                              meshes=(only_root,))
        result = check_skeleton(model)
        self.assertEqual(result.signal, Signal.YELLOW)
        self.assertEqual(result.value, 0.0)

    def test_no_rest_transforms_is_gray(self):
        self.assertEqual(check_skeleton(make_model(rest_local=None)).signal, Signal.GRAY)


# MARK: - 名前


class NamesTest(unittest.TestCase):
    def model_with_shapes(self, *names):
        shapes = tuple(ExportedBlendShape(name=n, offsets=np.zeros((1, 3)), point_indices=np.array([0]))
                       for n in names)
        return make_model(meshes=[make_mesh(blend_shapes=shapes)])

    def by_name(self, results):
        return {m.name: m for m in results}

    def test_all_good(self):
        results = self.by_name(check_names(RIG, self.model_with_shapes("a"), ["センター", "左腕", "右腕"], ["あ"],
                                           source_deform_bones=["左腕"]))
        for name in ("joint_names", "humanoid_bones", "shape_key_names"):
            self.assertEqual(results[name].signal, Signal.GREEN, name)
        self.assertEqual(results["humanoid_bones"].details["original"], {"hips": "センター"})

    def test_humanoid_bone_missing_from_skeleton(self):
        rig = {**RIG, "bones": {"hips": "root", "leftUpperArm": "leftUpperArm"}}
        result = self.by_name(check_names(rig, self.model_with_shapes("a"), ["センター", "左腕"], ["あ"]))
        self.assertEqual(result["humanoid_bones"].signal, Signal.RED)
        self.assertEqual(result["humanoid_bones"].details["missing"], {"leftUpperArm": "leftUpperArm"})

    def test_unmapped_and_unknown_joints(self):
        rig = {**RIG, "originalNames": {"bones": {"root": "センター"}, "shapeKeys": {"a": "あ"}}}
        result = self.by_name(check_names(rig, self.model_with_shapes("a"), ["センター"], ["あ"]))
        self.assertEqual(result["joint_names"].signal, Signal.RED)
        self.assertEqual(result["joint_names"].details["unmapped_joints"], ("arm",))
        result = self.by_name(check_names(RIG, self.model_with_shapes("a"), ["センター"], ["あ"]))
        self.assertEqual(result["joint_names"].details["unknown_originals"], {"arm": "左腕"})

    def test_deform_bone_not_exported(self):
        result = self.by_name(check_names(RIG, self.model_with_shapes("a"), ["センター", "左腕", "左ひじ"], ["あ"],
                                          source_deform_bones=["左腕", "左ひじ"]))
        self.assertEqual(result["joint_names"].signal, Signal.RED)
        self.assertEqual(result["joint_names"].details["deform_bones_not_exported"], ("左ひじ",))

    def test_shape_keys(self):
        rig = {**RIG, "expressions": {"a": [{"shapeKey": "a", "weight": 1.0}], "o": [{"shapeKey": "o", "weight": 1}]}}
        result = self.by_name(check_names(rig, self.model_with_shapes("a", "k_00"), ["センター", "左腕"], ["あ"]))
        self.assertEqual(result["shape_key_names"].signal, Signal.RED)
        self.assertEqual(result["shape_key_names"].details["unmapped_shape_keys"], ("k_00",))
        self.assertEqual(result["shape_key_names"].details["expression_keys_not_exported"], {"o": ("o",)})
        self.assertEqual(result["shape_key_names"].value, 2.0)

    def test_no_original_names_is_gray(self):
        rig = {"bones": {"hips": "root"}}
        result = self.by_name(check_names(rig, make_model(), ["センター"], []))
        self.assertEqual(result["joint_names"].signal, Signal.GRAY)
        self.assertEqual(result["humanoid_bones"].signal, Signal.GREEN)
        self.assertEqual(result["shape_key_names"].signal, Signal.GREEN)  # キーも表情も無い

    def test_no_bones_is_gray(self):
        result = self.by_name(check_names({"originalNames": RIG["originalNames"]}, make_model(), ["センター", "左腕"],
                                          []))
        self.assertEqual(result["humanoid_bones"].signal, Signal.GRAY)

    def test_empty_bones_is_gray(self):
        """bones が空（アプリが使う骨が1つも無い）のときは、「見つからない骨が 0」の緑にしない"""
        result = self.by_name(check_names({**RIG, "bones": {}}, make_model(), ["センター", "左腕"], []))
        self.assertEqual(result["humanoid_bones"].signal, Signal.GRAY)
        self.assertIsNone(result["humanoid_bones"].value)
        self.assertTrue(result["humanoid_bones"].reason)

    def test_skeleton_without_joints_is_gray(self):
        model = ExportedModel(joint_paths=(), bind=np.zeros((0, 4, 4)), rest_local=None, meshes=())
        result = self.by_name(check_names(RIG, model, ["センター", "左腕"], []))
        self.assertEqual(result["joint_names"].signal, Signal.GRAY)
        self.assertIsNone(result["joint_names"].value)


# MARK: - 骨の数と重み


def weights_model(ids, weights):
    count = len(ids)
    mesh = ExportedMesh(path="/m", name="m", points=np.zeros((count, 3)), triangles=np.zeros((0, 3), dtype=np.int64),
                        joint_indices=np.array(ids), joint_weights=np.array(weights, dtype=np.float64))
    paths = ("j0", "j0/j1", "j0/j1/j2", "j0/j1/j2/j3", "j0/j1/j2/j3/j4")
    return ExportedModel(joint_paths=paths, bind=np.stack([np.eye(4)] * 5), rest_local=None, meshes=(mesh,))


class InfluencesTest(unittest.TestCase):
    def test_four_is_green(self):
        influences, sums = check_influences(weights_model([[0, 1, 2, 3]], [[0.4, 0.3, 0.2, 0.1]]))
        self.assertEqual((influences.signal, influences.value), (Signal.GREEN, 4.0))
        self.assertEqual(sums.signal, Signal.GREEN)

    def test_five_is_red(self):
        influences, _ = check_influences(weights_model([[0, 1, 2, 3, 4], [0, 0, 0, 0, 0]],
                                                       [[0.5, 0.3, 0.1, 0.08, 0.02], [1.0, 0, 0, 0, 0]]))
        self.assertEqual((influences.signal, influences.value), (Signal.RED, 5.0))
        np.testing.assert_array_equal(influences.details["meshes"]["/m"]["over_limit"], [0])

    def test_wide_slots_with_four_weights_is_yellow(self):
        influences, _ = check_influences(weights_model([[0, 1, 2, 3, 0]], [[0.4, 0.3, 0.2, 0.1, 0.0]]))
        self.assertEqual((influences.signal, influences.value), (Signal.YELLOW, 4.0))

    def test_sums(self):
        _, sums = check_influences(weights_model([[0, 1]], [[0.5, 0.4]]))
        self.assertEqual(sums.signal, Signal.YELLOW)
        self.assertAlmostEqual(sums.value, 0.1, places=12)
        _, sums = check_influences(weights_model([[0, 1], [0, 1]], [[1.0, 0.0], [0.0, 0.0]]))
        self.assertEqual(sums.signal, Signal.RED)
        np.testing.assert_array_equal(sums.details["meshes"]["/m"]["unweighted"], [1])
        _, sums = check_influences(weights_model([[0, 1]], [[1.5, -0.5]]))
        self.assertEqual(sums.signal, Signal.RED)

    def test_no_skinned_mesh_is_gray(self):
        mesh = ExportedMesh(path="/m", name="m", points=np.zeros((1, 3)), triangles=np.zeros((0, 3), dtype=np.int64),
                            skinned=False)
        model = ExportedModel(joint_paths=("a",), bind=np.eye(4)[None], rest_local=None, meshes=(mesh,))
        self.assertEqual([m.signal for m in check_influences(model)], [Signal.GRAY, Signal.GRAY])


# MARK: - 物理の部品


class PhysicsTest(unittest.TestCase):
    def test_clean_is_green(self):
        model = make_model(prims=(("/root", "Xform"), ("/root/Arm", "SkelRoot")), object_names={"/root/Arm": "Arm"})
        self.assertEqual(check_no_physics(model, ["剛体1", "ジョイント1"]).signal, Signal.GREEN)

    def test_physics_object_by_blender_name(self):
        model = make_model(prims=(("/root/Arm/_1", "Xform"),), object_names={"/root/Arm/_1": "剛体1"})
        result = check_no_physics(model, ["剛体1"])
        self.assertEqual(result.signal, Signal.RED)
        self.assertEqual(result.details["physics_named"], {"/root/Arm/_1": "剛体1"})

    def test_physics_object_by_prim_name(self):
        model = make_model(prims=(("/root/Arm/rigid_head", "Xform"),))
        self.assertEqual(check_no_physics(model, ["rigid_head"]).signal, Signal.RED)

    def test_physics_schema_and_type(self):
        model = make_model(physics_schemas=(("/root/Arm/x", "PhysicsRigidBodyAPI"),))
        self.assertEqual(check_no_physics(model, []).signal, Signal.RED)
        model = make_model(other_prims=(("/root/scene", "PhysicsScene"),))
        self.assertEqual(check_no_physics(model, []).signal, Signal.RED)

    def test_unknown_type_is_yellow(self):
        model = make_model(other_prims=(("/root/Cam", "Camera"),))
        result = check_no_physics(model, [])
        self.assertEqual(result.signal, Signal.YELLOW)
        self.assertEqual(result.details["unknown_types"], (("/root/Cam", "Camera"),))


# MARK: - ポーズ


class PoseTest(unittest.TestCase):
    def test_posed_joint_matrices_through_original_names(self):
        bone_names = ["左腕", "センター", "右腕"]
        matrices = np.stack([translation(1, 0, 0), translation(2, 0, 0), translation(3, 0, 0)])
        result = posed_joint_matrices(["root", "arm", "b099"], {"root": "センター", "arm": "左腕", "b099": "無い骨"},
                                      bone_names, matrices)
        np.testing.assert_array_equal(result[0], translation(2, 0, 0))
        np.testing.assert_array_equal(result[1], translation(1, 0, 0))
        self.assertTrue(np.isnan(result[2]).all())
        self.assertTrue(np.isnan(posed_joint_matrices(["root"], None, bone_names, matrices)).all())

    def test_local_rotation_identity_rest(self):
        q = local_rotation(np.eye(4), (0, 0, 1), 90.0)
        np.testing.assert_allclose(q, [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])

    @staticmethod
    def quaternion_matrix(q):
        w, x, y, z = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    def posed_direction(self, rest, axis, degrees):
        """骨の向き（ローカルの Y）が、ポーズでどこを向くか：rest @ basis @ (0, 1, 0)"""
        basis = self.quaternion_matrix(local_rotation(rest, axis, degrees))
        return rest[:3, :3] @ basis @ np.array([0.0, 1.0, 0.0])

    def test_arms_up_raises_left_arm(self):
        down = math.radians(37.5)
        direction = np.array([math.cos(down), 0.0, -math.sin(down)])  # A ポーズの左腕（+X へ、下がる）
        rest = np.eye(4)
        rest[:3, :3] = rotation((0, 0, 1), -90.0)[:3, :3] @ rotation((1, 0, 0), -37.5)[:3, :3]
        np.testing.assert_allclose(rest[:3, :3] @ [0.0, 1.0, 0.0], direction, atol=1e-12)
        axis, degrees = roundtrip.TEST_POSES["arms_up"]["左腕"]
        expected = rotation(axis, degrees)[:3, :3] @ direction
        result = self.posed_direction(rest, axis, degrees)
        np.testing.assert_allclose(result, expected, atol=1e-12)
        self.assertGreater(result[2], direction[2] + 0.5)  # 上がった

    def test_knee_bends_backward_not_reversed(self):
        """ひざは、足先が後ろ（+Y）へ行く向きにだけ曲げる。もも は前（-Y）へ出す"""
        rest = np.eye(4)
        rest[:3, :3] = rotation((1, 0, 0), -90.0)[:3, :3]  # 下（-Z）を向く骨
        np.testing.assert_allclose(rest[:3, :3] @ [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], atol=1e-12)
        knee = self.posed_direction(rest, *roundtrip.TEST_POSES["knees"]["左ひざ"])
        self.assertGreater(knee[1], 0.8)
        thigh = self.posed_direction(rest, *roundtrip.TEST_POSES["knees"]["左足"])
        self.assertLess(thigh[1], -0.7)

    def test_resolve_pose(self):
        names = ["左ひじ", "上半身", "首"]
        rotations, missing = resolve_pose(roundtrip.TEST_POSES["elbows"], names, np.stack([np.eye(4)] * 3))
        self.assertEqual(missing, ("右ひじ",))
        np.testing.assert_allclose(rotations["左ひじ"], [math.cos(math.pi / 4), 0.0, 0.0, -math.sin(math.pi / 4)])

    def test_local_rotation_errors(self):
        with self.assertRaises(ValueError):
            local_rotation(np.eye(4), (0, 0, 0), 10.0)
        with self.assertRaises(ValueError):
            local_rotation(np.eye(2), (0, 0, 1), 10.0)


# MARK: - 形の確かめ・rig の読み込み


class ValidationTest(unittest.TestCase):
    def test_arrays_are_read_only(self):
        model = make_model()
        with self.assertRaises(ValueError):
            model.bind[0, 0, 0] = 2.0
        with self.assertRaises(ValueError):
            model.meshes[0].points[0, 0] = 2.0
        self.assertEqual(model.joint_names, ("root", "arm"))
        np.testing.assert_array_equal(model.joint_parents, [-1, 0])

    def test_shape_errors(self):
        with self.assertRaises(ValueError):
            make_model(bind=BIND[:1])
        with self.assertRaises(ValueError):
            make_mesh(joint_weights=JOINT_WEIGHTS[:2])
        with self.assertRaises(ValueError):
            make_mesh(joint_indices=np.array([[2, 0], [0, 1], [0, 1]]))
        # 骨格の並びのまま（joint_names なし）の番号は、骨格の骨の数より小さくする
        mesh = ExportedMesh(path="/m", name="m", points=POINTS_LOCAL, triangles=TRIANGLES,
                            joint_indices=np.array([[5], [0], [0]]), joint_weights=np.ones((3, 1)))
        with self.assertRaises(ValueError):
            make_model(meshes=[mesh])
        with self.assertRaises(ValueError):
            ExportedBlendShape(name="a", offsets=np.zeros((2, 3)), point_indices=np.array([0]))

    def test_source_name_defaults_to_name(self):
        self.assertEqual(make_mesh(source_name="").source_name, "Body")


class RigTest(unittest.TestCase):
    def test_load_from_file_and_dict(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "character_rig.json"
            path.write_text(json.dumps(RIG, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(load_rig(path), RIG)
            self.assertEqual(load_rig(str(path)), RIG)
            path.write_text("[1, 2]", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_rig(path)
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_rig(path)
            with self.assertRaises(ValueError):
                load_rig(Path(folder) / "none.json")
        self.assertEqual(load_rig(RIG), RIG)

    def test_original_names(self):
        self.assertEqual(original_names(RIG, "bones"), {"root": "センター", "arm": "左腕"})
        self.assertIsNone(original_names({}, "bones"))
        with self.assertRaises(ValueError):
            original_names({"originalNames": {"bones": {"a": 1}}}, "bones")


class PointMatchTypeTest(unittest.TestCase):
    def test_matches_must_be_point_match(self):
        model, _, _ = rest_setup()
        with self.assertRaises(ValueError):
            compare_pose(model, {"Body": POSED_SKEL}, POSED, {"/root/Arm/Body/Body": object()}, pose="bend")
        self.assertIsInstance(match_points(GRID, GRID), PointMatch)


if __name__ == "__main__":
    unittest.main()

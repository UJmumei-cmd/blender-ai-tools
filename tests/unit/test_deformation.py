"""A4 形のくずれと版の比べ（measure/deformation.py）を確かめる"""

import math
import unittest
from unittest import mock

import numpy as np

from blender_ai_tools import thresholds
from blender_ai_tools.measure import deformation as d
from blender_ai_tools.measure.results import Signal
from tests import shapes


def random_rotation(seed: int) -> np.ndarray:
    """鏡うつしを含まない、でたらめな回転（種を決める）"""
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def rigid(points: np.ndarray, seed: int, offset=(0.3, -1.2, 2.5)) -> np.ndarray:
    return points @ random_rotation(seed).T + np.asarray(offset)


def signed_area_2d(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """xy 平面に置いた三角形の、符号つきの面積の2倍（反時計回りなら正）"""
    a, b, c = (points[triangles[:, k]] for k in range(3))
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])


def neighbors(triangles: np.ndarray, vertex: int) -> set[int]:
    result: set[int] = set()
    for tri in triangles:
        if vertex in tri:
            result.update(int(x) for x in tri)
    result.discard(vertex)
    return result


def fold_vertex(points: np.ndarray, triangles: np.ndarray, vertex: int) -> tuple[np.ndarray, int]:
    """vertex を使う最初の三角形の、向かいの辺（を通る直線）について vertex を映す。返す物は（動いた形、その三角形の番号）"""
    first = int(np.flatnonzero((triangles == vertex).any(axis=1))[0])
    a, b = [int(x) for x in triangles[first] if x != vertex]
    direction = (points[b] - points[a]) / np.linalg.norm(points[b] - points[a])
    foot = points[a] + direction * ((points[vertex] - points[a]) @ direction)
    posed = points.copy()
    posed[vertex] = 2.0 * foot - points[vertex]
    return posed, first


def mean_edge_length(points: np.ndarray, triangles: np.ndarray, vertex: int) -> float:
    return float(np.mean([np.linalg.norm(points[j] - points[vertex]) for j in sorted(neighbors(triangles, vertex))]))


def tipped_strip(sag: float, across: int = 3, rows: int = 8, width: float = 0.01,
                 step: float = 0.01) -> tuple[np.ndarray, np.ndarray, int]:
    """髪の房のような細い帯（横に across 点、+Y へ rows 列）と、その先のとがった点（先の点）。

    帯の横の列は、Z へ sag × width だけ弓なりに曲げる。先の点は、最後の列の点と三角形の扇でつながるので、
    先の点のまわりの点は「ほぼ1本の線」に並ぶ（sag が 0 なら、ちょうど1本の線）。返す物は（点、三角形、先の点の番号）
    """
    xs = np.linspace(-width / 2.0, width / 2.0, across)
    bend = sag * width * (1.0 - (2.0 * xs / width) ** 2)
    points = [[x, row * step, z] for row in range(rows) for x, z in zip(xs, bend)]
    points.append([0.0, rows * step, 0.0])
    triangles = []
    for row in range(rows - 1):
        for i in range(across - 1):
            a = row * across + i
            triangles += [[a, a + 1, a + 1 + across], [a, a + 1 + across, a + across]]
    last = (rows - 1) * across
    tip = len(points) - 1
    triangles += [[last + i, last + i + 1, tip] for i in range(across - 1)]
    return np.asarray(points, dtype=np.float64), np.asarray(triangles, dtype=np.int64), tip


# MARK: - 版の比べ

class DisplacementTest(unittest.TestCase):
    def test_distances(self):
        before = np.zeros((3, 3))
        after = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [0.0, -2.0, 0.0]])
        np.testing.assert_array_equal(d.displacement(before, after), [5.0, 0.0, 2.0])

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            d.displacement(np.zeros((3, 3)), np.zeros((4, 3)))
        with self.assertRaises(ValueError):
            d.displacement(np.zeros((3, 2)), np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            d.displacement(np.zeros((1, 3)), [[np.nan, 0.0, 0.0]])
        with self.assertRaises(ValueError):
            d.displacement(np.zeros((1, 3), dtype=bool), np.zeros((1, 3), dtype=bool))

    def test_empty(self):
        self.assertEqual(d.displacement([], []).shape, (0,))


class ChangedVerticesTest(unittest.TestCase):
    def setUp(self):
        self.tol = thresholds.REPRO_TOLERANCE_LENGTH
        self.before = np.zeros((5, 3))
        self.after = np.zeros((5, 3))
        self.after[0, 0] = self.tol  # ちょうど幅：変化ではない
        self.after[1, 0] = np.nextafter(self.tol, 1.0)  # 幅をわずかに超える：変化
        self.after[2, 1] = -2.0 * self.tol  # 変化（ただし allowed で除く場合あり）
        self.after[3, 2] = 0.5 * self.tol  # 幅より小さい：変化ではない

    def test_tolerance_boundary(self):
        np.testing.assert_array_equal(d.changed_vertices(self.before, self.after), [1, 2])

    def test_allowed_mask_excludes(self):
        allowed = np.array([False, False, True, False, False])
        result = d.changed_vertices(self.before, self.after, allowed=allowed)
        np.testing.assert_array_equal(result, [1])
        self.assertEqual(result.dtype, np.int64)

    def test_allowed_where_nothing_moved_changes_nothing(self):
        allowed = np.array([False, False, False, False, True])
        np.testing.assert_array_equal(d.changed_vertices(self.before, self.after, allowed=allowed), [1, 2])

    def test_zero_tolerance(self):
        np.testing.assert_array_equal(d.changed_vertices(self.before, self.after, tolerance=0.0), [0, 1, 2, 3])
        np.testing.assert_array_equal(d.changed_vertices(self.before, self.before, tolerance=0.0), [])

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            d.changed_vertices(self.before, self.after, allowed=np.array([0, 0, 1, 0, 0]))
        with self.assertRaises(ValueError):
            d.changed_vertices(self.before, self.after, allowed=np.zeros(4, dtype=bool))
        with self.assertRaises(ValueError):
            d.changed_vertices(self.before, self.after, tolerance=-1.0)
        with self.assertRaises(ValueError):
            d.changed_vertices(self.before, self.after, tolerance=True)
        with self.assertRaises(ValueError):
            d.changed_vertices(self.before, self.after, tolerance=math.nan)


class SummarizeChangesTest(unittest.TestCase):
    def test_red_outside_allowed(self):
        before, _ = shapes.box()
        after = before.copy()
        after[3] += [0.0, 0.0, 0.002]
        after[5] += [0.001, 0.0, 0.0]
        allowed = np.zeros(8, dtype=bool)
        allowed[5] = True
        result = d.summarize_changes(before, after, allowed=allowed)
        self.assertEqual(result.name, "unexpected_changes")
        self.assertEqual(result.unit, "count")
        self.assertEqual(result.signal, Signal.RED)
        self.assertEqual(result.value, 1.0)
        np.testing.assert_array_equal(result.details["changed_indices"], [3])
        self.assertEqual(result.details["allowed_changed_count"], 1)
        self.assertAlmostEqual(result.details["max_unexpected_displacement"], 0.002, places=15)
        self.assertAlmostEqual(result.details["max_displacement"], 0.002, places=15)

    def test_green_when_changes_are_allowed_or_tiny(self):
        before, _ = shapes.box()
        after = before.copy()
        after[0] += [0.5 * thresholds.REPRO_TOLERANCE_LENGTH, 0.0, 0.0]
        after[1] += [0.1, 0.0, 0.0]
        allowed = np.zeros(8, dtype=bool)
        allowed[1] = True
        result = d.summarize_changes(before, after, allowed=allowed)
        self.assertEqual(result.signal, Signal.GREEN)
        self.assertEqual(result.value, 0.0)
        self.assertIsNone(result.details["max_unexpected_displacement"])

    def test_tolerance_boundary_matches_changed_vertices(self):
        """ちょうど幅だけ動いた頂点は変化ではない（緑）。わずかに超えると赤。数え方は changed_vertices と同じ"""
        tol = thresholds.REPRO_TOLERANCE_LENGTH
        before = np.zeros((3, 3))
        after = np.zeros((3, 3))
        after[0, 0] = tol
        self.assertEqual(d.summarize_changes(before, after).signal, Signal.GREEN)
        after[2, 1] = np.nextafter(tol, 1.0)
        result = d.summarize_changes(before, after)
        self.assertEqual((result.signal, result.value), (Signal.RED, 1.0))
        np.testing.assert_array_equal(result.details["changed_indices"], d.changed_vertices(before, after))
        np.testing.assert_array_equal(result.details["changed_indices"], [2])

    def test_gray_when_vertex_counts_differ(self):
        result = d.summarize_changes(np.zeros((3, 3)), np.zeros((4, 3)))
        self.assertEqual(result.signal, Signal.GRAY)
        self.assertTrue(result.reason)
        self.assertEqual(result.details["after_count"], 4)

    def test_gray_when_empty(self):
        self.assertEqual(d.summarize_changes([], []).signal, Signal.GRAY)

    def test_bad_points_raise(self):
        with self.assertRaises(ValueError):
            d.summarize_changes(np.zeros((3, 3)), np.full((3, 3), np.nan))


# MARK: - 辺

class UniqueEdgesTest(unittest.TestCase):
    def test_box_has_18_edges(self):
        _, t = shapes.box()
        edges = d.unique_edges(t)
        self.assertEqual(edges.shape, (18, 2))
        self.assertEqual(edges.dtype, np.int64)
        self.assertTrue((edges[:, 0] < edges[:, 1]).all())
        self.assertEqual(len({tuple(e) for e in edges.tolist()}), 18)
        self.assertEqual(edges.tolist(), sorted(edges.tolist()))

    def test_order_does_not_depend_on_input_order(self):
        _, t = shapes.box()
        shuffled = np.roll(t[np.random.default_rng(4).permutation(len(t))], 1, axis=1)
        np.testing.assert_array_equal(d.unique_edges(shuffled), d.unique_edges(t))

    def test_closed_mesh_counts(self):
        v, t = shapes.icosphere(subdivisions=1)
        edges = d.unique_edges(t)
        self.assertEqual(len(edges), 3 * len(t) // 2)
        self.assertEqual(len(v) - len(edges) + len(t), 2)  # オイラーの式

    def test_grid_counts(self):
        nx, ny = 3, 5
        _, t = shapes.grid(nx, ny)
        self.assertEqual(len(d.unique_edges(t)), nx * (ny + 1) + ny * (nx + 1) + nx * ny)

    def test_degenerate_triangle_has_no_self_edge(self):
        np.testing.assert_array_equal(d.unique_edges([[2, 2, 0]]), [[0, 2]])

    def test_empty_and_bad_inputs(self):
        self.assertEqual(d.unique_edges(np.zeros((0, 3), dtype=np.int64)).shape, (0, 2))
        self.assertEqual(d.unique_edges([]).shape, (0, 2))
        for bad in ([[0.0, 1.0, 2.0]], [[True, False, True]], [[0, -1, 2]], [[0, 1, 2, 3]], [0, 1, 2]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                d.unique_edges(np.asarray(bad))


class EdgeLengthRatiosTest(unittest.TestCase):
    def test_uniform_scale(self):
        v, t = shapes.icosphere(subdivisions=1)
        edges = d.unique_edges(t)
        scaled = (v - [0.2, 0.1, -0.3]) * 1.7 + [0.2, 0.1, -0.3]
        np.testing.assert_allclose(d.edge_length_ratios(v, scaled, edges), 1.7, rtol=1e-12)
        np.testing.assert_allclose(d.edge_length_ratios(v, rigid(v, 1), edges), 1.0, rtol=1e-12)

    def test_single_moved_vertex(self):
        v, t = shapes.grid(4, 4, size=(1.0, 1.0))
        edges = d.unique_edges(t)
        moved = 12
        h = 0.1
        posed = v.copy()
        posed[moved, 2] += h
        ratios = d.edge_length_ratios(v, posed, edges)
        for (a, b), ratio in zip(edges.tolist(), ratios):
            rest_length = np.linalg.norm(v[a] - v[b])
            if moved in (a, b):
                self.assertAlmostEqual(ratio, math.hypot(rest_length, h) / rest_length, places=12)
                self.assertGreater(ratio, 1.0)
            else:
                self.assertEqual(ratio, 1.0)

    def test_zero_rest_length_is_nan(self):
        rest = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        posed = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]])
        ratios = d.edge_length_ratios(rest, posed, [[0, 1], [0, 2]])
        self.assertTrue(math.isnan(ratios[0]))
        self.assertEqual(ratios[1], 2.0)

    def test_min_rest_length_boundary(self):
        tol = thresholds.REPRO_TOLERANCE_LENGTH
        rest = np.array([[0.0, 0.0, 0.0], [tol, 0.0, 0.0], [np.nextafter(tol, 1.0), 0.0, 0.0]])
        ratios = d.edge_length_ratios(rest, rest, [[0, 1], [0, 2]])
        self.assertTrue(math.isnan(ratios[0]))  # ちょうど幅は「ほぼ 0」
        self.assertEqual(ratios[1], 1.0)

    def test_posed_collapse_is_zero(self):
        rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        self.assertEqual(d.edge_length_ratios(rest, np.zeros((2, 3)), [[0, 1]])[0], 0.0)

    def test_bad_inputs(self):
        v, _ = shapes.box()
        with self.assertRaises(ValueError):
            d.edge_length_ratios(v, v, [[0, 8]])
        with self.assertRaises(ValueError):
            d.edge_length_ratios(v, v[:7], [[0, 1]])
        with self.assertRaises(ValueError):
            d.edge_length_ratios(v, v, [[0, 1, 2]])


# MARK: - 裏返り

class TriangleFlipsTest(unittest.TestCase):
    def folded_grid(self):
        """格子の真ん中の頂点を、1つの三角形の向かいの辺の反対側へ映す（その三角形が裏返る）"""
        v, t = shapes.grid(4, 4)
        posed, first = fold_vertex(v, t, 12)
        return v, posed, t, first

    def test_folded_face_is_found(self):
        v, posed, t, first = self.folded_grid()
        flipped, degenerate = d.triangle_flips(v, posed, t)
        expected = (signed_area_2d(v, t) > 0) & (signed_area_2d(posed, t) < 0)
        self.assertTrue(expected[first])
        np.testing.assert_array_equal(flipped, expected)
        self.assertFalse(degenerate.any())
        self.assertEqual(flipped.dtype, bool)

    def test_folded_face_is_found_after_rigid_motion(self):
        v, posed, t, _ = self.folded_grid()
        expected = (signed_area_2d(v, t) > 0) & (signed_area_2d(posed, t) < 0)
        for seed in range(3):
            flipped, _ = d.triangle_flips(v, rigid(posed, seed), t)
            np.testing.assert_array_equal(flipped, expected)

    def test_rigid_motion_has_no_flips(self):
        """腕が丸ごと回っただけなら、裏返りにしない（回転を除かない比べ方とは違う）"""
        v, t = shapes.icosphere(subdivisions=2)
        half_turn = np.diag([1.0, -1.0, -1.0])  # x 軸まわりに 180 度
        posed = v @ half_turn.T + [0.0, 0.5, 1.0]
        flipped, degenerate = d.triangle_flips(v, posed, t)
        self.assertFalse(flipped.any())
        self.assertFalse(degenerate.any())
        global_flipped, _ = d.triangle_flips(v, posed, t, local_rotation=False)
        normals = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
        expected = np.einsum("ij,ij->i", normals, normals @ half_turn.T) < 0
        self.assertTrue(expected.any())
        np.testing.assert_array_equal(global_flipped, expected)

    def test_folded_vertex_on_closed_mesh(self):
        """球の頂点を、三角形の向かいの辺の反対側へ映す。その三角形の法線はちょうど逆になる"""
        v, t = shapes.icosphere(subdivisions=2)
        for vertex in (0, 20, 100):
            posed, first = fold_vertex(v, t, vertex)
            incident = (t == vertex).any(axis=1)
            global_flipped, _ = d.triangle_flips(v, posed, t, local_rotation=False)
            self.assertTrue(global_flipped[first])
            self.assertFalse((global_flipped & ~incident).any())
            for moved in (posed, rigid(posed, 7)):
                flipped, degenerate = d.triangle_flips(v, moved, t)
                np.testing.assert_array_equal(flipped, global_flipped)
                self.assertFalse(degenerate.any())

    def test_dent_is_not_a_flip(self):
        """頂点を押し込んでくぼませても、面は外を向いたまま（裏返りではない）"""
        v, t = shapes.icosphere(subdivisions=2)
        posed = v.copy()
        posed[0] *= 0.8
        for local in (True, False):
            self.assertFalse(d.triangle_flips(v, posed, t, local_rotation=local)[0].any())

    def test_lone_mirrored_triangle(self):
        """1つだけの三角形を平面の中で鏡うつしにした物は、180 度回した物と同じ点になるので、まわりで比べると裏返りではない"""
        rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        posed = rest * [-1.0, 1.0, 1.0]
        tri = [[0, 1, 2]]
        self.assertFalse(d.triangle_flips(rest, posed, tri)[0][0])
        self.assertTrue(d.triangle_flips(rest, posed, tri, local_rotation=False)[0][0])

    def test_degenerate_triangles_are_reported_not_flipped(self):
        rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
        posed = rest.copy()
        posed[2] = [0.5, 0.0, 0.0]  # 三角形 0 は、動いた形でつぶれる
        triangles = [[0, 1, 2], [0, 0, 1], [0, 1, 3]]  # 1：同じ頂点を2回、2：一直線
        for local in (True, False):
            flipped, degenerate = d.triangle_flips(rest, posed, triangles, local_rotation=local)
            np.testing.assert_array_equal(degenerate, [True, True, True])
            np.testing.assert_array_equal(flipped, [False, False, False])

    def test_degenerate_area_boundary(self):
        rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # 面積 0.5
        _, degenerate = d.triangle_flips(rest, rest, [[0, 1, 2]], degenerate_area=0.5)
        self.assertTrue(degenerate[0])
        _, degenerate = d.triangle_flips(rest, rest, [[0, 1, 2]], degenerate_area=np.nextafter(0.5, 0.0))
        self.assertFalse(degenerate[0])

    def test_empty_and_bad_inputs(self):
        flipped, degenerate = d.triangle_flips(np.zeros((0, 3)), np.zeros((0, 3)), [])
        self.assertEqual((flipped.shape, degenerate.shape), ((0,), (0,)))
        v, t = shapes.box()
        with self.assertRaises(ValueError):
            d.triangle_flips(v, v, t, local_rotation=1)
        with self.assertRaises(ValueError):
            d.triangle_flips(v, v[:7], t)
        with self.assertRaises(ValueError):
            d.triangle_flips(v, v, t + 1)


# MARK: - 体積

class VolumeTest(unittest.TestCase):
    def test_box(self):
        v, t = shapes.box(size=(2.0, 3.0, 4.0), center=(1.0, -2.0, 5.0))
        self.assertAlmostEqual(d.mesh_volume(v, t), 24.0, places=12)
        self.assertAlmostEqual(d.mesh_volume(v, t[:, ::-1]), -24.0, places=12)

    def test_regular_icosahedron(self):
        v, t = shapes.icosphere(radius=1.0, subdivisions=0)
        edge = 4.0 / math.sqrt(10.0 + 2.0 * math.sqrt(5.0))  # 外接球の半径 1 の正二十面体の辺
        expected = 5.0 * (3.0 + math.sqrt(5.0)) / 12.0 * edge ** 3
        self.assertAlmostEqual(d.mesh_volume(v, t), expected, places=12)

    def test_icosphere(self):
        v, t = shapes.icosphere(radius=0.3, subdivisions=3, center=(0.1, 0.2, 0.3))
        volume = d.mesh_volume(v, t)
        self.assertAlmostEqual(volume, shapes.signed_volume(v, t), places=14)
        sphere = 4.0 / 3.0 * math.pi * 0.3 ** 3
        self.assertLess(volume, sphere)
        self.assertGreater(volume, 0.98 * sphere)

    def test_small_box_far_from_origin(self):
        """原点から遠い小さな形でも、けた落ちしない"""
        v, t = shapes.box(size=(1e-3, 1e-3, 1e-3), center=(1000.0, -2000.0, 500.0))
        self.assertAlmostEqual(d.mesh_volume(v, t) / 1e-9, 1.0, places=6)

    def test_empty(self):
        self.assertEqual(d.mesh_volume(np.zeros((3, 3)), np.zeros((0, 3), dtype=np.int64)), 0.0)

    def test_change_ratio_for_scaling(self):
        v, t = shapes.icosphere(subdivisions=2)
        for scale in (0.5, 0.9, 1.0, 1.25):
            with self.subTest(scale=scale):
                posed = (v - [1.0, 2.0, 3.0]) * scale + [4.0, -1.0, 0.5]
                self.assertAlmostEqual(d.volume_change_ratio(v, posed, t), scale ** 3 - 1.0, places=12)

    def test_change_ratio_rigid_is_zero(self):
        v, t = shapes.box(size=(1.0, 2.0, 3.0))
        self.assertAlmostEqual(d.volume_change_ratio(v, rigid(v, 2), t), 0.0, places=12)

    def test_change_ratio_with_inside_out_rest(self):
        v, t = shapes.box()
        self.assertAlmostEqual(d.volume_change_ratio(v, v * 0.5, t[:, ::-1]), -0.875, places=12)

    def test_posed_inside_out(self):
        v, t = shapes.box()
        self.assertAlmostEqual(d.volume_change_ratio(v, v * [-1.0, 1.0, 1.0], t), -2.0, places=12)

    def test_flat_rest_gives_none(self):
        v, t = shapes.grid(2, 2)
        pillow_t = np.concatenate([t, t[:, ::-1]])  # 表と裏が重なった、厚さ 0 の閉じた形
        self.assertIsNone(d.volume_change_ratio(v, v * 2.0, pillow_t))
        self.assertIsNone(d.volume_change_ratio(np.zeros((0, 3)), np.zeros((0, 3)), []))

    def test_open_or_misoriented_mesh_gives_none(self):
        """穴がある形・面の向きがそろっていない形の「体積」は、基準の点で値が変わる（体積ではない）ので測らない"""
        v, t = shapes.icosphere(subdivisions=2)
        missing_face = t[1:]
        reversed_face = np.concatenate([t[:1, ::-1], t[1:]])
        for triangles in (missing_face, reversed_face):
            self.assertIsNone(d.volume_change_ratio(v, v * 0.5, triangles))
        # 開いたお椀：底を深くしても浅くしても、中身の体積というものは無い
        bowl, grid_t = shapes.grid(8, 8)
        bowl[:, 2] = 0.5 * (bowl[:, 0] ** 2 + bowl[:, 1] ** 2)
        deeper = bowl.copy()
        deeper[:, 2] *= 1.6
        self.assertIsNone(d.volume_change_ratio(bowl, deeper, grid_t))
        self.assertEqual(d.summarize_volume(d.volume_change_ratio(bowl, deeper, grid_t)).signal, Signal.GRAY)

    def test_closed_shapes_sharing_an_edge_are_measured(self):
        """2つの箱が1本の辺だけを共有する（その辺を4つの面が使う）。行きと帰りの辺の数がそろうので、閉じた形として測る"""
        first, t1 = shapes.box()
        second, t2 = shapes.box(center=(1.0, 1.0, 0.0))
        # 2つ目の箱の点 0・4 は、1つ目の箱の点 2・6 と同じ位置（共有する辺）。残りの6点を後ろに足す
        rest_ids = [1, 2, 3, 5, 6, 7]
        mapping = np.empty(8, dtype=np.int64)
        mapping[[0, 4]] = [2, 6]
        mapping[rest_ids] = 8 + np.arange(6)
        v = np.vstack([first, second[rest_ids]])
        t = np.concatenate([t1, mapping[t2]])
        self.assertAlmostEqual(d.mesh_volume(v, t), 2.0, places=12)
        self.assertAlmostEqual(d.volume_change_ratio(v, v * 1.1, t), 1.1 ** 3 - 1.0, places=12)


# MARK: - とげ

class SpikeScoresTest(unittest.TestCase):
    def test_rigid_motion_is_zero(self):
        for v, t in (shapes.icosphere(subdivisions=2), shapes.box(), shapes.grid(3, 3)):
            for seed in range(3):
                scores = d.spike_scores(v, rigid(v, seed), t)
                self.assertFalse(np.isnan(scores).any())
                self.assertLess(float(scores.max()), 1e-9)

    def test_pulled_vertex(self):
        v, t = shapes.icosphere(subdivisions=2)
        vertex = 5
        pull = 0.3
        posed = v.copy()
        posed[vertex] *= 1.0 + pull  # 半径 1 なので、外へ pull だけ動く
        scores = d.spike_scores(v, posed, t)
        expected = pull / mean_edge_length(v, t, vertex)
        self.assertAlmostEqual(scores[vertex], expected, places=10)
        self.assertGreater(scores[vertex], thresholds.SPIKE_RED)
        ring = neighbors(t, vertex)
        for other in range(len(v)):
            if other == vertex:
                continue
            if other in ring:
                self.assertGreater(scores[other], 0.0)
                self.assertLess(scores[other], scores[vertex])
            else:
                self.assertLess(scores[other], 1e-12)
        # 形全体が回っていても同じ
        np.testing.assert_allclose(d.spike_scores(v, rigid(posed, 5), t), scores, atol=1e-9)

    def test_uniform_scale_matches_formula(self):
        """2倍にすると、まわりの回転は単位行列で、L が2倍になるので、強さ = |L| / 辺の長さの平均"""
        v, t = shapes.icosphere(subdivisions=1)
        scores = d.spike_scores(v, v * 2.0, t)
        for vertex in range(len(v)):
            ring = sorted(neighbors(t, vertex))
            laplacian = v[ring].mean(axis=0) - v[vertex]
            self.assertAlmostEqual(scores[vertex], np.linalg.norm(laplacian) / mean_edge_length(v, t, vertex),
                                   places=10)

    def test_corner_with_two_neighbors(self):
        """三角形1つだけに使われる角の頂点（まわりの点が2つで、1本の線になる）も測れる"""
        v, t = shapes.grid(4, 4)
        self.assertEqual(len(neighbors(t, 4)), 2)
        scores = d.spike_scores(v, rigid(v, 9), t)
        self.assertFalse(math.isnan(scores[4]))
        self.assertLess(scores[4], 1e-9)

    def test_nearly_collinear_ring_is_stable(self):
        """先の点のまわりが「ほぼ1本の線」の帯。横の列の弓なりが少しだけ逆になっても、とげ・裏返りにしない。

        まわりの点だけで回転を求めると、線のまわりの回転は弓なりの小さな曲がりだけで決まり、
        曲がりが逆になると 180 度回った答えになる（0.02mm 動いただけで、とげが赤になっていた）。
        """
        width = 0.01
        for sag in (1e-3, 0.05, 0.2):
            with self.subTest(sag=sag):
                v, t, tip = tipped_strip(sag, width=width)
                posed = v.copy()
                posed[:tip, 2] *= -1.0  # 横の列の弓なりを逆にする。先の点は動かさない
                scores = d.spike_scores(v, posed, t)
                self.assertFalse(np.isnan(scores).any())
                # 本当の変化：先の点のまわりの点の平均が、Z へ 2 × sag × width / 3 だけ動いた
                shift = 2.0 * sag * width / 3.0 / mean_edge_length(v, t, tip)
                self.assertLess(scores[tip], 1.5 * shift + 1e-6)
                self.assertLess(float(scores.max()), thresholds.SPIKE_YELLOW)
                self.assertFalse(d.triangle_flips(v, posed, t)[0].any())
                # 形全体が回っていても同じ
                np.testing.assert_allclose(d.spike_scores(v, rigid(posed, 3), t), scores, atol=1e-9)

    def test_thin_strip_interior_keeps_ring_rotation(self):
        """細長い格子（10:1）の頂点：まわりの点は細く並ぶが、頂点を入れても広がらないので、まわりの点だけの回転のまま。
        その頂点だけを、辺の長さの平均だけ面の外へ引くと、強さはちょうど 1（回転に吸われない）。
        6・11 は短い辺の縁の頂点（頂点を入れると、広がりがかえって悪くなる）、8・14・21 は中の頂点"""
        v, t = shapes.grid(5, 5, size=(1.0, 0.1))
        for vertex in (6, 11, 8, 14, 21):
            with self.subTest(vertex=vertex):
                posed = v.copy()
                posed[vertex, 2] += mean_edge_length(v, t, vertex)
                self.assertAlmostEqual(d.spike_scores(v, posed, t)[vertex], 1.0, places=9)

    def test_thin_rings_follow_their_own_part(self):
        """帯を途中で 90 度折る。折り目のまわりの点を使わない頂点（細い帯の先・角を含む）は、
        自分の側といっしょに回っただけなので、とげは 0。裏返りも無い"""
        v, t, tip = tipped_strip(0.05)
        hinge = 0.035  # 列 3（y = 0.03）と列 4（y = 0.04）の間
        turn = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # x 軸のまわりに 90 度
        far = v[:, 1] > hinge
        posed = v.copy()
        posed[far] = (v[far] - [0.0, hinge, 0.0]) @ turn.T + [0.0, hinge, 0.0]
        scores = d.spike_scores(v, posed, t)
        rows = np.round(v[:, 1] / 0.01).astype(int)
        away = (rows <= 2) | (rows >= 5)
        self.assertTrue(away[tip] and away[2])  # 先の点と、三角形1つだけの角
        self.assertLess(float(scores[away].max()), 1e-9)
        self.assertGreater(float(scores[~away].min()), 0.01)
        self.assertFalse(d.triangle_flips(v, posed, t)[0].any())

    def test_nearly_collinear_ring_still_finds_damage(self):
        """先の点のまわりが「ほぼ1本の線」でも、先の点を引き伸ばしたとげと、列の向こうへ折り返した裏返りは見つける"""
        v, t, tip = tipped_strip(1e-3)
        pulled = v.copy()
        pulled[tip, 1] += 0.02  # 帯の向きに、辺の長さの約2倍
        scores = d.spike_scores(v, pulled, t)
        self.assertEqual(int(np.nanargmax(scores)), tip)
        self.assertGreater(scores[tip], thresholds.SPIKE_RED)
        folded = v.copy()
        folded[tip, 1] -= 0.015  # 先の点を、最後の列の向こう側（帯の内側）へ折り返す
        flipped, _ = d.triangle_flips(v, folded, t)
        fan = (t == tip).any(axis=1)
        np.testing.assert_array_equal(flipped, fan)

    def test_unmeasurable_vertices_are_nan(self):
        v, t = shapes.box()
        extra = np.vstack([v, [[5.0, 5.0, 5.0]]])  # どの面にも使われていない頂点
        scores = d.spike_scores(extra, extra, t)
        self.assertTrue(math.isnan(scores[8]))
        self.assertFalse(np.isnan(scores[:8]).any())
        line = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])  # 一直線（回転が決まらない）
        self.assertTrue(np.isnan(d.spike_scores(line, line, [[0, 1, 2]])).all())

    def test_bad_inputs(self):
        v, t = shapes.box()
        with self.assertRaises(ValueError):
            d.spike_scores(v, v[:7], t)
        with self.assertRaises(ValueError):
            d.spike_scores(v, v, t.astype(np.float64))


class ChunkTest(unittest.TestCase):
    def test_small_chunks_give_the_same_answer(self):
        """小分けの大きさを変えても、答えは同じ（小分けのつなぎ目のまちがいを見つける）"""
        v, t = shapes.icosphere(subdivisions=2)
        posed, _ = fold_vertex(v, t, 20)
        posed[3] *= 1.4
        posed = rigid(posed, 4)
        flips = d.triangle_flips(v, posed, t)
        scores = d.spike_scores(v, posed, t)
        with mock.patch.object(d, "CHUNK_SIZE", 7), mock.patch.object(d, "EDGE_CHUNK_SIZE", 11):
            small_flips = d.triangle_flips(v, posed, t)
            small_scores = d.spike_scores(v, posed, t)
        np.testing.assert_array_equal(small_flips[0], flips[0])
        np.testing.assert_array_equal(small_flips[1], flips[1])
        self.assertTrue(flips[0].any())
        np.testing.assert_allclose(small_scores, scores, rtol=1e-9, atol=1e-12)

    def test_small_chunks_with_thin_rings(self):
        """まわりの点が細い頂点（帯の先・帯の縁・三角形1つだけの角）を求め直す所も、小分けで答えが変わらない"""
        v, t, tip = tipped_strip(0.05)
        posed = v.copy()
        posed[:tip, 2] *= -1.0
        posed[tip, 1] -= 0.015  # 先を折り返す（裏返りあり）
        posed = rigid(posed, 6)
        flips = d.triangle_flips(v, posed, t)
        scores = d.spike_scores(v, posed, t)
        with mock.patch.object(d, "CHUNK_SIZE", 2), mock.patch.object(d, "EDGE_CHUNK_SIZE", 3):
            small_flips = d.triangle_flips(v, posed, t)
            small_scores = d.spike_scores(v, posed, t)
        np.testing.assert_array_equal(small_flips[0], flips[0])
        np.testing.assert_array_equal(small_flips[1], flips[1])
        self.assertTrue(flips[0].any())
        np.testing.assert_allclose(small_scores, scores, rtol=1e-9, atol=1e-12)


# MARK: - 信号

class SummarizeStretchTest(unittest.TestCase):
    def test_signals(self):
        cases = [
            ([1.0, 1.29], Signal.GREEN, 1.29),
            ([1.0, thresholds.EDGE_STRETCH_YELLOW], Signal.YELLOW, thresholds.EDGE_STRETCH_YELLOW),
            ([1.0, thresholds.EDGE_STRETCH_RED], Signal.RED, thresholds.EDGE_STRETCH_RED),
            ([1.0, thresholds.EDGE_SHRINK_YELLOW], Signal.YELLOW, thresholds.EDGE_SHRINK_YELLOW),
            ([1.0, thresholds.EDGE_SHRINK_RED], Signal.RED, thresholds.EDGE_SHRINK_RED),
            ([1.7, 0.65], Signal.RED, 1.7),  # 伸びが赤、縮みが黄 → 伸び
            ([1.35, 0.45], Signal.RED, 0.45),  # 縮みが赤
            ([1.0, 0.0], Signal.RED, 0.0),  # 長さ 0 につぶれた
        ]
        for ratios, signal, value in cases:
            with self.subTest(ratios=ratios):
                result = d.summarize_stretch(np.asarray(ratios))
                self.assertEqual(result.signal, signal)
                self.assertEqual(result.value, value)
                self.assertEqual(result.name, "edge_stretch")
                self.assertEqual(result.unit, "ratio")

    def test_same_signal_picks_farther_from_one(self):
        result = d.summarize_stretch([1.1, 0.8])  # どちらも緑。1/0.8 = 1.25 > 1.1
        self.assertEqual(result.signal, Signal.GREEN)
        self.assertEqual(result.value, 0.8)
        self.assertEqual(result.details["value_kind"], "shrink")
        result = d.summarize_stretch([1.7, 0.4])  # どちらも赤。1/0.4 = 2.5 > 1.7
        self.assertEqual(result.value, 0.4)

    def test_details(self):
        result = d.summarize_stretch([1.0, 1.4, 1.7, 0.6, 0.3, math.nan])
        self.assertEqual(result.details["stretch_yellow_count"], 2)
        self.assertEqual(result.details["stretch_red_count"], 1)
        self.assertEqual(result.details["shrink_yellow_count"], 2)
        self.assertEqual(result.details["shrink_red_count"], 1)
        self.assertEqual(result.details["unmeasured_count"], 1)
        np.testing.assert_array_equal(result.details["worst_stretch_indices"], [2, 1])
        np.testing.assert_array_equal(result.details["worst_shrink_indices"], [4, 3])

    def test_partial_nan_is_gray_unless_red(self):
        """測れなかった辺があれば、赤でないかぎり灰（測れなかった物は緑にしない。灰は黄より重い）。値は測れた辺から決める"""
        cases = [
            ([1.0, math.nan], Signal.GRAY, 1.0),
            ([1.0, thresholds.EDGE_STRETCH_YELLOW, math.nan], Signal.GRAY, thresholds.EDGE_STRETCH_YELLOW),
            ([math.nan, thresholds.EDGE_SHRINK_YELLOW], Signal.GRAY, thresholds.EDGE_SHRINK_YELLOW),
            ([1.0, thresholds.EDGE_STRETCH_RED, math.nan], Signal.RED, thresholds.EDGE_STRETCH_RED),
            ([math.nan, thresholds.EDGE_SHRINK_RED], Signal.RED, thresholds.EDGE_SHRINK_RED),
        ]
        for ratios, signal, value in cases:
            with self.subTest(ratios=ratios):
                result = d.summarize_stretch(ratios)
                self.assertEqual(result.signal, signal)
                self.assertEqual(result.value, value)
                self.assertIn("測れなかった辺", result.reason)
                self.assertEqual(result.details["unmeasured_count"], 1)

    def test_gray(self):
        self.assertEqual(d.summarize_stretch([]).signal, Signal.GRAY)
        self.assertEqual(d.summarize_stretch([math.nan, math.nan]).signal, Signal.GRAY)

    def test_bad_inputs(self):
        for bad in ([-0.1], [math.inf], [True, False], [[1.0]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                d.summarize_stretch(np.asarray(bad))


class SummarizeFlipsTest(unittest.TestCase):
    def test_signals(self):
        result = d.summarize_flips(np.array([False, True, True]), np.array([False, False, False]))
        self.assertEqual((result.name, result.unit), ("flipped_faces", "count"))
        self.assertEqual(result.signal, Signal.RED)
        self.assertEqual(result.value, 2.0)
        np.testing.assert_array_equal(result.details["flipped_indices"], [1, 2])
        result = d.summarize_flips(np.array([False, False]))
        self.assertEqual((result.signal, result.value), (Signal.GREEN, 0.0))
        # 比べられなかった三角形があれば、裏返りが無くても灰（値は見つかった裏返りの数 0）。裏返りがあれば赤
        result = d.summarize_flips(np.array([False, False]), np.array([True, False]))
        self.assertEqual((result.signal, result.value), (Signal.GRAY, 0.0))
        self.assertEqual(result.details["degenerate_count"], 1)
        self.assertIn("向きを比べられなかった", result.reason)
        result = d.summarize_flips(np.array([False, True, False]), np.array([True, False, False]))
        self.assertEqual((result.signal, result.value), (Signal.RED, 1.0))
        self.assertIn("向きを比べられなかった", result.reason)

    def test_gray(self):
        self.assertEqual(d.summarize_flips(np.zeros(0, dtype=bool)).signal, Signal.GRAY)
        self.assertEqual(d.summarize_flips(np.array([False]), np.array([True])).signal, Signal.GRAY)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            d.summarize_flips(np.array([0, 1]))
        with self.assertRaises(ValueError):
            d.summarize_flips(np.array([True]), np.array([True]))
        with self.assertRaises(ValueError):
            d.summarize_flips(np.array([True]), np.array([False, False]))

    def test_with_triangle_flips(self):
        v, t = shapes.icosphere(subdivisions=1)
        self.assertEqual(d.summarize_flips(*d.triangle_flips(v, rigid(v, 3), t)).signal, Signal.GREEN)
        posed, first = fold_vertex(v, t, 0)
        result = d.summarize_flips(*d.triangle_flips(v, posed, t))
        self.assertEqual(result.signal, Signal.RED)
        self.assertIn(first, result.details["flipped_indices"].tolist())


class SummarizeVolumeTest(unittest.TestCase):
    def test_signals(self):
        cases = [
            (-0.05, Signal.GREEN, 0.05),
            (-thresholds.VOLUME_LOSS_YELLOW, Signal.YELLOW, thresholds.VOLUME_LOSS_YELLOW),
            (-thresholds.VOLUME_LOSS_RED, Signal.RED, thresholds.VOLUME_LOSS_RED),
            (0.5, Signal.GREEN, 0.0),  # ふくらみは信号にしない
            (-2.0, Signal.RED, 2.0),  # 中と外が入れ替わった
            (0, Signal.GREEN, 0.0),
        ]
        for ratio, signal, loss in cases:
            with self.subTest(ratio=ratio):
                result = d.summarize_volume(ratio)
                self.assertEqual((result.name, result.unit), ("volume_loss", "ratio"))
                self.assertEqual(result.signal, signal)
                self.assertEqual(result.value, loss)
        self.assertEqual(d.summarize_volume(0.5).details["gain"], 0.5)
        self.assertTrue(d.summarize_volume(-2.0).details["inside_out"])

    def test_gray(self):
        self.assertEqual(d.summarize_volume(None).signal, Signal.GRAY)
        self.assertEqual(d.summarize_volume(math.nan).signal, Signal.GRAY)
        self.assertEqual(d.summarize_volume(np.float64("nan")).signal, Signal.GRAY)

    def test_bad_inputs(self):
        for bad in (True, math.inf, "0.1"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                d.summarize_volume(bad)

    def test_with_volume_change_ratio(self):
        v, t = shapes.icosphere(subdivisions=2)
        result = d.summarize_volume(d.volume_change_ratio(v, v * 0.9, t))  # 0.9^3 = 0.729：27% 減る
        self.assertEqual(result.signal, Signal.RED)
        self.assertAlmostEqual(result.value, 0.271, places=12)


class SummarizeSpikesTest(unittest.TestCase):
    def test_signals(self):
        cases = [
            ([0.1, 0.0], Signal.GREEN, 0.1),
            ([0.0, thresholds.SPIKE_YELLOW], Signal.YELLOW, thresholds.SPIKE_YELLOW),
            ([thresholds.SPIKE_RED, 0.2], Signal.RED, thresholds.SPIKE_RED),
        ]
        for scores, signal, value in cases:
            with self.subTest(scores=scores):
                result = d.summarize_spikes(np.asarray(scores))
                self.assertEqual((result.name, result.unit), ("spike", "ratio"))
                self.assertEqual(result.signal, signal)
                self.assertEqual(result.value, value)
        result = d.summarize_spikes([0.2, 1.5, math.nan, 0.7])
        self.assertEqual(result.signal, Signal.RED)  # 測れなかった頂点があっても、赤は赤
        self.assertEqual(result.details["unmeasured_count"], 1)
        self.assertEqual(result.details["red_count"], 1)
        self.assertEqual(result.details["yellow_count"], 2)
        np.testing.assert_array_equal(result.details["worst_indices"], [1, 3, 0])

    def test_gray(self):
        self.assertEqual(d.summarize_spikes([]).signal, Signal.GRAY)
        self.assertEqual(d.summarize_spikes([math.nan]).signal, Signal.GRAY)

    def test_partial_nan_is_gray_unless_red(self):
        """測れなかった頂点があれば、赤でないかぎり灰（測れなかった物は緑にしない。灰は黄より重い）。値は測れた頂点から決める"""
        for scores, value in (([0.1, math.nan], 0.1), ([thresholds.SPIKE_YELLOW, math.nan], thresholds.SPIKE_YELLOW)):
            with self.subTest(scores=scores):
                result = d.summarize_spikes(scores)
                self.assertEqual((result.signal, result.value), (Signal.GRAY, value))
                self.assertIn("測れなかった頂点", result.reason)
        result = d.summarize_spikes([math.nan, thresholds.SPIKE_RED])
        self.assertEqual((result.signal, result.value), (Signal.RED, thresholds.SPIKE_RED))

    def test_bad_inputs(self):
        for bad in ([-0.1], [math.inf], [True]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                d.summarize_spikes(np.asarray(bad))

    def test_with_spike_scores(self):
        v, t = shapes.icosphere(subdivisions=2)
        self.assertEqual(d.summarize_spikes(d.spike_scores(v, rigid(v, 0), t)).signal, Signal.GREEN)
        posed = v.copy()
        posed[3] *= 1.5
        result = d.summarize_spikes(d.spike_scores(v, posed, t))
        self.assertEqual(result.signal, Signal.RED)
        self.assertEqual(result.details["worst_indices"][0], 3)


if __name__ == "__main__":
    unittest.main()

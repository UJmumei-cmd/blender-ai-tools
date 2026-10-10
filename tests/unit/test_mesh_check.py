"""A4 面の点検（measure/mesh_check.py）を確かめる"""

import unittest

import numpy as np

from blender_ai_tools import thresholds
from blender_ai_tools.measure import mesh_check
from blender_ai_tools.measure.mesh_check import (boundary_edges, degenerate_triangles, duplicate_vertex_groups,
                                                 edge_face_counts, face_components, inconsistent_edges,
                                                 loose_parts, non_manifold_edges, orient_consistently,
                                                 summarize_degenerate, summarize_duplicates, summarize_holes,
                                                 summarize_inconsistent, summarize_loose_parts, summarize_mesh,
                                                 summarize_non_manifold, unused_vertices)
from blender_ai_tools.measure.results import Signal, worst
from tests import shapes


def flip(triangles, rows):
    result = np.array(triangles, copy=True)
    result[rows] = result[rows][:, [0, 2, 1]]
    return result


def as_lists(groups):
    return [group.tolist() for group in groups]


def thin_l_prism():
    """細い L の字の柱（閉じた外向き）。へこんだ側の面（番号 12〜15）は、外向きの法線が形の重心のほうを向く"""
    polygon = np.array([(0, 0), (4, 0), (4, 0.5), (0.5, 0.5), (0.5, 4), (0, 4)], dtype=float)
    caps = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5)]
    k = len(polygon)
    vertices = np.vstack([np.column_stack([polygon, np.zeros(k)]), np.column_stack([polygon, np.ones(k)])])
    triangles = [[i, kk, j] for i, j, kk in caps] + [[k + i, k + j, k + kk] for i, j, kk in caps]
    for i in range(k):
        n = (i + 1) % k
        triangles += [[i, n, k + n], [i, k + n, k + i]]
    return vertices, np.asarray(triangles, dtype=np.int64)


def torus(major=1.0, minor=0.3, around=12, tube=8):
    """穴の開いた輪（閉じた外向き。へこんだ所がある）"""
    u = 2 * np.pi * np.arange(around) / around
    w = 2 * np.pi * np.arange(tube) / tube
    uu, ww = np.meshgrid(u, w, indexing="ij")
    vertices = np.stack([(major + minor * np.cos(ww)) * np.cos(uu), (major + minor * np.cos(ww)) * np.sin(uu),
                         minor * np.sin(ww)], axis=-1).reshape(-1, 3)
    triangles = []
    for i in range(around):
        for j in range(tube):
            a, b = i * tube + j, ((i + 1) % around) * tube + j
            c, d = ((i + 1) % around) * tube + (j + 1) % tube, i * tube + (j + 1) % tube
            triangles += [[a, b, c], [a, c, d]]
    return vertices, np.asarray(triangles, dtype=np.int64)


def naive_groups(points, distance):
    """素朴な方法：全部の組を比べ、つながった物をたどる"""
    n = len(points)
    linked = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2) <= distance
    seen = [False] * n
    groups = []
    for start in range(n):
        if seen[start]:
            continue
        stack, members = [start], []
        seen[start] = True
        while stack:
            i = stack.pop()
            members.append(i)
            for j in np.flatnonzero(linked[i]):
                if not seen[j]:
                    seen[j] = True
                    stack.append(int(j))
        if len(members) > 1:
            groups.append(sorted(members))
    return groups


# MARK: - 重なった点


class DuplicateVertexGroupsTest(unittest.TestCase):
    def test_simple_pair(self):
        points = [[0, 0, 0], [0.00005, 0, 0], [1, 1, 1]]
        self.assertEqual(as_lists(duplicate_vertex_groups(points)), [[0, 1]])

    def test_chain_joins_far_ends(self):
        # a〜b、b〜c が近ければ、a と c が離れていても1つ（距離 0.25、a と c は 0.4 離れている）
        points = np.array([[0.4, 0, 0], [5, 5, 5], [0.0, 0, 0], [0.2, 0, 0]])
        self.assertEqual(as_lists(duplicate_vertex_groups(points, 0.25)), [[0, 2, 3]])
        # 真ん中が無ければ、つながらない
        self.assertEqual(duplicate_vertex_groups(points[[0, 1, 2]], 0.25), [])

    def test_distance_exactly_equal_is_linked(self):
        d = 0.25
        self.assertEqual(as_lists(duplicate_vertex_groups([[0, 0, 0], [d, 0, 0]], d)), [[0, 1]])
        self.assertEqual(duplicate_vertex_groups([[0, 0, 0], [np.nextafter(d, 1), 0, 0]], d), [])
        default = thresholds.MERGE_DISTANCE
        self.assertEqual(as_lists(duplicate_vertex_groups([[0, 0, 0], [default, 0, 0]])), [[0, 1]])
        self.assertEqual(duplicate_vertex_groups([[0, 0, 0], [0, 0, np.nextafter(default, 1)]]), [])

    def test_cell_boundaries(self):
        d = 0.25
        cases = [
            [[0.24, 0, 0], [0.26, 0, 0]],  # 升目の境（0.25 付近）をまたぐ
            [[-0.001, 0, 0], [0.001, 0, 0]],  # 0 をまたぐ（負の升目）
            [[0.249, 0.249, 0.249], [0.251, 0.251, 0.251]],  # 3つの軸とも升目が違う（斜めのとなり）
            [[0.249, 0.251, 0.249], [0.251, 0.249, 0.251]],
            [[1e6 + 0.1, 0, 0], [1e6 + 0.1 + d, 0, 0]],  # 原点から遠い所
        ]
        for points in cases:
            with self.subTest(points=points):
                expected = naive_groups(np.asarray(points, dtype=float), d)
                self.assertEqual(as_lists(duplicate_vertex_groups(points, d)), expected)
                self.assertEqual(len(expected), 1)

    def test_order_is_deterministic(self):
        points = np.zeros((10, 3))
        points[:, 0] = np.arange(10) * 10.0
        points[7] = points[3]
        points[5] = points[1]
        points[9] = points[1] + [0, 0.00001, 0]
        self.assertEqual(as_lists(duplicate_vertex_groups(points)), [[1, 5, 9], [3, 7]])
        for group in duplicate_vertex_groups(points):
            self.assertEqual(group.dtype, np.int64)

    def test_matches_naive_on_random_clusters(self):
        rng = np.random.default_rng(99)
        centers = rng.uniform(-1, 1, size=(40, 3))
        points = centers[rng.integers(0, 40, size=400)] + rng.normal(scale=0.01, size=(400, 3))
        points[::17] = points[::17][::-1]  # ぴったり同じ点もまぜる
        for distance in (0.0, 0.003, 0.01, 0.02, 0.05, 0.5):
            with self.subTest(distance=distance):
                self.assertEqual(as_lists(duplicate_vertex_groups(points, distance)), naive_groups(points, distance))

    def test_zero_distance_means_exact_copies(self):
        points = [[0, 0, 0], [1e-12, 0, 0], [0, 0, 0], [-0.0, 0, 0]]
        self.assertEqual(as_lists(duplicate_vertex_groups(points, 0.0)), [[0, 2, 3]])

    def test_many_exact_copies(self):
        points = np.zeros((5000, 3))
        groups = duplicate_vertex_groups(points)
        self.assertEqual(len(groups), 1)
        np.testing.assert_array_equal(groups[0], np.arange(5000))

    def test_empty_single_and_invalid(self):
        self.assertEqual(duplicate_vertex_groups(np.zeros((0, 3))), [])
        self.assertEqual(duplicate_vertex_groups([[1, 2, 3]]), [])
        for bad in (-0.1, np.nan, np.inf, True, "0.1"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                duplicate_vertex_groups([[0, 0, 0], [1, 1, 1]], bad)
        with self.assertRaises(ValueError):
            duplicate_vertex_groups([[0, 0, np.nan]])


# MARK: - つぶれた三角形と辺


class DegenerateTrianglesTest(unittest.TestCase):
    def test_finds_repeated_and_zero_area(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [1e-7, 0, 0], [0, 1e-5, 0]], dtype=float)
        triangles = [[0, 1, 2], [0, 1, 3], [0, 0, 2], [0, 4, 5], [2, 2, 2], [1, 2, 3]]
        # 面積：0.5、0（一直線）、同じ頂点、5e-13（基準 1e-12 以下）、同じ頂点、0.5
        np.testing.assert_array_equal(degenerate_triangles(vertices, triangles), [1, 2, 3, 4])
        self.assertEqual(degenerate_triangles(vertices, triangles).dtype, np.int64)
        # 基準ちょうども含む
        np.testing.assert_array_equal(degenerate_triangles(vertices, triangles, area=0.5), [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(degenerate_triangles(vertices, triangles, area=0.0), [1, 2, 4])

    def test_box_has_none(self):
        self.assertEqual(len(degenerate_triangles(*shapes.box())), 0)
        with self.assertRaises(ValueError):
            degenerate_triangles(*shapes.box(), area=-1.0)


class EdgeTest(unittest.TestCase):
    def test_box_edges_all_used_twice(self):
        _, triangles = shapes.box()
        edges, counts = edge_face_counts(triangles)
        self.assertEqual(edges.shape, (18, 2))  # 箱の辺 12 本 + 面の対角線 6 本
        np.testing.assert_array_equal(counts, np.full(18, 2))
        self.assertTrue(np.all(edges[:, 0] < edges[:, 1]))
        self.assertEqual(edges.tolist(), sorted(edges.tolist()))
        self.assertEqual(len(boundary_edges(triangles)), 0)
        self.assertEqual(len(non_manifold_edges(triangles)), 0)
        self.assertEqual(len(inconsistent_edges(triangles)), 0)

    def test_grid_boundary(self):
        _, triangles = shapes.grid(2, 2)
        edges, counts = edge_face_counts(triangles)
        self.assertEqual(len(edges), 16)  # 縦横 12 本 + 対角線 4 本
        self.assertEqual(int((counts == 1).sum()), 8)
        border = boundary_edges(triangles)
        expected = [[0, 1], [0, 3], [1, 2], [2, 5], [3, 6], [5, 8], [6, 7], [7, 8]]
        self.assertEqual(border.tolist(), expected)

    def test_non_manifold_fin(self):
        vertices, triangles = shapes.box()
        fin = np.vstack([triangles, [[4, 5, 8]]])  # 上の面の辺 4-5 に、3つ目の面
        edges, counts = edge_face_counts(fin)
        self.assertEqual(non_manifold_edges(fin).tolist(), [[4, 5]])
        self.assertEqual(int(counts[(edges[:, 0] == 4) & (edges[:, 1] == 5)][0]), 3)
        self.assertEqual(boundary_edges(fin).tolist(), [[4, 8], [5, 8]])
        # 3つ以上の面の辺は、向きの点検に入れない
        self.assertEqual(len(inconsistent_edges(fin)), 0)

    def test_inconsistent_edges_of_one_flipped_face(self):
        _, triangles = shapes.box()
        flipped = flip(triangles, [0])  # [0, 2, 1] → [0, 1, 2]
        self.assertEqual(inconsistent_edges(flipped).tolist(), [[0, 1], [0, 2], [1, 2]])

    def test_large_vertex_numbers(self):
        big = 10 ** 12
        edges, counts = edge_face_counts([[big, 5, big + 7]])
        self.assertEqual(edges.tolist(), [[5, big], [5, big + 7], [big, big + 7]])
        np.testing.assert_array_equal(counts, [1, 1, 1])

    def test_degenerate_edges_are_skipped(self):
        edges, counts = edge_face_counts([[0, 0, 1]])
        self.assertEqual(edges.tolist(), [[0, 1]])
        self.assertEqual(counts.tolist(), [2])

    def test_empty_and_invalid(self):
        edges, counts = edge_face_counts(np.zeros((0, 3), dtype=np.int64))
        self.assertEqual(edges.shape, (0, 2))
        self.assertEqual(counts.shape, (0,))
        for bad in ([[0, 1]], [[0, 1, -1]], [[0.0, 1.0, 2.0]], np.array([[True, False, True]])):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                edge_face_counts(bad)


# MARK: - 面の向き


class OrientTest(unittest.TestCase):
    def assert_consistent(self, triangles):
        self.assertEqual(len(inconsistent_edges(triangles)), 0)

    def test_consistent_box_is_unchanged(self):
        vertices, triangles = shapes.box()
        fixed, flipped = orient_consistently(vertices, triangles)
        np.testing.assert_array_equal(fixed, triangles)
        self.assertFalse(flipped.any())
        self.assertEqual(flipped.dtype, bool)

    def test_box_with_some_faces_flipped(self):
        vertices, triangles = shapes.box()
        for rows in ([0], [0, 5, 9], [1, 2, 3, 4, 6, 7]):
            with self.subTest(rows=rows):
                fixed, flipped = orient_consistently(vertices, flip(triangles, rows))
                np.testing.assert_array_equal(fixed, triangles)
                np.testing.assert_array_equal(np.flatnonzero(flipped), rows)

    def test_inside_out_box_becomes_outward(self):
        vertices, triangles = shapes.box(size=(1.0, 2.0, 3.0), center=(5.0, -3.0, 2.0))
        inside_out = flip(triangles, np.arange(12))
        self.assertLess(shapes.signed_volume(vertices, inside_out), 0)
        fixed, flipped = orient_consistently(vertices, inside_out)
        self.assertTrue(flipped.all())
        np.testing.assert_array_equal(fixed, triangles)
        self.assertAlmostEqual(shapes.signed_volume(vertices, fixed), 6.0)
        # 大部分が裏返った箱（基準の面 0 も裏返っている）も外向きになる
        mostly = flip(triangles, [0, 1, 2, 3, 4, 5, 6, 7, 8])
        fixed, flipped = orient_consistently(vertices, mostly)
        np.testing.assert_array_equal(fixed, triangles)
        np.testing.assert_array_equal(np.flatnonzero(flipped), np.arange(9))

    def test_non_convex_inside_out_becomes_outward(self):
        # へこんだ形：基準の面（番号 0）を、へこんだ側の面にする。外向きの法線が重心のほうを向くので、
        # 「基準の面の法線が重心から外を向くか」で決める方法では、ここで逆にしてしまう（体積で決めれば正しい）
        vertices, triangles = thin_l_prism()
        self.assertAlmostEqual(shapes.signed_volume(vertices, triangles), 3.75)
        order = np.r_[12, np.setdiff1d(np.arange(len(triangles)), [12])]
        outward = triangles[order]
        normal = np.cross(*(vertices[outward[0, 1:]] - vertices[outward[0, 0]]))
        self.assertLess(float(normal @ (vertices[outward[0]].mean(axis=0) - vertices.mean(axis=0))), 0.0)
        fixed, flipped = orient_consistently(vertices, flip(outward, np.arange(len(outward))))
        np.testing.assert_array_equal(fixed, outward)
        self.assertTrue(flipped.all())
        # 穴の開いた輪（種数 1）でも、ばらばらに裏返した面が元に戻る
        ring_vertices, ring = torus()
        self.assertGreater(shapes.signed_volume(ring_vertices, ring), 0.0)
        self.assertEqual(len(inconsistent_edges(ring)), 0)
        rng = np.random.default_rng(31)
        for _ in range(5):
            rows = np.flatnonzero(rng.random(len(ring)) < rng.uniform(0.2, 0.9))
            fixed, flipped = orient_consistently(ring_vertices, flip(ring, rows))
            np.testing.assert_array_equal(fixed, ring)
            np.testing.assert_array_equal(np.flatnonzero(flipped), rows)

    def test_random_flips_on_sphere(self):
        vertices, triangles = shapes.icosphere(radius=0.3, subdivisions=2)
        rng = np.random.default_rng(8)
        for _ in range(5):
            rows = np.flatnonzero(rng.random(len(triangles)) < 0.4)
            fixed, flipped = orient_consistently(vertices, flip(triangles, rows))
            np.testing.assert_array_equal(fixed, triangles)
            np.testing.assert_array_equal(np.flatnonzero(flipped), rows)

    def test_open_grid_keeps_majority(self):
        vertices, triangles = shapes.grid(3, 3)
        fixed, flipped = orient_consistently(vertices, flip(triangles, [1, 4, 17]))
        np.testing.assert_array_equal(fixed, triangles)
        np.testing.assert_array_equal(np.flatnonzero(flipped), [1, 4, 17])
        # 大部分が裏返っていれば（-Z 向き）、そちらに合わせる
        rows = np.setdiff1d(np.arange(18), [2, 9])
        fixed, flipped = orient_consistently(vertices, flip(triangles, rows))
        np.testing.assert_array_equal(fixed, flip(triangles, np.arange(18)))
        np.testing.assert_array_equal(np.flatnonzero(flipped), [2, 9])

    def test_majority_is_by_area_and_tie_keeps_seed(self):
        # 大きい三角形 0（面積 50）と、小さい三角形 1・2（合わせて約 17）が辺でつながる。数ではなく面積の多いほうに合わせる
        vertices = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [5.5, 5.5, 0], [5.5, 10, 0]], dtype=float)
        triangles = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]])
        self.assertEqual(len(inconsistent_edges(triangles)), 0)
        fixed, flipped = orient_consistently(vertices, flip(triangles, [1, 2]))
        np.testing.assert_array_equal(np.flatnonzero(flipped), [1, 2])
        np.testing.assert_array_equal(fixed, triangles)
        # 大きい三角形だけが裏返っていても、面積はそちらが多いので、小さい2つを合わせる（数で決めると逆になる）
        fixed, flipped = orient_consistently(vertices, flip(triangles, [0]))
        np.testing.assert_array_equal(np.flatnonzero(flipped), [1, 2])
        np.testing.assert_array_equal(fixed, flip(triangles, [0, 1, 2]))
        # 同じ面積の2つ：基準（番号の小さい面）の向きのまま
        square = np.array([[0, 1, 2], [0, 2, 3]])
        quad = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        fixed, flipped = orient_consistently(quad, flip(square, [1]))
        np.testing.assert_array_equal(np.flatnonzero(flipped), [1])
        np.testing.assert_array_equal(fixed, square)

    def test_separate_parts_are_fixed_separately(self):
        vertices, triangles = shapes.box()
        both_vertices = np.vstack([vertices, vertices + [3.0, 0.0, 0.0]])
        both = np.vstack([flip(triangles, [2, 3]), flip(triangles + 8, np.arange(12))])
        fixed, flipped = orient_consistently(both_vertices, both)
        np.testing.assert_array_equal(fixed, np.vstack([triangles, triangles + 8]))
        np.testing.assert_array_equal(np.flatnonzero(flipped), [2, 3] + list(range(12, 24)))
        self.assert_consistent(fixed)

    def test_does_not_modify_input_and_empty(self):
        vertices, triangles = shapes.box()
        given = flip(triangles, [3])
        before = given.copy()
        orient_consistently(vertices, given)
        np.testing.assert_array_equal(given, before)
        fixed, flipped = orient_consistently(vertices, np.zeros((0, 3), dtype=np.int64))
        self.assertEqual(fixed.shape, (0, 3))
        self.assertEqual(flipped.shape, (0,))


# MARK: - 部品


class ComponentsTest(unittest.TestCase):
    def test_two_boxes(self):
        _, triangles = shapes.box()
        labels, count = face_components(np.vstack([triangles, triangles + 8]))
        self.assertEqual(count, 2)
        np.testing.assert_array_equal(labels, [0] * 12 + [1] * 12)
        self.assertEqual(labels.dtype, np.int64)

    def test_labels_follow_smallest_face_and_vertex_sharing(self):
        # 面 0 と 2 は頂点 0 だけを共有する（同じ部品）。面 1 は別
        triangles = [[0, 1, 2], [5, 6, 7], [0, 3, 4]]
        labels, count = face_components(triangles)
        self.assertEqual(count, 2)
        np.testing.assert_array_equal(labels, [0, 1, 0])
        labels, count = face_components([[5, 6, 7], [0, 1, 2]])
        np.testing.assert_array_equal(labels, [0, 1])
        labels, count = face_components([[0, 1, 2], [10 ** 12, 10 ** 12 + 1, 2], [7, 8, 9]])  # 大きな番号でもよい
        np.testing.assert_array_equal(labels, [0, 0, 1])
        labels, count = face_components(np.zeros((0, 3), dtype=np.int64))
        self.assertEqual((labels.shape, count), ((0,), 0))

    def test_loose_parts(self):
        vertices, triangles = shapes.box()
        stray = np.array([[9, 9, 9], [9, 9.1, 9], [9.1, 9, 9]])
        all_vertices = np.vstack([stray, vertices, vertices + 3.0])
        all_triangles = np.vstack([[[0, 1, 2]], triangles + 3, triangles + 11])
        parts = loose_parts(all_vertices, all_triangles)
        self.assertEqual(as_lists(parts), [[0]])
        # 基準を 12 にすると箱の片方も入る（いちばん大きい部品＝番号の小さい箱は入れない）
        self.assertEqual(as_lists(loose_parts(all_vertices, all_triangles, max_faces=12)), [[0], list(range(13, 25))])
        # 部品が1つだけなら、小さくても入れない
        self.assertEqual(loose_parts(stray, [[0, 1, 2]]), [])
        with self.assertRaises(ValueError):
            loose_parts(all_vertices, all_triangles, max_faces=True)
        with self.assertRaises(ValueError):
            loose_parts(all_vertices, all_triangles, max_faces=0)

    def test_unused_vertices(self):
        vertices = np.zeros((6, 3))
        np.testing.assert_array_equal(unused_vertices(vertices, [[0, 2, 3]]), [1, 4, 5])


# MARK: - 信号


class SummaryTest(unittest.TestCase):
    def test_clean_box_is_green(self):
        vertices, triangles = shapes.box()
        results = summarize_mesh(vertices, triangles, closed_required=True)
        self.assertEqual([r.name for r in results], ["duplicate_vertices", "degenerate_triangles", "non_manifold_edges",
                                                     "inconsistent_edges", "loose_parts", "holes"])
        self.assertEqual(worst(r.signal for r in results), Signal.GREEN)
        self.assertTrue(all(r.value == 0.0 and r.unit == "count" for r in results))

    def test_signals(self):
        vertices, triangles = shapes.box()
        duplicated = np.vstack([vertices, vertices[[0, 0, 3]]])
        result = summarize_duplicates(duplicated)
        self.assertEqual((result.signal, result.value, result.details["group_count"]), (Signal.YELLOW, 3.0, 2))
        self.assertEqual(summarize_inconsistent(flip(triangles, [4])).signal, Signal.RED)
        self.assertEqual(summarize_inconsistent(flip(triangles, [4])).value, 3.0)
        fin = np.vstack([triangles, [[4, 5, 0]]])
        self.assertEqual(summarize_non_manifold(fin).signal, Signal.YELLOW)
        open_box = triangles[:-1]
        self.assertEqual(summarize_holes(open_box).signal, Signal.RED)
        self.assertEqual(summarize_holes(open_box).value, 1.0)  # 穴は1つ（縁の辺は3本）
        self.assertEqual(summarize_holes(open_box).details["boundary_edge_count"], 3)
        self.assertEqual(summarize_degenerate(vertices, np.vstack([triangles, [[0, 0, 1]]])).signal, Signal.YELLOW)
        stray = np.vstack([vertices, [[9, 9, 9], [9, 9.1, 9], [9.1, 9, 9]]])
        self.assertEqual(summarize_loose_parts(stray, np.vstack([triangles, [[8, 9, 10]]])).signal, Signal.YELLOW)
        # 開いた形は、閉じている必要がなければ穴を調べない
        grid_vertices, grid_triangles = shapes.grid(2, 2)
        self.assertEqual(worst(r.signal for r in summarize_mesh(grid_vertices, grid_triangles)), Signal.GREEN)
        with self.assertRaises(ValueError):
            summarize_mesh(vertices, triangles, closed_required=1)

    def test_empty_mesh_is_gray_not_green(self):
        # 三角形が1つも無い形は、何も調べていない（測れなかった物は緑にしない）。
        # 取り出しをまちがえて空の体を渡したとき、「体の穴が0」で合格にしない
        vertices, _ = shapes.box()
        empty_triangles = np.zeros((0, 3), dtype=np.int64)
        for given_vertices in (vertices, np.zeros((0, 3))):
            results = summarize_mesh(given_vertices, empty_triangles, closed_required=True)
            by_name = {r.name: r for r in results}
            for name in ("degenerate_triangles", "non_manifold_edges", "inconsistent_edges", "loose_parts", "holes"):
                with self.subTest(name=name, vertices=len(given_vertices)):
                    self.assertEqual(by_name[name].signal, Signal.GRAY)
                    self.assertIsNone(by_name[name].value)
                    self.assertTrue(by_name[name].reason)
        self.assertEqual(summarize_holes(empty_triangles).signal, Signal.GRAY)
        self.assertEqual(summarize_inconsistent(empty_triangles).signal, Signal.GRAY)
        self.assertEqual(summarize_non_manifold(empty_triangles).signal, Signal.GRAY)
        self.assertEqual(summarize_degenerate(vertices, empty_triangles).signal, Signal.GRAY)
        self.assertEqual(summarize_loose_parts(vertices, empty_triangles).signal, Signal.GRAY)
        # 頂点が無ければ、重なった点も調べていない。頂点が1つなら、調べた上で重なりは無い（緑）
        self.assertEqual(summarize_duplicates(np.zeros((0, 3))).signal, Signal.GRAY)
        self.assertEqual(summarize_duplicates([[1.0, 2.0, 3.0]]).signal, Signal.GREEN)
        # 三角形が1つでもあれば、ふつうに調べる
        single = summarize_mesh(vertices, [[0, 1, 2]], closed_required=True)
        self.assertNotIn(Signal.GRAY, [r.signal for r in single])

    def test_holes_counts_boundary_loops_not_edges(self):
        # 値は「穴の数」（縁の辺がつながった輪の数）。報告の「穴が N 個」に入るので、辺の本数ではいけない
        _, triangles = shapes.box()
        cases = [
            (np.delete(triangles, [0, 2], axis=0), 2, 6),  # 下と上に、三角形の穴が1つずつ
            (np.delete(triangles, [0, 1], axis=0), 1, 4),  # 下の面の四角い穴が1つ
            (np.delete(triangles, [0, 1, 2, 3], axis=0), 2, 8),  # 上下が抜けた筒：輪が2つ
            (shapes.grid(3, 2)[1], 1, 10),  # 開いた格子：外の縁が1つ
            (triangles, 0, 0),
        ]
        for tris, holes, edges in cases:
            with self.subTest(holes=holes, edges=edges):
                result = summarize_holes(tris)
                self.assertEqual(result.value, float(holes))
                self.assertEqual(result.details["boundary_edge_count"], edges)
                self.assertEqual(len(result.details["edges"]), edges)
                self.assertEqual(result.signal, Signal.RED if holes else Signal.GREEN)

    def test_loose_part_threshold_is_provisional_constant(self):
        self.assertIsInstance(mesh_check.LOOSE_PART_MAX_FACES, int)
        self.assertGreaterEqual(mesh_check.LOOSE_PART_MAX_FACES, 1)


if __name__ == "__main__":
    unittest.main()

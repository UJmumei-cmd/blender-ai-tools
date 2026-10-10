"""A4 めり込み・すき間の形の計算（measure/geometry.py）を確かめる"""

import math
import tracemalloc
import unittest

import numpy as np

from blender_ai_tools import thresholds
from blender_ai_tools.measure import geometry
from blender_ai_tools.measure.geometry import (closest_points, gaps, penetration_depths, signed_distance,
                                               summarize_gap, summarize_penetration, triangle_areas,
                                               triangle_normals, vertex_normals, weld_exact)
from blender_ai_tools.measure.results import Signal
from tests import shapes

# MARK: - テスト用の形と、くらべるための素朴な計算


def prism(polygon, caps, z0=0.0, z1=1.0):
    """反時計回りの多角形を z0〜z1 に伸ばした、閉じた外向きの柱。caps は多角形の三角形分け（反時計回り）"""
    polygon = np.asarray(polygon, dtype=np.float64)
    k = len(polygon)
    vertices = np.vstack([np.column_stack([polygon, np.full(k, z0)]), np.column_stack([polygon, np.full(k, z1)])])
    triangles = [[i, kk, j] for i, j, kk in caps]  # 下（-Z 向き）
    triangles += [[k + i, k + j, k + kk] for i, j, kk in caps]  # 上（+Z 向き）
    for i in range(k):
        n = (i + 1) % k
        triangles += [[i, n, k + n], [i, k + n, k + i]]
    return vertices, np.asarray(triangles, dtype=np.int64)


# とがった辺を持つくさび：断面は (0,0)・(4,0)・(0,1)。(4,0) の角は約 14 度
WEDGE_POLYGON = [(0.0, 0.0), (4.0, 0.0), (0.0, 1.0)]
WEDGE_CAPS = [(0, 1, 2)]
# L の字（へこんだ辺がある）。(0,0) から扇に分けられる
L_POLYGON = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)]
L_CAPS = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5)]


def inside_wedge(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return (z > 0) & (z < 1) & (x > 0) & (y > 0) & (x + 4 * y < 4)


def inside_l(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return (z > 0) & (z < 1) & (((x > 0) & (x < 2) & (y > 0) & (y < 1)) | ((x > 0) & (x < 1) & (y > 0) & (y < 2)))


def split_vertices(vertices, triangles):
    """三角形ごとに頂点を別々にした形（UV で分かれた頂点のように、座標は同じでも番号が違う）"""
    return vertices[triangles.ravel()].copy(), np.arange(triangles.size, dtype=np.int64).reshape(-1, 3)


def naive_point_triangle(p, a, b, c):
    """素朴な方法：面に映した点が中にあればそれ、なければ3本の線分のいちばん近い点。(距離, 点)"""
    best = (math.inf, None)
    for s0, s1 in ((a, b), (b, c), (c, a)):
        e = s1 - s0
        length2 = float(e @ e)
        t = 0.0 if length2 == 0.0 else min(1.0, max(0.0, float((p - s0) @ e) / length2))
        q = s0 + t * e
        d = float(np.linalg.norm(p - q))
        if d < best[0]:
            best = (d, q)
    n = np.cross(b - a, c - a)
    longest2 = max(float((b - a) @ (b - a)), float((c - b) @ (c - b)), float((a - c) @ (a - c)))
    if float(np.linalg.norm(n)) > 1e-6 * longest2:
        q = p - float((p - a) @ n) / float(n @ n) * n
        inside = all(float(np.cross(s1 - s0, q - s0) @ n) >= 0 for s0, s1 in ((a, b), (b, c), (c, a)))
        if inside:
            d = float(np.linalg.norm(p - q))
            if d < best[0]:
                best = (d, q)
    return best


class NoFloatErrors:
    """0 での割り算・NaN を作る計算があれば、エラーにする（隠れた NaN を見つける）"""

    def __enter__(self):
        self.state = np.seterr(divide="raise", invalid="raise", over="raise")

    def __exit__(self, *args):
        np.seterr(**self.state)


# MARK: - 法線と面積


class NormalsAndAreasTest(unittest.TestCase):
    def test_box_triangle_normals_point_outward(self):
        vertices, triangles = shapes.box()
        expected = np.repeat([[0, 0, -1], [0, 0, 1], [0, -1, 0], [0, 1, 0], [1, 0, 0], [-1, 0, 0]], 2, axis=0)
        np.testing.assert_allclose(triangle_normals(vertices, triangles), expected, atol=1e-15)

    def test_areas(self):
        vertices, triangles = shapes.box(size=(2.0, 3.0, 4.0))
        areas = triangle_areas(vertices, triangles)
        np.testing.assert_allclose(areas, np.repeat([3.0, 3.0, 4.0, 4.0, 6.0, 6.0], 2))
        self.assertAlmostEqual(areas.sum(), 2 * (6 + 8 + 12))

    def test_degenerate_triangles_have_zero_normal_and_area(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=float)
        triangles = np.array([[0, 1, 2], [0, 0, 3], [3, 3, 3], [0, 1, 3]])
        with NoFloatErrors():
            normals = triangle_normals(vertices, triangles)
            areas = triangle_areas(vertices, triangles)
        np.testing.assert_array_equal(normals[:3], np.zeros((3, 3)))
        np.testing.assert_allclose(normals[3], [0, 0, 1])
        np.testing.assert_array_equal(areas, [0.0, 0.0, 0.0, 0.5])

    def test_vertex_normals_are_angle_weighted(self):
        # 箱の角では、3つの面が同じ重み（どの面も角は 90 度）。三角形の数で重みを付けると、ここがずれる
        # （左の面では、頂点 0 を使う三角形が1つ、下と前の面では2つ）
        vertices, triangles = shapes.box()
        normals = vertex_normals(vertices, triangles)
        expected = np.sign(vertices) / math.sqrt(3.0)
        np.testing.assert_allclose(normals, expected, atol=1e-12)

    def test_vertex_normals_independent_of_triangulation(self):
        # 上の面だけ、対角線を逆にして分けても、角で重みを付ければ同じ答え
        vertices, triangles = shapes.box()
        other = triangles.copy()
        other[2:4] = [[4, 5, 7], [5, 6, 7]]
        np.testing.assert_allclose(vertex_normals(vertices, other), vertex_normals(vertices, triangles), atol=1e-12)

    def test_unused_vertex_has_zero_normal(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]], dtype=float)
        normals = vertex_normals(vertices, [[0, 1, 2]])
        np.testing.assert_allclose(normals[:3], [[0, 0, 1]] * 3)
        np.testing.assert_array_equal(normals[3], [0, 0, 0])

    def test_weld_exact(self):
        vertices = np.array([[1, 2, 3], [0, 0, 0], [1, 2, 3], [-0.0, 0, 0], [1, 2, 3.0000001]], dtype=float)
        np.testing.assert_array_equal(weld_exact(vertices), [0, 1, 0, 1, 4])
        self.assertEqual(weld_exact(np.zeros((0, 3))).shape, (0,))


# MARK: - いちばん近い点


class ClosestPointsTest(unittest.TestCase):
    A, B, C = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)

    def single(self, query):
        vertices = np.array([self.A, self.B, self.C])
        return closest_points(np.asarray(query, dtype=float).reshape(-1, 3), vertices, [[0, 1, 2]])

    def test_each_voronoi_region_of_one_triangle(self):
        cases = [
            # (点, いちばん近い点, 場所の種類)
            ((0.25, 0.25, 1.0), (0.25, 0.25, 0.0), geometry.FEATURE_FACE),
            ((0.25, 0.5, -2.0), (0.25, 0.5, 0.0), geometry.FEATURE_FACE),
            ((-1.0, -1.0, 0.5), self.A, geometry.FEATURE_VERTEX_0),
            ((2.0, -1.0, 0.0), self.B, geometry.FEATURE_VERTEX_1),
            ((-1.0, 2.0, 3.0), self.C, geometry.FEATURE_VERTEX_2),
            ((0.5, -1.0, 2.0), (0.5, 0.0, 0.0), geometry.FEATURE_EDGE_01),
            ((1.0, 1.0, 0.0), (0.5, 0.5, 0.0), geometry.FEATURE_EDGE_12),
            ((0.75, 0.75, -1.0), (0.5, 0.5, 0.0), geometry.FEATURE_EDGE_12),
            ((-2.0, 0.5, -1.0), (0.0, 0.5, 0.0), geometry.FEATURE_EDGE_20),
        ]
        result = self.single([q for q, _, _ in cases])
        for row, (query, point, feature) in enumerate(cases):
            with self.subTest(query=query):
                np.testing.assert_allclose(result.points[row], point, atol=1e-15)
                self.assertAlmostEqual(result.distances[row], math.dist(query, point), places=14)
                self.assertEqual(int(result.features[row]), feature)
                self.assertEqual(int(result.triangles[row]), 0)

    def test_vertex_results_are_exact_vertex_coordinates(self):
        vertices = np.array([[0.1, 0.2, 0.3], [1.7, 0.2, 0.3], [0.1, 1.9, 0.3]])
        result = closest_points([[3.0, -1.0, 0.3], [0.1, 5.0, 0.3]], vertices, [[0, 1, 2]])
        np.testing.assert_array_equal(result.points, vertices[[1, 2]])
        np.testing.assert_array_equal(result.features, [geometry.FEATURE_VERTEX_1, geometry.FEATURE_VERTEX_2])

    def test_types_and_shapes(self):
        result = self.single([[0.2, 0.2, 1.0], [5, 5, 5]])
        self.assertEqual(result.points.shape, (2, 3))
        self.assertEqual(result.points.dtype, np.float64)
        self.assertEqual(result.distances.shape, (2,))
        self.assertEqual(result.triangles.dtype, np.int64)
        self.assertEqual(result.features.dtype, np.int8)

    def test_collinear_triangle_is_treated_as_segments(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        queries = [(1.5, 1.0, 0.0), (-1.0, 0.0, 0.0), (3.0, 0.0, 0.0), (1.0, 0.0, 2.0), (0.5, 0.0, -1.0)]
        with NoFloatErrors():
            result = closest_points(queries, vertices, [[0, 1, 2]])
        np.testing.assert_allclose(result.points, [[1.5, 0, 0], [0, 0, 0], [2, 0, 0], [1, 0, 0], [0.5, 0, 0]],
                                   atol=1e-15)
        np.testing.assert_allclose(result.distances, [1.0, 1.0, 1.0, 2.0, 1.0])
        self.assertFalse(np.isnan(result.points).any())
        np.testing.assert_array_equal(result.features[:3], [geometry.FEATURE_EDGE_12, geometry.FEATURE_VERTEX_0,
                                                            geometry.FEATURE_VERTEX_2])
        # (1, 0, 2) の近くは頂点 v1（v0-v1 の端）
        self.assertEqual(int(result.features[3]), geometry.FEATURE_VERTEX_1)

    def test_collapsed_triangles(self):
        # 2つの頂点が同じ（辺の長さ 0）、3つとも同じ（点）
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=float)
        queries = [(0.5, 2.0, 0.0), (-3.0, 0.0, 0.0), (1.0, 1.0, 3.0)]
        with NoFloatErrors():
            pair = closest_points(queries, vertices, [[0, 0, 1]])
            point = closest_points(queries, vertices, [[2, 2, 2]])
        np.testing.assert_allclose(pair.points, [[0.5, 0, 0], [0, 0, 0], [1, 0, 0]], atol=1e-15)
        np.testing.assert_allclose(pair.distances, [2.0, 3.0, math.sqrt(10.0)])
        np.testing.assert_array_equal(point.points, np.repeat([[1.0, 1.0, 1.0]], 3, axis=0))
        np.testing.assert_allclose(point.distances, [math.sqrt(0.25 + 1 + 1), math.sqrt(16 + 1 + 1), 2.0])
        np.testing.assert_array_equal(point.features, [geometry.FEATURE_VERTEX_0] * 3)

    def test_sliver_triangle_has_no_nan(self):
        # 高さがとても小さい三角形でも、NaN にならず、素朴な計算と同じ距離
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1e-14, 0]], dtype=float)
        queries = np.array([[0.5, 0.3, 0.2], [0.5, -0.3, 0.0], [2.0, 0.0, 0.0], [0.5, 0.0, 1.0]])
        with NoFloatErrors():
            result = closest_points(queries, vertices, [[0, 1, 2]])
        for row, query in enumerate(queries):
            expected, _ = naive_point_triangle(query, *vertices)
            self.assertAlmostEqual(result.distances[row], expected, places=12)

    def test_thin_triangle_face_region_is_accurate(self):
        # 細いが「線」とはみなさない三角形（高さ = 辺の 1e-5〜3e-8 倍。float32 で丸めた、ほぼ一直線の頂点で起きる）。
        # 面の上の点の距離は、面からの高さ。重心座標を |ab × ac|^2 で割って求めると、引き算の誤差が
        # 1 / 高さ^2 で大きくなり、点が三角形の外へ飛ぶ（高さ 1e-7 で約 0.6mm ずれた）
        rng = np.random.default_rng(21)
        for height in (1e-5, 1e-6, 1e-7, 3e-8):
            rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
            flat = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.4, height, 0.0]])
            vertices = flat @ rotation.T + [0.3, -0.2, 0.7]
            bary = np.array([[0.2, 0.3], [0.5, 0.25], [0.7, 0.1], [0.33, 0.33], [0.05, 0.9]])
            offsets = np.array([0.5, -0.25, 0.01, 1e-4, -0.03])
            base = vertices[0] + bary[:, :1] * (vertices[1] - vertices[0]) + bary[:, 1:] * (vertices[2] - vertices[0])
            queries = base + offsets[:, None] * rotation[:, 2]
            result = closest_points(queries, vertices, [[0, 1, 2]])
            for row, query in enumerate(queries):
                with self.subTest(height=height, row=row):
                    expected, point = naive_point_triangle(query, *vertices)
                    self.assertAlmostEqual(result.distances[row], expected, delta=1e-9)
                    self.assertLess(float(np.linalg.norm(result.points[row] - point)), 1e-6)
                    self.assertAlmostEqual(float(np.linalg.norm(query - result.points[row])), result.distances[row],
                                           delta=1e-12)

    def test_ties_choose_lowest_triangle_index(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [5, 5, 5], [6, 5, 5], [5, 6, 5]], dtype=float)
        # 同じ三角形が2つ：番号 1 と 2。遠い三角形 0 を先に置く
        triangles = [[4, 5, 6], [0, 1, 2], [0, 1, 2]]
        for chunk in (1, 2, 3, 1000):  # 小分けの境目をまたいでも、先の番号が残る
            result = closest_points([[0.7, 0.2, 1.0]], vertices, triangles, chunk_size=chunk)
            self.assertEqual(int(result.triangles[0]), 1)
        # 頂点 0 を共有する2つの三角形の、頂点の領域：どちらも同じ距離
        square = [[0, 1, 2], [0, 2, 3]]
        for order in (square, square[::-1]):
            result = closest_points([[-1.0, -1.0, 0.5]], vertices, order)
            self.assertEqual(int(result.triangles[0]), 0)
            np.testing.assert_array_equal(result.points[0], [0, 0, 0])

    def test_matches_naive_on_random_soup(self):
        rng = np.random.default_rng(12345)
        vertices = rng.normal(size=(40, 3))
        triangles = rng.integers(0, 40, size=(30, 3))
        # つぶれた三角形もまぜる
        vertices[5] = vertices[4]
        vertices[6] = 0.5 * (vertices[7] + vertices[8])
        triangles[:4] = [[4, 5, 9], [6, 7, 8], [10, 10, 11], [12, 12, 12]]
        queries = rng.normal(scale=1.5, size=(300, 3))
        result = closest_points(queries, vertices, triangles, chunk_size=97)
        for row, query in enumerate(queries):
            per_triangle = [naive_point_triangle(query, *vertices[tri]) for tri in triangles]
            distances = np.array([d for d, _ in per_triangle])
            best = int(np.argmin(distances))
            self.assertAlmostEqual(result.distances[row], distances[best], places=12)
            # いちばん近い点は、選んだ三角形の上にあり、その距離はちょうど distances
            self.assertAlmostEqual(float(np.linalg.norm(query - result.points[row])), distances[best], places=12)
            self.assertAlmostEqual(distances[int(result.triangles[row])], distances[best], places=12)
            if np.sort(distances)[1] - distances[best] > 1e-9:
                self.assertEqual(int(result.triangles[row]), best)
                np.testing.assert_allclose(result.points[row], per_triangle[best][1], atol=1e-12)

    def test_feature_matches_point(self):
        # 場所の種類と、いちばん近い点の座標が合っているか（頂点なら頂点、辺なら辺の上）
        rng = np.random.default_rng(7)
        vertices, triangles = shapes.icosphere(radius=0.3, subdivisions=1)
        queries = rng.normal(scale=0.5, size=(500, 3))
        result = closest_points(queries, vertices, triangles)
        for row in range(len(queries)):
            tri = vertices[triangles[result.triangles[row]]]
            feature = int(result.features[row])
            point = result.points[row]
            if feature >= geometry.FEATURE_VERTEX_0:
                np.testing.assert_array_equal(point, tri[feature - geometry.FEATURE_VERTEX_0])
            elif feature >= geometry.FEATURE_EDGE_01:
                start = tri[feature - 1]
                end = tri[feature % 3]
                along = np.cross(end - start, point - start)
                self.assertLess(float(np.linalg.norm(along)), 1e-12)
            else:
                normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                offset = (queries[row] - point) / max(result.distances[row], 1e-300)
                self.assertAlmostEqual(abs(float(offset @ normal)) / float(np.linalg.norm(normal)), 1.0, places=9)

    def test_chunk_size_does_not_change_results(self):
        rng = np.random.default_rng(3)
        vertices, triangles = shapes.icosphere(radius=0.4, subdivisions=2)
        queries = rng.normal(scale=0.5, size=(57, 3))
        reference = closest_points(queries, vertices, triangles)
        for chunk in (1, 2, 7, 320, 10 ** 7):
            result = closest_points(queries, vertices, triangles, chunk_size=chunk)
            np.testing.assert_array_equal(result.points, reference.points)
            np.testing.assert_array_equal(result.distances, reference.distances)
            np.testing.assert_array_equal(result.triangles, reference.triangles)
            np.testing.assert_array_equal(result.features, reference.features)

    def test_memory_is_bounded(self):
        rng = np.random.default_rng(0)
        vertices = rng.normal(size=(3000, 3))
        triangles = rng.integers(0, 3000, size=(1000, 3))
        queries = rng.normal(size=(1000, 3))  # 100 万組。小分けしなければ数百 MB になる
        tracemalloc.start()
        try:
            closest_points(queries, vertices, triangles)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 80 * 1024 * 1024)

    def test_empty_points(self):
        vertices, triangles = shapes.box()
        result = closest_points(np.zeros((0, 3)), vertices, triangles)
        self.assertEqual(result.points.shape, (0, 3))
        self.assertEqual(result.distances.shape, (0,))
        self.assertEqual(result.triangles.dtype, np.int64)
        self.assertEqual(result.features.dtype, np.int8)
        self.assertEqual(closest_points([], vertices, triangles).points.shape, (0, 3))
        self.assertEqual(signed_distance([], vertices, triangles).shape, (0,))

    def test_invalid_inputs(self):
        vertices, triangles = shapes.box()
        good = [[0.0, 0.0, 0.0]]
        bad_calls = [
            lambda: closest_points(good, vertices, np.zeros((0, 3), dtype=np.int64)),  # 三角形が無い
            lambda: closest_points([[0.0, 0.0]], vertices, triangles),  # 形が違う
            lambda: closest_points([[np.nan, 0.0, 0.0]], vertices, triangles),
            lambda: closest_points(good, np.vstack([vertices[:-1], [[np.inf, 0, 0]]]), triangles),
            lambda: closest_points(good, vertices, triangles + 1),  # 番号が頂点の数を超える
            lambda: closest_points(good, vertices, triangles - 1),  # 負の番号
            lambda: closest_points(good, vertices, triangles.astype(float)),  # 小数
            lambda: closest_points(good, vertices, triangles.astype(bool)),  # 真偽
            lambda: closest_points([[True, False, True]], vertices, triangles),
            lambda: closest_points(good, vertices, triangles, chunk_size=0),
            lambda: closest_points(good, vertices, triangles, chunk_size=True),  # 真偽は整数として受け付けない
            lambda: closest_points(good, vertices, triangles, chunk_size=2.0),
            lambda: closest_points(good, vertices, triangles[:, :2]),
            lambda: closest_points([["a", "b", "c"]], vertices, triangles),
        ]
        for number, call in enumerate(bad_calls):
            with self.subTest(number=number), self.assertRaises(ValueError):
                call()

    def test_ragged_input_gives_japanese_message(self):
        # 長さのそろわない並びも、日本語の説明と引数の名前を付けた ValueError にする（numpy の英語のままにしない）
        vertices, triangles = shapes.box()
        ragged = [[0.0, 0.0, 0.0], [1.0, 1.0]]
        calls = [
            ("points", lambda: closest_points(ragged, vertices, triangles)),
            ("triangles", lambda: closest_points([[0.0, 0.0, 0.0]], vertices, [[0, 1, 2], [0, 1]])),
            ("depths", lambda: summarize_penetration([[0.0], [0.1, 0.2]])),
            ("visible", lambda: summarize_penetration([0.0, 0.1], visible=[[True], [False, True]])),
        ]
        for name, call in calls:
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as caught:
                    call()
                self.assertIn(name, str(caught.exception))
                self.assertRegex(str(caught.exception), "[぀-ヿ]")


# MARK: - 符号つきの距離


class SignedDistanceTest(unittest.TestCase):
    def test_box_exact_values(self):
        vertices, triangles = shapes.box(size=(2.0, 2.0, 2.0))
        cases = [
            ((0.0, 0.0, 0.0), -1.0),  # 中心：半分の大きさ
            ((0.5, 0.0, 0.0), -0.5),
            ((0.0, -0.9, 0.2), -0.1),
            ((3.0, 0.0, 0.0), 2.0),  # 面の外
            ((0.3, 0.0, -1.25), 0.25),
            ((2.0, 2.0, 0.0), math.sqrt(2.0)),  # 辺の外
            ((0.5, -3.0, 4.0), math.sqrt(4.0 + 9.0)),
            ((2.0, 2.0, 2.0), math.sqrt(3.0)),  # 角の外
            ((-2.0, 1.5, -4.0), math.sqrt(1.0 + 0.25 + 9.0)),
            ((0.999, 0.999, 0.0), -0.001),  # 辺のすぐ内側
            ((0.999, 0.999, 0.999), -0.001),  # 角のすぐ内側
            ((-0.9995, 0.0, -0.999), -0.0005),
            ((1.001, 1.001, 0.0), 0.001 * math.sqrt(2.0)),  # 辺のすぐ外側
            ((1.0, 0.3, 0.2), 0.0),  # 面の上
            ((1.0, 1.0, 1.0), 0.0),  # 角の上
        ]
        result = signed_distance([p for p, _ in cases], vertices, triangles)
        for row, (point, expected) in enumerate(cases):
            with self.subTest(point=point):
                self.assertAlmostEqual(result[row], expected, places=12)

    def test_box_offset_center(self):
        vertices, triangles = shapes.box(size=(0.2, 0.4, 0.6), center=(1.0, -2.0, 0.5))
        points = [[1.0, -2.0, 0.5], [1.0, -2.0, 0.75], [1.0, -2.0, 1.0], [1.3, -2.0, 0.5]]
        result = signed_distance(points, vertices, triangles)
        np.testing.assert_allclose(result, [-0.1, -0.05, 0.2, 0.2], atol=1e-12)

    def assert_signs_match(self, vertices, triangles, points, inside):
        result = signed_distance(points, vertices, triangles)
        unsigned = closest_points(points, vertices, triangles).distances
        np.testing.assert_allclose(np.abs(result), unsigned, rtol=0, atol=0)
        decided = unsigned > 1e-9
        self.assertGreater(decided.sum(), 0.9 * len(points))
        np.testing.assert_array_equal(result[decided] < 0, inside[decided])

    def test_sharp_edge_and_vertex_of_wedge(self):
        vertices, triangles = prism(WEDGE_POLYGON, WEDGE_CAPS)
        self.assertGreater(shapes.signed_volume(vertices, triangles), 0)
        # とがった辺（x=4, y=0）の外。片方の面の法線だけで決めると、どちらかの点で符号をまちがえる
        points = np.array([[4.5, 0.05, 0.5], [4.5, -0.5, 0.5], [4.5, 0.05, -0.5], [4.5, -0.5, -0.5]])
        expected = [math.sqrt(0.25 + 0.0025), math.sqrt(0.5), math.sqrt(0.5 + 0.0025), math.sqrt(0.75)]
        closest = closest_points(points, vertices, triangles)
        self.assertTrue(np.all((closest.features[:2] >= 1) & (closest.features[:2] <= 3)), "辺がいちばん近い")
        self.assertTrue(np.all(closest.features[2:] >= 4), "頂点がいちばん近い")
        rng = np.random.default_rng(5)
        for _ in range(6):
            # 三角形の並びと、三角形の中の頂点の順（回すだけ）を変えても、答えは同じ
            order = rng.permutation(len(triangles))
            shuffled = np.array([np.roll(tri, int(rng.integers(3))) for tri in triangles[order]])
            np.testing.assert_allclose(signed_distance(points, vertices, shuffled), expected, atol=1e-12)

    def test_random_points_around_sharp_features(self):
        vertices, triangles = prism(WEDGE_POLYGON, WEDGE_CAPS)
        rng = np.random.default_rng(11)
        tip = rng.uniform([3.5, -0.5, -0.5], [4.5, 0.5, 0.5], size=(3000, 3))
        edge = rng.uniform([3.0, -0.6, 0.2], [4.6, 0.6, 0.8], size=(2000, 3))
        top = rng.uniform([3.5, -0.5, 0.5], [4.5, 0.5, 1.5], size=(2000, 3))
        points = np.vstack([tip, edge, top])
        self.assert_signs_match(vertices, triangles, points, inside_wedge(points))

    def test_split_vertices_keep_edges_connected(self):
        # 頂点が三角形ごとに分かれていても、同じ座標をまとめてつながりを調べるので、とがった辺の符号が正しい
        vertices, triangles = split_vertices(*prism(WEDGE_POLYGON, WEDGE_CAPS))
        rng = np.random.default_rng(12)
        points = rng.uniform([3.5, -0.5, -0.5], [4.5, 0.5, 1.5], size=(3000, 3))
        self.assert_signs_match(vertices, triangles, points, inside_wedge(points))

    def test_concave_l_prism(self):
        vertices, triangles = prism(L_POLYGON, L_CAPS)
        self.assertAlmostEqual(shapes.signed_volume(vertices, triangles), 3.0)
        rng = np.random.default_rng(13)
        points = np.vstack([rng.uniform(-0.5, 2.5, size=(3000, 3)),
                            rng.uniform([0.7, 0.7, -0.3], [1.3, 1.3, 1.3], size=(2000, 3))])  # へこんだ辺のまわり
        self.assert_signs_match(vertices, triangles, points, inside_l(points))
        # へこんだ辺 (1, 1, z) のすぐ内側：いちばん近いのはその辺
        result = signed_distance([[0.9, 0.9, 0.5]], vertices, triangles)
        self.assertAlmostEqual(result[0], -math.sqrt(0.02), places=12)

    def test_icosphere_within_exact_bounds(self):
        radius = 0.5
        vertices, triangles = shapes.icosphere(radius=radius, subdivisions=3)
        normals = triangle_normals(vertices, triangles)
        inner = float(np.min(np.einsum("ij,ij->i", normals, vertices[triangles[:, 0]])))
        self.assertGreater(inner, 0.98 * radius)
        rng = np.random.default_rng(17)
        directions = rng.normal(size=(400, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        lengths = rng.uniform(0.0, 1.0, size=400)
        points = directions * lengths[:, None]
        result = signed_distance(points, vertices, triangles)
        # 多面体は、半径 inner の球を含み、半径 radius の球に含まれる：|p| - radius <= 距離 <= |p| - inner
        self.assertTrue(np.all(result >= lengths - radius - 1e-12))
        self.assertTrue(np.all(result <= lengths - inner + 1e-12))
        np.testing.assert_allclose(result, lengths - radius, atol=radius - inner + 1e-12)

    def test_open_grid_sign_above_and_below(self):
        vertices, triangles = shapes.grid(4, 4)
        result = signed_distance([[0.1, 0.1, 0.5], [0.1, -0.2, -0.25]], vertices, triangles)
        np.testing.assert_allclose(result, [0.5, -0.25], atol=1e-15)

    def test_undecidable_sign_is_nan(self):
        # 表と裏の三角形が重なった板：辺の擬似法線は 0 になり、符号を決められない
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        triangles = [[0, 1, 2], [0, 2, 1]]
        result = signed_distance([[0.5, -1.0, 0.0], [0.2, 0.2, 1.0], [0.2, 0.2, 0.0]], vertices, triangles)
        self.assertTrue(math.isnan(result[0]))
        self.assertEqual(result[1], 1.0)  # 面の中：番号の小さい三角形（上向き）で決まる
        self.assertEqual(result[2], 0.0)  # 面の上は 0
        self.assertTrue(math.isnan(penetration_depths([[0.5, -1.0, 0.0]], vertices, triangles)[0]))
        self.assertTrue(math.isnan(gaps([[0.5, -1.0, 0.0]], vertices, triangles)[0]))


class PenetrationAndGapTest(unittest.TestCase):
    def test_clamps(self):
        vertices, triangles = shapes.box(size=(2.0, 2.0, 2.0))
        points = [[0.25, 0.0, 0.0], [1.5, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -0.9]]
        np.testing.assert_allclose(penetration_depths(points, vertices, triangles), [0.75, 0.0, 0.0, 0.1], atol=1e-15)
        np.testing.assert_allclose(gaps(points, vertices, triangles), [0.0, 0.5, 0.0, 0.0], atol=1e-15)
        self.assertFalse(np.signbit(penetration_depths(points, vertices, triangles)).any())  # -0.0 にならない


# MARK: - 信号


class SummarizePenetrationTest(unittest.TestCase):
    Y, R = thresholds.PENETRATION_YELLOW, thresholds.PENETRATION_RED

    def test_hidden_penetration_is_not_counted(self):
        depths = np.array([0.0, 2 * self.R, 0.7 * self.R, 0.0, 3 * self.R])
        visible = np.array([True, False, True, True, False])
        result = summarize_penetration(depths, visible=visible)
        self.assertEqual(result.name, "penetration_depth")
        self.assertEqual(result.unit, "m")
        self.assertAlmostEqual(result.value, 0.7 * self.R)
        self.assertEqual(result.signal, Signal.YELLOW)
        self.assertEqual(result.details["penetrating_visible_count"], 1)
        self.assertEqual(result.details["penetrating_hidden_count"], 2)
        self.assertAlmostEqual(result.details["max_hidden_depth"], 3 * self.R)
        np.testing.assert_array_equal(result.details["worst_visible_indices"], [2])
        np.testing.assert_array_equal(result.details["worst_hidden_indices"], [4, 1])
        # 見える印が無ければ、全部の点を数える
        self.assertEqual(summarize_penetration(depths).signal, Signal.RED)
        self.assertAlmostEqual(summarize_penetration(depths).value, 3 * self.R)

    def test_thresholds_are_inclusive(self):
        self.assertEqual(summarize_penetration([0.0, 0.0]).signal, Signal.GREEN)
        self.assertEqual(summarize_penetration([np.nextafter(self.Y, 0)]).signal, Signal.GREEN)
        self.assertEqual(summarize_penetration([self.Y]).signal, Signal.YELLOW)
        self.assertEqual(summarize_penetration([self.R]).signal, Signal.RED)

    def test_worst_order_and_ties(self):
        depths = [0.1, 0.3, 0.3, 0.0, 0.2]
        result = summarize_penetration(depths)
        np.testing.assert_array_equal(result.details["worst_visible_indices"], [1, 2, 4, 0])
        self.assertEqual(result.details["worst_visible_indices"].dtype, np.int64)
        self.assertEqual(result.details["over_red_visible_count"], 4)

    def test_gray_cases(self):
        self.assertEqual(summarize_penetration([]).signal, Signal.GRAY)
        all_hidden = summarize_penetration([0.0, 0.1], visible=np.array([False, False]))
        self.assertEqual(all_hidden.signal, Signal.GRAY)
        self.assertTrue(all_hidden.reason)
        self.assertEqual(summarize_penetration([np.nan, np.nan]).signal, Signal.GRAY)

    def test_unmeasured_visible_point_is_never_green(self):
        result = summarize_penetration([0.0, np.nan])
        self.assertEqual(result.signal, Signal.GRAY)
        self.assertEqual(result.value, 0.0)
        self.assertTrue(result.reason)
        # 赤は灰より重いので、赤のまま
        self.assertEqual(summarize_penetration([self.R, np.nan]).signal, Signal.RED)
        # 見えない点が測れなくても、見える点の信号は変わらない
        hidden_nan = summarize_penetration([0.0, np.nan], visible=np.array([True, False]))
        self.assertEqual(hidden_nan.signal, Signal.GREEN)
        self.assertEqual(hidden_nan.details["unmeasured_hidden_count"], 1)

    def test_invalid(self):
        for call in (lambda: summarize_penetration([-0.001]),
                     lambda: summarize_penetration([np.inf]),
                     lambda: summarize_penetration([[0.0]]),
                     lambda: summarize_penetration([0.0, 0.1], visible=np.array([True])),
                     lambda: summarize_penetration([0.0, 0.1], visible=np.array([1, 0]))):
            with self.assertRaises(ValueError):
                call()


class SummarizeGapTest(unittest.TestCase):
    LOW, HIGH = thresholds.CLOTH_GAP_MIN, thresholds.CLOTH_GAP_MAX

    def test_count_outside_range(self):
        values = [0.5 * self.LOW, 2 * self.LOW, 1.25 * self.HIGH, self.LOW, self.HIGH, -0.001]
        result = summarize_gap(values)
        self.assertEqual(result.name, "cloth_gap")
        self.assertEqual(result.unit, "count")
        self.assertEqual(result.value, 3.0)
        self.assertEqual(result.signal, Signal.YELLOW)
        self.assertEqual(result.details["too_close_count"], 2)
        self.assertEqual(result.details["too_far_count"], 1)
        np.testing.assert_array_equal(result.details["too_close_indices"], [0, 5])
        np.testing.assert_array_equal(result.details["too_far_indices"], [2])
        self.assertAlmostEqual(result.details["min"], -0.001)
        self.assertAlmostEqual(result.details["max"], 1.25 * self.HIGH)
        self.assertAlmostEqual(result.details["median"], float(np.median(values)))

    def test_all_inside_is_green(self):
        result = summarize_gap([self.LOW, self.HIGH, 0.5 * (self.LOW + self.HIGH)])
        self.assertEqual(result.signal, Signal.GREEN)
        self.assertEqual(result.value, 0.0)
        self.assertEqual(summarize_gap([np.nextafter(self.LOW, 0)]).signal, Signal.YELLOW)
        self.assertEqual(summarize_gap([np.nextafter(self.HIGH, 1)]).signal, Signal.YELLOW)

    def test_gray(self):
        self.assertEqual(summarize_gap([]).signal, Signal.GRAY)
        self.assertEqual(summarize_gap([np.nan]).signal, Signal.GRAY)
        partly = summarize_gap([2 * self.LOW, np.nan])
        self.assertEqual(partly.signal, Signal.GRAY)
        self.assertEqual(partly.details["unmeasured_count"], 1)
        with self.assertRaises(ValueError):
            summarize_gap([np.inf])


if __name__ == "__main__":
    unittest.main()

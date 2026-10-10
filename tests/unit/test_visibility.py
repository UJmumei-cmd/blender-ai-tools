"""A4 見える・見えない（奥行きの画像）を確かめる。

答えが分かっている形（箱・板・球・格子）で、塗るピクセル・奥行き・隠れ方・カメラの向きの名前を確かめる。
1ピクセルずつ光線を飛ばす、ゆっくりした別の計算（reference_render）とも比べる。
"""

import dataclasses
import itertools
import math
import unittest
from unittest import mock

import numpy as np

from blender_ai_tools import thresholds
from blender_ai_tools.measure import visibility
from blender_ai_tools.measure.results import Signal
from blender_ai_tools.measure.visibility import (
    Camera,
    DepthBuffer,
    look_at,
    project,
    rasterize,
    render_views,
    standard_views,
    summarize_visible_points,
    visible_fraction,
    visible_from_any,
    visible_points,
    visible_triangle_pixels,
    visible_triangles,
)
from tests.shapes import box, grid, icosphere

EPSILON = thresholds.VISIBILITY_DEPTH_EPSILON


# MARK: - テスト用の道具


def quad(corners) -> tuple[np.ndarray, np.ndarray]:
    """4つの角（順に回る）から、三角形2つの板を作る"""
    return np.asarray(corners, dtype=np.float64), np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)


def merge(*meshes) -> tuple[np.ndarray, np.ndarray]:
    """いくつかのメッシュを1つにする（三角形の番号は、渡した順に続く）"""
    vertices, triangles, offset = [], [], 0
    for v, t in meshes:
        vertices.append(np.asarray(v, dtype=np.float64))
        triangles.append(np.asarray(t, dtype=np.int64) + offset)
        offset += len(v)
    return np.concatenate(vertices), np.concatenate(triangles)


def front_ortho(width=200, height=100, ortho_scale=2.0, distance=5.0, **kwargs) -> Camera:
    """前（-Y）から +Y を見る平行のカメラ。画像の中心が世界の原点"""
    return look_at((0.0, -distance, 0.0), (0.0, 0.0, 0.0), kind="orthographic", width=width, height=height,
                   ortho_scale=ortho_scale, **kwargs)


def fit_size(camera: Camera) -> int:
    if camera.fov_axis == "horizontal":
        return camera.width
    if camera.fov_axis == "vertical":
        return camera.height
    return max(camera.width, camera.height)


def pixel_rays(camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    """ピクセルの中心を通る光線（カメラの座標）。origins (H, W, 3)、directions (H, W, 3)。
    向きの z は -1 にそろえるので、光線の長さ t がそのまま奥行きになる。
    （テストの側で、画角の決まりを別に書いて確かめる）"""
    u = np.arange(camera.width) + 0.5
    v = np.arange(camera.height) + 0.5
    uu, vv = np.meshgrid(u, v)
    du = uu - camera.width / 2.0
    dv = -(vv - camera.height / 2.0)
    if camera.kind == "perspective":
        focal = (fit_size(camera) / 2.0) / math.tan(math.radians(camera.fov_deg) / 2.0)
        directions = np.stack([du / focal, dv / focal, -np.ones_like(du)], axis=-1)
        origins = np.zeros_like(directions)
    else:
        scale = fit_size(camera) / camera.ortho_scale
        origins = np.stack([du / scale, dv / scale, np.zeros_like(du)], axis=-1)
        directions = np.broadcast_to(np.array([0.0, 0.0, -1.0]), origins.shape)
    return origins, directions


def reference_render(vertices, triangles, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    """1ピクセルずつ光線を飛ばして、いちばん近い三角形を探す（Möller–Trumbore。テスト用のゆっくりした答え）"""
    origins, directions = pixel_rays(camera)
    local = (np.asarray(vertices, dtype=np.float64) - camera.position) @ camera.rotation
    best_depth = np.full((camera.height, camera.width), np.inf)
    best_id = np.full((camera.height, camera.width), -1, dtype=np.int64)
    for index, (i0, i1, i2) in enumerate(np.asarray(triangles)):
        p0, p1, p2 = local[i0], local[i1], local[i2]
        edge1, edge2 = p1 - p0, p2 - p0
        pvec = np.cross(directions, edge2)
        det = pvec @ edge1
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tvec = origins - p0
        u = np.einsum("hwc,hwc->hw", tvec, pvec) * inv
        qvec = np.cross(tvec, edge1)
        v = np.einsum("hwc,hwc->hw", directions, qvec) * inv
        t = (qvec @ edge2) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= camera.near) & (t <= camera.far)
        closer = hit & (t < best_depth)
        best_depth[closer] = t[closer]
        best_id[closer] = index
    return best_depth, best_id


def plane_depths(camera: Camera, point, normal) -> np.ndarray:
    """ピクセルの中心を通る光線と、平面（point を通り normal に垂直）が交わる所の奥行き (H, W)"""
    origins, directions = pixel_rays(camera)
    local_point = (np.asarray(point, dtype=np.float64) - camera.position) @ camera.rotation
    local_normal = np.asarray(normal, dtype=np.float64) @ camera.rotation
    return ((local_point - origins) @ local_normal) / (directions @ local_normal)


def slanted_plane(point, normal, half_size) -> tuple[np.ndarray, np.ndarray]:
    """point を中心にした、normal に垂直な正方形の板"""
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    t1 = np.cross(normal, [0.0, 0.0, 1.0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    p = np.asarray(point, dtype=np.float64)
    h = half_size
    return quad([p - h * t1 - h * t2, p + h * t1 - h * t2, p + h * t1 + h * t2, p - h * t1 + h * t2])


def mean_column(buffer: DepthBuffer, triangle_range: range) -> float:
    mask = np.isin(buffer.triangle, list(triangle_range))
    return float(np.nonzero(mask)[1].mean() + 0.5)


def mean_row(buffer: DepthBuffer, triangle_range: range) -> float:
    mask = np.isin(buffer.triangle, list(triangle_range))
    return float(np.nonzero(mask)[0].mean() + 0.5)


def pixel_count(buffer: DepthBuffer, triangle_range: range) -> int:
    return int(np.isin(buffer.triangle, list(triangle_range)).sum())


# MARK: - カメラ


class CameraTest(unittest.TestCase):
    def test_defaults_come_from_thresholds(self):
        camera = look_at((0, -3, 0), (0, 0, 0))
        self.assertEqual(camera.kind, "perspective")
        self.assertEqual(camera.width, thresholds.VISIBILITY_RESOLUTION)
        self.assertEqual(camera.height, thresholds.VISIBILITY_RESOLUTION)
        self.assertEqual(camera.fov_deg, thresholds.APP_CAMERA_FOV_DEG)
        self.assertEqual(camera.fov_axis, "auto")

    def test_look_at_axes(self):
        camera = look_at((1.0, -4.0, 2.0), (0.0, 0.0, 1.0))
        expected_forward = np.array([-1.0, 4.0, -1.0]) / math.sqrt(18.0)
        np.testing.assert_allclose(camera.forward, expected_forward, atol=1e-12)
        np.testing.assert_allclose(camera.rotation.T @ camera.rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(camera.rotation), 1.0, places=12)
        self.assertAlmostEqual(float(camera.right[2]), 0.0, places=12)  # 右は水平
        self.assertGreater(camera.up[2], 0.0)  # 画像の上は、世界の上の側
        np.testing.assert_allclose(np.cross(camera.right, camera.up), -camera.forward, atol=1e-12)

    def test_look_at_front_view_axes(self):
        camera = look_at((0, -5, 0), (0, 0, 0))
        np.testing.assert_allclose(camera.right, [1, 0, 0], atol=1e-15)  # キャラの左（+X）は画像の右
        np.testing.assert_allclose(camera.up, [0, 0, 1], atol=1e-15)
        np.testing.assert_allclose(camera.forward, [0, 1, 0], atol=1e-15)

    def test_look_at_straight_down_uses_y_as_up(self):
        camera = look_at((0, 0, 5), (0, 0, 0), width=40, height=40)
        np.testing.assert_allclose(camera.up, [0, 1, 0], atol=1e-15)
        np.testing.assert_allclose(camera.right, [1, 0, 0], atol=1e-15)
        xy, _, _ = project([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], camera)
        self.assertLess(xy[0, 1], 20.0)  # +Y は画像の上
        self.assertGreater(xy[1, 0], 20.0)  # +X は画像の右

    def test_look_at_up_parallel_fallback_is_deterministic(self):
        a = look_at((0, -3, 0), (0, 0, 0), up=(0, 1, 0))  # 見る向き +Y と up が平行 → 次の候補 +Z
        np.testing.assert_allclose(a.up, [0, 0, 1], atol=1e-15)
        b = look_at((0, -3, 0), (0, 0, 0), up=(0, 1, 0))
        np.testing.assert_array_equal(a.rotation, b.rotation)
        c = look_at((0, 0, -5), (0, 0, 0))  # 真下から見上げる
        np.testing.assert_allclose(c.up, [0, 1, 0], atol=1e-15)
        np.testing.assert_allclose(c.rotation.T @ c.rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(c.rotation), 1.0, places=12)

    def test_look_at_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            look_at((1, 2, 3), (1, 2, 3))
        with self.assertRaises(ValueError):
            look_at((0, -3, 0), (0, 0, 0), up=(0, 0, 0))
        with self.assertRaises(ValueError):
            look_at((0, -3, math.nan), (0, 0, 0))
        with self.assertRaises(ValueError):
            look_at((0, -3), (0, 0, 0))

    def test_camera_validation(self):
        good = np.eye(3)
        Camera((0, 0, 0), good, width=np.int64(10), height=np.int32(8))  # numpy の整数は使える
        bad_cases = [
            dict(rotation=np.diag([1.0, 1.0, -1.0])),  # 鏡（左手系）
            dict(rotation=2.0 * np.eye(3)),  # 大きさを含む
            dict(rotation=np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])),  # 直交しない
            dict(rotation=np.eye(2)),
            dict(rotation=np.full((3, 3), math.nan)),
            dict(position=(0.0, math.nan, 0.0)),
            dict(position=(0.0, 0.0)),
            dict(kind="fisheye"),
            dict(fov_axis="diagonal"),
            dict(width=True),  # bool は整数として使えない
            dict(width=0),
            dict(height=-1),
            dict(width=10.5),
            dict(fov_deg=0.0),
            dict(fov_deg=180.0),
            dict(fov_deg=math.nan),
            dict(fov_deg=True),
            dict(kind="orthographic"),  # ortho_scale が無い
            dict(kind="orthographic", ortho_scale=0.0),
            dict(kind="orthographic", ortho_scale=math.inf),
            dict(near=0.0),
            dict(near=-1.0),
            dict(near=math.nan),
            dict(near=1.0, far=1.0),
            dict(far=math.nan),
        ]
        for case in bad_cases:
            arguments = dict(position=(0.0, 0.0, 0.0), rotation=good)
            arguments.update(case)
            with self.subTest(case=case), self.assertRaises(ValueError):
                Camera(**arguments)

    def test_camera_accepts_float32_rotation(self):
        """Blender の行列（float32）から作った回転も使える"""
        rotation = look_at((0.3, -4.0, 1.7), (0.0, 0.0, 1.0)).rotation.astype(np.float32)
        camera = Camera((0.3, -4.0, 1.7), rotation)
        self.assertEqual(camera.rotation.dtype, np.float64)

    def test_camera_is_immutable(self):
        position = np.array([0.0, -3.0, 0.0])
        camera = look_at(position, (0, 0, 0))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            camera.width = 3
        with self.assertRaises(ValueError):
            camera.position[0] = 1.0
        with self.assertRaises(ValueError):
            camera.rotation[0, 0] = 1.0
        position[1] = 100.0  # 元の並びを変えても、カメラは変わらない
        self.assertEqual(camera.position[1], -3.0)


class StandardViewsTest(unittest.TestCase):
    def test_positions_and_directions(self):
        center = np.array([0.1, 0.2, 1.0])
        views = standard_views(center, 3.0, width=32, height=32)
        self.assertEqual(tuple(views), visibility.STANDARD_VIEW_NAMES)
        s = math.sqrt(0.5) * 3.0
        expected = {
            "front": (0.0, -3.0, 0.0), "back": (0.0, 3.0, 0.0), "left": (3.0, 0.0, 0.0), "right": (-3.0, 0.0, 0.0),
            "front_left": (s, -s, 0.0), "front_right": (-s, -s, 0.0),
        }
        for name, offset in expected.items():
            with self.subTest(view=name):
                camera = views[name]
                np.testing.assert_allclose(camera.position, center + offset, atol=1e-12)
                np.testing.assert_allclose(camera.forward, -np.asarray(offset) / 3.0, atol=1e-12)
                np.testing.assert_allclose(camera.up, [0, 0, 1], atol=1e-12)

    def test_passes_settings(self):
        views = standard_views((0, 0, 0), 2.0, width=30, height=20, kind="orthographic", ortho_scale=1.5,
                               fov_axis="vertical", near=0.5, far=9.0)
        for camera in views.values():
            self.assertEqual((camera.width, camera.height, camera.kind), (30, 20, "orthographic"))
            self.assertEqual((camera.ortho_scale, camera.fov_axis, camera.near, camera.far),
                             (1.5, "vertical", 0.5, 9.0))

    def test_rejects_bad_distance(self):
        for distance in (0.0, -1.0, math.nan, math.inf, True):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                standard_views((0, 0, 0), distance)

    def test_left_right_front_naming(self):
        """キャラの左は +X、前は -Y。名前は、キャラから見た向きにする"""
        body = box((0.4, 0.2, 1.0))
        left_marker = box((0.1, 0.1, 0.1), center=(0.4, 0.0, 0.0))  # キャラの左
        front_marker = box((0.1, 0.1, 0.1), center=(0.0, -0.5, 0.3))  # キャラの前、少し上
        vertices, triangles = merge(body, left_marker, front_marker)
        left_ids, front_ids = range(12, 24), range(24, 36)
        renders = render_views(vertices, triangles, standard_views((0, 0, 0), 3.0, width=64, height=64, fov_deg=40))
        buffers = {name: buffer for name, (_, buffer) in renders.items()}

        # 前から：キャラの左は画像の右。前の印は、上の側（画像の上半分）
        self.assertGreater(mean_column(buffers["front"], left_ids), 40.0)
        self.assertLess(mean_row(buffers["front"], front_ids), 32.0)
        # うしろから：キャラの左は画像の左。前の印は体に隠れる
        self.assertLess(mean_column(buffers["back"], left_ids), 24.0)
        self.assertEqual(pixel_count(buffers["back"], front_ids), 0)
        # 左（+X）から：左の印が見え、キャラの前は画像の左
        self.assertGreater(pixel_count(buffers["left"], left_ids), 0)
        self.assertLess(mean_column(buffers["left"], front_ids), 24.0)
        # 右（-X）から：左の印は体に隠れ、キャラの前は画像の右
        self.assertEqual(pixel_count(buffers["right"], left_ids), 0)
        self.assertGreater(mean_column(buffers["right"], front_ids), 40.0)
        # 前と左のななめから：左の印も前の印も見える
        self.assertGreater(pixel_count(buffers["front_left"], left_ids), 0)
        self.assertGreater(pixel_count(buffers["front_left"], front_ids), 0)
        # どの横の向きでも、上（+Z）は画像の上
        for name in visibility.STANDARD_VIEW_NAMES:
            if pixel_count(buffers[name], front_ids):
                with self.subTest(view=name):
                    self.assertLess(mean_row(buffers[name], front_ids), 32.0)


# MARK: - 写す


class ProjectTest(unittest.TestCase):
    def test_perspective_known_answers(self):
        camera = look_at((0, -2, 0), (0, 0, 0), width=100, height=50, fov_deg=90.0)  # 横に合わせる：焦点 50px
        points = [[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 2, 0], [-1, 2, -1]]
        xy, depth, in_front = project(points, camera)
        np.testing.assert_allclose(xy, [[50, 25], [75, 25], [50, 0], [50, 25], [50 - 12.5, 25 + 12.5]], atol=1e-12)
        np.testing.assert_allclose(depth, [2, 2, 2, 4, 4], atol=1e-12)
        self.assertTrue(in_front.all())

    def test_perspective_fov_axis(self):
        point = [[1.0, 0.0, 0.0]]
        expected = {"horizontal": 75.0, "auto": 75.0, "vertical": 62.5}  # 焦点 50px か 25px
        for axis, x in expected.items():
            camera = look_at((0, -2, 0), (0, 0, 0), width=100, height=50, fov_deg=90.0, fov_axis=axis)
            with self.subTest(axis=axis):
                self.assertAlmostEqual(project(point, camera)[0][0, 0], x, places=12)
        portrait = look_at((0, -2, 0), (0, 0, 0), width=50, height=100, fov_deg=90.0)  # auto は長い縦に合わせる
        self.assertAlmostEqual(project(point, portrait)[0][0, 0], 25.0 + 50.0 * 0.5, places=12)

    def test_orthographic_known_answers(self):
        camera = front_ortho(width=100, height=50, ortho_scale=4.0)  # 横 4m：25px/m
        xy, depth, in_front = project([[1, 0, 0], [0, 0, 1], [0, 3, -1]], camera)
        np.testing.assert_allclose(xy, [[75, 25], [50, 0], [50, 50]], atol=1e-12)
        np.testing.assert_allclose(depth, [5, 5, 8], atol=1e-12)
        self.assertTrue(in_front.all())
        vertical = front_ortho(width=100, height=50, ortho_scale=4.0, fov_axis="vertical")  # 縦 4m：12.5px/m
        self.assertAlmostEqual(project([[1, 0, 0]], vertical)[0][0, 0], 62.5, places=12)

    def test_behind_and_clip_range(self):
        camera = look_at((0, -2, 0), (0, 0, 0), width=100, height=50, fov_deg=90.0, near=0.01, far=10.0)
        points = [[0, -3, 0], [0.1, -1.995, 0], [0, 8, 0], [0, 8.5, 0]]
        xy, depth, in_front = project(points, camera)
        np.testing.assert_allclose(depth, [-1, 0.005, 10, 10.5], atol=1e-12)
        np.testing.assert_array_equal(in_front, [False, False, True, False])
        self.assertTrue(np.isnan(xy[0]).all())  # カメラのうしろは写せない
        self.assertTrue(np.isfinite(xy[1:]).all())

    def test_orthographic_behind_is_not_in_front(self):
        camera = front_ortho()
        xy, depth, in_front = project([[0, -6, 0]], camera)
        self.assertEqual(depth[0], -1.0)
        self.assertFalse(in_front[0])
        self.assertTrue(np.isfinite(xy).all())

    def test_empty_and_bad_points(self):
        camera = front_ortho()
        xy, depth, in_front = project(np.zeros((0, 3)), camera)
        self.assertEqual((xy.shape, depth.shape, in_front.shape), ((0, 2), (0,), (0,)))
        self.assertEqual(project([], camera)[0].shape, (0, 2))
        for bad in ([1.0, 2.0, 3.0], [[1.0, 2.0]], [[math.nan, 0.0, 0.0]], [[True, False, True]], [["a", "b", "c"]]):
            with self.subTest(points=bad), self.assertRaises(ValueError):
                project(bad, camera)
        # 空でも形が違う並び（列が無い・列が3でない）は、点が無いとはみなさない
        for bad in (np.zeros((5, 0)), np.zeros((0, 2)), [[]], np.zeros((0, 3, 1))):
            with self.subTest(shape=np.shape(bad)), self.assertRaises(ValueError):
                project(bad, camera)
        with self.assertRaises(ValueError):
            project([[0.0, 0.0, 0.0]], "camera")


# MARK: - 描く


class RasterizeBoxTest(unittest.TestCase):
    def setUp(self):
        # 中心を原点からずらした箱（左右・上下が逆になると、答えがずれる）
        self.vertices, self.triangles = box((0.6, 0.2, 0.4), center=(0.25, 0.0, 0.1))
        self.camera = front_ortho()  # 200x100、100px/m、原点が (100, 50)

    def test_orthographic_coverage_and_depth(self):
        buffer = rasterize(self.vertices, self.triangles, self.camera)
        self.assertEqual(buffer.depth.shape, (100, 200))
        expected = np.zeros((100, 200), dtype=bool)
        expected[20:60, 95:155] = True  # x は -0.05〜0.55m、z は -0.1〜0.3m
        np.testing.assert_array_equal(buffer.covered, expected)
        np.testing.assert_allclose(buffer.depth[expected], 4.9, rtol=0, atol=1e-12)  # 前の面は y=-0.1
        self.assertTrue(np.isposinf(buffer.depth[~expected]).all())
        self.assertEqual(set(np.unique(buffer.triangle[expected])), {4, 5})  # 前（-Y）の面の三角形
        counts = visible_triangle_pixels(buffer, len(self.triangles))
        self.assertEqual(int(counts.sum()), 40 * 60)
        self.assertEqual(int(counts[4] + counts[5]), 40 * 60)

    def test_back_faces_are_drawn_and_occlude(self):
        """前の面を取ると、裏向きのうしろの面が写り、さらにうしろの物を隠す"""
        without_front = np.delete(self.triangles, [4, 5], axis=0)
        hidden = box((0.1, 0.1, 0.1), center=(0.25, 1.0, 0.1))
        vertices, triangles = merge((self.vertices, without_front), hidden)
        buffer = rasterize(vertices, triangles, self.camera)
        expected = np.zeros((100, 200), dtype=bool)
        expected[20:60, 95:155] = True
        np.testing.assert_array_equal(buffer.covered, expected)
        np.testing.assert_allclose(buffer.depth[expected], 5.1, atol=1e-12)  # うしろの面（y=+0.1）
        self.assertEqual(pixel_count(buffer, range(10, 22)), 0)

    def test_small_box_behind_is_hidden_and_in_front_is_visible(self):
        big = box((1.0, 1.0, 1.0))
        behind = box((0.2, 0.2, 0.2), center=(0.1, 2.0, 0.05))
        ahead = box((0.2, 0.2, 0.2), center=(0.1, -2.0, 0.05))
        for camera in (look_at((0, -6, 0), (0, 0, 0), width=80, height=80, fov_deg=40), front_ortho(80, 80, 3.0)):
            with self.subTest(kind=camera.kind):
                vertices, triangles = merge(big, behind)
                counts = visible_triangle_pixels(rasterize(vertices, triangles, camera), len(triangles))
                self.assertGreater(int(counts[:12].sum()), 0)
                self.assertEqual(int(counts[12:].sum()), 0)
                vertices, triangles = merge(big, ahead)
                buffer = rasterize(vertices, triangles, camera)
                counts = visible_triangle_pixels(buffer, len(triangles))
                alone = rasterize(*ahead, camera)
                self.assertEqual(int(counts[12:].sum()), int(alone.covered.sum()))  # 小さい箱は全部見える
                np.testing.assert_array_equal(buffer.depth[alone.covered], alone.depth[alone.covered])
                self.assertEqual(int(counts[:12].sum()), int(rasterize(*big, camera).covered.sum())
                                 - int(alone.covered.sum()))  # 大きい箱は、小さい箱の分だけ隠れる

    def test_winding_does_not_matter(self):
        flipped = self.triangles[:, ::-1]
        a = rasterize(self.vertices, self.triangles, self.camera)
        b = rasterize(self.vertices, flipped, self.camera)
        np.testing.assert_array_equal(a.depth, b.depth)
        np.testing.assert_array_equal(a.covered, b.covered)

    def test_deterministic(self):
        vertices, triangles = icosphere(1.0, 2)
        camera = look_at((0.3, -3.0, 0.5), (0, 0, 0), width=48, height=40)
        a = rasterize(vertices, triangles, camera)
        b = rasterize(vertices, triangles, camera)
        np.testing.assert_array_equal(a.depth, b.depth)
        np.testing.assert_array_equal(a.triangle, b.triangle)


class PerspectiveTest(unittest.TestCase):
    def test_twice_as_far_is_half_as_large(self):
        camera = look_at((0, 0, 0), (0, 1, 0), width=200, height=200, fov_deg=60.0)
        focal = 100.0 / math.tan(math.radians(30.0))
        sizes = {}
        for distance in (2.0, 4.0):
            plate = quad([(-0.5, distance, -0.5), (0.5, distance, -0.5), (0.5, distance, 0.5), (-0.5, distance, 0.5)])
            buffer = rasterize(*plate, camera)
            columns = int(buffer.covered.any(axis=0).sum())
            rows = int(buffer.covered.any(axis=1).sum())
            with self.subTest(distance=distance):
                self.assertLessEqual(abs(columns - focal / distance), 1.0)
                self.assertLessEqual(abs(rows - focal / distance), 1.0)
                np.testing.assert_allclose(buffer.depth[buffer.covered], distance, atol=1e-12)
            sizes[distance] = columns
        self.assertAlmostEqual(sizes[4.0] / sizes[2.0], 0.5, delta=0.03)

    def test_slanted_plane_depth_is_perspective_correct(self):
        camera = look_at((0.0, -3.0, 0.2), (0.1, 0.0, 0.0), width=80, height=60, fov_deg=50.0)
        point, normal = (0.0, 0.5, 0.0), (0.6, -1.0, 0.5)
        buffer = rasterize(*slanted_plane(point, normal, 4.0), camera)
        expected = plane_depths(camera, point, normal)
        covered = buffer.covered
        self.assertGreater(int(covered.sum()), 2000)
        depths = buffer.depth[covered]
        self.assertGreater(depths.max() / depths.min(), 1.3)  # 奥行きが大きく変わる（まっすぐ混ぜると違いが出る）
        np.testing.assert_allclose(depths, expected[covered], rtol=1e-6)
        # まっすぐ（画像の上で）奥行きを混ぜたら、この確かめに落ちることも確かめる
        vertices, triangles = slanted_plane(point, normal, 4.0)
        xy, corner_depth, _ = project(vertices, camera)
        wrong = []
        for row, column in zip(*np.nonzero(covered)):
            p = np.array([column + 0.5, row + 0.5])
            for t in triangles:
                a, b, c = xy[t]
                matrix = np.column_stack([b - a, c - a])
                l1, l2 = np.linalg.solve(matrix, p - a)
                if l1 >= 0 and l2 >= 0 and l1 + l2 <= 1:
                    d = corner_depth[t]
                    wrong.append((1 - l1 - l2) * d[0] + l1 * d[1] + l2 * d[2])
                    break
        wrong = np.array(wrong)
        self.assertGreater(np.max(np.abs(wrong - depths) / depths), 1e-3)

    def test_slanted_plane_orthographic(self):
        camera = look_at((0.0, -3.0, 0.2), (0.1, 0.0, 0.0), kind="orthographic", ortho_scale=3.0, width=60, height=60)
        point, normal = (0.0, 0.5, 0.0), (0.6, -1.0, 0.5)
        buffer = rasterize(*slanted_plane(point, normal, 4.0), camera)
        expected = plane_depths(camera, point, normal)
        self.assertEqual(int(buffer.covered.sum()), 60 * 60)
        np.testing.assert_allclose(buffer.depth, expected, rtol=1e-9)


class ReferenceComparisonTest(unittest.TestCase):
    """乱数で作った三角形（near の面を横切る物を含む）を、1ピクセルずつ光線を飛ばす答えと比べる"""

    def check(self, camera, seed):
        rng = np.random.default_rng(seed)
        count = 14
        centers = rng.uniform([-1.0, -3.6, -1.0], [1.0, 1.5, 1.0], size=(count, 1, 3))
        vertices = (centers + rng.normal(scale=0.8, size=(count, 3, 3))).reshape(-1, 3)
        triangles = np.arange(3 * count, dtype=np.int64).reshape(count, 3)
        buffer = rasterize(vertices, triangles, camera)
        depth, owner = reference_render(vertices, triangles, camera)
        corner_depth = -((vertices - camera.position) @ camera.rotation)[:, 2][triangles]
        crossing = int(((corner_depth < camera.near).any(axis=1) & (corner_depth >= camera.near).any(axis=1)).sum())
        self.assertGreater(crossing, 0)  # near の面を横切る三角形がある
        self.assertGreater(int((owner >= 0).sum()), 200)
        np.testing.assert_array_equal(buffer.triangle, owner)
        covered = owner >= 0
        np.testing.assert_allclose(buffer.depth[covered], depth[covered], rtol=1e-9)
        self.assertTrue((buffer.depth[covered] >= camera.near).all())

    def test_perspective(self):
        camera = look_at((0.0, -3.0, 0.0), (0.0, 0.0, 0.0), width=64, height=48, fov_deg=60.0, near=0.1, far=6.0)
        for seed in range(6):
            with self.subTest(seed=seed):
                self.check(camera, seed)

    def test_orthographic(self):
        camera = look_at((0.0, -3.0, 0.0), (0.0, 0.0, 0.0), kind="orthographic", ortho_scale=4.0, width=48,
                         height=64, near=0.1, far=6.0)
        for seed in range(6):
            with self.subTest(seed=seed):
                self.check(camera, seed)

    def test_tilted_camera(self):
        camera = look_at((1.2, -2.5, 0.8), (0.0, -0.5, 0.0), width=40, height=40, fov_deg=75.0, near=0.2)
        for seed in range(3):
            with self.subTest(seed=seed):
                self.check(camera, seed)


class NearPlaneTest(unittest.TestCase):
    def setUp(self):
        # 原点から +Y を見る。焦点 32px（画角 90 度、64x64）
        self.camera = look_at((0, 0, 0), (0, 1, 0), width=64, height=64, fov_deg=90.0, near=0.1)

    def test_floor_through_camera_plane(self):
        """カメラの下を通って、うしろまで続く床。前の部分だけが、地平線より下に写る"""
        floor = quad([(-10, -10, -1), (10, -10, -1), (10, 10, -1), (-10, 10, -1)])
        buffer = rasterize(*floor, self.camera)
        expected = np.zeros((64, 64), dtype=bool)
        expected[35:, :] = True  # 行 j の床までの奥行きは 32/(j+0.5-32)。奥の端（y=10）より手前の行だけ
        np.testing.assert_array_equal(buffer.covered, expected)
        rows = np.arange(64) + 0.5
        want = np.broadcast_to((32.0 / (rows - 32.0))[:, None], (64, 64))
        np.testing.assert_allclose(buffer.depth[expected], want[expected], rtol=1e-9)
        self.assertTrue((buffer.depth[expected] >= self.camera.near).all())

    def test_wall_through_camera_plane(self):
        wall = quad([(0.5, -10, -10), (0.5, 10, -10), (0.5, 10, 10), (0.5, -10, 10)])
        buffer = rasterize(*wall, self.camera)
        expected = np.zeros((64, 64), dtype=bool)
        expected[:, 34:] = True  # 列 i の壁までの奥行きは 16/(i+0.5-32)。奥の端（y=10）より手前の列だけ
        np.testing.assert_array_equal(buffer.covered, expected)
        columns = np.arange(64) + 0.5
        want = np.broadcast_to(16.0 / (columns - 32.0), (64, 64))
        np.testing.assert_allclose(buffer.depth[expected], want[expected], rtol=1e-9)

    def test_triangle_around_the_camera(self):
        """near の面を横切り、角がカメラのうしろにある大きな三角形：前の部分だけが描かれる"""
        # 面はカメラの約 0.5m 上を通る。ピクセルの中心が辺にちょうど乗らないように、少しずらした数にする
        # （ちょうど乗るときの決まりは CoverageRuleTest で確かめる）
        vertices = np.array([[-5.03, -5.0, -0.5], [4.97, -4.9, -0.47], [0.013, 5.0, 1.5]])
        buffer = rasterize(vertices, [[0, 1, 2]], self.camera)
        depth, owner = reference_render(vertices, [[0, 1, 2]], self.camera)
        np.testing.assert_array_equal(buffer.triangle, owner)
        self.assertGreater(int(buffer.covered.sum()), 100)
        np.testing.assert_allclose(buffer.depth[buffer.covered], depth[buffer.covered], rtol=1e-9)
        self.assertTrue((buffer.depth[buffer.covered] >= self.camera.near).all())

    def test_orthographic_slanted_quad_is_cut_at_near(self):
        camera = look_at((0, 0, 0), (0, 1, 0), kind="orthographic", ortho_scale=8.0, width=40, height=40, near=0.13)
        plate = quad([(-5, -3, -5), (5, -3, -5), (5, 7, 5), (-5, 7, 5)])  # y = z + 2
        buffer = rasterize(*plate, camera)
        expected = np.zeros((40, 40), dtype=bool)
        expected[:29, :] = True  # 行 j の z は (20-(j+0.5))/5。奥行き z+2 が near 以上の行だけ
        np.testing.assert_array_equal(buffer.covered, expected)
        rows = np.arange(40) + 0.5
        want = np.broadcast_to(((20.0 - rows) / 5.0 + 2.0)[:, None], (40, 40))
        np.testing.assert_allclose(buffer.depth[expected], want[expected], rtol=1e-9)

    def test_grid_crossing_near_is_watertight(self):
        """near の面を横切る格子：切った所の点が、となりの三角形とビット単位で同じになり、すき間も重なりも無い"""
        vertices, triangles = grid(6, 6, size=(3.0, 3.0), z=-0.4)
        vertices = vertices + np.array([0.0, 0.9, 0.0])  # y は -0.6〜2.4（カメラのうしろから前まで）
        camera = look_at((0, 0, 0), (0, 1, 0), width=48, height=48, fov_deg=100.0, near=0.25)
        whole = rasterize(vertices, triangles, camera)
        single_total = 0
        union = np.zeros(whole.covered.shape, dtype=bool)
        for index in range(len(triangles)):
            alone = rasterize(vertices, triangles[index:index + 1], camera).covered
            single_total += int(alone.sum())
            union |= alone
        self.assertGreater(int(whole.covered.sum()), 300)
        self.assertEqual(single_total, int(whole.covered.sum()))
        np.testing.assert_array_equal(union, whole.covered)

    def test_near_crossing_does_not_depend_on_edge_direction(self):
        rng = np.random.default_rng(7)
        for _ in range(200):
            a = rng.normal(size=3) + [0.0, 0.0, 0.5]
            b = rng.normal(size=3) + [0.0, 0.0, -1.5]
            if (-a[2] >= 0.1) == (-b[2] >= 0.1):
                continue
            forward = visibility._near_crossing(a, b, 0.1)
            backward = visibility._near_crossing(b, a, 0.1)
            np.testing.assert_array_equal(forward, backward)
            self.assertEqual(forward[2], -0.1)

    def test_nothing_behind_the_camera(self):
        behind = box((1.0, 1.0, 1.0), center=(0.0, -3.0, 0.0))
        self.assertFalse(rasterize(*behind, self.camera).covered.any())
        too_close = quad([(-1, 0.05, -1), (1, 0.05, -1), (1, 0.05, 1), (-1, 0.05, 1)])  # near より手前
        self.assertFalse(rasterize(*too_close, self.camera).covered.any())

    def test_far_plane(self):
        camera = look_at((0, 0, 0), (0, 1, 0), width=64, height=64, fov_deg=90.0, near=0.1, far=5.0)
        far_plate = quad([(-1, 6, -1), (1, 6, -1), (1, 6, 1), (-1, 6, 1)])
        self.assertFalse(rasterize(*far_plate, camera).covered.any())
        floor = quad([(-10, 0.5, -1), (10, 0.5, -1), (10, 10, -1), (-10, 10, -1)])
        buffer = rasterize(*floor, camera)
        expected = np.zeros((64, 64), dtype=bool)
        expected[38:, :] = True  # 奥行き 32/(j+0.5-32) が far（5m）以下の行（j >= 38）
        np.testing.assert_array_equal(buffer.covered, expected)
        self.assertLessEqual(float(buffer.depth[buffer.covered].max()), 5.0)


class CoverageRuleTest(unittest.TestCase):
    def test_grid_is_watertight_with_top_left_rule(self):
        """ピクセルの中心が辺や角の上にちょうど乗る格子：すき間も重なりも無く、上と左の辺だけを含む"""
        vertices, triangles = grid(4, 4, size=(2.0, 2.0))
        # 0.0625 ずらすと、格子の線がピクセルの中心をちょうど通る（2進数で割り切れる数だけを使う）
        camera = look_at((0.0625, 0.0625, 5.0), (0.0625, 0.0625, 0.0), kind="orthographic", ortho_scale=4.0,
                         width=32, height=32)
        buffer = rasterize(vertices, triangles, camera)
        expected = np.zeros((32, 32), dtype=bool)
        expected[8:24, 7:23] = True  # 画像の x は 7.5〜23.5、y は 8.5〜24.5（右と下の端は含まない）
        np.testing.assert_array_equal(buffer.covered, expected)
        single_total = 0
        union = np.zeros((32, 32), dtype=bool)
        for index in range(len(triangles)):
            alone = rasterize(vertices, triangles[index:index + 1], camera).covered
            single_total += int(alone.sum())
            union |= alone
        self.assertEqual(single_total, int(expected.sum()))  # 重なりが無い
        np.testing.assert_array_equal(union, expected)  # すき間が無い

    def test_rectangle_count_matches_projected_area(self):
        camera = front_ortho(width=64, height=64, ortho_scale=4.0)  # 16px/m
        # 辺がピクセルの中心をちょうど通る板（画像の x は 24.5〜44.5、y は 24.5〜36.5）。2進数で割り切れる数だけを使う
        plate = quad([(-0.46875, 0, -0.28125), (0.78125, 0, -0.28125), (0.78125, 0, 0.46875), (-0.46875, 0, 0.46875)])
        buffer = rasterize(*plate, camera)
        self.assertEqual(int(buffer.covered.sum()), int(1.25 * 16) * int(0.75 * 16))  # 面積 × 16px/m の2乗
        expected = np.zeros((64, 64), dtype=bool)
        expected[24:36, 24:44] = True
        np.testing.assert_array_equal(buffer.covered, expected)

    def test_sphere_has_no_cracks_and_front_faces_win(self):
        vertices, triangles = icosphere(1.0, 3)
        camera = front_ortho(width=100, height=100, ortho_scale=2.5)  # 40px/m
        buffer = rasterize(vertices, triangles, camera)
        centers = (np.arange(100) + 0.5 - 50.0) / 40.0
        x, z = np.meshgrid(centers, -centers)
        radius = np.hypot(x, z)
        self.assertTrue(buffer.covered[radius < 0.97].all())
        self.assertFalse(buffer.covered[radius > 1.0].any())
        a, b, c = (vertices[triangles[:, i]] for i in range(3))
        normals = np.cross(b - a, c - a)
        owners = buffer.triangle[buffer.covered]
        self.assertTrue((normals[owners, 1] < 0.0).all())  # 写るのは、カメラ（-Y）を向いた面だけ
        inner = radius < 0.9
        np.testing.assert_allclose(buffer.depth[inner], 5.0 - np.sqrt(1.0 - radius[inner] ** 2), atol=0.01)

    def test_chunking_does_not_change_the_answer(self):
        sphere = icosphere(0.8, 2, center=(0.1, 0.0, 0.0))
        floor = quad([(-3, -4, -0.5), (3, -4, -0.5), (3, 3, -0.5), (-3, 3, -0.5)])
        vertices, triangles = merge(sphere, floor)
        camera = look_at((0.2, -2.5, 0.3), (0, 0, 0), width=28, height=20, fov_deg=70.0, near=0.1)
        normal = rasterize(vertices, triangles, camera)
        with mock.patch.object(visibility, "_CHUNK_PIXELS", 5):
            small = rasterize(vertices, triangles, camera)
        np.testing.assert_array_equal(normal.depth, small.depth)
        np.testing.assert_array_equal(normal.triangle, small.triangle)
        self.assertGreater(int(normal.covered.sum()), 100)


class TieTest(unittest.TestCase):
    def test_duplicate_triangles_lower_index_wins(self):
        vertices = np.array([[-0.5, 0.0, -0.5], [0.5, 0.0, -0.5], [0.0, 0.0, 0.5]])
        camera = front_ortho(width=40, height=40, ortho_scale=2.0)
        for triangles in ([[0, 1, 2], [0, 1, 2], [2, 1, 0]], [[2, 1, 0], [0, 1, 2], [1, 2, 0]]):
            with self.subTest(triangles=triangles):
                buffer = rasterize(vertices, triangles, camera)
                self.assertGreater(int(buffer.covered.sum()), 0)
                self.assertEqual(set(np.unique(buffer.triangle[buffer.covered])), {0})

    def test_duplicate_slanted_triangles_lower_index_wins(self):
        """斜めの面で、角の順や向きだけが違う同じ三角形（裏返して複製した両面の服など）も、番号の小さい三角形が残る。
        角の順で奥行きの丸めが変わると、番号ではなく丸めで決まってしまう（その誤りを見つける）"""
        vertices = np.array([[-0.9, -0.4, -0.6], [0.8, 0.5, -0.3], [0.1, -0.2, 0.9]])
        orders = np.array(list(itertools.permutations(range(3))), dtype=np.int64)
        cameras = (look_at((0.3, -3.0, 0.4), (0, 0, 0), width=96, height=80, fov_deg=60.0),
                   look_at((0.3, -3.0, 0.4), (0, 0, 0), kind="orthographic", ortho_scale=3.0, width=96, height=80))
        for camera in cameras:
            for shift in range(len(orders)):
                triangles = np.roll(orders, -shift, axis=0)
                with self.subTest(kind=camera.kind, first=triangles[0].tolist()):
                    buffer = rasterize(vertices, triangles, camera)
                    self.assertGreater(int(buffer.covered.sum()), 500)
                    self.assertEqual(set(np.unique(buffer.triangle[buffer.covered])), {0})
            # 点も複製した三角形（番号は違うが、同じ場所の点。裏返しの向き）
            doubled = np.concatenate([vertices, vertices[::-1]])
            for triangles in ([[3, 4, 5], [0, 1, 2]], [[0, 1, 2], [3, 4, 5]]):
                with self.subTest(kind=camera.kind, doubled=triangles):
                    buffer = rasterize(doubled, triangles, camera)
                    self.assertEqual(set(np.unique(buffer.triangle[buffer.covered])), {0})

    def test_duplicate_triangles_crossing_near_lower_index_wins(self):
        """near の面で切る三角形でも、角の順だけが違う同じ三角形は、番号の小さい三角形が残る"""
        camera = look_at((0, 0, 0), (0, 1, 0), width=64, height=64, fov_deg=90.0, near=0.1)
        vertices = np.array([[-1.3, -0.7, -0.6], [1.1, 2.5, -0.45], [0.2, 1.8, 1.3]])  # 1つの角がカメラのうしろ
        orders = np.array(list(itertools.permutations(range(3))), dtype=np.int64)
        for shift in range(len(orders)):
            triangles = np.roll(orders, -shift, axis=0)
            with self.subTest(first=triangles[0].tolist()):
                buffer = rasterize(vertices, triangles, camera)
                self.assertGreater(int(buffer.covered.sum()), 1000)
                self.assertEqual(set(np.unique(buffer.triangle[buffer.covered])), {0})

    def test_ties_across_chunks(self):
        """候補を小分けにして、同じ奥行きの三角形が別の組に入っても、番号の小さい三角形が残る"""
        a = quad([(-0.5, 0, -0.5), (0.5, 0, -0.5), (0.5, 0, 0.5), (-0.5, 0, 0.5)])
        vertices, triangles = merge(a, a, a)
        camera = front_ortho(width=24, height=24, ortho_scale=2.0)
        with mock.patch.object(visibility, "_CHUNK_PIXELS", 5):
            buffer = rasterize(vertices, triangles, camera)
            reversed_order = rasterize(vertices, triangles[::-1], camera)
        self.assertGreater(int(buffer.covered.sum()), 50)
        self.assertTrue(np.isin(buffer.triangle[buffer.covered], [0, 1]).all())
        # 逆の順に並べると、元の 4・5 番（いまの 0・1 番）が残る
        self.assertTrue(np.isin(reversed_order.triangle[reversed_order.covered], [0, 1]).all())

    def test_coplanar_overlap_lower_index_wins(self):
        a = quad([(-0.5, 0, -0.5), (0.5, 0, -0.5), (0.5, 0, 0.5), (-0.5, 0, 0.5)])
        b = quad([(0.0, 0, -0.25), (1.0, 0, -0.25), (1.0, 0, 0.25), (0.0, 0, 0.25)])
        for camera in (front_ortho(width=64, height=64, ortho_scale=4.0),
                       look_at((0, -3, 0), (0, 0, 0), width=64, height=64, fov_deg=60)):
            with self.subTest(kind=camera.kind):
                first = rasterize(*merge(a, b), camera)
                second = rasterize(*merge(b, a), camera)
                overlap = rasterize(*a, camera).covered & rasterize(*b, camera).covered
                self.assertGreater(int(overlap.sum()), 20)
                self.assertTrue(np.isin(first.triangle[overlap], [0, 1]).all())  # a が先
                self.assertTrue(np.isin(second.triangle[overlap], [0, 1]).all())  # b が先
                np.testing.assert_array_equal(first.covered, second.covered)


class EmptyAndValidationTest(unittest.TestCase):
    def test_empty_mesh(self):
        camera = front_ortho(width=30, height=20)
        for vertices, triangles in ((np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)),
                                    (np.ones((4, 3)), []), ([], [])):
            buffer = rasterize(vertices, triangles, camera)
            self.assertEqual(buffer.depth.shape, (20, 30))
            self.assertTrue(np.isposinf(buffer.depth).all())
            self.assertTrue((buffer.triangle == -1).all())
            self.assertEqual(buffer.triangle.dtype, np.int64)
            self.assertEqual(visible_triangle_pixels(buffer, 0).shape, (0,))

    def test_triangle_outside_image_is_skipped(self):
        camera = front_ortho(width=30, height=20)
        far_right = quad([(5, 0, -0.1), (6, 0, -0.1), (6, 0, 0.1), (5, 0, 0.1)])
        self.assertFalse(rasterize(*far_right, camera).covered.any())

    def test_degenerate_triangles(self):
        camera = front_ortho(width=30, height=20)
        vertices = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        buffer = rasterize(vertices, [[0, 1, 2], [0, 3, 1], [0, 0, 0]], camera)  # 線と点
        self.assertFalse(buffer.covered.any())
        edge_on = quad([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)])  # 真横から見た板
        self.assertFalse(rasterize(*edge_on, camera).covered.any())

    def test_mesh_validation(self):
        camera = front_ortho(width=10, height=10)
        vertices = np.zeros((3, 3))
        bad = [
            (vertices, [[0, 1, 3]]),  # 番号が大きすぎる
            (vertices, [[0, -1, 2]]),
            (vertices, [[0.0, 1.0, 2.0]]),  # 小数
            (vertices, [[True, False, True]]),  # bool
            (vertices, [[0, 1]]),
            (vertices, [0, 1, 2]),
            (np.zeros((3, 2)), [[0, 1, 2]]),
            (np.array([[0, 0, 0], [1, 0, 0], [math.nan, 0, 0]]), [[0, 1, 2]]),
            (np.array([[0, 0, 0], [1, 0, 0], [math.inf, 0, 0]]), [[0, 1, 2]]),
            (vertices, np.zeros((2, 0), dtype=np.int64)),  # 空でも形が違う
            (np.zeros((3, 0)), []),
        ]
        for v, t in bad:
            with self.subTest(vertices=np.asarray(v).shape, triangles=t), self.assertRaises(ValueError):
                rasterize(v, t, camera)
        with self.assertRaises(ValueError):
            rasterize(vertices, [[0, 1, 2]], None)
        rasterize(vertices, np.array([[0, 1, 2]], dtype=np.uint32), camera)  # 符号なしの整数は使える

    def test_depth_buffer_validation(self):
        DepthBuffer(np.full((2, 3), np.inf), np.full((2, 3), -1))
        DepthBuffer(np.array([[1.0, np.inf]], dtype=np.float32), np.array([[0, -1]], dtype=np.int32))
        bad = [
            (np.full((2, 3), np.inf), np.full((3, 2), -1)),
            (np.full((2, 3), np.inf), np.full((2, 3), -1.0)),
            (np.full(3, np.inf), np.full(3, -1)),
            (np.array([[1.0, np.inf]]), np.array([[-1, -1]])),  # 奥行きがあるのに三角形が無い
            (np.array([[np.inf, np.inf]]), np.array([[0, -1]])),  # 三角形があるのに奥行きが無い
            (np.array([[math.nan, np.inf]]), np.array([[0, -1]])),
            (np.array([[-1.0, np.inf]]), np.array([[0, -1]])),
            (np.array([[1.0, np.inf]]), np.array([[-2, -1]])),
        ]
        for depth, triangle in bad:
            with self.subTest(depth=depth.tolist()), self.assertRaises(ValueError):
                DepthBuffer(depth, triangle)
        empty = DepthBuffer.empty(4, 5)
        self.assertEqual((empty.height, empty.width), (4, 5))
        self.assertFalse(empty.covered.any())

    def test_visible_triangle_pixels_validation(self):
        buffer = rasterize(*box(), front_ortho(width=20, height=20, ortho_scale=3.0))
        counts = visible_triangle_pixels(buffer, 12)
        self.assertEqual(counts.dtype, np.int64)
        self.assertEqual(int(counts.sum()), int(buffer.covered.sum()))
        self.assertEqual(visible_triangle_pixels(buffer, 20).shape, (20,))
        for n in (True, -1, 2.0, 5):  # 5 は、写った三角形の番号より小さい
            with self.subTest(n=n), self.assertRaises(ValueError):
                visible_triangle_pixels(buffer, n)
        with self.assertRaises(ValueError):
            visible_triangle_pixels("buffer", 12)
        mask = visible_triangles(buffer, 12)
        np.testing.assert_array_equal(mask, counts > 0)
        np.testing.assert_array_equal(visible_triangles(buffer, 12, min_pixels=int(counts.max())),
                                      counts == counts.max())
        for least in (0, True):
            with self.subTest(min_pixels=least), self.assertRaises(ValueError):
                visible_triangles(buffer, 12, min_pixels=least)


# MARK: - 見えるかどうか


class VisiblePointsTest(unittest.TestCase):
    def setUp(self):
        self.camera = front_ortho()
        self.buffer = rasterize(*box((0.6, 0.2, 0.4), center=(0.25, 0.0, 0.1)), self.camera)

    def test_points(self):
        points = np.array([
            [0.3, -0.1, 0.12],  # 前の面の上
            [0.3, -0.1 + 0.5 * EPSILON, 0.12],  # 面の少し奥（幅の中）
            [0.3, -0.1 + 2.0 * EPSILON, 0.12],  # 面の奥（幅の外）
            [0.3, 0.1, 0.12],  # うしろの面（前の面に隠れる）
            [0.3, -0.5, 0.12],  # 箱の手前
            [-0.5, 0.0, 0.0],  # 何も写っていないピクセル
            [5.0, 0.0, 0.0],  # 画像の外
            [0.3, -6.0, 0.12],  # カメラのうしろ
        ])
        np.testing.assert_array_equal(visible_points(points, self.buffer, self.camera),
                                      [True, True, False, False, True, True, False, False])
        exact = visible_points(points[:2], self.buffer, self.camera, epsilon=0.0)
        np.testing.assert_array_equal(exact, [True, False])  # 同じ奥行きは見える（<=）
        self.assertAlmostEqual(visible_fraction(points, self.buffer, self.camera), 4 / 8)

    def test_image_border(self):
        # 画像の枠：x は [0, 200)、y は [0, 100)
        points = np.array([[-1.0, 0, 0], [-1.0 - 1e-9, 0, 0], [0.99, 0, 0], [1.0, 0, 0], [0, 0, 0.5], [0, 0, -0.5]])
        np.testing.assert_array_equal(visible_points(points, self.buffer, self.camera),
                                      [True, False, True, False, True, False])

    def test_far_points_are_not_visible(self):
        camera = front_ortho(far=6.0)
        buffer = rasterize(np.zeros((0, 3)), [], camera)
        np.testing.assert_array_equal(visible_points([[0, 0.9, 0], [0, 1.1, 0]], buffer, camera), [True, False])

    def test_perspective_point_on_slanted_surface(self):
        """斜めの面の上の点：ピクセルの中心で測った奥行きとの差が、幅の中なら見える。光線の上で奥へ動かすと隠れる"""
        camera = look_at((0, -3, 0), (0, 0, 0), width=800, height=800, fov_deg=20)  # 1ピクセルは約 1.3mm
        point, normal = (0.0, 0.0, 0.0), (0.3, -1.0, 0.2)
        buffer = rasterize(*slanted_plane(point, normal, 2.0), camera)
        n = np.asarray(normal) / np.linalg.norm(normal)
        t1 = np.cross(n, [0, 0, 1])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        on_surface = np.array([0.05 * t1, -0.1 * t1 + 0.07 * t2, np.zeros(3), 0.2 * t2])
        self.assertTrue(visible_points(on_surface, buffer, camera).all())
        rays = on_surface - camera.position
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        behind = on_surface + 4.0 * EPSILON * rays  # 同じ光線の上で、面の 4mm 奥
        self.assertFalse(visible_points(behind, buffer, camera).any())
        ahead = on_surface - 4.0 * EPSILON * rays
        self.assertTrue(visible_points(ahead, buffer, camera).all())

    def coarse_slanted_case(self, camera):
        """粗い画像（1ピクセル数 cm）と、45度より傾いた面。面の上の点 (P, 3) を返す"""
        point, normal = (0.0, 0.0, 0.0), (1.0, -1.0, 0.3)
        plate = slanted_plane(point, normal, 2.0)
        n = np.asarray(normal) / np.linalg.norm(normal)
        t1 = np.cross(n, [0, 0, 1])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        a, b = np.meshgrid(np.linspace(-0.7, 0.7, 13), np.linspace(-0.7, 0.7, 11))
        on_surface = a.reshape(-1, 1) * t1 + b.reshape(-1, 1) * t2
        return plate, on_surface

    def test_mesh_option_measures_slanted_surface_exactly(self):
        for camera in (look_at((0, -3, 0), (0, 0, 0), width=64, height=64, fov_deg=70),
                       front_ortho(width=48, height=48, ortho_scale=3.0)):
            with self.subTest(kind=camera.kind):
                plate, on_surface = self.coarse_slanted_case(camera)
                buffer = rasterize(*plate, camera)
                xy, _, in_front = project(on_surface, camera)
                self.assertTrue(in_front.all() and (xy >= 0).all() and (xy < 48).all())
                # ピクセルの中心で比べると、面の上の点でも「隠れている」とまちがえる点がある（mesh を使う理由）
                self.assertFalse(visible_points(on_surface, buffer, camera).all())
                self.assertTrue(visible_points(on_surface, buffer, camera, mesh=plate).all())
                if camera.kind == "perspective":
                    rays = on_surface - camera.position
                    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
                else:
                    rays = np.broadcast_to(camera.forward, on_surface.shape)
                behind = on_surface + 2.0 * EPSILON * rays
                ahead = on_surface - 2.0 * EPSILON * rays
                self.assertFalse(visible_points(behind, buffer, camera, mesh=plate).any())
                self.assertTrue(visible_points(ahead, buffer, camera, mesh=plate).all())
                self.assertAlmostEqual(visible_fraction(on_surface, buffer, camera, mesh=plate), 1.0)

    def test_mesh_option_keeps_occlusion(self):
        camera = look_at((0, -3, 0), (0, 0, 0), width=64, height=64, fov_deg=70)
        plate, on_surface = self.coarse_slanted_case(camera)
        blocker = box((0.6, 0.1, 0.6), center=(0.0, -1.0, 0.0))
        vertices, triangles = merge(plate, blocker)
        buffer = rasterize(vertices, triangles, camera)
        visible = visible_points(on_surface, buffer, camera, mesh=(vertices, triangles))
        xy, _, _ = project(on_surface, camera)
        owner = buffer.triangle[np.floor(xy[:, 1]).astype(int), np.floor(xy[:, 0]).astype(int)]
        behind_blocker = owner >= 2
        self.assertGreater(int(behind_blocker.sum()), 5)
        self.assertFalse(visible[behind_blocker].any())  # 箱のうしろの点は隠れる
        self.assertTrue(visible[~behind_blocker].all())  # ほかの点は見える
        renders = {"front": (camera, buffer)}
        result = summarize_visible_points(on_surface, renders, mesh=(vertices, triangles))
        self.assertEqual(result.value, float((~behind_blocker).sum()))
        self.assertEqual(result.details["hidden_count"], int(behind_blocker.sum()))
        np.testing.assert_array_equal(visible_from_any(on_surface, renders, mesh=(vertices, triangles)), visible)

    def test_mesh_option_validation(self):
        vertices, triangles = box((0.6, 0.2, 0.4), center=(0.25, 0.0, 0.1))
        point = [[0.3, -0.2, 0.12]]
        with self.assertRaises(ValueError):
            visible_points(point, self.buffer, self.camera, mesh=(vertices, triangles[:3]))  # 三角形が足りない
        for bad in ((vertices,), vertices, (vertices, triangles.astype(float))):
            with self.subTest(mesh=type(bad).__name__), self.assertRaises(ValueError):
                visible_points(point, self.buffer, self.camera, mesh=bad)

    def test_empty_points_and_validation(self):
        self.assertEqual(visible_points(np.zeros((0, 3)), self.buffer, self.camera).shape, (0,))
        self.assertTrue(math.isnan(visible_fraction([], self.buffer, self.camera)))
        other = front_ortho(width=10, height=10)
        with self.assertRaises(ValueError):
            visible_points([[0, 0, 0]], self.buffer, other)  # 大きさの違うカメラ
        for epsilon in (-1.0, math.nan, math.inf, True):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                visible_points([[0, 0, 0]], self.buffer, self.camera, epsilon=epsilon)
        with self.assertRaises(ValueError):
            visible_points([[0, math.nan, 0]], self.buffer, self.camera)


class SummaryTest(unittest.TestCase):
    def setUp(self):
        self.vertices, self.triangles = box((0.4, 0.2, 1.0))
        self.views = standard_views((0, 0, 0), 3.0, width=48, height=48, fov_deg=40)
        self.renders = render_views(self.vertices, self.triangles, self.views)
        self.on_front = [0.05, -0.1, 0.2]  # 前の面の上（前から見える）
        self.inside = [0.0, 0.0, 0.0]  # 体の中（服の下のめり込みのような物。どこからも見えない）

    def test_visible_problem_is_red(self):
        result = summarize_visible_points([self.on_front, self.inside], self.renders)
        self.assertEqual(result.signal, Signal.RED)
        self.assertEqual(result.value, 1.0)
        self.assertEqual(result.unit, "count")
        np.testing.assert_array_equal(result.details["visible"], [True, False])
        self.assertEqual(result.details["hidden_count"], 1)
        self.assertGreater(result.details["per_view"]["front"], 0)
        self.assertEqual(result.details["per_view"]["back"], 0)
        np.testing.assert_array_equal(visible_from_any([self.on_front, self.inside], self.renders), [True, False])

    def test_hidden_problem_is_only_recorded(self):
        result = summarize_visible_points([self.inside], self.renders)
        self.assertEqual(result.signal, Signal.GREEN)
        self.assertEqual(result.value, 0.0)
        self.assertEqual(result.details["hidden_count"], 1)
        back_only = {"back": self.renders["back"]}
        self.assertEqual(summarize_visible_points([self.on_front], back_only).signal, Signal.GREEN)

    def test_signal_if_visible(self):
        result = summarize_visible_points([self.on_front], self.renders, signal_if_visible=Signal.YELLOW)
        self.assertEqual(result.signal, Signal.YELLOW)
        with self.assertRaises(ValueError):
            summarize_visible_points([self.on_front], self.renders, signal_if_visible="red")
        # 見える問題を緑や灰（測れなかった）にはできない
        for signal in (Signal.GREEN, Signal.GRAY):
            with self.subTest(signal=signal), self.assertRaises(ValueError):
                summarize_visible_points([self.on_front], self.renders, signal_if_visible=signal)

    def test_gray_outranks_yellow(self):
        """見える点の信号が黄でも、測れなかった点やカメラがあれば灰にする（赤 > 灰 > 黄 > 緑）"""
        outside = [0.0, 0.0, 100.0]  # どのカメラでも枠の外
        result = summarize_visible_points([self.on_front, outside], self.renders, signal_if_visible=Signal.YELLOW)
        self.assertEqual(result.signal, Signal.GRAY)
        self.assertEqual(result.value, 1.0)  # 見えた点の数は残す（少なくともこの数）
        self.assertIn("見える点が 1 個", result.reason)
        self.assertIn("枠の外", result.reason)
        away = look_at((0, -3, 0), (0, -6, 0), width=48, height=48)  # 何も写らないカメラ
        renders = dict(self.renders)
        renders["away"] = (away, rasterize(self.vertices, self.triangles, away))
        empty_view = summarize_visible_points([self.on_front], renders, signal_if_visible=Signal.YELLOW)
        self.assertEqual(empty_view.signal, Signal.GRAY)
        self.assertIn("away", empty_view.reason)
        # 赤は灰より重いので、赤のまま
        self.assertEqual(summarize_visible_points([self.on_front, outside], self.renders).signal, Signal.RED)

    def test_bad_points_shape_is_not_green(self):
        """形の違う空の並び（列が無いなど）を「点が無い」として緑にしない"""
        for bad in (np.zeros((4, 0)), np.zeros((0, 2))):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                summarize_visible_points(bad, self.renders)

    def test_renders_validation(self):
        camera, buffer = self.renders["front"]
        bad_cases = {
            "camera only": {"front": camera},
            "one item": {"front": (camera,)},
            "swapped": {"front": (buffer, camera)},
            "list": [(camera, buffer)],
            "other buffer": {"front": (camera, DepthBuffer.empty(3, 3))},
        }
        for label, renders in bad_cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    summarize_visible_points([self.on_front], renders)
                with self.assertRaises(ValueError):
                    visible_from_any([self.on_front], renders)

    def test_no_points_is_green(self):
        result = summarize_visible_points(np.zeros((0, 3)), self.renders)
        self.assertEqual(result.signal, Signal.GREEN)
        self.assertEqual(result.value, 0.0)

    def test_gray_when_cannot_judge(self):
        no_cameras = summarize_visible_points([self.on_front], {})
        self.assertEqual(no_cameras.signal, Signal.GRAY)
        self.assertIsNone(no_cameras.value)

        away = look_at((0, -3, 0), (0, -6, 0), width=48, height=48)  # 体の反対を見ている
        renders = dict(self.renders)
        renders["away"] = (away, rasterize(self.vertices, self.triangles, away))
        empty_view = summarize_visible_points([self.inside], renders)
        self.assertEqual(empty_view.signal, Signal.GRAY)
        self.assertIn("away", empty_view.reason)
        self.assertEqual(empty_view.details["empty_views"], ["away"])

        outside = [0.0, 0.0, 100.0]  # どのカメラでも枠の外（ずっと上）
        out = summarize_visible_points([self.inside, outside], self.renders)
        self.assertEqual(out.signal, Signal.GRAY)
        self.assertEqual(out.details["unmeasured_count"], 1)
        # 見える点があれば、灰の理由があっても赤（赤 > 灰）
        both = summarize_visible_points([self.on_front, outside], self.renders)
        self.assertEqual(both.signal, Signal.RED)
        self.assertEqual(both.value, 1.0)
        self.assertIn("枠の外", both.reason)


if __name__ == "__main__":
    unittest.main()

"""A4 見える・見えない：カメラから見た「奥行きの画像」（z バッファ）を、numpy だけで作る。

アプリのカメラから見える問題だけを数えるために使う（ロードマップ段階1の基準2：
「アプリのカメラから見える赤の出来事が0」）。服の下に隠れためり込みは、記録するだけにする。
Blender の側では、あとで GPU で描く版を作るかもしれない。この numpy の版は、そのときの「正しい答え」にする。

決まり：
- 向きは Blender のまま：Z が上、モデルは -Y を向く、キャラの左は +X。
- カメラの中の向きは Blender のカメラと同じ：カメラは自分の -Z の向きを見る。自分の +Y が上、+X が右。
- 画像の座標：ピクセル（列 i、行 j）は、x が [i, i+1)、y が [j, j+1) の四角。中心は (i+0.5, j+0.5)。
  y は下向き（画像と同じ。行 0 がいちばん上）。
- 奥行き（depth）：カメラの見る向きに測った長さ（カメラの -Z の向きの成分）。カメラの前が正。
  カメラから点までの、まっすぐの距離ではない。
- 長さはメートル、角度は度。
- 三角形は、表も裏も描く（裏向きの面も、うしろの物を隠す）。
- ピクセルの中心が三角形の中にあれば、そのピクセルを塗る。中心が辺の上にちょうど乗ったときは
  「左上の規則」で決める（三角形の上の辺と左の辺は含み、右の辺と下の辺は含まない）。
  となり合う三角形の境目のピクセルは、どちらか1つだけが塗る（すき間も重なりもできない）。
- 奥行きが同じときは、番号の小さい三角形を残す（いつも同じ答えになる）。
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import thresholds
from ..thresholds import CAMERA_FAR, CAMERA_NEAR
from .geometry import as_array
from .results import Measurement, Signal, worst

__all__ = [
    "CAMERA_FAR",
    "CAMERA_NEAR",
    "STANDARD_VIEW_NAMES",
    "VISIBLE_POINTS_NAME",
    "Camera",
    "DepthBuffer",
    "look_at",
    "project",
    "rasterize",
    "render_views",
    "standard_views",
    "summarize_visible_points",
    "visible_fraction",
    "visible_from_any",
    "visible_points",
    "visible_triangle_pixels",
    "visible_triangles",
]

# MARK: - 決まった数字

# カメラの手前と奥の切り取り（Blender の clip_start・clip_end）の既定は、thresholds.CAMERA_NEAR・CAMERA_FAR（仮）。
# 上で読み込む。near より近い物・far より遠い物は写さない

# 回転の行列を「回転」とみなす幅（Blender の float32 の行列から作っても通るように、少しゆるめる）
_ROTATION_TOLERANCE = 1e-5
# 上の向きと見る向きが平行とみなす幅（2つの単位ベクトルの外積の長さ）
_PARALLEL_TOLERANCE = 1e-6
# 見る向きと上の向きが平行なときに、代わりに使う上の向き（この順に試す）。
# 真上から見下ろすと、Blender の「上から」の画面と同じく +Y が画像の上になる
_FALLBACK_UPS = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
# 一度に調べる候補のピクセルの数の目安。1回の計算のメモリを数十 MB までに抑える
_CHUNK_PIXELS = 1 << 17

_KINDS = ("perspective", "orthographic")
# summarize_visible_points の結果の名前（既定）。名前は、ほかの測る道具と同じ英語の小文字と _ だけ（snake_case）
VISIBLE_POINTS_NAME = "visible_points"
_MEASUREMENT_NAME = re.compile(r"[a-z][a-z0-9_]*")
_FOV_AXES = ("horizontal", "vertical", "auto")

# 決まった向きのカメラ：名前 → 中心から見たカメラの向き（単位ベクトル）。
# 左右はキャラから見た左右（キャラの左は +X）。Blender の「右から」（テンキー3）は +X から見るので、
# ここでは "left"（キャラの左側を見る）になる
_HALF_SQRT2 = math.sqrt(0.5)
_STANDARD_DIRECTIONS: dict[str, tuple[float, float, float]] = {
    "front": (0.0, -1.0, 0.0),
    "back": (0.0, 1.0, 0.0),
    "left": (1.0, 0.0, 0.0),
    "right": (-1.0, 0.0, 0.0),
    "front_left": (_HALF_SQRT2, -_HALF_SQRT2, 0.0),
    "front_right": (-_HALF_SQRT2, -_HALF_SQRT2, 0.0),
}
STANDARD_VIEW_NAMES = tuple(_STANDARD_DIRECTIONS)

# MARK: - カメラ


@dataclass(frozen=True, eq=False)
class Camera:
    """カメラ（Blender のカメラと同じ決まり）。

    position: カメラの場所 (3,)（メートル）
    rotation: (3, 3) の回転。列が、カメラの X（右）・Y（上）・Z（うしろ）の向きを世界の向きで表す
              （Blender の matrix_world の 3x3 の部分と同じ。カメラは -Z の向きを見る）
    kind: "perspective"（遠近あり）か "orthographic"（平行に写す）
    width, height: 画像の大きさ（ピクセル）。奥行きの画像は width * height * 16 バイトを使う
    fov_deg: 画角（度。perspective で使う）
    fov_axis: 画角と ortho_scale を、どちらの辺に合わせるか。
              "horizontal"（横）・"vertical"（縦）・"auto"（長いほうの辺。Blender の sensor_fit の AUTO と同じ）
    ortho_scale: 写す幅（メートル。orthographic で使う）。fov_axis で決めた辺の長さになる
    near, far: 写す奥行きの範囲（Blender の clip_start・clip_end）。この外の物は写さない

    position と rotation は、書き換えられない numpy の並びとして持つ。
    == で比べると、同じ物かどうかだけを見る（numpy の並びどうしの比べを避けるため）。
    """

    position: np.ndarray
    rotation: np.ndarray
    kind: str = "perspective"
    width: int = thresholds.VISIBILITY_RESOLUTION
    height: int = thresholds.VISIBILITY_RESOLUTION
    fov_deg: float = thresholds.APP_CAMERA_FOV_DEG
    fov_axis: str = "auto"
    ortho_scale: float | None = None
    near: float = CAMERA_NEAR
    far: float = CAMERA_FAR

    def __post_init__(self) -> None:
        position = _as_vector3(self.position, "position")
        rotation = _as_rotation(self.rotation)
        if self.kind not in _KINDS:
            raise ValueError(f"kind は {_KINDS} のどれかにしてください（いまは {self.kind!r}）")
        if self.fov_axis not in _FOV_AXES:
            raise ValueError(f"fov_axis は {_FOV_AXES} のどれかにしてください（いまは {self.fov_axis!r}）")
        width = _integer(self.width, "width")
        height = _integer(self.height, "height")
        if width < 1 or height < 1:
            raise ValueError(f"画像の大きさは 1 ピクセル以上にしてください（いまは {width} x {height}）")
        fov_deg = _real(self.fov_deg, "fov_deg")
        if not 0.0 < fov_deg < 180.0:
            raise ValueError(f"fov_deg は 0 より大きく 180 より小さくしてください（いまは {fov_deg}）")
        ortho_scale = None
        if self.ortho_scale is not None:
            ortho_scale = _real(self.ortho_scale, "ortho_scale")
            if not (math.isfinite(ortho_scale) and ortho_scale > 0.0):
                raise ValueError(f"ortho_scale は 0 より大きい数にしてください（いまは {ortho_scale}）")
        elif self.kind == "orthographic":
            raise ValueError("orthographic のカメラには ortho_scale（写す幅、メートル）が要ります")
        near = _real(self.near, "near")
        far = _real(self.far, "far")
        if not (math.isfinite(near) and near > 0.0):
            raise ValueError(f"near は 0 より大きい数にしてください（いまは {near}）")
        if not far > near:
            raise ValueError(f"far は near より大きくしてください（near {near}、far {far}）")
        for name, value in (("position", position), ("rotation", rotation), ("width", width), ("height", height),
                            ("fov_deg", fov_deg), ("ortho_scale", ortho_scale), ("near", near), ("far", far)):
            object.__setattr__(self, name, value)

    @property
    def right(self) -> np.ndarray:
        """画像の右の向き（世界の向き）"""
        return self.rotation[:, 0].copy()

    @property
    def up(self) -> np.ndarray:
        """画像の上の向き（世界の向き）"""
        return self.rotation[:, 1].copy()

    @property
    def forward(self) -> np.ndarray:
        """カメラの見る向き（世界の向き。カメラの -Z）"""
        return -self.rotation[:, 2]

    @property
    def pixel_scale(self) -> float:
        """長さとピクセルの比。
        perspective：奥行き 1m の所で、1m が何ピクセルになるか（焦点距離をピクセルで表した物）。
        orthographic：1m が何ピクセルになるか。
        """
        fit = {"horizontal": self.width, "vertical": self.height}.get(self.fov_axis, max(self.width, self.height))
        if self.kind == "perspective":
            return (fit / 2.0) / math.tan(math.radians(self.fov_deg) / 2.0)
        return fit / self.ortho_scale


def look_at(position, target, *, up=(0.0, 0.0, 1.0), kind: str = "perspective",
            width: int = thresholds.VISIBILITY_RESOLUTION, height: int = thresholds.VISIBILITY_RESOLUTION,
            fov_deg: float = thresholds.APP_CAMERA_FOV_DEG, fov_axis: str = "auto",
            ortho_scale: float | None = None, near: float = CAMERA_NEAR, far: float = CAMERA_FAR) -> Camera:
    """position から target を見るカメラを作る。画像の上は、up になるべく近い向きにする。

    見る向きと up が平行なとき（真上・真下から見るときなど）は、代わりに +Y、+Z、+X の順に、
    平行でない最初の向きを up に使う（いつも同じ答えになる）。
    """
    eye = _as_vector3(position, "position")
    goal = _as_vector3(target, "target")
    up_vector = _as_vector3(up, "up")
    forward = goal - eye
    length = float(np.linalg.norm(forward))
    if length == 0.0:
        raise ValueError("position と target が同じ場所です（見る向きが決まりません）")
    forward = forward / length
    up_length = float(np.linalg.norm(up_vector))
    if up_length == 0.0:
        raise ValueError("up の長さが 0 です")
    right = np.cross(forward, up_vector / up_length)
    if np.linalg.norm(right) < _PARALLEL_TOLERANCE:
        for candidate in _FALLBACK_UPS:
            right = np.cross(forward, np.asarray(candidate, dtype=np.float64))
            if np.linalg.norm(right) >= _PARALLEL_TOLERANCE:
                break
    right = right / np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    rotation = np.column_stack([right, camera_up, -forward])
    return Camera(eye, rotation, kind=kind, width=width, height=height, fov_deg=fov_deg, fov_axis=fov_axis,
                  ortho_scale=ortho_scale, near=near, far=far)


def standard_views(center, distance: float, *, width: int = thresholds.VISIBILITY_RESOLUTION,
                   height: int = thresholds.VISIBILITY_RESOLUTION, kind: str = "perspective",
                   fov_deg: float = thresholds.APP_CAMERA_FOV_DEG, fov_axis: str = "auto",
                   ortho_scale: float | None = None, near: float = CAMERA_NEAR,
                   far: float = CAMERA_FAR) -> dict[str, Camera]:
    """決まった6つの向きのカメラ。どれも center から distance 離れた、同じ高さから center を見る（上は +Z）。

    "front"：前（-Y の側）から見る。キャラの左（+X）は、画像の右に写る。
    "back"：うしろ（+Y の側）から見る。
    "left"：キャラの左（+X の側）から見る。キャラの前（-Y）は、画像の左に写る。
    "right"：キャラの右（-X の側）から見る。キャラの前（-Y）は、画像の右に写る。
    "front_left"・"front_right"：前と左（右）の、45度のななめから見る。
    """
    middle = _as_vector3(center, "center")
    distance = _real(distance, "distance")
    if not (math.isfinite(distance) and distance > 0.0):
        raise ValueError(f"distance は 0 より大きい数にしてください（いまは {distance}）")
    views: dict[str, Camera] = {}
    for name, direction in _STANDARD_DIRECTIONS.items():
        position = middle + distance * np.asarray(direction, dtype=np.float64)
        views[name] = look_at(position, middle, up=(0.0, 0.0, 1.0), kind=kind, width=width, height=height,
                              fov_deg=fov_deg, fov_axis=fov_axis, ortho_scale=ortho_scale, near=near, far=far)
    return views


# MARK: - 写す（点を画像の座標に）


def project(points, camera: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """点を画像に写す。

    返す物：
    - xy (P, 2)：画像の座標（ピクセル）。ピクセル (i, j) の中心は (i+0.5, j+0.5)、y は下向き。
      perspective でカメラの面より後ろ（depth <= 0）の点は、写せないので NaN。
    - depth (P,)：奥行き（カメラの見る向きに測った長さ。前が正）
    - in_front (P,) bool：写る奥行きの範囲（near 以上 far 以下）にあるか。
      画像の枠の中かどうかは見ない。
    """
    _check_camera(camera)
    array = _as_points(points, "points")
    sx, sy, depth = _to_screen(_to_camera_space(array, camera), camera)
    in_front = (depth >= camera.near) & (depth <= camera.far)
    return np.stack([sx, sy], axis=1), depth, in_front


def _to_camera_space(points: np.ndarray, camera: Camera) -> np.ndarray:
    """世界の座標をカメラの座標にする（rotation の転置をかける）"""
    return (points - camera.position) @ camera.rotation


def _to_screen(camera_points: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """カメラの座標を、画像の座標 (x, y) と奥行きにする。

    描く所と点を調べる所で、必ずこの関数を使う（同じ点が、ビット単位で同じ座標になるように）。
    """
    depth = -camera_points[:, 2]
    scale = camera.pixel_scale
    half_width = camera.width / 2.0
    half_height = camera.height / 2.0
    if camera.kind == "perspective":
        ahead = depth > 0.0
        safe = np.where(ahead, depth, 1.0)
        sx = half_width + scale * (camera_points[:, 0] / safe)
        sy = half_height - scale * (camera_points[:, 1] / safe)
        sx = np.where(ahead, sx, np.nan)
        sy = np.where(ahead, sy, np.nan)
    else:
        sx = half_width + scale * camera_points[:, 0]
        sy = half_height - scale * camera_points[:, 1]
    return sx, sy, depth


# MARK: - 奥行きの画像


@dataclass(eq=False)
class DepthBuffer:
    """奥行きの画像。

    depth (H, W) float64：いちばん手前の面の奥行き（メートル）。何も写っていないピクセルは +inf
    triangle (H, W) int64：そのピクセルに写った三角形の番号。何も写っていないピクセルは -1
    """

    depth: np.ndarray
    triangle: np.ndarray

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth)
        triangle = np.asarray(self.triangle)
        if depth.dtype.kind not in "iuf" or depth.ndim != 2:
            raise ValueError(f"depth は (H, W) の数の並びにしてください（いまは {depth.dtype}、{depth.shape}）")
        if triangle.dtype.kind not in "iu" or triangle.shape != depth.shape:
            raise ValueError(f"triangle は depth と同じ形 {depth.shape} の整数の並びにしてください"
                             f"（いまは {triangle.dtype}、{triangle.shape}）")
        depth = depth.astype(np.float64, copy=False)
        triangle = triangle.astype(np.int64, copy=False)
        if np.isnan(depth).any():
            raise ValueError("depth に NaN が入っています")
        if triangle.size and triangle.min() < -1:
            raise ValueError("triangle の番号は -1（何も無い）か 0 以上にしてください")
        empty = triangle < 0
        if not np.array_equal(empty, np.isposinf(depth)):
            raise ValueError("何も写っていないピクセルは、depth を +inf、triangle を -1 にしてください（両方そろえる）")
        if not (depth[~empty] > 0.0).all():
            raise ValueError("写っているピクセルの depth は、0 より大きくしてください")
        self.depth = depth
        self.triangle = triangle

    @classmethod
    def empty(cls, height: int, width: int) -> DepthBuffer:
        """何も写っていない奥行きの画像"""
        height = _integer(height, "height")
        width = _integer(width, "width")
        if height < 0 or width < 0:
            raise ValueError(f"画像の大きさは 0 以上にしてください（いまは {width} x {height}）")
        return cls(np.full((height, width), np.inf), np.full((height, width), -1, dtype=np.int64))

    @property
    def height(self) -> int:
        """画像の高さ（ピクセル）"""
        return int(self.depth.shape[0])

    @property
    def width(self) -> int:
        """画像の幅（ピクセル）"""
        return int(self.depth.shape[1])

    @property
    def covered(self) -> np.ndarray:
        """何かが写っているピクセル (H, W) bool"""
        return self.triangle >= 0


def rasterize(vertices, triangles, camera: Camera) -> DepthBuffer:
    """三角形のメッシュを、カメラから見た奥行きの画像にする。

    - 表も裏も描く（裏向きの面も、うしろの物を隠す）。
    - ピクセルの中心が三角形の中にあれば塗る。辺の上にちょうど乗った中心は「左上の規則」で決める
      （上の辺・左の辺は含み、右の辺・下の辺は含まない）。となり合う三角形の境目は、どちらか1つだけが塗る。
    - 奥行きは、perspective では画像の上で 1/奥行き を、orthographic では奥行きを、そのまま割り合いで混ぜる
      （遠近のある画像でも、面の上の正しい奥行きになる）。
    - near の面を横切る三角形は、カメラの座標で near の面で切ってから描く（Sutherland-Hodgman）。
      near より手前（カメラのうしろを含む）の部分は描かない。far より奥のピクセルも描かない。
    - 奥行きが同じピクセルは、番号の小さい三角形を残す。
    """
    _check_camera(camera)
    array, faces = _as_mesh(vertices, triangles)
    return _rasterize(array, faces, camera)


def render_views(vertices, triangles, cameras: Mapping[str, Camera]) -> dict[str, tuple[Camera, DepthBuffer]]:
    """いくつかのカメラで、同じメッシュを描く。名前 → (カメラ, 奥行きの画像)"""
    array, faces = _as_mesh(vertices, triangles)
    renders: dict[str, tuple[Camera, DepthBuffer]] = {}
    for name, camera in cameras.items():
        _check_camera(camera)
        renders[name] = (camera, _rasterize(array, faces, camera))
    return renders


def _rasterize(vertices: np.ndarray, triangles: np.ndarray, camera: Camera) -> DepthBuffer:
    height, width = camera.height, camera.width
    depth = np.full(height * width, np.inf)
    owner = np.full(height * width, -1, dtype=np.int64)
    if len(triangles):
        xs, ys, ds, ids = _screen_triangles(vertices, triangles, camera)
        if len(ids):
            _draw(xs, ys, ds, ids, camera, depth, owner)
    return DepthBuffer(depth.reshape(height, width), owner.reshape(height, width))


def _screen_triangles(vertices: np.ndarray, triangles: np.ndarray,
                      camera: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """描く三角形を、画像の座標の三角形にする。near の面を横切る三角形は切って、1つか2つにする。

    角は、決まった順に並べ直す（切る前はカメラの座標の x・y・z、切ったあとは画像の x・y の小さいほうから）。
    角の順や向きだけが違う同じ三角形（裏返して複製した両面の面など）が、ビット単位で同じ奥行きになり、
    「奥行きが同じなら番号の小さい三角形」の決まりが、丸めの誤差に負けないようにするため。

    返す物：x (K, 3)、y (K, 3)、奥行き (K, 3)、元の三角形の番号 (K,)
    """
    camera_points = _to_camera_space(vertices, camera)
    sx, sy, vertex_depth = _to_screen(camera_points, camera)
    corner_depth = vertex_depth[triangles]
    in_near = corner_depth >= camera.near
    inside_count = in_near.sum(axis=1)
    # 3つの角がどれも far より奥なら、どこも写らない
    not_beyond_far = (corner_depth <= camera.far).any(axis=1)
    whole = np.flatnonzero((inside_count == 3) & not_beyond_far)
    xs = [sx[triangles[whole]]]
    ys = [sy[triangles[whole]]]
    ds = [corner_depth[whole]]
    ids = [whole]
    for index in np.flatnonzero((inside_count > 0) & (inside_count < 3) & not_beyond_far):
        corners = camera_points[triangles[index]]
        corners = corners[np.lexsort(corners.T[::-1])]  # x、y、z の順に小さいほうから
        polygon = _clip_near(corners, camera.near)
        px, py, pd = _to_screen(polygon, camera)
        # 切った形は凸なので、最初の角から扇の形に三角形に分ける
        fans = np.array([[0, corner, corner + 1] for corner in range(1, len(polygon) - 1)], dtype=np.int64)
        xs.append(px[fans])
        ys.append(py[fans])
        ds.append(pd[fans])
        ids.append(np.full(len(fans), index, dtype=np.int64))
    x, y, d = np.concatenate(xs), np.concatenate(ys), np.concatenate(ds)
    order = np.lexsort((y, x))  # 三角形ごとに、画像の x、同じなら y の小さいほうから
    return (np.take_along_axis(x, order, axis=1), np.take_along_axis(y, order, axis=1),
            np.take_along_axis(d, order, axis=1), np.concatenate(ids))


def _clip_near(corners: np.ndarray, near: float) -> np.ndarray:
    """三角形（カメラの座標 (3, 3)）を、奥行き >= near の側だけ残すように切る（Sutherland-Hodgman）。

    返す物：残った形の角 (3, 3) か (4, 3)。切った所の角は、奥行きがちょうど near になる。
    """
    kept: list[np.ndarray] = []
    for i in range(3):
        a = corners[i]
        b = corners[(i + 1) % 3]
        a_inside = -a[2] >= near
        b_inside = -b[2] >= near
        if a_inside:
            kept.append(a)
        if a_inside != b_inside:
            kept.append(_near_crossing(a, b, near))
    return np.array(kept)


def _near_crossing(a: np.ndarray, b: np.ndarray, near: float) -> np.ndarray:
    """辺 a-b が near の面と交わる点。向きを決まった順にしてから計算する
    （となりの三角形と共有する辺で、ビット単位で同じ点になるように）"""
    if tuple(a) > tuple(b):
        a, b = b, a
    depth_a = -a[2]
    depth_b = -b[2]
    t = (near - depth_a) / (depth_b - depth_a)
    point = a + t * (b - a)
    point[2] = -near
    return point


def _draw(xs: np.ndarray, ys: np.ndarray, ds: np.ndarray, ids: np.ndarray, camera: Camera,
          depth: np.ndarray, owner: np.ndarray) -> None:
    """画像の座標の三角形を、奥行きの画像（平らにした depth と owner）に描く"""
    width, height = camera.width, camera.height
    # ピクセルの中心 i+0.5 が [最小, 最大] に入る i の範囲（画像の中に切り詰める）
    ix0 = np.maximum(np.ceil(np.clip(xs.min(axis=1) - 0.5, -1.0, width)), 0.0).astype(np.int64)
    ix1 = np.minimum(np.floor(np.clip(xs.max(axis=1) - 0.5, -1.0, width)), width - 1).astype(np.int64)
    iy0 = np.maximum(np.ceil(np.clip(ys.min(axis=1) - 0.5, -1.0, height)), 0.0).astype(np.int64)
    iy1 = np.minimum(np.floor(np.clip(ys.max(axis=1) - 0.5, -1.0, height)), height - 1).astype(np.int64)
    edges = _edge_data(xs, ys)
    keep = (ix0 <= ix1) & (iy0 <= iy1) & (edges["orientation"] != 0.0)
    if not keep.any():
        return
    ix0, ix1, iy0, iy1 = ix0[keep], ix1[keep], iy0[keep], iy1[keep]
    origin_x, origin_y = edges["origin_x"][keep], edges["origin_y"][keep]
    step_x, step_y = edges["step_x"][keep], edges["step_y"][keep]
    sign, top_left = edges["sign"][keep], edges["top_left"][keep]
    ds, ids = ds[keep], ids[keep]
    # 辺 k（a→b、b→c、c→a）の向かいの角は c、a、b
    opposite_depth = ds[:, [2, 0, 1]]
    opposite_weight = 1.0 / opposite_depth if camera.kind == "perspective" else opposite_depth
    lowest = ds.min(axis=1)
    highest = ds.max(axis=1)

    # 大きな三角形は、何行かずつの帯（タイル）に分ける。1つのタイルは _CHUNK_PIXELS か1行まで
    box_width = ix1 - ix0 + 1
    box_height = iy1 - iy0 + 1
    rows_per_tile = np.maximum(1, _CHUNK_PIXELS // box_width)
    tile_count = (box_height + rows_per_tile - 1) // rows_per_tile
    tile_triangle = np.repeat(np.arange(len(ids)), tile_count)
    tile_index = np.arange(len(tile_triangle)) - np.repeat(np.cumsum(tile_count) - tile_count, tile_count)
    tile_y0 = iy0[tile_triangle] + tile_index * rows_per_tile[tile_triangle]
    tile_rows = np.minimum(rows_per_tile[tile_triangle], iy1[tile_triangle] - tile_y0 + 1)
    tile_x0 = ix0[tile_triangle]
    tile_width = box_width[tile_triangle]
    tile_pixels = tile_width * tile_rows
    # タイルを、候補のピクセルが _CHUNK_PIXELS くらいずつになるように組にする
    tile_start = np.cumsum(tile_pixels) - tile_pixels
    group = tile_start // _CHUNK_PIXELS
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(group)) + 1, [len(group)]])

    for lo, hi in zip(bounds[:-1], bounds[1:]):
        counts = tile_pixels[lo:hi]
        total = int(counts.sum())
        local_tile = np.repeat(np.arange(hi - lo), counts)
        offset = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        row_width = tile_width[lo:hi][local_tile]
        column = tile_x0[lo:hi][local_tile] + offset % row_width
        row = tile_y0[lo:hi][local_tile] + offset // row_width
        k = tile_triangle[lo:hi][local_tile]
        del local_tile, offset, row_width
        center_x = column + 0.5
        center_y = row + 0.5
        inside = np.ones(total, dtype=bool)
        values = []
        for e in range(3):
            value = sign[k, e] * (step_x[k, e] * (center_y - origin_y[k, e])
                                  - step_y[k, e] * (center_x - origin_x[k, e]))
            inside &= (value > 0.0) | ((value == 0.0) & top_left[k, e])
            values.append(value)
        if not inside.any():
            continue
        k = k[inside]
        values = [value[inside] for value in values]
        pixel = row[inside] * width + column[inside]
        total_weight = values[0] + values[1] + values[2]
        mixed = (values[0] * opposite_weight[k, 0] + values[1] * opposite_weight[k, 1]
                 + values[2] * opposite_weight[k, 2]) / np.where(total_weight > 0.0, total_weight, 1.0)
        pixel_depth = 1.0 / mixed if camera.kind == "perspective" else mixed
        # 丸めの誤差で、三角形の角の奥行きの範囲から出ないようにする（平らな面はぴったり同じ奥行きになる）
        pixel_depth = np.clip(pixel_depth, lowest[k], highest[k])
        good = (total_weight > 0.0) & (pixel_depth <= camera.far)
        _merge(pixel[good], pixel_depth[good], ids[k[good]], depth, owner)


def _edge_data(xs: np.ndarray, ys: np.ndarray) -> dict[str, np.ndarray]:
    """三角形の3つの辺の式を作る。

    辺 a→b の式 E(p) = (b.x-a.x)(p.y-a.y) - (b.y-a.y)(p.x-a.x) を、2つの端を決まった順
    （x、同じなら y の小さいほうから）にして計算し、向きの符号をかける。
    こうすると、となり合う三角形が共有する辺で、式の値がビット単位で同じ（符号だけ逆）になる。
    sign は、三角形の中で E が正になるようにそろえる符号。
    top_left は、中心がその辺の上に乗ったときに塗るか（左上の規則）。
    """
    origin_x = np.empty_like(xs)
    origin_y = np.empty_like(ys)
    step_x = np.empty_like(xs)
    step_y = np.empty_like(ys)
    direction = np.empty_like(xs)
    for e, (u, v) in enumerate(((0, 1), (1, 2), (2, 0))):
        ux, uy, vx, vy = xs[:, u], ys[:, u], xs[:, v], ys[:, v]
        swap = (ux > vx) | ((ux == vx) & (uy > vy))
        origin_x[:, e] = np.where(swap, vx, ux)
        origin_y[:, e] = np.where(swap, vy, uy)
        step_x[:, e] = np.where(swap, ux - vx, vx - ux)
        step_y[:, e] = np.where(swap, uy - vy, vy - uy)
        direction[:, e] = np.where(swap, -1.0, 1.0)
    # 辺 a→b の式に c を入れた値で、三角形の向き（画像の上で）を決める
    area = direction[:, 0] * (step_x[:, 0] * (ys[:, 2] - origin_y[:, 0]) - step_y[:, 0] * (xs[:, 2] - origin_x[:, 0]))
    orientation = np.sign(area)
    sign = direction * orientation[:, None]
    # そろえた向きでの辺の向き (dx, dy)。y は下向きなので、dy < 0 が左の辺、dy == 0 で dx > 0 が上の辺
    dx = sign * step_x
    dy = sign * step_y
    top_left = (dy < 0.0) | ((dy == 0.0) & (dx > 0.0))
    return {"origin_x": origin_x, "origin_y": origin_y, "step_x": step_x, "step_y": step_y, "sign": sign,
            "top_left": top_left, "orientation": orientation}


def _merge(pixel: np.ndarray, pixel_depth: np.ndarray, triangle: np.ndarray, depth: np.ndarray,
           owner: np.ndarray) -> None:
    """候補を奥行きの画像に入れる。手前の物を残し、奥行きが同じなら番号の小さい三角形を残す"""
    if not len(pixel):
        return
    order = np.lexsort((triangle, pixel_depth, pixel))
    sorted_pixel = pixel[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = sorted_pixel[1:] != sorted_pixel[:-1]
    best = order[first]
    where = pixel[best]
    new_depth = pixel_depth[best]
    new_owner = triangle[best]
    current_depth = depth[where]
    better = (new_depth < current_depth) | ((new_depth == current_depth) & (new_owner < owner[where]))
    depth[where[better]] = new_depth[better]
    owner[where[better]] = new_owner[better]


# MARK: - 見えるかどうか


def visible_triangle_pixels(buffer: DepthBuffer, n_triangles: int) -> np.ndarray:
    """三角形ごとの、写ったピクセルの数 (M,) int64。M は n_triangles（メッシュの三角形の数）"""
    _check_buffer(buffer)
    count = _integer(n_triangles, "n_triangles")
    if count < 0:
        raise ValueError(f"n_triangles は 0 以上にしてください（いまは {count}）")
    ids = buffer.triangle[buffer.triangle >= 0]
    if ids.size and int(ids.max()) >= count:
        raise ValueError(f"奥行きの画像に、番号 {int(ids.max())} の三角形があります（n_triangles {count} より大きい）")
    return np.bincount(ids, minlength=count).astype(np.int64)


def visible_triangles(buffer: DepthBuffer, n_triangles: int, *, min_pixels: int = 1) -> np.ndarray:
    """min_pixels ピクセル以上写った三角形 (M,) bool"""
    least = _integer(min_pixels, "min_pixels")
    if least < 1:
        raise ValueError(f"min_pixels は 1 以上にしてください（いまは {least}）")
    return visible_triangle_pixels(buffer, n_triangles) >= least


def visible_points(points, buffer: DepthBuffer, camera: Camera, *,
                   epsilon: float = thresholds.VISIBILITY_DEPTH_EPSILON, mesh=None) -> np.ndarray:
    """点がカメラから見えるか (P,) bool。

    見える：画像の枠の中にあり、写る奥行きの範囲（near〜far）にあり、
    点の奥行き <= その位置の面の奥行き + epsilon（何も写っていないピクセルなら見える）。

    面の奥行き：
    - mesh が無いとき：点のピクセルの、中心での奥行き。面が斜めだと、同じ面の上の点でも、
      中心からのずれの分だけ奥行きが違い、見える点を「隠れている」とまちがえることがある
      （512 ピクセル・1ピクセル約 8mm で 45 度の面なら、差は数 mm になり、epsilon の 1mm を超える）。
    - mesh=(vertices, triangles)（描いたときと同じメッシュ）を渡すとき：そのピクセルに写った三角形の面を、
      点のちょうどの位置まで延ばした奥行き。斜めの面でも、面の上の点は面と同じ奥行きになる。
      見える問題を見落とさないために、なるべく mesh を渡す。
    """
    _check_camera(camera)
    array = _as_points(points, "points")
    margin = _epsilon(epsilon)
    surface_mesh = _optional_mesh(mesh)
    measured, depth, surface = _lookup(array, buffer, camera, surface_mesh)
    return measured & (depth <= surface + margin)


def visible_fraction(points, buffer: DepthBuffer, camera: Camera, *,
                     epsilon: float = thresholds.VISIBILITY_DEPTH_EPSILON, mesh=None) -> float:
    """点のうち、見える点の割合（0〜1）。点が無いときは NaN（測れない）。mesh は visible_points と同じ"""
    mask = visible_points(points, buffer, camera, epsilon=epsilon, mesh=mesh)
    return float(mask.mean()) if mask.size else math.nan


def visible_from_any(points, renders: Mapping[str, tuple[Camera, DepthBuffer]], *,
                     epsilon: float = thresholds.VISIBILITY_DEPTH_EPSILON, mesh=None) -> np.ndarray:
    """どれか1つのカメラから見える点 (P,) bool。renders は render_views の返す物（名前 → (カメラ, 奥行きの画像)）。
    mesh は visible_points と同じ"""
    array = _as_points(points, "points")
    margin = _epsilon(epsilon)
    surface_mesh = _optional_mesh(mesh)
    seen = np.zeros(len(array), dtype=bool)
    for _, camera, buffer in _render_items(renders):
        measured, depth, surface = _lookup(array, buffer, camera, surface_mesh)
        seen |= measured & (depth <= surface + margin)
    return seen


def _lookup(points: np.ndarray, buffer: DepthBuffer, camera: Camera,
            mesh: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """点ごとに：調べられたか（枠の中で、写る奥行きの範囲）、点の奥行き、その位置の面の奥行き（調べられない点は NaN）"""
    _check_camera(camera)
    _check_buffer(buffer, camera)
    local = _to_camera_space(points, camera)
    sx, sy, depth = _to_screen(local, camera)
    with np.errstate(invalid="ignore"):
        measured = ((sx >= 0.0) & (sx < camera.width) & (sy >= 0.0) & (sy < camera.height)
                    & (depth >= camera.near) & (depth <= camera.far))
    surface = np.full(len(points), np.nan)
    index = np.flatnonzero(measured)
    if len(index):
        column = np.floor(sx[index]).astype(np.int64)
        row = np.floor(sy[index]).astype(np.int64)
        surface[index] = buffer.depth[row, column]
        if mesh is not None:
            owner = buffer.triangle[row, column]
            hit = owner >= 0
            if hit.any():
                surface[index[hit]] = _plane_depth(local[index[hit]], owner[hit], surface[index[hit]], mesh, camera)
    return measured, depth, surface


def _plane_depth(local_points: np.ndarray, owners: np.ndarray, fallback: np.ndarray,
                 mesh: tuple[np.ndarray, np.ndarray], camera: Camera) -> np.ndarray:
    """ピクセルに写った三角形の面を、点のちょうどの位置（画像の上の位置）まで延ばした奥行き。

    perspective：カメラから点への光線と、面の交わる所。orthographic：点を通る、見る向きの線と面の交わる所。
    値は、三角形の角の奥行きの範囲（near より手前は near）に収める（真横に近い面で、値が飛ばないように）。
    計算できないとき（面が光線と平行など）は、fallback（ピクセルの中心の奥行き）を使う。
    """
    vertices, triangles = mesh
    if int(owners.max()) >= len(triangles):
        raise ValueError(f"奥行きの画像に、番号 {int(owners.max())} の三角形があります（mesh の三角形は "
                         f"{len(triangles)} 個。描いたときと同じメッシュを渡してください）")
    corners = _to_camera_space(vertices[triangles[owners]].reshape(-1, 3), camera).reshape(-1, 3, 3)
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    offset = np.einsum("ij,ij->i", normal, corners[:, 0])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if camera.kind == "perspective":
            # 光線 r = s * q（q は点）。面 n・r = offset との交わりは s = offset / (n・q)。奥行きは s * (点の奥行き)
            exact = offset / np.einsum("ij,ij->i", normal, local_points) * -local_points[:, 2]
        else:
            # 線 r = (q.x, q.y, z)。n・r = offset から z を求める。奥行きは -z
            exact = -(offset - normal[:, 0] * local_points[:, 0] - normal[:, 1] * local_points[:, 1]) / normal[:, 2]
    corner_depth = -corners[:, :, 2]
    low = np.maximum(corner_depth.min(axis=1), camera.near)
    high = corner_depth.max(axis=1)
    return np.where(np.isfinite(exact), np.clip(exact, low, high), fallback)


# MARK: - まとめ（Measurement）


def summarize_visible_points(points, renders: Mapping[str, tuple[Camera, DepthBuffer]], *,
                             name: str = VISIBLE_POINTS_NAME, signal_if_visible: Signal = Signal.RED,
                             epsilon: float = thresholds.VISIBILITY_DEPTH_EPSILON, mesh=None) -> Measurement:
    """問題の点（赤のめり込みの場所など）が、カメラから見えるかをまとめる。
    mesh（描いたときと同じメッシュ）は visible_points と同じ。見落としを減らすため、なるべく渡す。
    name は結果の名前（英語の小文字・数字・_ だけ。例 "visible_penetration"）。決まりに合わないときは ValueError。

    値：どれか1つのカメラから見える点の数（count）。
    信号：見える点があれば signal_if_visible（赤か黄。ふつうは赤）。見える点が無ければ緑（隠れた点は記録だけ）。
    灰（測れない）：カメラが無い、何も写っていないカメラがある（向き・距離の設定がおかしい）、
    どのカメラでも枠の外か、写る奥行きの範囲の外にある点がある（見えるかどうか分からない）。
    見える点があって灰の理由もあるときは、重いほうの信号にする（赤 > 灰 > 黄）。
    値は見えた点の数のまま（少なくともこの数）にし、理由の文には両方を書く。

    details：
    - "visible" (P,) bool：どれかのカメラから見える点
    - "unmeasured" (P,) bool：どのカメラでも調べられなかった点
    - "per_view"：カメラの名前 → 見える点の数
    - "visible_count"・"hidden_count"・"unmeasured_count"：点の数
    - "empty_views"：何も写っていないカメラの名前
    """
    if not isinstance(name, str) or not _MEASUREMENT_NAME.fullmatch(name):
        raise ValueError(f"name は英語の小文字・数字・_ だけの名前にしてください（例 \"visible_points\"。いまは {name!r}）")
    array = _as_points(points, "points")
    margin = _epsilon(epsilon)
    surface_mesh = _optional_mesh(mesh)
    if not isinstance(signal_if_visible, Signal) or signal_if_visible not in (Signal.YELLOW, Signal.RED):
        raise ValueError(f"signal_if_visible は Signal.RED か Signal.YELLOW にしてください（いまは {signal_if_visible!r}。"
                         "見える問題を緑や灰にはしない）")
    items = _render_items(renders)
    if not items:
        return Measurement.unmeasured(name, "count", "カメラがありません")
    seen = np.zeros(len(array), dtype=bool)
    measured_any = np.zeros(len(array), dtype=bool)
    per_view: dict[str, int] = {}
    empty_views: list[str] = []
    for view_name, camera, buffer in items:
        measured, depth, surface = _lookup(array, buffer, camera, surface_mesh)
        visible = measured & (depth <= surface + margin)
        per_view[view_name] = int(visible.sum())
        seen |= visible
        measured_any |= measured
        if not buffer.covered.any():
            empty_views.append(view_name)
    visible_count = int(seen.sum())
    unmeasured_count = int((~measured_any).sum())
    hidden_count = len(array) - visible_count - unmeasured_count
    details: dict[str, Any] = {
        "visible": seen, "unmeasured": ~measured_any, "per_view": per_view, "visible_count": visible_count,
        "hidden_count": hidden_count, "unmeasured_count": unmeasured_count, "empty_views": empty_views,
    }
    problems = []
    if empty_views:
        problems.append(f"何も写っていないカメラがあります（{'、'.join(empty_views)}。向き・距離を確かめてください）")
    if unmeasured_count:
        problems.append(f"どのカメラでも枠の外にある点が {unmeasured_count} 個あります（カメラの範囲を確かめてください）")
    if visible_count:
        reason = f"カメラから見える点が {visible_count} 個あります（隠れた点 {hidden_count} 個）"
        signal = worst([signal_if_visible, Signal.GRAY]) if problems else signal_if_visible
        return Measurement(name, float(visible_count), "count", signal, "。".join([reason, *problems]), details)
    if problems:
        return Measurement.unmeasured(name, "count", "。".join(problems), details)
    return Measurement(name, 0.0, "count", Signal.GREEN, f"カメラから見える点はありません（隠れた点 {hidden_count} 個）",
                       details)


# MARK: - 入力の確かめ


def _check_camera(camera: Any) -> None:
    if not isinstance(camera, Camera):
        raise ValueError(f"camera は Camera にしてください（いまは {type(camera).__name__}）")


def _check_buffer(buffer: Any, camera: Camera | None = None) -> None:
    if not isinstance(buffer, DepthBuffer):
        raise ValueError(f"buffer は DepthBuffer にしてください（いまは {type(buffer).__name__}）")
    if camera is not None and buffer.depth.shape != (camera.height, camera.width):
        raise ValueError(f"奥行きの画像の大きさ {buffer.depth.shape} が、カメラの画像 ({camera.height}, {camera.width}) と"
                         "違います（同じカメラで描いた物を使ってください）")


def _render_items(renders: Any) -> list[tuple[Any, Camera, DepthBuffer]]:
    """renders（名前 → (カメラ, 奥行きの画像)）を確かめて、(名前, カメラ, 奥行きの画像) の並びにする"""
    if not isinstance(renders, Mapping):
        raise ValueError(f"renders は「名前 → (カメラ, 奥行きの画像)」の辞書にしてください（render_views の返す物。"
                         f"いまは {type(renders).__name__}）")
    items = []
    for name, value in renders.items():
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError(f"renders[{name!r}] は (カメラ, 奥行きの画像) の組にしてください")
        camera, buffer = value
        _check_camera(camera)
        _check_buffer(buffer, camera)
        items.append((name, camera, buffer))
    return items


def _real(value: Any, name: str) -> float:
    """数（bool は使えない）を float にする。NaN は使えない"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} は数にしてください（いまは {value!r}）")
    result = float(value)
    if math.isnan(result):
        raise ValueError(f"{name} が NaN です")
    return result


def _integer(value: Any, name: str) -> int:
    """整数（bool は使えない）を int にする"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} は整数にしてください（いまは {value!r}）")
    return int(value)


def _epsilon(value: Any) -> float:
    margin = _real(value, "epsilon")
    if not (math.isfinite(margin) and margin >= 0.0):
        raise ValueError(f"epsilon は 0 以上の数にしてください（いまは {margin}）")
    return margin


def _float_array(value: Any, name: str) -> np.ndarray:
    """数の並びを float64 にする（空の並びは型を問わない）。並びにできない物は日本語の ValueError"""
    raw = as_array(value, name)
    if raw.size and raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} は数の並びにしてください（いまは {raw.dtype}）")
    return raw.astype(np.float64)


def _as_vector3(value: Any, name: str) -> np.ndarray:
    """(3,) の、NaN・無限大の無い float64 の並び（書き換えられない写し）"""
    array = _float_array(value, name)
    if array.shape != (3,):
        raise ValueError(f"{name} は (3,) の並びにしてください（いまは {array.shape}）")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} に NaN か無限大が入っています")
    array.setflags(write=False)
    return array


def _as_rotation(value: Any) -> np.ndarray:
    """(3, 3) の回転の行列（列が直交する単位ベクトルで、右手系）。書き換えられない写し"""
    array = _float_array(value, "rotation")
    if array.shape != (3, 3):
        raise ValueError(f"rotation は (3, 3) の並びにしてください（いまは {array.shape}）")
    if not np.isfinite(array).all():
        raise ValueError("rotation に NaN か無限大が入っています")
    if np.abs(array.T @ array - np.eye(3)).max() > _ROTATION_TOLERANCE:
        raise ValueError("rotation の列は、長さ 1 で互いに直交する向きにしてください（大きさや傾きを含めない）")
    if np.linalg.det(array) <= 0.0:
        raise ValueError("rotation は右手系にしてください（鏡に映した向きは使えません）")
    array.setflags(write=False)
    return array


def _as_points(value: Any, name: str) -> np.ndarray:
    """(P, 3) の、NaN・無限大の無い float64 の並び。空（[] か (0, 3)）なら (0, 3)。
    (5, 0) のような、空でも形の違う並びは使えない（まちがった入力を「点が無い」とみなして緑にしないため）"""
    array = _float_array(value, name)
    if array.size == 0:
        if array.shape not in ((0,), (0, 3)):
            raise ValueError(f"{name} は (N, 3) の並びにしてください（いまは {array.shape}）")
        return np.zeros((0, 3))
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} は (N, 3) の並びにしてください（いまは {array.shape}）")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} に NaN か無限大が入っています")
    return array


def _optional_mesh(mesh: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """mesh=(vertices, triangles) を確かめる。None ならそのまま"""
    if mesh is None:
        return None
    if not isinstance(mesh, (tuple, list)) or len(mesh) != 2:
        raise ValueError("mesh は (vertices, triangles) の組にしてください")
    return _as_mesh(mesh[0], mesh[1])


def _as_mesh(vertices: Any, triangles: Any) -> tuple[np.ndarray, np.ndarray]:
    """点 (N, 3) float64 と三角形 (M, 3) int64。三角形の番号が点の数の中にあるかも確かめる"""
    array = _as_points(vertices, "vertices")
    raw = as_array(triangles, "triangles")
    if raw.size == 0:
        if raw.shape not in ((0,), (0, 3)):
            raise ValueError(f"triangles は (M, 3) の並びにしてください（いまは {raw.shape}）")
        return array, np.zeros((0, 3), dtype=np.int64)
    if raw.dtype.kind not in "iu":
        raise ValueError(f"triangles は整数の並びにしてください（bool や小数は使えません。いまは {raw.dtype}）")
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(f"triangles は (M, 3) の並びにしてください（いまは {raw.shape}）")
    faces = raw.astype(np.int64)
    if faces.min() < 0 or faces.max() >= len(array):
        raise ValueError(f"triangles の番号は 0 以上 {len(array)} 未満にしてください"
                         f"（いまは {int(faces.min())}〜{int(faces.max())}）")
    return array, faces

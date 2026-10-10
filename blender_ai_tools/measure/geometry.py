"""A4 めり込み・すき間を測るための、形の計算（bpy なし・numpy だけ）。

- 三角形の向き（法線）と面積、頂点の向き（角度で重みを付けた法線）
- 点から三角形の面への、いちばん近い点（closest_points）
- 符号つきの距離（signed_distance）：形の中は負、外は正
- めり込みの深さ・すき間と、その信号（summarize_penetration・summarize_gap）

Blender の中では、あとで mathutils の BVHTree を使う速い版を作る。ここの numpy 版は、それと比べるための
「正しい答え」（基準の実装）なので、速さより正しさを大事にする。点と三角形の組を全部調べるが、
小分けにして、メモリが増えすぎないようにする。

決まり（CLAUDE.md 7章）：長さはメートル。点は (N, 3) の float64、三角形は (M, 3) の int64。
面は外から見て反時計回り（法線が外を向く）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import thresholds
from .results import Measurement, Signal, classify, worst

# MARK: - 計算のための数字（判定の基準ではない）

# 点と三角形の組を、一度にいくつまで計算するか。1組あたり約 300 バイトなので、約 20MB になる
DEFAULT_CHUNK_PAIRS = 1 << 16
# 細い三角形を「線」とみなす割合。高さが、いちばん長い辺のこの割合以下なら、面の向きが決まらないので、
# 3本の辺（線分）として扱う。これで生まれる距離の差は、高さ（辺の長さの 1e-10 倍）より小さい
SLIVER_RATIO = 1e-10
# 擬似法線（下の signed_distance）の長さがこれ以下なら、向きが決まらないとみなす。
# 単位の長さの法線を足した物なので、長さはふつう 0.1 より大きい
PSEUDO_NORMAL_EPS = 1e-9
# details に入れる「いちばん悪い点」の数
WORST_LIST_LENGTH = 10

# いちばん近い点が、三角形のどこにあるか（ClosestResult.features の値）
FEATURE_FACE = 0  # 面の中
FEATURE_EDGE_01 = 1  # 辺 v0-v1
FEATURE_EDGE_12 = 2  # 辺 v1-v2
FEATURE_EDGE_20 = 3  # 辺 v2-v0
FEATURE_VERTEX_0 = 4  # 頂点 v0
FEATURE_VERTEX_1 = 5  # 頂点 v1
FEATURE_VERTEX_2 = 6  # 頂点 v2

_INT64_MAX = np.iinfo(np.int64).max


# MARK: - 入力の確かめ

def as_array(value: Any, name: str) -> np.ndarray:
    """np.asarray と同じ（型も形も変えない）。行の長さがそろわない並びなど、並びにできない物は、
    numpy の英語のエラーではなく、日本語の ValueError にする。

    measure/ の下のモジュールは、入力を受け取る所で np.asarray の代わりにこれを使う（エラーの文を日本語にそろえる）。
    """
    try:
        return np.asarray(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} は形のそろった並びにしてください（行の長さが違うなど）") from error


def as_points(points: Any, name: str = "points") -> np.ndarray:
    """点の並びを (N, 3) の float64 にする。形・型が違う、NaN・無限大があるときは ValueError。

    空の並び（[] や shape (0,)）は (0, 3) にする。真偽（bool）の並びは受け付けない。
    """
    array = as_array(points, name)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} は数の並びにしてください（いまの型：{array.dtype}）")
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 1 and array.size == 0:
        array = array.reshape(0, 3)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} は (N, 3) の並びにしてください（いまの形：{array.shape}）")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} に NaN か無限大が入っています")
    return array


def as_triangles(triangles: Any, vertex_count: int | None = None, name: str = "triangles") -> np.ndarray:
    """三角形の並びを (M, 3) の int64 にする。

    整数でない（小数・真偽）、形が違う、番号が負か vertex_count 以上のときは ValueError。
    vertex_count が None のときは、番号の上限を確かめない。
    """
    array = as_array(triangles, name)
    if array.size == 0:
        if array.dtype.kind == "b":
            raise ValueError(f"{name} は整数の並びにしてください（真偽の並びは使えません）")
        array = array.reshape(0, 3) if array.ndim == 1 else array
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(f"{name} は (M, 3) の並びにしてください（いまの形：{array.shape}）")
        return np.zeros((0, 3), dtype=np.int64)
    if array.dtype.kind not in "iu":
        raise ValueError(f"{name} は整数の並びにしてください（いまの型：{array.dtype}）")
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} は (M, 3) の並びにしてください（いまの形：{array.shape}）")
    if array.dtype.kind == "u" and int(array.max()) > _INT64_MAX:
        raise ValueError(f"{name} の番号が大きすぎます")
    array = np.asarray(array, dtype=np.int64)
    if int(array.min()) < 0:
        raise ValueError(f"{name} に負の番号があります")
    if vertex_count is not None and int(array.max()) >= vertex_count:
        raise ValueError(f"{name} の番号が頂点の数（{vertex_count}）を超えています")
    return array


def as_mask(mask: Any, length: int, name: str) -> np.ndarray:
    """真偽の並び (length,) にする。0/1 の整数の並びは、番号の並びとまちがえやすいので受け付けない"""
    array = as_array(mask, name)
    if array.dtype.kind != "b":
        raise ValueError(f"{name} は真偽（bool）の並びにしてください（いまの型：{array.dtype}）")
    if array.shape != (length,):
        raise ValueError(f"{name} の形は ({length},) にしてください（いまの形：{array.shape}）")
    return array


def as_positive_int(value: Any, name: str) -> int:
    """1 以上の整数にする。真偽（True・False）は整数として受け付けない"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} は 1 以上の整数にしてください（いまの値：{value!r}）")
    if value < 1:
        raise ValueError(f"{name} は 1 以上の整数にしてください（いまの値：{value!r}）")
    return int(value)


def as_nonnegative_float(value: Any, name: str) -> float:
    """0 以上の有限の数にする。真偽は受け付けない"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} は 0 以上の数にしてください（いまの値：{value!r}）")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} は 0 以上の有限の数にしてください（いまの値：{value!r}）")
    return result


def as_values(values: Any, name: str, *, allow_nan: bool) -> np.ndarray:
    """1次元の float64 の並びにする。無限大は受け付けない。allow_nan=False のときは NaN も受け付けない"""
    array = as_array(values, name)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} は数の並びにしてください（いまの型：{array.dtype}）")
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} は1次元の並びにしてください（いまの形：{array.shape}）")
    if np.isinf(array).any():
        raise ValueError(f"{name} に無限大が入っています")
    if not allow_nan and np.isnan(array).any():
        raise ValueError(f"{name} に NaN が入っています")
    return array


# MARK: - 三角形の向きと面積

def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """最後の軸（x, y, z）の内積。足す順を決めておき、小分けの仕方で答えが変わらないようにする"""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _corners(vertices: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]


def _sliver_mask(a: np.ndarray, b: np.ndarray, c: np.ndarray, cross_length: np.ndarray) -> np.ndarray:
    """面の向きが決まらない（細すぎる・つぶれた）三角形の印。

    |ab × ac| は「いちばん長い辺 × その辺への高さ」なので、高さ <= SLIVER_RATIO × いちばん長い辺 のとき真
    """
    longest2 = np.maximum(np.maximum(_dot(b - a, b - a), _dot(c - b, c - b)), _dot(a - c, a - c))
    return cross_length <= SLIVER_RATIO * longest2


def triangle_areas(vertices: Any, triangles: Any) -> np.ndarray:
    """三角形ごとの面積 (M,)（平方メートル）"""
    v = as_points(vertices, "vertices")
    t = as_triangles(triangles, len(v))
    a, b, c = _corners(v, t)
    return 0.5 * np.sqrt(_dot(np.cross(b - a, c - a), np.cross(b - a, c - a)))


def _triangle_normals(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    a, b, c = _corners(v, t)
    cross = np.cross(b - a, c - a)
    length = np.sqrt(_dot(cross, cross))
    good = ~_sliver_mask(a, b, c, length)
    normals = np.zeros_like(cross)
    normals[good] = cross[good] / length[good, None]
    return normals


def triangle_normals(vertices: Any, triangles: Any) -> np.ndarray:
    """三角形ごとの、長さ 1 の法線 (M, 3)。反時計回りに見える側を向く。

    つぶれた・細すぎる三角形（高さが、いちばん長い辺の SLIVER_RATIO 倍以下）は、向きが決まらないので 0 のベクトル。
    """
    v = as_points(vertices, "vertices")
    return _triangle_normals(v, as_triangles(triangles, len(v)))


def _angle_weighted_sums(v: np.ndarray, t: np.ndarray, face_normals: np.ndarray) -> np.ndarray:
    """頂点ごとに「角の大きさ × 面の法線」を足した物 (N, 3)（長さはそろえない）"""
    total = np.zeros((len(v), 3))
    if len(t) == 0:
        return total
    for k in range(3):
        here = v[t[:, k]]
        e1 = v[t[:, (k + 1) % 3]] - here
        e2 = v[t[:, (k + 2) % 3]] - here
        cross = np.cross(e1, e2)
        # atan2 は、辺の長さが 0 でも NaN にならない（0 になる）
        angle = np.arctan2(np.sqrt(_dot(cross, cross)), _dot(e1, e2))
        weighted = face_normals * angle[:, None]
        for axis in range(3):
            total[:, axis] += np.bincount(t[:, k], weights=weighted[:, axis], minlength=len(v))
    return total


def vertex_normals(vertices: Any, triangles: Any) -> np.ndarray:
    """頂点ごとの、長さ 1 の法線 (N, 3)。まわりの面の法線に、その頂点での角の大きさで重みを付けて足す。

    角で重みを付けるので、面の分け方（四角形をどちら向きの三角形に分けたか）で向きが変わらない。
    どの面にも使われていない頂点や、向きが打ち消し合う頂点は 0 のベクトル。
    """
    v = as_points(vertices, "vertices")
    t = as_triangles(triangles, len(v))
    total = _angle_weighted_sums(v, t, _triangle_normals(v, t))
    length = np.sqrt(_dot(total, total))
    result = np.zeros_like(total)
    good = length > PSEUDO_NORMAL_EPS
    result[good] = total[good] / length[good, None]
    return result


def weld_exact(vertices: Any) -> np.ndarray:
    """座標がまったく同じ頂点をまとめる。頂点ごとに、同じ座標の頂点の中でいちばん小さい番号を返す (N,)。

    距離の幅は使わない（ぴったり同じだけ）。-0.0 と 0.0 は同じとみなす。
    """
    v = as_points(vertices, "vertices") + 0.0  # -0.0 を 0.0 にする
    n = len(v)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort((v[:, 2], v[:, 1], v[:, 0]))  # 安定な並べ替えなので、同じ座標の中では番号の小さい順
    sorted_v = v[order]
    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    new_group[1:] = np.any(sorted_v[1:] != sorted_v[:-1], axis=1)
    group = np.cumsum(new_group) - 1
    representative = order[new_group]  # 各まとまりの先頭 = いちばん小さい番号
    result = np.empty(n, dtype=np.int64)
    result[order] = representative[group]
    return result


# MARK: - いちばん近い点

@dataclass(frozen=True)
class ClosestResult:
    """closest_points の答え。

    points: (P, 3) 面の上の、いちばん近い点
    distances: (P,) その点までの距離（メートル）
    triangles: (P,) int64 その点がある三角形の番号（同じ距離なら、いちばん小さい番号）
    features: (P,) int8 三角形のどこか。0 = 面の中、1〜3 = 辺（v0-v1、v1-v2、v2-v0）、4〜6 = 頂点（v0、v1、v2）
    """

    points: np.ndarray
    distances: np.ndarray
    triangles: np.ndarray
    features: np.ndarray


class _TriangleData:
    """三角形ごとに、前もって計算しておく値"""

    def __init__(self, v: np.ndarray, t: np.ndarray) -> None:
        self.a, self.b, self.c = _corners(v, t)
        self.ab = self.b - self.a
        self.ac = self.c - self.a
        self.bc = self.c - self.b
        ab2, ac2, bc2 = _dot(self.ab, self.ab), _dot(self.ac, self.ac), _dot(self.bc, self.bc)
        cross = np.cross(self.ab, self.ac)
        den = _dot(cross, cross)  # = |ab|^2 |ac|^2 - (ab・ac)^2
        self.normal = cross  # 長さをそろえない法線（面の中の点は、これに沿って面へ下ろす）
        self.sliver = _sliver_mask(self.a, self.b, self.c, np.sqrt(den))
        self.ab_ac = _dot(self.ab, self.ac)
        self.bc_ab = _dot(self.bc, self.ab)
        self.ab2, self.ac2, self.bc2 = ab2, ac2, bc2
        # 0 で割らないように、長さ 0 の辺・つぶれた三角形では 0 を入れる（その答えは、あとで線分の計算で置き換える）
        self.inv_ab2 = _safe_inverse(ab2)
        self.inv_ac2 = _safe_inverse(ac2)
        self.inv_bc2 = _safe_inverse(bc2)
        self.inv_den = np.where(self.sliver, 0.0, _safe_inverse(den))

    def take(self, start: int, stop: int) -> dict[str, np.ndarray]:
        """三角形 start〜stop-1 の分の値"""
        return {name: getattr(self, name)[start:stop] for name in (
            "a", "b", "c", "ab", "ac", "bc", "normal", "ab2", "ac2", "bc2", "ab_ac", "bc_ab",
            "inv_ab2", "inv_ac2", "inv_bc2", "inv_den", "sliver")}


def _safe_inverse(values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    np.divide(1.0, values, out=result, where=values > 0.0)
    return result


def _differences(feature: np.ndarray, s: np.ndarray, t: np.ndarray, ap: np.ndarray, bp: np.ndarray,
                 cp: np.ndarray, ab: np.ndarray, ac: np.ndarray, bc: np.ndarray) -> np.ndarray:
    """「点 − いちばん近い点」のベクトル。場所の種類（feature）と係数 s・t から作る。

    頂点のときは s = t = 0 で、ap・bp・cp をそのまま使う（同じ頂点を共有する三角形どうしで、距離がぴったり同じになる）。
    - 辺 v0-v1（1）：p - (a + s ab)、辺 v1-v2（2）：p - (b + s bc)、辺 v2-v0（3）：p - (a + t ac)
    - 面の中（0）は、呼ぶ側（_closest_block）が、面へ垂直に下ろしたベクトルに置き換える
    """
    use_b = (feature == FEATURE_VERTEX_1) | (feature == FEATURE_EDGE_12)
    use_c = feature == FEATURE_VERTEX_2
    base = np.where(use_b[..., None], bp, np.where(use_c[..., None], cp, ap))
    first_edge = np.where((feature == FEATURE_EDGE_12)[..., None], bc, ab)
    return base - s[..., None] * first_edge - t[..., None] * ac


def _closest_block(points: np.ndarray, tri: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    """点の小分け × 三角形の小分けの、全部の組について、距離の2乗・場所の種類・係数を求める。

    Ericson「Real-Time Collision Detection」5.1.5 の、ボロノイ領域で場所を決める方法。
    ただし割り算の分母は、三角形だけで決まる値（辺の長さの2乗・|ab × ac|^2）を使い、点に関わる値の引き算を減らす。
    面の中のときは、重心座標（vb / |ab × ac|^2 など）を使わず、点から面へ垂直に下ろす。
    重心座標の誤差は、細い三角形では 1 / 高さ^2 で大きくなり、点が三角形の外へ飛ぶため。
    垂直に下ろせば、誤差は法線の向きの誤差（1 / 高さ）だけになる。
    """
    p = points[:, None, :]
    ab, ac, bc = tri["ab"][None], tri["ac"][None], tri["bc"][None]
    ap = p - tri["a"][None]
    bp = p - tri["b"][None]
    cp = p - tri["c"][None]
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    d3, d6 = _dot(ab, bp), _dot(ac, cp)
    bc_bp, bc_cp = _dot(bc, bp), _dot(bc, cp)
    ab2, ac2, bc2 = tri["ab2"][None], tri["ac2"][None], tri["bc2"][None]
    ab_ac, bc_ab = tri["ab_ac"][None], tri["bc_ab"][None]
    # 面に映した点の重心座標（の |ab × ac|^2 倍）。va・vb・vc がそれぞれ頂点 a・b・c の重み
    vc = ab2 * d2 - ab_ac * d1
    vb = ac2 * d1 - ab_ac * d2
    va = bc_ab * bc_bp - bc2 * d3

    in_a = (d1 <= 0.0) & (d2 <= 0.0)
    in_b = (d3 >= 0.0) & (bc_bp <= 0.0)
    in_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    in_c = (d6 >= 0.0) & (bc_cp >= 0.0)
    in_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    in_bc = (va <= 0.0) & (bc_bp >= 0.0) & (bc_cp <= 0.0)
    feature = np.select(
        [in_a, in_b, in_ab, in_c, in_ac, in_bc],
        [FEATURE_VERTEX_0, FEATURE_VERTEX_1, FEATURE_EDGE_01, FEATURE_VERTEX_2, FEATURE_EDGE_20, FEATURE_EDGE_12],
        default=FEATURE_FACE).astype(np.int8)
    # 係数は辺のときだけ使う（面の中は、下で垂直に下ろす。頂点は 0）
    s = np.select([feature == FEATURE_EDGE_01, feature == FEATURE_EDGE_12],
                  [d1 * tri["inv_ab2"][None], bc_bp * tri["inv_bc2"][None]], default=0.0)
    t = np.where(feature == FEATURE_EDGE_20, d2 * tri["inv_ac2"][None], 0.0)

    columns = np.flatnonzero(tri["sliver"])
    if columns.size:
        # 細い・つぶれた三角形は、3本の線分（v0-v1、v1-v2、v0-v2）のいちばん近い点にする
        sub = (slice(None), columns)
        f_seg, s_seg, t_seg = _segment_features(d1[sub], d2[sub], bc_bp[sub], tri["inv_ab2"][columns],
                                                tri["inv_ac2"][columns], tri["inv_bc2"][columns],
                                                ap[sub], bp[sub], cp[sub], ab[sub], ac[sub], bc[sub])
        feature[sub], s[sub], t[sub] = f_seg, s_seg, t_seg

    diff = _differences(feature, s, t, ap, bp, cp, ab, ac, bc)
    # 面の中：点 − 面の上の点 = 法線 n × (n・ap) / |n|^2（細い三角形でも、点が三角形の外へ飛ばない）。
    # 細い・つぶれた三角形は、上で線分にしたので、ここには来ない（inv_den も 0）
    normal = tri["normal"][None]
    along = _dot(normal, ap) * tri["inv_den"][None]
    diff = np.where((feature == FEATURE_FACE)[..., None], normal * along[..., None], diff)
    return _dot(diff, diff), feature, s, t


def _on_segment(param: np.ndarray, start: int, end: int, edge: int) -> tuple[np.ndarray, np.ndarray]:
    """線分の上の位置（0〜1 の係数）から、場所の種類と係数を決める。端なら頂点（係数 0）"""
    feature = np.where(param <= 0.0, start, np.where(param >= 1.0, end, edge)).astype(np.int8)
    return feature, np.where(feature == edge, param, 0.0)


def _segment_features(d1, d2, bc_bp, inv_ab2, inv_ac2, inv_bc2, ap, bp, cp, ab, ac, bc):
    """つぶれた三角形を3本の線分として、いちばん近い場所を選ぶ。

    長さ 0 の辺では係数が 0 になり、始まりの頂点になる。同じ距離なら v0-v1、v1-v2、v2-v0 の順で先の物。
    """
    zero = np.zeros_like(d1)
    f_ab, s_ab = _on_segment(d1 * inv_ab2, FEATURE_VERTEX_0, FEATURE_VERTEX_1, FEATURE_EDGE_01)
    f_bc, s_bc = _on_segment(bc_bp * inv_bc2, FEATURE_VERTEX_1, FEATURE_VERTEX_2, FEATURE_EDGE_12)
    f_ac, t_ac = _on_segment(d2 * inv_ac2, FEATURE_VERTEX_0, FEATURE_VERTEX_2, FEATURE_EDGE_20)
    best_f, best_s, best_t = f_ab, s_ab, zero
    diff = _differences(best_f, best_s, best_t, ap, bp, cp, ab, ac, bc)
    best_d2 = _dot(diff, diff)
    for f_k, s_k, t_k in ((f_bc, s_bc, zero), (f_ac, zero, t_ac)):
        diff = _differences(f_k, s_k, t_k, ap, bp, cp, ab, ac, bc)
        d2_k = _dot(diff, diff)
        better = d2_k < best_d2
        best_f = np.where(better, f_k, best_f)
        best_s = np.where(better, s_k, best_s)
        best_t = np.where(better, t_k, best_t)
        best_d2 = np.where(better, d2_k, best_d2)
    return best_f, best_s, best_t


def _empty_closest() -> ClosestResult:
    return ClosestResult(points=np.zeros((0, 3)), distances=np.zeros(0), triangles=np.zeros(0, dtype=np.int64),
                         features=np.zeros(0, dtype=np.int8))


def closest_points(points: Any, vertices: Any, triangles: Any, *,
                   chunk_size: int = DEFAULT_CHUNK_PAIRS) -> ClosestResult:
    """点ごとに、三角形の面の上で、いちばん近い点を求める（全部の三角形を調べる、正しい答え）。

    chunk_size: 一度に計算する「点と三角形の組」の数の上限（メモリは 1組 約 300 バイト）。
    同じ距離の三角形が2つ以上あるときは、いちばん小さい番号を選ぶ（距離が浮動小数点で同じときだけ）。
    細い・つぶれた三角形（面積 0 など）は、線分や点として扱うので、NaN にならない。
    点が空なら空の答えを返す。三角形が1つも無いのに点があるときは ValueError。
    """
    p = as_points(points, "points")
    v = as_points(vertices, "vertices")
    t = as_triangles(triangles, len(v))
    chunk = as_positive_int(chunk_size, "chunk_size")
    count = len(p)
    if count == 0:
        return _empty_closest()
    if len(t) == 0:
        raise ValueError("三角形が1つもありません（いちばん近い点を決められません）")

    data = _TriangleData(v, t)
    best_d2 = np.zeros(count)
    best_tri = np.zeros(count, dtype=np.int64)
    best_feature = np.zeros(count, dtype=np.int8)
    best_s = np.zeros(count)
    best_t = np.zeros(count)
    tri_block = min(len(t), chunk)
    point_block = max(1, chunk // tri_block)
    # 三角形は小さい番号の小分けから順に調べ、「より近い」ときだけ置き換える（同じ距離なら先の番号が残る）
    for t0 in range(0, len(t), tri_block):
        t1 = min(t0 + tri_block, len(t))
        tri = data.take(t0, t1)
        for p0 in range(0, count, point_block):
            p1 = min(p0 + point_block, count)
            d2, feature, s, tt = _closest_block(p[p0:p1], tri)
            column = np.argmin(d2, axis=1)  # 同じ値なら最初（小さい番号）
            rows = np.arange(p1 - p0)
            candidate = d2[rows, column]
            better = np.ones(p1 - p0, dtype=bool) if t0 == 0 else candidate < best_d2[p0:p1]
            target = np.arange(p0, p1)[better]
            best_d2[target] = candidate[better]
            best_tri[target] = column[better] + t0
            best_feature[target] = feature[rows, column][better]
            best_s[target] = s[rows, column][better]
            best_t[target] = tt[rows, column][better]

    return ClosestResult(points=_closest_positions(p, data, best_tri, best_feature, best_s, best_t),
                         distances=np.sqrt(best_d2), triangles=best_tri, features=best_feature)


def _closest_positions(p: np.ndarray, data: _TriangleData, tri: np.ndarray, feature: np.ndarray, s: np.ndarray,
                       tt: np.ndarray) -> np.ndarray:
    """選んだ三角形の上の点の座標。頂点のときは、頂点の座標そのもの。面の中のときは、点から面へ垂直に下ろした所"""
    a, b, c = data.a[tri], data.b[tri], data.c[tri]
    ab, ac, bc = data.ab[tri], data.ac[tri], data.bc[tri]
    normal = data.normal[tri]
    foot = p - normal * (_dot(normal, p - a) * data.inv_den[tri])[:, None]
    s_, t_ = s[:, None], tt[:, None]
    f = feature[:, None]
    return np.select(
        [f == FEATURE_VERTEX_0, f == FEATURE_VERTEX_1, f == FEATURE_VERTEX_2,
         f == FEATURE_EDGE_01, f == FEATURE_EDGE_12, f == FEATURE_EDGE_20],
        [a, b, c, a + s_ * ab, b + s_ * bc, a + t_ * ac],
        default=foot)


# MARK: - 符号つきの距離

def _edge_pseudo_normals(welded: np.ndarray, face_normals: np.ndarray,
                         vertex_count: int) -> tuple[np.ndarray, np.ndarray]:
    """辺ごとの擬似法線（その辺を使う面の法線の和）。

    返す物：三角形の辺（v0-v1、v1-v2、v2-v0）ごとの辺の番号 (M, 3) と、辺ごとの擬似法線 (E, 3)
    """
    first = welded
    second = welded[:, [1, 2, 0]]
    low = np.minimum(first, second).ravel()
    high = np.maximum(first, second).ravel()
    key = low * vertex_count + high
    unique_keys, inverse = np.unique(key, return_inverse=True)
    inverse = inverse.reshape(-1)
    normals = np.repeat(face_normals, 3, axis=0)
    sums = np.zeros((len(unique_keys), 3))
    for axis in range(3):
        sums[:, axis] = np.bincount(inverse, weights=normals[:, axis], minlength=len(unique_keys))
    return inverse.reshape(-1, 3), sums


def signed_distance(points: Any, vertices: Any, triangles: Any, *,
                    chunk_size: int = DEFAULT_CHUNK_PAIRS) -> np.ndarray:
    """符号つきの距離 (P,)：形の中は負、外は正、面の上は 0（メートル）。

    前提：形は閉じていて（穴が無い）、面の向きがそろって外を向いていること（mesh_check で確かめられる）。
    開いた形では、符号が意味を持つのは、いちばん近い点が面の中にあるときだけ（縁の近くでは当てにならない）。

    符号は、いちばん近い点での「擬似法線」との内積で決める（Baerentzen・Aanaes 2005）。
    - 面の中：その面の法線
    - 辺：その辺を使う2つの面の法線の和
    - 頂点：まわりの面の法線を、その頂点での角の大きさで重みを付けて足した物
    これで、いちばん近い点が辺や頂点のとき（とがった所の近く）も、符号が正しくなる。
    辺と頂点のつながりは、座標がまったく同じ頂点をまとめて調べる（UV などで分かれた頂点があっても、つながりが切れない）。

    擬似法線が決まらないとき（つぶれた面だけが集まる所、表と裏の面が重なった所）は、符号を決めずに NaN を返す
    （測れなかった物は緑にしない。めり込みの信号では灰になる）。
    """
    p = as_points(points, "points")
    v = as_points(vertices, "vertices")
    t = as_triangles(triangles, len(v))
    closest = closest_points(p, v, t, chunk_size=chunk_size)
    if len(p) == 0:
        return np.zeros(0)

    welded = weld_exact(v)[t]
    face_normals = _triangle_normals(v, t)
    vertex_pseudo = _angle_weighted_sums(v, welded, face_normals)
    edge_ids, edge_pseudo = _edge_pseudo_normals(welded, face_normals, len(v))

    feature = closest.features.astype(np.int64)
    tri = closest.triangles
    normal = np.zeros((len(p), 3))
    face = feature == FEATURE_FACE
    normal[face] = face_normals[tri[face]]
    edge = (feature >= FEATURE_EDGE_01) & (feature <= FEATURE_EDGE_20)
    normal[edge] = edge_pseudo[edge_ids[tri[edge], feature[edge] - FEATURE_EDGE_01]]
    vertex = feature >= FEATURE_VERTEX_0
    normal[vertex] = vertex_pseudo[welded[tri[vertex], feature[vertex] - FEATURE_VERTEX_0]]

    along = _dot(normal, p - closest.points)
    decided = (np.sqrt(_dot(normal, normal)) > PSEUDO_NORMAL_EPS) & (along != 0.0)
    sign = np.where(along > 0.0, 1.0, -1.0)
    result = np.where(decided, sign * closest.distances, np.nan)
    return np.where(closest.distances == 0.0, 0.0, result)


def penetration_depths(points: Any, vertices: Any, triangles: Any, *,
                       chunk_size: int = DEFAULT_CHUNK_PAIRS) -> np.ndarray:
    """めり込みの深さ (P,) = max(0, -符号つきの距離)。形の中の点だけが正になる。符号が決まらない点は NaN"""
    signed = signed_distance(points, vertices, triangles, chunk_size=chunk_size)
    return np.where(np.isnan(signed), np.nan, np.where(signed < 0.0, -signed, 0.0))


def gaps(points: Any, vertices: Any, triangles: Any, *, chunk_size: int = DEFAULT_CHUNK_PAIRS) -> np.ndarray:
    """すき間 (P,) = max(0, 符号つきの距離)。形の外の点だけが正になる。符号が決まらない点は NaN"""
    signed = signed_distance(points, vertices, triangles, chunk_size=chunk_size)
    return np.where(np.isnan(signed), np.nan, np.where(signed > 0.0, signed, 0.0))


# MARK: - 信号にまとめる

def worst_indices(values: np.ndarray, mask: np.ndarray, count: int = WORST_LIST_LENGTH) -> np.ndarray:
    """mask が真の所から、値の大きい順に count 個の番号 (int64)。同じ値なら番号の小さい順"""
    index = np.flatnonzero(mask)
    order = np.lexsort((index, -values[index]))
    return index[order][:count].astype(np.int64)


PENETRATION_NAME = "penetration_depth"
GAP_NAME = "cloth_gap"


def summarize_penetration(depths: Any, *, visible: Any = None) -> Measurement:
    """めり込みの深さを、1つの信号にまとめる（名前 "penetration_depth"、単位 "m"）。

    - 値は「見える点」の中でいちばん深いめり込み。基準は thresholds.PENETRATION_YELLOW・PENETRATION_RED。
    - visible（真偽の並び）を渡すと、見えない点のめり込みは数えない（ロードマップ：見える問題だけを数える）。
      見えない点は details に記録するだけ。visible が None なら、全部の点を見える点とする。
    - depths の NaN は「測れなかった点」。見える点に1つでもあれば、赤でないかぎり灰にする。
    - 点が無い、見える点が無い、見える点を全部測れなかったときは灰。
    - 負の深さ・無限大は ValueError。
    """
    d = as_values(depths, "depths", allow_nan=True)
    if (d < 0.0).any():
        raise ValueError("depths に負の値があります（めり込みの深さは 0 以上）")
    count = len(d)
    if count == 0:
        return Measurement.unmeasured(PENETRATION_NAME, "m", "点がありません")
    seen = np.ones(count, dtype=bool) if visible is None else as_mask(visible, count, "visible")
    measured = ~np.isnan(d)
    safe = np.where(measured, d, 0.0)
    penetrating = measured & (safe > 0.0)
    hidden = ~seen
    details: dict[str, Any] = {
        "point_count": count,
        "visible_count": int(seen.sum()),
        "penetrating_visible_count": int((penetrating & seen).sum()),
        "penetrating_hidden_count": int((penetrating & hidden).sum()),
        "over_yellow_visible_count": int((measured & seen & (safe >= thresholds.PENETRATION_YELLOW)).sum()),
        "over_red_visible_count": int((measured & seen & (safe >= thresholds.PENETRATION_RED)).sum()),
        "unmeasured_visible_count": int((~measured & seen).sum()),
        "unmeasured_hidden_count": int((~measured & hidden).sum()),
        "max_hidden_depth": float(safe[measured & hidden].max()) if (measured & hidden).any() else None,
        "worst_visible_indices": worst_indices(safe, penetrating & seen),
        "worst_hidden_indices": worst_indices(safe, penetrating & hidden),
    }
    if not seen.any():
        return Measurement.unmeasured(PENETRATION_NAME, "m", "見える点が1つもありません（見え方の計算を確かめてください）",
                                      details)
    usable = measured & seen
    if not usable.any():
        return Measurement.unmeasured(PENETRATION_NAME, "m", "見える点のめり込みを測れませんでした", details)
    value = float(safe[usable].max())
    signal = classify(value, red=thresholds.PENETRATION_RED, yellow=thresholds.PENETRATION_YELLOW)
    reason = ""
    if details["unmeasured_visible_count"]:
        signal = worst([signal, Signal.GRAY])
        reason = "測れなかった見える点があります（そこに深いめり込みが隠れているかもしれません）"
    return Measurement(PENETRATION_NAME, value, "m", signal, reason, details)


def summarize_gap(gap_values: Any) -> Measurement:
    """服と体のすき間を、1つの信号にまとめる（名前 "cloth_gap"、単位 "count"）。

    - 値は、すき間が thresholds.CLOTH_GAP_MIN〜CLOTH_GAP_MAX（ロードマップ R2：1〜4mm。両端を含む）の外にある点の数。
    - 1点でも外にあれば黄（仮。赤にするかどうかは、本物のモデルで決める）。
    - gap_values は gaps() の値（0 以上）でも、符号つきの距離（めり込みは負）でもよい。負の値は「近すぎる」に数える。
    - 体から離れていてよい所（スカートのすそなど）は、呼ぶ側で除いてから渡す。
    - NaN は「測れなかった点」。1つでもあれば灰（値が全部 NaN か、点が無いときも灰）。
    """
    g = as_values(gap_values, "gap_values", allow_nan=True)
    count = len(g)
    if count == 0:
        return Measurement.unmeasured(GAP_NAME, "count", "点がありません")
    measured = ~np.isnan(g)
    if not measured.any():
        return Measurement.unmeasured(GAP_NAME, "count", "すき間を測れた点がありません",
                                      {"point_count": count, "unmeasured_count": count})
    safe = np.where(measured, g, 0.0)
    too_close = measured & (safe < thresholds.CLOTH_GAP_MIN)
    too_far = measured & (safe > thresholds.CLOTH_GAP_MAX)
    outside = int(too_close.sum() + too_far.sum())
    values = g[measured]
    details: dict[str, Any] = {
        "point_count": count,
        "unmeasured_count": int((~measured).sum()),
        "too_close_count": int(too_close.sum()),
        "too_far_count": int(too_far.sum()),
        "too_close_indices": np.flatnonzero(too_close).astype(np.int64),
        "too_far_indices": np.flatnonzero(too_far).astype(np.int64),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(np.median(values)),
    }
    signal = Signal.YELLOW if outside else Signal.GREEN
    reasons = ["すき間が基準の外の点があります（基準は仮）"] if outside else []
    if details["unmeasured_count"]:
        signal = worst([signal, Signal.GRAY])
        reasons.append("すき間を測れなかった点があります")
    reason = "。".join(reasons)
    return Measurement(GAP_NAME, float(outside), "count", signal, reason, details)

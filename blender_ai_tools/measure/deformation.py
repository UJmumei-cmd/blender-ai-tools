"""A4 形のくずれと、版の比べ（bpy なし・numpy だけ）。

関節を曲げたときに形がくずれていないかを、休みの形（rest）と動いた形（posed）を比べて数字にする。
2つの形は、同じ頂点・同じ三角形の並びで渡す（頂点の番号どうしで比べる）。

- 辺が伸びる・縮む（edge_length_ratios → summarize_stretch）
- 面が裏返る（triangle_flips → summarize_flips）
- 細くやせる・ふくらむ（mesh_volume・volume_change_ratio → summarize_volume）
- とげ（spike_scores → summarize_spikes）
- 版の比べ：「変えてよい所」の外の変化（changed_vertices → summarize_changes）

関節を曲げると、腕などは丸ごと回る。丸ごと回っただけの所を「裏返った」「とげ」と数えないように、
裏返りととげは、頂点ごとに「まわりの点がどう回ったか」（まわりの回転）を求め、それを除いてから比べる。

決まり（CLAUDE.md 7章）：長さはメートル。点は (N, 3) の float64、三角形は (M, 3) の int64。
面は外から見て反時計回り（法線が外を向く）。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .. import thresholds
from ..thresholds import VOLUME_MIN_RATIO
from .geometry import WORST_LIST_LENGTH, as_array, as_mask, as_nonnegative_float, as_points, worst_indices
from .results import Measurement, Signal, classify, worst

# 体積の変わり方を測らない、休みの形の体積の小ささは thresholds.VOLUME_MIN_RATIO（仮）。上で読み込む

# MARK: - 計算のための数字（判定の基準ではない）

# まわりの点の広がりが「ほぼ1本の線（か1点）」とみなす割合。回転が決まらない。
# 広がりの行列の2番目の特異値が、1番目のこの割合以下なら、回転を決めない
ROTATION_RANK_RATIO = 1e-10
# まわりの点の広がりが「細い」とみなす割合（2番目の特異値 / 1番目の特異値）。
# 細い広がりでは、線のまわりの回転が、線からの小さなずれだけで決まり、少しの形の変化で大きく変わる
# （細い帯の先の点で、0.02mm 動いただけで 180 度回った答えになり、とげが赤になっていた）。
# これ以下の頂点は、その頂点も入れて求め直し、広がりのよいほうの回転を使う
THIN_RING_RATIO = 0.1
# 3つの頂点の回転で回した法線を足した物の長さ（1 の長さの法線を最大3つ足す）が、これ以下なら向きが決まらない
DIRECTION_EPS = 1e-9
# 一度に計算する頂点・三角形の数（3×3 の行列の並びで、1つあたり 72 バイト）
CHUNK_SIZE = 1 << 16
# まわりの点の広がりを足すとき、一度に扱う向きのある辺の数（1本あたり約 100 バイト）
EDGE_CHUNK_SIZE = 1 << 18
# details に入れる「いちばん悪い物」の数は、geometry.WORST_LIST_LENGTH と同じ（上で読み込む）

_INT64_MAX = np.iinfo(np.int64).max

CHANGES_NAME = "unexpected_changes"
STRETCH_NAME = "edge_stretch"
FLIPS_NAME = "flipped_faces"
VOLUME_NAME = "volume_loss"
SPIKES_NAME = "spike"


# MARK: - 入力の確かめ

def _as_point_pair(first: Any, second: Any, first_name: str, second_name: str) -> tuple[np.ndarray, np.ndarray]:
    """同じ頂点の並びの、2つの形。頂点の数が違うときは ValueError"""
    a = as_points(first, first_name)
    b = as_points(second, second_name)
    if a.shape != b.shape:
        raise ValueError(f"{first_name} と {second_name} の頂点の数が違います（{len(a)} と {len(b)}）")
    return a, b


def _as_indices(array_like: Any, width: int, vertex_count: int | None, name: str) -> np.ndarray:
    """頂点の番号の並びを (M, width) の int64 にする（三角形なら width=3、辺なら width=2）。

    整数でない（小数・真偽）、形が違う、番号が負か vertex_count 以上のときは ValueError。
    空の並びは、数の型なら (0, width) にする。
    """
    array = as_array(array_like, name)
    if array.dtype.kind == "b" or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} は整数の並びにしてください（いまの型：{array.dtype}）")
    if array.size == 0:
        if array.ndim == 1 or (array.ndim == 2 and array.shape[1] == width):
            return np.zeros((0, width), dtype=np.int64)
        raise ValueError(f"{name} は (M, {width}) の並びにしてください（いまの形：{array.shape}）")
    if array.dtype.kind == "f":
        raise ValueError(f"{name} は整数の並びにしてください（いまの型：{array.dtype}）")
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} は (M, {width}) の並びにしてください（いまの形：{array.shape}）")
    if array.dtype.kind == "u" and int(array.max()) > _INT64_MAX:
        raise ValueError(f"{name} の番号が大きすぎます")
    array = np.asarray(array, dtype=np.int64)
    if int(array.min()) < 0:
        raise ValueError(f"{name} に負の番号があります")
    if vertex_count is not None and int(array.max()) >= vertex_count:
        raise ValueError(f"{name} の番号が頂点の数（{vertex_count}）を超えています")
    return array


def _as_bool(value: Any, name: str) -> bool:
    """True か False だけを受け付ける（1・0 などの数は受け付けない）"""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} は True か False にしてください（いまの値：{value!r}）")
    return bool(value)


def _as_values(values: Any, name: str) -> np.ndarray:
    """1次元の float64 の並びにする。NaN は「測れなかった」としてそのまま通す。無限大・負の値は ValueError"""
    array = as_array(values, name)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} は数の並びにしてください（いまの型：{array.dtype}）")
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} は1次元の並びにしてください（いまの形：{array.shape}）")
    if np.isinf(array).any():
        raise ValueError(f"{name} に無限大が入っています")
    if (array < 0.0).any():
        raise ValueError(f"{name} に負の値があります（0 以上のはずです）")
    return array


# MARK: - 小さな計算

def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """最後の軸（x, y, z）の内積。足す順を決めておき、小分けの仕方で答えが変わらないようにする"""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _lengths(vectors: np.ndarray) -> np.ndarray:
    return np.sqrt(_dot(vectors, vectors))


def _triangle_cross(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """三角形ごとの (v1 - v0) × (v2 - v0)。向きは法線、長さは面積の2倍"""
    a = vertices[triangles[:, 0]]
    return np.cross(vertices[triangles[:, 1]] - a, vertices[triangles[:, 2]] - a)


def _worst_indices(values: np.ndarray, mask: np.ndarray, *, largest: bool = True) -> np.ndarray:
    """mask が真の所から、悪い順（largest なら大きい順、そうでなければ小さい順）に番号を返す。同じ値なら番号の小さい順。
    geometry.worst_indices と同じ決まり（小さい順は、値の符号を変えて渡す）"""
    return worst_indices(values if largest else -values, mask, WORST_LIST_LENGTH)


# MARK: - 版の比べ（変えてよい所の外の変化）

def displacement(before: Any, after: Any) -> np.ndarray:
    """頂点ごとの動いた距離 (N,)（メートル）。before と after の頂点の数が違うときは ValueError"""
    b, a = _as_point_pair(before, after, "before", "after")
    return _lengths(a - b)


def changed_vertices(before: Any, after: Any, *, tolerance: float = thresholds.REPRO_TOLERANCE_LENGTH,
                     allowed: Any = None) -> np.ndarray:
    """「変えてよい所」の外で動いた頂点の番号 (int64、小さい順)。

    - 動いた距離が tolerance より大きい（ちょうどは含まない）頂点を「変わった」とする。
      再現性の幅（thresholds.REPRO_TOLERANCE_LENGTH）より小さい差は、変化として数えない（CLAUDE.md 7章）。
    - allowed（真偽の並び (N,)）が真の頂点は、変えてよい所なので数えない。None なら、どこも変えてはいけない。
    """
    d = displacement(before, after)
    limit = as_nonnegative_float(tolerance, "tolerance")
    moved = d > limit
    if allowed is not None:
        moved &= ~as_mask(allowed, len(d), "allowed")
    return np.flatnonzero(moved).astype(np.int64)


def summarize_changes(before: Any, after: Any, *, tolerance: float = thresholds.REPRO_TOLERANCE_LENGTH,
                      allowed: Any = None) -> Measurement:
    """版の比べを、1つの信号にまとめる（名前 "unexpected_changes"、単位 "count"）。

    - 値は「変えてよい所」の外で動いた頂点の数（changed_vertices と同じ数え方）。1つでもあれば赤
      （ロードマップ段階1 基準5：変えてよい所の外の変化が0）。
    - 頂点の数が違う（形のつながりが変わった）ときは、点ごとに比べられないので灰。
    - 頂点が無いときも灰（比べた物が無い）。
    - 並びの形が (N, 3) でない、NaN があるときは ValueError。
    """
    b = as_points(before, "before")
    a = as_points(after, "after")
    limit = as_nonnegative_float(tolerance, "tolerance")
    if len(b) != len(a):
        return Measurement.unmeasured(CHANGES_NAME, "count", "頂点の数が違うので、頂点ごとに比べられません",
                                      {"before_count": len(b), "after_count": len(a)})
    if len(b) == 0:
        return Measurement.unmeasured(CHANGES_NAME, "count", "頂点がありません")
    free = np.zeros(len(b), dtype=bool) if allowed is None else as_mask(allowed, len(b), "allowed")
    d = _lengths(a - b)
    moved = d > limit
    unexpected = moved & ~free
    count = int(unexpected.sum())
    details: dict[str, Any] = {
        "vertex_count": len(b),
        "tolerance": limit,
        "changed_indices": np.flatnonzero(unexpected).astype(np.int64),
        "allowed_vertex_count": int(free.sum()),
        "allowed_changed_count": int((moved & free).sum()),
        "max_displacement": float(d.max()),
        "max_unexpected_displacement": float(d[unexpected].max()) if count else None,
        "worst_indices": _worst_indices(d, unexpected),
    }
    if count:
        return Measurement(CHANGES_NAME, float(count), "count", Signal.RED, "変えてよい所の外で動いた頂点があります",
                           details)
    return Measurement(CHANGES_NAME, 0.0, "count", Signal.GREEN, "", details)


# MARK: - 辺の伸び縮み

def unique_edges(triangles: Any) -> np.ndarray:
    """三角形の辺を、重なり無しで (E, 2) の int64 にする。

    各行は「小さい番号, 大きい番号」。行は (1列目, 2列目) の小さい順に並ぶ（いつも同じ順）。
    同じ頂点どうしの辺（同じ頂点を2回使う、つぶれた三角形の辺）は入れない。
    """
    t = _as_indices(triangles, 3, None, "triangles")
    if len(t) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    start = t.ravel()
    end = t[:, [1, 2, 0]].ravel()
    low = np.minimum(start, end)
    high = np.maximum(start, end)
    keep = low != high
    low, high = low[keep], high[keep]
    order = np.lexsort((high, low))
    low, high = low[order], high[order]
    first = np.ones(len(low), dtype=bool)
    first[1:] = (low[1:] != low[:-1]) | (high[1:] != high[:-1])
    return np.stack([low[first], high[first]], axis=1)


def edge_length_ratios(rest: Any, posed: Any, edges: Any, *,
                       min_rest_length: float = thresholds.REPRO_TOLERANCE_LENGTH) -> np.ndarray:
    """辺ごとの「動いた形の長さ / 休みの形の長さ」(E,)。1 より大きいと伸びた、小さいと縮んだ。

    休みの形の長さが min_rest_length 以下（ほぼ 0）の辺は、割合が決まらないので NaN。
    既定は再現性の幅（thresholds.REPRO_TOLERANCE_LENGTH）：これより短い辺の長さは、測るたびの差に埋もれる。
    """
    r, p = _as_point_pair(rest, posed, "rest", "posed")
    e = _as_indices(edges, 2, len(r), "edges")
    limit = as_nonnegative_float(min_rest_length, "min_rest_length")
    rest_length = _lengths(r[e[:, 1]] - r[e[:, 0]])
    posed_length = _lengths(p[e[:, 1]] - p[e[:, 0]])
    ratios = np.full(len(e), np.nan)
    usable = rest_length > limit
    ratios[usable] = posed_length[usable] / rest_length[usable]
    return ratios


# MARK: - まわりの回転

class _Neighborhood:
    """頂点ごとの、まわりの点（辺でつながった点）の情報。

    degree: まわりの点の数 (N,)
    rest_center・posed_center: まわりの点の平均の位置 (N, 3)（まわりの点が無い頂点は 0）
    rotations: まわりの点の、休みの形から動いた形への回転 (N, 3, 3)。決まらない頂点は単位行列
    determined: 回転が決まったか (N,)
    spread: 回転を求めた点の広がりの割合（2番目の特異値 / 1番目の特異値）(N,)。小さいほど、回転があいまい
    """

    def __init__(self, rest: np.ndarray, posed: np.ndarray, edges: np.ndarray) -> None:
        count = len(rest)
        # 向きのある辺（両向き）：source の頂点から見た、まわりの点 target
        self.source = np.concatenate([edges[:, 0], edges[:, 1]])
        self.target = np.concatenate([edges[:, 1], edges[:, 0]])
        self.degree = np.bincount(self.source, minlength=count).astype(np.int64)
        has = self.degree > 0
        self.rest_center = self._neighbor_mean(rest, has)
        self.posed_center = self._neighbor_mean(posed, has)
        # まず、まわりの点だけで回転を求める（その頂点が自分で動いた分に、回転が引きずられないように）
        covariance = self._covariance(rest, posed, self.rest_center, self.posed_center, self.source, self.target)
        self.rotations, self.determined, self.spread = _rotations_from_covariance(covariance)
        self.determined &= has
        # まわりの点が1本の線か、細く並ぶ頂点（三角形1つだけの角、細い帯の先の点など）は、
        # その頂点も入れて求め直し、広がりのよいほうを使う
        retry = has & (self.spread <= THIN_RING_RATIO)
        if retry.any():
            self._retry_with_self(rest, posed, retry)

    def _neighbor_mean(self, points: np.ndarray, has: np.ndarray) -> np.ndarray:
        count = len(points)
        center = np.zeros((count, 3))
        for axis in range(3):
            total = np.bincount(self.source, weights=points[self.target, axis], minlength=count)
            center[has, axis] = total[has] / self.degree[has]
        return center

    @staticmethod
    def _covariance(rest: np.ndarray, posed: np.ndarray, rest_center: np.ndarray, posed_center: np.ndarray,
                    source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """頂点ごとの H = Σ (休みの点 - 休みの中心)(動いた点 - 動いた中心)ᵀ (N, 3, 3)。

        中心を先に引いてから掛けるので、モデルが原点から遠くても、けた落ちしない。
        辺を EDGE_CHUNK_SIZE 本ずつ足していく（メモリが増えすぎないように）。
        """
        count = len(rest)
        covariance = np.zeros((count, 3, 3))
        for start in range(0, len(source), EDGE_CHUNK_SIZE):
            here = source[start:start + EDGE_CHUNK_SIZE]
            there = target[start:start + EDGE_CHUNK_SIZE]
            x = rest[there] - rest_center[here]
            y = posed[there] - posed_center[here]
            for i in range(3):
                for j in range(3):
                    covariance[:, i, j] += np.bincount(here, weights=x[:, i] * y[:, j], minlength=count)
        return covariance

    def _retry_with_self(self, rest: np.ndarray, posed: np.ndarray, retry: np.ndarray) -> None:
        count = len(rest)
        chosen = retry[self.source]
        source = np.concatenate([self.source[chosen], np.flatnonzero(retry)])
        target = np.concatenate([self.target[chosen], np.flatnonzero(retry)])
        size = np.bincount(source, minlength=count)
        rest_center = np.zeros((count, 3))
        posed_center = np.zeros((count, 3))
        for axis in range(3):
            rest_center[retry, axis] = (np.bincount(source, weights=rest[target, axis], minlength=count)[retry]
                                        / size[retry])
            posed_center[retry, axis] = (np.bincount(source, weights=posed[target, axis], minlength=count)[retry]
                                         / size[retry])
        covariance = self._covariance(rest, posed, rest_center, posed_center, source, target)
        rotations, determined, spread = _rotations_from_covariance(covariance[retry])
        # 広がりがよくなった頂点だけ、求め直した回転に替える（まわりの点だけで十分に広い・
        # 頂点を入れても広がらない頂点は、まわりの点だけの回転のまま）
        better = spread > self.spread[retry]
        index = np.flatnonzero(retry)[better]
        self.rotations[index] = rotations[better]
        self.determined[index] = determined[better]
        self.spread[index] = spread[better]


def _rotations_from_covariance(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """H (n, 3, 3) から、休みの形を動いた形にいちばんよく重ねる回転（鏡うつしは含まない）を求める（Kabsch の方法）。

    H = U S Vᵀ のとき、回転は V diag(1, 1, d) Uᵀ（d は回転の行列式を +1 にする符号）。
    広がりが1本の線か1点のとき（2番目の特異値がほぼ 0）は、回転が決まらないので、単位行列と「決まらない」を返す。
    返す物は（回転 (n, 3, 3)、決まったか (n,)、広がりの割合 s₁ / s₀ (n,)。1番目の特異値が 0 なら 0）。
    """
    count = len(covariance)
    rotations = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    determined = np.zeros(count, dtype=bool)
    spread = np.zeros(count)
    for start in range(0, count, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, count)
        u, s, vt = np.linalg.svd(covariance[start:stop])
        sign = np.where(np.linalg.det(u) * np.linalg.det(vt) < 0.0, -1.0, 1.0)
        vt[:, 2, :] *= sign[:, None]
        block = np.matmul(np.swapaxes(vt, 1, 2), np.swapaxes(u, 1, 2))
        positive = s[:, 0] > 0.0
        good = positive & (s[:, 1] > ROTATION_RANK_RATIO * s[:, 0])
        rotations[start:stop][good] = block[good]
        determined[start:stop] = good
        spread[start:stop][positive] = s[positive, 1] / s[positive, 0]
    return rotations, determined, spread


# MARK: - 面の裏返り

def triangle_flips(rest: Any, posed: Any, triangles: Any, *, local_rotation: bool = True,
                   degenerate_area: float = thresholds.DEGENERATE_AREA) -> tuple[np.ndarray, np.ndarray]:
    """面が裏返った三角形を調べる。返す物は (flipped, degenerate)。どちらも (M,) の真偽。

    flipped: 裏返った三角形。動いた形の法線と、比べる向きの内積が負（ちょうど直角は裏返りにしない）。
    degenerate: 向きを比べられなかった三角形。休みか動いた形で面積が degenerate_area 以下（同じ頂点を2回使う物を含む）、
        または、まわりの回転が決まらない物。これらは flipped にしない（別に数える）。

    比べる向き：
    - local_rotation=True（既定）：休みの法線を、3つの頂点の「まわりの回転」で回した物（の和）。
      腕を曲げて、面がまわりごと丸ごと回っただけなら裏返りにしない。まわりに対して裏返った面だけを数える。
    - local_rotation=False：休みの法線そのもの（回転を除かない）。同じ姿勢の2つの形を比べるとき用。

    注意：平らな所で、まわりごと鏡うつしになった物は、180度回った物と見分けが付かない（点の位置が同じになる）。
    まわりに裏返っていない面がある所だけを見つけられる。形全体の裏返り（中と外が入れ替わる）は、体積の符号で分かる。
    """
    r, p = _as_point_pair(rest, posed, "rest", "posed")
    t = _as_indices(triangles, 3, len(r), "triangles")
    use_rotation = _as_bool(local_rotation, "local_rotation")
    limit = as_nonnegative_float(degenerate_area, "degenerate_area")
    rest_cross = _triangle_cross(r, t)
    posed_cross = _triangle_cross(p, t)
    rest_double_area = _lengths(rest_cross)
    degenerate = (0.5 * rest_double_area <= limit) | (0.5 * _lengths(posed_cross) <= limit)
    if not use_rotation:
        reference = rest_cross
    else:
        neighborhood = _Neighborhood(r, p, unique_edges(t))
        rest_normals = np.zeros_like(rest_cross)
        good = ~degenerate
        rest_normals[good] = rest_cross[good] / rest_double_area[good, None]
        reference = np.zeros_like(rest_cross)
        for start in range(0, len(t), CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, len(t))
            for corner in range(3):
                vertex = t[start:stop, corner]
                rotated = np.einsum("nij,nj->ni", neighborhood.rotations[vertex], rest_normals[start:stop])
                reference[start:stop] += np.where(neighborhood.determined[vertex, None], rotated, 0.0)
        degenerate |= _lengths(reference) <= DIRECTION_EPS
    flipped = ~degenerate & (_dot(reference, posed_cross) < 0.0)
    return flipped, degenerate


# MARK: - 体積

def mesh_volume(vertices: Any, triangles: Any) -> float:
    """符号つきの体積（立方メートル）。閉じた形で、面が外向きなら正、中と外が入れ替わっていれば負。

    発散定理：三角形ごとの四面体（基準の点・v0・v1・v2）の体積 (a · (b × c)) / 6 を足す。
    基準の点は、三角形が使う頂点の平均にする（原点から遠いモデルでも、けた落ちしない）。
    閉じていない形では、値が基準の点によって変わるので、体積としての意味は無い（volume_change_ratio は None を返す）。
    三角形が無いときは 0。
    """
    v = as_points(vertices, "vertices")
    t = _as_indices(triangles, 3, len(v), "triangles")
    if len(t) == 0:
        return 0.0
    center = v[np.unique(t)].mean(axis=0)
    a, b, c = (v[t[:, k]] - center for k in range(3))
    return float(np.sum(_dot(a, np.cross(b, c))) / 6.0)


def _is_closed(triangles: np.ndarray) -> bool:
    """三角形が、穴が無く、面の向きがそろった閉じた形か。

    どの辺も、向き a → b で使われた数と、逆向き b → a で使われた数が同じなら、閉じている
    （3つ以上の面が使う辺も、行きと帰りの数が同じならよい）。このときだけ、体積が基準の点によらずに決まる。
    同じ頂点どうしの辺（つぶれた三角形）は数えない。
    """
    start = triangles.ravel()
    end = triangles[:, [1, 2, 0]].ravel()
    keep = start != end
    start, end = start[keep], end[keep]
    if len(start) == 0:
        return True
    low = np.minimum(start, end)
    high = np.maximum(start, end)
    direction = np.where(start < end, 1, -1).astype(np.int64)
    order = np.lexsort((high, low))
    low, high, direction = low[order], high[order], direction[order]
    first = np.ones(len(low), dtype=bool)
    first[1:] = (low[1:] != low[:-1]) | (high[1:] != high[:-1])
    return not np.add.reduceat(direction, np.flatnonzero(first)).any()


def volume_change_ratio(rest: Any, posed: Any, triangles: Any) -> float | None:
    """体積の変わり方 (動いた形の体積 - 休みの体積) / 休みの体積。負なら減った（やせた）、正なら増えた（ふくらんだ）。

    割合が決まらないときは None：
    - 閉じていない形（穴の縁の辺がある、面の向きがそろっていない）。体積が基準の点で変わり、体積の意味が無い。
    - 休みの体積が小さすぎる（|体積| <= VOLUME_MIN_RATIO × 休みの表面積^1.5）。
    -1 より小さいときは、動いた形の中と外が入れ替わっている。
    """
    r, p = _as_point_pair(rest, posed, "rest", "posed")
    t = _as_indices(triangles, 3, len(r), "triangles")
    if not _is_closed(t):
        return None
    rest_volume = mesh_volume(r, t)
    rest_area = float(np.sum(_lengths(_triangle_cross(r, t)))) * 0.5
    if abs(rest_volume) <= VOLUME_MIN_RATIO * rest_area ** 1.5:
        return None
    return (mesh_volume(p, t) - rest_volume) / rest_volume


# MARK: - とげ

def spike_scores(rest: Any, posed: Any, triangles: Any) -> np.ndarray:
    """頂点ごとの、とげの強さ (N,)。

    ラプラシアン L = (まわりの点の平均) - (その頂点) は、頂点がまわりからどれだけ出っ張っているかを表す。
    強さ = |L(動いた形) - R L(休みの形)| / (休みの形での、その頂点の辺の長さの平均)。
    R は、まわりの点の回転（腕が丸ごと回っただけなら、R で回すと L が重なり、強さは 0 になる）。
    まわりの点が動かず、その頂点だけが d 動いたときは、強さ = d / 辺の長さの平均。

    まわりの点が細く並ぶ頂点（三角形1つだけの角、細い帯の先の点など。THIN_RING_RATIO）は、その頂点も入れて
    回転を求める。その頂点が自分で動いた分の一部が回転に吸われるので、強さは小さめに出る
    （例：帯の先の点を、辺の長さの平均だけ面の外へ引くと、約 0.44）。
    どの三角形にも使われていない頂点、まわりの回転が決まらない頂点、辺の長さの平均が 0 の頂点は NaN。
    """
    r, p = _as_point_pair(rest, posed, "rest", "posed")
    t = _as_indices(triangles, 3, len(r), "triangles")
    edges = unique_edges(t)
    neighborhood = _Neighborhood(r, p, edges)
    count = len(r)
    degree = neighborhood.degree
    edge_length = _lengths(r[edges[:, 1]] - r[edges[:, 0]])
    total_length = (np.bincount(edges[:, 0], weights=edge_length, minlength=count)
                    + np.bincount(edges[:, 1], weights=edge_length, minlength=count))
    usable = neighborhood.determined & (degree > 0) & (total_length > 0.0)
    scores = np.full(count, np.nan)
    if not usable.any():
        return scores
    index = np.flatnonzero(usable)
    rest_laplacian = neighborhood.rest_center[index] - r[index]
    posed_laplacian = neighborhood.posed_center[index] - p[index]
    difference = posed_laplacian - np.einsum("nij,nj->ni", neighborhood.rotations[index], rest_laplacian)
    mean_length = total_length[index] / degree[index]
    scores[index] = _lengths(difference) / mean_length
    return scores


# MARK: - 信号にまとめる

def summarize_stretch(ratios: Any) -> Measurement:
    """辺の伸び縮みを、1つの信号にまとめる（名前 "edge_stretch"、単位 "ratio"）。

    - 伸び：いちばん大きい割合を thresholds.EDGE_STRETCH_YELLOW・EDGE_STRETCH_RED と比べる（大きいほど悪い）。
    - 縮み：いちばん小さい割合を thresholds.EDGE_SHRINK_YELLOW・EDGE_SHRINK_RED と比べる（小さいほど悪い）。
    - 信号は、2つの重いほう。値は、信号を決めたほうの割合（同じ信号なら、1 から割合で遠いほう。同じなら伸び）。
      details["value_kind"] は "stretch" か "shrink"。
    - NaN は「測れなかった辺」（休みの長さがほぼ 0）。数は details に入れる。
      測れなかった辺が1つでもあれば、赤でないかぎり灰にする（そこに伸び縮みが隠れているかもしれない。
      ほかの測る道具と同じ決まり：測れなかった物は緑にしない）。値は、測れた辺から決めたまま。
      辺が無い、全部 NaN のときも灰。負の値・無限大は ValueError。
    """
    values = _as_values(ratios, "ratios")
    count = len(values)
    if count == 0:
        return Measurement.unmeasured(STRETCH_NAME, "ratio", "辺がありません")
    measured = ~np.isnan(values)
    unmeasured = int(count - measured.sum())
    if not measured.any():
        return Measurement.unmeasured(STRETCH_NAME, "ratio", "伸び縮みを測れた辺がありません（休みの長さがほぼ 0）",
                                      {"edge_count": count, "unmeasured_count": unmeasured})
    high = np.where(measured, values, -np.inf)
    low = np.where(measured, values, np.inf)
    largest = float(high.max())
    smallest = float(low.min())
    stretch_signal = classify(largest, red=thresholds.EDGE_STRETCH_RED, yellow=thresholds.EDGE_STRETCH_YELLOW)
    shrink_signal = classify(smallest, red=thresholds.EDGE_SHRINK_RED, yellow=thresholds.EDGE_SHRINK_YELLOW,
                             higher_is_worse=False)
    if shrink_signal != stretch_signal:
        use_shrink = shrink_signal > stretch_signal
    else:
        use_shrink = smallest == 0.0 or -math.log(smallest) > math.log(largest)
    details: dict[str, Any] = {
        "edge_count": count,
        "unmeasured_count": unmeasured,
        "max_ratio": largest,
        "min_ratio": smallest,
        "value_kind": "shrink" if use_shrink else "stretch",
        "stretch_signal": stretch_signal,
        "shrink_signal": shrink_signal,
        "stretch_yellow_count": int((high >= thresholds.EDGE_STRETCH_YELLOW).sum()),
        "stretch_red_count": int((high >= thresholds.EDGE_STRETCH_RED).sum()),
        "shrink_yellow_count": int((low <= thresholds.EDGE_SHRINK_YELLOW).sum()),
        "shrink_red_count": int((low <= thresholds.EDGE_SHRINK_RED).sum()),
        "worst_stretch_indices": _worst_indices(high, measured & (high > 1.0)),
        "worst_shrink_indices": _worst_indices(low, measured & (low < 1.0), largest=False),
    }
    signal = worst([stretch_signal, shrink_signal])
    reason = ""
    if unmeasured:
        signal = worst([signal, Signal.GRAY])
        reason = "休みの長さがほぼ 0 で、測れなかった辺があります（そこに伸び縮みが隠れているかもしれません）"
    return Measurement(STRETCH_NAME, smallest if use_shrink else largest, "ratio", signal, reason, details)


def summarize_flips(flipped: Any, degenerate: Any = None) -> Measurement:
    """面の裏返りを、1つの信号にまとめる（名前 "flipped_faces"、単位 "count"）。

    - 値は裏返った三角形の数。1つでもあれば赤。
    - degenerate（triangle_flips の2つ目）は、向きを比べられなかった三角形。数は details に入れる。
      裏返りが無く、比べられなかった三角形が1つでもあれば灰（そこに裏返りが隠れているかもしれない。
      ほかの測る道具と同じ決まり：測れなかった物は緑にしない）。値は 0（見つかった裏返りの数）のまま。
      つぶれた三角形は mesh_check.summarize_degenerate でも見つかるので、先に直すと灰が消える。
    - 三角形が無い、全部比べられなかったときも灰。
    - 同じ三角形が flipped と degenerate の両方で真のときは ValueError（triangle_flips はそうしない）。
    """
    flips = as_array(flipped, "flipped")
    if flips.dtype.kind != "b" or flips.ndim != 1:
        raise ValueError(f"flipped は1次元の真偽（bool）の並びにしてください（いまの型・形：{flips.dtype}・{flips.shape}）")
    count = len(flips)
    skipped = np.zeros(count, dtype=bool) if degenerate is None else as_mask(degenerate, count, "degenerate")
    if (flips & skipped).any():
        raise ValueError("flipped と degenerate の両方で真の三角形があります")
    if count == 0:
        return Measurement.unmeasured(FLIPS_NAME, "count", "三角形がありません")
    details: dict[str, Any] = {
        "triangle_count": count,
        "degenerate_count": int(skipped.sum()),
        "flipped_indices": np.flatnonzero(flips).astype(np.int64),
    }
    if skipped.all():
        return Measurement.unmeasured(FLIPS_NAME, "count", "向きを比べられる三角形がありません（全部つぶれています）",
                                      details)
    flipped_count = int(flips.sum())
    skipped_reason = "つぶれていて、向きを比べられなかった三角形があります"
    if flipped_count:
        reason = "裏返った面があります"
        if details["degenerate_count"]:
            reason += f"（{skipped_reason}）"
        return Measurement(FLIPS_NAME, float(flipped_count), "count", Signal.RED, reason, details)
    if details["degenerate_count"]:
        return Measurement(FLIPS_NAME, 0.0, "count", Signal.GRAY,
                           f"{skipped_reason}（そこに裏返りが隠れているかもしれません）", details)
    return Measurement(FLIPS_NAME, 0.0, "count", Signal.GREEN, "", details)


def summarize_volume(ratio: float | None) -> Measurement:
    """体積の変わり方（volume_change_ratio の値）を、1つの信号にまとめる（名前 "volume_loss"、単位 "ratio"）。

    - 値は減った割合（loss = max(0, -ratio)）。基準は thresholds.VOLUME_LOSS_YELLOW・VOLUME_LOSS_RED。
    - 増えた分（ふくらみ）は信号にせず、details["gain"] に入れるだけ（基準がまだ無い）。
    - ratio が None・NaN のときは灰。真偽・無限大は ValueError。
    """
    if ratio is None or (isinstance(ratio, (float, np.floating)) and math.isnan(ratio)):
        return Measurement.unmeasured(VOLUME_NAME, "ratio",
                                      "体積の変わり方を測れませんでした（休みの体積がほぼ 0 か、閉じていない形）")
    if isinstance(ratio, (bool, np.bool_)) or not isinstance(ratio, (int, float, np.integer, np.floating)):
        raise ValueError(f"ratio は数か None にしてください（いまの値：{ratio!r}）")
    value = float(ratio)
    if not math.isfinite(value):
        raise ValueError(f"ratio に無限大が入っています（いまの値：{ratio!r}）")
    loss = max(0.0, -value)
    details: dict[str, Any] = {"ratio": value, "loss": loss, "gain": max(0.0, value), "inside_out": value < -1.0}
    reason = "動いた形の中と外が入れ替わっています" if value < -1.0 else ""
    return Measurement.from_value(VOLUME_NAME, loss, "ratio", red=thresholds.VOLUME_LOSS_RED,
                                  yellow=thresholds.VOLUME_LOSS_YELLOW, reason=reason, details=details)


def summarize_spikes(scores: Any) -> Measurement:
    """とげの強さ（spike_scores の値）を、1つの信号にまとめる（名前 "spike"、単位 "ratio"）。

    - 値は NaN を除いた中で、いちばん強いとげ。基準は thresholds.SPIKE_YELLOW・SPIKE_RED。
    - NaN は「測れなかった頂点」（どの面にも使われていない頂点など）。数は details に入れる。
      測れなかった頂点が1つでもあれば、赤でないかぎり灰にする（ほかの測る道具と同じ決まり：測れなかった物は緑にしない）。
      値は、測れた頂点から決めたまま。どの面にも使われていない頂点は、呼ぶ側で除いてから渡すとよい
      （mesh_check.unused_vertices で見つかる）。
    - 頂点が無い、全部 NaN のときも灰。負の値・無限大は ValueError。
    """
    values = _as_values(scores, "scores")
    count = len(values)
    if count == 0:
        return Measurement.unmeasured(SPIKES_NAME, "ratio", "頂点がありません")
    measured = ~np.isnan(values)
    safe = np.where(measured, values, 0.0)
    details: dict[str, Any] = {
        "vertex_count": count,
        "unmeasured_count": int(count - measured.sum()),
        "yellow_count": int((safe >= thresholds.SPIKE_YELLOW).sum()),
        "red_count": int((safe >= thresholds.SPIKE_RED).sum()),
        "worst_indices": _worst_indices(safe, measured & (safe > 0.0)),
    }
    if not measured.any():
        return Measurement.unmeasured(SPIKES_NAME, "ratio", "とげを測れた頂点がありません", details)
    value = float(safe[measured].max())
    signal = classify(value, red=thresholds.SPIKE_RED, yellow=thresholds.SPIKE_YELLOW)
    reason = ""
    if details["unmeasured_count"]:
        signal = worst([signal, Signal.GRAY])
        reason = "とげを測れなかった頂点があります（どの面にも使われていない頂点など。そこにとげが隠れているかもしれません）"
    return Measurement(SPIKES_NAME, value, "ratio", signal, reason, details)

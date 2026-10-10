"""A4 スキンのウェイト（bpy なし・numpy だけ）。

ウェイトは、頂点ごとに (N, K) の骨の番号（空きは -1）と (N, K) の重み（空きは 0）の2つの並びで持つ（CLAUDE.md 7章）。
K は1つの頂点が持てる枠の数。骨の番号は、アーマチュアの骨の並びの番号（0 から）。

- 点検（check・summarize）：合計が 1 か、1点に効く骨が4本までか、どの骨にも付いていない点が無いか、
  負の重み、同じ骨が2回入っていないか。
- 安全な直し（normalize_and_limit）：「ウェイトを合計1にして1点4本までにする」（CLAUDE.md 2章）。
  すでに正しい頂点は動かさない。
- 形の計算（linear_blend_skinning）：骨の行列で頂点を動かす（重みは渡されたまま使う）。

形を動かす骨（deform）：準標準のモデルでは、脚のウェイトが 足D・ひざD・足首D に乗っていることが多く、
足・ひざ・足首 は IK と VMD のための骨（CLAUDE.md 3章）。数えるのは、形を動かす骨（Blender の use_deform）だけにする。
deform は (B,) の真偽の並びで、骨の番号で引く。None なら全部の骨が形を動かす。
形を動かさない骨の重みは、数えず、直さず、そのまま残す（convert_blend.py も、use_deform の骨だけを見る）。
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import thresholds
from .geometry import as_array, as_nonnegative_float, as_points
from .results import Measurement, Signal

# MARK: - 計算のための数字（判定の基準ではない）

# 骨の行列の下の行が (0, 0, 0, 1) からずれてよい幅。行列を転置して渡すまちがいを見つけるため
AFFINE_TOLERANCE = 1e-9
# 一度に計算する頂点の数（点検・直し・linear_blend_skinning。1頂点あたり、枠の数 K × 約 100 バイト）
CHUNK_SIZE = 1 << 16

_INT64_MAX = np.iinfo(np.int64).max

# check の返す辞書の鍵（この順に並べる）
CHECK_KEYS = ("unweighted", "over_limit", "not_normalized", "negative", "duplicate_ids")

OVER_LIMIT_NAME = "bone_influences_over_limit"
UNWEIGHTED_NAME = "unweighted_vertices"
NOT_NORMALIZED_NAME = "weights_not_normalized"
NEGATIVE_NAME = "negative_weights"
DUPLICATE_NAME = "duplicate_bone_ids"


# MARK: - 入力の確かめ

def _as_id_matrix(value: Any, name: str) -> np.ndarray:
    """骨の番号の (N, K) の並びを int64 にする。整数でない（小数・真偽）と ValueError。空の並びは数の型なら受け付ける"""
    array = as_array(value, name)
    if array.dtype.kind == "b" or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} は整数の並びにしてください（いまの型：{array.dtype}）")
    if array.ndim != 2:
        raise ValueError(f"{name} は (N, K) の並びにしてください（いまの形：{array.shape}）")
    if array.size == 0:
        return np.zeros(array.shape, dtype=np.int64)
    if array.dtype.kind == "f":
        raise ValueError(f"{name} は整数の並びにしてください（いまの型：{array.dtype}）")
    if array.dtype.kind == "u" and int(array.max()) > _INT64_MAX:
        raise ValueError(f"{name} の番号が大きすぎます")
    return np.asarray(array, dtype=np.int64)


def _as_weight_matrix(value: Any, name: str) -> np.ndarray:
    """重みの (N, K) の並びを float64 にする。数でない（真偽など）と ValueError"""
    array = as_array(value, name)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} は数の並びにしてください（いまの型：{array.dtype}）")
    if array.ndim != 2:
        raise ValueError(f"{name} は (N, K) の並びにしてください（いまの形：{array.shape}）")
    return np.asarray(array, dtype=np.float64)


def _as_count(value: Any, name: str, minimum: int) -> int:
    """minimum 以上の整数にする。真偽（True・False）は整数として受け付けない"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} は {minimum} 以上の整数にしてください（いまの値：{value!r}）")
    return int(value)


def _as_matrices(matrices: Any) -> np.ndarray:
    """骨の行列 (B, 4, 4) の float64 にする。下の行が (0, 0, 0, 1) でない（転置した行列など）と ValueError"""
    array = as_array(matrices, "bone_matrices")
    if array.dtype.kind not in "iuf":
        raise ValueError(f"bone_matrices は数の並びにしてください（いまの型：{array.dtype}）")
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (4, 4):
        raise ValueError(f"bone_matrices は (B, 4, 4) の並びにしてください（いまの形：{array.shape}）")
    if not np.isfinite(array).all():
        raise ValueError("bone_matrices に NaN か無限大が入っています")
    if (np.abs(array[:, 3, :] - np.array([0.0, 0.0, 0.0, 1.0])) > AFFINE_TOLERANCE).any():
        raise ValueError("bone_matrices の下の行が (0, 0, 0, 1) ではありません（行列が転置されていませんか）")
    return array


# MARK: - ウェイトの表

@dataclass(frozen=True, eq=False)
class WeightTable:
    """頂点ごとのウェイト。

    bone_ids: (N, K) の int64。骨の番号（0 以上）。空きの枠は -1
    weights: (N, K) の float64。重み。空きの枠は 0

    - 作るときに確かめる：2つの形が同じ、番号は -1 以上、空きの枠の重みは 0、重みは有限（NaN・無限大は ValueError）。
    - 負の重みと、同じ骨が1つの頂点に2回入ることは、表としては受け付ける（check が見つけて報告する）。
    - 中の並びは写しを取り、書き換えられないようにする。配列どうしの == は使えないので、比べるときは中の並びを比べる。
    """

    bone_ids: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        ids = _as_id_matrix(self.bone_ids, "bone_ids")
        weights = _as_weight_matrix(self.weights, "weights")
        if ids.shape != weights.shape:
            raise ValueError(f"bone_ids と weights の形が違います（{ids.shape} と {weights.shape}）")
        if not np.isfinite(weights).all():
            raise ValueError("weights に NaN か無限大が入っています")
        if (ids < -1).any():
            raise ValueError("bone_ids に -1 より小さい番号があります（空きの枠は -1）")
        if (weights[ids == -1] != 0.0).any():
            raise ValueError("空きの枠（bone_ids が -1）の重みは 0 にしてください")
        ids = ids.copy()
        weights = weights.copy()
        ids.flags.writeable = False
        weights.flags.writeable = False
        object.__setattr__(self, "bone_ids", ids)
        object.__setattr__(self, "weights", weights)

    @property
    def vertex_count(self) -> int:
        """頂点の数 N"""
        return int(self.bone_ids.shape[0])

    @property
    def width(self) -> int:
        """1つの頂点の枠の数 K"""
        return int(self.bone_ids.shape[1])

    @classmethod
    def from_lists(cls, per_vertex: Sequence[Sequence[tuple[int, float]]]) -> WeightTable:
        """頂点ごとの [(骨の番号, 重み), ...] から作る。枠の数 K は、いちばん長い頂点に合わせる（並びの順はそのまま）。

        骨の番号は 0 以上の整数（真偽は不可）、重みは有限の数（真偽は不可）。
        """
        if isinstance(per_vertex, (str, bytes)) or not isinstance(per_vertex, (Sequence, np.ndarray)):
            raise ValueError("per_vertex は、頂点ごとの並びの並びにしてください")
        rows = list(per_vertex)
        for index, row in enumerate(rows):
            if isinstance(row, (str, bytes)) or not isinstance(row, (Sequence, np.ndarray)):
                raise ValueError(f"頂点 {index} のウェイトは (骨の番号, 重み) の並びにしてください")
        width = max((len(row) for row in rows), default=0)
        ids = np.full((len(rows), width), -1, dtype=np.int64)
        weights = np.zeros((len(rows), width), dtype=np.float64)
        for index, row in enumerate(rows):
            for slot, entry in enumerate(row):
                bone, weight = _parse_entry(entry, index)
                ids[index, slot] = bone
                weights[index, slot] = weight
        return cls(ids, weights)

    @classmethod
    def from_dense(cls, dense: Any, *, epsilon: float = thresholds.WEIGHT_EPSILON) -> WeightTable:
        """(N, B) の並び（行が頂点、列が骨の番号）から作る。

        |重み| が epsilon より大きい物だけを入れる（負の重みも、check で見つけられるように入れる）。
        頂点の中は骨の番号の小さい順。枠の数 K は、入れた物がいちばん多い頂点に合わせる。
        """
        array = as_array(dense, "dense")
        if array.dtype.kind not in "iuf":
            raise ValueError(f"dense は数の並びにしてください（いまの型：{array.dtype}）")
        array = np.asarray(array, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"dense は (N, B) の並びにしてください（いまの形：{array.shape}）")
        if not np.isfinite(array).all():
            raise ValueError("dense に NaN か無限大が入っています")
        limit = as_nonnegative_float(epsilon, "epsilon")
        keep = np.abs(array) > limit
        counts = keep.sum(axis=1)
        width = int(counts.max()) if len(counts) else 0
        ids = np.full((len(array), width), -1, dtype=np.int64)
        weights = np.zeros((len(array), width), dtype=np.float64)
        rows, columns = np.nonzero(keep)  # 行ごとに、列（骨の番号）の小さい順に出てくる
        slots = np.arange(len(rows)) - np.repeat(np.cumsum(counts) - counts, counts)
        ids[rows, slots] = columns
        weights[rows, slots] = array[rows, columns]
        return cls(ids, weights)

    def to_dense(self, n_bones: int) -> np.ndarray:
        """(N, n_bones) の並びにする。同じ骨が2回入っている頂点は、重みを足す。番号が n_bones 以上なら ValueError"""
        bones = _as_count(n_bones, "n_bones", 0)
        used = self.bone_ids >= 0
        if used.any() and int(self.bone_ids.max()) >= bones:
            raise ValueError(f"bone_ids に n_bones（{bones}）以上の番号があります")
        rows = np.broadcast_to(np.arange(self.vertex_count)[:, None], self.bone_ids.shape)
        flat = rows[used] * bones + self.bone_ids[used]
        dense = np.bincount(flat, weights=self.weights[used], minlength=self.vertex_count * bones)
        return dense.reshape(self.vertex_count, bones)


def _parse_entry(entry: Any, index: int) -> tuple[int, float]:
    """(骨の番号, 重み) の1つを確かめる"""
    if isinstance(entry, (str, bytes)) or not isinstance(entry, (Sequence, np.ndarray)) or len(entry) != 2:
        raise ValueError(f"頂点 {index} のウェイトは (骨の番号, 重み) の組にしてください（いまの値：{entry!r}）")
    bone, weight = entry
    if isinstance(bone, (bool, np.bool_)) or not isinstance(bone, (int, np.integer)) or bone < 0:
        raise ValueError(f"頂点 {index} の骨の番号は 0 以上の整数にしてください（いまの値：{bone!r}）")
    if isinstance(weight, (bool, np.bool_)) or not isinstance(weight, (int, float, np.integer, np.floating)):
        raise ValueError(f"頂点 {index} の重みは数にしてください（いまの値：{weight!r}）")
    try:
        value = float(weight)
    except OverflowError:  # float にできない大きな整数
        raise ValueError(f"頂点 {index} の重みが大きすぎます") from None
    if not math.isfinite(value):
        raise ValueError(f"頂点 {index} の重みに NaN か無限大が入っています")
    if int(bone) > _INT64_MAX:
        raise ValueError(f"頂点 {index} の骨の番号が大きすぎます")
    return int(bone), value


# MARK: - 形を動かす骨

def _as_table(table: Any) -> WeightTable:
    if not isinstance(table, WeightTable):
        raise ValueError(f"table は WeightTable にしてください（いまの型：{type(table).__name__}）")
    return table


def _deform_mask(table: WeightTable, deform: Any) -> np.ndarray | None:
    """deform を確かめて (B,) の真偽の並びにする。None ならそのまま None（全部の骨が形を動かす）。

    真偽の1次元の並びでない、表の骨の番号が B 以上のときは ValueError。
    """
    if deform is None:
        return None
    mask = as_array(deform, "deform")
    if mask.dtype.kind != "b" or mask.ndim != 1:
        raise ValueError(f"deform は1次元の真偽（bool）の並びにしてください（いまの型・形：{mask.dtype}・{mask.shape}）")
    if (table.bone_ids >= 0).any() and int(table.bone_ids.max()) >= len(mask):
        raise ValueError(f"bone_ids に deform の長さ（{len(mask)}）以上の番号があります")
    return mask


def _deform_slots(ids: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """枠ごとに「形を動かす骨の重みか」(n, K)。空きの枠は偽。mask は _deform_mask で確かめた物"""
    filled = ids >= 0
    if mask is None or len(mask) == 0:
        return filled
    return filled & mask[np.maximum(ids, 0)]


def _row_blocks(count: int) -> Iterator[slice]:
    """頂点を CHUNK_SIZE 個ずつに分ける（メモリが増えすぎないように）"""
    for start in range(0, count, CHUNK_SIZE):
        yield slice(start, min(start + CHUNK_SIZE, count))


class _Merged:
    """形を動かす骨の重みを、頂点ごとに骨でまとめた物（頂点のひとかたまり分）。

    ids・weights: (n, K)。骨の番号の小さい順に、同じ骨の重みを足した物（負の重みは 0 として足す）。残りの枠は -1・0
    deform_slot: 枠ごとに「形を動かす骨の重みか」(n, K)（元の枠の並び）
    deform_sum: 形を動かす骨の重みの、そのままの合計 (n,)（負の重みも、ごく小さい重みも入れる）
    negative: 形を動かす骨に、負の重みがあるか (n,)
    duplicate: 形を動かす骨が、同じ頂点に2回以上入っているか (n,)
    """

    def __init__(self, ids: np.ndarray, weights: np.ndarray, mask: np.ndarray | None) -> None:
        count, width = ids.shape
        self.deform_slot = _deform_slots(ids, mask)
        self.deform_sum = np.where(self.deform_slot, weights, 0.0).sum(axis=1)
        self.negative = (self.deform_slot & (weights < 0.0)).any(axis=1)
        # 形を動かす骨の重みだけを、骨の番号の順に並べる（ほかの枠は、いちばん後ろへ）
        key = np.where(self.deform_slot, ids, _INT64_MAX)
        order = np.argsort(key, axis=1, kind="stable")
        sorted_ids = np.take_along_axis(key, order, axis=1)
        sorted_weights = np.take_along_axis(np.where(self.deform_slot, np.maximum(weights, 0.0), 0.0), order, axis=1)
        used = sorted_ids != _INT64_MAX
        start = used.copy()
        start[:, 1:] &= sorted_ids[:, 1:] != sorted_ids[:, :-1]
        self.duplicate = (used & ~start).any(axis=1)
        group = np.cumsum(start, axis=1) - 1
        rows = np.broadcast_to(np.arange(count)[:, None], (count, width))
        flat = (rows * width + group)[used]
        self.weights = np.bincount(flat, weights=sorted_weights[used], minlength=count * width).reshape(count, width)
        self.ids = np.full((count, width), -1, dtype=np.int64)
        self.ids[rows[start], group[start]] = sorted_ids[start]

    def influences(self, epsilon: float) -> np.ndarray:
        """重みが epsilon より大きい、形を動かす骨の印 (n, K)（self.ids の枠の並び）"""
        return (self.ids >= 0) & (self.weights > epsilon)

    def flags(self, limit: int, epsilon: float, tolerance: float) -> dict[str, np.ndarray]:
        """check の印（このかたまり分）"""
        counts = self.influences(epsilon).sum(axis=1)
        unweighted = counts == 0
        return {
            "unweighted": unweighted,
            "over_limit": counts > limit,
            "not_normalized": ~unweighted & (np.abs(self.deform_sum - 1.0) > tolerance),
            "negative": self.negative,
            "duplicate_ids": self.duplicate,
        }


def _needs_fix(flags: dict[str, np.ndarray]) -> np.ndarray:
    """安全な直しをかける頂点：どの骨かに付いていて、正しくない物"""
    return ~flags["unweighted"] & (flags["over_limit"] | flags["not_normalized"] | flags["negative"]
                                   | flags["duplicate_ids"])


class _Analysis:
    """表全体の点検。頂点のかたまりごとに計算して、頂点ごとの結果 (N,) をつなぐ。

    flags: check の印。counts: 効いている骨の数。deform_sum: 形を動かす骨の重みの合計。
    min_deform_weight: 形を動かす骨の重みの、いちばん小さい値（無ければ None）
    """

    def __init__(self, table: WeightTable, mask: np.ndarray | None, limit: int, epsilon: float,
                 tolerance: float) -> None:
        count = table.vertex_count
        self.flags = {key: np.zeros(count, dtype=bool) for key in CHECK_KEYS}
        self.counts = np.zeros(count, dtype=np.int64)
        self.deform_sum = np.zeros(count)
        self.min_deform_weight: float | None = None
        for rows in _row_blocks(count):
            merged = _Merged(table.bone_ids[rows], table.weights[rows], mask)
            for key, value in merged.flags(limit, epsilon, tolerance).items():
                self.flags[key][rows] = value
            self.counts[rows] = merged.influences(epsilon).sum(axis=1)
            self.deform_sum[rows] = merged.deform_sum
            if merged.deform_slot.any():
                smallest = float(table.weights[rows][merged.deform_slot].min())
                self.min_deform_weight = smallest if self.min_deform_weight is None else min(self.min_deform_weight,
                                                                                            smallest)


# MARK: - 数える・点検

def sums(table: WeightTable, *, deform: Any = None) -> np.ndarray:
    """頂点ごとの重みの合計 (N,)。deform を渡すと、形を動かす骨の重みだけを足す。負の重みもそのまま足す"""
    t = _as_table(table)
    mask = _deform_mask(t, deform)
    result = np.zeros(t.vertex_count)
    for rows in _row_blocks(t.vertex_count):
        result[rows] = np.where(_deform_slots(t.bone_ids[rows], mask), t.weights[rows], 0.0).sum(axis=1)
    return result


def influence_counts(table: WeightTable, *, epsilon: float = thresholds.WEIGHT_EPSILON,
                     deform: Any = None) -> np.ndarray:
    """頂点ごとの、効いている骨の数 (N,) の int64。

    同じ骨が2回入っていれば重みを足してから（負の重みは 0 として）、重みが epsilon より大きい骨を数える。
    deform を渡すと、形を動かす骨だけを数える。
    """
    t = _as_table(table)
    small = as_nonnegative_float(epsilon, "epsilon")
    mask = _deform_mask(t, deform)
    result = np.zeros(t.vertex_count, dtype=np.int64)
    for rows in _row_blocks(t.vertex_count):
        result[rows] = _Merged(t.bone_ids[rows], t.weights[rows], mask).influences(small).sum(axis=1)
    return result


def check(table: WeightTable, *, limit: int = thresholds.MAX_BONE_INFLUENCES, deform: Any = None,
          epsilon: float = thresholds.WEIGHT_EPSILON,
          tolerance: float = thresholds.WEIGHT_SUM_TOLERANCE) -> dict[str, np.ndarray]:
    """ウェイトを点検する。返す物は、鍵（CHECK_KEYS の順）ごとの (N,) の真偽の並び。

    - "unweighted"：形を動かす骨の重みが、どれも epsilon 以下（どの骨にも付いていない点）
    - "over_limit"：効いている骨（influence_counts）が limit 本を超える
    - "not_normalized"：形を動かす骨の重みの合計が、1 から tolerance より離れている（unweighted の点は除く）
    - "negative"：形を動かす骨に、負の重みがある
    - "duplicate_ids"：形を動かす骨が、同じ頂点に2回以上入っている
    形を動かさない骨の重みは、どれにも数えない。
    """
    t = _as_table(table)
    analysis = _Analysis(t, _deform_mask(t, deform), _as_count(limit, "limit", 1),
                         as_nonnegative_float(epsilon, "epsilon"), as_nonnegative_float(tolerance, "tolerance"))
    return analysis.flags


# MARK: - 安全な直し

def normalize_and_limit(table: WeightTable, *, limit: int = thresholds.MAX_BONE_INFLUENCES, deform: Any = None,
                        epsilon: float = thresholds.WEIGHT_EPSILON,
                        tolerance: float = thresholds.WEIGHT_SUM_TOLERANCE) -> tuple[WeightTable, np.ndarray]:
    """安全な直し「ウェイトを合計1にして1点4本までにする」。返す物は (直した表, 変わった頂点の印 (N,))。

    直すのは、check で over_limit・not_normalized・negative・duplicate_ids のどれかが真の頂点だけ。
    すでに正しい頂点（骨が limit 本まで、合計が 1 から tolerance 以内、負の重みも重なりも無い）は、まったく変えない。
    どの骨にも付いていない頂点（unweighted）も変えない（重みを作り出さない。check で報告されたまま残る）。

    直し方（形を動かす骨だけ）：
    1. 負の重みを 0 にし、同じ骨の重みを足して1つにする。
    2. 重みが epsilon より大きい骨から、重みの大きい順に limit 本を残す（同じ重みなら、骨の番号の小さいほう）。
       残りは消す。
    3. 残した重みを、合計が 1 になるように割る。
    形を動かさない骨の枠は、元の順のまま、形を動かす骨の後ろに置く（重みは変えない）。
    枠の数 K は変えない。直した頂点の枠は、形を動かす骨（重みの大きい順）・形を動かさない骨・空き の順に並ぶ。
    同じ入力なら、いつも同じ答えになる。
    """
    t = _as_table(table)
    cap = _as_count(limit, "limit", 1)
    small = as_nonnegative_float(epsilon, "epsilon")
    gap = as_nonnegative_float(tolerance, "tolerance")
    mask = _deform_mask(t, deform)
    new_ids = t.bone_ids.copy()
    new_weights = t.weights.copy()
    width = t.width
    for rows in _row_blocks(t.vertex_count):
        ids, weights = t.bone_ids[rows], t.weights[rows]
        merged = _Merged(ids, weights, mask)
        local = np.flatnonzero(_needs_fix(merged.flags(cap, small, gap)))
        if len(local) == 0:
            continue
        merged_ids = merged.ids[local]
        merged_weights = merged.weights[local]
        positive = merged.influences(small)[local]
        # 重みの大きい順（同じなら番号の小さい順）。効いていない枠は後ろへ
        order = np.lexsort((merged_ids, -merged_weights, ~positive), axis=1)
        kept = np.take_along_axis(positive, order, axis=1) & (np.arange(width) < cap)
        kept_ids = np.where(kept, np.take_along_axis(merged_ids, order, axis=1), -1)
        kept_weights = np.where(kept, np.take_along_axis(merged_weights, order, axis=1), 0.0)
        total = kept_weights.sum(axis=1)
        if not np.isfinite(total).all():
            # 黙って割ると、重みが全部 0（どの骨にも付いていない点）か NaN になる
            raise ValueError("重みが大きすぎて、合計が無限大になります（合計を 1 にできません）")
        kept_weights = kept_weights / total[:, None]
        other = (ids[local] >= 0) & ~merged.deform_slot[local]
        joined_ids = np.concatenate([kept_ids, np.where(other, ids[local], -1)], axis=1)
        joined_weights = np.concatenate([kept_weights, np.where(other, weights[local], 0.0)], axis=1)
        # 空きでない枠を前に詰める（入る物は、元の空きでない枠の数より多くならないので、K に収まる）
        packed = np.argsort(joined_ids == -1, axis=1, kind="stable")[:, :width]
        target = rows.start + local
        new_ids[target] = np.take_along_axis(joined_ids, packed, axis=1)
        new_weights[target] = np.take_along_axis(joined_weights, packed, axis=1)
    changed = (new_ids != t.bone_ids).any(axis=1) | (new_weights != t.weights).any(axis=1)
    return WeightTable(new_ids, new_weights), changed


# MARK: - 骨で形を動かす

def linear_blend_skinning(rest_vertices: Any, table: WeightTable, bone_matrices: Any, *, deform: Any = None,
                          chunk_size: int = CHUNK_SIZE) -> np.ndarray:
    """骨の行列で頂点を動かす（線形ブレンドスキニング）。返す物は (N, 3)。

    動いた位置 = Σ_k w_k × (M_k @ [v, 1]) の x, y, z。M_k は骨 k の行列（休みの姿勢から今の姿勢へ動かす 4×4 の行列、
    列ベクトルに左から掛ける形。下の行は (0, 0, 0, 1)）。
    - 重みは渡されたまま使う（合計が 1 でなくても割らない。Blender のアーマチュアは合計で割る）。
      合計を 1 にしたいときは、先に normalize_and_limit を呼ぶ。重みが1つも無い頂点は (0, 0, 0) になる。
    - 空きの枠は使わない。deform（(B,) の真偽）を渡すと、形を動かさない骨の枠も使わない。
    - 骨の番号が B 以上なら ValueError。
    頂点を chunk_size 個ずつ計算する（メモリが増えすぎないように）。
    """
    t = _as_table(table)
    vertices = as_points(rest_vertices, "rest_vertices")
    if len(vertices) != t.vertex_count:
        raise ValueError(f"rest_vertices の数（{len(vertices)}）が、ウェイトの頂点の数（{t.vertex_count}）と違います")
    matrices = _as_matrices(bone_matrices)
    step = _as_count(chunk_size, "chunk_size", 1)
    if (t.bone_ids >= 0).any() and int(t.bone_ids.max()) >= len(matrices):
        raise ValueError(f"bone_ids に bone_matrices の数（{len(matrices)}）以上の番号があります")
    if deform is not None and as_array(deform, "deform").shape != (len(matrices),):
        raise ValueError(f"deform の形は ({len(matrices)},) にしてください（いまの形：{as_array(deform, 'deform').shape}）")
    mask = _deform_mask(t, deform)
    affine = matrices[:, :3, :]
    result = np.zeros((len(vertices), 3))
    if len(matrices) == 0:
        return result  # 骨が無ければ、空きでない枠も無い（上で確かめた）
    for start in range(0, len(vertices), step):
        rows = slice(start, min(start + step, len(vertices)))
        use = _deform_slots(t.bone_ids[rows], mask)
        safe_ids = np.where(use, t.bone_ids[rows], 0)
        used_weights = np.where(use, t.weights[rows], 0.0)
        blend = np.zeros((len(use), 3, 4))
        for slot in range(t.width):
            blend += used_weights[:, slot, None, None] * affine[safe_ids[:, slot]]
        result[rows] = np.einsum("nij,nj->ni", blend[:, :, :3], vertices[rows]) + blend[:, :, 3]
    return result


# MARK: - 信号にまとめる

def summarize(table: WeightTable, *, limit: int = thresholds.MAX_BONE_INFLUENCES, deform: Any = None,
              epsilon: float = thresholds.WEIGHT_EPSILON,
              tolerance: float = thresholds.WEIGHT_SUM_TOLERANCE) -> list[Measurement]:
    """ウェイトの点検を、信号にまとめる。単位はどれも "count"（その頂点の数）。

    - "bone_influences_over_limit"：1点に効く骨が limit 本を超える頂点。1つでもあれば赤（アプリの上限は4本）
    - "unweighted_vertices"：どの骨にも付いていない頂点。1つでもあれば赤
    - "weights_not_normalized"：合計が 1 でない頂点。1つでもあれば黄
    - "negative_weights"：負の重みがある頂点。1つでもあれば赤
    - "duplicate_bone_ids"：同じ骨が2回入っている頂点。1つでもあれば黄
    頂点が無いときは、どれも灰。over_limit・not_normalized・negative・duplicate は normalize_and_limit で直せる。
    """
    t = _as_table(table)
    cap = _as_count(limit, "limit", 1)
    gap = as_nonnegative_float(tolerance, "tolerance")
    analysis = _Analysis(t, _deform_mask(t, deform), cap, as_nonnegative_float(epsilon, "epsilon"), gap)
    names = (OVER_LIMIT_NAME, UNWEIGHTED_NAME, NOT_NORMALIZED_NAME, NEGATIVE_NAME, DUPLICATE_NAME)
    if t.vertex_count == 0:
        return [Measurement.unmeasured(name, "count", "頂点がありません") for name in names]
    flags = analysis.flags
    weighted = ~flags["unweighted"]
    deviation = np.abs(analysis.deform_sum - 1.0)

    def make(name: str, mask: np.ndarray, bad: Signal, reason: str, extra: dict[str, Any]) -> Measurement:
        count = int(mask.sum())
        details = {"vertex_count": t.vertex_count, "indices": np.flatnonzero(mask).astype(np.int64), **extra}
        if count:
            return Measurement(name, float(count), "count", bad, reason, details)
        return Measurement(name, 0.0, "count", Signal.GREEN, "", details)

    return [
        make(OVER_LIMIT_NAME, flags["over_limit"], Signal.RED, "1点に効く骨が上限を超える頂点があります",
             {"limit": cap, "max_influences": int(analysis.counts.max())}),
        make(UNWEIGHTED_NAME, flags["unweighted"], Signal.RED, "どの骨にも付いていない頂点があります", {}),
        make(NOT_NORMALIZED_NAME, flags["not_normalized"], Signal.YELLOW, "重みの合計が 1 でない頂点があります",
             {"tolerance": gap, "max_deviation": float(deviation[weighted].max()) if weighted.any() else None}),
        make(NEGATIVE_NAME, flags["negative"], Signal.RED, "負の重みがある頂点があります",
             {"min_weight": analysis.min_deform_weight}),
        make(DUPLICATE_NAME, flags["duplicate_ids"], Signal.YELLOW, "同じ骨が2回入っている頂点があります", {}),
    ]

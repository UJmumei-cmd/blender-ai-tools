"""A4 面の点検（bpy なし・numpy だけ）。

安全な直し（CLAUDE.md 2章）で自動で直す物：
- 重なった点をまとめる（duplicate_vertex_groups で見つける）
- 面の向き（orient_consistently で、そろえた三角形を作る）

見つけて一覧にするだけの物（直すのはユーザーの確認のあと）：
- つぶれた三角形（degenerate_triangles）
- 穴の縁の辺（boundary_edges）、3つ以上の面が使う辺（non_manifold_edges）
- 小さな離れた部品（loose_parts）、どの面にも使われていない頂点（unused_vertices）

信号の決め方（summarize_*）：
- 重なった点・つぶれた三角形・3つ以上の面が使う辺・小さな離れた部品：1つでもあれば黄。
  3つ以上の面が使う辺は、MMD のモデルではふつうに見られ、見た目が壊れているとは限らないので赤にしない。
  ただし、その近くでは「中か外か」（geometry.signed_distance）が当てにならない。
- 向きの合わない辺：1つでもあれば赤（アプリで面が裏返って見えなくなる。安全な直しで直せる）。
- 穴（閉じているはずの形だけ。summarize_holes。値は縁の辺の輪の数）：1つでもあれば赤（ロードマップ段階1：体の穴が0）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import thresholds
from ..thresholds import LOOSE_PART_MAX_FACES
from .geometry import (as_nonnegative_float, as_points, as_positive_int, as_triangles, triangle_areas,
                       weld_exact)
from .results import Measurement, Signal

# 離れた部品を「小さい」とみなす三角形の数（いちばん大きい部品は除く）は thresholds.LOOSE_PART_MAX_FACES（仮）。上で読み込む

# 点の組を、一度にいくつまで距離を調べるか（重なった点さがし）
_PAIR_BUDGET = 1 << 20
# 閉じた形の体積が、ほぼ 0 とみなす割合（体積 <= この数 × 表面積^1.5）。平らに重なった形で向きを決めないため
_FLAT_VOLUME_RATIO = 1e-9

# 結果の名前（summarize_*）
DUPLICATES_NAME = "duplicate_vertices"
DEGENERATE_NAME = "degenerate_triangles"
NON_MANIFOLD_NAME = "non_manifold_edges"
INCONSISTENT_NAME = "inconsistent_edges"
HOLES_NAME = "holes"
LOOSE_PARTS_NAME = "loose_parts"


# MARK: - つながりをまとめる（union-find）

class _UnionFind:
    """番号どうしのつながりをまとめる。根（まとまりの代表）は、そのまとまりのいちばん小さい番号になる。

    つなぐ（union）たびに、すべての番号が直接、根を指すようにしておく。
    """

    def __init__(self, count: int) -> None:
        self.parent = np.arange(count, dtype=np.int64)

    def union(self, first: np.ndarray, second: np.ndarray) -> None:
        """first[k] と second[k] を、同じまとまりにつなぐ"""
        first = np.asarray(first, dtype=np.int64)
        second = np.asarray(second, dtype=np.int64)
        while first.size:
            root_first, root_second = self.parent[first], self.parent[second]
            differ = root_first != root_second
            if not differ.any():
                return
            first, second = first[differ], second[differ]
            root_first, root_second = root_first[differ], root_second[differ]
            # 大きいほうの根を、小さいほうの根につなぐ（親はいつも自分以下なので、輪にならない）
            np.minimum.at(self.parent, np.maximum(root_first, root_second), np.minimum(root_first, root_second))
            self._compress()

    def _compress(self) -> None:
        while True:
            grand = self.parent[self.parent]
            if np.array_equal(grand, self.parent):
                return
            self.parent = grand

    def roots(self) -> np.ndarray:
        """番号ごとの根（まとまりのいちばん小さい番号）"""
        return self.parent


def _groups_from_roots(roots: np.ndarray, minimum_size: int) -> list[np.ndarray]:
    """根が同じ番号をまとめる。中は小さい順、まとまりは「いちばん小さい番号」の順"""
    sizes = np.bincount(roots, minlength=len(roots))
    members = np.flatnonzero(sizes[roots] >= minimum_size)
    if members.size == 0:
        return []
    order = np.argsort(roots[members], kind="stable")
    members = members[order]
    keys = roots[members]
    cuts = np.flatnonzero(keys[1:] != keys[:-1]) + 1
    return [part.astype(np.int64) for part in np.split(members, cuts)]


# MARK: - 重なった点

def duplicate_vertex_groups(vertices: Any, distance: float = thresholds.MERGE_DISTANCE) -> list[np.ndarray]:
    """距離 distance 以内（distance ちょうども含む）の頂点をつなぎ、つながった物を1つのまとまりにする。

    a と b、b と c が近ければ、a と c が離れていても同じまとまり（Blender の「距離でマージ」で1点になる物）。
    返す物：まとまり（2点以上）ごとの頂点の番号 (int64)。中は小さい順、まとまりは「いちばん小さい番号」の順。
    distance = 0 なら、座標がまったく同じ頂点だけ。

    方法：まず座標がまったく同じ頂点をまとめ、残りを、一辺が distance より少し大きい升目に分ける。
    同じ升目と、となりの升目の点だけを比べるので、点の数にほぼ比例する時間で済む。
    """
    v = as_points(vertices, "vertices")
    limit = as_nonnegative_float(distance, "distance")
    count = len(v)
    if count < 2:
        return []
    groups = _UnionFind(count)
    representative = weld_exact(v)
    copies = np.flatnonzero(representative != np.arange(count))
    groups.union(copies, representative[copies])
    if limit > 0.0:
        unique = np.flatnonzero(representative == np.arange(count))
        pending_first: list[np.ndarray] = []
        pending_second: list[np.ndarray] = []
        pending = 0
        for first, second in _close_pairs(v[unique], limit):
            pending_first.append(unique[first])
            pending_second.append(unique[second])
            pending += len(first)
            if pending >= _PAIR_BUDGET:
                groups.union(np.concatenate(pending_first), np.concatenate(pending_second))
                pending_first, pending_second, pending = [], [], 0
        if pending_first:
            groups.union(np.concatenate(pending_first), np.concatenate(pending_second))
    return _groups_from_roots(groups.roots(), 2)


# 自分の升目のほかに調べる、となりの升目（片側の13個。反対側は、相手の升目から調べるので要らない）
_FORWARD_OFFSETS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                    if (dx, dy, dz) > (0, 0, 0)]


def _close_pairs(points: np.ndarray, limit: float):
    """距離 limit 以内の点の組 (i, j) を、小分けにして順に返す（i と j は points の番号。同じ組は1回だけ）。

    升目の番号は、軸ごとに「使われている升目の値の順位」に詰めてから、1つの整数（key）にする。
    key は順位について1次式なので、となりの升目の key は、自分の key に決まった数を足すだけで求まる。
    """
    count = len(points)
    largest = float(np.abs(points).max()) if count else 0.0
    # 升目は limit より少しだけ大きくする。割り算の丸めで、limit 以内の2点が2つ離れた升目に入るのを防ぐ
    cell = limit * (1.0 + 1e-6) + 1e-15 * largest
    ranks, has_next, sizes = [], [], []
    for axis in range(3):
        values, rank = np.unique(np.floor(points[:, axis] / cell).astype(np.int64), return_inverse=True)
        ranks.append(rank.reshape(-1))
        has_next.append(values[1:] == values[:-1] + 1)  # 順位 r の升目の、すぐ次の升目が使われているか
        sizes.append(len(values))
    if sizes[0] * sizes[1] * sizes[2] >= 2 ** 62:
        raise ValueError("点が広がりすぎていて、升目に分けられません（distance を大きくしてください）")
    keys = (ranks[0] * sizes[1] + ranks[1]) * sizes[2] + ranks[2]
    order = np.argsort(keys, kind="stable")
    cell_keys, cell_starts, cell_counts = np.unique(keys[order], return_index=True, return_counts=True)
    first_point = order[cell_starts]
    cell_ranks = [rank[first_point] for rank in ranks]
    del keys, ranks

    triples = []  # (点の位置, 相手の升目の始まり, 相手の点の数)。位置は order の中の位置
    # 同じ升目：自分より後ろの点
    crowded = np.flatnonzero(cell_counts >= 2)
    if crowded.size:
        position = _expand_ranges(cell_starts[crowded], cell_counts[crowded])
        end = np.repeat(cell_starts[crowded] + cell_counts[crowded], cell_counts[crowded])
        triples.append((position, position + 1, end - position - 1))
    for offset in _FORWARD_OFFSETS:
        valid = np.ones(len(cell_keys), dtype=bool)
        for axis, step in enumerate(offset):
            rank = cell_ranks[axis]
            if step == 1:
                valid &= (rank < sizes[axis] - 1) & np.append(has_next[axis], False)[rank]
            elif step == -1:
                valid &= (rank > 0) & np.insert(has_next[axis], 0, False)[rank]
        source_cells = np.flatnonzero(valid)
        target_keys = cell_keys[source_cells] + (offset[0] * sizes[1] + offset[1]) * sizes[2] + offset[2]
        slot = np.minimum(np.searchsorted(cell_keys, target_keys), len(cell_keys) - 1)
        found = cell_keys[slot] == target_keys
        source_cells, slot = source_cells[found], slot[found]
        if source_cells.size == 0:
            continue
        position = _expand_ranges(cell_starts[source_cells], cell_counts[source_cells])
        triples.append((position, np.repeat(cell_starts[slot], cell_counts[source_cells]),
                        np.repeat(cell_counts[slot], cell_counts[source_cells])))
    if not triples:
        return
    source = np.concatenate([t[0] for t in triples])
    low = np.concatenate([t[1] for t in triples])
    counts = np.concatenate([t[2] for t in triples])
    keep = counts > 0
    source, low, counts = source[keep], low[keep], counts[keep]

    total = np.cumsum(counts)
    start = 0
    while start < len(counts):
        before = int(total[start - 1]) if start else 0
        stop = max(int(np.searchsorted(total, before + _PAIR_BUDGET, side="right")), start + 1)
        part = counts[start:stop]
        first = order[np.repeat(source[start:stop], part)]
        second = order[_expand_ranges(low[start:stop], part)]
        diff = points[first] - points[second]
        length = np.sqrt(diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1] + diff[:, 2] * diff[:, 2])
        close = length <= limit
        if close.any():
            yield first[close], second[close]
        start = stop


def _expand_ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """[starts[k], starts[k] + lengths[k]) を、順につなげた番号の並び"""
    total = int(lengths.sum())
    return np.repeat(starts, lengths) + np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)


# MARK: - つぶれた三角形

def degenerate_triangles(vertices: Any, triangles: Any, area: float = thresholds.DEGENERATE_AREA) -> np.ndarray:
    """つぶれた三角形の番号 (int64、小さい順)：面積が area 以下（ちょうども含む）か、同じ頂点を2回以上使う物"""
    v = as_points(vertices, "vertices")
    t = as_triangles(triangles, len(v))
    limit = as_nonnegative_float(area, "area")
    repeated = (t[:, 0] == t[:, 1]) | (t[:, 1] == t[:, 2]) | (t[:, 2] == t[:, 0])
    return np.flatnonzero(repeated | (triangle_areas(v, t) <= limit)).astype(np.int64)


# MARK: - 辺

class _HalfEdges:
    """三角形の向きのある辺（v0→v1、v1→v2、v2→v0）。同じ頂点どうしの辺（つぶれた三角形の辺）は除く。

    key: 向きを無くした辺の番号（小さいほうの頂点 × 頂点の数 + 大きいほうの頂点。頂点は使われている物だけに詰めた番号）
    forward: 小さい番号から大きい番号へ向かうか
    face: その辺を持つ三角形の番号
    """

    def __init__(self, t: np.ndarray) -> None:
        # 使われている頂点だけに番号を詰める（並びの順は変わらないので、辺の並びの順も変わらない）
        self.used, compact = np.unique(t.ravel(), return_inverse=True)
        compact = compact.reshape(-1, 3)
        self.vertex_count = len(self.used)
        start = compact.ravel()
        end = compact[:, [1, 2, 0]].ravel()
        face = np.repeat(np.arange(len(t), dtype=np.int64), 3)
        valid = start != end
        start, end, face = start[valid], end[valid], face[valid]
        self.forward = start < end
        self.face = face
        self.key = np.minimum(start, end) * self.vertex_count + np.maximum(start, end)
        self.unique_keys, self.edge_of, self.counts = np.unique(self.key, return_inverse=True, return_counts=True)
        self.edge_of = self.edge_of.reshape(-1)

    def edges(self, selected: np.ndarray | None = None) -> np.ndarray:
        """辺 (E, 2)（もとの頂点の番号。各行は小さい番号が先、行は小さい順）"""
        keys = self.unique_keys if selected is None else self.unique_keys[selected]
        if self.vertex_count == 0 or keys.size == 0:
            return np.zeros((0, 2), dtype=np.int64)
        return np.stack([self.used[keys // self.vertex_count], self.used[keys % self.vertex_count]],
                        axis=1).astype(np.int64)

    def pairs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ちょうど2回使われる辺について、(辺の番号, 1つ目の向きのある辺, 2つ目の向きのある辺) の番号"""
        order = np.argsort(self.edge_of, kind="stable")
        two = np.flatnonzero(self.counts == 2)
        starts = np.searchsorted(self.edge_of[order], two)
        return two, order[starts], order[starts + 1]


def edge_face_counts(triangles: Any) -> tuple[np.ndarray, np.ndarray]:
    """辺 (E, 2) と、その辺を使う回数 (E,)。

    辺の各行は小さい番号が先、行は小さい順。同じ頂点どうしの辺（つぶれた三角形の辺）は数えない。
    閉じた形なら、どの辺も 2。穴の縁は 1、3つ以上の面が集まる辺は 3 以上。
    """
    half = _HalfEdges(as_triangles(triangles))
    return half.edges(), half.counts.astype(np.int64)


def boundary_edges(triangles: Any) -> np.ndarray:
    """穴の縁の辺 (K, 2)：1つの面だけが使う辺"""
    half = _HalfEdges(as_triangles(triangles))
    return half.edges(half.counts == 1)


def non_manifold_edges(triangles: Any) -> np.ndarray:
    """3つ以上の面が使う辺 (K, 2)"""
    half = _HalfEdges(as_triangles(triangles))
    return half.edges(half.counts >= 3)


def inconsistent_edges(triangles: Any) -> np.ndarray:
    """向きの合わない辺 (K, 2)：ちょうど2つの面が使い、2つの面が同じ向きにたどる辺。

    向きがそろっていれば、となりの面は共有する辺を逆向きにたどる。
    3つ以上の面が使う辺は、向きがそろうかどうか決まらないので、ここには入れない（non_manifold_edges を見る）。
    """
    half = _HalfEdges(as_triangles(triangles))
    edge, first, second = half.pairs()
    same = half.forward[first] == half.forward[second]
    return half.edges(edge[same])


# MARK: - 面の向きをそろえる

def orient_consistently(vertices: Any, triangles: Any) -> tuple[np.ndarray, np.ndarray]:
    """面の向きをそろえた三角形 (M, 3) と、裏返した面の印 (M,) を返す（元の並びは変えない）。

    - ちょうど2つの面が使う辺でつながった面を、1つのまとまりとして、幅優先でたどる。
      まとまりのいちばん小さい番号の面を基準にし、となりの面が共有する辺を逆向きにたどるように裏返す。
    - 閉じたまとまり（どの辺もちょうど2回使う）は、最後に外向き（符号つきの体積が正）にする。
      体積がほぼ 0（平らに重なった形）のときは、基準の面の向きのまま。
    - 開いたまとまりは、もとの向きのままの面積が多いほうに合わせる（同じなら基準の面の向き）。
    - 裏返すときは (v0, v1, v2) を (v0, v2, v1) にする。
    向きがそろっていて外向きの形は、何も変わらない（印は全部 False）。
    メビウスの帯のように向きをそろえられない形は、たどった順で決まり、向きの合わない辺が残る。
    """
    v = as_points(vertices, "vertices")
    t = as_triangles(triangles, len(v))
    count = len(t)
    if count == 0:
        return t.copy(), np.zeros(0, dtype=bool)
    half = _HalfEdges(t)
    _, first, second = half.pairs()
    face_a, face_b = half.face[first], half.face[second]
    distinct = face_a != face_b
    face_a, face_b = face_a[distinct], face_b[distinct]
    # 同じ向きにたどっていれば、片方を裏返す必要がある（1）
    relation = (half.forward[first] == half.forward[second])[distinct].astype(np.int8)

    parts = _UnionFind(count)
    parts.union(face_a, face_b)
    roots = parts.roots()
    parity = _propagate_parity(count, face_a, face_b, relation, np.unique(roots))

    # 閉じているか：まとまりの面の辺が、どれもちょうど2回使われているか
    open_face = np.zeros(count, dtype=bool)
    open_face[half.face[half.counts[half.edge_of] != 2]] = True
    open_part = np.zeros(count, dtype=bool)
    open_part[roots[open_face]] = True

    areas = triangle_areas(v, t)
    flipped_now = parity == 1
    invert = np.zeros(count, dtype=bool)  # まとまり（根の番号）ごとに、全体を裏返すか

    # 開いたまとまり：もとの向きのままの面積が多いほうに合わせる
    flipped_area = np.bincount(roots, weights=np.where(flipped_now, areas, 0.0), minlength=count)
    kept_area = np.bincount(roots, weights=np.where(flipped_now, 0.0, areas), minlength=count)
    invert |= open_part & (flipped_area > kept_area)

    # 閉じたまとまり：外向き（体積が正）にする
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    face_count = np.bincount(roots, minlength=count).astype(np.float64)
    centroid_sum = np.stack([np.bincount(roots, weights=((a + b + c) / 3.0)[:, axis], minlength=count)
                             for axis in range(3)], axis=1)
    origin = (centroid_sum / np.maximum(face_count, 1.0)[:, None])[roots]
    a, b, c = a - origin, b - origin, c - origin
    volume_part = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    volume = np.bincount(roots, weights=np.where(flipped_now, -volume_part, volume_part), minlength=count)
    surface = np.bincount(roots, weights=areas, minlength=count)
    flat = np.abs(volume) <= _FLAT_VOLUME_RATIO * surface ** 1.5
    invert |= ~open_part & ~flat & (volume < 0.0)

    flipped = flipped_now ^ invert[roots]
    fixed = t.copy()
    fixed[flipped, 1], fixed[flipped, 2] = t[flipped, 2], t[flipped, 1]
    return fixed, flipped


def _propagate_parity(count: int, face_a: np.ndarray, face_b: np.ndarray, relation: np.ndarray,
                      seeds: np.ndarray) -> np.ndarray:
    """面ごとに、基準の面と比べて裏返すか（0 / 1）を、幅優先でたどって決める。

    同じ段で2つ以上の面から届いたときは、番号の小さい面から届いた値を使う（決まった答えにするため）。
    """
    source = np.concatenate([face_a, face_b])
    target = np.concatenate([face_b, face_a])
    rel = np.concatenate([relation, relation])
    order = np.argsort(source, kind="stable")
    source, target, rel = source[order], target[order], rel[order]
    starts = np.searchsorted(source, np.arange(count + 1))

    parity = np.full(count, -1, dtype=np.int8)
    parity[seeds] = 0
    frontier = np.asarray(seeds, dtype=np.int64)
    while frontier.size:
        lengths = starts[frontier + 1] - starts[frontier]
        total = int(lengths.sum())
        if total == 0:
            break
        origin = np.repeat(frontier, lengths)
        position = np.repeat(starts[frontier], lengths) + np.arange(total) - np.repeat(np.cumsum(lengths) - lengths,
                                                                                        lengths)
        reached = target[position]
        new = parity[reached] < 0
        reached, origin, step = reached[new], origin[new], rel[position[new]]
        if reached.size == 0:
            break
        pick = np.lexsort((origin, reached))
        reached, origin, step = reached[pick], origin[pick], step[pick]
        first = np.ones(len(reached), dtype=bool)
        first[1:] = reached[1:] != reached[:-1]
        reached, origin, step = reached[first], origin[first], step[first]
        parity[reached] = parity[origin] ^ step
        frontier = reached
    return parity


# MARK: - 部品

def face_components(triangles: Any) -> tuple[np.ndarray, int]:
    """頂点を共有する面をつないだ部品ごとの番号 (M,) int64 と、部品の数。

    部品の番号は、いちばん小さい面の番号の順に 0, 1, 2 …。頂点1つだけを共有する面も同じ部品。
    """
    t = as_triangles(triangles)
    count = len(t)
    if count == 0:
        return np.zeros(0, dtype=np.int64), 0
    # 使われている頂点だけに番号を詰める（大きな番号があっても、並びが大きくならないように）
    used, compact = np.unique(t.ravel(), return_inverse=True)
    # 面（0〜M-1）と頂点（M〜）をつなぐ。根はいちばん小さい番号なので、面のある部品の根は面の番号になる
    nodes = _UnionFind(count + len(used))
    nodes.union(np.repeat(np.arange(count), 3), count + compact.reshape(-1))
    roots = nodes.roots()[:count]
    unique_roots, labels = np.unique(roots, return_inverse=True)
    return labels.reshape(-1).astype(np.int64), len(unique_roots)


def loose_parts(vertices: Any, triangles: Any, *, max_faces: int = LOOSE_PART_MAX_FACES) -> list[np.ndarray]:
    """小さな離れた部品：三角形が max_faces 個以下の部品の、面の番号 (int64) の一覧。

    いちばん大きい部品（同じ大きさなら番号の小さいほう）は、小さくても入れない（モデルそのものを消さないため）。
    並びは、いちばん小さい面の番号の順。消すかどうかは、ユーザーが決める。
    """
    v = as_points(vertices, "vertices")
    t = as_triangles(triangles, len(v))
    limit = as_positive_int(max_faces, "max_faces")
    labels, count = face_components(t)
    if count <= 1:
        return []
    sizes = np.bincount(labels, minlength=count)
    small = sizes <= limit
    small[int(np.argmax(sizes))] = False  # 同じ大きさなら、最初（番号の小さい部品）
    faces = np.flatnonzero(small[labels])
    if faces.size == 0:
        return []
    # 部品の番号は「いちばん小さい面の番号」の順なので、番号で並べればその順になる
    faces = faces[np.argsort(labels[faces], kind="stable")]
    keys = labels[faces]
    cuts = np.flatnonzero(keys[1:] != keys[:-1]) + 1
    return [part.astype(np.int64) for part in np.split(faces, cuts)]


def unused_vertices(vertices: Any, triangles: Any) -> np.ndarray:
    """どの三角形にも使われていない頂点の番号 (int64、小さい順)"""
    v = as_points(vertices, "vertices")
    t = as_triangles(triangles, len(v))
    return np.flatnonzero(np.bincount(t.ravel(), minlength=len(v)) == 0).astype(np.int64)


# MARK: - 信号にまとめる

# 調べる物が無いときの理由（灰）。何も調べていないのに緑にしない（CLAUDE.md 2章5）
_NO_TRIANGLES = "三角形が1つもありません（調べる面がありません）"
_NO_VERTICES = "頂点が1つもありません（調べる点がありません）"


def _count_measurement(name: str, count: int, signal_if_any: Signal, reason_if_any: str,
                       details: dict[str, Any], *, empty_reason: str = "") -> Measurement:
    """数を信号にする（0 なら緑、1つでもあれば signal_if_any）。empty_reason があれば、調べる物が無いので灰"""
    if empty_reason:
        return Measurement.unmeasured(name, "count", empty_reason, details)
    signal = signal_if_any if count else Signal.GREEN
    return Measurement(name, float(count), "count", signal, reason_if_any if count else "", details)


def summarize_duplicates(vertices: Any, distance: float = thresholds.MERGE_DISTANCE) -> Measurement:
    """重なった点（名前 "duplicate_vertices"）。値は、まとめると減る頂点の数。1つでもあれば黄（安全な直しで直せる）。
    頂点が1つも無いときは灰"""
    v = as_points(vertices, "vertices")
    groups = duplicate_vertex_groups(v, distance)
    removed = sum(len(group) - 1 for group in groups)
    return _count_measurement(DUPLICATES_NAME, removed, Signal.YELLOW, "重なった点があります（まとめられます）",
                              {"group_count": len(groups), "groups": groups},
                              empty_reason="" if len(v) else _NO_VERTICES)


def summarize_degenerate(vertices: Any, triangles: Any, area: float = thresholds.DEGENERATE_AREA) -> Measurement:
    """つぶれた三角形（名前 "degenerate_triangles"）。1つでもあれば黄。三角形が1つも無いときは灰"""
    v = as_points(vertices, "vertices")
    t = as_triangles(triangles, len(v))
    indices = degenerate_triangles(v, t, area)
    return _count_measurement(DEGENERATE_NAME, len(indices), Signal.YELLOW, "つぶれた三角形があります",
                              {"indices": indices}, empty_reason="" if len(t) else _NO_TRIANGLES)


def summarize_non_manifold(triangles: Any) -> Measurement:
    """3つ以上の面が使う辺（名前 "non_manifold_edges"）。1つでもあれば黄（その近くでは中か外かが当てにならない）。
    三角形が1つも無いときは灰"""
    t = as_triangles(triangles)
    edges = non_manifold_edges(t)
    return _count_measurement(NON_MANIFOLD_NAME, len(edges), Signal.YELLOW,
                              "3つ以上の面が集まる辺があります（その近くでは、めり込みの向きが当てになりません）",
                              {"edges": edges}, empty_reason="" if len(t) else _NO_TRIANGLES)


def summarize_inconsistent(triangles: Any) -> Measurement:
    """向きの合わない辺（名前 "inconsistent_edges"）。1つでもあれば赤（面の向きの直しで直せる）。
    三角形が1つも無いときは灰"""
    t = as_triangles(triangles)
    edges = inconsistent_edges(t)
    return _count_measurement(INCONSISTENT_NAME, len(edges), Signal.RED,
                              "面の向きがそろっていません（面の向きの直しで直せます）", {"edges": edges},
                              empty_reason="" if len(t) else _NO_TRIANGLES)


def summarize_holes(triangles: Any) -> Measurement:
    """穴（名前 "holes"）。閉じているはずの形（体など）に使う。1つでもあれば赤。

    値は穴の数：穴の縁の辺（1つの面だけが使う辺）が、頂点でつながった輪の数。
    三角形が1つ抜けた穴は、縁の辺が3本でも、穴は1つと数える（報告の「穴が N 個」に入る値なので）。
    details：縁の辺 "edges" (K, 2) と、その本数 "boundary_edge_count"。
    三角形が1つも無いときは灰（空の形を「穴が0」で合格にしない）。
    """
    t = as_triangles(triangles)
    edges = boundary_edges(t)
    loops = 0
    if len(edges):
        used, compact = np.unique(edges.ravel(), return_inverse=True)
        compact = compact.reshape(-1, 2)
        ends = _UnionFind(len(used))
        ends.union(compact[:, 0], compact[:, 1])
        loops = len(np.unique(ends.roots()))
    return _count_measurement(HOLES_NAME, loops, Signal.RED, "閉じているはずの形に穴があります",
                              {"edges": edges, "boundary_edge_count": len(edges)},
                              empty_reason="" if len(t) else _NO_TRIANGLES)


def summarize_loose_parts(vertices: Any, triangles: Any, *, max_faces: int = LOOSE_PART_MAX_FACES) -> Measurement:
    """小さな離れた部品（名前 "loose_parts"）。値は部品の数。1つでもあれば黄（消すかどうかはユーザーが決める）。
    三角形が1つも無いときは灰"""
    v = as_points(vertices, "vertices")
    t = as_triangles(triangles, len(v))
    parts = loose_parts(v, t, max_faces=max_faces)
    return _count_measurement(LOOSE_PARTS_NAME, len(parts), Signal.YELLOW, "小さな離れた部品があります",
                              {"parts": parts, "max_faces": max_faces}, empty_reason="" if len(t) else _NO_TRIANGLES)


def summarize_mesh(vertices: Any, triangles: Any, *, closed_required: bool = False) -> list[Measurement]:
    """面の点検をまとめて行う。closed_required=True（体など閉じているはずの形）のときは、穴も調べる。
    三角形が1つも無いときは、三角形を使う点検をすべて灰にする（何も調べていないので、緑にしない）"""
    if not isinstance(closed_required, (bool, np.bool_)):
        raise ValueError("closed_required は True か False にしてください")
    v = as_points(vertices, "vertices")
    t = as_triangles(triangles, len(v))
    results = [summarize_duplicates(v), summarize_degenerate(v, t), summarize_non_manifold(t),
               summarize_inconsistent(t), summarize_loose_parts(v, t)]
    if closed_required:
        results.append(summarize_holes(t))
    return results

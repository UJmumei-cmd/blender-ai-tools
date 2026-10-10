"""E2 往復の確認（bpy なし・numpy だけ）。

書き出した USDZ（convert_blend.py の character.usdz）を読み戻した物と、Blender の元のモデルを比べる。
ロードマップ段階1 基準4：いくつかの姿勢で、読み戻した形のずれが thresholds.ROUNDTRIP_MAX_DEVIATION（0.5mm）未満。
名前は character_rig.json の originalNames を通して比べる。1点に付く骨は4本まで。物理の部品が入っていない。

USD を読む所（pxr）は usd_read_bpy.py、Blender から形とポーズを取り出して比べる所は roundtrip_bpy.py。
ここは、その間の計算だけを持つ（Blender の外でもテストできる）。

決まり：
- 比べる空間は「骨格の空間」＝ Blender のアーマチュアのオブジェクトの中の座標（Z が上。アーマチュアのオブジェクトに
  大きさが無ければメートル）。
  convert_blend.py の USD では、骨格（Skeleton）の bindTransforms が Blender の bone.matrix_local と同じ空間にあり、
  Z が上から Y が上への向きの変え方は、いちばん上の /root の Xform にだけ入っている。
- 長さ（ずれ・対応の距離）は、アプリで見える長さ（メートル）で測る（skeleton_to_meters）。骨格の空間の 1 は、
  アーマチュアのオブジェクトに大きさ（スケール）があると 1m ではない。アプリは骨格の空間を、骨格の位置の行列
  （skeleton_world）と metersPerUnit で大きくして見せるので、骨格の空間の差を、その行列の回転・大きさの部分で移してから測る。
- 行列は (…, 4, 4) の float64 で、列ベクトルに左から掛ける形（平行移動は右の列）。
  USD（Gf）の行列は行ベクトルの形（平行移動は下の行）なので、読むときに転置する（gf_to_column）。
- USD の骨格の決まり（UsdSkel）：
  - 骨の親は、骨の場所（"a/b/c"）の先祖のうち、骨の一覧にある、いちばん近い物（joint_parents）。
    一覧に無い先祖（"a/b/c" の "b"）は飛ばす。
  - restTransforms は、親の骨から見た休みの姿勢。bindTransforms は、骨格の空間での休みの姿勢。
    アプリ（RealityKit）は休みの姿勢を restTransforms から組み立てるので、組み立てた物が bindTransforms と
    違うと、何もしなくても形がずれる（check_skeleton）。
  - 骨で動かした点 = Σ_k w_k (posed[j_k] @ inv(bind[j_k])) @ (geomBindTransform @ p)。
    骨で動く形では、メッシュ自身の位置の行列は使わない（結果は骨格の空間に出る）。
- 書き出した点と元の頂点の対応（match_points）は、休みの姿勢の位置で決める。
  convert_blend.py は頂点の並びを変えない（服で隠す MASK で消した頂点だけが抜ける）ので、まず並びのまま対応させ、
  合わないときだけ、いちばん近い点で対応させる。
- 信号：測れなかった点・骨があるときは、緑にしない（灰。CLAUDE.md 2章5）。
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import bones, thresholds
from ..thresholds import ORDER_MATCH_MIN_FRACTION, POSE_MOVED_TOLERANCE, SKELETON_MATRIX_TOLERANCE
from ..measure.geometry import (WORST_LIST_LENGTH, as_array, as_nonnegative_float, as_points, as_triangles,
                                weld_exact, worst_indices)
from ..measure.results import Measurement, Signal, classify, worst
from ..measure.symmetry import nearest_distances
from ..measure.weights import WeightTable, linear_blend_skinning

__all__ = [
    "COVERAGE_NAME",
    "DEVIATION_NAME",
    "EXPECTED_PRIM_TYPES",
    "HUMANOID_NAME",
    "INFLUENCES_NAME",
    "JOINT_NAMES_NAME",
    "PHYSICS_NAME",
    "REST_POSE",
    "SHAPE_KEYS_NAME",
    "SKELETON_NAME",
    "TEST_POSES",
    "WEIGHT_SUMS_NAME",
    "ExportedBlendShape",
    "ExportedMesh",
    "ExportedModel",
    "PointMatch",
    "check_coverage",
    "check_influences",
    "check_names",
    "check_no_physics",
    "check_skeleton",
    "compare_pose",
    "compare_rest",
    "compose_rest",
    "exported_weight_table",
    "fan_triangulate",
    "gf_to_column",
    "joint_parents",
    "load_rig",
    "local_rotation",
    "match_meshes",
    "match_points",
    "original_names",
    "path_gaps",
    "posed_joint_matrices",
    "resolve_pose",
    "rest_points",
    "skeleton_to_meters",
    "skin_exported",
]

# MARK: - 決まり

# 測った結果の名前
DEVIATION_NAME = "roundtrip_deviation"
COVERAGE_NAME = "roundtrip_coverage"
SKELETON_NAME = "skeleton_rest_consistency"
JOINT_NAMES_NAME = "joint_names"
HUMANOID_NAME = "humanoid_bones"
SHAPE_KEYS_NAME = "shape_key_names"
INFLUENCES_NAME = "max_bone_influences"
WEIGHT_SUMS_NAME = "joint_weight_sums"
PHYSICS_NAME = "physics_parts"

# 休みの姿勢の名前（compare_rest の details["pose"]）
REST_POSE = "rest"

# 書き出しに入っていてよい USD の部品の種類。ほかの種類は other_prims に入れて、check_no_physics で調べる。
# SkelAnimation は、Blender がシェイプキーの値を書くために、動きを書き出さないときでも作る（Blender 5.2.2 で確かめた）
EXPECTED_PRIM_TYPES = frozenset({
    "Xform", "Scope", "SkelRoot", "Skeleton", "SkelAnimation", "Mesh", "GeomSubset", "BlendShape",
    "Material", "Shader", "NodeGraph",
})
# 物理の部品の印：この字で始まる USD の種類と、当てた API（UsdPhysics の PhysicsRigidBodyAPI など）
PHYSICS_PREFIX = "Physics"

# 確かめのポーズ：ポーズの名前 → {共通の名前（MMD 準標準）: (アーマチュアの空間での回す軸, 角度（度））}。
# 回す軸は、休みの姿勢での向き（Z が上、モデルは -Y を向く、キャラの左は +X）。骨の根元を中心に回す。
# ひざは、ふつうの向き（足先が後ろの +Y へ行く向き）にだけ曲げる（逆曲がりにしない）。
# 準標準のモデルでは、足・ひざ を回すと、足D・ひざD が回転付与で同じだけ回る
TEST_POSES: dict[str, dict[str, tuple[tuple[float, float, float], float]]] = {
    "arms_up": {"左腕": ((0.0, -1.0, 0.0), 60.0), "右腕": ((0.0, 1.0, 0.0), 60.0)},
    "elbows": {"左ひじ": ((0.0, 0.0, -1.0), 90.0), "右ひじ": ((0.0, 0.0, 1.0), 90.0)},
    "knees": {"左足": ((-1.0, 0.0, 0.0), 45.0), "右足": ((-1.0, 0.0, 0.0), 45.0),
              "左ひざ": ((1.0, 0.0, 0.0), 60.0), "右ひざ": ((1.0, 0.0, 0.0), 60.0)},
    "twist": {"上半身": ((0.0, 0.0, 1.0), 30.0), "首": ((0.0, 0.0, 1.0), 20.0)},
}

# MARK: - 計算のための数字（判定の基準ではない）

# 行列の下の行が (0, 0, 0, 1) からずれてよい幅。USD の float の誤差は許し、行列の転置のまちがいは見つける
MATRIX_BOTTOM_ROW_TOLERANCE = 1e-6

# 基準の数字（thresholds.py）：ORDER_MATCH_MIN_FRACTION（並びのまま対応させる割合）、
# SKELETON_MATRIX_TOLERANCE（休みの姿勢の行列の差）、POSE_MOVED_TOLERANCE（ポーズで骨が動いたとみなす差）


# MARK: - 入力の確かめ


def _as_matrix(value: Any, name: str) -> np.ndarray:
    """(4, 4) の行列にする（列ベクトルの形）。下の行が (0, 0, 0, 1) でないと ValueError。下の行はぴったりにそろえる"""
    array = as_array(value, name)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} は数の並びにしてください（いまの型：{array.dtype}）")
    array = np.array(array, dtype=np.float64)
    if array.shape != (4, 4):
        raise ValueError(f"{name} は (4, 4) の行列にしてください（いまの形：{array.shape}）")
    return _check_affine(array[None], name)[0]


def _as_matrices(value: Any, name: str, count: int | None = None, *, allow_nan_rows: bool = False) -> np.ndarray:
    """(J, 4, 4) の行列の並びにする（列ベクトルの形）。

    allow_nan_rows=True のときは、行列全体が NaN の物（分からない骨）を受け付ける。それ以外の NaN・無限大は ValueError
    """
    array = as_array(value, name)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} は数の並びにしてください（いまの型：{array.dtype}）")
    array = np.array(array, dtype=np.float64)
    if array.ndim == 1 and array.size == 0:
        array = array.reshape(0, 4, 4)
    if array.ndim != 3 or array.shape[1:] != (4, 4):
        raise ValueError(f"{name} は (J, 4, 4) の並びにしてください（いまの形：{array.shape}）")
    if count is not None and len(array) != count:
        raise ValueError(f"{name} の数（{len(array)}）が骨の数（{count}）と違います")
    unknown = np.isnan(array).all(axis=(1, 2))
    if unknown.any() and not allow_nan_rows:
        raise ValueError(f"{name} に NaN の行列があります")
    known = ~unknown
    array[known] = _check_affine(array[known], name)
    return array


def _check_affine(array: np.ndarray, name: str) -> np.ndarray:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} に NaN か無限大が入っています")
    bottom = np.array([0.0, 0.0, 0.0, 1.0])
    if (np.abs(array[:, 3, :] - bottom) > MATRIX_BOTTOM_ROW_TOLERANCE).any():
        raise ValueError(f"{name} の下の行が (0, 0, 0, 1) ではありません（USD の行列を転置していませんか）")
    array[:, 3, :] = bottom
    return array


def _readonly(array: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(array)
    array.flags.writeable = False
    return array


def _as_names(value: Any, name: str) -> tuple[str, ...]:
    """名前の並びにする。1つの文字列・文字列でない物は ValueError"""
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f"{name} は名前（文字列）の並びにしてください")
    names = tuple(value)
    for item in names:
        if not isinstance(item, str):
            raise ValueError(f"{name} の中身は文字列にしてください（いまの値：{item!r}）")
    return names


def _transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(N, 3) の点に、列ベクトルの形の 4×4 の行列を掛ける"""
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def gf_to_column(rows: Any) -> np.ndarray:
    """USD（Gf）の行ベクトルの形の行列 (…, 4, 4)（平行移動は下の行）を、列ベクトルの形（平行移動は右の列）にする。

    Gf の行列 M は、点を v' = v M（行ベクトルに右から掛ける）で動かす。列ベクトルの形は、その転置。
    """
    array = as_array(rows, "rows")
    if array.dtype.kind not in "iuf":
        raise ValueError(f"rows は数の並びにしてください（いまの型：{array.dtype}）")
    array = np.asarray(array, dtype=np.float64)
    if array.ndim < 2 or array.shape[-2:] != (4, 4):
        raise ValueError(f"rows は (…, 4, 4) の並びにしてください（いまの形：{array.shape}）")
    return np.ascontiguousarray(np.swapaxes(array, -1, -2))


# MARK: - 骨格の並び（UsdSkel の決まり）


def _path_parts(path: str) -> list[str]:
    parts = path.split("/")
    if not path or any(not part for part in parts):
        raise ValueError(f"骨の場所の形が正しくありません：{path!r}")
    return parts


def joint_parents(paths: Sequence[str]) -> np.ndarray:
    """骨の親の番号 (J,) int64（根は -1）。UsdSkelTopology と同じ決まり：

    骨の場所（"a/b/c"）の先祖を近い順に調べ、骨の一覧にある最初の物を親にする（"a/b" が無ければ "a"）。
    同じ場所が2回ある・形の正しくない場所は ValueError。
    """
    names = _as_names(paths, "paths")
    lookup: dict[str, int] = {}
    for index, path in enumerate(names):
        _path_parts(path)
        if path in lookup:
            raise ValueError(f"骨の場所 {path!r} が2回あります")
        lookup[path] = index
    parents = np.full(len(names), -1, dtype=np.int64)
    for index, path in enumerate(names):
        parts = path.split("/")
        for depth in range(len(parts) - 1, 0, -1):
            found = lookup.get("/".join(parts[:depth]))
            if found is not None:
                parents[index] = found
                break
    return parents


def path_gaps(paths: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """骨の場所の途中にあるのに、骨の一覧に無い物。骨の場所 → 抜けている途中の名前（根に近い順）。

    例：一覧が "a"・"a/b/c" なら {"a/b/c": ("b",)}。抜けた骨は、休みの姿勢（restTransforms）の親が
    書き出した側と読む側で食い違う元になる。抜けが無い骨は入れない。
    """
    names = _as_names(paths, "paths")
    known = set(names)
    result: dict[str, tuple[str, ...]] = {}
    for path in names:
        parts = _path_parts(path)
        missing: list[str] = []
        for depth in range(len(parts) - 1, 0, -1):
            if "/".join(parts[:depth]) in known:
                break
            missing.append(parts[depth - 1])
        if missing:
            result[path] = tuple(reversed(missing))
    return result


def compose_rest(rest_local: Any, parents: Any) -> np.ndarray:
    """restTransforms（親から見た休みの姿勢）を、親から順に掛けて、骨格の空間の休みの姿勢 (J, 4, 4) にする。

    world[j] = world[parent[j]] @ rest_local[j]（根は rest_local[j] のまま）。
    UsdSkel では、親は子より前に並ぶ決まり。親が子より後ろ（自分も含む）なら ValueError。
    """
    local = _as_matrices(rest_local, "rest_local")
    parent = as_array(parents, "parents")
    if parent.dtype.kind not in "iu" or parent.shape != (len(local),):
        raise ValueError(f"parents は ({len(local)},) の整数の並びにしてください（いまの形：{parent.shape}）")
    world = np.empty_like(local)
    for index in range(len(local)):
        p = int(parent[index])
        if p >= index:
            raise ValueError(f"骨の並びが USD の決まりに合いません（骨 {index} の親 {p} が、子より前にありません）")
        world[index] = local[index] if p < 0 else world[p] @ local[index]
    return world


def fan_triangulate(counts: Any, indices: Any, point_count: int, *, left_handed: bool = False) -> np.ndarray:
    """USD の面（faceVertexCounts・faceVertexIndices）を三角形 (M, 3) int64 にする。

    n 角形 (v0, v1, …, vn-1) は (v0, vi, vi+1)（i = 1 … n-2）に分ける（扇の形。どの面も同じ分け方）。
    left_handed（USD の orientation が "leftHanded"）のときは、回る向きを逆にして、反時計回りが表になるようにする。
    3 より少ない頂点の面は、面積が無いので飛ばす。数が合わない・番号が範囲の外のときは ValueError。
    """
    c = as_array(counts, "counts")
    i = as_array(indices, "indices")
    if c.size == 0 and i.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    if c.dtype.kind not in "iu" or i.dtype.kind not in "iu" or c.ndim != 1 or i.ndim != 1:
        raise ValueError("faceVertexCounts と faceVertexIndices は、整数の1次元の並びにしてください")
    c = c.astype(np.int64)
    i = i.astype(np.int64)
    if (c < 0).any() or int(c.sum()) != len(i):
        raise ValueError(f"faceVertexCounts の合計（{int(c.sum())}）が faceVertexIndices の数（{len(i)}）と合いません")
    starts = np.cumsum(c) - c
    fans = np.maximum(c - 2, 0)
    face = np.repeat(np.arange(len(c)), fans)
    k = np.arange(int(fans.sum())) - np.repeat(np.cumsum(fans) - fans, fans) + 1
    first = i[starts[face]]
    second = i[starts[face] + k]
    third = i[starts[face] + k + 1]
    triangles = np.stack([first, third, second] if left_handed else [first, second, third], axis=1)
    return as_triangles(triangles, point_count) if len(triangles) else np.zeros((0, 3), dtype=np.int64)


# MARK: - 書き出した物の形


@dataclass(frozen=True, eq=False)
class ExportedBlendShape:
    """書き出したシェイプキー（USD の BlendShape）。

    name: 名前（メッシュの skel:blendShapes の名前。書き出した ASCII の名前）
    offsets: (P, 3) 基本の形からの差（メッシュの中の座標）
    point_indices: (P,) int64 差のある点の番号。None なら全部の点（P = 頂点の数）
    """

    name: str
    offsets: np.ndarray
    point_indices: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ValueError("シェイプキーの名前は文字列にしてください")
        object.__setattr__(self, "offsets", _readonly(as_points(self.offsets, "offsets")))
        if self.point_indices is not None:
            indices = as_array(self.point_indices, "point_indices")
            if indices.size == 0:
                indices = np.zeros(0, dtype=np.int64)
            if indices.dtype.kind not in "iu" or indices.shape != (len(self.offsets),):
                raise ValueError(f"point_indices は offsets と同じ数の整数にしてください（いまの形：{indices.shape}）")
            if len(indices) and int(indices.min()) < 0:
                raise ValueError("point_indices に負の番号があります")
            object.__setattr__(self, "point_indices", _readonly(indices.astype(np.int64)))


@dataclass(frozen=True, eq=False)
class ExportedMesh:
    """書き出した形（USD の Mesh）1つ。

    path: USD の中の場所（"/root/TestArmature/Body/Body" など）
    name: USD の部品の名前
    points: (N, 3) 頂点（メッシュの中の座標。USD の points）
    triangles: (M, 3) int64 三角形（fan_triangulate。反時計回りが表）
    geom_bind: (4, 4) skel:geomBindTransform（メッシュの中の座標 → 骨格の空間。無ければ単位行列）
    joint_indices: (N, K) int64 頂点ごとの骨の番号（この形の骨の並び joint_names の番号。空きの枠は重み 0）
    joint_weights: (N, K) float64 重み
    joint_names: この形の骨の並び（skel:joints があればその並び、無ければ骨格の並び）の名前（場所の最後の部分）
    joint_map: (len(joint_names),) int64 この形の骨の番号 → 骨格の骨の番号（骨格に無い骨は -1）。None なら骨格と同じ並び
    skinned: 骨で動く形か（骨格に結び付いて、骨の重みがある）。False なら points を local_to_skeleton で置くだけ
    local_to_skeleton: (4, 4) 骨で動かない形の、メッシュの中の座標 → 骨格の空間
    source_name: Blender の元のオブジェクトの名前（USD の userProperties:blender:object_name など。無ければ name）
    blend_shapes: 書き出したシェイプキー
    """

    path: str
    name: str
    points: np.ndarray
    triangles: np.ndarray
    geom_bind: np.ndarray | None = None
    joint_indices: np.ndarray | None = None
    joint_weights: np.ndarray | None = None
    joint_names: tuple[str, ...] = ()
    joint_map: np.ndarray | None = None
    skinned: bool = True
    local_to_skeleton: np.ndarray | None = None
    source_name: str = ""
    blend_shapes: tuple[ExportedBlendShape, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not isinstance(self.name, str):
            raise ValueError("path と name は文字列にしてください")
        points = as_points(self.points, "points")
        count = len(points)
        object.__setattr__(self, "points", _readonly(points))
        object.__setattr__(self, "triangles", _readonly(as_triangles(self.triangles, count)))
        for key in ("geom_bind", "local_to_skeleton"):
            value = getattr(self, key)
            matrix = np.eye(4) if value is None else _as_matrix(value, key)
            object.__setattr__(self, key, _readonly(matrix))
        ids = np.zeros((count, 0), dtype=np.int64) if self.joint_indices is None else as_array(self.joint_indices,
                                                                                               "joint_indices")
        weights = np.zeros((count, 0)) if self.joint_weights is None else as_array(self.joint_weights,
                                                                                  "joint_weights")
        if ids.size == 0 and ids.ndim != 2:
            ids = np.zeros((count, 0), dtype=np.int64)
        if weights.size == 0 and weights.ndim != 2:
            weights = np.zeros((count, 0))
        if ids.dtype.kind not in "iu" or ids.ndim != 2 or ids.shape[0] != count:
            raise ValueError(f"joint_indices は ({count}, K) の整数の並びにしてください（いまの形：{ids.shape}）")
        if weights.dtype.kind not in "iuf" or weights.shape != ids.shape:
            raise ValueError(f"joint_weights は joint_indices と同じ形 {ids.shape} の数の並びにしてください"
                             f"（いまの形：{weights.shape}）")
        weights = np.asarray(weights, dtype=np.float64)
        if not np.isfinite(weights).all():
            raise ValueError("joint_weights に NaN か無限大が入っています")
        ids = ids.astype(np.int64)
        names = _as_names(self.joint_names, "joint_names")
        if ids.size and int(ids.min()) < 0:
            raise ValueError("joint_indices に負の番号があります")
        if self.joint_map is not None:
            mapping = as_array(self.joint_map, "joint_map")
            if mapping.size == 0:
                mapping = np.zeros(0, dtype=np.int64)
            if mapping.dtype.kind not in "iu" or mapping.shape != (len(names),):
                raise ValueError(f"joint_map は ({len(names)},) の整数の並びにしてください（いまの形：{mapping.shape}）")
            if len(mapping) and int(mapping.min()) < -1:
                raise ValueError("joint_map の番号は -1 以上にしてください（骨格に無い骨は -1）")
            object.__setattr__(self, "joint_map", _readonly(mapping.astype(np.int64)))
        limit = len(names) if (self.joint_map is not None or names) else None
        if limit is not None and ids.size and int(ids.max()) >= limit:
            raise ValueError(f"joint_indices に、骨の数（{limit}）以上の番号があります")
        object.__setattr__(self, "joint_indices", _readonly(ids))
        object.__setattr__(self, "joint_weights", _readonly(weights))
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "skinned", bool(self.skinned))
        if not isinstance(self.source_name, str):
            raise ValueError("source_name は文字列にしてください")
        if not self.source_name:
            object.__setattr__(self, "source_name", self.name)
        shapes = tuple(self.blend_shapes)
        for shape in shapes:
            if not isinstance(shape, ExportedBlendShape):
                raise ValueError("blend_shapes の中身は ExportedBlendShape にしてください")
            if shape.point_indices is None and len(shape.offsets) != count:
                raise ValueError(f"シェイプキー {shape.name} の offsets の数が、頂点の数（{count}）と違います")
            if shape.point_indices is not None and len(shape.point_indices) and int(shape.point_indices.max()) >= count:
                raise ValueError(f"シェイプキー {shape.name} の point_indices が頂点の数（{count}）を超えています")
        object.__setattr__(self, "blend_shapes", shapes)

    @property
    def vertex_count(self) -> int:
        """頂点の数 N"""
        return int(len(self.points))

    @property
    def element_size(self) -> int:
        """1つの頂点の骨の枠の数 K（USD の elementSize）"""
        return int(self.joint_indices.shape[1])


@dataclass(frozen=True, eq=False)
class ExportedModel:
    """書き出したファイル（USDZ）全体。

    joint_paths: 骨格の骨の場所（Skeleton の joints。"hips"・"hips/spine" など）
    bind: (J, 4, 4) bindTransforms（骨格の空間。列ベクトルの形）
    rest_local: (J, 4, 4) restTransforms（親の骨から見た休みの姿勢）。書かれていなければ None
    meshes: 形（Mesh）の並び（USD の中の順）
    path: 読んだファイルの場所
    up_axis・meters_per_unit: USD の上の向き（"Y" など）と、1 の長さ（メートル。0 より大きい有限の数）
    root_matrix: (4, 4) いちばん上の部品（defaultPrim。convert_blend.py では /root）の、世界への行列
    skeleton_path: 骨格（Skeleton）の場所
    skeleton_world: (4, 4) 骨格の空間 → USD の世界（Y が上）の行列（つぶれていない物。長さを測るときに使う）
    prims: 全部の部品 ((場所, 種類), …)（USD の中の順。種類の無い部品は ""）
    other_prims: EXPECTED_PRIM_TYPES に無い種類の部品 ((場所, 種類), …)
    physics_schemas: 物理の API（PhysicsRigidBodyAPI など）を当てた部品 ((場所, API の名前), …)
    object_names: 部品の場所 → Blender のオブジェクトの名前（userProperties:blender:object_name がある部品だけ）
    joint_names: 骨の名前（骨の場所の最後の部分。作るときに決まる）
    joint_parents: (J,) int64 骨の親の番号（joint_parents。作るときに決まる）
    """

    joint_paths: tuple[str, ...]
    bind: np.ndarray
    rest_local: np.ndarray | None
    meshes: tuple[ExportedMesh, ...]
    path: str = ""
    up_axis: str = "Y"
    meters_per_unit: float = 1.0
    root_matrix: np.ndarray | None = None
    skeleton_path: str = ""
    skeleton_world: np.ndarray | None = None
    prims: tuple[tuple[str, str], ...] = ()
    other_prims: tuple[tuple[str, str], ...] = ()
    physics_schemas: tuple[tuple[str, str], ...] = ()
    object_names: Mapping[str, str] = field(default_factory=dict)
    joint_names: tuple[str, ...] = field(init=False)
    joint_parents: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        paths = _as_names(self.joint_paths, "joint_paths")
        parents = joint_parents(paths)
        object.__setattr__(self, "joint_paths", paths)
        object.__setattr__(self, "joint_names", tuple(path.split("/")[-1] for path in paths))
        object.__setattr__(self, "joint_parents", _readonly(parents))
        count = len(paths)
        object.__setattr__(self, "bind", _readonly(_as_matrices(self.bind, "bind", count)))
        rest = self.rest_local
        if rest is not None:
            rest = as_array(rest, "rest_local")
            rest = None if rest.size == 0 and count else _readonly(_as_matrices(rest, "rest_local", count))
        object.__setattr__(self, "rest_local", rest)
        for key in ("root_matrix", "skeleton_world"):
            value = getattr(self, key)
            object.__setattr__(self, key, _readonly(np.eye(4) if value is None else _as_matrix(value, key)))
        meshes = tuple(self.meshes)
        for mesh in meshes:
            if not isinstance(mesh, ExportedMesh):
                raise ValueError("meshes の中身は ExportedMesh にしてください")
            if mesh.joint_map is not None:
                if len(mesh.joint_map) and int(mesh.joint_map.max()) >= count:
                    raise ValueError(f"{mesh.path} の joint_map に、骨格の骨の数（{count}）以上の番号があります")
            elif mesh.joint_indices.size and int(mesh.joint_indices.max()) >= count:
                raise ValueError(f"{mesh.path} の joint_indices に、骨格の骨の数（{count}）以上の番号があります")
        object.__setattr__(self, "meshes", meshes)
        if not isinstance(self.up_axis, str) or not isinstance(self.path, str) or not isinstance(self.skeleton_path,
                                                                                                    str):
            raise ValueError("up_axis・path・skeleton_path は文字列にしてください")
        if isinstance(self.meters_per_unit, bool) or not isinstance(self.meters_per_unit,
                                                                    (int, float, np.integer, np.floating)):
            raise ValueError(f"meters_per_unit は数にしてください（いまの値：{self.meters_per_unit!r}）")
        meters = float(self.meters_per_unit)
        if not math.isfinite(meters) or meters <= 0.0:
            raise ValueError(f"meters_per_unit は 0 より大きい有限の数にしてください（いまの値：{meters!r}）")
        object.__setattr__(self, "meters_per_unit", meters)
        if not np.isfinite(np.linalg.cond(self.skeleton_world[:3, :3])):
            raise ValueError("skeleton_world がつぶれています（長さを測れません）")
        object.__setattr__(self, "prims", tuple((str(p), str(t)) for p, t in self.prims))
        object.__setattr__(self, "other_prims", tuple((str(p), str(t)) for p, t in self.other_prims))
        object.__setattr__(self, "physics_schemas", tuple((str(p), str(s)) for p, s in self.physics_schemas))
        object.__setattr__(self, "object_names", {str(k): str(v) for k, v in dict(self.object_names).items()})

    @property
    def joint_count(self) -> int:
        """骨格の骨の数 J"""
        return len(self.joint_paths)


def _as_model(value: Any) -> ExportedModel:
    if not isinstance(value, ExportedModel):
        raise ValueError(f"exported は ExportedModel にしてください（いまの型：{type(value).__name__}）")
    return value


def _as_mesh(value: Any) -> ExportedMesh:
    if not isinstance(value, ExportedMesh):
        raise ValueError(f"mesh は ExportedMesh にしてください（いまの型：{type(value).__name__}）")
    return value


def skeleton_to_meters(exported: ExportedModel) -> np.ndarray:
    """骨格の空間の差（ベクトル）を、アプリで見える長さ（メートル）の差にする (3, 3) の行列。

    = skeleton_world の回転・大きさの部分 × meters_per_unit。アーマチュアのオブジェクトの大きさが 2 なら、
    骨格の空間の 0.3mm の差は、アプリでは 0.6mm（Blender の世界の長さも同じ）。位置（平行移動）は長さを変えないので使わない。
    """
    model = _as_model(exported)
    return np.ascontiguousarray(model.skeleton_world[:3, :3] * model.meters_per_unit)


def _lengths(to_meters: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """骨格の空間の差 (N, 3) の、アプリで見える長さ (N,)（メートル）"""
    return np.sqrt(((vectors @ to_meters.T) ** 2).sum(axis=1))


# MARK: - 骨で動かす


def _skeleton_ids(mesh: ExportedMesh) -> np.ndarray:
    """枠ごとの、骨格の骨の番号 (N, K)（骨格に無い骨は -1）"""
    if mesh.joint_map is None:
        return mesh.joint_indices
    if len(mesh.joint_map) == 0:
        return np.full(mesh.joint_indices.shape, -1, dtype=np.int64)
    return mesh.joint_map[mesh.joint_indices]


def exported_weight_table(mesh: ExportedMesh, joint_count: int) -> WeightTable:
    """書き出したウェイトを、骨格の骨の番号の WeightTable にする（重みが 0 の枠は空きにする）。

    USD は空きの枠を「番号 0・重み 0」で埋めるので、重みが 0 の枠は使わない。
    重みが 0 でないのに、骨格に無い骨・joint_count 以上の番号を指す枠があれば ValueError。
    """
    m = _as_mesh(mesh)
    if isinstance(joint_count, bool) or not isinstance(joint_count, (int, np.integer)) or joint_count < 0:
        raise ValueError(f"joint_count は 0 以上の整数にしてください（いまの値：{joint_count!r}）")
    ids = _skeleton_ids(m)
    used = m.joint_weights != 0.0
    bad = used & ((ids < 0) | (ids >= joint_count))
    if bad.any():
        raise ValueError(f"{m.path}：骨格に無い骨を指す重みがあります（頂点 {np.flatnonzero(bad.any(axis=1))[:10].tolist()}）")
    return WeightTable(np.where(used, ids, -1), np.where(used, m.joint_weights, 0.0))


def rest_points(mesh: ExportedMesh) -> np.ndarray:
    """休みの姿勢の点 (N, 3)（骨格の空間）。骨で動く形は geom_bind @ p、動かない形は local_to_skeleton @ p"""
    m = _as_mesh(mesh)
    return _transform(m.geom_bind if m.skinned else m.local_to_skeleton, m.points)


def _inverse(matrices: np.ndarray, name: str) -> np.ndarray:
    try:
        inverse = np.linalg.inv(matrices)
    except np.linalg.LinAlgError:
        raise ValueError(f"{name} に、逆行列の無い（つぶれた）行列があります") from None
    if len(inverse):
        inverse[:, 3, :] = (0.0, 0.0, 0.0, 1.0)
    return inverse


def _skinning(bind: np.ndarray, posed: np.ndarray) -> np.ndarray:
    """休みの姿勢からポーズへ動かす行列 (J, 4, 4) = posed @ inv(bind)"""
    skin = posed @ _inverse(bind, "bind")
    if len(skin):
        skin[:, 3, :] = (0.0, 0.0, 0.0, 1.0)
    return skin


def skin_exported(mesh: ExportedMesh, bind: Any, posed: Any) -> np.ndarray:
    """書き出した形を、骨の行列で動かした点 (N, 3)（骨格の空間）。UsdSkel の線形ブレンドスキニングと同じ計算：

        p_skel = geom_bind @ p
        動いた点 = Σ_k w_k (posed[j_k] @ inv(bind[j_k])) @ p_skel

    bind・posed：骨格の骨ごとの (J, 4, 4)（列ベクトルの形。骨格の空間）。posed = bind なら休みの姿勢（p_skel）になる。
    重みは書き出されたまま使う（合計で割らない。合計は check_influences で確かめる）。重みが 0 の枠は使わない。
    骨で動かない形（skinned が False）は、local_to_skeleton で置いた点を返す（ポーズで動かない）。
    """
    m = _as_mesh(mesh)
    b = _as_matrices(bind, "bind")
    p = _as_matrices(posed, "posed", len(b))
    if not m.skinned:
        return rest_points(m)
    table = exported_weight_table(m, len(b))
    if len(b) == 0:
        return np.zeros((m.vertex_count, 3))
    return linear_blend_skinning(rest_points(m), table, _skinning(b, p))


def _skin_known(mesh: ExportedMesh, bind: np.ndarray, posed: np.ndarray,
                inverse_bind: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """skin_exported と同じ計算。ただし、行列が分からない骨（posed が NaN）・骨格に無い骨に重みのある頂点は、
    測れない点として印を付ける。返す物：(点 (N, 3), 測れない点の印 (N,))"""
    count = mesh.vertex_count
    if not mesh.skinned:
        return rest_points(mesh), np.zeros(count, dtype=bool)
    ids = _skeleton_ids(mesh)
    used = mesh.joint_weights != 0.0
    known = ~np.isnan(posed).all(axis=(1, 2))
    valid = (ids >= 0) & (ids < len(bind))
    good = valid & known[np.clip(ids, 0, max(len(bind) - 1, 0))] if len(bind) else np.zeros_like(valid)
    unknown = (used & ~good).any(axis=1)
    keep = used & good
    table = WeightTable(np.where(keep, ids, -1), np.where(keep, mesh.joint_weights, 0.0))
    if len(bind) == 0:
        return np.zeros((count, 3)), unknown | used.any(axis=1)
    filled = np.where(known[:, None, None], posed, bind)
    skin = filled @ inverse_bind
    skin[:, 3, :] = (0.0, 0.0, 0.0, 1.0)
    return linear_blend_skinning(rest_points(mesh), table, skin), unknown


# MARK: - 点の対応


@dataclass(frozen=True, eq=False)
class PointMatch:
    """書き出した点と、元の頂点の対応。

    index: (P,) int64 書き出した点ごとの、元の頂点の番号（対応が無ければ -1）
    distance: (P,) float64 対応した元の頂点までの距離（"nearest" で対応が無い点は、いちばん近い頂点までの距離）
    unmatched_source: (k,) int64 どの書き出した点にも対応しなかった元の頂点（小さい順）
    method: "order"（並びのまま）か "nearest"（いちばん近い点）
    source_count: 元の頂点の数
    """

    index: np.ndarray
    distance: np.ndarray
    unmatched_source: np.ndarray
    method: str
    source_count: int

    @property
    def matched(self) -> np.ndarray:
        """書き出した点ごとの、対応があるかの印 (P,)"""
        return self.index >= 0


def _make_match(index: np.ndarray, distance: np.ndarray, method: str, source_count: int) -> PointMatch:
    used = np.zeros(source_count, dtype=bool)
    used[index[index >= 0]] = True
    return PointMatch(index=_readonly(index.astype(np.int64)), distance=_readonly(distance.astype(np.float64)),
                      unmatched_source=_readonly(np.flatnonzero(~used).astype(np.int64)), method=method,
                      source_count=source_count)


def match_points(source: Any, exported: Any, *, distance: float = thresholds.ROUNDTRIP_MATCH_DISTANCE,
                 candidates: Any = None) -> PointMatch:
    """書き出した点 (P, 3) を、元の頂点 (S, 3) に対応させる（どちらも休みの姿勢、同じ空間）。

    candidates：書き出されるはずの元の頂点の印 (S,) bool（MASK で消える頂点は False）。None なら全部。
    1. 書き出した点の数が candidates の数と同じなら、並びのまま対応させる（convert_blend.py は並びを変えない）。
       distance の中に入る点が ORDER_MATCH_MIN_FRACTION 以上なら、それで決める。ずれた点も対応したまま残す
       （ずれは compare_rest で測る）。
    2. そうでなければ、全部の元の頂点から、いちばん近い点を探す（measure.symmetry.nearest_distances。同じ距離なら
       小さい番号）。distance より遠い点は対応なし（-1）。ただし 1 の並びのほうが多く合うときは 1 を使う
       （全体がずれた形を「対応なし」にせず、ずれとして測るため）。
    """
    s = as_points(source, "source")
    e = as_points(exported, "exported")
    tol = as_nonnegative_float(distance, "distance")
    if candidates is None:
        allowed = np.ones(len(s), dtype=bool)
    else:
        allowed = as_array(candidates, "candidates")
        if allowed.dtype.kind != "b" or allowed.shape != (len(s),):
            raise ValueError(f"candidates は ({len(s)},) の真偽の並びにしてください（いまの形：{allowed.shape}）")
    order_index = np.flatnonzero(allowed)
    order_fraction: float | None = None
    order_distance = None
    if len(order_index) == len(e):
        order_distance = np.sqrt(((e - s[order_index]) ** 2).sum(axis=1))
        order_fraction = float((order_distance <= tol).mean()) if len(e) else 1.0
        if order_fraction >= ORDER_MATCH_MIN_FRACTION:
            return _make_match(order_index, order_distance, "order", len(s))
    if len(e) == 0:
        return _make_match(np.zeros(0, dtype=np.int64), np.zeros(0), "nearest", len(s))
    if len(s) == 0:
        return _make_match(np.full(len(e), -1), np.full(len(e), np.inf), "nearest", len(s))
    near_distance, near_index = nearest_distances(e, s)
    near_ok = near_distance <= tol
    if order_fraction is not None and float(near_ok.mean()) <= order_fraction:
        return _make_match(order_index, order_distance, "order", len(s))
    index = _spread_coincident(s, np.where(near_ok, near_index, -1))
    return _make_match(index, near_distance, "nearest", len(s))


def _spread_coincident(source: np.ndarray, index: np.ndarray) -> np.ndarray:
    """いちばん近い点で対応させたとき、座標がまったく同じ元の頂点（UV の切れ目で分かれた頂点など）は、
    いちばん小さい番号にばかり対応する（nearest_distances は同じ距離なら小さい番号を選ぶ）。
    同じ頂点に対応した書き出した点を、書き出した点の順に、同じ座標の頂点へ小さい番号から1つずつ割り振る
    （頂点の数より多い分は、いちばん小さい番号のまま）"""
    matched = np.flatnonzero(index >= 0)
    if len(matched) == 0:
        return index
    representative = weld_exact(source)  # 同じ座標の頂点の中で、いちばん小さい番号
    members = np.argsort(representative, kind="stable")  # 同じ代表の頂点が、番号の小さい順に並ぶ
    sizes = np.bincount(representative, minlength=len(source))
    starts = np.cumsum(sizes) - sizes
    targets = index[matched]
    order = np.argsort(targets, kind="stable")
    sorted_targets = targets[order]
    first = np.r_[True, sorted_targets[1:] != sorted_targets[:-1]]
    rank = np.arange(len(order)) - np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
    spread = np.where(rank < sizes[sorted_targets], members[starts[sorted_targets] + np.minimum(
        rank, sizes[sorted_targets] - 1)], sorted_targets)
    result = index.copy()
    result[matched[order]] = spread
    return result


def _as_point_map(value: Any, name: str) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} は メッシュの名前 → 点 (N, 3) の辞書にしてください")
    result = {}
    for key, points in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{name} の鍵は、メッシュの名前（文字列）にしてください")
        result[key] = as_points(points, f"{name}[{key!r}]")
    return result


def _as_mask_map(value: Any, counts: Mapping[str, int], name: str) -> dict[str, np.ndarray]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} は メッシュの名前 → 真偽の並び (S,) の辞書にしてください")
    result = {}
    for key, mask in value.items():
        array = as_array(mask, f"{name}[{key!r}]")
        expected = counts.get(key)
        if array.dtype.kind != "b" or array.ndim != 1 or (expected is not None and len(array) != expected):
            raise ValueError(f"{name}[{key!r}] は ({expected},) の真偽の並びにしてください（いまの形：{array.shape}）")
        result[key] = array
    return result


def match_meshes(exported: ExportedModel, source_rest: Mapping[str, Any], *, expected_missing: Any = None,
                 distance: float = thresholds.ROUNDTRIP_MATCH_DISTANCE) -> dict[str, PointMatch]:
    """書き出した形ごとに、元のメッシュの頂点との対応を作る。返す物：書き出した形の場所（path）→ PointMatch。

    source_rest：元のメッシュの名前 → 休みの姿勢の頂点 (S, 3)（骨格の空間）。形は source_name で元のメッシュと組にする。
    元のメッシュが無い形・同じ元のメッシュの2つ目の形は入れない（check_coverage が報告する）。
    expected_missing：元のメッシュの名前 → MASK で消えるはずの頂点の印 (S,)。
    対応の距離（distance と PointMatch.distance）は、アプリで見える長さ（メートル。skeleton_to_meters）。
    """
    model = _as_model(exported)
    sources = _as_point_map(source_rest, "source_rest")
    hidden = _as_mask_map(expected_missing, {k: len(v) for k, v in sources.items()}, "expected_missing")
    to_meters = skeleton_to_meters(model)
    result: dict[str, PointMatch] = {}
    taken: set[str] = set()
    for mesh in model.meshes:
        name = mesh.source_name
        if name not in sources or name in taken:
            continue
        taken.add(name)
        mask = hidden.get(name)
        result[mesh.path] = match_points(sources[name] @ to_meters.T, rest_points(mesh) @ to_meters.T,
                                         distance=distance, candidates=None if mask is None else ~mask)
    return result


# MARK: - 形のずれ


def _as_match_map(value: Any) -> dict[str, PointMatch]:
    if not isinstance(value, Mapping) or not all(isinstance(m, PointMatch) for m in value.values()):
        raise ValueError("matches は 形の場所 → PointMatch の辞書にしてください（match_meshes の答え）")
    return dict(value)


def _moved_joints(posed: np.ndarray, rest: np.ndarray) -> int:
    """休みの姿勢から動いた骨の数（どちらかが分からない骨は数えない）"""
    known = ~np.isnan(posed).all(axis=(1, 2)) & ~np.isnan(rest).all(axis=(1, 2))
    if not known.any():
        return 0
    difference = np.abs(posed[known] - rest[known]).max(axis=(1, 2))
    return int((difference > POSE_MOVED_TOLERANCE).sum())


def compare_pose(exported: ExportedModel, source_points: Mapping[str, Any], posed: Any,
                 matches: Mapping[str, PointMatch], *, pose: str, rest: Any = None,
                 extra_details: Mapping[str, Any] | None = None) -> Measurement:
    """1つのポーズで、書き出した形を骨で動かした点と、Blender で動かした元の頂点を比べる（"roundtrip_deviation"、"m"）。

    source_points：元のメッシュの名前 → そのポーズの頂点 (S, 3)（骨格の空間。頂点の並びは match の元の頂点と同じ）
    posed：(J, 4, 4) 骨格の骨ごとのポーズの行列（骨格の空間）。分からない骨は、行列全体を NaN にする
        （posed_joint_matrices の答え）。
    matches：match_meshes の答え。rest：休みの姿勢の骨の行列 (J, 4, 4)。渡すと、どの骨も動いていないとき灰にする
    （基準を超えた点があれば赤のまま）。
    - 値は、測れた点の中でいちばん大きいずれ（アプリで見える長さ。skeleton_to_meters）。
      thresholds.ROUNDTRIP_MAX_DEVIATION 以上で赤。
    - 測れない点（対応する元の頂点が無い・行列が分からない骨に重みがある・元のメッシュが無い）があれば、赤でないかぎり灰。
    - details["meshes"][形の場所]："deviations"（書き出した点ごとのずれ。測れない点は NaN）、"source_index"、
      "worst_exported"・"worst_source"（ずれの大きい順）、"max" など。
    """
    model = _as_model(exported)
    if not isinstance(pose, str) or not pose:
        raise ValueError("pose はポーズの名前（文字列）にしてください")
    sources = _as_point_map(source_points, "source_points")
    joints = _as_matrices(posed, "posed", model.joint_count, allow_nan_rows=True)
    rest_joints = None if rest is None else _as_matrices(rest, "rest", model.joint_count, allow_nan_rows=True)
    match_map = _as_match_map(matches)
    inverse_bind = _inverse(model.bind, "bind")
    to_meters = skeleton_to_meters(model)
    unknown_joints = tuple(model.joint_names[j] for j in np.flatnonzero(np.isnan(joints).all(axis=(1, 2))))

    mesh_details: dict[str, dict[str, Any]] = {}
    measured_values: list[np.ndarray] = []
    worst_rows: list[tuple[float, str, int, int]] = []
    unmeasured_total = 0
    point_total = 0
    for mesh in model.meshes:
        count = mesh.vertex_count
        point_total += count
        deviations = np.full(count, np.nan)
        source_index = np.full(count, -1, dtype=np.int64)
        match = match_map.get(mesh.path)
        source = sources.get(mesh.source_name)
        if match is not None and len(match.index) != count:
            raise ValueError(f"{mesh.path}：matches の点の数（{len(match.index)}）が形の頂点の数（{count}）と違います")
        if match is not None and source is not None:
            if len(source) != match.source_count:
                raise ValueError(f"{mesh.source_name}：元の頂点の数（{len(source)}）が、対応を作ったとき"
                                 f"（{match.source_count}）と違います")
            points, unknown = _skin_known(mesh, model.bind, joints, inverse_bind)
            measured = match.matched & ~unknown
            source_index = match.index.copy()
            deviations[measured] = _lengths(to_meters, points[measured] - source[match.index[measured]])
        measured = ~np.isnan(deviations)
        safe = np.where(measured, deviations, 0.0)
        worst_exported = worst_indices(safe, measured)
        unmeasured_total += int((~measured).sum())
        if measured.any():
            measured_values.append(deviations[measured])
            for index in worst_exported:
                worst_rows.append((float(deviations[index]), mesh.path, int(index), int(source_index[index])))
        mesh_details[mesh.path] = {
            "source_name": mesh.source_name,
            "point_count": count,
            "measured_count": int(measured.sum()),
            "unmeasured_count": int((~measured).sum()),
            "max": float(deviations[measured].max()) if measured.any() else None,
            "over_limit_count": int((safe >= thresholds.ROUNDTRIP_MAX_DEVIATION).sum()),
            "deviations": deviations,
            "source_index": source_index,
            "worst_exported": worst_exported,
            "worst_source": source_index[worst_exported],
        }

    worst_rows.sort(key=lambda row: (-row[0], row[1], row[2]))
    all_values = np.concatenate(measured_values) if measured_values else np.zeros(0)
    details: dict[str, Any] = {
        "pose": pose,
        "limit": thresholds.ROUNDTRIP_MAX_DEVIATION,
        "point_count": point_total,
        "measured_count": int(len(all_values)),
        "unmeasured_count": unmeasured_total,
        "over_limit_count": int((all_values >= thresholds.ROUNDTRIP_MAX_DEVIATION).sum()),
        "unknown_joints": unknown_joints,
        "worst": [{"mesh": path, "exported_index": e, "source_index": s, "deviation": d}
                  for d, path, e, s in worst_rows[:WORST_LIST_LENGTH]],
        "meshes": mesh_details,
    }
    if rest_joints is not None:
        details["moved_joint_count"] = _moved_joints(joints, rest_joints)
    if extra_details:
        details.update(extra_details)
    if not len(all_values):
        return Measurement.unmeasured(DEVIATION_NAME, "m", "比べられる点がありません（対応する元の頂点・骨の行列がありません）",
                                      details)
    value = float(all_values.max())
    signal = classify(value, red=thresholds.ROUNDTRIP_MAX_DEVIATION)
    not_moved = "ポーズで動いた骨がありません（ポーズを確かめられませんでした）"
    if rest_joints is not None and details["moved_joint_count"] == 0 and signal != Signal.RED:
        return Measurement.unmeasured(DEVIATION_NAME, "m", not_moved, details)
    reasons = []
    if signal == Signal.RED:
        reasons.append("書き出した形が、元の形から基準以上ずれている点があります")
        if rest_joints is not None and details["moved_joint_count"] == 0:
            reasons.append(not_moved)  # 赤は灰で隠さない（赤 > 灰）
    if unmeasured_total:
        signal = worst([signal, Signal.GRAY])
        reasons.append("比べられない点があります（対応する元の頂点が無い、または骨の行列が分かりません）")
    return Measurement(DEVIATION_NAME, value, "m", signal, "。".join(reasons), details)


def compare_rest(exported: ExportedModel, source_rest: Mapping[str, Any], rest: Any,
                 matches: Mapping[str, PointMatch]) -> Measurement:
    """休みの姿勢で比べる（compare_pose の pose="rest"）。

    rest：元のアーマチュアの休みの行列（bone.matrix_local）を、骨格の骨の並びにした物 (J, 4, 4)
    （posed_joint_matrices の答え）。bindTransforms が元と違えば、ここでずれになる。
    """
    return compare_pose(exported, source_rest, rest, matches, pose=REST_POSE)


def check_coverage(exported: ExportedModel, source_counts: Mapping[str, int], matches: Mapping[str, PointMatch], *,
                   expected_missing: Any = None, unmeasured: Mapping[str, str] | None = None) -> Measurement:
    """形と頂点が、抜けも余りも無く書き出されたか（"roundtrip_coverage"、"count"）。

    source_counts：書き出されるはずの元のメッシュの名前 → 頂点の数
    unmeasured：元のメッシュの名前 → Blender で形を計算できなかった理由（そのメッシュは調べられないので灰）
    - 赤：書き出されていない元のメッシュ、元に無い形、同じ元のメッシュの形が2つ、書き出しで消えた頂点（MASK で
      消えるはずの頂点は除く）
    - 黄：MASK で消えるはずなのに書き出された頂点、2つの書き出した点が同じ元の頂点に対応した物
    - 灰：対応する元の頂点が見つからない書き出した点、計算できなかったメッシュ。
      元のメッシュも書き出した形も1つも無いとき（調べる物が無い。値は None）
    値は、赤と黄の物の数。
    """
    model = _as_model(exported)
    if not isinstance(source_counts, Mapping):
        raise ValueError("source_counts は メッシュの名前 → 頂点の数 の辞書にしてください")
    counts = {str(k): int(v) for k, v in source_counts.items()}
    hidden = _as_mask_map(expected_missing, counts, "expected_missing")
    skipped = dict(unmeasured or {})
    match_map = _as_match_map(matches)

    seen: dict[str, str] = {}
    extra: list[str] = []
    duplicate_meshes: list[str] = []
    lost: dict[str, np.ndarray] = {}
    hidden_exported: dict[str, np.ndarray] = {}
    shared: dict[str, np.ndarray] = {}
    unmatched_exported: dict[str, int] = {}
    methods: dict[str, str] = {}
    for mesh in model.meshes:
        name = mesh.source_name
        if name in skipped:
            continue
        if name not in counts:
            extra.append(mesh.path)
            continue
        if name in seen:
            duplicate_meshes.append(mesh.path)
            continue
        seen[name] = mesh.path
        match = match_map.get(mesh.path)
        if match is None:
            raise ValueError(f"{mesh.path} の対応（matches）がありません。match_meshes の答えを渡してください")
        if match.source_count != counts[name]:
            raise ValueError(f"{name}：元の頂点の数（{counts[name]}）が、対応を作ったとき（{match.source_count}）と違います")
        methods[mesh.path] = match.method
        mask = hidden.get(name, np.zeros(counts[name], dtype=bool))
        lost_here = match.unmatched_source[~mask[match.unmatched_source]]
        if len(lost_here):
            lost[name] = lost_here
        exported_hidden = np.unique(match.index[match.matched & mask[np.maximum(match.index, 0)]])
        if len(exported_hidden):
            hidden_exported[name] = exported_hidden.astype(np.int64)
        times = np.bincount(match.index[match.matched], minlength=counts[name])
        if (times > 1).any():
            shared[name] = np.flatnonzero(times > 1).astype(np.int64)
        if (~match.matched).any():
            unmatched_exported[mesh.path] = int((~match.matched).sum())
    missing = [name for name in counts if name not in seen and name not in skipped]
    if not counts and not model.meshes:
        return Measurement.unmeasured(COVERAGE_NAME, "count", "元のメッシュも書き出した形も無いので、調べる物がありません",
                                      {"unmeasured_meshes": skipped})

    red_count = len(missing) + len(extra) + len(duplicate_meshes) + sum(len(v) for v in lost.values())
    yellow_count = sum(len(v) for v in hidden_exported.values()) + sum(len(v) for v in shared.values())
    details: dict[str, Any] = {
        "missing_meshes": tuple(missing),
        "extra_meshes": tuple(extra),
        "duplicate_meshes": tuple(duplicate_meshes),
        "lost_vertices": lost,
        "hidden_exported": hidden_exported,
        "shared_source_vertices": shared,
        "unmatched_exported": unmatched_exported,
        "expected_hidden_counts": {name: int(mask.sum()) for name, mask in hidden.items()},
        "unmeasured_meshes": skipped,
        "match_methods": methods,
    }
    reasons = []
    signal = Signal.GREEN
    if red_count:
        signal = Signal.RED
        reasons.append("書き出されていない形・頂点、または元に無い形があります")
    elif yellow_count:
        signal = Signal.YELLOW
        reasons.append("服で隠れるはずの頂点が書き出されたか、2つの点が同じ元の頂点に対応しました")
    if unmatched_exported or skipped:
        signal = worst([signal, Signal.GRAY])
        reasons.append("対応する元の頂点が見つからない点か、調べられなかったメッシュがあります")
    return Measurement(COVERAGE_NAME, float(red_count + yellow_count), "count", signal, "。".join(reasons), details)


# MARK: - 骨格の休みの姿勢


def check_skeleton(exported: ExportedModel, original_bones: Mapping[str, str] | None = None) -> Measurement:
    """restTransforms から組み立てた休みの姿勢が、bindTransforms と合うか（"skeleton_rest_consistency"、"m"）。

    アプリは休みの姿勢を restTransforms と骨の親子（joint_parents）から組み立て、bindTransforms の逆を掛けて形を動かす。
    合わない骨があると、ポーズを付けなくても、その骨に付いた形がずれる。
    - 値は、組み立てた休みの姿勢で書き出した形を動かしたときの、いちばん大きい点のずれ（アプリで見える長さ。
      skeleton_to_meters）。ROUNDTRIP_MAX_DEVIATION 以上で赤。
    - 合わない骨があるのに、形が基準より動かない（重みの無い骨など）ときは黄。
    - restTransforms が無い・骨の並びが USD の決まりに合わない・骨で動く形が無いときは灰。
    details："mismatched_joints"（合わない骨）、"path_gaps"（骨の場所の途中で、骨の一覧に無い物。抜けた骨の後ろの
    骨は、親を取り違える）、"original"（元の名前。original_bones を渡したとき）など。
    """
    model = _as_model(exported)
    table = dict(original_bones) if original_bones is not None else {}
    names = model.joint_names
    gaps = path_gaps(model.joint_paths)
    gap_names = {path.split("/")[-1]: missing for path, missing in gaps.items()}
    details: dict[str, Any] = {
        "joint_count": model.joint_count,
        "path_gaps": gap_names,
        "path_gaps_original": {table.get(name, name): tuple(table.get(m, m) for m in missing)
                               for name, missing in gap_names.items()},
        "limit": thresholds.ROUNDTRIP_MAX_DEVIATION,
    }
    if model.rest_local is None:
        return Measurement.unmeasured(SKELETON_NAME, "m", "骨格に休みの姿勢（restTransforms）がありません", details)
    try:
        composed = compose_rest(model.rest_local, model.joint_parents)
    except ValueError as error:
        return Measurement.unmeasured(SKELETON_NAME, "m", f"休みの姿勢を組み立てられません：{error}", details)
    error_per_joint = np.abs(composed - model.bind).max(axis=(1, 2)) if model.joint_count else np.zeros(0)
    mismatched = np.flatnonzero(error_per_joint > SKELETON_MATRIX_TOLERANCE)
    details["max_matrix_error"] = float(error_per_joint.max()) if len(error_per_joint) else 0.0
    details["mismatched_joints"] = tuple(names[j] for j in mismatched)
    details["mismatched_original"] = tuple(table.get(names[j], names[j]) for j in mismatched)
    details["joint_errors"] = error_per_joint

    inverse_bind = _inverse(model.bind, "bind")
    to_meters = skeleton_to_meters(model)
    mesh_details: dict[str, dict[str, Any]] = {}
    values: list[float] = []
    for mesh in model.meshes:
        if not mesh.skinned:
            continue
        moved, unknown = _skin_known(mesh, model.bind, composed, inverse_bind)
        deviation = _lengths(to_meters, moved - rest_points(mesh))
        usable = ~unknown
        if usable.any():
            values.append(float(deviation[usable].max()))
        mesh_details[mesh.path] = {
            "max": float(deviation[usable].max()) if usable.any() else None,
            "affected_count": int((usable & (deviation >= thresholds.ROUNDTRIP_MAX_DEVIATION)).sum()),
            "worst_exported": worst_indices(np.where(usable, deviation, 0.0), usable & (deviation > 0.0)),
        }
    details["meshes"] = mesh_details
    if not values:
        return Measurement.unmeasured(SKELETON_NAME, "m", "骨で動く形が無いので、形のずれを測れません", details)
    value = max(values)
    signal = classify(value, red=thresholds.ROUNDTRIP_MAX_DEVIATION)
    reason = ""
    if signal == Signal.RED:
        reason = "骨格の休みの姿勢（restTransforms）が bindTransforms と合わず、アプリでは何もしなくても形がずれます"
    elif len(mismatched):
        signal = Signal.YELLOW
        reason = "休みの姿勢が bindTransforms と合わない骨があります（いまは形を動かしていません）"
    return Measurement(SKELETON_NAME, value, "m", signal, reason, details)


# MARK: - 名前


def load_rig(source: Any) -> dict[str, Any]:
    """character_rig.json を読む。source はファイルの場所か、読んだ辞書。形が違えば ValueError"""
    if isinstance(source, Mapping):
        return dict(source)
    if not isinstance(source, (str, os.PathLike)):
        raise ValueError(f"rig は character_rig.json の場所か辞書にしてください（いまの型：{type(source).__name__}）")
    path = Path(source)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path.name} を読めませんでした：{error}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} のいちばん外が {{}} ではありません")
    return data


def original_names(rig: Mapping[str, Any], key: str) -> dict[str, str] | None:
    """rig の originalNames[key]（書き出した名前 → 元の名前）。無ければ None。形が違えば ValueError"""
    if not isinstance(rig, Mapping):
        raise ValueError("rig は辞書にしてください")
    table = rig.get("originalNames")
    if table is None:
        return None
    if not isinstance(table, Mapping):
        raise ValueError("character_rig.json の originalNames が {} ではありません")
    names = table.get(key)
    if names is None:
        return None
    if not isinstance(names, Mapping) or not all(isinstance(k, str) and isinstance(v, str) for k, v in names.items()):
        raise ValueError(f"character_rig.json の originalNames.{key} は、名前 → 名前 の {{}} にしてください")
    return dict(names)


def _duplicates(names: Iterable[str]) -> tuple[str, ...]:
    seen: dict[str, int] = {}
    for name in names:
        seen[name] = seen.get(name, 0) + 1
    return tuple(name for name, times in seen.items() if times > 1)


def _count_measurement(name: str, problems: int, reason: str, details: dict[str, Any]) -> Measurement:
    if problems:
        return Measurement(name, float(problems), "count", Signal.RED, reason, details)
    return Measurement(name, 0.0, "count", Signal.GREEN, "", details)


def check_names(rig: Mapping[str, Any], exported: ExportedModel, source_bone_names: Iterable[str],
                source_shape_keys: Iterable[str], *,
                source_deform_bones: Iterable[str] | None = None) -> list[Measurement]:
    """名前を、character_rig.json の originalNames を通して元のモデルと比べる。返す物（どれも "count"、問題の数）：

    - "joint_names"：骨格の骨が、どれも originalNames.bones で元の骨の名前に戻せて、その骨が元のモデルにあるか。
      同じ名前の骨・同じ元の骨に戻る骨が無いか。source_deform_bones（形を動かす骨）がどれも書き出されたか。
    - "humanoid_bones"：rig の bones（ヒューマノイドの名前 → 書き出した骨の名前）の骨が、どれも骨格にあるか。
      bones が空なら灰（調べる骨が無い。アプリの骨が足りないことは、E1 の報告（missing_humanoid）が黄で知らせる）。
    - "shape_key_names"：書き出したシェイプキーが、どれも originalNames.shapeKeys で元のシェイプキーに戻せるか。
      rig の expressions が使うシェイプキーが、どれも書き出されたか。
    問題があれば赤。表が無い・形が違う・骨格に骨が無いときは灰。
    """
    model = _as_model(exported)
    if not isinstance(rig, Mapping):
        raise ValueError("rig は character_rig.json の中身（辞書）にしてください")
    source_bones = set(_as_names(source_bone_names, "source_bone_names"))
    source_keys = set(_as_names(source_shape_keys, "source_shape_keys"))
    deform = None if source_deform_bones is None else _as_names(source_deform_bones, "source_deform_bones")
    joint_names = model.joint_names
    results: list[Measurement] = []

    # 骨の名前
    try:
        bone_table = original_names(rig, "bones")
    except ValueError as error:
        bone_table, table_error = None, str(error)
    else:
        table_error = "character_rig.json に originalNames.bones がありません（骨の名前を元に戻せません）"
    if bone_table is None:
        results.append(Measurement.unmeasured(JOINT_NAMES_NAME, "count", table_error,
                                              {"joint_count": model.joint_count}))
    elif model.joint_count == 0:
        results.append(Measurement.unmeasured(JOINT_NAMES_NAME, "count", "書き出した骨格に骨が無いので、名前を調べられません",
                                              {"joint_count": 0}))
    else:
        unmapped = tuple(name for name in joint_names if name not in bone_table)
        unknown = tuple(name for name in joint_names if name in bone_table and bone_table[name] not in source_bones)
        originals = [bone_table[name] for name in dict.fromkeys(joint_names) if name in bone_table]
        exported_originals = set(originals)
        not_exported = tuple(b for b in deform if b not in exported_originals) if deform is not None else ()
        details = {
            "joint_count": model.joint_count,
            "unmapped_joints": unmapped,
            "unknown_originals": {name: bone_table[name] for name in unknown},
            "duplicate_joint_names": _duplicates(joint_names),
            "duplicate_originals": _duplicates(originals),
            "deform_bones_not_exported": not_exported,
        }
        problems = (len(unmapped) + len(unknown) + len(details["duplicate_joint_names"])
                    + len(details["duplicate_originals"]) + len(not_exported))
        results.append(_count_measurement(JOINT_NAMES_NAME, problems,
                                          "元の骨の名前に戻せない骨、または書き出されていない骨があります", details))

    # ヒューマノイドの骨
    humanoid = rig.get("bones")
    if not isinstance(humanoid, Mapping):
        results.append(Measurement.unmeasured(HUMANOID_NAME, "count", "character_rig.json に bones がありません"))
    elif not humanoid:
        results.append(Measurement.unmeasured(HUMANOID_NAME, "count",
                                              "character_rig.json の bones が空です（アプリが使う骨が1つもありません）",
                                              {"humanoid_count": 0}))
    else:
        joints = set(joint_names)
        missing = {str(k): v for k, v in humanoid.items() if not isinstance(v, str) or v not in joints}
        table = bone_table or {}
        details = {
            "humanoid_count": len(humanoid),
            "missing": missing,
            "original": {str(k): table.get(v, None) if isinstance(v, str) else None for k, v in humanoid.items()},
        }
        results.append(_count_measurement(HUMANOID_NAME, len(missing),
                                          "アプリが使う骨（bones）が、書き出した骨格にありません", details))

    # シェイプキーの名前
    shapes = tuple(dict.fromkeys(shape.name for mesh in model.meshes for shape in mesh.blend_shapes))
    expressions = rig.get("expressions", {})
    try:
        key_table = original_names(rig, "shapeKeys")
    except ValueError as error:
        results.append(Measurement.unmeasured(SHAPE_KEYS_NAME, "count", str(error)))
        return results
    if key_table is None and shapes:
        results.append(Measurement.unmeasured(SHAPE_KEYS_NAME, "count",
                                              "character_rig.json に originalNames.shapeKeys がありません"))
        return results
    key_table = key_table or {}
    unmapped_keys = tuple(name for name in shapes if name not in key_table)
    unknown_keys = {name: key_table[name] for name in shapes
                    if name in key_table and key_table[name] not in source_keys}
    missing_in_export: dict[str, tuple[str, ...]] = {}
    invalid_presets: list[str] = []
    if not isinstance(expressions, Mapping):
        invalid_presets.append("expressions")
    else:
        available = set(shapes)
        for preset, entries in expressions.items():
            if not isinstance(entries, list) or not all(isinstance(e, Mapping) and isinstance(e.get("shapeKey"), str)
                                                        for e in entries):
                invalid_presets.append(str(preset))
                continue
            absent = tuple(e["shapeKey"] for e in entries if e["shapeKey"] not in available)
            if absent:
                missing_in_export[str(preset)] = absent
    details = {
        "blend_shape_count": len(shapes),
        "unmapped_shape_keys": unmapped_keys,
        "unknown_originals": unknown_keys,
        "expression_keys_not_exported": missing_in_export,
        "invalid_presets": tuple(invalid_presets),
    }
    problems = (len(unmapped_keys) + len(unknown_keys) + sum(len(v) for v in missing_in_export.values())
                + len(invalid_presets))
    results.append(_count_measurement(SHAPE_KEYS_NAME, problems,
                                      "元の名前に戻せないシェイプキー、または書き出されていない表情のキーがあります", details))
    return results


# MARK: - 骨の数と重み


def check_influences(exported: ExportedModel) -> list[Measurement]:
    """書き出したウェイトの点検。返す物：

    - "max_bone_influences"（"count"）：1つの頂点に効く骨（重みが 0 でない枠）のいちばん多い数。
      thresholds.MAX_BONE_INFLUENCES を超える頂点があれば赤。数は守っていても、枠の数（elementSize）が上限を
      超える形があれば黄（アプリが枠の数で断るかもしれない）。
    - "joint_weight_sums"（"ratio"）：頂点ごとの重みの合計の、1 からのいちばん大きいずれ。
      thresholds.WEIGHT_SUM_TOLERANCE を超えれば黄。どの骨にも付いていない点・負の重み・骨格に無い骨を指す重みは赤。
    骨で動く形が1つも無いときは、どちらも灰。
    """
    model = _as_model(exported)
    limit = thresholds.MAX_BONE_INFLUENCES
    skinned = [mesh for mesh in model.meshes if mesh.skinned]
    if not skinned:
        reason = "骨で動く形がありません"
        return [Measurement.unmeasured(INFLUENCES_NAME, "count", reason),
                Measurement.unmeasured(WEIGHT_SUMS_NAME, "ratio", reason)]
    count_details: dict[str, dict[str, Any]] = {}
    sum_details: dict[str, dict[str, Any]] = {}
    max_count = 0
    max_deviation = 0.0
    over_total = 0
    wide = []
    red_sums = False
    yellow_sums = False
    for mesh in skinned:
        weights = mesh.joint_weights
        used = weights != 0.0
        counts = used.sum(axis=1)
        over = np.flatnonzero(counts > limit)
        over_total += len(over)
        max_count = max(max_count, int(counts.max()) if len(counts) else 0)
        if mesh.element_size > limit:
            wide.append(mesh.path)
        count_details[mesh.path] = {"max": int(counts.max()) if len(counts) else 0, "element_size": mesh.element_size,
                                    "over_limit": over.astype(np.int64)}
        ids = _skeleton_ids(mesh)
        invalid = np.flatnonzero((used & ((ids < 0) | (ids >= model.joint_count))).any(axis=1))
        totals = weights.sum(axis=1)
        unweighted = np.flatnonzero(~used.any(axis=1))
        negative = np.flatnonzero((weights < 0.0).any(axis=1))
        deviation = np.abs(totals - 1.0)
        not_normalized = np.flatnonzero(used.any(axis=1) & (deviation > thresholds.WEIGHT_SUM_TOLERANCE))
        if len(deviation):
            max_deviation = max(max_deviation, float(deviation.max()))
        red_sums |= bool(len(invalid) or len(unweighted) or len(negative))
        yellow_sums |= bool(len(not_normalized))
        sum_details[mesh.path] = {"unweighted": unweighted.astype(np.int64), "negative": negative.astype(np.int64),
                                  "invalid_joint": invalid.astype(np.int64),
                                  "not_normalized": not_normalized.astype(np.int64),
                                  "max_deviation": float(deviation.max()) if len(deviation) else 0.0}

    details = {"limit": limit, "over_limit_count": over_total, "wide_meshes": tuple(wide), "meshes": count_details}
    if over_total:
        influences = Measurement(INFLUENCES_NAME, float(max_count), "count", Signal.RED,
                                 "1つの頂点に効く骨が、上限の数を超えています", details)
    elif wide:
        influences = Measurement(INFLUENCES_NAME, float(max_count), "count", Signal.YELLOW,
                                 "骨の枠の数（elementSize）が上限の数を超える形があります（重みが 0 の枠）", details)
    else:
        influences = Measurement(INFLUENCES_NAME, float(max_count), "count", Signal.GREEN, "", details)

    sums_details = {"tolerance": thresholds.WEIGHT_SUM_TOLERANCE, "meshes": sum_details}
    if red_sums:
        sums = Measurement(WEIGHT_SUMS_NAME, max_deviation, "ratio", Signal.RED,
                           "どの骨にも付いていない点、負の重み、または骨格に無い骨を指す重みがあります", sums_details)
    elif yellow_sums:
        sums = Measurement(WEIGHT_SUMS_NAME, max_deviation, "ratio", Signal.YELLOW, "重みの合計が 1 でない頂点があります",
                           sums_details)
    else:
        sums = Measurement(WEIGHT_SUMS_NAME, max_deviation, "ratio", Signal.GREEN, "", sums_details)
    return [influences, sums]


# MARK: - 物理の部品


def check_no_physics(exported: ExportedModel, source_physics_object_names: Iterable[str]) -> Measurement:
    """書き出しに物理の部品が入っていないか（"physics_parts"、"count"）。

    - 赤：元のモデルの剛体・ジョイント・Empty（source_physics_object_names）と同じ名前の部品、物理の API を当てた部品、
      "Physics" で始まる種類の部品。名前は、Blender のオブジェクトの名前（object_names）がある部品はそれで、
      無い部品は USD の部品の名前（場所の最後の部分）で比べる
    - 黄：EXPECTED_PRIM_TYPES に無い、ほかの種類の部品（カメラ・種類の無い部品など）
    値は、赤と黄の部品の数。
    """
    model = _as_model(exported)
    physics_names = set(_as_names(source_physics_object_names, "source_physics_object_names"))
    prim_paths = dict.fromkeys(path for path, _ in model.prims)
    prim_paths.update(dict.fromkeys(model.object_names))
    prim_paths.update(dict.fromkeys(path for path, _ in model.other_prims))
    prim_paths.update(dict.fromkeys(mesh.path for mesh in model.meshes))
    named: dict[str, str] = {}
    for path in prim_paths:
        name = model.object_names.get(path, path.rstrip("/").split("/")[-1])
        if name in physics_names:
            named[path] = name
    physics_types = tuple((p, t) for p, t in model.other_prims if t.startswith(PHYSICS_PREFIX))
    unknown_types = tuple((p, t) for p, t in model.other_prims if not t.startswith(PHYSICS_PREFIX))
    schemas = tuple((p, s) for p, s in model.physics_schemas)
    red = len(named) + len(physics_types) + len(schemas)
    details = {
        "physics_named": named,
        "physics_types": physics_types,
        "physics_schemas": schemas,
        "unknown_types": unknown_types,
        "source_physics_count": len(physics_names),
    }
    if red:
        return Measurement(PHYSICS_NAME, float(red + len(unknown_types)), "count", Signal.RED,
                           "物理の部品（剛体・ジョイントなど）が書き出されています", details)
    if unknown_types:
        return Measurement(PHYSICS_NAME, float(len(unknown_types)), "count", Signal.YELLOW,
                           "知らない種類の部品が書き出されています", details)
    return Measurement(PHYSICS_NAME, 0.0, "count", Signal.GREEN, "", details)


# MARK: - ポーズ


def posed_joint_matrices(joint_names: Sequence[str], original_bones: Mapping[str, str] | None,
                         bone_names: Sequence[str], matrices: Any) -> np.ndarray:
    """元のアーマチュアの骨の行列 (B, 4, 4) を、骨格の骨の並び (J, 4, 4) にする。

    書き出した骨の名前 → originalNames.bones → 元の骨の名前 → bone_names の番号 で引く。
    戻せない骨（表に無い・元のアーマチュアに無い）は、行列全体を NaN にする（compare_pose が測れない点にする）。
    original_bones が None なら、全部 NaN。
    """
    names = _as_names(joint_names, "joint_names")
    bones_ = _as_names(bone_names, "bone_names")
    source = _as_matrices(matrices, "matrices", len(bones_))
    table = {} if original_bones is None else dict(original_bones)
    lookup = {name: index for index, name in enumerate(bones_)}
    result = np.full((len(names), 4, 4), np.nan)
    for j, name in enumerate(names):
        original = table.get(name)
        index = None if original is None else lookup.get(original)
        if index is not None:
            result[j] = source[index]
    return result


def _rotation_part(matrix: np.ndarray) -> np.ndarray:
    """いちばん近い回転の行列（大きさを外す。SVD の極分解）"""
    u, _, vt = np.linalg.svd(matrix[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u = u.copy()
        u[:, 2] *= -1.0
        rotation = u @ vt
    return rotation


def local_rotation(rest_matrix: Any, axis: Sequence[float], angle_deg: float) -> tuple[float, float, float, float]:
    """アーマチュアの空間の軸 axis のまわりに angle_deg 度回すことを、骨のローカルの回転（w, x, y, z）にする。

    Blender のポーズ：pose = parent @ rest_rel @ basis。親が休みの姿勢なら、骨の行列は rest @ basis なので、
    アーマチュアの空間で Q だけ回すには basis = rest⁻¹ Q rest（軸 rest⁻¹ axis のまわりの同じ角度）。
    rest_matrix は骨の休みの行列（bone.matrix_local。(4, 4) か (3, 3)）。軸の長さが 0 なら ValueError。
    """
    matrix = as_array(rest_matrix, "rest_matrix")
    if matrix.dtype.kind not in "iuf" or matrix.shape not in ((4, 4), (3, 3)):
        raise ValueError(f"rest_matrix は (4, 4) か (3, 3) の行列にしてください（いまの形：{matrix.shape}）")
    a = as_array(axis, "axis")
    if a.dtype.kind not in "iuf" or a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError("axis は3つの有限の数にしてください")
    if isinstance(angle_deg, bool) or not isinstance(angle_deg, (int, float, np.integer, np.floating)) \
            or not math.isfinite(float(angle_deg)):
        raise ValueError(f"angle_deg は有限の数にしてください（いまの値：{angle_deg!r}）")
    length = float(np.linalg.norm(a))
    if length == 0.0:
        raise ValueError("axis の長さが 0 です")
    local_axis = _rotation_part(np.asarray(matrix, dtype=np.float64)).T @ (np.asarray(a, dtype=np.float64) / length)
    half = math.radians(float(angle_deg)) / 2.0
    sine = math.sin(half)
    return (math.cos(half), float(local_axis[0] * sine), float(local_axis[1] * sine), float(local_axis[2] * sine))


def resolve_pose(pose: Mapping[str, tuple[Sequence[float], float]], bone_names: Sequence[str],
                 rest_matrices: Any) -> tuple[dict[str, tuple[float, float, float, float]], tuple[str, ...]]:
    """確かめのポーズ（共通の名前 → (軸, 角度)）を、モデルの骨の名前 → ローカルの回転（w, x, y, z）にする。

    共通の名前は bones.map_bones で、モデルの骨の名前に対応させる。rest_matrices は bone_names の順の休みの行列
    (B, 4, 4)（アーマチュアの空間）。返す物：(回転, モデルに無かった共通の名前)
    """
    if not isinstance(pose, Mapping):
        raise ValueError("pose は 共通の名前 → (軸, 角度) の辞書にしてください")
    names = _as_names(bone_names, "bone_names")
    matrices = _as_matrices(rest_matrices, "rest_matrices", len(names))
    by_canonical = bones.map_bones(names).by_canonical
    lookup = {name: index for index, name in enumerate(names)}
    rotations: dict[str, tuple[float, float, float, float]] = {}
    missing: list[str] = []
    for canonical, value in pose.items():
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError(f"{canonical} の値は (軸, 角度) にしてください")
        actual = by_canonical.get(canonical)
        if actual is None:
            missing.append(canonical)
            continue
        axis, angle = value
        rotations[actual] = local_rotation(matrices[lookup[actual]], axis, angle)
    return rotations, tuple(missing)

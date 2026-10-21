"""A6 版の記録（snapshot）：版を保存したときの、オブジェクト・形・シェイプキー・ウェイト・骨・ドライバー・材質・
モディファイアの控え（bpy なし・numpy だけ）。

2つの版の違い（diff.py）は、この記録どうしを比べて出す。Blender から作る所は snapshot_bpy.py。

決まり（CLAUDE.md 7章）：
- 長さはメートル。点は (N, 3) の float64、三角形は (M, 3) の int64。行列は (4, 4) の float64
  （列ベクトルに左から掛ける形。mathutils の Matrix を np.array にした形と同じ）。
- メッシュの頂点は、オブジェクトの中の座標（オブジェクトの置き方を通さない）。置き方は objects の matrix で別に比べる。
- 骨の位置は、アーマチュアのオブジェクトの中の座標（Blender の head_local・tail_local・matrix_local）。
- 記録の並びは丸めない（float64 のまま。Blender の float32 の値は、そのまま正しく入る）。比べるときに幅を使う。

保存の形（save_snapshot・load_snapshot）：
- フォルダの中に snapshot.json（名前・種類などと、並びの鍵）と arrays.npz（数の並び。np.savez_compressed）を置く。
- 並びの鍵は "a0"・"a1" … の決まった名前（オブジェクトの名前を鍵にしない。どんな文字の名前でも困らないように）。
- arrays.npz の sha256 を snapshot.json に書き、読むときに確かめる（別の版の並びと取りちがえない）。
- 読むときは np.load(allow_pickle=False)。pickle（Python の物をそのまま入れる形）は決して読まない
  （ファイルの中のコードが動いてしまうため。CLAUDE.md 2章1）。
- 上書きしない：同じ名前のファイルがあれば FileExistsError。書きかけのファイルが残らないように、一時ファイルに
  書いてから置く。snapshot.json を最後に置く（snapshot.json があれば、記録はそろっている）。
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ..measure.geometry import as_array, as_points, as_triangles
from ..measure.weights import WeightTable

# MARK: - 決まり

# 記録の形の版。項目の意味を変えたら上げる
SCHEMA_VERSION = 1
FORMAT_NAME = "blender_ai_tools.versions.snapshot"
JSON_NAME = "snapshot.json"
ARRAYS_NAME = "arrays.npz"

# ファイルを読むときの1回の大きさ（1 MiB）
_CHUNK_SIZE = 1024 * 1024
_INT64_MAX = np.iinfo(np.int64).max


# MARK: - 入力の確かめ

def _readonly(array: np.ndarray) -> np.ndarray:
    array = np.array(array, copy=True)
    array.flags.writeable = False
    return array


def _text(value: Any, what: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{what} は文字にしてください（いまの値：{value!r}）")
    if not allow_empty and not value:
        raise ValueError(f"{what} が空です")
    return value


def _texts(values: Any, what: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError(f"{what} は文字の並びにしてください（いまの値：{values!r}）")
    return tuple(_text(value, f"{what} の中身") for value in values)


def _unique(names: Sequence[str], what: str) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"{what} に同じ名前 {name!r} が2回あります")
        seen.add(name)


def _matrix(value: Any, what: str) -> np.ndarray:
    array = as_array(value, what)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{what} は数の並びにしてください（いまの型：{array.dtype}）")
    array = np.asarray(array, dtype=np.float64)
    if array.shape != (4, 4):
        raise ValueError(f"{what} は (4, 4) の並びにしてください（いまの形：{array.shape}）")
    if not np.isfinite(array).all():
        raise ValueError(f"{what} に NaN か無限大が入っています")
    return _readonly(array)


def _int_vector(value: Any, what: str, length: int | None = None) -> np.ndarray:
    """1次元の int64 の並び。整数でない（小数・真偽）と ValueError。空の並びは数の型なら受け付ける"""
    array = as_array(value, what)
    if array.dtype.kind == "b" or array.dtype.kind not in "iuf":
        raise ValueError(f"{what} は整数の並びにしてください（いまの型：{array.dtype}）")
    if array.ndim != 1:
        raise ValueError(f"{what} は1次元の並びにしてください（いまの形：{array.shape}）")
    if array.size and array.dtype.kind == "f":
        raise ValueError(f"{what} は整数の並びにしてください（いまの型：{array.dtype}）")
    if array.size and array.dtype.kind == "u" and int(array.max()) > _INT64_MAX:
        raise ValueError(f"{what} の数が大きすぎます")
    array = np.asarray(array, dtype=np.int64)
    if length is not None and len(array) != length:
        raise ValueError(f"{what} の長さは {length} にしてください（いまの長さ：{len(array)}）")
    return _readonly(array)


def _bool_vector(value: Any, what: str, length: int) -> np.ndarray:
    array = as_array(value, what)
    if array.dtype.kind != "b" or array.shape != (length,):
        raise ValueError(f"{what} は ({length},) の真偽（bool）の並びにしてください"
                         f"（いまの型・形：{array.dtype}・{array.shape}）")
    return _readonly(array)


# MARK: - 記録の中身

@dataclass(frozen=True, eq=False)
class ObjectState:
    """1つのオブジェクトの控え。

    type: Blender のオブジェクトの種類（"MESH"・"ARMATURE"・"EMPTY" など）
    parent: 親のオブジェクトの名前。親が無ければ None
    matrix: オブジェクト自身の置き方 (4, 4) float64（Blender の matrix_basis：位置・回転・大きさ。
        親の動き・ポーズでは変わらない）
    modifiers: モディファイアの (名前, 種類, 描画で使うか show_render) の並び（上から順）。名前は重ならない
    materials: 材質の枠ごとの材質の名前（枠の順。材質の入っていない枠は ""）
    parent_bone: 骨を親にしているときの骨の名前（ほかは ""）
    """

    type: str
    parent: str | None
    matrix: np.ndarray
    modifiers: tuple[tuple[str, str, bool], ...] = ()
    materials: tuple[str, ...] = ()
    parent_bone: str = ""

    def __post_init__(self) -> None:
        _text(self.type, "type", allow_empty=False)
        if self.parent is not None:
            _text(self.parent, "parent", allow_empty=False)
        object.__setattr__(self, "matrix", _matrix(self.matrix, "matrix"))
        modifiers = []
        if isinstance(self.modifiers, (str, bytes)) or not isinstance(self.modifiers, Iterable):
            raise ValueError("modifiers は (名前, 種類, 描画で使うか) の並びにしてください")
        for entry in self.modifiers:
            if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence) or len(entry) != 3:
                raise ValueError(f"modifiers の中身は (名前, 種類, 描画で使うか) の組にしてください（いまの値：{entry!r}）")
            name, kind, render = entry
            if not isinstance(render, (bool, np.bool_)):
                raise ValueError(f"モディファイア {name!r} の描画で使うか（show_render）は True か False にしてください")
            modifiers.append((_text(name, "モディファイアの名前", allow_empty=False),
                              _text(kind, "モディファイアの種類", allow_empty=False), bool(render)))
        _unique([m[0] for m in modifiers], "modifiers")
        object.__setattr__(self, "modifiers", tuple(modifiers))
        object.__setattr__(self, "materials", _texts(self.materials, "materials"))
        _text(self.parent_bone, "parent_bone")


@dataclass(frozen=True, eq=False)
class MeshState:
    """1つのメッシュのオブジェクトの形の控え。

    vertices: 頂点 (N, 3) float64。オブジェクトの中の座標。シェイプキーがあれば基本のキー（Basis）の形
    triangles: 三角形 (M, 3) int64（Blender の loop_triangles。面の向きは元の面のまま）
    shape_keys: シェイプキーの名前 → その形 (N, 3) float64（オブジェクトの中の座標。Blender の並びの順。基本のキーも入る）
    weights: 頂点グループのウェイト。骨の番号の代わりに、group_names の番号を入れる（WeightTable の決まりのまま）
    group_names: 頂点グループの名前（Blender の頂点グループの番号の順）。重ならない
    face_sizes: 面ごとの角の数 (F,) int64。face_vertices: 面の角の頂点の番号を、面の順につないだ物 (sum(face_sizes),) int64。
        どちらも None なら、三角形を面とみなす（三角形だけの記録。テストなど）。形のつながりは、この面で比べる
        （四角形を三角形に分ける向きは、頂点が動くだけで変わることがあるため、三角形では比べない）
    face_materials: 面ごとの材質の枠の番号 (F,) int64。None なら全部 0
    """

    vertices: np.ndarray
    triangles: np.ndarray
    shape_keys: Mapping[str, np.ndarray] = field(default_factory=dict)
    weights: WeightTable | None = None
    group_names: tuple[str, ...] = ()
    face_sizes: np.ndarray | None = None
    face_vertices: np.ndarray | None = None
    face_materials: np.ndarray | None = None

    def __post_init__(self) -> None:
        vertices = _readonly(as_points(self.vertices, "vertices"))
        count = len(vertices)
        object.__setattr__(self, "vertices", vertices)
        triangles = _readonly(as_triangles(self.triangles, count, "triangles"))
        object.__setattr__(self, "triangles", triangles)

        if not isinstance(self.shape_keys, Mapping):
            raise ValueError("shape_keys は {名前: (N, 3) の並び} にしてください")
        keys = {}
        for name, points in self.shape_keys.items():
            _text(name, "シェイプキーの名前", allow_empty=False)
            array = as_points(points, f"シェイプキー {name!r}")
            if len(array) != count:
                raise ValueError(f"シェイプキー {name!r} の頂点の数（{len(array)}）が、頂点の数（{count}）と違います")
            keys[name] = _readonly(array)
        object.__setattr__(self, "shape_keys", MappingProxyType(keys))

        groups = _texts(self.group_names, "group_names")
        _unique(groups, "group_names")
        object.__setattr__(self, "group_names", groups)
        weights = self.weights
        if weights is None:
            weights = WeightTable(np.zeros((count, 0), dtype=np.int64), np.zeros((count, 0), dtype=np.float64))
        if not isinstance(weights, WeightTable):
            raise ValueError(f"weights は WeightTable にしてください（いまの型：{type(weights).__name__}）")
        if weights.vertex_count != count:
            raise ValueError(f"weights の頂点の数（{weights.vertex_count}）が、頂点の数（{count}）と違います")
        used = weights.bone_ids[weights.bone_ids >= 0]
        if used.size and int(used.max()) >= len(groups):
            raise ValueError(f"weights に、頂点グループの数（{len(groups)}）以上の番号があります")
        object.__setattr__(self, "weights", weights)

        if (self.face_sizes is None) != (self.face_vertices is None):
            raise ValueError("face_sizes と face_vertices は、両方とも渡すか、両方とも None にしてください")
        if self.face_sizes is None:
            sizes = np.full(len(triangles), 3, dtype=np.int64)
            corners = triangles.reshape(-1)
        else:
            sizes = _int_vector(self.face_sizes, "face_sizes")
            corners = _int_vector(self.face_vertices, "face_vertices")
            if sizes.size and int(sizes.min()) < 3:
                raise ValueError("face_sizes に 3 より小さい数があります（面の角は3つ以上）")
            if int(sizes.sum()) != len(corners):
                raise ValueError(f"face_vertices の長さ（{len(corners)}）が、face_sizes の合計（{int(sizes.sum())}）と違います")
            if corners.size and (int(corners.min()) < 0 or int(corners.max()) >= count):
                raise ValueError(f"face_vertices に、負の番号か頂点の数（{count}）以上の番号があります")
        object.__setattr__(self, "face_sizes", _readonly(sizes))
        object.__setattr__(self, "face_vertices", _readonly(corners))
        if self.face_materials is None:
            materials = np.zeros(len(sizes), dtype=np.int64)
        else:
            materials = _int_vector(self.face_materials, "face_materials", len(sizes))
        object.__setattr__(self, "face_materials", _readonly(materials))

    @property
    def vertex_count(self) -> int:
        return int(len(self.vertices))

    @property
    def face_count(self) -> int:
        return int(len(self.face_sizes))


@dataclass(frozen=True, eq=False)
class ArmatureState:
    """1つのアーマチュアのオブジェクトの骨の控え（extract_bpy.BoneRest に似た形。並びは armature.data.bones の順）。

    names: 骨の名前 (B,)。重ならない
    parents: 親の骨の番号 (B,) int64。根の骨は -1
    heads・tails: 骨の根元・先 (B, 3) float64（アーマチュアのオブジェクトの中の座標）
    deform: 形を動かす骨か (B,) bool（Blender の use_deform）
    matrices: 骨の休みの行列 (B, 4, 4) float64（Blender の bone.matrix_local）。ねじれ（ロール）を比べるのに使う。
        None なら、ねじれは比べない
    """

    names: tuple[str, ...]
    parents: np.ndarray
    heads: np.ndarray
    tails: np.ndarray
    deform: np.ndarray
    matrices: np.ndarray | None = None

    def __post_init__(self) -> None:
        names = _texts(self.names, "names")
        for name in names:
            _text(name, "骨の名前", allow_empty=False)
        _unique(names, "names")
        count = len(names)
        object.__setattr__(self, "names", names)
        parents = _int_vector(self.parents, "parents", count)
        if parents.size and (int(parents.min()) < -1 or int(parents.max()) >= count):
            raise ValueError("parents に -1 より小さい番号か、骨の数以上の番号があります")
        if (parents == np.arange(count)).any():
            raise ValueError("parents に、自分を親にした骨があります")
        object.__setattr__(self, "parents", parents)
        for name in ("heads", "tails"):
            array = as_points(getattr(self, name), name)
            if len(array) != count:
                raise ValueError(f"{name} の数（{len(array)}）が、骨の数（{count}）と違います")
            object.__setattr__(self, name, _readonly(array))
        object.__setattr__(self, "deform", _bool_vector(self.deform, "deform", count))
        if self.matrices is not None:
            matrices = as_array(self.matrices, "matrices")
            if matrices.dtype.kind not in "iuf":
                raise ValueError(f"matrices は数の並びにしてください（いまの型：{matrices.dtype}）")
            matrices = np.asarray(matrices, dtype=np.float64)
            if matrices.shape != (count, 4, 4):
                raise ValueError(f"matrices は ({count}, 4, 4) の並びにしてください（いまの形：{matrices.shape}）")
            if not np.isfinite(matrices).all():
                raise ValueError("matrices に NaN か無限大が入っています")
            object.__setattr__(self, "matrices", _readonly(matrices))

    @property
    def bone_count(self) -> int:
        return len(self.names)


@dataclass(frozen=True)
class Driver:
    """1つのドライバーの控え。

    owner: ドライバーの付いたオブジェクトの名前（メッシュ・シェイプキー・アーマチュアのデータのドライバーは、
        そのデータを使うオブジェクト）
    data_path: オブジェクトから見た、動かす物の場所（例 'data.shape_keys.key_blocks["あ"].value'・
        'pose.bones["左足D"].rotation_quaternion'）
    expression: 式（Blender の driver.expression。式を使わない種類でも、そのまま入れる）
    index: 並びの何番目を動かすか（Blender の array_index。並びでない物は 0）
    kind: ドライバーの種類（Blender の driver.type："SCRIPTED"・"AVERAGE"・"SUM" など）
    variables: 変数ごとの説明の文字（名前・種類・相手。snapshot_bpy.py が決まった形で作る）
    """

    owner: str
    data_path: str
    expression: str
    index: int = 0
    kind: str = "SCRIPTED"
    variables: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.owner, "owner", allow_empty=False)
        _text(self.data_path, "data_path", allow_empty=False)
        _text(self.expression, "expression")
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError(f"index は 0 以上の整数にしてください（いまの値：{self.index!r}）")
        _text(self.kind, "kind", allow_empty=False)
        object.__setattr__(self, "variables", _texts(self.variables, "variables"))

    @property
    def key(self) -> tuple[str, str, int]:
        """どのドライバーかを決める組：(owner, data_path, index)"""
        return self.owner, self.data_path, self.index


@dataclass(frozen=True, eq=False)
class Snapshot:
    """版の記録。

    objects: オブジェクトの名前 → ObjectState（並びは記録したときの順）
    meshes: メッシュのオブジェクトの名前 → MeshState。種類が "MESH" のオブジェクトには、どれにも入れる
    armatures: アーマチュアのオブジェクトの名前 → ArmatureState。種類が "ARMATURE" のオブジェクトには、どれにも入れる
    drivers: ドライバーの並び。owner は objects にある名前
    schema_version: 記録の形の版（いまは 1 だけ）
    """

    objects: Mapping[str, ObjectState]
    meshes: Mapping[str, MeshState] = field(default_factory=dict)
    armatures: Mapping[str, ArmatureState] = field(default_factory=dict)
    drivers: tuple[Driver, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"記録の形の版 {self.schema_version!r} は読めません（読めるのは {SCHEMA_VERSION}）")
        objects = _typed_mapping(self.objects, ObjectState, "objects")
        meshes = _typed_mapping(self.meshes, MeshState, "meshes")
        armatures = _typed_mapping(self.armatures, ArmatureState, "armatures")
        for what, table, kind in (("meshes", meshes, "MESH"), ("armatures", armatures, "ARMATURE")):
            for name in table:
                if name not in objects or objects[name].type != kind:
                    raise ValueError(f"{what} の {name!r} は、objects の中の種類 {kind} のオブジェクトにしてください")
            for name, state in objects.items():
                if state.type == kind and name not in table:
                    raise ValueError(f"種類 {kind} のオブジェクト {name!r} の控えが {what} にありません")
        if isinstance(self.drivers, (str, bytes)) or not isinstance(self.drivers, Iterable):
            raise ValueError("drivers は Driver の並びにしてください")
        drivers = tuple(self.drivers)
        for driver in drivers:
            if not isinstance(driver, Driver):
                raise ValueError(f"drivers の中身は Driver にしてください（いまの型：{type(driver).__name__}）")
            if driver.owner not in objects:
                raise ValueError(f"ドライバーの owner {driver.owner!r} が objects にありません")
        object.__setattr__(self, "objects", MappingProxyType(objects))
        object.__setattr__(self, "meshes", MappingProxyType(meshes))
        object.__setattr__(self, "armatures", MappingProxyType(armatures))
        object.__setattr__(self, "drivers", drivers)


def _typed_mapping(value: Any, kind: type, what: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{what} は {{名前: {kind.__name__}}} にしてください")
    result = {}
    for name, item in value.items():
        _text(name, f"{what} の名前", allow_empty=False)
        if not isinstance(item, kind):
            raise ValueError(f"{what} の {name!r} は {kind.__name__} にしてください（いまの型：{type(item).__name__}）")
        result[name] = item
    return result


# MARK: - まったく同じか

def _same_array(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and a.shape == b.shape and bool(np.array_equal(a, b))


def _same_object(a: ObjectState, b: ObjectState) -> bool:
    return (a.type == b.type and a.parent == b.parent and a.parent_bone == b.parent_bone
            and _same_array(a.matrix, b.matrix) and a.modifiers == b.modifiers and a.materials == b.materials)


def _same_mesh(a: MeshState, b: MeshState) -> bool:
    if list(a.shape_keys) != list(b.shape_keys):
        return False
    return (all(_same_array(a.shape_keys[k], b.shape_keys[k]) for k in a.shape_keys)
            and _same_array(a.vertices, b.vertices) and _same_array(a.triangles, b.triangles)
            and a.group_names == b.group_names
            and _same_array(a.weights.bone_ids, b.weights.bone_ids) and _same_array(a.weights.weights, b.weights.weights)
            and _same_array(a.face_sizes, b.face_sizes) and _same_array(a.face_vertices, b.face_vertices)
            and _same_array(a.face_materials, b.face_materials))


def _same_armature(a: ArmatureState, b: ArmatureState) -> bool:
    return (a.names == b.names and _same_array(a.parents, b.parents) and _same_array(a.heads, b.heads)
            and _same_array(a.tails, b.tails) and _same_array(a.deform, b.deform)
            and _same_array(a.matrices, b.matrices))


def snapshots_equal(a: Snapshot, b: Snapshot) -> bool:
    """2つの記録が、並びの順・数の型・数まで、まったく同じか（幅を使わない。保存と読み込みの確かめに使う）"""
    if not isinstance(a, Snapshot) or not isinstance(b, Snapshot):
        raise ValueError("snapshots_equal には Snapshot を2つ渡してください")
    return (a.schema_version == b.schema_version
            and list(a.objects) == list(b.objects) and all(_same_object(a.objects[n], b.objects[n]) for n in a.objects)
            and list(a.meshes) == list(b.meshes) and all(_same_mesh(a.meshes[n], b.meshes[n]) for n in a.meshes)
            and list(a.armatures) == list(b.armatures)
            and all(_same_armature(a.armatures[n], b.armatures[n]) for n in a.armatures)
            and a.drivers == b.drivers)


# MARK: - JSON と npz にする

class _ArrayStore:
    """並びに "a0"・"a1" … の鍵を付けて集める"""

    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}

    def put(self, array: np.ndarray | None) -> str | None:
        if array is None:
            return None
        key = f"a{len(self.arrays)}"
        self.arrays[key] = np.ascontiguousarray(array)
        return key


def _to_data(snapshot: Snapshot) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    store = _ArrayStore()
    objects = [{"name": name, "type": state.type, "parent": state.parent, "parent_bone": state.parent_bone,
                "matrix": store.put(state.matrix), "modifiers": [list(m) for m in state.modifiers],
                "materials": list(state.materials)} for name, state in snapshot.objects.items()]
    meshes = [{"object": name, "vertices": store.put(mesh.vertices), "triangles": store.put(mesh.triangles),
               "face_sizes": store.put(mesh.face_sizes), "face_vertices": store.put(mesh.face_vertices),
               "face_materials": store.put(mesh.face_materials),
               "shape_keys": [[key, store.put(points)] for key, points in mesh.shape_keys.items()],
               "group_names": list(mesh.group_names), "weight_ids": store.put(mesh.weights.bone_ids),
               "weight_values": store.put(mesh.weights.weights)} for name, mesh in snapshot.meshes.items()]
    armatures = [{"object": name, "names": list(arm.names), "parents": store.put(arm.parents),
                  "heads": store.put(arm.heads), "tails": store.put(arm.tails), "deform": store.put(arm.deform),
                  "matrices": store.put(arm.matrices)} for name, arm in snapshot.armatures.items()]
    drivers = [{"owner": d.owner, "data_path": d.data_path, "index": d.index, "kind": d.kind,
                "expression": d.expression, "variables": list(d.variables)} for d in snapshot.drivers]
    data = {"format": FORMAT_NAME, "schema_version": snapshot.schema_version, "arrays": ARRAYS_NAME,
            "arrays_sha256": "", "objects": objects, "meshes": meshes, "armatures": armatures, "drivers": drivers}
    return data, store.arrays


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _open_new(path: Path) -> int:
    """まだ無いファイルを作って開く。あるときは FileExistsError（上書きしない。リンクもたどらない）"""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    return os.open(path, flags, 0o666)


def _write_temp(folder: Path, write: Any) -> Path:
    """同じフォルダに一時ファイルを作り、write(stream) で書く"""
    temp = folder / f".snapshot.{secrets.token_hex(8)}.tmp"
    descriptor = _open_new(temp)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return temp


def _place(temp: Path, target: Path) -> None:
    """一時ファイルを target に置く。target があれば上書きせずに FileExistsError（一時ファイルは消す）"""
    try:
        os.close(_open_new(target))  # 名前を取っておく（ほかの物があれば、ここで止まる）
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    try:
        os.replace(temp, target)  # 取っておいた、自分で作った空のファイルを置き換える
    except BaseException:
        target.unlink(missing_ok=True)
        temp.unlink(missing_ok=True)
        raise


def save_snapshot(snapshot: Snapshot, directory: str | os.PathLike[str]) -> Path:
    """記録を directory の中に snapshot.json と arrays.npz で保存し、snapshot.json の場所を返す。

    フォルダが無ければ作る。snapshot.json か arrays.npz がすでにあれば、上書きせずに FileExistsError。
    """
    if not isinstance(snapshot, Snapshot):
        raise ValueError(f"保存できるのは Snapshot だけです（いまの型：{type(snapshot).__name__}）")
    folder = Path(directory)
    if folder.exists() and not folder.is_dir():
        raise ValueError(f"記録を置く場所がフォルダではありません：{folder}")
    json_path = folder / JSON_NAME
    arrays_path = folder / ARRAYS_NAME
    for path in (json_path, arrays_path):
        if os.path.lexists(path):
            raise FileExistsError(f"記録がすでにあります（上書きしません）：{path}")
    data, arrays = _to_data(snapshot)
    folder.mkdir(parents=True, exist_ok=True)
    arrays_temp = _write_temp(folder, lambda stream: np.savez_compressed(stream, **arrays))
    data["arrays_sha256"] = _sha256(arrays_temp)
    text = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=1) + "\n"
    try:
        json_temp = _write_temp(folder, lambda stream: stream.write(text.encode("utf-8")))
    except BaseException:
        arrays_temp.unlink(missing_ok=True)
        raise
    try:
        _place(arrays_temp, arrays_path)
    except BaseException:
        json_temp.unlink(missing_ok=True)
        raise
    _place(json_temp, json_path)
    return json_path


# MARK: - JSON と npz から読む

def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"記録の JSON に同じキー {key!r} が2回あります")
        result[key] = value
    return result


def _reject_constant(name: str) -> Any:
    raise ValueError(f"記録の JSON に {name} は書けません")


def _require_dict(value: Any, keys: set[str], what: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        found = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"記録の {what} は {sorted(keys)} の項目にしてください（いまは {found}）")
    return value


def _require_list(value: Any, what: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"記録の {what} は [] の形にしてください（いまの型：{type(value).__name__}）")
    return value


class _ArrayReader:
    """npz の中の並びを、鍵で1回ずつ取り出す。数の型を確かめる。使わなかった鍵が残ったら、まちがい"""

    _KINDS = {"f": np.dtype(np.float64), "i": np.dtype(np.int64), "b": np.dtype(np.bool_)}

    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.arrays = arrays
        self.used: set[str] = set()

    def get(self, key: Any, kind: str, what: str, *, optional: bool = False) -> np.ndarray | None:
        if key is None and optional:
            return None
        if not isinstance(key, str) or key not in self.arrays:
            raise ValueError(f"記録の {what} の並びの鍵 {key!r} が {ARRAYS_NAME} にありません")
        if key in self.used:
            raise ValueError(f"記録の並びの鍵 {key!r} が2回使われています")
        self.used.add(key)
        array = self.arrays[key]
        if array.dtype != self._KINDS[kind]:
            raise ValueError(f"記録の {what} の並びの型が違います（いる：{self._KINDS[kind]}、いまの型：{array.dtype}）")
        return array

    def check_all_used(self) -> None:
        left = sorted(set(self.arrays) - self.used)
        if left:
            raise ValueError(f"{ARRAYS_NAME} に、記録で使わない並びがあります：{left}")


def _read_arrays(path: Path) -> dict[str, np.ndarray]:
    """npz を pickle なしで読み、全部の並びを取り出す。pickle の入った並び（object の並び）は ValueError。

    読めない物は、どんな壊れ方でも ValueError にする（zip として壊れている物は zipfile.BadZipFile、
    .npy でない物（NpzFile は bytes を返す）など。読む側は ValueError を灰にする）。
    """
    try:
        loaded = np.load(path, allow_pickle=False)
    except ValueError as error:
        raise ValueError(f"{path.name} を読めません（pickle は読みません）：{error}") from error
    except Exception as error:  # zip として壊れている（zipfile.BadZipFile）・読めない（OSError）など
        raise ValueError(f"{path.name} を読めません：{error}") from error
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError(f"{path.name} が npz（並びを集めた zip）ではありません")
    with loaded:
        arrays = {}
        for key in loaded.files:
            try:
                array = loaded[key]
            except ValueError as error:
                raise ValueError(f"{path.name} の並び {key!r} を読めません（pickle の入った並びは読みません）：{error}") \
                    from error
            except Exception as error:  # zip の中身が壊れている など
                raise ValueError(f"{path.name} の並び {key!r} を読めません：{error}") from error
            if not isinstance(array, np.ndarray):
                raise ValueError(f"{path.name} の {key!r} が数の並び（.npy）ではありません（{type(array).__name__}）")
            arrays[key] = array
    return arrays


def _object_from_data(item: Any, reader: _ArrayReader) -> tuple[str, ObjectState]:
    item = _require_dict(item, {"name", "type", "parent", "parent_bone", "matrix", "modifiers", "materials"},
                         "objects の中身")
    modifiers = []
    for entry in _require_list(item["modifiers"], "modifiers"):
        if not isinstance(entry, list) or len(entry) != 3:
            raise ValueError("記録の modifiers の中身は [名前, 種類, 描画で使うか] にしてください")
        modifiers.append(tuple(entry))
    state = ObjectState(type=item["type"], parent=item["parent"], matrix=reader.get(item["matrix"], "f", "matrix"),
                        modifiers=tuple(modifiers), materials=tuple(_require_list(item["materials"], "materials")),
                        parent_bone=item["parent_bone"])
    return _text(item["name"], "objects の name", allow_empty=False), state


def _mesh_from_data(item: Any, reader: _ArrayReader) -> tuple[str, MeshState]:
    item = _require_dict(item, {"object", "vertices", "triangles", "face_sizes", "face_vertices", "face_materials",
                                "shape_keys", "group_names", "weight_ids", "weight_values"}, "meshes の中身")
    keys: dict[str, np.ndarray] = {}
    for entry in _require_list(item["shape_keys"], "shape_keys"):
        if not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], str):
            raise ValueError("記録の shape_keys の中身は [名前, 並びの鍵] にしてください")
        if entry[0] in keys:
            raise ValueError(f"記録の shape_keys に同じ名前 {entry[0]!r} が2回あります")
        keys[entry[0]] = reader.get(entry[1], "f", f"シェイプキー {entry[0]!r}")
    weights = WeightTable(reader.get(item["weight_ids"], "i", "weight_ids"),
                          reader.get(item["weight_values"], "f", "weight_values"))
    state = MeshState(vertices=reader.get(item["vertices"], "f", "vertices"),
                      triangles=reader.get(item["triangles"], "i", "triangles"), shape_keys=keys, weights=weights,
                      group_names=tuple(_require_list(item["group_names"], "group_names")),
                      face_sizes=reader.get(item["face_sizes"], "i", "face_sizes"),
                      face_vertices=reader.get(item["face_vertices"], "i", "face_vertices"),
                      face_materials=reader.get(item["face_materials"], "i", "face_materials"))
    return _text(item["object"], "meshes の object", allow_empty=False), state


def _armature_from_data(item: Any, reader: _ArrayReader) -> tuple[str, ArmatureState]:
    item = _require_dict(item, {"object", "names", "parents", "heads", "tails", "deform", "matrices"},
                         "armatures の中身")
    state = ArmatureState(names=tuple(_require_list(item["names"], "names")),
                          parents=reader.get(item["parents"], "i", "parents"),
                          heads=reader.get(item["heads"], "f", "heads"), tails=reader.get(item["tails"], "f", "tails"),
                          deform=reader.get(item["deform"], "b", "deform"),
                          matrices=reader.get(item["matrices"], "f", "matrices", optional=True))
    return _text(item["object"], "armatures の object", allow_empty=False), state


def _driver_from_data(item: Any) -> Driver:
    item = _require_dict(item, {"owner", "data_path", "index", "kind", "expression", "variables"}, "drivers の中身")
    return Driver(owner=item["owner"], data_path=item["data_path"], expression=item["expression"],
                  index=item["index"], kind=item["kind"],
                  variables=tuple(_require_list(item["variables"], "variables")))


def _unique_entries(pairs: Iterable[tuple[str, Any]], what: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"記録の {what} に同じ名前 {name!r} が2回あります")
        result[name] = value
    return result


def load_snapshot(path: str | os.PathLike[str]) -> Snapshot:
    """save_snapshot で保存した記録を読む。path は記録のフォルダか、その中の snapshot.json。

    ファイルが無い・形が違う・arrays.npz の sha256 が合わない・pickle の入った並びがあるときは ValueError。
    """
    given = Path(path)
    json_path = given / JSON_NAME if given.is_dir() else given
    if not json_path.is_file():
        raise ValueError(f"記録のファイルがありません：{json_path}")
    try:
        text = json_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"記録のファイルが UTF-8 ではありません：{json_path}") from error
    try:
        data = json.loads(text, object_pairs_hook=_object_without_duplicates, parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise ValueError(f"記録のファイルが JSON として読めません：{error}") from error
    data = _require_dict(data, {"format", "schema_version", "arrays", "arrays_sha256", "objects", "meshes",
                                "armatures", "drivers"}, "いちばん外")
    if data["format"] != FORMAT_NAME:
        raise ValueError(f"版の記録のファイルではありません（format：{data['format']!r}）")
    version = data["schema_version"]
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        raise ValueError(f"記録の形の版 {version!r} は読めません（読めるのは {SCHEMA_VERSION}）")
    if data["arrays"] != ARRAYS_NAME:
        raise ValueError(f"記録の並びのファイルの名前は {ARRAYS_NAME} にしてください（いまの値：{data['arrays']!r}）")
    arrays_path = json_path.parent / ARRAYS_NAME
    if not arrays_path.is_file():
        raise ValueError(f"記録の並びのファイルがありません：{arrays_path}")
    if _sha256(arrays_path) != data["arrays_sha256"]:
        raise ValueError(f"{ARRAYS_NAME} の中身が、記録を保存したときと違います（sha256 が合いません）")
    reader = _ArrayReader(_read_arrays(arrays_path))
    objects = _unique_entries((_object_from_data(item, reader) for item in _require_list(data["objects"], "objects")),
                              "objects")
    meshes = _unique_entries((_mesh_from_data(item, reader) for item in _require_list(data["meshes"], "meshes")),
                             "meshes")
    armatures = _unique_entries(
        (_armature_from_data(item, reader) for item in _require_list(data["armatures"], "armatures")), "armatures")
    drivers = tuple(_driver_from_data(item) for item in _require_list(data["drivers"], "drivers"))
    reader.check_all_used()
    return Snapshot(objects=objects, meshes=meshes, armatures=armatures, drivers=drivers,
                    schema_version=version)

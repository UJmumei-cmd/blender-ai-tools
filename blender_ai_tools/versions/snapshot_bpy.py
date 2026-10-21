"""A6 いまの Blender の中の物から、版の記録（snapshot.Snapshot）を作る薄い所。

読むだけで、何も書き換えない（フレーム・モード・ポーズ・選択・アクティブなオブジェクトも変えない）。
編集モードの物は、編集中の形を先にデータへ写してから読む（extract_bpy.sync_edit_mode。モードはそのまま）。

読む物（シーンの中の、全部のオブジェクト。名前の順）：
- オブジェクトの名前は name_full：ふつうの（このファイルの）オブジェクトは name と同じ。ほかのファイルからリンクした
  オブジェクトは「名前 [ファイル名]」になる。同じ名前のオブジェクトが、このファイルの物とリンクした物で重なっても、
  記録の中で片方がもう片方を消さないようにする（消すと、変わった所が無いと見えてしまう）。親・ドライバーの持ち主も同じ
- オブジェクト：種類・親（骨を親にしていれば、その骨）・matrix_basis・モディファイア（名前・種類・show_render）・材質の枠
- メッシュ：オブジェクトの中の座標の頂点（シェイプキーがあれば基本のキーの形）・三角形（loop_triangles）・
  面（角の頂点の並び）・面の材質の番号・シェイプキー・頂点グループの名前とウェイト（重みが 0 の物は入れない）
- アーマチュア：骨の名前・親・根元と先（head_local・tail_local）・形を動かすか（use_deform）・matrix_local
- ドライバー：オブジェクト・メッシュ・シェイプキー（Key）・アーマチュアのデータに付いた物。場所はオブジェクトから見た
  場所にする（メッシュなら 'data.'、シェイプキーなら 'data.shape_keys.' を前に付ける）

形は、モディファイア・ポーズを通さない元のデータ（版の比べは、ファイルに入っている物どうしを比べる）。
"""

from __future__ import annotations

from typing import Any

import bpy
import numpy as np

from ..measure.weights import WeightTable
from ..model.extract_bpy import sync_edit_mode
from .snapshot import ArmatureState, Driver, MeshState, ObjectState, Snapshot


# MARK: - 小さな読み取り

def _matrix(value: Any) -> np.ndarray:
    return np.array(value, dtype=np.float64).reshape(4, 4)


def _points(collection: Any) -> np.ndarray:
    """co を持つ物の並びから (N, 3) float64 を読む（Blender の float32 の値のまま）"""
    count = len(collection)
    flat = np.empty(count * 3, dtype=np.float32)
    collection.foreach_get("co", flat)
    return flat.reshape(count, 3).astype(np.float64)


def _ints(collection: Any, attribute: str, width: int = 1) -> np.ndarray:
    flat = np.empty(len(collection) * width, dtype=np.int32)
    collection.foreach_get(attribute, flat)
    return flat.astype(np.int64)


# MARK: - オブジェクト

def _object_state(obj: Any) -> ObjectState:
    parent = obj.parent
    parent_bone = obj.parent_bone if parent is not None and obj.parent_type == "BONE" else ""
    modifiers = tuple((m.name, m.type, bool(m.show_render)) for m in obj.modifiers)
    materials = tuple("" if slot.material is None else slot.material.name for slot in obj.material_slots)
    return ObjectState(type=obj.type, parent=None if parent is None else parent.name_full,
                       matrix=_matrix(obj.matrix_basis),
                       modifiers=modifiers, materials=materials, parent_bone=parent_bone)


# MARK: - メッシュ

def _faces(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    """面ごとの角の数 (F,) と、角の頂点の番号を面の順につないだ物"""
    polygons = mesh.polygons
    starts = _ints(polygons, "loop_start")
    sizes = _ints(polygons, "loop_total")
    loop_vertices = _ints(mesh.loops, "vertex_index")
    if len(sizes) == 0:
        return sizes, np.zeros(0, dtype=np.int64)
    offsets = np.arange(int(sizes.sum()), dtype=np.int64) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    return sizes, loop_vertices[np.repeat(starts, sizes) + offsets]


def _vertex_groups(obj: Any, count: int) -> tuple[WeightTable, tuple[str, ...]]:
    """頂点グループのウェイト（番号は頂点グループの番号。頂点の中は番号の小さい順）と、グループの名前"""
    names = tuple(group.name for group in sorted(obj.vertex_groups, key=lambda g: g.index))
    group_count = len(names)
    rows: list[list[tuple[int, float]]] = []
    for vertex in obj.data.vertices:
        row = []
        for element in vertex.groups:
            weight = float(element.weight)
            if 0 <= element.group < group_count and weight != 0.0:
                row.append((int(element.group), weight))
        row.sort()
        rows.append(row)
    width = max((len(row) for row in rows), default=0)
    ids = np.full((count, width), -1, dtype=np.int64)
    weights = np.zeros((count, width), dtype=np.float64)
    for index, row in enumerate(rows):
        for slot, (group, weight) in enumerate(row):
            ids[index, slot] = group
            weights[index, slot] = weight
    return WeightTable(ids, weights), names


def _mesh_state(obj: Any) -> MeshState:
    mesh = obj.data
    count = len(mesh.vertices)
    keys = mesh.shape_keys
    shape_keys = {}
    vertices = None
    if keys is not None:
        for block in keys.key_blocks:
            if len(block.data) == count:
                shape_keys[block.name] = _points(block.data)
        reference = keys.reference_key
        if reference is not None and reference.name in shape_keys:
            vertices = shape_keys[reference.name]
    if vertices is None:
        vertices = _points(mesh.vertices)
    triangles = _ints(mesh.loop_triangles, "vertices", 3).reshape(-1, 3)
    sizes, corners = _faces(mesh)
    weights, groups = _vertex_groups(obj, count)
    return MeshState(vertices=vertices, triangles=triangles, shape_keys=shape_keys, weights=weights,
                     group_names=groups, face_sizes=sizes, face_vertices=corners,
                     face_materials=_ints(mesh.polygons, "material_index"))


# MARK: - アーマチュア

def _armature_state(obj: Any) -> ArmatureState:
    bones = list(obj.data.bones)
    names = tuple(bone.name for bone in bones)
    lookup = {name: i for i, name in enumerate(names)}
    parents = np.array([-1 if bone.parent is None else lookup[bone.parent.name] for bone in bones], dtype=np.int64)
    heads = np.array([tuple(bone.head_local) for bone in bones], dtype=np.float64).reshape(-1, 3)
    tails = np.array([tuple(bone.tail_local) for bone in bones], dtype=np.float64).reshape(-1, 3)
    deform = np.array([bool(bone.use_deform) for bone in bones], dtype=bool)
    matrices = np.array([_matrix(bone.matrix_local) for bone in bones], dtype=np.float64).reshape(-1, 4, 4)
    return ArmatureState(names=names, parents=parents, heads=heads, tails=tails, deform=deform, matrices=matrices)


# MARK: - ドライバー

def _variable_text(variable: Any) -> str:
    """変数の説明の文字（名前・種類・相手）。決まった形にして、変わったかどうかを文字で比べられるようにする"""
    parts = [variable.name, variable.type]
    for target in variable.targets:  # 変数の種類で使う相手だけが入っている（1つか2つ）
        target_id = target.id
        parts.append("|".join((
            str(target.id_type),
            "" if target_id is None else target_id.name_full,
            str(target.data_path),
            str(target.bone_target),
            str(target.transform_type),
            str(target.transform_space),
            str(target.rotation_mode),
            str(getattr(target, "context_property", "")),
        )))
    return " ; ".join(parts)


def _drivers_of(owner: str, id_data: Any, prefix: str) -> list[Driver]:
    animation = getattr(id_data, "animation_data", None)
    if animation is None:
        return []
    found = []
    for curve in animation.drivers:
        driver = curve.driver
        found.append(Driver(owner=owner, data_path=prefix + curve.data_path, expression=str(driver.expression),
                            index=int(curve.array_index), kind=str(driver.type),
                            variables=tuple(_variable_text(v) for v in driver.variables)))
    return found


def _object_drivers(obj: Any) -> list[Driver]:
    owner = obj.name_full
    drivers = _drivers_of(owner, obj, "")
    data = obj.data
    if data is not None and obj.type in ("MESH", "ARMATURE"):
        drivers += _drivers_of(owner, data, "data.")
        if obj.type == "MESH" and data.shape_keys is not None:
            drivers += _drivers_of(owner, data.shape_keys, "data.shape_keys.")
    return drivers


# MARK: - 記録を作る

def take_snapshot(scene: Any = None) -> Snapshot:
    """いまのシーン（scene を渡せば、そのシーン）の中の物から、版の記録を作る（何も書き換えない）"""
    if scene is None:
        scene = bpy.context.scene
    if not isinstance(scene, bpy.types.Scene):
        raise ValueError(f"scene はシーンにしてください（いまの物：{type(scene).__name__}）")
    objects_in_scene = sorted(scene.objects, key=lambda obj: obj.name_full)
    sync_edit_mode(*objects_in_scene)
    objects, meshes, armatures, drivers = {}, {}, {}, []
    for obj in objects_in_scene:
        name = obj.name_full  # リンクしたオブジェクトは「名前 [ファイル名]」（このファイルの物と重ならない）
        objects[name] = _object_state(obj)
        if obj.type == "MESH":
            meshes[name] = _mesh_state(obj)
        elif obj.type == "ARMATURE":
            armatures[name] = _armature_state(obj)
        drivers += _object_drivers(obj)
    return Snapshot(objects=objects, meshes=meshes, armatures=armatures, drivers=tuple(drivers))

"""A1・A4：Blender の物（アーマチュア・メッシュ・ウェイト・ポーズ）を、numpy の並びにする薄い所。

測る計算（measure/）は numpy の並びを受け取る。ここは bpy から並びを取り出すだけにする（CLAUDE.md 7章）。

決まり：
- 長さはメートル。点は (N, 3) の float64、三角形は (M, 3) の int64（四角形などは Blender の loop_triangles で三角形に分ける。
  面の回る向きは元の面のまま）。行列は (…, 4, 4) の float64 で、列ベクトルに左から掛ける形（平行移動は右の列）。
  mathutils の Matrix を np.array にした形と同じ。
- 向きは Blender のまま：Z が上。座標の空間は space で選ぶ：
    "world"：世界の座標（オブジェクトの matrix_world を通す）
    "armature"：アーマチュアのオブジェクトの中の座標（世界の座標に、アーマチュアの matrix_world の逆を掛けた物）。
      アーマチュアのオブジェクトを動かしても変わらないので、骨の休みの形・ウェイトと比べるときはこちらを使う。
- 骨の並び（番号）は armature.data.bones の並び（親から子へ、深さ優先）。bone_rest・bone_weights・pose_matrices・
  pose_rotations はどれもこの並び。convert_pmx.rename_bones の b000 などの番号も、同じ並びで付く。
- 回転（クォータニオン）は Blender の順 (w, x, y, z)。q と -q は同じ回転なので、w が負にならない方を返す。
- 元のデータは書き換えない。フレームを変えたときは、元のフレームとアーマチュアのポーズ（キーの無い変更も）に戻す。
  モード・アクティブなオブジェクト・選択は変えない。
  編集モードの物は、編集中の形を先にデータへ写してから読む（update_from_editmode。モードはそのまま）。
- いまのシーン（bpy.context）の中の物を扱う。
- 位置（matrix_world）・ポーズ・モディファイアを通した形は、いまのビューレイヤーの depsgraph で計算した物を読む
  （位置を変えたばかりで、まだ計算していなくても、新しい値になる）。
  depsgraph の計算に入らない物（ビューポートで無効 hide_viewport・除外したコレクション・ほかのシーン）は、
  古い値のまま残るので読まずに ValueError（黙って古い値を返さない）。
  書き出し（convert_blend.py）は hide_viewport の物も入れるので、読むときは写しで有効にしてから読む。
- アーマチュアが「レストポジション」（data.pose_position が "REST"）のときは、ポーズもアニメーションも使われない。
  ポーズを読む関数（pose_matrices・skinning_matrices・pose_rotations・joint_positions）は ValueError にする
  （休みの形をポーズとして読むと、動きの確かめがどれも「問題なし」に見えてしまうため）。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import bpy
import numpy as np
from mathutils import Quaternion

# QUATERNION_MIN_NORM：これより短いクォータニオンは、向きを表せないとみなす（set_pose で ValueError）
from ..measure.kinematics import QUATERNION_MIN_NORM
from ..measure.weights import WeightTable

SPACES = ("world", "armature")


# MARK: - 入力の確かめ


def _check_space(space: str) -> str:
    if space not in SPACES:
        raise ValueError(f"space は {' か '.join(repr(s) for s in SPACES)} にしてください（いまの値：{space!r}）")
    return space


def _check_object(obj: Any, kind: str, name: str) -> Any:
    if not isinstance(obj, bpy.types.Object) or obj.type != kind:
        found = getattr(obj, "type", type(obj).__name__)
        raise ValueError(f"{name} は {kind} のオブジェクトにしてください（いまの物：{found}）")
    return obj


def _matrix(value: Any) -> np.ndarray:
    """mathutils の Matrix を (4, 4) の float64 にする（列ベクトルに左から掛ける形）"""
    return np.array(value, dtype=np.float64).reshape(4, 4)


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(N, 3) の点に 4×4 の行列を掛ける"""
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _sync_edit_mode(*objects: Any) -> None:
    """編集モードの物は、編集中の形をデータに写す（モードは変えない）。写さないと、古い形を読んでしまう"""
    for obj in objects:
        if obj is not None and obj.mode == "EDIT":
            obj.update_from_editmode()


def _evaluated(obj: Any, depsgraph: Any = None) -> Any:
    """depsgraph で計算した物（depsgraph を渡さなければ、いまのビューレイヤーの物を計算して使う）。

    計算に入っていない物は、元の物が返ってくる。その matrix_world・ポーズ・形は古いままなので、ValueError にする。
    """
    if depsgraph is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    if not evaluated.is_evaluated:
        raise ValueError(f"{obj.name} は、いまのビューレイヤーで計算されていません（ビューポートで無効 hide_viewport・"
                         f"除外したコレクション・ほかのシーンなど）。位置・形・ポーズが古いままかもしれないので読めません。"
                         f"写しで有効にしてから読んでください")
    return evaluated


def _world_matrix(obj: Any, depsgraph: Any = None) -> np.ndarray:
    """計算した matrix_world (4, 4)。元の物の matrix_world は、計算するまで古いままのことがあるので使わない"""
    return _matrix(_evaluated(obj, depsgraph).matrix_world)


def _check_pose_position(armature: Any) -> None:
    """レストポジションのアーマチュアは、ポーズを使わない。ポーズとして読ませない"""
    if armature.data.pose_position != "POSE":
        raise ValueError(f"アーマチュア {armature.name} は「レストポジション」になっていて、ポーズとアニメーションが"
                         f"使われません。ポーズを読むときは「ポーズポジション」（data.pose_position = 'POSE'）にしてください")


# MARK: - モデルを探す


@dataclass(frozen=True)
class ModelObjects:
    """書き出しの対象になる物（convert_blend.py の find_model と同じ決まりで選ぶ）。

    armature: アーマチュアのオブジェクト
    meshes: それに付いた、描画するメッシュのオブジェクト（探した並びの順）
    """

    armature: Any
    meshes: tuple[Any, ...]


def _uses_armature(obj: Any, armature: Any) -> bool:
    """convert_blend.py の uses_armature と同じ：アーマチュアの子か、それを使う ARMATURE モディファイアがある"""
    return obj.parent == armature or any(m.type == "ARMATURE" and m.object == armature for m in obj.modifiers)


def find_model(scene: Any = None) -> ModelObjects:
    """書き出しの対象を、convert_blend.py の find_model とまったく同じ決まりで選ぶ。

    - アーマチュア：骨がいちばん多い物（同じ数なら、並びで先の物。Python の max と同じ）
    - メッシュ：hide_render が False で、そのアーマチュアの子か、それを使う ARMATURE モディファイアがある物
    探す物の並びは bpy.data.objects（convert_blend.py と同じ。ファイルの中の全部のオブジェクト）。
    scene を渡したときは、その中のオブジェクトだけを、同じ並びで探す。
    見つからないとき（convert_blend.py は終わる所）は、日本語の ValueError。
    """
    objects = list(bpy.data.objects)
    if scene is not None:
        in_scene = set(scene.objects)
        objects = [obj for obj in objects if obj in in_scene]
    armatures = [obj for obj in objects if obj.type == "ARMATURE"]
    if not armatures:
        raise ValueError("アーマチュアがありません")
    armature = max(armatures, key=lambda obj: len(obj.data.bones))
    meshes = tuple(obj for obj in objects
                   if obj.type == "MESH" and not obj.hide_render and _uses_armature(obj, armature))
    if not meshes:
        raise ValueError(f"骨格（{armature.name}）に付いた描画するメッシュがありません")
    return ModelObjects(armature=armature, meshes=meshes)


def armature_of(obj: Any) -> Any:
    """メッシュを動かすアーマチュア：最初の ARMATURE モディファイアの相手。無ければ、親がアーマチュアならその親。
    どちらも無いときは ValueError"""
    for modifier in obj.modifiers:
        if modifier.type == "ARMATURE" and modifier.object is not None:
            return modifier.object
    if obj.parent is not None and obj.parent.type == "ARMATURE":
        return obj.parent
    raise ValueError(f"{obj.name} を動かすアーマチュアが見つかりません（ARMATURE モディファイアも、アーマチュアの親もありません）")


# MARK: - 骨の休みの形


def _readonly(array: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(array)
    array.flags.writeable = False
    return array


@dataclass(frozen=True, eq=False)
class BoneRest:
    """骨の休みの形（エディットモードで決めた形）。並びは armature.data.bones の順。

    names: 骨の名前 (B,)
    parents: 親の骨の番号 (B,) int64。根の骨は -1
    heads・tails: 骨の根元・先 (B, 3) float64
    matrices: 骨の休みの行列 (B, 4, 4) float64（Blender の bone.matrix_local。space が "world" なら、
        アーマチュアの matrix_world を左から掛けた物）。列 0・1・2 が骨の X・Y・Z 軸（Y が骨の向き）、列 3 が根元
    deform: 形を動かす骨か (B,) bool（Blender の use_deform）
    space: "world" か "armature"
    """

    names: tuple[str, ...]
    parents: np.ndarray
    heads: np.ndarray
    tails: np.ndarray
    matrices: np.ndarray
    deform: np.ndarray
    space: str

    def __post_init__(self) -> None:
        for name in ("parents", "heads", "tails", "matrices", "deform"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    @property
    def bone_count(self) -> int:
        """骨の数 B"""
        return len(self.names)

    def index(self, name: str) -> int:
        """骨の番号。無い名前は ValueError"""
        try:
            return self.names.index(name)
        except ValueError:
            raise ValueError(f"骨 {name!r} がありません") from None

    def indices(self, names: Iterable[str]) -> np.ndarray:
        """骨の番号の並び (J,) int64。無い名前があれば、全部を書いて ValueError"""
        lookup = {name: i for i, name in enumerate(self.names)}
        wanted = list(names)
        unknown = [name for name in wanted if name not in lookup]
        if unknown:
            raise ValueError(f"アーマチュアに無い骨があります：{unknown}")
        return np.array([lookup[name] for name in wanted], dtype=np.int64)


def bone_names(armature: Any) -> tuple[str, ...]:
    """骨の名前（armature.data.bones の順）"""
    _check_object(armature, "ARMATURE", "armature")
    _sync_edit_mode(armature)
    return tuple(bone.name for bone in armature.data.bones)


def bone_rest(armature: Any, *, space: str = "world") -> BoneRest:
    """骨の休みの形。並びは armature.data.bones の順（bone_weights の骨の番号と同じ）。

    "world" はアーマチュアの計算した matrix_world を使う（計算に入っていないアーマチュアは ValueError）。
    "armature" は行列を使わないので、計算に入っていないアーマチュアでも読める。
    """
    _check_object(armature, "ARMATURE", "armature")
    _check_space(space)
    _sync_edit_mode(armature)
    bones = list(armature.data.bones)
    names = tuple(bone.name for bone in bones)
    lookup = {name: i for i, name in enumerate(names)}
    parents = np.array([-1 if bone.parent is None else lookup[bone.parent.name] for bone in bones], dtype=np.int64)
    heads = np.array([tuple(bone.head_local) for bone in bones], dtype=np.float64).reshape(-1, 3)
    tails = np.array([tuple(bone.tail_local) for bone in bones], dtype=np.float64).reshape(-1, 3)
    matrices = np.array([_matrix(bone.matrix_local) for bone in bones], dtype=np.float64).reshape(-1, 4, 4)
    deform = np.array([bool(bone.use_deform) for bone in bones], dtype=bool)
    if space == "world":
        world = _world_matrix(armature)
        heads = _transform_points(world, heads)
        tails = _transform_points(world, tails)
        matrices = world @ matrices
    return BoneRest(names=names, parents=parents, heads=heads, tails=tails, matrices=matrices, deform=deform,
                    space=space)


# MARK: - メッシュ


def _vertex_coordinates(mesh: Any) -> np.ndarray:
    count = len(mesh.vertices)
    flat = np.empty(count * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", flat)
    return flat.reshape(count, 3).astype(np.float64)


def _rest_coordinates(mesh: Any) -> np.ndarray:
    """休みの形の頂点 (N, 3)。シェイプキーがあれば基本のキー（reference_key。Basis）の形。

    Blender はシェイプキーのあるメッシュを、基本のキーから計算する。スクリプトで頂点だけを動かすと、
    mesh.vertices の座標と基本のキーがずれ、Blender の形（mesh_evaluated）は基本のキーの方になる。
    """
    keys = mesh.shape_keys
    reference = None if keys is None else keys.reference_key
    if reference is None or len(reference.data) != len(mesh.vertices):
        return _vertex_coordinates(mesh)
    flat = np.empty(len(reference.data) * 3, dtype=np.float32)
    reference.data.foreach_get("co", flat)
    return flat.reshape(-1, 3).astype(np.float64)


def _triangles(mesh: Any) -> np.ndarray:
    triangles = mesh.loop_triangles
    flat = np.empty(len(triangles) * 3, dtype=np.int32)
    triangles.foreach_get("vertices", flat)
    return flat.reshape(-1, 3).astype(np.int64)


def _to_space(obj_matrix: np.ndarray, armature_matrix: np.ndarray | None, space: str) -> np.ndarray:
    """メッシュの中の座標から、space の座標へ移す行列"""
    if space == "world":
        return obj_matrix
    return np.linalg.inv(armature_matrix) @ obj_matrix


def mesh_rest(obj: Any, *, space: str = "world", armature: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """モディファイアを通さない、元の形：(頂点 (N, 3) float64, 三角形 (M, 3) int64)。

    頂点は obj.data の座標（シェイプキーがあれば、基本のキーの形。_rest_coordinates）。space が "armature" のときは
    armature の中の座標（armature を渡さなければ armature_of(obj)）。三角形は loop_triangles（面の向きは元の面のまま）。
    位置の行列は、計算した matrix_world（計算に入っていない物は ValueError。モジュールの説明を見る）。
    """
    _check_object(obj, "MESH", "obj")
    _check_space(space)
    if space == "armature":
        armature = _check_object(armature if armature is not None else armature_of(obj), "ARMATURE", "armature")
    _sync_edit_mode(obj)
    mesh = obj.data
    vertices = _rest_coordinates(mesh)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    arm_matrix = None if space == "world" else _world_matrix(armature, depsgraph)
    matrix = _to_space(_world_matrix(obj, depsgraph), arm_matrix, space)
    return _transform_points(matrix, vertices), _triangles(mesh)


def mesh_evaluated(obj: Any, depsgraph: Any = None, *, space: str = "world", armature: Any = None) -> np.ndarray:
    """モディファイア（アーマチュアなど）とシェイプキーを通した、いまのフレーム・ポーズの形 (N, 3) float64。

    頂点の並びは元の形（mesh_rest）と同じ。頂点の数が変わるモディファイア（MASK・削減など）があると、並びが
    合わなくなるので ValueError。depsgraph を渡さなければ、いまの view layer の物を使う。
    depsgraph の計算に入っていないメッシュ（ビューポートで無効など）は、モディファイアを通さない形しか取れないので
    ValueError。アーマチュアがレストポジションなら、ポーズは使われない（Blender の画面と同じ形）。
    """
    _check_object(obj, "MESH", "obj")
    _check_space(space)
    if space == "armature":
        armature = _check_object(armature if armature is not None else armature_of(obj), "ARMATURE", "armature")
    _sync_edit_mode(obj, armature)
    if depsgraph is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = _evaluated(obj, depsgraph)
    mesh = evaluated.to_mesh()
    try:
        count, original = len(mesh.vertices), len(obj.data.vertices)
        if count != original:
            changing = [m.name for m in obj.modifiers if m.show_viewport and m.type != "ARMATURE"]
            raise ValueError(f"{obj.name}：モディファイアを通すと頂点の数が変わります（{original} → {count}）。"
                             f"頂点の並びが元の形と合わないので読めません。頂点の数を変えるモディファイア"
                             f"（MASK など。候補：{changing}）を外すか、写しで焼き込んでから読んでください")
        vertices = _vertex_coordinates(mesh)
    finally:
        evaluated.to_mesh_clear()
    arm_matrix = None if space == "world" else _world_matrix(armature, depsgraph)
    return _transform_points(_to_space(_matrix(evaluated.matrix_world), arm_matrix, space), vertices)


# MARK: - ウェイト


def _group_bone_ids(obj: Any, names: Sequence[str]) -> np.ndarray:
    """頂点グループの番号 → 骨の番号（骨でないグループは -1）"""
    lookup = {name: i for i, name in enumerate(names)}
    mapping = np.full(len(obj.vertex_groups), -1, dtype=np.int64)
    for group in obj.vertex_groups:
        mapping[group.index] = lookup.get(group.name, -1)
    return mapping


def non_bone_groups(obj: Any, armature: Any) -> tuple[str, ...]:
    """骨の名前でない頂点グループの名前（頂点グループの順）。bone_weights は、これらを使わない"""
    _check_object(obj, "MESH", "obj")
    names = set(bone_names(armature))
    return tuple(group.name for group in obj.vertex_groups if group.name not in names)


def bone_weights(obj: Any, armature: Any) -> tuple[WeightTable, tuple[str, ...]]:
    """頂点ごとの骨のウェイト：(WeightTable, 骨の名前)。

    骨の番号は、返す骨の名前（= bone_rest(armature).names。armature.data.bones の順）の番号。
    - 頂点グループの名前と骨の名前が、字までまったく同じときだけ骨とみなす（Blender のアーマチュアと同じ）。
      骨でないグループ（non_bone_groups）は使わない。
    - 重みが 0 の物は入れない。枠の数 K は、いちばん多くの骨（重みが 0 でない物）が付いた頂点に合わせる。
    - 頂点の中は、骨の番号の小さい順。形を動かさない骨（use_deform が False）のグループも入れる
      （分けるときは bone_rest(...).deform を使う）。重みは Blender の値のまま（合計で割らない）。
    """
    _check_object(obj, "MESH", "obj")
    names = bone_names(armature)
    _sync_edit_mode(obj)
    mapping = _group_bone_ids(obj, names)
    group_count = len(mapping)
    rows: list[list[tuple[int, float]]] = []
    for vertex in obj.data.vertices:
        row = []
        for element in vertex.groups:
            group = element.group
            if 0 <= group < group_count and mapping[group] >= 0:
                weight = float(element.weight)
                if weight != 0.0:
                    row.append((int(mapping[group]), weight))
        row.sort()
        rows.append(row)
    width = max((len(row) for row in rows), default=0)
    ids = np.full((len(rows), width), -1, dtype=np.int64)
    weights = np.zeros((len(rows), width), dtype=np.float64)
    for index, row in enumerate(rows):
        for slot, (bone, weight) in enumerate(row):
            ids[index, slot] = bone
            weights[index, slot] = weight
    return WeightTable(ids, weights), names


# MARK: - フレームとポーズ


def _frame_values(frames: Any) -> list[tuple[int, float]]:
    """フレームの並びを (整数のフレーム, 小数の部分) の並びにする。数でない・NaN・無限大は ValueError"""
    if isinstance(frames, (str, bytes)) or not isinstance(frames, (Iterable, np.ndarray)):
        raise ValueError(f"frames はフレームの番号の並びにしてください（いまの型：{type(frames).__name__}）")
    if isinstance(frames, np.ndarray) and frames.ndim != 1:
        raise ValueError(f"frames は 1 次元の並びにしてください（いまの形：{frames.shape}）")
    values = []
    for frame in frames:
        if isinstance(frame, (bool, np.bool_)) or not isinstance(frame, (int, float, np.integer, np.floating)):
            raise ValueError(f"フレームの番号は数にしてください（いまの値：{frame!r}）")
        number = float(frame)
        if not math.isfinite(number):
            raise ValueError(f"フレームの番号に NaN か無限大があります（{frame!r}）")
        whole = math.floor(number)
        values.append((int(whole), number - whole))
    return values


def _pose_snapshot(armature: Any) -> list[tuple[Any, str, tuple[float, ...]]]:
    """アーマチュアのオブジェクトと骨の、動かせる値（位置・回転・大きさ）の写し"""
    snapshot = []
    for owner in [armature, *armature.pose.bones]:
        for attribute in ("location", "rotation_quaternion", "rotation_euler", "rotation_axis_angle", "scale"):
            snapshot.append((owner, attribute, tuple(getattr(owner, attribute))))
    return snapshot


def _restore_snapshot(snapshot: list[tuple[Any, str, tuple[float, ...]]]) -> None:
    for owner, attribute, value in snapshot:
        if tuple(getattr(owner, attribute)) != value:
            setattr(owner, attribute, value)


@contextmanager
def _frames_restored(armature: Any) -> Iterator[Any]:
    """フレームを動かしたあと、元のフレームと、アーマチュアのポーズ（キーの無い変更も）に戻す"""
    scene = bpy.context.scene
    frame, subframe = scene.frame_current, scene.frame_subframe
    snapshot = _pose_snapshot(armature)
    try:
        yield scene
    finally:
        scene.frame_set(frame, subframe=subframe)
        _restore_snapshot(snapshot)
        bpy.context.view_layer.update()


def _read_pose(armature: Any, names: Sequence[str], space: str) -> np.ndarray:
    """いまのフレームの、骨のポーズの行列 (B, 4, 4)。並びは names。
    計算に入っていないアーマチュア（ポーズが古いまま）と、レストポジションのアーマチュアは ValueError"""
    _check_pose_position(armature)
    evaluated = _evaluated(armature)
    pose_bones = evaluated.pose.bones
    matrices = np.array([_matrix(pose_bones[name].matrix) for name in names], dtype=np.float64).reshape(-1, 4, 4)
    if space == "world":
        matrices = _matrix(evaluated.matrix_world) @ matrices
    return matrices


def _sample(armature: Any, frames: Any, read: Any) -> np.ndarray:
    """frames の各フレームで read() を呼んで並べる。frames が None なら、いまのフレームだけ"""
    if frames is None:
        return np.stack([read()])
    values = _frame_values(frames)
    if not values:
        return np.stack([read()])[:0]
    results = []
    with _frames_restored(armature) as scene:
        for whole, fraction in values:
            scene.frame_set(whole, subframe=fraction)
            results.append(read())
    return np.stack(results)


def pose_matrices(armature: Any, frames: Any = None, *, space: str = "world") -> np.ndarray:
    """骨のポーズの行列 (F, B, 4, 4) float64。並びは armature.data.bones の順。

    "armature" の空間では Blender の pose_bone.matrix（拘束を計算したあと）、"world" ではアーマチュアの
    matrix_world（そのフレームの物）を左から掛けた物。frames が None なら、いまのフレームだけ（F = 1）。
    フレームを変えたときは、元のフレームとポーズに戻す。
    depsgraph の計算に入っていないアーマチュア（ポーズが古いまま）と、レストポジションのアーマチュアは ValueError。
    """
    _check_object(armature, "ARMATURE", "armature")
    _check_space(space)
    names = bone_names(armature)
    return _sample(armature, frames, lambda: _read_pose(armature, names, space))


def skinning_matrices(armature: Any, frames: Any = None, *, space: str = "armature") -> np.ndarray:
    """休みの形から、そのフレームの形へ動かす行列 (F, B, 4, 4) = ポーズの行列 @ 休みの行列の逆。

    measure.weights.linear_blend_skinning にそのまま渡せる。休みの形の頂点も同じ space で渡す
    （"world" の休みの行列は、呼んだときのアーマチュアの matrix_world で作る）。
    """
    rest = bone_rest(armature, space=space)
    posed = pose_matrices(armature, frames, space=space)
    return posed @ np.linalg.inv(rest.matrices)[None]


def _orthonormalize(matrices: np.ndarray) -> np.ndarray:
    """3×3 の行列を、いちばん近い回転にする（大きさを外す。SVD の極分解）"""
    u, _, vt = np.linalg.svd(matrices)
    rotation = u @ vt
    flip = np.linalg.det(rotation) < 0.0
    if flip.any():
        u = u.copy()
        u[flip, :, 2] *= -1.0
        rotation = u @ vt
    return rotation


def quaternions_from_rotations(rotations: np.ndarray) -> np.ndarray:
    """回転の行列 (..., 3, 3)（列ベクトルに左から掛ける形）を、クォータニオン (..., 4)（w, x, y, z）にする。
    いちばん大きい成分から計算する（180 度近くでも正しい）。w が負にならない方を返す"""
    r = np.asarray(rotations, dtype=np.float64)
    m00, m01, m02 = r[..., 0, 0], r[..., 0, 1], r[..., 0, 2]
    m10, m11, m12 = r[..., 1, 0], r[..., 1, 1], r[..., 1, 2]
    m20, m21, m22 = r[..., 2, 0], r[..., 2, 1], r[..., 2, 2]
    squares = np.stack([1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22,
                        1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], axis=-1) / 4.0
    largest = np.argmax(squares, axis=-1)
    big = np.sqrt(np.maximum(np.take_along_axis(squares, largest[..., None], axis=-1)[..., 0], 0.0))
    scale = 0.25 / np.maximum(big, 1e-300)
    candidates = np.stack([
        np.stack([big, (m21 - m12) * scale, (m02 - m20) * scale, (m10 - m01) * scale], axis=-1),
        np.stack([(m21 - m12) * scale, big, (m01 + m10) * scale, (m02 + m20) * scale], axis=-1),
        np.stack([(m02 - m20) * scale, (m01 + m10) * scale, big, (m12 + m21) * scale], axis=-1),
        np.stack([(m10 - m01) * scale, (m02 + m20) * scale, (m12 + m21) * scale, big], axis=-1),
    ], axis=-2)
    q = np.take_along_axis(candidates, largest[..., None, None], axis=-2)[..., 0, :]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(q[..., :1] < 0.0, -q, q)


def pose_rotations(armature: Any, frames: Any = None) -> np.ndarray:
    """骨を、休みの向きからポーズの向きへ回す、世界の向きでの回転 (F, B, 4)（w, x, y, z）。並びは bones の順。

    = ポーズの世界の回転 @ 休みの世界の回転の逆。measure.kinematics.knee_bend_from_thigh の thigh_rotation に渡す。
    休みの世界の向きは、呼んだときの bone_rest(armature, space="world")（knee_rest_axis に渡す位置と同じ物）。
    行列に大きさ（スケール）が入っていても、外してから回転にする。休みの姿勢では、どの骨も (1, 0, 0, 0)。
    """
    rest = bone_rest(armature, space="world")
    posed = pose_matrices(armature, frames, space="world")
    rest_rotation = _orthonormalize(rest.matrices[:, :3, :3])
    posed_rotation = _orthonormalize(posed[..., :3, :3])
    delta = posed_rotation @ np.swapaxes(rest_rotation, -1, -2)[None]
    return quaternions_from_rotations(delta)


def joint_positions(armature: Any, names: Iterable[str], frames: Any = None, *, space: str = "world") -> np.ndarray:
    """骨の根元（関節）の位置 (F, J, 3) float64。並びは names。アーマチュアに無い名前は ValueError"""
    _check_object(armature, "ARMATURE", "armature")
    _check_space(space)
    if isinstance(names, (str, bytes)):
        raise ValueError("names は骨の名前の並び（リストなど）にしてください。1つの文字列は渡せません")
    wanted = list(names)
    all_names = bone_names(armature)
    lookup = {name: i for i, name in enumerate(all_names)}
    unknown = [name for name in wanted if name not in lookup]
    if unknown:
        raise ValueError(f"アーマチュア {armature.name} に無い骨があります：{unknown}")
    matrices = _sample(armature, frames, lambda: _read_pose(armature, wanted, space))
    return matrices[..., :3, 3].copy()


# MARK: - ポーズを決める（テストと、E2 の確かめのポーズ）


def _as_quaternion(value: Any, name: str) -> tuple[float, float, float, float]:
    """(w, x, y, z) を長さ 1 にする。形が違う・NaN・長さがほぼ 0 なら ValueError"""
    try:
        q = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{name} の回転は (w, x, y, z) の4つの数にしてください") from None
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError(f"{name} の回転は (w, x, y, z) の4つの有限の数にしてください（いまの形：{q.shape}）")
    norm = float(np.linalg.norm(q))
    if norm <= QUATERNION_MIN_NORM:
        raise ValueError(f"{name} の回転の長さが 0 です")
    q = q / norm
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def set_pose(armature: Any, rotations: Mapping[str, Any]) -> None:
    """骨の回転を決める。rotations：骨の名前 → クォータニオン (w, x, y, z)（骨のローカルの向き。休みの向きからの回転）。

    ほかの骨は変えない。骨の回転の形（rotation_mode）がクォータニオンでなければ、その形に直して入れる。
    無い骨の名前・形の違う回転があれば、どれも入れずに ValueError。
    """
    _check_object(armature, "ARMATURE", "armature")
    if not isinstance(rotations, Mapping):
        raise ValueError("rotations は 骨の名前 → (w, x, y, z) の辞書にしてください")
    pose_bones = armature.pose.bones
    unknown = [name for name in rotations if not isinstance(name, str) or pose_bones.get(name) is None]
    if unknown:
        raise ValueError(f"アーマチュア {armature.name} に無い骨があります：{unknown}")
    values = {name: _as_quaternion(value, name) for name, value in rotations.items()}
    for name, q in values.items():
        pose_bone = pose_bones[name]
        mode = pose_bone.rotation_mode
        if mode == "QUATERNION":
            pose_bone.rotation_quaternion = q
        elif mode == "AXIS_ANGLE":
            axis, angle = Quaternion(q).to_axis_angle()
            pose_bone.rotation_axis_angle = (angle, axis[0], axis[1], axis[2])
        else:
            pose_bone.rotation_euler = Quaternion(q).to_euler(mode)
    bpy.context.view_layer.update()


def reset_pose(armature: Any) -> None:
    """全部の骨の位置・回転・大きさを、休みの姿勢に戻す（アクションのキーは消さない）"""
    _check_object(armature, "ARMATURE", "armature")
    for pose_bone in armature.pose.bones:
        pose_bone.location = (0.0, 0.0, 0.0)
        pose_bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pose_bone.rotation_euler = (0.0, 0.0, 0.0)
        pose_bone.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
        pose_bone.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()

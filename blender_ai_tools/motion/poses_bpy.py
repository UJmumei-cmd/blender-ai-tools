"""A5 確かめの動き：決まったポーズ（poses.POSES）を、いまのシーンのモデルに付ける所（Blender の中だけ）。

計算（骨の名前の対応・ローカルの回転・センター の動き・決めた向きになったか）は poses.py（bpy なし）。
ここは extract_bpy で骨とメッシュを読み、ポーズの骨に値を入れるだけにする。

使い方（ポーズを付けて測り、終わったら必ず元に戻す）：

    with posing(armature):
        applied = apply_pose(armature, pose, ground_meshes=body_meshes, floor_z=floor_z)
        ...（extract_bpy.mesh_evaluated などで測る）

- posing() は、中で変えた物を、終わったら（失敗しても）元に戻す：骨の位置・回転・大きさ（extract_bpy.pose_kept）、
  ポーズとぶつかる拘束の「無効」の印、アーマチュアのレストポジションの設定。フレーム・モード・選択・アクティブな
  オブジェクトは変えない。アーマチュアが編集モードのときは、ポーズを付けられないので ValueError。
- ポーズとぶつかる拘束：IK（mmd_tools のモデルでは ひざ に付いている）・スプライン IK と、mmd_tools が IK の角度の
  制限に付ける回転の制限（名前が "mmd_ik_limit" で始まる物）。これらが効いていると、手で決めた回転（FK）が
  IK に上書きされてしまうので、posing() の間だけ無効にする。D の骨の回転付与（Copy Rotation など）はそのまま
  （足 を回すと、モデルの拘束で 足D が付いて来る）。
- apply_pose は posing() の中で呼ぶ（ぶつかる拘束が効いているときは ValueError）。毎回、休みの姿勢に戻してから付ける。
- 元の .blend は保存しない。アクションのキーは消さない・変えない。
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import bpy
import numpy as np

from ..model import extract_bpy
from . import poses
from .poses import AppliedPose, PoseSpec

__all__ = ["FIGHTING_CONSTRAINT_TYPES", "MMD_IK_LIMIT_PREFIX", "apply_pose", "fighting_constraints", "posing"]

# ポーズ（手で決めた回転）とぶつかる拘束の種類
FIGHTING_CONSTRAINT_TYPES = ("IK", "SPLINE_IK")
# mmd_tools が IK の角度の制限のために付ける拘束（回転の制限）の名前の始まり
MMD_IK_LIMIT_PREFIX = "mmd_ik_limit"


# MARK: - 入力の確かめ


def _check_armature(armature: Any) -> Any:
    if not isinstance(armature, bpy.types.Object) or armature.type != "ARMATURE":
        found = getattr(armature, "type", type(armature).__name__)
        raise ValueError(f"armature はアーマチュアのオブジェクトにしてください（いまの物：{found}）")
    return armature


def _is_fighting(constraint: Any) -> bool:
    if constraint.type in FIGHTING_CONSTRAINT_TYPES:
        return True
    return constraint.type == "LIMIT_ROTATION" and constraint.name.startswith(MMD_IK_LIMIT_PREFIX)


def fighting_constraints(armature: Any) -> list[tuple[str, str]]:
    """ポーズとぶつかる拘束のうち、いま効いている物：(ポーズの骨の名前, 拘束の名前) の並び（骨の順）"""
    _check_armature(armature)
    return [(pose_bone.name, constraint.name)
            for pose_bone in armature.pose.bones for constraint in pose_bone.constraints
            if _is_fighting(constraint) and not constraint.mute]


# MARK: - ポーズを付ける間


@contextmanager
def posing(armature: Any) -> Iterator[None]:
    """ポーズを付けて測る間だけ、ぶつかる拘束を無効にし、アーマチュアを「ポーズポジション」にする。

    終わったら（失敗しても）、拘束の印・レストポジションの設定・骨の位置と回転と大きさを元に戻す。
    アーマチュアが編集モードのときは ValueError（モードは変えない）。
    """
    _check_armature(armature)
    if armature.mode == "EDIT":
        raise ValueError(f"アーマチュア {armature.name} は編集モードです。ポーズを付けられないので、"
                         f"編集モードを終えてから動かしてください")
    pose_position = armature.data.pose_position
    muted: list[Any] = []
    with extract_bpy.pose_kept(armature):
        try:
            for pose_bone in armature.pose.bones:
                for constraint in pose_bone.constraints:
                    if _is_fighting(constraint) and not constraint.mute:
                        constraint.mute = True
                        muted.append(constraint)
            if armature.data.pose_position != "POSE":
                armature.data.pose_position = "POSE"
            bpy.context.view_layer.update()
            yield
        finally:
            for constraint in muted:
                constraint.mute = False
            if armature.data.pose_position != pose_position:
                armature.data.pose_position = pose_position
            # 骨の値は pose_kept が戻し、view_layer を計算し直す


# MARK: - 床に合わせる


def _lowest_mesh_z(meshes: Sequence[Any], masks: Sequence[np.ndarray | None]) -> float:
    """メッシュの、いまのポーズの形の、世界のいちばん低い高さ（masks があれば、印の付いた頂点だけ）。頂点が無ければ NaN"""
    lowest = math.inf
    for obj, mask in zip(meshes, masks):
        points = extract_bpy.mesh_evaluated(obj, space="world")
        if mask is not None:
            points = points[mask]
        if len(points):
            lowest = min(lowest, float(points[:, 2].min()))
    return lowest if math.isfinite(lowest) else math.nan


def _ground_masks(meshes: Sequence[Any], masks: Sequence[Any] | None) -> list[np.ndarray | None]:
    """ground_vertex_masks を確かめる（メッシュごとに、頂点の数の真偽の並びか None）"""
    if masks is None:
        return [None] * len(meshes)
    values = list(masks)
    if len(values) != len(meshes):
        raise ValueError(f"ground_vertex_masks の数（{len(values)}）を ground_meshes の数（{len(meshes)}）にそろえてください")
    result: list[np.ndarray | None] = []
    for obj, mask in zip(meshes, values):
        if mask is None:
            result.append(None)
            continue
        array = np.asarray(mask)
        if array.dtype != bool or array.shape != (len(obj.data.vertices),):
            raise ValueError(f"{obj.name} の ground_vertex_masks は、頂点の数（{len(obj.data.vertices)}）の真偽の並びに"
                             f"してください（いまの形：{array.shape}、型：{array.dtype}）")
        result.append(array)
    return result


def _bone_points(rest: extract_bpy.BoneRest, posed_world: np.ndarray) -> np.ndarray:
    """形を動かす骨（無ければ全部の骨）の、根元と先の世界の位置 (2K, 3)。posed_world はアーマチュアの骨の行列（世界）"""
    selected = np.flatnonzero(rest.deform) if rest.deform.any() else np.arange(rest.bone_count)
    lengths = np.linalg.norm(rest.tails[selected] - rest.heads[selected], axis=1)
    matrices = posed_world[selected]
    heads = matrices[:, :3, 3]
    tails = heads + matrices[:, :3, 1] * lengths[:, None]
    return np.concatenate([heads, tails])


def _location_basis(armature: Any, rest: extract_bpy.BoneRest, posed: np.ndarray, index: int) -> np.ndarray:
    """骨の位置（pose_bone.location）を 1 動かすと、アーマチュアの中で骨がどう動くか (3, 3)。

    Blender（armature.cc の BKE_bone_parent_transform_calc_from_matrices）：骨の位置は
    親のポーズ @ (親の休み⁻¹ @ 休み) の向きで当たる（根の骨は休みの向き）。
    「ローカルの位置」を使わない骨（use_local_location が False）は、アーマチュアの向きのまま当たる。
    """
    if not armature.data.bones[rest.names[index]].use_local_location:
        return np.eye(3)
    own = rest.matrices[index, :3, :3]
    parent = int(rest.parents[index])
    if parent < 0:
        return own
    return posed[parent, :3, :3] @ np.linalg.inv(rest.matrices[parent, :3, :3]) @ own


def _shift_center(armature: Any, rest: extract_bpy.BoneRest, center: str, world_dz: float) -> None:
    """センター を、世界の上下に world_dz（メートル）動かす（骨の位置に足す）"""
    index = rest.index(center)
    posed = extract_bpy.pose_matrices(armature, space="armature")[0]
    arm_world = extract_bpy.world_matrix(armature)
    in_armature = np.linalg.solve(arm_world[:3, :3], np.array([0.0, 0.0, world_dz]))
    local = np.linalg.solve(_location_basis(armature, rest, posed, index), in_armature)
    pose_bone = armature.pose.bones[center]
    pose_bone.location = tuple(float(a + b) for a, b in zip(pose_bone.location, local))
    bpy.context.view_layer.update()


# MARK: - ポーズを付ける


def apply_pose(armature: Any, pose: PoseSpec, *, leg_length: float | None = None,
               ground_meshes: Sequence[Any] = (), ground_vertex_masks: Sequence[Any] | None = None,
               floor_z: float | None = None) -> AppliedPose:
    """ポーズを付ける（posing() の中で呼ぶ）。返す物は AppliedPose（回した骨・無かった骨・決めた向きにならなかった骨）。

    1. 休みの姿勢に戻す（extract_bpy.reset_pose）。
    2. 共通の名前をモデルの骨の名前に対応させ（bones.map_bones）、骨のローカルの回転を入れる（extract_bpy.set_pose）。
       モデルに無い骨は飛ばし、missing に入れる（足りないポーズ。確かめは灰にする）。
    3. center_offset_legs があれば、センター を 脚の長さ × center_offset_legs だけ動かす。leg_length は
       アーマチュアの中の長さ（None なら、骨から測る：poses.leg_length_from_bones）。
    4. ground なら、いちばん低い所を床に合わせるように、センター を世界の上下に動かす：
       - ground_meshes があれば、その形（いまのポーズ、世界の座標）のいちばん低い頂点を、floor_z（None なら、
         休みの姿勢の同じメッシュのいちばん低い高さ）に合わせる。ground_vertex_masks（メッシュごとの頂点の印か None）を
         渡すと、印の付いた頂点だけを見る（1つのメッシュに体と服の材質があるとき、体の頂点だけで合わせる）。
       - 無ければ、形を動かす骨の根元と先のいちばん低い所を、休みの姿勢での同じ点のいちばん低い高さに合わせる（目安）。
    5. 骨の向きが決めた向きになったかを確かめる（poses.find_overridden）。
    ポーズとぶつかる拘束が効いている（posing() の外で呼んだ）とき、レストポジションのときは ValueError。
    """
    _check_armature(armature)
    if not isinstance(pose, PoseSpec):
        raise ValueError(f"pose は PoseSpec にしてください（いまは {type(pose).__name__}）")
    active = fighting_constraints(armature)
    if active:
        names = "、".join(f"{bone}（{constraint}）" for bone, constraint in active)
        raise ValueError(f"ポーズとぶつかる拘束が効いています：{names}。posing() の中で呼んでください")
    if armature.data.pose_position != "POSE":
        raise ValueError(f"アーマチュア {armature.name} はレストポジションです。posing() の中で呼んでください")
    meshes = tuple(ground_meshes)
    masks = _ground_masks(meshes, ground_vertex_masks)
    if floor_z is not None and (isinstance(floor_z, bool) or not isinstance(floor_z, (int, float))
                                or not math.isfinite(float(floor_z))):
        raise ValueError(f"floor_z は有限の数にしてください（いまは {floor_z!r}）")

    extract_bpy.reset_pose(armature)
    rest = extract_bpy.bone_rest(armature, space="armature")
    used_leg = None
    if any(v != 0.0 for v in pose.center_offset_legs):
        used_leg = leg_length if leg_length is not None else poses.leg_length_from_bones(rest.names, rest.heads)
    reference = None
    if pose.ground:
        if meshes:
            reference = float(floor_z) if floor_z is not None else _lowest_mesh_z(meshes, masks)
        else:
            rest_world = extract_bpy.bone_rest(armature, space="world")
            reference = float(_bone_points(rest_world, rest_world.matrices)[:, 2].min())

    resolved = poses.resolve_pose(pose, rest.names, rest.matrices, leg_length=used_leg)
    if resolved.rotations:
        extract_bpy.set_pose(armature, resolved.rotations)
    pose_bones = armature.pose.bones
    if resolved.center is not None and any(v != 0.0 for v in resolved.center_location):
        pose_bones[resolved.center].location = resolved.center_location
        bpy.context.view_layer.update()

    shift = None
    grounded_by = None
    if pose.ground and resolved.center is not None and reference is not None and math.isfinite(reference):
        if meshes:
            lowest = _lowest_mesh_z(meshes, masks)
            grounded_by = "mesh"
        else:
            rest_world = extract_bpy.bone_rest(armature, space="world")
            posed_world = extract_bpy.pose_matrices(armature, space="world")[0]
            lowest = float(_bone_points(rest_world, posed_world)[:, 2].min())
            grounded_by = "bones"
        if math.isfinite(lowest):
            shift = reference - lowest
            _shift_center(armature, rest, resolved.center, shift)
        else:
            grounded_by = None

    deltas = poses.rotation_deltas(rest.matrices, extract_bpy.pose_matrices(armature, space="armature")[0])
    overridden = poses.find_overridden(resolved, rest.names, rest.parents, deltas)
    location = (0.0, 0.0, 0.0)
    if resolved.center is not None:
        location = tuple(float(v) for v in pose_bones[resolved.center].location)
    posed = {resolved.canonical[actual]: actual for actual in resolved.rotations}
    return AppliedPose(pose=pose.name, posed=posed, missing=resolved.missing, overridden=overridden,
                       center_location=location, ground_shift=shift, grounded_by=grounded_by,
                       leg_length=None if used_leg is None else float(used_leg))

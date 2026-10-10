"""E2 往復の確認：いまの Blender のモデルと、書き出した USDZ を比べる（bpy を使う所）。

ロードマップ段階1 基準4：いくつかの姿勢で、読み戻した形のずれが thresholds.ROUNDTRIP_MAX_DEVIATION（0.5mm）未満。
名前は originalNames を通して比べる。1点に付く骨は4本まで。物理の部品が入っていない。

run_roundtrip の流れ：
1. 書き出した USDZ（usd_read_bpy.read_usdz）と character_rig.json を読む。
2. 書き出しの対象（extract_bpy.find_model。convert_blend.py と同じ決まり）を探す。
3. 休みの姿勢（アーマチュアを「レストポジション」にする）で Blender の形を計算し、書き出した点と対応させる
   （roundtrip.match_meshes）。抜け・余り（check_coverage）と、休みの姿勢のずれ（compare_rest）を測る。
4. 確かめのポーズ（roundtrip.TEST_POSES）を1つずつ付け、Blender の形と、書き出した形を同じ骨の行列で動かした点を
   比べる（compare_pose）。骨の行列は、書き出した骨の名前 → originalNames.bones → 元の骨の pose_bone.matrix。
5. 骨格の休みの姿勢（check_skeleton）・名前（check_names）・骨の数と重み（check_influences）・物理の部品
   （check_no_physics）を調べる。

Blender の形は、書き出しと同じ設定で計算する：
- 書き出し（convert_blend.py と Blender の USD の書き出し）は、描画（レンダー）の設定を使う。そこで計算の間だけ、
  モディファイアのビューポートの表示を描画の表示と同じにし、MASK は外す（MASK で消える頂点は、
  mask_hidden_vertices で「消えるはずの頂点」として数える）。隠した物（hide_viewport・目のマーク）も表示する。
- シェイプキーは、いまの値のまま（convert_blend.py も、表情に使わないキーを、いまの値で形に焼き込む）。
- 編集モードの物は、編集中の形を先にデータへ写してから読む（extract_bpy.sync_edit_mode。モードはそのまま）。
ずれの長さは、アプリで見える長さ（メートル。roundtrip.skeleton_to_meters）で測る。
終わったら、ポーズ・レストポジションの設定・モディファイアと物の表示を、元に戻す。フレーム・モード・選択は変えない。
元の .blend にも、書き出したファイルにも、何も書かない。
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import bpy
import numpy as np

from ..measure.results import Measurement
from ..model import extract_bpy
from . import roundtrip
from .roundtrip import DEVIATION_NAME, REST_POSE, TEST_POSES
from .usd_read_bpy import read_usdz

__all__ = ["REST_POSE", "TEST_POSES", "mask_hidden_vertices", "physics_object_names", "pose_label", "run_roundtrip",
           "source_shape_keys"]

# mmd_tools の部品の種類（Object.mmd_type）のうち、ふつうの物
_PLAIN_MMD_TYPES = ("", "NONE")


# MARK: - 元のモデルを調べる


def mask_hidden_vertices(obj: Any) -> np.ndarray:
    """convert_blend.py が書き出しで消す頂点の印 (N,) bool（apply_masks と同じ決まり）。

    描画で有効（show_render）な MASK モディファイアのうち、頂点グループで選ぶ物（mode が "VERTEX_GROUP"）：
    頂点の重みが threshold より大きければ「グループの中」。中か外かが invert_vertex_group と同じなら消える。
    MASK が2つ以上なら、どれかで消える頂点を全部。骨で選ぶ MASK・グループが無い MASK では消えない。
    """
    if not isinstance(obj, bpy.types.Object) or obj.type != "MESH":
        raise ValueError(f"obj はメッシュのオブジェクトにしてください（いまの物：{getattr(obj, 'type', type(obj).__name__)}）")
    vertices = obj.data.vertices
    hidden = np.zeros(len(vertices), dtype=bool)
    for modifier in obj.modifiers:
        if modifier.type != "MASK" or not modifier.show_render or modifier.mode != "VERTEX_GROUP":
            continue
        group = obj.vertex_groups.get(modifier.vertex_group)
        if group is None:
            continue
        index = group.index
        for vertex in vertices:
            weight = next((g.weight for g in vertex.groups if g.group == index), 0.0)
            inside = weight > modifier.threshold
            if inside == modifier.invert_vertex_group:
                hidden[vertex.index] = True
    return hidden


def physics_object_names(scene: Any = None) -> tuple[str, ...]:
    """書き出してはいけない物（剛体・ジョイント・Empty・mmd_tools の部品）のオブジェクトの名前。

    Empty、剛体（rigid_body）、剛体のつなぎ（rigid_body_constraint）、mmd_tools の mmd_type がふつうでない物。
    探す物の並びは bpy.data.objects（convert_blend.py と同じ）。scene を渡すと、その中の物だけ。
    """
    objects = list(bpy.data.objects)
    if scene is not None:
        in_scene = set(scene.objects)
        objects = [obj for obj in objects if obj in in_scene]
    names = []
    for obj in objects:
        mmd_type = str(getattr(obj, "mmd_type", "NONE"))
        if (obj.type == "EMPTY" or obj.rigid_body is not None or obj.rigid_body_constraint is not None
                or mmd_type not in _PLAIN_MMD_TYPES):
            names.append(obj.name)
    return tuple(names)


def source_shape_keys(meshes: Sequence[Any]) -> tuple[str, ...]:
    """メッシュのシェイプキーの名前（基本の形のキーを除く。重なりは1つにする）"""
    names: dict[str, None] = {}
    for obj in meshes:
        keys = obj.data.shape_keys
        if keys is None:
            continue
        reference = keys.reference_key
        for block in keys.key_blocks:
            if block != reference:
                names.setdefault(block.name, None)
    return tuple(names)


# MARK: - 計算の間だけ変える物


@contextmanager
def _evaluation_state(armature: Any, meshes: Sequence[Any]) -> Iterator[None]:
    """書き出しと同じ設定で形を計算するために変える物を覚えておき、終わったら元に戻す。

    変える物：骨の位置・回転・大きさ（extract_bpy.pose_kept が戻す）、レストポジションの設定、
    物の表示（hide_viewport・目のマーク）、モディファイアのビューポートの表示（描画の表示と同じにし、MASK は外す）
    """
    with extract_bpy.pose_kept(armature):
        with _display_state(armature, meshes):
            yield


@contextmanager
def _display_state(armature: Any, meshes: Sequence[Any]) -> Iterator[None]:
    """レストポジションの設定と、物・モディファイアの表示を、終わったら元に戻す（_evaluation_state の中身）"""
    view_layer = bpy.context.view_layer
    pose_position = armature.data.pose_position
    visibility = []
    for obj in (armature, *meshes):
        in_layer = view_layer.objects.get(obj.name) is not None
        visibility.append((obj, obj.hide_viewport, obj.hide_get() if in_layer else None))
    modifiers = [(modifier, modifier.show_viewport) for obj in meshes for modifier in obj.modifiers]
    try:
        for obj, _, hidden in visibility:
            obj.hide_viewport = False
            if hidden:
                obj.hide_set(False)
        for obj in meshes:
            for modifier in obj.modifiers:
                modifier.show_viewport = False if modifier.type == "MASK" else modifier.show_render
        view_layer.update()
        yield
    finally:
        for modifier, shown in modifiers:
            if modifier.show_viewport != shown:
                modifier.show_viewport = shown
        for obj, hide_viewport, hidden in visibility:
            if obj.hide_viewport != hide_viewport:
                obj.hide_viewport = hide_viewport
            if hidden is not None and obj.hide_get() != hidden:
                obj.hide_set(hidden)
        if armature.data.pose_position != pose_position:
            armature.data.pose_position = pose_position
        view_layer.update()


def _evaluate(meshes: Sequence[Any], armature: Any,
              names: Mapping[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """いまのポーズの形（骨格＝アーマチュアの空間）。返す物：(名前 → 頂点 (N, 3), 計算できなかった名前 → 理由)。
    names を渡すと、その中の名前のメッシュだけ"""
    points: dict[str, np.ndarray] = {}
    failed: dict[str, str] = {}
    for obj in meshes:
        if names is not None and obj.name not in names:
            continue
        try:
            points[obj.name] = extract_bpy.mesh_evaluated(obj, space="armature", armature=armature)
        except ValueError as error:
            failed[obj.name] = str(error)
    return points, failed


# MARK: - 往復の確認


def _gray(reason: str, pose: str | None = None) -> Measurement:
    return Measurement.unmeasured(DEVIATION_NAME, "m", reason, {} if pose is None else {"pose": pose})


def run_roundtrip(usdz_path: str | os.PathLike[str], rig_json: Any, *, scene: Any = None,
                  poses: Mapping[str, Mapping[str, tuple[Sequence[float], float]]] | None = None) -> list[Measurement]:
    """書き出した USDZ（usdz_path）と character_rig.json（rig_json：場所か、読んだ辞書）を、いまのモデルと比べる。

    返す物（この順）：
        "roundtrip_coverage"、"roundtrip_deviation"（休みの姿勢。details["pose"] = "rest"）、
        "roundtrip_deviation"（ポーズごと。details["pose"] はポーズの名前）、"skeleton_rest_consistency"、
        "joint_names"・"humanoid_bones"・"shape_key_names"、"max_bone_influences"・"joint_weight_sums"、"physics_parts"
    poses：確かめのポーズ（None なら roundtrip.TEST_POSES）。scene：None か、いまのシーン（ほかのシーンは ValueError）。
    ファイルが読めない・モデルが見つからないときは、灰の "roundtrip_deviation" 1つだけを返す。
    ポーズ・形・レストポジションの設定・表示は、終わったら元に戻す。
    """
    if scene is not None and scene != bpy.context.scene:
        raise ValueError("scene は、いまのシーン（bpy.context.scene）にしてください（ほかのシーンの形は計算されません）")
    test_poses = TEST_POSES if poses is None else poses
    if not isinstance(test_poses, Mapping):
        raise ValueError("poses は ポーズの名前 → {共通の名前: (軸, 角度)} の辞書にしてください")
    try:
        exported = read_usdz(usdz_path)
        rig = roundtrip.load_rig(rig_json)
        original_bones = roundtrip.original_names(rig, "bones")
    except ValueError as error:
        return [_gray(f"書き出したファイルを読めませんでした：{error}")]
    try:
        model = extract_bpy.find_model(scene)
    except ValueError as error:
        return [_gray(f"書き出しの対象のモデルが見つかりません：{error}")]
    armature, meshes = model.armature, model.meshes
    # 編集モードの物は、編集中の形を先にデータへ写す（モードは変えない。extract_bpy と同じ決まり）。
    # 写さないと、頂点の数・MASK で消える頂点を古いデータで数え、計算した形と数が合わなくなる
    extract_bpy.sync_edit_mode(armature, *meshes)
    rest = extract_bpy.bone_rest(armature, space="armature")
    resolved = {name: roundtrip.resolve_pose(pose, rest.names, rest.matrices) for name, pose in test_poses.items()}
    rest_joints = roundtrip.posed_joint_matrices(exported.joint_names, original_bones, rest.names, rest.matrices)
    hidden = {obj.name: mask_hidden_vertices(obj) for obj in meshes}
    counts = {obj.name: len(obj.data.vertices) for obj in meshes}

    results: list[Measurement] = []
    with _evaluation_state(armature, meshes):
        armature.data.pose_position = "REST"
        bpy.context.view_layer.update()
        source_rest, failed = _evaluate(meshes, armature)
        matches = roundtrip.match_meshes(exported, source_rest,
                                         expected_missing={k: v for k, v in hidden.items() if k in source_rest})
        results.append(roundtrip.check_coverage(exported, counts, matches, expected_missing=hidden,
                                                unmeasured=failed))
        results.append(roundtrip.compare_rest(exported, source_rest, rest_joints, matches))
        armature.data.pose_position = "POSE"
        for pose_name, (rotations, missing) in resolved.items():
            extra = {"applied_bones": tuple(rotations), "missing_bones": missing}
            try:
                extract_bpy.reset_pose(armature)
                if rotations:
                    extract_bpy.set_pose(armature, rotations)
                posed = extract_bpy.pose_matrices(armature, space="armature")[0]
                source_posed, _ = _evaluate(meshes, armature, source_rest)
            except ValueError as error:
                results.append(Measurement.unmeasured(DEVIATION_NAME, "m", f"ポーズの形を計算できませんでした：{error}",
                                                      {"pose": pose_name, **extra}))
                continue
            joints = roundtrip.posed_joint_matrices(exported.joint_names, original_bones, rest.names, posed)
            results.append(roundtrip.compare_pose(exported, source_posed, joints, matches, pose=pose_name,
                                                  rest=rest_joints, extra_details=extra))

    deform = tuple(name for name, flag in zip(rest.names, rest.deform) if flag)
    results.append(roundtrip.check_skeleton(exported, original_bones))
    results.extend(roundtrip.check_names(rig, exported, rest.names, source_shape_keys(meshes),
                                         source_deform_bones=deform))
    results.extend(roundtrip.check_influences(exported))
    results.append(roundtrip.check_no_physics(exported, physics_object_names(scene)))
    return results


def pose_label(measurement: Measurement) -> str | None:
    """"roundtrip_deviation" の結果のポーズの名前（休みの姿勢は REST_POSE）。ほかの結果は None"""
    if measurement.name != DEVIATION_NAME:
        return None
    return measurement.details.get("pose")


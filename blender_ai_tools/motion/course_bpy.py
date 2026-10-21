"""A5 確かめの動き：いまのシーンのモデルに、決まったポーズを1つずつ付けて測る（Blender の中だけ）。

計算は course.py・poses.py（bpy なし）。ここは extract_bpy で並びを取り出し、poses_bpy でポーズを付けるだけにする。

run_course の流れ：
1. 書き出しの対象（extract_bpy.find_model。convert_blend.py と同じ決まり）を探す。
2. メッシュを「体」と「服」に分ける。A1 の登録カード（card。渡さなければ、シーンに置いたカードを読む）が
   同じアーマチュアの物なら、カードの部品と材質の種類を使う。カードに無い部品は、名前から推測する
   （model.card.guess_part_kind・guess_material。推測はただの案なので、カードで決めたほうがよい）。
   材質ごとに種類のある部品は、三角形ごとに材質の種類で分ける（1つのメッシュに肌と服がある MMD のモデル）。
3. posing() の中で：休みの姿勢の体の形から、床の高さ（体のいちばん低い高さ）と、体のつながった部品（閉じた形か）を
   調べる。ポーズを1つずつ付け（apply_pose。床に合わせるポーズは、体の頂点で合わせる）、形と骨を読んで、
   course の確かめ（ひざの逆曲がり・床への沈み・服からの体のはみ出し）をする。
4. 終わったら（失敗しても）ポーズ・拘束・レストポジションの設定を元に戻す（posing）。フレーム・モード・選択・
   アクティブなオブジェクトは変えない。元の .blend は保存しない。
5. アプリの動き（motions.json）の灰の出来事を足す（course.app_motion_events）。

形を読めないメッシュ（ビューポートで無効・頂点の数が変わるモディファイア など）は使わず、それを使う確かめを灰にする。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import bpy
import numpy as np

from .. import bones, texts
from ..jobs import progress
from ..measure.geometry import PENETRATION_NAME
from ..measure.kinematics import FLOOR_SINK_NAME, REVERSE_BEND_NAME
from ..measure.results import Measurement
from ..measure.visibility import render_views
from ..model import card_bpy, extract_bpy
from ..model.card import LEG_LENGTH, ModelCard, guess_material, guess_part_kind
from . import course, poses_bpy
from .course import CHECKS, CLOTH_CHECK, KNEE_CHECK, SINK_CHECK, Event
from .poses import POSES, PoseSpec

__all__ = ["MeshRoles", "classify_meshes", "run_course"]

# ひざを測る骨（左。右も同じ）：形を動かす D の骨を先に使い、無ければ見える骨
_KNEE_CHAINS = (("左足D", "左ひざD", "左足首D"), ("左足", "左ひざ", "左足首"))
# 確かめの名前 → 測った結果の名前（灰にするときに使う）
_MEASUREMENT_NAMES = {KNEE_CHECK: (REVERSE_BEND_NAME, "deg"), SINK_CHECK: (FLOOR_SINK_NAME, "m"),
                      CLOTH_CHECK: (PENETRATION_NAME, "m")}


# MARK: - 体と服に分ける


@dataclass(frozen=True, eq=False)
class MeshRoles:
    """1つのメッシュの、体と服の分け方。

    obj: メッシュのオブジェクト
    triangles: 三角形 (M, 3)（loop_triangles。mesh_evaluated の頂点の番号）
    body_triangles: 体の三角形の印 (M,) bool
    cloth_vertices: 服の三角形が使う頂点の印 (N,) bool
    source: 決め方の記録（"card"：カードの種類、"guess"：名前からの推測）
    """

    obj: Any
    triangles: np.ndarray
    body_triangles: np.ndarray
    cloth_vertices: np.ndarray
    source: str

    @property
    def body_vertices(self) -> np.ndarray:
        """体の三角形が使う頂点の印 (N,) bool"""
        mask = np.zeros(len(self.cloth_vertices), dtype=bool)
        mask[self.triangles[self.body_triangles].ravel()] = True
        return mask

    @property
    def has_body(self) -> bool:
        return bool(self.body_triangles.any())

    @property
    def has_cloth(self) -> bool:
        return bool(self.cloth_vertices.any())


def _part_kinds(obj: Any, card: ModelCard | None) -> tuple[str, dict[str, str], str]:
    """(部品の種類, 材質の名前 → 種類, 決め方)。カードに部品があればカードの種類、無ければ名前からの推測"""
    _, slot_names = extract_bpy.triangle_materials(obj)
    names = tuple(dict.fromkeys(name for name in slot_names if name))
    if card is not None:
        try:
            part = card.part(obj.name)
        except ValueError:
            part = None
        if part is not None:
            return part.kind, {item.name: item.kind for item in part.materials}, "card"
    kind, _ = guess_part_kind(obj.name, names)
    materials = {name: guess_material(name).kind for name in names} if len(names) >= 2 else {}
    return kind, materials, "guess"


def _mesh_triangles(obj: Any) -> np.ndarray:
    """メッシュの三角形 (M, 3) int64（loop_triangles。extract_bpy.mesh_rest と同じ並び）。位置の計算は要らないので、
    ビューポートで無効のメッシュでも読める"""
    extract_bpy.sync_edit_mode(obj)
    triangles = obj.data.loop_triangles
    flat = np.empty(len(triangles) * 3, dtype=np.int32)
    triangles.foreach_get("vertices", flat)
    return flat.reshape(-1, 3).astype(np.int64)


def classify_meshes(meshes: Sequence[Any], card: ModelCard | None = None) -> list[MeshRoles]:
    """メッシュごとに、体の三角形と服の頂点を決める（モジュールの説明の 2）。種類が "body" の物が体、"cloth" の物が服"""
    roles = []
    for obj in meshes:
        kind, materials, source = _part_kinds(obj, card)
        triangles = _mesh_triangles(obj)
        indices, slot_names = extract_bpy.triangle_materials(obj)
        slot_kinds = [materials.get(name, kind) if name else kind for name in slot_names]
        kinds = np.array(slot_kinds + [kind], dtype=object)
        inside = (indices >= 0) & (indices < len(slot_names))
        triangle_kinds = kinds[np.where(inside, indices, len(slot_names))]
        body = triangle_kinds == "body"
        cloth = np.zeros(len(obj.data.vertices), dtype=bool)
        cloth[triangles[triangle_kinds == "cloth"].ravel()] = True
        roles.append(MeshRoles(obj=obj, triangles=triangles, body_triangles=np.asarray(body, dtype=bool),
                               cloth_vertices=cloth, source=source))
    return roles


# MARK: - いまの形を読む


@dataclass
class _Shapes:
    """いまのポーズの形（世界の座標）。読めなかったメッシュは unreadable に入れる"""

    vertices: dict[str, np.ndarray] = field(default_factory=dict)
    unreadable: dict[str, str] = field(default_factory=dict)


def _read_shapes(wanted: Sequence[MeshRoles]) -> _Shapes:
    shapes = _Shapes()
    for role in wanted:
        try:
            shapes.vertices[role.obj.name] = extract_bpy.mesh_evaluated(role.obj, space="world")
        except ValueError as error:
            shapes.unreadable[role.obj.name] = str(error)
    return shapes


def _body_arrays(roles: Sequence[MeshRoles], shapes: _Shapes) -> tuple[np.ndarray, np.ndarray]:
    """体の頂点（体の三角形が使う頂点だけ）と三角形（その並びの番号）を、メッシュをまたいでつなぐ"""
    points, triangles = [], []
    offset = 0
    for role in roles:
        if not role.has_body or role.obj.name not in shapes.vertices:
            continue
        used = np.flatnonzero(role.body_vertices)
        remap = np.full(len(role.cloth_vertices), -1, dtype=np.int64)
        remap[used] = np.arange(len(used)) + offset
        points.append(shapes.vertices[role.obj.name][used])
        triangles.append(remap[role.triangles[role.body_triangles]])
        offset += len(used)
    if not points:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(points), np.concatenate(triangles)


def _cloth_points(roles: Sequence[MeshRoles], shapes: _Shapes) -> np.ndarray:
    points = [shapes.vertices[role.obj.name][role.cloth_vertices]
              for role in roles if role.has_cloth and role.obj.name in shapes.vertices]
    return np.concatenate(points) if points else np.zeros((0, 3))


def _scene_mesh(roles: Sequence[MeshRoles], shapes: _Shapes) -> tuple[np.ndarray, np.ndarray]:
    """見え方を調べるための、読めた全部のメッシュをつないだ形"""
    points, triangles = [], []
    offset = 0
    for role in roles:
        vertices = shapes.vertices.get(role.obj.name)
        if vertices is None:
            continue
        points.append(vertices)
        triangles.append(role.triangles + offset)
        offset += len(vertices)
    return np.concatenate(points), np.concatenate(triangles)


# MARK: - ひざ


@dataclass(frozen=True)
class _KneeBones:
    """ひざを測る骨（左, 右の順。無い側は None）"""

    chains: tuple[tuple[str, str, str] | None, tuple[str, str, str] | None]
    missing: tuple[str, ...]


def _knee_bones(names: Sequence[str]) -> _KneeBones:
    by_canonical = bones.map_bones(names).by_canonical
    chains: list[tuple[str, str, str] | None] = []
    missing: list[str] = []
    for side in ("左", "右"):
        found = None
        for chain in _KNEE_CHAINS:
            canonical = tuple(bone.replace("左", side, 1) for bone in chain)
            actual = [by_canonical.get(bone) for bone in canonical]
            if None not in actual:
                found = tuple(actual)
                break
        if found is None:
            visible = tuple(bone.replace("左", side, 1) for bone in _KNEE_CHAINS[1])
            missing.extend(bone for bone in visible if bone not in by_canonical)
        chains.append(found)  # type: ignore[arg-type]
    return _KneeBones(chains=(chains[0], chains[1]), missing=tuple(missing))


def _knee_measurement(armature: Any, knee: _KneeBones, rest_world: extract_bpy.BoneRest) -> Measurement:
    """いまのポーズの、左右のひざの逆曲がり"""
    nan3 = np.full(3, np.nan)
    rest_hips, rest_knees, hips, knees, ankles = ([nan3, nan3] for _ in range(5))
    thighs = [np.full(4, np.nan), np.full(4, np.nan)]
    rotations = extract_bpy.pose_rotations(armature)[0]
    for side, chain in enumerate(knee.chains):
        if chain is None:
            continue
        hip, knee_bone, ankle = chain
        rest_hips[side] = rest_world.heads[rest_world.index(hip)]
        rest_knees[side] = rest_world.heads[rest_world.index(knee_bone)]
        posed = extract_bpy.joint_positions(armature, chain, space="world")[0]
        hips[side], knees[side], ankles[side] = posed
        thighs[side] = rotations[rest_world.index(hip)]
    # キャラの左の向き：左右の股関節の差。片側が無いときは、アーマチュアの +X（キャラの左）の世界の向き
    lateral = np.asarray(rest_hips[0]) - np.asarray(rest_hips[1])
    if not np.isfinite(lateral).all() or np.linalg.norm(lateral) == 0.0:
        lateral = extract_bpy.world_matrix(armature)[:3, 0]
    return course.check_knees(rest_hips=np.array(rest_hips), rest_knees=np.array(rest_knees), hips=np.array(hips),
                              knees=np.array(knees), ankles=np.array(ankles), thigh_rotations=np.array(thighs),
                              lateral=lateral, missing_bones=knee.missing)


# MARK: - 確かめの動き


def _check_scene(scene: Any) -> None:
    if scene is not None and scene != bpy.context.scene:
        raise ValueError("確かめの動きは、いまのシーンのモデルだけを動かせます（ほかのシーンは、いまのシーンにしてから）")


def _load_card() -> ModelCard | None:
    """シーンに置いたカード。無い・読めないときは None（名前から推測する）"""
    try:
        return card_bpy.load_card()
    except ValueError:
        return None


def _leg_length(card: ModelCard | None, armature: Any) -> float | None:
    """カードの脚の長さ（世界の長さ）を、アーマチュアの中の長さにする。使えないときは None（骨から測る）"""
    if card is None or card.armature_name != armature.name:
        return None
    values = [card.value(LEG_LENGTH[side]) for side in ("L", "R")]
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    scale = abs(float(np.linalg.det(extract_bpy.world_matrix(armature)[:3, :3]))) ** (1.0 / 3.0)
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    return float(np.mean(values)) / scale


def _gray(check: str, reason: str) -> Measurement:
    name, unit = _MEASUREMENT_NAMES[check]
    return Measurement.unmeasured(name, unit, reason)


def _check_inputs(poses: Sequence[PoseSpec], checks: Sequence[str]) -> tuple[list[PoseSpec], list[str]]:
    pose_list = list(poses)
    if not all(isinstance(pose, PoseSpec) for pose in pose_list):
        raise ValueError("poses は PoseSpec の並びにしてください")
    if len({pose.name for pose in pose_list}) != len(pose_list):
        raise ValueError("poses に同じ名前のポーズがあります")
    if isinstance(checks, str):
        raise ValueError("checks は確かめの名前の並びにしてください")
    check_list = list(checks)
    unknown = [check for check in check_list if check not in CHECKS]
    if unknown:
        raise ValueError(f"知らない確かめの名前があります：{unknown}（{'・'.join(CHECKS)} のどれか）")
    return pose_list, [check for check in CHECKS if check in check_list]


def run_course(scene: Any = None, *, poses: Sequence[PoseSpec] = POSES, checks: Sequence[str] = CHECKS,
               visibility: bool = False, card: ModelCard | None = None, app_motions: Sequence[str] | None = None,
               orbit: Mapping[str, Any] | None = None) -> list[Event]:
    """いまのシーンのモデルに、ポーズを1つずつ付けて確かめ、出来事の並びを返す（ポーズの順、確かめは CHECKS の順。
    最後にアプリの動きの灰の出来事）。

    - poses：確かめるポーズ（既定は POSES の全部）。checks：する確かめ（CHECKS の中から）。
    - visibility：True なら、服からの体のはみ出しを、まわり全体のカメラ（course.orbit_cameras。orbit は
      orbit_views に渡す設定）から見える物だけ数える。重い（ポーズごとに全部のカメラで描く）。
    - card：A1 の登録カード。None なら、シーンに置いたカード（card_bpy.load_card）を読む。カードは、アーマチュアの
      名前が同じときだけ、部品の種類と脚の長さに使う。
    - app_motions：アプリの動きの名前（motions.json から読んだ物）。None なら、まとめて1つの灰の出来事。
    - scene：None か、いまのシーン（ほかのシーンは ValueError）。モデルが見つからないときも ValueError
      （extract_bpy.find_model の日本語の説明のまま）。
    ポーズ・拘束・レストポジションの設定は、終わったら（失敗しても）元に戻す。フレーム・モード・選択は変えない。
    """
    _check_scene(scene)
    pose_list, check_list = _check_inputs(poses, checks)
    if not isinstance(visibility, bool):
        raise ValueError("visibility は True か False にしてください")
    if card is not None and not isinstance(card, ModelCard):
        raise ValueError(f"card は ModelCard にしてください（いまは {type(card).__name__}）")
    orbit_options = dict(orbit or {})
    model = extract_bpy.find_model(scene)
    armature = model.armature
    if card is None:
        card = _load_card()
    usable_card = card if card is not None and card.armature_name == armature.name else None
    roles = classify_meshes(model.meshes, usable_card)
    body_roles = [role for role in roles if role.has_body]
    cloth_roles = [role for role in roles if role.has_cloth]
    wanted = roles if visibility else [role for role in roles if role.has_body or role.has_cloth]
    leg_length = _leg_length(usable_card, armature)

    events: list[Event] = []
    with poses_bpy.posing(armature):
        extract_bpy.reset_pose(armature)
        rest_bones = extract_bpy.bone_rest(armature, space="armature")
        rest_world = extract_bpy.bone_rest(armature, space="world")
        knee = _knee_bones(rest_bones.names)
        rest_shapes = _read_shapes(wanted)
        rest_body, rest_body_triangles = _body_arrays(roles, rest_shapes)
        floor_z = float(rest_body[:, 2].min()) if len(rest_body) else None
        parts = course.body_parts(rest_body, rest_body_triangles)
        body_unreadable = [role.obj.name for role in body_roles if role.obj.name in rest_shapes.unreadable]
        cloth_unreadable = [role.obj.name for role in cloth_roles if role.obj.name in rest_shapes.unreadable]
        readable_body = [role for role in body_roles if role.obj.name not in rest_shapes.unreadable]
        ground_meshes = [role.obj for role in readable_body] if floor_z is not None else []
        ground_masks = [role.body_vertices for role in readable_body] if floor_z is not None else None
        for index, pose in enumerate(pose_list):
            # 裏の係（A10）の中なら進み具合を書く（中止が頼まれていれば JobCancelled が上がり、posing() が元に戻す）。
            # 画面の中では何もしない
            progress.report(index, len(pose_list), "motion.progress_pose", {"pose": course.pose_label(pose.name)})
            try:
                applied = poses_bpy.apply_pose(armature, pose, leg_length=leg_length, ground_meshes=ground_meshes,
                                               ground_vertex_masks=ground_masks, floor_z=floor_z)
                shapes = _read_shapes(wanted)
                measurements = _measure(pose, check_list, armature, knee, rest_world, roles, shapes, parts,
                                        floor_z=floor_z, body_unreadable=body_unreadable,
                                        cloth_unreadable=cloth_unreadable, has_body=bool(body_roles),
                                        has_cloth=bool(cloth_roles), visibility=visibility, orbit=orbit_options)
            except ValueError as error:
                reason = texts.text("motion.reason_pose_failed", error=str(error))
                events.extend(course.pose_events(pose, {check: _gray(check, reason) for check in check_list}))
                continue
            events.extend(course.pose_events(pose, measurements, applied))
    events.extend(course.app_motion_events(app_motions))
    return events


def _measure(pose: PoseSpec, checks: Sequence[str], armature: Any, knee: _KneeBones,
             rest_world: extract_bpy.BoneRest, roles: Sequence[MeshRoles], shapes: _Shapes,
             parts: course.BodyParts, *, floor_z: float | None, body_unreadable: Sequence[str],
             cloth_unreadable: Sequence[str], has_body: bool, has_cloth: bool, visibility: bool,
             orbit: Mapping[str, Any]) -> dict[str, Measurement]:
    """1つのポーズの確かめ（ポーズを付けたあとに呼ぶ）"""
    sep = texts.text("motion.list_separator")
    measurements: dict[str, Measurement] = {}
    if KNEE_CHECK in checks:
        measurements[KNEE_CHECK] = _knee_measurement(armature, knee, rest_world)
    body, body_triangles = _body_arrays(roles, shapes)
    body_problem = None
    if not has_body:
        body_problem = texts.text("motion.reason_no_body")
    elif body_unreadable or floor_z is None or len(body) != parts.vertex_count:
        names = sep.join(body_unreadable or [role.obj.name for role in roles if role.obj.name in shapes.unreadable])
        body_problem = texts.text("motion.reason_unreadable_meshes", names=names)
    if SINK_CHECK in checks and not pose.lying:
        measurements[SINK_CHECK] = (_gray(SINK_CHECK, body_problem) if body_problem is not None
                                    else course.check_floor_sink(body, floor_z=floor_z))
    if CLOTH_CHECK in checks:
        if body_problem is not None:
            measurements[CLOTH_CHECK] = _gray(CLOTH_CHECK, body_problem)
        elif not has_cloth:
            measurements[CLOTH_CHECK] = _gray(CLOTH_CHECK, texts.text("motion.reason_no_cloth"))
        elif cloth_unreadable:
            measurements[CLOTH_CHECK] = _gray(CLOTH_CHECK, texts.text("motion.reason_unreadable_meshes",
                                                                      names=sep.join(cloth_unreadable)))
        else:
            renders = scene_mesh = None
            if visibility:
                scene_mesh = _scene_mesh(roles, shapes)
                renders = render_views(scene_mesh[0], scene_mesh[1],
                                       course.orbit_cameras(scene_mesh[0], **orbit))
            measurements[CLOTH_CHECK] = course.check_cloth_penetration(
                _cloth_points(roles, shapes), body, parts, renders=renders, scene_mesh=scene_mesh)
    return measurements

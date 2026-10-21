"""A4 測る道具の自己テストの見本（bpy なし・numpy だけ）。

ロードマップ段階1 基準1（CLAUDE.md 10章「できあがりの基準」1）：
「わざと壊した見本（約20種）を全部見つける。壊していない見本では赤が0。同じ入力を2回測ると、決めた幅の中で同じ数字になる。」
このための見本を、乱数の種を決めて作る（何度作っても、まったく同じ数字になる）。

- clean_sample()：壊していない見本（Sample）。
  - 体：閉じていて、面の向きがそろい、左右がそろった楕円の胴（tests/shapes.py の球を伸ばした物）。
  - 服：体のまわりの殻。すき間は thresholds.CLOTH_GAP_MIN〜CLOTH_GAP_MAX のまん中。
  - ウェイト：(N, K) の骨の番号（空きは -1）と重み（合計 1、1点 MAX_BONE_INFLUENCES 本まで）。
    準標準の骨の名前で、形を動かさない骨（左足ＩＫ）も1本持つ。
  - ポーズ：上半身を少し前に曲げる（骨の行列）。動いた形は skin()（Blender のアーマチュアと同じ決まり）で作る。
    ポーズのときだけ効くシェイプキー（関節の補正の形）pose_shape も足す（壊していない見本では 0）。
  - 前の版：再現性の幅（REPRO_TOLERANCE_LENGTH）より小さい差と、「変えてよい所」の中だけの変化。
  - 動き：その場で足踏みする2本の脚（股関節・ひざ・足首・足の裏・IK の目標・脚の骨の回転）。
- BREAKAGES：わざと壊した見本（名前 → Breakage）。apply が壊した写しを作る（もとの見本は変えない）。
  finds の測る物（Measurement の名前）が expected の信号になる。also は、一緒に変わってよい測る物と、その信号
  （なぜ一緒に変わるかは、apply の説明に書く）。
  体の形を壊す見本は、前の版にも同じ壊れ方を入れる（壊れた所は前からあった。版の比べには出ない）。
- GRAY_CASES：測れない見本。finds の測る物が灰になる（赤にしてはいけない。CLAUDE.md 11章：ひざの逆曲がりは、
  まちがえて赤にしない）。
- HARMLESS：形も動きも変えない書き換え（どの信号も変わらない）。
- KNOWN_MISSES：いまの測り方では見つからない壊れ方（見つかるのが正しい。直したら BREAKAGES へ移す）。
- measure_sample()：見本を、A4 の測る道具（measure/ の summarize_*）で測る。

壊し方の大きさは、基準の数字（thresholds）から決める。基準が変わっても、壊れ方が基準のどちら側にあるかは変わらない。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any

import numpy as np

from blender_ai_tools import thresholds
from blender_ai_tools.measure import deformation, geometry, kinematics, mesh_check, symmetry, weights
from blender_ai_tools.measure.results import Measurement, Signal
from tests import shapes

__all__ = [
    "BREAKAGES",
    "GRAY_CASES",
    "HARMLESS",
    "KNOWN_MISSES",
    "Breakage",
    "Sample",
    "clean_sample",
    "measure_sample",
    "skin",
]

# MARK: - 見本の形（基準の数字ではない。見本の大きさ）

# 体（楕円の胴）：中心と、x・y・z の半径（m）。球を分ける回数（2 で頂点 162、三角形 320）
BODY_CENTER = np.array([0.0, 0.0, 1.05])
BODY_RADII = np.array([0.16, 0.11, 0.30])
BODY_SUBDIVISIONS = 2
# 腰（上半身を曲げる軸の高さ）と、上半身・下半身のウェイトを混ぜる幅（m）
WAIST_HEIGHT = 1.10
WAIST_BAND = 0.15
# この高さより下は、下半身と足D を混ぜる（m）。混ぜる幅と、足D の重みのいちばん大きい割合
LEG_HEIGHT = 0.90
LEG_BAND = 0.10
LEG_SHARE = 0.5
# この高さより上は、上半身と上半身2 を混ぜる（m）。上半身2 の重みのいちばん大きい割合
CHEST_HEIGHT = 1.22
CHEST_BAND = 0.10
CHEST_SHARE = 0.5
# ポーズ：上半身を前（-Y）へ曲げる角度（度）。ゆるく曲げる（形のくずれの基準より、ずっと内側）
POSE_BEND_DEG = 10.0

# 骨（準標準の名前）。左足ＩＫ は形を動かさない骨（IK の目標。ウェイトを乗せても数えない）
BONE_NAMES = ("センター", "下半身", "上半身", "上半身2", "首", "左足D", "右足D", "腰", "左足ＩＫ")
CENTER, LOWER, UPPER, UPPER2, NECK, LEFT_LEG, RIGHT_LEG, WAIST, LEFT_IK = range(len(BONE_NAMES))
# ポーズで曲げる骨（ほかの骨は動かない）
BENT_BONES = (UPPER, UPPER2, NECK)

# 動き：フレームの数と速さ。左足は前半が床に着き、後半に上がる。右足はその逆
FRAME_COUNT = 60
FPS = 30.0
SWING_FRAMES = 30
# 脚の長さ（m）：太もも・すね。股関節の高さと、左右の間（中心から）。足首から足の裏までの高さ
THIGH_LENGTH = 0.40
SHIN_LENGTH = 0.40
HIP_HEIGHT = 0.85
HIP_HALF_WIDTH = 0.09
ANKLE_HEIGHT = 0.06
# 上げた足の高さと、前（-Y）へ出す長さ（m）
STEP_HEIGHT = 0.08
STEP_FORWARD = 0.05
# 脚の骨の回転を記録する骨（太もも・すねの、休みの向き（真下）からの世界の向きの回転）
LEG_BONE_NAMES = ("左足", "左ひざ", "右足", "右ひざ")
SIDES = ("left", "right")

# MARK: - 壊し方の大きさ（基準の数字に掛ける割合）

# 基準をはっきり超える：基準の何倍にするか
OVER = 2.0
# 基準の内側にとどめる：基準の何倍にするか
UNDER = 0.5
# ポーズ（POSE_BEND_DEG）で、もともと縮む辺のいちばん小さい割合の見込み。shrunk_pose の縮め方を決めるのに使う
# （test_self_check が、壊していない見本の実際の値がこれより大きいことを確かめる）
CLEAN_MIN_EDGE_RATIO = 0.85


# MARK: - 見本

@dataclass(frozen=True, eq=False)
class Sample:
    """測る見本。長さはメートル、世界の座標（Z が上、モデルは -Y を向く、キャラの左は +X）。

    体：body_vertices (N, 3)・body_triangles (M, 3)（休みの姿勢。閉じた形で、外から見て反時計回り）。
    服：cloth_vertices (C, 3)・cloth_triangles（休みの姿勢）。
    ウェイト：bone_ids (N, K)（空きは -1）・bone_weights (N, K)（空きは 0）。骨は bone_names (B,)、
        形を動かす骨の印 deform_bones (B,)。
    ポーズ：pose_matrices (B, 4, 4)（休みからポーズへ動かす、骨ごとの行列）。pose_shape (N, 3) は、ポーズのときだけ効く
        シェイプキー（関節の補正の形）の、休みの座標での動き。動いた形は posed_body_vertices()。
    前の版：previous_body_vertices (N, 3) と、変えてよい頂点の印 allowed_changes (N,)。
    動き（F フレーム、左・右の順）：hips・knees・ankles・soles（足の裏）・ik_targets は (F, 2, 3)。
        leg_lengths (2,)（太もも＋すね）、lateral (3,)（キャラの左の向き）、fps、floor_z（床の高さ）。
        leg_rotations (F, R, 4)（w, x, y, z）は leg_bone_names (R,) の骨の、世界の向きの回転。
    seed：前の版の小さな差に使った乱数の種。

    中の並びは写しを取り、書き換えられないようにする。壊すときは新しい並びを作り、dataclasses.replace で入れ替える。
    """

    body_vertices: np.ndarray
    body_triangles: np.ndarray
    cloth_vertices: np.ndarray
    cloth_triangles: np.ndarray
    bone_names: tuple[str, ...]
    deform_bones: np.ndarray
    bone_ids: np.ndarray
    bone_weights: np.ndarray
    pose_matrices: np.ndarray
    pose_shape: np.ndarray
    previous_body_vertices: np.ndarray
    allowed_changes: np.ndarray
    fps: float
    floor_z: float
    lateral: np.ndarray
    leg_lengths: np.ndarray
    hips: np.ndarray
    knees: np.ndarray
    ankles: np.ndarray
    soles: np.ndarray
    ik_targets: np.ndarray
    leg_bone_names: tuple[str, ...]
    leg_rotations: np.ndarray
    seed: int

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, np.ndarray):
                copy = np.array(value)
                copy.flags.writeable = False
                object.__setattr__(self, item.name, copy)
        count = len(self.body_vertices)
        for name in ("bone_ids", "bone_weights", "pose_shape", "previous_body_vertices", "allowed_changes"):
            if len(getattr(self, name)) != count:
                raise ValueError(f"{name} の行の数が、体の頂点の数（{count}）と違います")
        if self.bone_ids.shape != self.bone_weights.shape:
            raise ValueError("bone_ids と bone_weights の形が違います")
        bones = len(self.bone_names)
        if self.deform_bones.shape != (bones,) or self.pose_matrices.shape != (bones, 4, 4):
            raise ValueError(f"deform_bones・pose_matrices の形が、骨の数（{bones}）と合いません")
        frames = len(self.hips)
        for name in ("knees", "ankles", "soles", "ik_targets"):
            if getattr(self, name).shape != (frames, 2, 3):
                raise ValueError(f"{name} の形は ({frames}, 2, 3) にしてください")
        if self.leg_rotations.shape != (frames, len(self.leg_bone_names), 4):
            raise ValueError("leg_rotations の形が、フレームの数と骨の数に合いません")

    def posed_body_vertices(self) -> np.ndarray:
        """ポーズを付けた体の頂点 (N, 3)：休みの形に pose_shape を足してから、骨で動かす（skin）"""
        return skin(self.body_vertices + self.pose_shape, self.bone_ids, self.bone_weights, self.pose_matrices,
                    self.deform_bones)

    def weight_table(self) -> weights.WeightTable:
        return weights.WeightTable(self.bone_ids, self.bone_weights)


def skin(points: Any, bone_ids: Any, bone_weights: Any, matrices: Any, deform: Any) -> np.ndarray:
    """Blender のアーマチュア（Armature モディファイア）と同じ決まりで、休みの形を骨で動かす（bpy なし）。

    - 形を動かす骨（deform が真）の、正の重みだけを使い、その合計で割る（合計が 1 でなくてもよい）。
    - 形を動かす骨の重みが1つも無い頂点は、動かさない（休みの位置のまま）。
    - 動いた位置 = Σ w_k (M_k @ [p, 1]) / Σ w_k。M_k は骨 k の 4×4 の行列（列ベクトルに左から掛ける）。
    測る道具（measure/weights.py の linear_blend_skinning）とは別に書く（見本を、測る道具の答えに頼らずに作るため）。
    """
    p = np.asarray(points, dtype=np.float64)
    ids = np.asarray(bone_ids, dtype=np.int64)
    w = np.asarray(bone_weights, dtype=np.float64)
    m = np.asarray(matrices, dtype=np.float64)
    use = (ids >= 0) & (w > 0.0)
    use &= np.asarray(deform, dtype=bool)[np.maximum(ids, 0)]
    w = np.where(use, w, 0.0)
    total = w.sum(axis=1)
    blended = np.einsum("nk,nkij->nij", w, m[np.where(use, ids, 0)])
    moved = np.einsum("nij,nj->ni", blended[:, :3, :3], p) + blended[:, :3, 3]
    result = p.copy()
    has = total > 0.0
    result[has] = moved[has] / total[has, None]
    return result


# MARK: - 小さな計算

def _rotation_x(angle_deg: float, pivot: np.ndarray) -> np.ndarray:
    """pivot を通る +X の軸のまわりに angle_deg 度回す 4×4 の行列（正の角度で、上の点が前 -Y へ動く）"""
    angle = math.radians(angle_deg)
    rotation = np.array([[1.0, 0.0, 0.0],
                         [0.0, math.cos(angle), -math.sin(angle)],
                         [0.0, math.sin(angle), math.cos(angle)]])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = pivot - rotation @ pivot
    return matrix


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def _area_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """頂点ごとの外向きの向き（まわりの面の、面積で重みを付けた法線の和を長さ 1 にした物）。
    測る道具（geometry.vertex_normals）とは別に書く"""
    a, b, c = (vertices[triangles[:, k]] for k in range(3))
    cross = np.cross(b - a, c - a)
    total = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(total, triangles[:, k], cross)
    return _unit(total)


def _table_from_dense(dense: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """(N, B) の重みを、(N, width) の骨の番号（空きは -1）と重み（空きは 0）にする。重みの大きい順、同じなら骨の番号の順"""
    order = np.lexsort((np.broadcast_to(np.arange(dense.shape[1]), dense.shape), -dense), axis=1)
    ids = np.full((len(dense), width), -1, dtype=np.int64)
    values = np.zeros((len(dense), width))
    for row in range(len(dense)):
        kept = [bone for bone in order[row] if dense[row, bone] > 0.0]
        if len(kept) > width:
            raise ValueError(f"頂点 {row} の骨が {width} 本を超えます（見本の作り方のまちがい）")
        ids[row, :len(kept)] = kept
        values[row, :len(kept)] = dense[row, kept]
    return ids, values


def _between(low: float, high: float, what: str) -> float:
    """low と high のまん中。low < high でないときは ValueError（基準が変わって、壊し方を決められなくなった）"""
    if not low < high:
        raise ValueError(f"{what}：基準の数字の組み合わせで、壊し方の大きさを決められません（{low} 〜 {high}）")
    return (low + high) / 2.0


def _quaternion_x(angle: np.ndarray) -> np.ndarray:
    """+X の軸のまわりの回転（ラジアン）のクォータニオン (..., 4)（w, x, y, z）"""
    half = np.asarray(angle) / 2.0
    zeros = np.zeros_like(half)
    return np.stack([np.cos(half), np.sin(half), zeros, zeros], axis=-1)


def _quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """クォータニオンの積 first × second（w, x, y, z）"""
    w1, x1, y1, z1 = np.moveaxis(np.asarray(first), -1, 0)
    w2, x2, y2, z2 = np.moveaxis(np.asarray(second), -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


# MARK: - 壊していない見本

def _body_weights(vertices: np.ndarray) -> np.ndarray:
    """体の頂点ごとの重み (N, B)：高さで下半身・上半身を混ぜ、下は足D、上は上半身2 も混ぜる。左右はそろえる"""
    x, z = vertices[:, 0], vertices[:, 2]
    upper = np.clip((z - WAIST_HEIGHT) / WAIST_BAND + 0.5, 0.0, 1.0)
    lower = 1.0 - upper
    leg = lower * np.clip((LEG_HEIGHT - z) / LEG_BAND, 0.0, 1.0) * LEG_SHARE
    chest = upper * np.clip((z - CHEST_HEIGHT) / CHEST_BAND, 0.0, 1.0) * CHEST_SHARE
    dense = np.zeros((len(vertices), len(BONE_NAMES)))
    dense[:, LOWER] = lower - leg
    dense[:, UPPER] = upper - chest
    dense[:, UPPER2] = chest
    left, right = x > 0.0, x < 0.0
    middle = ~left & ~right
    dense[left, LEFT_LEG] = leg[left]
    dense[right, RIGHT_LEG] = leg[right]
    dense[middle, LEFT_LEG] = leg[middle] / 2.0
    dense[middle, RIGHT_LEG] = leg[middle] / 2.0
    return dense


def _leg_from_sole(hip: np.ndarray, sole: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """足の裏の位置から、足首とひざの位置を決める（2本の骨の IK。ひざは前 -Y へ曲がる）。hip・sole は (F, 3)"""
    ankle = sole + np.array([0.0, 0.0, ANKLE_HEIGHT])
    reach = ankle - hip
    distance = np.linalg.norm(reach, axis=1)
    if (distance > THIGH_LENGTH + SHIN_LENGTH).any():
        raise ValueError("足の裏が、脚の長さより遠くにあります（見本の作り方のまちがい）")
    along = reach / distance[:, None]
    # 股関節から足首への向きに垂直で、前（-Y）を向く向き（脚は YZ の面の中にある）
    forward = np.stack([np.zeros(len(along)), along[:, 2], -along[:, 1]], axis=1)
    a = (THIGH_LENGTH ** 2 - SHIN_LENGTH ** 2 + distance ** 2) / (2.0 * distance)
    b = np.sqrt(np.maximum(THIGH_LENGTH ** 2 - a ** 2, 0.0))
    knee = hip + a[:, None] * along + b[:, None] * forward
    return ankle, knee


def _bone_rotations(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """休みの向き（真下 -Z）から、start → end の向き（YZ の面の中）へ回す、+X まわりの回転 (F, 4)"""
    direction = end - start
    return _quaternion_x(np.arctan2(direction[:, 1], -direction[:, 2]))


def _motion() -> dict[str, Any]:
    """その場で足踏みする2本の脚。左足は前半が床に着き後半に上がる。右足はその逆。上げた足は前へ出て、同じ所へ戻る"""
    frames = np.arange(FRAME_COUNT)
    hips = np.zeros((FRAME_COUNT, 2, 3))
    soles = np.zeros((FRAME_COUNT, 2, 3))
    for side, sign in enumerate((1.0, -1.0)):
        hips[:, side] = [sign * HIP_HALF_WIDTH, 0.0, HIP_HEIGHT]
        swing_start = SWING_FRAMES if side == 0 else 0
        phase = (frames - swing_start) / SWING_FRAMES
        swinging = (phase >= 0.0) & (phase < 1.0)
        arc = np.where(swinging, np.sin(np.pi * np.clip(phase, 0.0, 1.0)), 0.0)
        soles[:, side] = np.stack([np.full(FRAME_COUNT, sign * HIP_HALF_WIDTH), -STEP_FORWARD * arc,
                                   STEP_HEIGHT * arc], axis=1)
    ankles = np.zeros_like(soles)
    knees = np.zeros_like(soles)
    rotations = np.zeros((FRAME_COUNT, len(LEG_BONE_NAMES), 4))
    for side in range(2):
        ankles[:, side], knees[:, side] = _leg_from_sole(hips[:, side], soles[:, side])
        rotations[:, 2 * side] = _bone_rotations(hips[:, side], knees[:, side])
        rotations[:, 2 * side + 1] = _bone_rotations(knees[:, side], ankles[:, side])
    return {
        "hips": hips, "knees": knees, "ankles": ankles, "soles": soles, "ik_targets": ankles.copy(),
        "leg_rotations": rotations, "lateral": hips[0, 0] - hips[0, 1],
        "leg_lengths": np.full(2, THIGH_LENGTH + SHIN_LENGTH),
    }


def clean_sample(seed: int = thresholds.DEFAULT_SEED) -> Sample:
    """壊していない見本。どの測る道具でも赤が出ず、全部を測れる（灰が出ない）ように作る"""
    unit_vertices, triangles = shapes.icosphere(radius=1.0, subdivisions=BODY_SUBDIVISIONS)
    body = unit_vertices * BODY_RADII + BODY_CENTER
    normals = _area_normals(body, triangles)
    gap = (thresholds.CLOTH_GAP_MIN + thresholds.CLOTH_GAP_MAX) / 2.0
    cloth = body + gap * normals
    ids, values = _table_from_dense(_body_weights(body), thresholds.MAX_BONE_INFLUENCES)
    deform = np.array([name != BONE_NAMES[LEFT_IK] for name in BONE_NAMES])
    pose = np.broadcast_to(np.eye(4), (len(BONE_NAMES), 4, 4)).copy()
    for bone in BENT_BONES:
        pose[bone] = _rotation_x(POSE_BEND_DEG, np.array([0.0, 0.0, WAIST_HEIGHT]))
    # 前の版：どの頂点にも、再現性の幅の半分の小さな差（変化として数えない）。胸（変えてよい所）は、はっきり違う
    rng = np.random.default_rng(seed)
    noise = _unit(rng.normal(size=body.shape)) * (UNDER * thresholds.REPRO_TOLERANCE_LENGTH)
    allowed = body[:, 2] > CHEST_HEIGHT
    previous = body + noise
    previous[allowed] += normals[allowed] * (OVER * OVER * thresholds.REPRO_TOLERANCE_LENGTH)
    return Sample(body_vertices=body, body_triangles=triangles, cloth_vertices=cloth, cloth_triangles=triangles.copy(),
                  bone_names=BONE_NAMES, deform_bones=deform, bone_ids=ids, bone_weights=values, pose_matrices=pose,
                  pose_shape=np.zeros_like(body), previous_body_vertices=previous, allowed_changes=allowed,
                  fps=FPS, floor_z=0.0, leg_bone_names=LEG_BONE_NAMES, seed=seed, **_motion())


# MARK: - 見本を測る

def measure_sample(sample: Sample, *, chunk_size: int | None = None) -> dict[str, Measurement]:
    """見本を A4 の測る道具で測る。返す物：「どこ/名前」→ Measurement（名前は Measurement.name）。

    - body/…：体の面の点検（mesh_check.summarize_mesh、閉じているはずの形）と左右のずれ（symmetry）。
    - cloth/…：服の点の、体の中へのめり込み（geometry.summarize_penetration）と、体とのすき間（summarize_gap）。
    - weights/…：ウェイトの点検（weights.summarize。形を動かす骨だけを数える）。
    - pose/…：休みの形と、ポーズを付けた形の比べ（deformation の伸び縮み・裏返り・体積・とげ）。
      とげは、どの面にも使われていない頂点を除いてから測る（summarize_spikes の説明のとおり）。
    - version/…：前の版との比べ（deformation.summarize_changes。変えてよい所の外の変化）。
    - left/…・right/…：脚ごとの、ひざの逆曲がり・足すべり・届かないフレーム（kinematics）。
    - feet/floor_sink：両足の裏の、床への沈み。motion/bone_jumps：脚の骨の跳び。
    chunk_size を渡すと、小分けにして計算する道具（めり込み・すき間・左右のずれ・骨の跳び）に渡す
    （小分けの仕方で答えが変わらないことを確かめるため）。
    """
    chunk: dict[str, int] = {} if chunk_size is None else {"chunk_size": chunk_size}
    body, triangles = sample.body_vertices, sample.body_triangles
    results: dict[str, Measurement] = {}
    for measurement in mesh_check.summarize_mesh(body, triangles, closed_required=True):
        results[f"body/{measurement.name}"] = measurement
    results[f"body/{symmetry.ASYMMETRY_NAME}"] = symmetry.summarize_asymmetry(symmetry.asymmetry(body, **chunk))
    results[f"cloth/{geometry.PENETRATION_NAME}"] = geometry.summarize_penetration(
        geometry.penetration_depths(sample.cloth_vertices, body, triangles, **chunk))
    results[f"cloth/{geometry.GAP_NAME}"] = geometry.summarize_gap(
        geometry.gaps(sample.cloth_vertices, body, triangles, **chunk))
    for measurement in weights.summarize(sample.weight_table(), deform=sample.deform_bones):
        results[f"weights/{measurement.name}"] = measurement
    posed = sample.posed_body_vertices()
    edges = deformation.unique_edges(triangles)
    results[f"pose/{deformation.STRETCH_NAME}"] = deformation.summarize_stretch(
        deformation.edge_length_ratios(body, posed, edges))
    results[f"pose/{deformation.FLIPS_NAME}"] = deformation.summarize_flips(
        *deformation.triangle_flips(body, posed, triangles))
    results[f"pose/{deformation.VOLUME_NAME}"] = deformation.summarize_volume(
        deformation.volume_change_ratio(body, posed, triangles))
    used = np.bincount(triangles.ravel(), minlength=len(body)) > 0
    results[f"pose/{deformation.SPIKES_NAME}"] = deformation.summarize_spikes(
        deformation.spike_scores(body, posed, triangles)[used])
    results[f"version/{deformation.CHANGES_NAME}"] = deformation.summarize_changes(
        sample.previous_body_vertices, body, allowed=sample.allowed_changes)
    for side, label in enumerate(SIDES):
        hip, knee, ankle = sample.hips[:, side], sample.knees[:, side], sample.ankles[:, side]
        sole = sample.soles[:, side]
        results[f"{label}/{kinematics.REVERSE_BEND_NAME}"] = kinematics.summarize_reverse_bend(
            kinematics.knee_bend(hip, knee, ankle, sample.lateral))
        contacts = kinematics.foot_contacts(sole, fps=sample.fps, floor_z=sample.floor_z)
        results[f"{label}/{kinematics.FOOT_SLIDE_NAME}"] = kinematics.summarize_slide(
            kinematics.slide_distances(sole, contacts))
        results[f"{label}/{kinematics.UNREACHABLE_NAME}"] = kinematics.summarize_unreachable(
            kinematics.unreachable_frames(hip, sample.ik_targets[:, side], sample.leg_lengths[side]))
    results[f"feet/{kinematics.FLOOR_SINK_NAME}"] = kinematics.summarize_sink(
        kinematics.sink_depths(sample.soles, floor_z=sample.floor_z))
    results[f"motion/{kinematics.BONE_JUMPS_NAME}"] = kinematics.summarize_jumps(
        kinematics.bone_jump_frames(sample.leg_rotations, **chunk), bone_names=sample.leg_bone_names)
    return results


# MARK: - 壊し方の部品

def _writable(array: np.ndarray) -> np.ndarray:
    return np.array(array)


def _static_bones(sample: Sample) -> np.ndarray:
    """ポーズで動かない骨の印 (B,)（行列が単位行列）"""
    return np.all(sample.pose_matrices == np.eye(4), axis=(1, 2))


def _static_vertices(sample: Sample) -> np.ndarray:
    """ポーズで動かない頂点の印 (N,)：形を動かす骨の重みが、全部ポーズで動かない骨にある（下半身のあたり）。
    ここを壊すと、ポーズを付けた形も同じだけ変わり、形のくずれ（pose/…）は変わらない"""
    ids = sample.bone_ids
    filled = (ids >= 0) & (sample.bone_weights > 0.0)
    moving = filled & ~_static_bones(sample)[np.maximum(ids, 0)]
    return filled.any(axis=1) & ~moving.any(axis=1)


def _pick(mask: np.ndarray, key: np.ndarray, count: int = 1) -> np.ndarray:
    """mask が真の頂点から、key の小さい順に count 個の番号（同じなら番号の小さい順）"""
    index = np.flatnonzero(mask)
    if len(index) < count:
        raise ValueError("壊す頂点が足りません（見本の作り方のまちがい）")
    order = np.lexsort((index, key[index]))
    return index[order][:count]


def _front_static(sample: Sample, count: int) -> np.ndarray:
    """動かない頂点のうち、いちばん前（-Y）にある物から count 個"""
    return _pick(_static_vertices(sample), sample.body_vertices[:, 1], count)


def _side_static(sample: Sample) -> int:
    """動かない頂点のうち、キャラの左（+X）にいちばん出ている物（左右の真ん中の面の上にはない）"""
    return int(_pick(_static_vertices(sample), -sample.body_vertices[:, 0])[0])


def _middle_static_edge(sample: Sample) -> tuple[int, int]:
    """両端が左右の真ん中の面（x = 0）の上にあり、どちらも動かない体の辺 (a, b)。いちばん低い物"""
    edges = deformation.unique_edges(sample.body_triangles)
    x = sample.body_vertices[:, 0]
    static = _static_vertices(sample)
    keep = (x[edges[:, 0]] == 0.0) & (x[edges[:, 1]] == 0.0) & static[edges[:, 0]] & static[edges[:, 1]]
    if not keep.any():
        raise ValueError("左右の真ん中の、動かない辺がありません（見本の作り方のまちがい）")
    candidates = edges[keep]
    height = sample.body_vertices[candidates].mean(axis=1)[:, 2]
    a, b = candidates[int(np.lexsort((np.arange(len(candidates)), height))[0])]
    return int(a), int(b)


def _triangle_with_edge(triangles: np.ndarray, start: int, end: int) -> tuple[int, int]:
    """向きのある辺 start → end を持つ三角形の番号と、その中での start の位置"""
    for corner in range(3):
        rows = np.flatnonzero((triangles[:, corner] == start) & (triangles[:, (corner + 1) % 3] == end))
        if len(rows):
            return int(rows[0]), corner
    raise ValueError(f"辺 {start} → {end} を持つ三角形がありません")


def _normals(sample: Sample) -> np.ndarray:
    return _area_normals(sample.body_vertices, sample.body_triangles)


def _move_body(sample: Sample, index: np.ndarray, offsets: np.ndarray, *, with_cloth: bool) -> Sample:
    """体の頂点を動かす。前の版にも同じだけ動かす（壊れた所は前からあった）。with_cloth なら、同じ番号の服の点も動かす
    （服は体の殻なので、番号が同じ）"""
    body = _writable(sample.body_vertices)
    previous = _writable(sample.previous_body_vertices)
    body[index] += offsets
    previous[index] += offsets
    changes: dict[str, Any] = {"body_vertices": body, "previous_body_vertices": previous}
    if with_cloth:
        cloth = _writable(sample.cloth_vertices)
        cloth[index] += offsets
        changes["cloth_vertices"] = cloth
    return replace(sample, **changes)


def _append_body(sample: Sample, points: np.ndarray, weight_rows: np.ndarray,
                 triangles: np.ndarray | None = None) -> Sample:
    """体に頂点を足す（ウェイトは weight_rows の頂点の写し）。前の版にも同じ頂点を足す。triangles を渡すと入れ替える"""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    rows = np.asarray(weight_rows, dtype=np.int64).reshape(-1)
    return replace(
        sample,
        body_vertices=np.vstack([sample.body_vertices, points]),
        body_triangles=sample.body_triangles if triangles is None else triangles,
        bone_ids=np.vstack([sample.bone_ids, sample.bone_ids[rows]]),
        bone_weights=np.vstack([sample.bone_weights, sample.bone_weights[rows]]),
        pose_shape=np.vstack([sample.pose_shape, np.zeros_like(points)]),
        previous_body_vertices=np.vstack([sample.previous_body_vertices, points]),
        allowed_changes=np.concatenate([sample.allowed_changes, np.zeros(len(points), dtype=bool)]),
    )


def _set_weight_row(ids: np.ndarray, values: np.ndarray, row: int, entries: list[tuple[int, float]]) -> None:
    """1つの頂点の枠を、(骨, 重み) の並びに入れ替える（残りの枠は空き）"""
    ids[row] = -1
    values[row] = 0.0
    for slot, (bone, weight) in enumerate(entries):
        ids[row, slot] = bone
        values[row, slot] = weight


def _with_pose_shape(sample: Sample, offsets: np.ndarray) -> Sample:
    return replace(sample, pose_shape=sample.pose_shape + offsets)


def _mean_edge_length(sample: Sample, vertex: int) -> float:
    edges = deformation.unique_edges(sample.body_triangles)
    touching = edges[(edges[:, 0] == vertex) | (edges[:, 1] == vertex)]
    lengths = np.linalg.norm(sample.body_vertices[touching[:, 1]] - sample.body_vertices[touching[:, 0]], axis=1)
    return float(lengths.mean())


# MARK: - わざと壊した見本：体の面

def duplicate_vertex(sample: Sample) -> Sample:
    """重なった点：動かない頂点（左右の真ん中の面の上）のすぐそば（MERGE_DISTANCE の半分、+Y の向き）に、
    どの面にも使われない点を1つ足す（ウェイトは元の頂点と同じ）。
    どの面にも使われない点は、とげの測りから除く（measure_sample）ので、ほかの測る物は変わらない"""
    a, _ = _middle_static_edge(sample)
    point = sample.body_vertices[a] + np.array([0.0, UNDER * thresholds.MERGE_DISTANCE, 0.0])
    return _append_body(sample, point, [a])


def split_seam(sample: Sample) -> Sample:
    """重なった点（UV の切れ目）：動かない頂点1つを、まったく同じ座標の2つに分け、まわりの面のうち後ろ（+Y）側の物を
    新しいほうへつなぎ替える（PMX から読んだモデルによくある形）。
    頂点の番号の上では切れ目ができるので、穴（holes）が赤になり、閉じていないので体積の変わり方（volume_loss）を
    測れず灰になる（重なった点をまとめる安全な直しのあとは、どちらも消える）"""
    a, _ = _middle_static_edge(sample)
    triangles = _writable(sample.body_triangles)
    vertices = sample.body_vertices
    fan = np.flatnonzero((triangles == a).any(axis=1))
    centers = vertices[triangles[fan]].mean(axis=1)
    back = fan[centers[:, 1] > vertices[a, 1]]
    new_index = len(vertices)
    triangles[back] = np.where(triangles[back] == a, new_index, triangles[back])
    return _append_body(sample, vertices[a], [a], triangles)


def degenerate_triangle(sample: Sample) -> Sample:
    """つぶれた三角形：左右の真ん中の、動かない辺 a–b のまん中に頂点 m を足し、片側の三角形を2つに分け、
    面積 0 の三角形 (b, m, a) で T 字のすき間をふさぐ（閉じた形のまま、向きもそろったまま）。
    向きを比べられない三角形ができるので、面の裏返り（flipped_faces）は灰になる（つぶれた三角形を直すと消える）"""
    a, b = _middle_static_edge(sample)
    triangles = _writable(sample.body_triangles)
    row, corner = _triangle_with_edge(triangles, a, b)
    c = int(triangles[row, (corner + 2) % 3])
    m = len(sample.body_vertices)
    triangles[row] = [a, m, c]
    triangles = np.vstack([triangles, [[m, b, c], [b, m, a]]])
    midpoint = (sample.body_vertices[a] + sample.body_vertices[b]) / 2.0
    return _append_body(sample, midpoint, [a], triangles)


def non_manifold_fin(sample: Sample) -> Sample:
    """3つ以上の面が使う辺：左右の真ん中の、動かない辺 a–b に、体の中へ向かう両面のひれ（三角形 (a, b, f) と (b, a, f)）を
    付ける。辺 a–b を4つの面が使う。ひれは閉じていて（どの辺も行きと帰りがある）体積が 0、左右の真ん中の面の上にあり、
    体の中にあるので、ほかの測る物は変わらない"""
    a, b = _middle_static_edge(sample)
    vertices = sample.body_vertices
    midpoint = (vertices[a] + vertices[b]) / 2.0
    inward = -_unit(_normals(sample)[a] + _normals(sample)[b])
    tip = midpoint + inward * (UNDER * np.linalg.norm(vertices[b] - vertices[a]))
    tip[0] = 0.0
    f = len(vertices)
    triangles = np.vstack([sample.body_triangles, [[a, b, f], [b, a, f]]])
    return _append_body(sample, tip, [a], triangles)


def flipped_triangle(sample: Sample) -> Sample:
    """向きの合わない辺：動かない三角形1つの向きを裏返す（頂点の順を入れ替える）。まわりの3本の辺で向きが合わない。
    閉じた形ではなくなる（辺の行きと帰りがそろわない）ので、体積の変わり方（volume_loss）を測れず灰になる"""
    static = _static_vertices(sample)
    triangles = _writable(sample.body_triangles)
    row = int(np.flatnonzero(static[triangles].all(axis=1))[0])
    triangles[row, 1], triangles[row, 2] = sample.body_triangles[row, 2], sample.body_triangles[row, 1]
    return replace(sample, body_triangles=triangles)


def hole(sample: Sample) -> Sample:
    """穴：動かない三角形1つを消す。閉じた形ではなくなるので、体積の変わり方（volume_loss）を測れず灰になる"""
    static = _static_vertices(sample)
    row = int(np.flatnonzero(static[sample.body_triangles].all(axis=1))[0])
    return replace(sample, body_triangles=np.delete(sample.body_triangles, row, axis=0))


def loose_part(sample: Sample) -> Sample:
    """小さな離れた部品：体の中（下半身のあたり、左右の真ん中）に、閉じた小さな四面体（三角形 4 つ）を足す。
    左右がそろい、動かない骨に付けるので、ほかの測る物は変わらない（体積は四面体の分だけ増えるが、信号は変わらない）"""
    size = 0.02
    center = np.array([0.0, 0.0, LEG_HEIGHT + LEG_BAND])
    corners = center + np.array([[0.0, -size, -size], [0.0, size, -size], [-size, 0.0, size], [size, 0.0, size]])
    faces = []
    for face in ((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)):
        a, b, c = (corners[k] for k in face)
        outward = np.dot(np.cross(b - a, c - a), (a + b + c) / 3.0 - center) > 0.0
        faces.append(face if outward else (face[0], face[2], face[1]))
    start = len(sample.body_vertices)
    triangles = np.vstack([sample.body_triangles, np.asarray(faces, dtype=np.int64) + start])
    anchor = _front_static(sample, 1)
    return _append_body(sample, corners, np.repeat(anchor, 4), triangles)


def asymmetric_vertex(sample: Sample) -> Sample:
    """左右のずれ：動かない頂点（キャラの左にいちばん出ている物）を、面に沿って ASYMMETRY_RED の2倍だけ上へずらす。
    服の点も同じだけ動かす（服は体の殻）ので、すき間・めり込みは変わらない。前の版にも同じずれを入れる"""
    vertex = _side_static(sample)
    normal = _normals(sample)[vertex]
    up = np.array([0.0, 0.0, 1.0])
    tangent = _unit(up - np.dot(up, normal) * normal)
    return _move_body(sample, np.array([vertex]), tangent[None] * (OVER * thresholds.ASYMMETRY_RED), with_cloth=True)


# MARK: - わざと壊した見本：服

def cloth_inside(sample: Sample) -> Sample:
    """めり込み：服の点 3 つ（前の、動かない所）を、体の面から PENETRATION_RED の2倍だけ中へ入れる。
    体の中の点は、すき間が 0（近すぎる）なので、すき間（cloth_gap）も黄になる"""
    index = _front_static(sample, 3)
    cloth = _writable(sample.cloth_vertices)
    cloth[index] = sample.body_vertices[index] - _normals(sample)[index] * (OVER * thresholds.PENETRATION_RED)
    return replace(sample, cloth_vertices=cloth)


def cloth_too_close(sample: Sample) -> Sample:
    """すき間（近すぎる）：服の点 3 つを、体の外の CLOTH_GAP_MIN の半分の所へ寄せる。体の外なので、めり込みは 0 のまま"""
    index = _front_static(sample, 3)
    cloth = _writable(sample.cloth_vertices)
    cloth[index] = sample.body_vertices[index] + _normals(sample)[index] * (UNDER * thresholds.CLOTH_GAP_MIN)
    return replace(sample, cloth_vertices=cloth)


def cloth_too_far(sample: Sample) -> Sample:
    """すき間（離れすぎ）：服の点 3 つを、体の外の CLOTH_GAP_MAX の2倍の所へ離す"""
    index = _front_static(sample, 3)
    cloth = _writable(sample.cloth_vertices)
    cloth[index] = sample.body_vertices[index] + _normals(sample)[index] * (OVER * thresholds.CLOTH_GAP_MAX)
    return replace(sample, cloth_vertices=cloth)


# MARK: - わざと壊した見本：ウェイト（どれも、ポーズで動かない頂点を、ポーズで動かない骨だけを使って壊す。
# skin は正の重みだけを合計で割って使い、付いていない点は動かさないので、ポーズを付けた形は変わらない）

def unweighted(sample: Sample) -> Sample:
    """どの骨にも付いていない点：動かない頂点 3 つのうち、2 つは重みを全部消し、1 つは重みを全部、形を動かさない骨
    （左足ＩＫ）へ移す（形を動かす骨だけを数えるので、これも「付いていない」）。
    Blender と同じく、付いていない点は休みの位置のままなので、ポーズを付けた形は変わらない"""
    index = _front_static(sample, 3)
    ids, values = _writable(sample.bone_ids), _writable(sample.bone_weights)
    for row in index[:2]:
        _set_weight_row(ids, values, int(row), [])
    _set_weight_row(ids, values, int(index[2]), [(LEFT_IK, 1.0)])
    return replace(sample, bone_ids=ids, bone_weights=values)


def negative_weight(sample: Sample) -> Sample:
    """負の重み：動かない頂点 1 つを、下半身 1.25・腰 -0.25 にする（合計は 1、効く骨は 1 本）"""
    row = int(_front_static(sample, 1)[0])
    ids, values = _writable(sample.bone_ids), _writable(sample.bone_weights)
    extra = 1.0 / (OVER * OVER)
    _set_weight_row(ids, values, row, [(LOWER, 1.0 + extra), (WAIST, -extra)])
    return replace(sample, bone_ids=ids, bone_weights=values)


def not_normalized(sample: Sample) -> Sample:
    """合計が 1 でない重み：動かない頂点 3 つの重みを半分にする（合計 0.5）"""
    index = _front_static(sample, 3)
    values = _writable(sample.bone_weights)
    values[index] *= UNDER
    return replace(sample, bone_weights=values)


def over_limit(sample: Sample) -> Sample:
    """1点に効く骨が多すぎる：動かない頂点 1 つに、ポーズで動かない5本の骨を 0.2 ずつ付ける（枠を1つ増やす）"""
    row = int(_front_static(sample, 1)[0])
    count = thresholds.MAX_BONE_INFLUENCES + 1
    bones = [int(bone) for bone in np.flatnonzero(_static_bones(sample) & sample.deform_bones)][:count]
    if len(bones) < count:
        raise ValueError("ポーズで動かない、形を動かす骨が足りません（見本の作り方のまちがい）")
    ids = np.hstack([sample.bone_ids, np.full((len(sample.bone_ids), 1), -1, dtype=np.int64)])
    values = np.hstack([sample.bone_weights, np.zeros((len(sample.bone_weights), 1))])
    _set_weight_row(ids, values, row, [(bone, 1.0 / count) for bone in bones])
    return replace(sample, bone_ids=ids, bone_weights=values)


def duplicate_bone(sample: Sample) -> Sample:
    """同じ骨が2回入っている：動かない頂点 1 つを、下半身 0.5・下半身 0.5 にする（合計は 1、効く骨は 1 本）"""
    row = int(_front_static(sample, 1)[0])
    ids, values = _writable(sample.bone_ids), _writable(sample.bone_weights)
    _set_weight_row(ids, values, row, [(LOWER, UNDER), (LOWER, UNDER)])
    return replace(sample, bone_ids=ids, bone_weights=values)


# MARK: - わざと壊した見本：ポーズを付けた形（ポーズのときだけ効くシェイプキーが壊れた）

def stretched_pose(sample: Sample) -> Sample:
    """面の伸び：ポーズのときだけ効くシェイプキーが、体を縦（Z）に伸ばす（EDGE_STRETCH_RED より少し大きい割合）。
    縦に伸ばすだけなので、縮む辺・裏返り・とげは出ない（体積は増えるが、増えた分は信号にしない）"""
    factor = thresholds.EDGE_STRETCH_RED * (1.0 + UNDER * UNDER)
    offsets = np.zeros_like(sample.body_vertices)
    offsets[:, 2] = (factor - 1.0) * (sample.body_vertices[:, 2] - BODY_CENTER[2])
    return _with_pose_shape(sample, offsets)


def shrunk_pose(sample: Sample) -> Sample:
    """体積の減り：ポーズのときだけ効くシェイプキーが、体を中心へ縮める（全部の向きに同じ割合）。
    割合は、体積が VOLUME_LOSS_RED より多く減り、辺は EDGE_SHRINK_YELLOW より縮まない所（ポーズで少し縮む辺の分も見込む）"""
    largest = (1.0 - thresholds.VOLUME_LOSS_RED) ** (1.0 / 3.0)
    # ポーズ（上半身を曲げる）で、もともと縮む辺の割合の見込み（clean_sample の作り方で決まる。test_self_check が確かめる）
    smallest = thresholds.EDGE_SHRINK_YELLOW / CLEAN_MIN_EDGE_RATIO
    factor = _between(smallest, largest, "体積の減り")
    return _with_pose_shape(sample, (factor - 1.0) * (sample.body_vertices - BODY_CENTER))


def _push_depth_limit(sample: Sample, vertex: int, ratio: float) -> float:
    """頂点を体の中へ（法線の逆へ）d だけ押すとき、まわりの辺がどれも ratio 倍より伸びない、いちばん大きい d。

    辺 e = (まわりの点 − 頂点)、長さ l、c = 法線・e のとき、押したあとの長さの2乗は l² + 2cd + d²。
    これが (ratio × l)² になる d（正の答え）を辺ごとに求め、いちばん小さい物を返す。
    """
    edges = deformation.unique_edges(sample.body_triangles)
    touching = edges[(edges[:, 0] == vertex) | (edges[:, 1] == vertex)]
    others = np.where(touching[:, 0] == vertex, touching[:, 1], touching[:, 0])
    vectors = sample.body_vertices[others] - sample.body_vertices[vertex]
    lengths2 = (vectors * vectors).sum(axis=1)
    along = vectors @ _normals(sample)[vertex]
    return float((-along + np.sqrt(along * along + lengths2 * (ratio * ratio - 1.0))).min())


def spiked_pose(sample: Sample) -> Sample:
    """とげ：ポーズのときだけ効くシェイプキーが、動かない頂点 1 つを体の中へ押し込む。押す長さは、とげの強さ
    （押した長さ ÷ まわりの辺の長さの平均）が SPIKE_YELLOW 以上 SPIKE_RED 未満で、まわりの辺の伸びが
    EDGE_STRETCH_YELLOW 未満になる範囲のまん中（とげは黄。面の伸びは変わらない。外へ引くと、辺が伸びすぎる）"""
    vertex = _side_static(sample)
    mean_length = _mean_edge_length(sample, vertex)
    deepest = min(thresholds.SPIKE_RED * mean_length,
                  _push_depth_limit(sample, vertex, thresholds.EDGE_STRETCH_YELLOW))
    depth = _between(thresholds.SPIKE_YELLOW * mean_length, deepest, "とげ")
    offsets = np.zeros_like(sample.body_vertices)
    offsets[vertex] = -_normals(sample)[vertex] * depth
    return _with_pose_shape(sample, offsets)


def folded_pose(sample: Sample) -> Sample:
    """面の裏返り：ポーズのときだけ効くシェイプキーが、動かない頂点 1 つを、面に沿って、まわりの三角形1つの向かいの辺の
    まん中を半分だけ越えた所まで動かす（面が折れ重なる）。折れるときは、まわりの辺が大きく伸び縮みし、頂点がまわりから
    大きくずれるので、面の伸び（edge_stretch）ととげ（spike）も赤になる（1つの頂点で面を折ると、必ずそうなる）"""
    vertex = _side_static(sample)
    triangles = sample.body_triangles
    fan = triangles[(triangles == vertex).any(axis=1)][0]
    others = [int(k) for k in fan if k != vertex]
    target = sample.body_vertices[others].mean(axis=0)
    offsets = np.zeros_like(sample.body_vertices)
    offsets[vertex] = (1.0 + UNDER) * (target - sample.body_vertices[vertex])
    return _with_pose_shape(sample, offsets)


# MARK: - わざと壊した見本：版の比べ

def unexpected_change(sample: Sample) -> Sample:
    """変えてよい所の外の変化：前の版の、変えてはいけない頂点 1 つを、再現性の幅の2倍だけずらす
    （今の版の形は変えないので、ほかの測る物は変わらない）"""
    row = int(_front_static(sample, 1)[0])
    previous = _writable(sample.previous_body_vertices)
    previous[row, 2] += OVER * thresholds.REPRO_TOLERANCE_LENGTH
    return replace(sample, previous_body_vertices=previous)


# MARK: - わざと壊した見本：動き（左脚だけを壊す。ほかの並びは変えない）

def _replace_track(sample: Sample, name: str, side: int, frames: slice, values: np.ndarray) -> Sample:
    track = _writable(getattr(sample, name))
    track[frames, side] = values
    return replace(sample, **{name: track})


def reverse_knee(sample: Sample) -> Sample:
    """ひざの逆曲がり：左脚の、足を上げているフレーム（いちばん曲がる所）5 つで、ひざを股関節と足首を結ぶ線の
    反対側（後ろ）へ映す（IK がひざを逆に解いた形。股関節と足首の位置は変えない）"""
    frames = slice(SWING_FRAMES + SWING_FRAMES // 2 - 2, SWING_FRAMES + SWING_FRAMES // 2 + 3)
    hip, knee, ankle = (getattr(sample, name)[frames, 0] for name in ("hips", "knees", "ankles"))
    along = _unit(ankle - hip)
    offset = knee - hip
    projected = (offset * along).sum(axis=1, keepdims=True) * along
    mirrored = hip + projected - (offset - projected)
    return _replace_track(sample, "knees", 0, frames, mirrored)


def sunk_foot(sample: Sample) -> Sample:
    """床への沈み：左足の裏を、床に着いているフレーム 10 個で、床より FLOOR_SINK_TOLERANCE の2倍だけ下げる。
    下げる・戻すときの速さは FOOT_CONTACT_SPEED より遅いので、床に着いている区間（足すべり）は変わらない"""
    frames = slice(10, 20)
    soles = _writable(sample.soles)
    soles[frames, 0, 2] = sample.floor_z - OVER * thresholds.FLOOR_SINK_TOLERANCE
    return replace(sample, soles=soles)


def sliding_foot(sample: Sample) -> Sample:
    """足すべり：左足の裏が、床に着いている間（前半）に、ゆっくり後ろ（+Y）へ FOOT_SLIDE_TOLERANCE の2倍だけ滑る。
    速さは FOOT_CONTACT_SPEED よりずっと遅いので、床に着いたまま（高さも変えないので、床への沈みは変わらない）"""
    frames = slice(0, SWING_FRAMES)
    soles = _writable(sample.soles)
    soles[frames, 0, 1] += np.linspace(0.0, OVER * thresholds.FOOT_SLIDE_TOLERANCE, SWING_FRAMES)
    return replace(sample, soles=soles)


def unreachable_target(sample: Sample) -> Sample:
    """足が届かない：左足の IK の目標を、フレーム 3 つで、股関節から脚の長さの (1 + 10 × IK_REACH_TOLERANCE) 倍の
    所（真下）へ動かす（骨の位置は変えない。IK が届かずに止まった形）"""
    frames = slice(5, 8)
    reach = sample.leg_lengths[0] * (1.0 + 10.0 * thresholds.IK_REACH_TOLERANCE)
    target = sample.hips[frames, 0] - np.array([0.0, 0.0, reach])
    return _replace_track(sample, "ik_targets", 0, frames, target)


def jumping_bone(sample: Sample) -> Sample:
    """骨の跳び：左足（太もも）の骨の回転を、フレーム 1 つだけ、Z のまわりに BONE_JUMP_DEG の3倍だけ余分に回す
    （その前後の2つのフレームの間で跳ぶ）。骨の位置の並びは変えない"""
    frame = 20
    rotations = _writable(sample.leg_rotations)
    angle = math.radians(3.0 * thresholds.BONE_JUMP_DEG)
    extra = np.array([math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)])
    rotations[frame, 0] = _quaternion_multiply(extra, rotations[frame, 0])
    return replace(sample, leg_rotations=rotations)


# MARK: - 測れない見本（灰）

def sideways_leg(sample: Sample) -> Sample:
    """ひざの軸が決まらない：左脚を、フレーム 3 つで真横（+X）へ上げ、ひざはふつうに曲げる。太ももがキャラの左右の向きに
    近すぎて、位置だけではひざの軸が決まらないので、逆曲がりは灰（赤にしない。CLAUDE.md 11章）"""
    frames = slice(10, 13)
    hip = sample.hips[frames, 0]
    tilt = math.radians(UNDER * thresholds.KNEE_AXIS_MIN_ANGLE_DEG)
    knee = hip + THIGH_LENGTH * np.array([math.cos(tilt), 0.0, -math.sin(tilt)])
    ankle = knee + SHIN_LENGTH * np.array([0.0, math.sin(math.radians(45.0)), -math.cos(math.radians(45.0))])
    return _replace_track(_replace_track(sample, "knees", 0, frames, knee), "ankles", 0, frames, ankle)


def foot_in_air(sample: Sample) -> Sample:
    """足が一度も床に着かない：左足の裏を、全部のフレームで FOOT_CONTACT_HEIGHT の2倍だけ上げる（跳んでいる動き）。
    床に着いている区間が無いので、足すべりは測れず灰（すべっていないとは言えない）"""
    soles = _writable(sample.soles)
    soles[:, 0, 2] += OVER * thresholds.FOOT_CONTACT_HEIGHT
    return replace(sample, soles=soles)


# MARK: - いまの測り方では見つからない壊れ方（KNOWN_MISSES）

def skating_foot(sample: Sample) -> Sample:
    """速い足すべり：左足の裏が、床の高さのまま（z = 0）、前半の後ろの半分のフレームで、FOOT_CONTACT_SPEED の4倍の速さで
    後ろ（+Y）へ滑る（そのあとの足の上げ下ろしも、滑った所から続ける）。
    いまの foot_contacts は「低くて、遅い」フレームだけを床に着いているとみなすので、速く滑る足は床に着いていないことになり、
    足すべり（foot_slide）として数えられない（緑のまま）。足の裏の並びのほかは変えない"""
    frames = np.arange(len(sample.soles))
    start = SWING_FRAMES // 2
    soles = _writable(sample.soles)
    step = OVER * OVER * thresholds.FOOT_CONTACT_SPEED / sample.fps
    soles[:, 0, 1] += step * np.clip(frames - (start - 1), 0, SWING_FRAMES - start)
    return replace(sample, soles=soles)


def _ring(triangles: np.ndarray, vertex: int) -> list[int]:
    """頂点のまわりの点を、まわりの三角形の向き（外から見て反時計回り）の順に並べる（閉じた扇の形のとき）"""
    following: dict[int, int] = {}
    for row in triangles[(triangles == vertex).any(axis=1)]:
        corner = list(row).index(vertex)
        following[int(row[(corner + 1) % 3])] = int(row[(corner + 2) % 3])
    ring = [min(following)]
    while len(ring) < len(following):
        ring.append(following[ring[-1]])
    return ring


def coarse_cloth(sample: Sample) -> Sample:
    """服の面を突き抜ける体：服の点のうち1つ（動かない所の、いちばん前）を消し、まわりの点を結んだ三角形で穴をふさぐ
    （その所だけ、服の面が粗くなる。閉じた殻で、向きもそろったまま）。粗くなった服の面は体の近くを平らに通るので、
    その下の体の頂点が、服の外へ PENETRATION_RED より深く出る。
    いまのめり込み（penetration_depth）は、服の頂点が体の中にあるかだけを調べるので、服の面のまん中を体が突き抜けても
    見つからない（緑のまま）。服の点はどれも体から CLOTH_GAP_MIN〜CLOTH_GAP_MAX の所にあるので、すき間も緑のまま"""
    vertex = int(_front_static(sample, 1)[0])
    triangles = sample.cloth_triangles
    ring = _ring(triangles, vertex)
    kept = triangles[~(triangles == vertex).any(axis=1)]
    patch = np.array([[ring[0], ring[k], ring[k + 1]] for k in range(1, len(ring) - 1)], dtype=np.int64)
    joined = np.vstack([kept, patch])
    return replace(sample, cloth_vertices=np.delete(sample.cloth_vertices, vertex, axis=0),
                   cloth_triangles=np.where(joined > vertex, joined - 1, joined))


# MARK: - 何も変えない書き換え

def flip_quaternion_signs(sample: Sample) -> Sample:
    """1つおきのフレームで、脚の骨の回転の符号を変える（q と -q は同じ向きなので、跳びではない）"""
    rotations = _writable(sample.leg_rotations)
    rotations[1::2] *= -1.0
    return replace(sample, leg_rotations=rotations)


def rotate_triangle_corners(sample: Sample) -> Sample:
    """体と服の三角形の頂点の順を、ぐるりと1つずらす（(a, b, c) → (b, c, a)。向きは変わらない）"""
    return replace(sample, body_triangles=np.roll(sample.body_triangles, 1, axis=1),
                   cloth_triangles=np.roll(sample.cloth_triangles, 1, axis=1))


def reverse_triangle_order(sample: Sample) -> Sample:
    """体と服の三角形の並びの順を逆にする"""
    return replace(sample, body_triangles=sample.body_triangles[::-1], cloth_triangles=sample.cloth_triangles[::-1])


# MARK: - 一覧

@dataclass(frozen=True)
class Breakage:
    """わざと壊した見本1つ。

    apply：壊していない見本から、壊した写しを作る関数（もとの見本は変えない）。説明（docstring）に、何を壊したかと、
        also の測る物が一緒に変わる理由を書く。
    finds：見つけるはずの測る物（Measurement の名前）。どれも expected の信号になり、値は 0 より大きい。
    expected：finds の信号（赤か黄。GRAY_CASES では灰）。
    also：一緒に変わってよい測る物（名前 → その信号）。ここに無い測る物の信号は、壊していない見本と同じでなければならない。
    """

    apply: Callable[[Sample], Sample]
    finds: tuple[str, ...]
    expected: Signal
    also: Mapping[str, Signal] = field(default_factory=dict)


BREAKAGES: dict[str, Breakage] = {
    # 体の面
    "duplicate_vertex": Breakage(duplicate_vertex, (mesh_check.DUPLICATES_NAME,), Signal.YELLOW),
    "split_seam": Breakage(split_seam, (mesh_check.DUPLICATES_NAME,), Signal.YELLOW,
                           also={mesh_check.HOLES_NAME: Signal.RED, deformation.VOLUME_NAME: Signal.GRAY}),
    "degenerate_triangle": Breakage(degenerate_triangle, (mesh_check.DEGENERATE_NAME,), Signal.YELLOW,
                                    also={deformation.FLIPS_NAME: Signal.GRAY}),
    "non_manifold_fin": Breakage(non_manifold_fin, (mesh_check.NON_MANIFOLD_NAME,), Signal.YELLOW),
    "flipped_triangle": Breakage(flipped_triangle, (mesh_check.INCONSISTENT_NAME,), Signal.RED,
                                 also={deformation.VOLUME_NAME: Signal.GRAY}),
    "hole": Breakage(hole, (mesh_check.HOLES_NAME,), Signal.RED, also={deformation.VOLUME_NAME: Signal.GRAY}),
    "loose_part": Breakage(loose_part, (mesh_check.LOOSE_PARTS_NAME,), Signal.YELLOW),
    "asymmetric_vertex": Breakage(asymmetric_vertex, (symmetry.ASYMMETRY_NAME,), Signal.RED),
    # 服
    "cloth_inside": Breakage(cloth_inside, (geometry.PENETRATION_NAME,), Signal.RED,
                             also={geometry.GAP_NAME: Signal.YELLOW}),
    "cloth_too_close": Breakage(cloth_too_close, (geometry.GAP_NAME,), Signal.YELLOW),
    "cloth_too_far": Breakage(cloth_too_far, (geometry.GAP_NAME,), Signal.YELLOW),
    # ウェイト
    "unweighted": Breakage(unweighted, (weights.UNWEIGHTED_NAME,), Signal.RED),
    "negative_weight": Breakage(negative_weight, (weights.NEGATIVE_NAME,), Signal.RED),
    "not_normalized": Breakage(not_normalized, (weights.NOT_NORMALIZED_NAME,), Signal.YELLOW),
    "over_limit": Breakage(over_limit, (weights.OVER_LIMIT_NAME,), Signal.RED),
    "duplicate_bone": Breakage(duplicate_bone, (weights.DUPLICATE_NAME,), Signal.YELLOW),
    # ポーズを付けた形
    "stretched_pose": Breakage(stretched_pose, (deformation.STRETCH_NAME,), Signal.RED),
    "shrunk_pose": Breakage(shrunk_pose, (deformation.VOLUME_NAME,), Signal.RED),
    "spiked_pose": Breakage(spiked_pose, (deformation.SPIKES_NAME,), Signal.YELLOW),
    "folded_pose": Breakage(folded_pose, (deformation.FLIPS_NAME,), Signal.RED,
                            also={deformation.STRETCH_NAME: Signal.RED, deformation.SPIKES_NAME: Signal.RED}),
    # 版の比べ
    "unexpected_change": Breakage(unexpected_change, (deformation.CHANGES_NAME,), Signal.RED),
    # 動き
    "reverse_knee": Breakage(reverse_knee, (kinematics.REVERSE_BEND_NAME,), Signal.RED),
    "sunk_foot": Breakage(sunk_foot, (kinematics.FLOOR_SINK_NAME,), Signal.RED),
    "sliding_foot": Breakage(sliding_foot, (kinematics.FOOT_SLIDE_NAME,), Signal.RED),
    "unreachable_target": Breakage(unreachable_target, (kinematics.UNREACHABLE_NAME,), Signal.YELLOW),
    "jumping_bone": Breakage(jumping_bone, (kinematics.BONE_JUMPS_NAME,), Signal.RED),
}

GRAY_CASES: dict[str, Breakage] = {
    "sideways_leg": Breakage(sideways_leg, (kinematics.REVERSE_BEND_NAME,), Signal.GRAY),
    "foot_in_air": Breakage(foot_in_air, (kinematics.FOOT_SLIDE_NAME,), Signal.GRAY),
}

# いまの測り方では見つからない壊れ方。finds が expected になるのが正しいが、いまはならない
# （test_self_check の KnownMissTest で、unittest.expectedFailure にしている。直したら BREAKAGES へ移す）
KNOWN_MISSES: dict[str, Breakage] = {
    "skating_foot": Breakage(skating_foot, (kinematics.FOOT_SLIDE_NAME,), Signal.RED),
    "coarse_cloth": Breakage(coarse_cloth, (geometry.PENETRATION_NAME,), Signal.RED),
}

HARMLESS: dict[str, Callable[[Sample], Sample]] = {
    "flip_quaternion_signs": flip_quaternion_signs,
    "rotate_triangle_corners": rotate_triangle_corners,
    "reverse_triangle_order": reverse_triangle_order,
}

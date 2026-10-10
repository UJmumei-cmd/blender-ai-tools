"""テスト用の、MMD に似た人型のモデルを作る（Blender の中で動かす）。

いまのシーンに、次の物を作る。乱数は使わない（何度作っても、まったく同じ数字になる）。

- アーマチュア "TestArmature"：MMD 準標準の骨（bones.canonical_bones() の 96 本）を、日本語の名前のまま全部持つ。
- メッシュ "Body"：閉じた部品（胴・首・頭・鼻・目・肩・腕・手・指・もも・すね・足）を集めた、低い解像度の人型。
  どの部品も閉じていて、面は外を向く。アーマチュアの子で、Armature モディファイアと骨と同じ名前の頂点グループを持つ。
- メッシュ "Skirt"（with_cloth のとき）：腰のまわりの、開いた筒のスカート。体より少し外にある。

向きと大きさ（CLAUDE.md 7章）：Z が上、モデルは -Y を向く、キャラの左は +X。足の裏は z=0、頭のてっぺんは z=height。
姿勢は MMD の A ポーズ（上腕は水平から ARM_DOWN_DEG 度下がる）。

ウェイト（本物の準標準のモデルに似せる）：
- 部品の骨に固く付ける。ひじ（左腕／左ひじ）とひざ（左足D／左ひざD）の近くは、2本の骨で混ぜる。
- 脚のウェイトは D の骨（足D・ひざD・足首D・足先EX）に乗せる。足・ひざ・足首（IK と VMD の骨）には乗せない。
- 前腕の頂点の一部は 左手捩 に乗せる。上腕の一部は 左腕捩 に乗せる。
- five_influences のとき、前腕の頂点のうち片腕2つに、5本の骨を付ける。5本目の重みは小さい（SMALL_FIFTH_WEIGHT）。
  1点4本までにすると、形が少しだけ変わる（その差を測れるように）。
- 頂点グループは名前の順に作る。骨の並びの順とわざと違えて、番号のまちがいを見つけられるようにする。

骨の付与：足D・ひざD・足首D は、足・ひざ・足首 の回転を写す（Copy Rotation、ローカルどうし。MMD の回転付与、割合 1）。
足 を回すと、見える脚が曲がる。IK の拘束は付けない（足ＩＫ などは、名前と位置だけ）。

使い方：
    # Blender の中で（テストなど）
    bpy.ops.wm.read_homefile(use_empty=True)
    info = build_model()
    # コマンドで .blend に保存する
    blender --background --python tests/fixtures/mmd_model.py -- <out.blend> [--no-cloth] [--no-shape-keys]
        [--four-influences] [--height 1.45]
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import bpy
import numpy as np

# MARK: - 決まり

# 形を決めるときの、もとの背の高さ（m）。ほかの高さは、この形を同じ割合で大きく・小さくする
REFERENCE_HEIGHT = 1.45
# A ポーズの上腕の下がり角（度）。convert_pmx.rest_arm_angle（左腕 → 左ひじ の向き）で測ると、この値になる
ARM_DOWN_DEG = 37.5
# 5本目の骨の、小さい重み
SMALL_FIFTH_WEIGHT = 0.02

ARMATURE_NAME = "TestArmature"
BODY_NAME = "Body"
SKIRT_NAME = "Skirt"
SHAPE_KEY_NAMES = ("Basis", "まばたき", "あ")
# Armature モディファイアの名前
MODIFIER_NAME = "Armature"
# 回転付与の拘束の名前
COPY_ROTATION_NAME = "回転付与"

# 回転付与：D の骨 → 回転を写す元の骨（左。右も同じ）
ROTATION_SOURCES = {"左足D": "左足", "左ひざD": "左ひざ", "左足首D": "左足首"}

# Body の部品（左右のある部品は、左・右の順）。どれも閉じた形
BODY_PARTS = (
    "torso", "neck", "head", "nose", "eye_L", "eye_R",
    "shoulder_L", "upper_arm_L", "forearm_L", "hand_L",
    "thumb_L", "index_L", "middle_L", "ring_L", "little_L",
    "shoulder_R", "upper_arm_R", "forearm_R", "hand_R",
    "thumb_R", "index_R", "middle_R", "ring_R", "little_R",
    "thigh_L", "shin_L", "foot_L", "thigh_R", "shin_R", "foot_R",
)


def mirror_name(name: str) -> str:
    """左の名前を右の名前にする（左腕 → 右腕、腰キャンセル左 → 腰キャンセル右、hand_L → hand_R）。体の中心はそのまま"""
    if name.endswith("_L"):
        return name[:-2] + "_R"
    return name.replace("左", "右")


# MARK: - 骨の並び


@dataclass(frozen=True)
class BoneSpec:
    """骨1本。位置は REFERENCE_HEIGHT の大きさのとき（m）"""

    name: str
    parent: str | None
    head: tuple[float, float, float]
    tail: tuple[float, float, float]
    deform: bool


def _along(start: Sequence[float], direction: np.ndarray, length: float) -> tuple[float, float, float]:
    point = np.asarray(start, dtype=np.float64) + direction * length
    return float(point[0]), float(point[1]), float(point[2])


def _arm_direction() -> np.ndarray:
    """左腕の向き（長さ 1）。+X へ伸び、水平から ARM_DOWN_DEG 度下がる"""
    angle = math.radians(ARM_DOWN_DEG)
    return np.array([math.cos(angle), 0.0, -math.sin(angle)])


# 腕の長さ（m）
UPPER_ARM_LENGTH = 0.24
FOREARM_LENGTH = 0.21
# 左腕の付け根（左腕 の head）
SHOULDER_JOINT = (0.15, 0.0, 1.13)
# 脚の関節（左）
HIP_JOINT = (0.08, 0.0, 0.74)
KNEE_JOINT = (0.08, -0.012, 0.40)
ANKLE_JOINT = (0.08, 0.0, 0.075)
TOE_POINT = (0.08, -0.11, 0.01)
BALL_POINT = (0.08, -0.07, 0.015)

# 指：（名前の前半, 手首からの前への距離, 横（-Y が前・親指の側）, 3つの節の長さ）
_FINGERS = (
    ("人指", 0.080, -0.022, (0.025, 0.018, 0.016)),
    ("中指", 0.085, -0.007, (0.027, 0.020, 0.017)),
    ("薬指", 0.080, 0.008, (0.025, 0.018, 0.016)),
    ("小指", 0.075, 0.022, (0.020, 0.015, 0.014)),
)
# 指の節の数字（全角。辞書の字のまま）
_DIGITS = "０１２３"


def _joints() -> dict[str, tuple[float, float, float]]:
    """腕・手・指の関節の位置（左。REFERENCE_HEIGHT の大きさ）"""
    d = _arm_direction()
    shoulder = SHOULDER_JOINT
    elbow = _along(shoulder, d, UPPER_ARM_LENGTH)
    wrist = _along(elbow, d, FOREARM_LENGTH)
    return {"shoulder": shoulder, "elbow": elbow, "wrist": wrist}


def _thumb_direction() -> np.ndarray:
    """親指の向き（長さ 1）。腕の向きから、前（-Y）へ開く"""
    v = _arm_direction() + np.array([0.0, -0.7, 0.0])
    return v / np.linalg.norm(v)


def _finger_chains() -> list[tuple[str, list[tuple[float, float, float]]]]:
    """指ごとの（指の名前の前半, 関節の位置 4つ：付け根・2つの節目・先）（左）"""
    d = _arm_direction()
    wrist = _joints()["wrist"]
    chains = []
    thumb_dir = _thumb_direction()
    start = np.asarray(_along(wrist, d, 0.015)) + np.array([0.0, -0.02, -0.005])
    points = [start]
    for length in (0.025, 0.025, 0.02):
        points.append(points[-1] + thumb_dir * length)
    chains.append(("親指", [tuple(float(c) for c in p) for p in points]))
    for prefix, forward, side, lengths in _FINGERS:
        start = np.asarray(_along(wrist, d, forward)) + np.array([0.0, side, 0.0])
        points = [start]
        for length in lengths:
            points.append(points[-1] + d * length)
        chains.append((prefix, [tuple(float(c) for c in p) for p in points]))
    return chains


def _center_bones() -> list[BoneSpec]:
    return [
        BoneSpec("全ての親", None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.10), False),
        BoneSpec("センター", "全ての親", (0.0, 0.0, 0.62), (0.0, 0.0, 0.67), False),
        BoneSpec("グルーブ", "センター", (0.0, 0.0, 0.66), (0.0, 0.0, 0.71), False),
        BoneSpec("腰", "グルーブ", (0.0, 0.02, 0.80), (0.0, 0.0, 0.86), False),
        BoneSpec("上半身", "腰", (0.0, 0.0, 0.86), (0.0, 0.0, 0.98), True),
        BoneSpec("上半身2", "上半身", (0.0, 0.0, 0.98), (0.0, 0.0, 1.15), True),
        BoneSpec("首", "上半身2", (0.0, 0.0, 1.17), (0.0, 0.0, 1.25), True),
        BoneSpec("頭", "首", (0.0, 0.0, 1.25), (0.0, 0.0, 1.38), True),
        BoneSpec("両目", "頭", (0.0, -0.12, 1.40), (0.0, -0.12, 1.43), False),
        BoneSpec("下半身", "腰", (0.0, 0.0, 0.86), (0.0, 0.0, 0.74), True),
    ]


def _left_bones() -> list[BoneSpec]:
    d = _arm_direction()
    j = _joints()
    shoulder, elbow, wrist = j["shoulder"], j["elbow"], j["wrist"]
    bones = [
        BoneSpec("左目", "頭", (0.032, -0.07, 1.335), (0.032, -0.07, 1.355), True),
        # 腕
        BoneSpec("左肩P", "上半身2", (0.025, 0.0, 1.15), (0.025, 0.0, 1.18), False),
        BoneSpec("左肩", "左肩P", (0.025, 0.0, 1.15), shoulder, True),
        BoneSpec("左肩C", "左肩", shoulder, _along(shoulder, d, 0.02), False),
        BoneSpec("左腕", "左肩", shoulder, elbow, True),
        BoneSpec("左腕捩", "左腕", _along(shoulder, d, 0.55 * UPPER_ARM_LENGTH),
                 _along(shoulder, d, 0.75 * UPPER_ARM_LENGTH), True),
    ]
    for number, fraction in enumerate((0.25, 0.5, 0.75), start=1):
        bones.append(BoneSpec(f"左腕捩{number}", "左腕", _along(shoulder, d, fraction * UPPER_ARM_LENGTH),
                              _along(shoulder, d, (fraction + 0.05) * UPPER_ARM_LENGTH), False))
    bones += [
        BoneSpec("左ひじ", "左腕捩", elbow, wrist, True),
        BoneSpec("左手捩", "左ひじ", _along(elbow, d, 0.55 * FOREARM_LENGTH),
                 _along(elbow, d, 0.75 * FOREARM_LENGTH), True),
    ]
    for number, fraction in enumerate((0.25, 0.5, 0.75), start=1):
        bones.append(BoneSpec(f"左手捩{number}", "左ひじ", _along(elbow, d, fraction * FOREARM_LENGTH),
                              _along(elbow, d, (fraction + 0.05) * FOREARM_LENGTH), False))
    bones += [
        BoneSpec("左手首", "左手捩", wrist, _along(wrist, d, 0.07), True),
        BoneSpec("左ダミー", "左手首", _along(wrist, d, 0.04), _along(wrist, d, 0.06), False),
    ]
    # 指（親指は ０〜２、ほかは １〜３）
    for prefix, points in _finger_chains():
        first = 0 if prefix == "親指" else 1
        parent = "左手首"
        for k in range(3):
            name = f"左{prefix}{_DIGITS[first + k]}"
            bones.append(BoneSpec(name, parent, points[k], points[k + 1], True))
            parent = name
    # 脚（見える骨：IK と VMD のための骨。形は動かさない）
    bones += [
        BoneSpec("腰キャンセル左", "下半身", HIP_JOINT, (0.08, 0.0, 0.70), False),
        BoneSpec("左足", "腰キャンセル左", HIP_JOINT, KNEE_JOINT, False),
        BoneSpec("左ひざ", "左足", KNEE_JOINT, ANKLE_JOINT, False),
        BoneSpec("左足首", "左ひざ", ANKLE_JOINT, TOE_POINT, False),
        BoneSpec("左つま先", "左足首", TOE_POINT, (0.08, -0.13, 0.01), False),
        # D の骨（形を動かす骨。足・ひざ・足首 の回転を写す）
        BoneSpec("左足D", "腰キャンセル左", HIP_JOINT, KNEE_JOINT, True),
        BoneSpec("左ひざD", "左足D", KNEE_JOINT, ANKLE_JOINT, True),
        BoneSpec("左足首D", "左ひざD", ANKLE_JOINT, TOE_POINT, True),
        BoneSpec("左足先EX", "左足首D", BALL_POINT, (0.08, -0.11, 0.012), True),
        # IK
        BoneSpec("左足IK親", "全ての親", (0.08, 0.0, 0.0), (0.08, 0.0, 0.05), False),
        BoneSpec("左足ＩＫ", "左足IK親", ANKLE_JOINT, (0.08, 0.10, 0.075), False),
        BoneSpec("左つま先ＩＫ", "左足ＩＫ", TOE_POINT, (0.08, -0.14, 0.01), False),
    ]
    return bones


def _mirror_bone(bone: BoneSpec) -> BoneSpec:
    def flip(p: tuple[float, float, float]) -> tuple[float, float, float]:
        return -p[0], p[1], p[2]

    parent = None if bone.parent is None else mirror_name(bone.parent)
    return BoneSpec(mirror_name(bone.name), parent, flip(bone.head), flip(bone.tail), bone.deform)


def bone_specs() -> list[BoneSpec]:
    """骨の一覧（体の中心、左、右の順。REFERENCE_HEIGHT の大きさ）。右は左を映して作る"""
    left = _left_bones()
    return _center_bones() + left + [_mirror_bone(bone) for bone in left]


def deform_bone_names() -> frozenset[str]:
    """形を動かす骨（use_deform が True）の名前"""
    return frozenset(bone.name for bone in bone_specs() if bone.deform)


def bone_parents() -> dict[str, str | None]:
    """骨の名前 → 親の骨の名前（根は None）"""
    return {bone.name: bone.parent for bone in bone_specs()}


# MARK: - メッシュを組み立てる

Weights = dict[str, float]


class _MeshBuilder:
    """頂点・面・ウェイト・部品を集める。位置は REFERENCE_HEIGHT の大きさ"""

    def __init__(self) -> None:
        self.vertices: list[np.ndarray] = []
        self.faces: list[tuple[int, ...]] = []
        self.weights: list[Weights] = []
        self.parts: dict[str, tuple[int, int]] = {}
        self._part_start: int | None = None

    def begin(self, part: str) -> None:
        if part in self.parts:
            raise ValueError(f"部品 {part} が2回あります")
        self._part_start = len(self.vertices)
        self._part_name = part

    def end(self) -> None:
        self.parts[self._part_name] = (self._part_start, len(self.vertices))
        self._part_start = None

    def vertex(self, point: Sequence[float], weights: Weights) -> int:
        cleaned = {bone: float(w) for bone, w in weights.items() if w > 0.0}
        if not cleaned:
            raise ValueError("ウェイトの無い頂点は作らない")
        self.vertices.append(np.asarray(point, dtype=np.float64))
        self.weights.append(cleaned)
        return len(self.vertices) - 1

    def tube(self, part: str, centers: Sequence[Sequence[float]], profiles: Sequence[Sequence[tuple[float, float]]],
             hint: Sequence[float], weights_at: Callable[[int, int, np.ndarray], Weights], *,
             start_pole: Sequence[float] | None = None, end_pole: Sequence[float] | None = None,
             start_weights: Weights | None = None, end_weights: Weights | None = None,
             axis: Sequence[float] | None = None) -> None:
        """輪をつないだ筒。輪 i の頂点 j = centers[i] + profiles[i][j] の (u, v)。

        軸 a は axis（無ければ、最初の中心から最後の中心への向き）。輪は a の向きに進むように並べる。
        u は hint を a に垂直にした向き、v = a × u（u × v = a の右手系）。
        profiles の点は (u, v) の平面で反時計回りに並べる。こうすると、側面は外を向く。
        start_pole・end_pole があれば、両端を三角形の扇でふさぐ（閉じた形）。無ければ開いた筒。
        """
        c = [np.asarray(p, dtype=np.float64) for p in centers]
        axis = c[-1] - c[0] if axis is None else np.asarray(axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        h = np.asarray(hint, dtype=np.float64)
        u = h - np.dot(h, axis) * axis
        u = u / np.linalg.norm(u)
        v = np.cross(axis, u)
        sides = len(profiles[0])
        if any(len(profile) != sides for profile in profiles) or len(profiles) != len(c):
            raise ValueError(f"{part}: 輪の数と点の数をそろえてください")
        self.begin(part)
        rings = []
        for i, (center, profile) in enumerate(zip(c, profiles)):
            ring = []
            for j, (pu, pv) in enumerate(profile):
                point = center + pu * u + pv * v
                ring.append(self.vertex(point, weights_at(i, j, point)))
            rings.append(ring)
        for i in range(len(rings) - 1):
            for j in range(sides):
                k = (j + 1) % sides
                self.faces.append((rings[i][j], rings[i][k], rings[i + 1][k], rings[i + 1][j]))
        if start_pole is not None:
            pole = self.vertex(start_pole, start_weights or {})
            for j in range(sides):
                self.faces.append((pole, rings[0][(j + 1) % sides], rings[0][j]))
        if end_pole is not None:
            pole = self.vertex(end_pole, end_weights or {})
            last = rings[-1]
            for j in range(sides):
                self.faces.append((pole, last[j], last[(j + 1) % sides]))
        self.end()

    def octahedron(self, part: str, center: Sequence[float], radius: float, weights: Weights) -> list[int]:
        """閉じた正八面体。頂点の並びは +x, -x, +y, -y, +z, -z。頂点の番号を返す"""
        c = np.asarray(center, dtype=np.float64)
        self.begin(part)
        index = []
        for axis in range(3):
            for sign in (1.0, -1.0):
                offset = np.zeros(3)
                offset[axis] = sign * radius
                index.append(self.vertex(c + offset, weights))
        for sx in (0, 1):
            for sy in (0, 1):
                for sz in (0, 1):
                    a, b, cc = index[sx], index[2 + sy], index[4 + sz]
                    # 負の向きが奇数個なら、回る向きを逆にする（どの面も外を向く）
                    self.faces.append((a, b, cc) if (sx + sy + sz) % 2 == 0 else (a, cc, b))
        self.end()
        return index

    def pyramid(self, part: str, base_center: Sequence[float], half: float, apex: Sequence[float],
                weights: Weights) -> int:
        """-Y を向いた、閉じた四角すい（底は y = base_center の y の、XZ 平面の正方形）。とがった頂点の番号を返す"""
        x, y, z = (float(value) for value in base_center)
        self.begin(part)
        base = [self.vertex((x - half, y, z - half), weights), self.vertex((x + half, y, z - half), weights),
                self.vertex((x + half, y, z + half), weights), self.vertex((x - half, y, z + half), weights)]
        tip = self.vertex(apex, weights)
        for k in range(4):
            self.faces.append((base[k], base[(k + 1) % 4], tip))
        self.faces.append((base[0], base[3], base[2], base[1]))
        self.end()
        return tip


def _ellipse(rx: float, ry: float, sides: int, phase: float = 0.0) -> list[tuple[float, float]]:
    """(u, v) の平面の楕円（反時計回り）"""
    return [(rx * math.cos(phase + 2.0 * math.pi * j / sides), ry * math.sin(phase + 2.0 * math.pi * j / sides))
            for j in range(sides)]


def _rectangle(half_u: float, half_v: float) -> list[tuple[float, float]]:
    """(u, v) の平面の長方形（反時計回り）"""
    return [(half_u, half_v), (-half_u, half_v), (-half_u, -half_v), (half_u, -half_v)]


def _side(weights: Weights, side: str) -> Weights:
    """左の骨の名前で書いたウェイトを、その側の名前にする"""
    return weights if side == "L" else {mirror_name(bone): w for bone, w in weights.items()}


def _flip(point: Sequence[float], side: str) -> np.ndarray:
    p = np.asarray(point, dtype=np.float64)
    return p if side == "L" else p * np.array([-1.0, 1.0, 1.0])


def _rings(weights: Sequence[Weights]) -> Callable[[int, int, np.ndarray], Weights]:
    return lambda i, j, point: weights[i]


# MARK: - Body の部品


def _build_trunk(builder: _MeshBuilder) -> dict[str, list[int]]:
    """胴・首・頭・鼻・目。シェイプキーで動かす頂点の番号を返す"""
    # 胴：(z, 横の半径, 前後の半径, ウェイト)
    torso = [
        (0.70, 0.100, 0.080, {"下半身": 1.0}),
        (0.76, 0.140, 0.095, {"下半身": 1.0}),
        (0.86, 0.115, 0.085, {"下半身": 0.5, "上半身": 0.5}),
        (0.98, 0.130, 0.095, {"上半身": 0.5, "上半身2": 0.5}),
        (1.10, 0.140, 0.090, {"上半身2": 1.0}),
        (1.17, 0.070, 0.060, {"上半身2": 0.7, "首": 0.3}),
    ]
    builder.tube("torso", [(0.0, 0.0, z) for z, *_ in torso], [_ellipse(rx, ry, 12) for _, rx, ry, _ in torso],
                 (1.0, 0.0, 0.0), _rings([w for *_, w in torso]),
                 start_pole=(0.0, 0.0, 0.69), end_pole=(0.0, 0.0, 1.18),
                 start_weights={"下半身": 1.0}, end_weights={"上半身2": 0.7, "首": 0.3})
    neck = [(1.15, 0.045, {"首": 0.7, "上半身2": 0.3}), (1.27, 0.042, {"首": 0.5, "頭": 0.5})]
    builder.tube("neck", [(0.0, 0.0, z) for z, *_ in neck], [_ellipse(r, r, 8) for _, r, _ in neck],
                 (1.0, 0.0, 0.0), _rings([w for *_, w in neck]),
                 start_pole=(0.0, 0.0, 1.15), end_pole=(0.0, 0.0, 1.27),
                 start_weights=neck[0][2], end_weights=neck[1][2])
    # 頭：中心 z=1.345 の楕円体。上の端（てっぺん）は z=1.45 ＝ REFERENCE_HEIGHT
    center_z, radius_z, radius_x, radius_y = 1.345, 0.105, 0.085, 0.095
    latitudes = (-60.0, -30.0, 0.0, 30.0, 60.0)
    centers = [(0.0, 0.0, center_z + radius_z * math.sin(math.radians(a))) for a in latitudes]
    profiles = [_ellipse(radius_x * math.cos(math.radians(a)), radius_y * math.cos(math.radians(a)), 12)
                for a in latitudes]
    start = len(builder.vertices)
    builder.tube("head", centers, profiles, (1.0, 0.0, 0.0), _rings([{"頭": 1.0}] * len(latitudes)),
                 start_pole=(0.0, 0.0, center_z - radius_z), end_pole=(0.0, 0.0, REFERENCE_HEIGHT),
                 start_weights={"頭": 1.0}, end_weights={"頭": 1.0})
    # 「あ」で下げる口の頂点：-30 度の輪の、前（-Y）の3つ（j = 8, 9, 10。j=9 が真ん前）
    mouth = [start + 12 * 1 + j for j in (8, 9, 10)]
    builder.pyramid("nose", (0.0, -0.085, 1.31), 0.012, (0.0, -0.115, 1.305), {"頭": 1.0})
    eyes = []
    for side in ("L", "R"):
        index = builder.octahedron(f"eye_{side}", _flip((0.032, -0.083, 1.335), side), 0.012,
                                   {"左目" if side == "L" else "右目": 1.0})
        eyes.append(index[4])  # 上（+z）の頂点
    return {"mouth": mouth, "eyelids": eyes}


def _build_arm(builder: _MeshBuilder, side: str, five_influences: bool) -> list[int]:
    """肩・上腕・前腕・手・指（side の側）。5本の骨を付けた頂点の番号を返す"""
    d = _arm_direction()
    j = _joints()
    shoulder, elbow, wrist = (np.asarray(j[key]) for key in ("shoulder", "elbow", "wrist"))
    flip = lambda p: _flip(p, side)  # noqa: E731
    hint = flip((0.0, 1.0, 0.0))  # 腕の輪の u：後ろ（+Y）

    # 肩：上半身2 から 左腕 の付け根まで
    builder.tube(f"shoulder_{side}", [flip((0.025, 0.0, 1.15)), flip(shoulder)],
                 [_ellipse(0.045, 0.045, 8)] * 2, hint,
                 _rings([_side({"上半身2": 0.4, "左肩": 0.6}, side), _side({"左肩": 0.5, "左腕": 0.5}, side)]),
                 start_pole=flip((0.0, 0.0, 1.15)), end_pole=flip(shoulder + d * 0.01),
                 start_weights=_side({"上半身2": 0.4, "左肩": 0.6}, side),
                 end_weights=_side({"左肩": 0.5, "左腕": 0.5}, side))
    # 上腕：ひじの輪は 左腕／左ひじ で混ぜる。途中の輪は 左腕捩 にも乗せる
    upper = [
        (-0.02, 0.040, {"左肩": 0.2, "左腕": 0.8}),
        (0.35, 0.038, {"左腕": 1.0}),
        (0.70, 0.035, {"左腕": 0.5, "左腕捩": 0.5}),
        (1.00, 0.033, {"左腕": 0.5, "左ひじ": 0.5}),
    ]
    builder.tube(f"upper_arm_{side}", [flip(shoulder + d * t * UPPER_ARM_LENGTH) for t, *_ in upper],
                 [_ellipse(r, r, 8) for _, r, _ in upper], hint,
                 _rings([_side(w, side) for *_, w in upper]),
                 start_pole=flip(shoulder - d * 0.03), end_pole=flip(elbow),
                 start_weights=_side(upper[0][2], side), end_weights=_side(upper[-1][2], side))
    # 前腕：真ん中の輪は 左ひじ と 左手捩。five_influences のときは、そのうち2つを5本の骨にする
    fore = [
        (0.0, 0.033, {"左腕": 0.5, "左ひじ": 0.5}),
        (0.5, 0.030, {"左ひじ": 0.6, "左手捩": 0.4}),
        (1.0, 0.026, {"左手捩": 0.5, "左手首": 0.5}),
    ]
    five = {"左ひじ": 0.5, "左手捩": 0.3, "左腕": 0.1, "左腕捩": 0.08, "左手首": SMALL_FIFTH_WEIGHT}
    five_slots = (0, 4) if five_influences else ()
    start = len(builder.vertices)

    def fore_weights(i: int, k: int, point: np.ndarray) -> Weights:
        if i == 1 and k in five_slots:
            return _side(five, side)
        return _side(fore[i][2], side)

    builder.tube(f"forearm_{side}", [flip(elbow + d * t * FOREARM_LENGTH) for t, *_ in fore],
                 [_ellipse(r, r, 8) for _, r, _ in fore], hint, fore_weights,
                 start_pole=flip(elbow - d * 0.01), end_pole=flip(wrist),
                 start_weights=_side(fore[0][2], side), end_weights=_side(fore[-1][2], side))
    five_vertices = [start + 8 + k for k in five_slots]
    # 手：手のひらは平たい（横 = Y の向きに広い）
    hand_centers = [wrist + d * t for t in (0.0, 0.045, 0.085)]
    builder.tube(f"hand_{side}", [flip(p) for p in hand_centers], [_rectangle(0.038, 0.012)] * 3, hint,
                 _rings([_side({"左手首": 1.0}, side)] * 3),
                 start_pole=flip(wrist - d * 0.01), end_pole=flip(wrist + d * 0.09),
                 start_weights=_side({"左手首": 1.0}, side), end_weights=_side({"左手首": 1.0}, side))
    # 指：関節ごとの輪。輪 k は k 番目の節の骨、先の輪は最後の節の骨
    names = {"親指": "thumb", "人指": "index", "中指": "middle", "薬指": "ring", "小指": "little"}
    for prefix, points in _finger_chains():
        first = 0 if prefix == "親指" else 1
        bones = [f"左{prefix}{_DIGITS[first + k]}" for k in range(3)]
        ring_weights = [_side({bones[min(k, 2)]: 1.0}, side) for k in range(4)]
        p = [np.asarray(point) for point in points]
        direction = (p[-1] - p[0]) / np.linalg.norm(p[-1] - p[0])
        builder.tube(f"{names[prefix]}_{side}", [flip(point) for point in p], [_ellipse(0.008, 0.008, 4)] * 4,
                     hint, _rings(ring_weights),
                     start_pole=flip(p[0] - direction * 0.004), end_pole=flip(p[-1] + direction * 0.004),
                     start_weights=ring_weights[0], end_weights=ring_weights[-1])
    return five_vertices


def _build_leg(builder: _MeshBuilder, side: str) -> None:
    """もも・すね・足（side の側）。ウェイトは D の骨だけ（と、ももの付け根の 下半身）"""
    flip = lambda p: _flip(p, side)  # noqa: E731
    hip, knee, ankle = (np.asarray(p) for p in (HIP_JOINT, KNEE_JOINT, ANKLE_JOINT))
    top = np.array([hip[0], hip[1], 0.80])
    # もも：（もも の上の端から ひざ への割合, 半径, ウェイト）
    thigh = [
        (0.00, 0.070, {"下半身": 0.5, "左足D": 0.5}),
        (0.15, 0.068, {"下半身": 0.2, "左足D": 0.8}),
        (0.55, 0.058, {"左足D": 1.0}),
        (1.00, 0.045, {"左足D": 0.5, "左ひざD": 0.5}),
    ]
    builder.tube(f"thigh_{side}", [flip(top + (knee - top) * t) for t, *_ in thigh],
                 [_ellipse(r, r, 10) for _, r, _ in thigh], flip((1.0, 0.0, 0.0)),
                 _rings([_side(w, side) for *_, w in thigh]),
                 start_pole=flip(top + np.array([0.0, 0.0, 0.01])), end_pole=flip(knee),
                 start_weights=_side(thigh[0][2], side), end_weights=_side(thigh[-1][2], side))
    shin = [
        (0.0, 0.045, {"左足D": 0.5, "左ひざD": 0.5}),
        (0.5, 0.040, {"左ひざD": 1.0}),
        (1.0, 0.030, {"左ひざD": 0.3, "左足首D": 0.7}),
    ]
    builder.tube(f"shin_{side}", [flip(knee + (ankle - knee) * t) for t, *_ in shin],
                 [_ellipse(r, r, 10) for _, r, _ in shin], flip((1.0, 0.0, 0.0)),
                 _rings([_side(w, side) for *_, w in shin]),
                 start_pole=flip(knee), end_pole=flip(ankle),
                 start_weights=_side(shin[0][2], side), end_weights=_side(shin[-1][2], side))
    # 足：かかと（+Y）からつま先（-Y）への四角い筒。底は z=0
    foot = [
        (0.04, 0.100, {"左足首D": 1.0}),
        (-0.03, 0.080, {"左足首D": 1.0}),
        (-0.08, 0.050, {"左足首D": 0.5, "左足先EX": 0.5}),
        (-0.14, 0.030, {"左足先EX": 1.0}),
    ]
    x = HIP_JOINT[0]
    # 軸は -Y。u = +X（左右で同じ向きにする）、v = (-Y) × (+X) = +Z。中心の高さを半分にすると、底がちょうど z=0
    builder.tube(f"foot_{side}", [flip((x, y, h / 2.0)) for y, h, _ in foot],
                 [_rectangle(0.0375, h / 2.0) for _, h, _ in foot], (1.0, 0.0, 0.0),
                 _rings([_side(w, side) for *_, w in foot]),
                 start_pole=flip((x, 0.05, 0.05)), end_pole=flip((x, -0.15, 0.015)),
                 start_weights=_side(foot[0][2], side), end_weights=_side(foot[-1][2], side),
                 axis=(0.0, -1.0, 0.0))


def _build_skirt(builder: _MeshBuilder) -> None:
    """腰のまわりの、開いた筒（ふさがない）。体より少し外にある。上は 下半身、下へ行くほど もも（足D）に乗せる"""
    rings = [(0.86, 0.135, 0.105), (0.78, 0.170, 0.135), (0.68, 0.205, 0.165), (0.58, 0.235, 0.190)]

    def weights_at(i: int, j: int, point: np.ndarray) -> Weights:
        t = i / (len(rings) - 1)
        leg = 0.6 * t
        left = min(1.0, max(0.0, 0.5 + point[0] / (2.0 * rings[i][1])))
        return {"下半身": 1.0 - leg, "左足D": leg * left, "右足D": leg * (1.0 - left)}

    builder.tube("skirt", [(0.0, 0.0, z) for z, *_ in rings], [_ellipse(rx, ry, 16) for _, rx, ry in rings],
                 (1.0, 0.0, 0.0), weights_at)


# MARK: - Blender の物を作る


def _check_height(height: float) -> float:
    if isinstance(height, bool) or not isinstance(height, (int, float)) or not math.isfinite(height) or height <= 0:
        raise ValueError(f"height は正の数（m）にしてください（いまの値：{height!r}）")
    return float(height)


def _create_armature(scale: float):
    data = bpy.data.armatures.new(ARMATURE_NAME)
    obj = bpy.data.objects.new(ARMATURE_NAME, data)
    bpy.context.scene.collection.objects.link(obj)
    view_layer = bpy.context.view_layer
    view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        specs = bone_specs()
        for spec in specs:
            bone = data.edit_bones.new(spec.name)
            bone.head = tuple(c * scale for c in spec.head)
            bone.tail = tuple(c * scale for c in spec.tail)
            bone.roll = 0.0
            bone.use_deform = spec.deform
            bone.use_connect = False
        for spec in specs:
            if spec.parent is not None:
                data.edit_bones[spec.name].parent = data.edit_bones[spec.parent]
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    for pose_bone in obj.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"
    for side in ("L", "R"):
        for target, source in ROTATION_SOURCES.items():
            side_name = (lambda name: name) if side == "L" else mirror_name
            constraint = obj.pose.bones[side_name(target)].constraints.new("COPY_ROTATION")
            constraint.name = COPY_ROTATION_NAME
            constraint.target = obj
            constraint.subtarget = side_name(source)
            constraint.owner_space = "LOCAL"
            constraint.target_space = "LOCAL"
            constraint.mix_mode = "AFTER"
    return obj


def _create_mesh(name: str, builder: _MeshBuilder, armature, scale: float):
    data = bpy.data.meshes.new(name)
    vertices = [tuple(float(c) * scale for c in v) for v in builder.vertices]
    data.from_pydata(vertices, [], builder.faces)
    data.validate(verbose=False)
    data.update()
    if len(data.vertices) != len(vertices) or len(data.polygons) != len(builder.faces):
        raise RuntimeError(f"{name}: メッシュの頂点か面の数が変わりました（作り方のまちがい）")
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = armature
    modifier = obj.modifiers.new(MODIFIER_NAME, "ARMATURE")
    modifier.object = armature
    modifier.use_vertex_groups = True
    modifier.use_bone_envelopes = False
    modifier.use_deform_preserve_volume = False
    # 頂点グループは名前の順（骨の並びの順とわざと違える）
    bones = sorted({bone for weights in builder.weights for bone in weights})
    groups = {bone: obj.vertex_groups.new(name=bone) for bone in bones}
    by_group: dict[tuple[str, float], list[int]] = {}
    for index, weights in enumerate(builder.weights):
        for bone, weight in weights.items():
            by_group.setdefault((bone, weight), []).append(index)
    for (bone, weight), indices in sorted(by_group.items()):
        groups[bone].add(indices, weight, "REPLACE")
    return obj


def _add_shape_keys(obj, moves: dict[str, list[int]], scale: float) -> None:
    """Basis と、顔のキー（まばたき：目の上の頂点を下げる、あ：口の頂点を下げる）"""
    obj.shape_key_add(name=SHAPE_KEY_NAMES[0], from_mix=False)
    for key_name, part, drop in ((SHAPE_KEY_NAMES[1], "eyelids", 0.012), (SHAPE_KEY_NAMES[2], "mouth", 0.010)):
        key = obj.shape_key_add(name=key_name, from_mix=False)
        key.value = 0.0  # 作ったばかりのキーは 1.0 になっている。ふだんの顔（キーを使わない形）にする
        for index in moves[part]:
            co = key.data[index].co
            key.data[index].co = (co[0], co[1], co[2] - drop * scale)


def build_model(*, height: float = REFERENCE_HEIGHT, with_cloth: bool = True, with_shape_keys: bool = True,
                five_influences: bool = True) -> dict:
    """いまのシーンにテスト用のモデルを作る（呼ぶ側は bpy.ops.wm.read_homefile(use_empty=True) から始める）。

    返す物：
        armature: アーマチュアのオブジェクトの名前
        meshes: メッシュのオブジェクトの名前（Body、Skirt の順）
        height: 背の高さ（m）
        bone_count: 骨の数
        vertex_counts: メッシュの名前 → 頂点の数
        parts: メッシュの名前 → {部品の名前: (最初の頂点の番号, 最後の次の番号)}
        five_influence_vertices: 5本の骨を付けた Body の頂点の番号（five_influences が False なら空）
        shape_keys: Body のシェイプキーの名前（with_shape_keys が False なら空）
    """
    height = _check_height(height)
    for name in (ARMATURE_NAME, BODY_NAME, SKIRT_NAME):
        if bpy.data.objects.get(name) is not None:
            raise ValueError(f"すでに {name} というオブジェクトがあります。空のシーンから作ってください")
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    scale = height / REFERENCE_HEIGHT
    armature = _create_armature(scale)

    body = _MeshBuilder()
    moves = _build_trunk(body)
    five_vertices = []
    for side in ("L", "R"):
        five_vertices += _build_arm(body, side, five_influences)
    for side in ("L", "R"):
        _build_leg(body, side)
    if tuple(body.parts) != BODY_PARTS:
        raise RuntimeError(f"Body の部品の並びが決まりと違います：{tuple(body.parts)}")
    body_obj = _create_mesh(BODY_NAME, body, armature, scale)
    if with_shape_keys:
        _add_shape_keys(body_obj, moves, scale)
    meshes = [body_obj]
    parts = {BODY_NAME: dict(body.parts)}
    if with_cloth:
        skirt = _MeshBuilder()
        _build_skirt(skirt)
        meshes.append(_create_mesh(SKIRT_NAME, skirt, armature, scale))
        parts[SKIRT_NAME] = dict(skirt.parts)

    bpy.context.view_layer.objects.active = armature
    bpy.context.view_layer.update()
    return {
        "armature": armature.name,
        "meshes": [obj.name for obj in meshes],
        "height": height,
        "bone_count": len(armature.data.bones),
        "vertex_counts": {obj.name: len(obj.data.vertices) for obj in meshes},
        "parts": parts,
        "five_influence_vertices": five_vertices,
        "shape_keys": list(SHAPE_KEY_NAMES) if with_shape_keys else [],
    }


def save_model(path: str | Path) -> Path:
    """いまのファイルを path に .blend で保存する（写しとして保存する。いま開いているファイルの場所は変えない）"""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    result = bpy.ops.wm.save_as_mainfile(filepath=str(target), copy=True, compress=False)
    if "FINISHED" not in result or not target.exists():
        raise RuntimeError(f"保存できませんでした：{target}")
    return target


# MARK: - コマンド


def main(argv: Sequence[str] | None = None) -> None:
    """blender --background --python tests/fixtures/mmd_model.py -- <out.blend> [--no-cloth] ..."""
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(prog="mmd_model.py", description="テスト用の MMD に似た人型のモデルを .blend に保存する")
    parser.add_argument("output", type=Path, help="保存する .blend")
    parser.add_argument("--no-cloth", action="store_true", help="スカートを作らない")
    parser.add_argument("--no-shape-keys", action="store_true", help="シェイプキーを作らない")
    parser.add_argument("--four-influences", action="store_true", help="5本の骨を付けた頂点を作らない")
    parser.add_argument("--height", type=float, default=REFERENCE_HEIGHT, help="背の高さ（m）")
    args = parser.parse_args(list(argv))
    bpy.ops.wm.read_homefile(use_factory_startup=True, use_empty=True)
    info = build_model(height=args.height, with_cloth=not args.no_cloth, with_shape_keys=not args.no_shape_keys,
                       five_influences=not args.four_influences)
    target = save_model(args.output)
    print(f"保存しました：{target}（骨 {info['bone_count']}、頂点 {info['vertex_counts']}）")


if __name__ == "__main__":
    main()

"""A5 確かめの動き：どの MMD 準標準のモデルにも使える、決まったポーズの並び（bpy なし・numpy だけ）。

ポーズは3つの組に分ける：
- "rom"：関節の動く範囲。関節を1つずつ、ふつうの人が動かせる端の近くまで動かす。
- "daily"：ふだんの動き（両手を上げる・腕を組む・手をたたく・おじぎ・歩く・しゃがむ・正座 など）。
- "lying"：横になる（全ての親 で体全体を回す）。

ポーズの書き方（PoseSpec）：
- 骨の名前は、MMD 準標準の共通の名前（bones の辞書の字のまま）。モデルの骨の名前には、bones.map_bones で対応させる。
- 回転は (骨, 軸, 角度) を順に当てる。骨の根元を中心に、その骨と子の骨を回す。
- 軸は「アーマチュアの休みの向き」で書く：+X がキャラの左、+Y がキャラの後ろ（モデルは -Y を向く）、+Z が上。
  親の骨が回っていれば、軸も親と一緒に回る（体の向きに付いた軸。骨のロール（ねじれの向き）には左右されない）。
  骨のローカルの回転は conj(q_rest) * q_axis * q_rest（local_rotation）。
- 同じ骨に2つ以上の回転があるときは、書いた順に当てる。あとの回転の軸も、親に付いた休みの向きで書く
  （前の回転で骨が動いたあとの向きで考える。合わせた回転は q_2 * q_1）。
- 角度は度で、いつも正（向きは軸で決める）。ひざ・ひじは、ふつうに曲がる向きにだけ回す（逆曲がりにしない）。
- center_offset_legs：センター を動かす量（アーマチュアの休みの向き。単位はモデルの脚の長さ）。座る・しゃがむ姿勢で使う。
- ground：ポーズを付けたあと、センター を上下に動かして、いちばん低い所を床（休みの姿勢の体のいちばん低い所）に合わせる。
- lying：横になる姿勢（全ての親 を回す）。横になる姿勢だけが 全ての親 を回す。

左右のある骨は「左」で書き、右は鏡に映して作る（_mirrored）。鏡に映した軸は (x, -y, -z)（回転の軸は、映すと向きが変わる）。
腕の角度は、MMD のふつうの A ポーズ（腕が水平から 30〜45 度下がる）を前提にしている。
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import bones
from ..thresholds import POSE_FOLLOW_TOLERANCE_DEG
from ..measure.geometry import as_array
from ..measure.kinematics import quat_angle_deg

__all__ = [
    "BACK",
    "CATEGORIES",
    "CENTER_BONE",
    "D_BONE_SOURCES",
    "JOINT_LIMITS_DEG",
    "LEFT",
    "POSES",
    "POSE_FOLLOW_TOLERANCE_DEG",
    "ROOT_BONE",
    "UP",
    "AppliedPose",
    "PoseSpec",
    "ResolvedPose",
    "axis_quaternion",
    "find_overridden",
    "leg_length_from_bones",
    "local_rotation",
    "pose_by_name",
    "quat_multiply",
    "resolve_pose",
    "rotation_deltas",
]

# MARK: - 決まり

CATEGORIES = ("rom", "daily", "lying")
# 横になる姿勢で回す骨（体全体の根）と、座る・しゃがむ・床に合わせるときに動かす骨
ROOT_BONE = "全ての親"
CENTER_BONE = "センター"

# アーマチュアの休みの向き（体の向きの軸）
LEFT = (1.0, 0.0, 0.0)  # キャラの左
BACK = (0.0, 1.0, 0.0)  # キャラの後ろ（モデルは -Y を向く）
UP = (0.0, 0.0, 1.0)  # 上

# 回転付与で、元の骨の回転を写す D の骨 → 元の骨（左右とも）。準標準のモデルでは、脚のウェイトは D の骨に乗る
D_BONE_SOURCES = {"左足D": "左足", "左ひざD": "左ひざ", "左足首D": "左足首",
                  "右足D": "右足", "右ひざD": "右ひざ", "右足首D": "右足首"}


# ポーズの角度の上限（度。骨1本に付ける回転の角度の合計）。ふつうの人の関節の動く範囲の目安で、
# ポーズの並び（POSES）を作るときの決まり（測った値の判定には使わない）。左右のある骨は左の名前で書く。
# ここに無い骨は、ポーズで回さない
JOINT_LIMITS_DEG: dict[str, float] = {
    "全ての親": 90.0,  # 横になる（体全体を回す）
    "上半身": 50.0,
    "上半身2": 40.0,
    "首": 60.0,
    "頭": 40.0,
    "左肩": 20.0,
    "左腕": 150.0,  # 肩の関節（上げる・下げる・前後・ひねりの合計）
    "左ひじ": 145.0,
    "左手捩": 90.0,
    "左手首": 80.0,
    "左親指０": 30.0,
    "左親指１": 60.0,
    "左親指２": 80.0,
    "左人指１": 90.0, "左人指２": 100.0, "左人指３": 80.0,
    "左中指１": 90.0, "左中指２": 100.0, "左中指３": 80.0,
    "左薬指１": 90.0, "左薬指２": 100.0, "左薬指３": 80.0,
    "左小指１": 90.0, "左小指２": 100.0, "左小指３": 80.0,
    "左足": 120.0,  # 股関節
    "左ひざ": 160.0,  # 正座で約 160 度
    "左足首": 80.0,  # 正座で足の甲を床に向けると約 75 度（足首と足の甲の曲がりを合わせた物）
    "左足先EX": 60.0,
}

_NAME = re.compile(r"[a-z][a-z0-9_]*")
# 回転の軸の長さが、これより短いと向きが決まらない
_MIN_AXIS_LENGTH = 1e-9


# MARK: - 回転（クォータニオン。Blender の順 w, x, y, z）


def _unit(vector: Sequence[float]) -> tuple[float, float, float]:
    """長さ 1 の向き（タプル）"""
    v = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(v))
    if not math.isfinite(length) or length <= _MIN_AXIS_LENGTH:
        raise ValueError(f"軸の向きが決まりません（長さ {length}）")
    v = v / length
    return float(v[0]), float(v[1]), float(v[2])


def _as_axis(axis: Any, name: str = "axis") -> np.ndarray:
    a = as_array(axis, name)
    if a.dtype.kind not in "iuf" or a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError(f"{name} は3つの有限の数にしてください")
    length = float(np.linalg.norm(a))
    if length <= _MIN_AXIS_LENGTH:
        raise ValueError(f"{name} の長さが 0 です")
    return np.asarray(a, dtype=np.float64) / length


def _as_angle(angle_deg: Any, name: str = "angle_deg") -> float:
    if isinstance(angle_deg, bool) or not isinstance(angle_deg, (int, float, np.integer, np.floating)) \
            or not math.isfinite(float(angle_deg)):
        raise ValueError(f"{name} は有限の数にしてください（いまの値：{angle_deg!r}）")
    return float(angle_deg)


def axis_quaternion(axis: Any, angle_deg: float) -> np.ndarray:
    """axis のまわりに angle_deg 度回すクォータニオン (4,)（w, x, y, z。右手の決まり）。軸の長さは何でもよい"""
    a = _as_axis(axis)
    half = math.radians(_as_angle(angle_deg)) / 2.0
    return np.concatenate(([math.cos(half)], math.sin(half) * a))


def quat_multiply(a: Any, b: Any) -> np.ndarray:
    """ハミルトンの積 a * b（先に b、あとに a を当てる回転）。(..., 4) どうし"""
    p = np.asarray(a, dtype=np.float64)
    q = np.asarray(b, dtype=np.float64)
    w1, x1, y1, z1 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    w2, x2, y2, z2 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def _conjugate(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def _rotation_part(matrix: np.ndarray) -> np.ndarray:
    """(..., 3, 3) の行列に、いちばん近い回転（大きさを外す。SVD の極分解）"""
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    flip = np.linalg.det(rotation) < 0.0
    if np.any(flip):
        u = u.copy()
        u[flip, :, 2] *= -1.0
        rotation = u @ vt
    return rotation


def _matrix_quaternion(rotations: np.ndarray) -> np.ndarray:
    """回転の行列 (..., 3, 3)（列ベクトルに左から掛ける形）をクォータニオン (..., 4) にする。w が負にならない方を返す。
    いちばん大きい成分から計算する（180 度の近くでも正しい）"""
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


def _as_matrix(value: Any, name: str) -> np.ndarray:
    matrix = as_array(value, name)
    if matrix.dtype.kind not in "iuf" or matrix.shape not in ((4, 4), (3, 3)) \
            or not np.isfinite(np.asarray(matrix, dtype=np.float64)).all():
        raise ValueError(f"{name} は有限の数の (4, 4) か (3, 3) の行列にしてください（いまの形：{matrix.shape}）")
    return np.asarray(matrix, dtype=np.float64)[:3, :3]


def local_rotation(rest_matrix: Any, axis: Any, angle_deg: float) -> tuple[float, float, float, float]:
    """アーマチュアの休みの向きの軸 axis のまわりに angle_deg 度回すことを、骨のローカルの回転（w, x, y, z）にする。

    = conj(q_rest) * q_axis * q_rest（q_rest は骨の休みの行列の回転の部分）。Blender のポーズの骨の行列は
    親 @ (親の休み⁻¹ @ 休み) @ basis なので、親が休みの姿勢なら、骨はアーマチュアの中で q_axis だけ回る。
    親が回っていれば、軸も親と一緒に回る。骨のロール（休みの行列の Y 軸まわりのねじれ）が違っても、
    アーマチュアの中での回り方は同じになる。rest_matrix は bone.matrix_local（(4, 4) か (3, 3)）。
    """
    rest = _rotation_part(_as_matrix(rest_matrix, "rest_matrix"))
    q_rest = _matrix_quaternion(rest)
    q = quat_multiply(_conjugate(q_rest), quat_multiply(axis_quaternion(axis, angle_deg), q_rest))
    q = q / np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def rotation_deltas(rest_matrices: Any, posed_matrices: Any) -> np.ndarray:
    """骨を、休みの向きからポーズの向きへ回す回転 (B, 4)（w, x, y, z）= ポーズの回転 @ 休みの回転の逆。

    rest_matrices・posed_matrices は同じ空間（アーマチュアの中など）の (B, 4, 4) か (B, 3, 3)。大きさは外してから回転にする
    """
    rest = as_array(rest_matrices, "rest_matrices")
    posed = as_array(posed_matrices, "posed_matrices")
    for value, name in ((rest, "rest_matrices"), (posed, "posed_matrices")):
        if value.dtype.kind not in "iuf" or value.ndim != 3 or value.shape[1:] not in ((4, 4), (3, 3)):
            raise ValueError(f"{name} は (B, 4, 4) か (B, 3, 3) の行列の並びにしてください（いまの形：{value.shape}）")
    if len(rest) != len(posed):
        raise ValueError(f"rest_matrices と posed_matrices の数が違います（{len(rest)} と {len(posed)}）")
    if len(rest) == 0:
        return np.zeros((0, 4))
    r = _rotation_part(np.asarray(rest, dtype=np.float64)[:, :3, :3])
    p = _rotation_part(np.asarray(posed, dtype=np.float64)[:, :3, :3])
    return _matrix_quaternion(p @ np.swapaxes(r, -1, -2))


# MARK: - ポーズの形


Rotation = tuple[str, tuple[float, float, float], float]


@dataclass(frozen=True)
class PoseSpec:
    """1つの確かめのポーズ。

    name: 英語の名前（小文字・数字・_）
    text_key: 画面と報告の名前の文のキー（"motion.pose_<name>"）
    category: "rom"（関節の動く範囲）・"daily"（ふだんの動き）・"lying"（横になる）
    rotations: (共通の名前, 軸 (3,), 角度（度、正）) の並び。書いた順に当てる（モジュールの説明）
    center_offset_legs: センター を動かす量 (dx, dy, dz)（アーマチュアの休みの向き。単位はモデルの脚の長さ）
    ground: ポーズのあと、いちばん低い所を床に合わせる
    lying: 横になる姿勢（全ての親 を回す）
    """

    name: str
    text_key: str
    category: str
    rotations: tuple[Rotation, ...] = ()
    center_offset_legs: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ground: bool = False
    lying: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME.fullmatch(self.name):
            raise ValueError(f"ポーズの名前は英語の小文字・数字・_ だけにしてください（いまは {self.name!r}）")
        if self.text_key != f"motion.pose_{self.name}":
            raise ValueError(f"{self.name}：text_key は \"motion.pose_{self.name}\" にしてください（いまは {self.text_key!r}）")
        if self.category not in CATEGORIES:
            raise ValueError(f"{self.name}：category は {'・'.join(CATEGORIES)} のどれかにしてください（いまは {self.category!r}）")
        if not isinstance(self.ground, bool) or not isinstance(self.lying, bool):
            raise ValueError(f"{self.name}：ground と lying は True か False にしてください")
        if isinstance(self.rotations, (str, bytes)) or not isinstance(self.rotations, Iterable):
            raise ValueError(f"{self.name}：rotations は (骨, 軸, 角度) の並びにしてください")
        rotations = []
        for item in self.rotations:
            if not isinstance(item, (tuple, list)) or len(item) != 3:
                raise ValueError(f"{self.name}：回転は (骨, 軸, 角度) の3つの組にしてください（いまは {item!r}）")
            bone, axis, angle = item
            canonical = bones.canonical_bone(bone) if isinstance(bone, str) else None
            if canonical is None or canonical.name != bone:
                raise ValueError(f"{self.name}：骨の名前は共通の名前（MMD 準標準の字のまま）にしてください（いまは {bone!r}）")
            unit_axis = _as_axis(axis, f"{self.name} の {bone} の軸")
            value = _as_angle(angle, f"{self.name} の {bone} の角度")
            if not 0.0 < value <= 180.0:
                raise ValueError(f"{self.name}：{bone} の角度は 0 より大きく 180 以下にしてください（向きは軸で決める。"
                                 f"いまは {value}）")
            rotations.append((bone, (float(unit_axis[0]), float(unit_axis[1]), float(unit_axis[2])), value))
        object.__setattr__(self, "rotations", tuple(rotations))
        offset = as_array(self.center_offset_legs, f"{self.name} の center_offset_legs")
        if offset.dtype.kind not in "iuf" or offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError(f"{self.name}：center_offset_legs は3つの有限の数にしてください")
        object.__setattr__(self, "center_offset_legs", tuple(float(v) for v in offset))
        rotates_root = any(bone == ROOT_BONE for bone, _, _ in rotations)
        if self.lying != rotates_root or self.lying != (self.category == "lying"):
            raise ValueError(f"{self.name}：横になる姿勢（lying・category \"lying\"）だけが、{ROOT_BONE} を回します")

    @property
    def needs_center(self) -> bool:
        """センター を動かすか（center_offset_legs が 0 でない、または ground）"""
        return self.ground or any(v != 0.0 for v in self.center_offset_legs)

    @property
    def bones(self) -> tuple[str, ...]:
        """このポーズに要る共通の名前（回す骨の順。センター を動かすときは最後に センター）"""
        names = dict.fromkeys(bone for bone, _, _ in self.rotations)
        if self.needs_center:
            names.setdefault(CENTER_BONE, None)
        return tuple(names)

    def axis_rotations(self) -> dict[str, np.ndarray]:
        """共通の名前 → その骨の回転を合わせたクォータニオン（アーマチュアの休みの向き）。書いた順に当てる（q_2 * q_1）"""
        result: dict[str, np.ndarray] = {}
        for bone, axis, angle in self.rotations:
            q = axis_quaternion(axis, angle)
            result[bone] = q if bone not in result else quat_multiply(q, result[bone])
        return result


def _mirror_axis(axis: Sequence[float]) -> tuple[float, float, float]:
    """左右を鏡に映した回転の軸：(x, -y, -z)（回転の軸は、映すと向きが変わる）"""
    return float(axis[0]), -float(axis[1]), -float(axis[2])


def _mirrored(rotations: Iterable[Rotation]) -> tuple[Rotation, ...]:
    """左の回転を、右に映す（体の中心の骨は、名前はそのままで軸を映す）"""
    result = []
    for bone, axis, angle in rotations:
        other = bones.mirror_name(bone)
        result.append((bone if other is None else other, _mirror_axis(axis), angle))
    return tuple(result)


def _both(*left: Rotation) -> tuple[Rotation, ...]:
    """左の回転と、それを右に映した回転（左右そろった姿勢）"""
    return tuple(left) + _mirrored(left)


def _pose(name: str, category: str, rotations: Iterable[Rotation], *,
          center_offset_legs: tuple[float, float, float] = (0.0, 0.0, 0.0), ground: bool = False) -> PoseSpec:
    return PoseSpec(name=name, text_key=f"motion.pose_{name}", category=category, rotations=tuple(rotations),
                    center_offset_legs=center_offset_legs, ground=ground, lying=category == "lying")


# MARK: - 体の向きの軸（左の骨で書く。右は鏡に映す）

# 体の中心の骨（上半身・首・頭）：前へ曲げる・後ろへ反らす・左へ傾ける・左を向く
_FORWARD = (1.0, 0.0, 0.0)  # 上を向いた骨の先が前（-Y）へ。下を向いた骨（すね）の先は後ろへ
_BACKWARD = (-1.0, 0.0, 0.0)
_LEAN_LEFT = (0.0, 1.0, 0.0)  # 上を向いた骨の先が左（+X）へ
_TURN_LEFT = (0.0, 0.0, 1.0)  # 前（-Y）が左（+X）へ向く
# 左腕（休みの姿勢で +X へ伸び、少し下がる）
_ARM_RAISE = (0.0, -1.0, 0.0)  # 横から上へ上げる
_ARM_LOWER = (0.0, 1.0, 0.0)  # 横から下へ下ろす
_ARM_SWING_FORWARD = (0.0, 0.0, -1.0)  # 水平に前へ回す
_HANGING_FORWARD = (-1.0, 0.0, 0.0)  # 下ろした腕・脚を前へ振る
_HANGING_BACK = (1.0, 0.0, 0.0)  # 下ろした腕・脚を後ろへ振る
_ARM_TURN_IN = (0.0, 0.0, -1.0)  # 下ろした腕を、縦の軸のまわりに内へひねる（前腕が体の前へ来る）
# ひじを曲げる軸：上腕に垂直で、前腕が前へ曲がる向き（水平から 30 度下がった上腕に合わせる。A ポーズの腕に近い）
_ELBOW_FLEX = _unit((-math.sin(math.radians(30.0)), 0.0, -math.cos(math.radians(30.0))))
# 前腕・手首のひねりの軸（前腕の向き。水平から 35 度下がった向き）
_FOREARM_TWIST = _unit((math.cos(math.radians(35.0)), 0.0, -math.sin(math.radians(35.0))))
# 指・手首を手のひらの側へ曲げる（手のひらは下・内を向く）
_CURL = (0.0, 1.0, 0.0)
# 親指を手のひらの側へ曲げる
_THUMB_CURL = _unit((0.45, 0.82, -0.35))
# 左脚
_HIP_FLEX = (-1.0, 0.0, 0.0)  # 脚を前へ上げる
_HIP_EXTEND = (1.0, 0.0, 0.0)  # 脚を後ろへ引く
_HIP_ABDUCT = (0.0, -1.0, 0.0)  # 脚を横（左）へ開く
_THIGH_OUT = (0.0, 0.0, 1.0)  # 前へ上げた太ももを、外（左）へ開く
_KNEE_FLEX = (1.0, 0.0, 0.0)  # ひざを曲げる（すねの先が後ろへ。ふつうの向き）
_ANKLE_DOWN = (1.0, 0.0, 0.0)  # つま先を下げる（つま先立ち）
_ANKLE_UP = (-1.0, 0.0, 0.0)  # つま先を上げる
_TOES_UP = (-1.0, 0.0, 0.0)  # 足の指を上へ曲げる


def _r(bone: str, axis: Sequence[float], angle: float) -> Rotation:
    return bone, (float(axis[0]), float(axis[1]), float(axis[2])), float(angle)


def _wave_twist_axis() -> tuple[float, float, float]:
    """手を振るとき、右の上腕をひねる軸：右腕を横へ 50 度上げたあとの、上腕の向き（A ポーズ 37.5 度から）"""
    down = math.radians(37.5 - 50.0)
    return _unit((-math.cos(down), 0.0, -math.sin(down)))


# MARK: - ポーズの並び

_FINGERS = ("人指", "中指", "薬指", "小指")


def _fist(side_rotations: list[Rotation]) -> None:
    for finger in _FINGERS:
        side_rotations += [_r(f"左{finger}１", _CURL, 80.0), _r(f"左{finger}２", _CURL, 90.0),
                           _r(f"左{finger}３", _CURL, 60.0)]
    side_rotations += [_r("左親指０", _THUMB_CURL, 15.0), _r("左親指１", _THUMB_CURL, 35.0),
                       _r("左親指２", _THUMB_CURL, 45.0)]


def _rom_poses() -> list[PoseSpec]:
    fist: list[Rotation] = [_r("左手首", _CURL, 30.0)]
    _fist(fist)
    return [
        _pose("head_down", "rom", [_r("首", _FORWARD, 30.0), _r("頭", _FORWARD, 25.0)]),
        _pose("head_up", "rom", [_r("首", _BACKWARD, 35.0), _r("頭", _BACKWARD, 25.0)]),
        _pose("head_turn_left", "rom", [_r("首", _TURN_LEFT, 50.0), _r("頭", _TURN_LEFT, 20.0)]),
        _pose("head_turn_right", "rom", _mirrored([_r("首", _TURN_LEFT, 50.0), _r("頭", _TURN_LEFT, 20.0)])),
        _pose("head_tilt_left", "rom", [_r("首", _LEAN_LEFT, 25.0), _r("頭", _LEAN_LEFT, 15.0)]),
        _pose("head_tilt_right", "rom", _mirrored([_r("首", _LEAN_LEFT, 25.0), _r("頭", _LEAN_LEFT, 15.0)])),
        _pose("torso_forward", "rom", [_r("上半身", _FORWARD, 45.0), _r("上半身2", _FORWARD, 30.0)]),
        _pose("torso_back", "rom", [_r("上半身", _BACKWARD, 20.0), _r("上半身2", _BACKWARD, 15.0)]),
        _pose("torso_side_left", "rom", [_r("上半身", _LEAN_LEFT, 20.0), _r("上半身2", _LEAN_LEFT, 15.0)]),
        _pose("torso_side_right", "rom",
              _mirrored([_r("上半身", _LEAN_LEFT, 20.0), _r("上半身2", _LEAN_LEFT, 15.0)])),
        _pose("torso_twist_left", "rom", [_r("上半身", _TURN_LEFT, 25.0), _r("上半身2", _TURN_LEFT, 20.0)]),
        _pose("torso_twist_right", "rom",
              _mirrored([_r("上半身", _TURN_LEFT, 25.0), _r("上半身2", _TURN_LEFT, 20.0)])),
        _pose("shoulders_up", "rom", _both(_r("左肩", _ARM_RAISE, 15.0))),
        _pose("arms_down", "rom", _both(_r("左腕", _ARM_LOWER, 40.0))),
        _pose("arms_back", "rom", _both(_r("左腕", _ARM_LOWER, 40.0), _r("左腕", _HANGING_BACK, 40.0))),
        _pose("elbows_bend", "rom", _both(_r("左ひじ", _ELBOW_FLEX, 130.0))),
        _pose("forearms_twist", "rom", _both(_r("左手捩", _FOREARM_TWIST, 80.0))),
        _pose("fists", "rom", _both(*fist)),
        _pose("leg_front_left", "rom", [_r("左足", _HIP_FLEX, 70.0)]),
        _pose("leg_front_right", "rom", _mirrored([_r("左足", _HIP_FLEX, 70.0)])),
        _pose("leg_side_left", "rom", [_r("左足", _HIP_ABDUCT, 40.0)]),
        _pose("leg_side_right", "rom", _mirrored([_r("左足", _HIP_ABDUCT, 40.0)])),
        _pose("leg_back_left", "rom", [_r("左足", _HIP_EXTEND, 25.0)]),
        _pose("leg_back_right", "rom", _mirrored([_r("左足", _HIP_EXTEND, 25.0)])),
        _pose("knee_bend_left", "rom", [_r("左足", _HIP_FLEX, 10.0), _r("左ひざ", _KNEE_FLEX, 130.0)]),
        _pose("knee_bend_right", "rom", _mirrored([_r("左足", _HIP_FLEX, 10.0), _r("左ひざ", _KNEE_FLEX, 130.0)])),
        _pose("tiptoe", "rom", _both(_r("左足首", _ANKLE_DOWN, 35.0), _r("左足先EX", _TOES_UP, 35.0)), ground=True),
    ]


def _daily_poses() -> list[PoseSpec]:
    return [
        _pose("arms_up", "daily", _both(_r("左肩", _ARM_RAISE, 15.0), _r("左腕", _ARM_RAISE, 105.0))),
        # 腕を組む：上腕を下ろして内へひねり、前腕を胸の前で重ねる（右の前腕を少し上にする）
        _pose("arms_crossed", "daily", [
            *_both(_r("左腕", _ARM_LOWER, 45.0), _r("左腕", _HANGING_FORWARD, 30.0), _r("左腕", _ARM_TURN_IN, 65.0)),
            _r("左ひじ", _ELBOW_FLEX, 100.0), *_mirrored([_r("左ひじ", _ELBOW_FLEX, 110.0)]),
        ]),
        _pose("clap", "daily", _both(_r("左腕", _ARM_LOWER, 45.0), _r("左腕", _HANGING_FORWARD, 40.0),
                                     _r("左腕", _ARM_TURN_IN, 35.0), _r("左ひじ", _ELBOW_FLEX, 70.0))),
        _pose("hands_on_cheeks", "daily", _both(_r("左腕", _ARM_LOWER, 45.0), _r("左腕", _HANGING_FORWARD, 45.0),
                                                _r("左腕", _ARM_TURN_IN, 20.0), _r("左ひじ", _ELBOW_FLEX, 140.0))),
        _pose("hand_on_chin", "daily", [
            *_mirrored([_r("左腕", _ARM_LOWER, 45.0), _r("左腕", _HANGING_FORWARD, 50.0), _r("左腕", _ARM_TURN_IN, 35.0),
                        _r("左ひじ", _ELBOW_FLEX, 140.0)]),
            _r("首", _FORWARD, 10.0),
        ]),
        _pose("bow", "daily", [_r("上半身", _FORWARD, 30.0), _r("上半身2", _FORWARD, 15.0), _r("首", _FORWARD, 10.0),
                               *_both(_r("左腕", _ARM_LOWER, 40.0), _r("左腕", _HANGING_FORWARD, 45.0))]),
        _pose("look_back", "daily", [_r("上半身", _TURN_LEFT, 25.0), _r("上半身2", _TURN_LEFT, 20.0),
                                     _r("首", _TURN_LEFT, 50.0), _r("頭", _TURN_LEFT, 20.0)]),
        _pose("walk", "daily", [
            _r("左足", _HIP_FLEX, 25.0), _r("左ひざ", _KNEE_FLEX, 10.0), _r("左足首", _ANKLE_UP, 10.0),
            *_mirrored([_r("左足", _HIP_EXTEND, 15.0), _r("左ひざ", _KNEE_FLEX, 25.0), _r("左足首", _ANKLE_DOWN, 15.0)]),
            _r("左腕", _ARM_LOWER, 40.0), _r("左腕", _HANGING_BACK, 20.0),
            *_mirrored([_r("左腕", _ARM_LOWER, 40.0), _r("左腕", _HANGING_FORWARD, 20.0)]),
            *_both(_r("左ひじ", _ELBOW_FLEX, 15.0)),
        ], ground=True),
        _pose("run", "daily", [
            _r("上半身", _FORWARD, 10.0),
            _r("左足", _HIP_FLEX, 50.0), _r("左ひざ", _KNEE_FLEX, 80.0), _r("左足首", _ANKLE_DOWN, 10.0),
            *_mirrored([_r("左足", _HIP_EXTEND, 20.0), _r("左ひざ", _KNEE_FLEX, 30.0), _r("左足首", _ANKLE_DOWN, 25.0)]),
            _r("左腕", _ARM_LOWER, 40.0), _r("左腕", _HANGING_BACK, 35.0),
            *_mirrored([_r("左腕", _ARM_LOWER, 40.0), _r("左腕", _HANGING_FORWARD, 35.0)]),
            *_both(_r("左ひじ", _ELBOW_FLEX, 90.0)),
        ], ground=True),
        _pose("squat", "daily", [
            _r("上半身", _FORWARD, 25.0),
            *_both(_r("左足", _HIP_FLEX, 85.0), _r("左足", _THIGH_OUT, 15.0), _r("左ひざ", _KNEE_FLEX, 110.0),
                   _r("左足首", _ANKLE_UP, 25.0),
                   _r("左腕", _ARM_LOWER, 40.0), _r("左腕", _HANGING_FORWARD, 60.0)),
        ], center_offset_legs=(0.0, 0.25, -0.5), ground=True),
        _pose("chair_sit", "daily", _both(
            _r("左足", _HIP_FLEX, 85.0), _r("左ひざ", _KNEE_FLEX, 85.0),
            _r("左腕", _ARM_LOWER, 40.0), _r("左腕", _HANGING_FORWARD, 30.0), _r("左ひじ", _ELBOW_FLEX, 30.0),
        ), center_offset_legs=(0.0, 0.0, -0.45), ground=True),
        # 正座：すねを床に沿わせ（太もも 70 度 + ひざ 160 度）、足の甲を床に向ける
        _pose("seiza", "daily", _both(
            _r("左足", _HIP_FLEX, 70.0), _r("左ひざ", _KNEE_FLEX, 160.0), _r("左足首", _ANKLE_DOWN, 75.0),
            _r("左腕", _ARM_LOWER, 40.0), _r("左腕", _HANGING_FORWARD, 35.0), _r("左ひじ", _ELBOW_FLEX, 40.0),
        ), center_offset_legs=(0.0, 0.0, -0.7), ground=True),
        _pose("hands_on_knees", "daily", [
            _r("上半身", _FORWARD, 35.0), _r("上半身2", _FORWARD, 10.0),
            *_both(_r("左足", _HIP_FLEX, 30.0), _r("左ひざ", _KNEE_FLEX, 45.0), _r("左足首", _ANKLE_UP, 15.0),
                   _r("左腕", _ARM_LOWER, 45.0), _r("左腕", _HANGING_FORWARD, 45.0), _r("左腕", _ARM_TURN_IN, 15.0)),
        ], ground=True),
        _pose("wave", "daily", [_r("右腕", _mirror_axis(_ARM_RAISE), 50.0), _r("右腕", _wave_twist_axis(), 90.0),
                                _r("右ひじ", _mirror_axis(_ELBOW_FLEX), 100.0)]),
        # 刀を構える（中段）：右足を前、左足を後ろ（かかとを上げる）。両手を体の前の真ん中で、右手を前にする
        _pose("sword_stance", "daily", [
            *_mirrored([_r("左足", _HIP_FLEX, 20.0), _r("左ひざ", _KNEE_FLEX, 15.0)]),
            _r("左足", _HIP_EXTEND, 15.0), _r("左ひざ", _KNEE_FLEX, 10.0), _r("左足首", _ANKLE_DOWN, 20.0),
            _r("左腕", _ARM_LOWER, 45.0), _r("左腕", _HANGING_FORWARD, 40.0), _r("左腕", _ARM_TURN_IN, 32.0),
            _r("左ひじ", _ELBOW_FLEX, 45.0),
            *_mirrored([_r("左腕", _ARM_LOWER, 45.0), _r("左腕", _HANGING_FORWARD, 55.0), _r("左腕", _ARM_TURN_IN, 28.0),
                        _r("左ひじ", _ELBOW_FLEX, 25.0)]),
        ], ground=True),
    ]


def _lying_poses() -> list[PoseSpec]:
    return [
        # あおむけ：前（-Y）が上を向くように、体全体を X のまわりに回す
        _pose("lying_back", "lying", [_r(ROOT_BONE, _BACKWARD, 90.0), *_both(_r("左腕", _ARM_LOWER, 30.0))],
              ground=True),
        # 左を下にした横向き：キャラの左（+X）が下を向くように、Y のまわりに回す。下の腕は前へ出し、ひざを少し曲げる
        _pose("lying_side", "lying", [
            _r(ROOT_BONE, _LEAN_LEFT, 90.0),
            _r("左腕", _ARM_SWING_FORWARD, 70.0),
            *_mirrored([_r("左腕", _ARM_LOWER, 40.0)]),
            *_both(_r("左足", _HIP_FLEX, 30.0), _r("左ひざ", _KNEE_FLEX, 40.0)),
        ], ground=True),
    ]


POSES: tuple[PoseSpec, ...] = tuple(_rom_poses() + _daily_poses() + _lying_poses())


def pose_by_name(name: str) -> PoseSpec:
    """名前でポーズを引く。無い名前は ValueError"""
    for pose in POSES:
        if pose.name == name:
            return pose
    raise ValueError(f"ポーズ {name!r} はありません")


# MARK: - モデルの骨に当てる（bpy なしの計算）


@dataclass(frozen=True)
class ResolvedPose:
    """ポーズを、あるモデルの骨に対応させた物（poses_bpy.apply_pose が Blender の骨に入れる）。

    rotations: モデルの骨の名前 → 骨のローカルの回転（w, x, y, z。pose_bone.rotation_quaternion に入れる値）
    axis_rotations: モデルの骨の名前 → アーマチュアの休みの向きでの回転（w, x, y, z。決めた向きと比べるのに使う）
    canonical: モデルの骨の名前 → 共通の名前
    center: センター のモデルの骨の名前（無ければ None）
    center_location: センター の位置の動き（骨のローカル。pose_bone.location に入れる値。メートル）
    missing: モデルに無かった共通の名前（ポーズに要る骨の順）
    """

    pose: str
    rotations: dict[str, tuple[float, float, float, float]]
    axis_rotations: dict[str, tuple[float, float, float, float]]
    canonical: dict[str, str]
    center: str | None
    center_location: tuple[float, float, float]
    missing: tuple[str, ...]


def _as_names(names: Any, what: str) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)) or not isinstance(names, Iterable):
        raise ValueError(f"{what} は名前の並びにしてください")
    values = tuple(names)
    if not all(isinstance(name, str) for name in values):
        raise ValueError(f"{what} の中は文字にしてください")
    return values


def _as_matrices(value: Any, name: str, count: int) -> np.ndarray:
    array = as_array(value, name)
    if array.dtype.kind not in "iuf" or array.ndim != 3 or array.shape[1:] not in ((4, 4), (3, 3)):
        raise ValueError(f"{name} は (B, 4, 4) か (B, 3, 3) の行列の並びにしてください（いまの形：{array.shape}）")
    if len(array) != count:
        raise ValueError(f"{name} の数（{len(array)}）が骨の数（{count}）と違います")
    if not np.isfinite(np.asarray(array, dtype=np.float64)).all():
        raise ValueError(f"{name} に NaN か無限大があります")
    return np.asarray(array, dtype=np.float64)


def resolve_pose(pose: PoseSpec, bone_names: Sequence[str], rest_matrices: Any, *,
                 leg_length: float | None = None) -> ResolvedPose:
    """ポーズを、モデルの骨の名前と休みの行列（bone_names の順。(B, 4, 4) のアーマチュアの空間）に対応させる。

    共通の名前は bones.map_bones でモデルの骨の名前に対応させる。対応する骨が無い共通の名前は missing に入れ、
    残りの骨だけを回す（そのポーズは「足りない」ポーズ。確かめは灰にする）。
    センター の位置の動き = 休みの回転⁻¹ @ (center_offset_legs × leg_length)（体の向きの軸を、骨のローカルにする）。
    center_offset_legs が 0 でないのに leg_length が無い（None・0 以下）ときは ValueError。
    """
    if not isinstance(pose, PoseSpec):
        raise ValueError(f"pose は PoseSpec にしてください（いまは {type(pose).__name__}）")
    names = _as_names(bone_names, "bone_names")
    matrices = _as_matrices(rest_matrices, "rest_matrices", len(names))
    lookup = {name: index for index, name in enumerate(names)}
    by_canonical = bones.map_bones(names).by_canonical
    missing = [bone for bone in pose.bones if bone not in by_canonical]
    rotations: dict[str, tuple[float, float, float, float]] = {}
    axis_rotations: dict[str, tuple[float, float, float, float]] = {}
    canonical: dict[str, str] = {}
    for bone, q_axis in pose.axis_rotations().items():
        actual = by_canonical.get(bone)
        if actual is None:
            continue
        rest = _rotation_part(matrices[lookup[actual], :3, :3])
        q_rest = _matrix_quaternion(rest)
        q = quat_multiply(_conjugate(q_rest), quat_multiply(q_axis, q_rest))
        q = q / np.linalg.norm(q)
        q = -q if q[0] < 0.0 else q
        rotations[actual] = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        axis_rotations[actual] = tuple(float(v) for v in q_axis)  # type: ignore[assignment]
        canonical[actual] = bone
    center = by_canonical.get(CENTER_BONE)
    location = (0.0, 0.0, 0.0)
    offset = np.asarray(pose.center_offset_legs, dtype=np.float64)
    if np.any(offset != 0.0) and leg_length is None:
        # 脚の長さを測る骨が無いときは、センター を動かせない（足りない骨として数える）
        absent = _absent_leg_bones(by_canonical)
        if not absent:
            raise ValueError(f"{pose.name}：センター を動かすには、脚の長さ（leg_length）を渡してください")
        missing.extend(bone for bone in absent if bone not in missing)
    elif np.any(offset != 0.0):
        if isinstance(leg_length, bool) or not isinstance(leg_length, (int, float, np.integer, np.floating)) \
                or not math.isfinite(float(leg_length)) or float(leg_length) <= 0.0:
            raise ValueError(f"{pose.name}：脚の長さ（leg_length）は正の数にしてください（いまは {leg_length!r}）")
        if center is not None:
            rest = _rotation_part(matrices[lookup[center], :3, :3])
            local = rest.T @ (offset * float(leg_length))
            location = (float(local[0]), float(local[1]), float(local[2]))
    if center is not None:
        canonical.setdefault(center, CENTER_BONE)
    return ResolvedPose(pose=pose.name, rotations=rotations, axis_rotations=axis_rotations, canonical=canonical,
                        center=center, center_location=location, missing=tuple(missing))


def _leg_chains(side: str) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    """脚の長さを測る骨の並び（見える骨、D の骨の順）。side は "左" か "右" """
    return (f"{side}足", f"{side}ひざ", f"{side}足首"), (f"{side}足D", f"{side}ひざD", f"{side}足首D")


def _absent_leg_bones(by_canonical: Mapping[str, str]) -> list[str]:
    """脚の長さを測れない側の、足りない骨の共通の名前（見える骨の並びで書く）"""
    absent: list[str] = []
    for side in ("左", "右"):
        chains = _leg_chains(side)
        if not any(all(bone in by_canonical for bone in chain) for chain in chains):
            absent.extend(bone for bone in chains[0] if bone not in by_canonical)
    return absent


def leg_length_from_bones(bone_names: Sequence[str], heads: Any) -> float | None:
    """脚の長さ（足 → ひざ → 足首 の根元をつないだ長さの、左右の平均。無ければ D の骨で測る）。測れないときは None。

    heads は bone_names の順の骨の根元 (B, 3)（アーマチュアの中の座標なら、アーマチュアの中の長さになる）
    """
    names = _as_names(bone_names, "bone_names")
    points = as_array(heads, "heads")
    if points.dtype.kind not in "iuf" or points.shape != (len(names), 3):
        raise ValueError(f"heads は ({len(names)}, 3) の並びにしてください（いまの形：{points.shape}）")
    points = np.asarray(points, dtype=np.float64)
    lookup = {name: index for index, name in enumerate(names)}
    by_canonical = bones.map_bones(names).by_canonical
    lengths = []
    for side in ("左", "右"):
        for chain in _leg_chains(side):
            actual = [by_canonical.get(bone) for bone in chain]
            if None in actual:
                continue
            p = points[[lookup[name] for name in actual]]
            length = float(np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[1]))
            if math.isfinite(length) and length > 0.0:
                lengths.append(length)
                break
    return float(np.mean(lengths)) if lengths else None


# MARK: - 決めた向きになったか


def find_overridden(resolved: ResolvedPose, bone_names: Sequence[str], parents: Any, deltas: Any, *,
                    tolerance_deg: float = POSE_FOLLOW_TOLERANCE_DEG) -> tuple[str, ...]:
    """ポーズを付けたあとの骨の向きが、決めた向きにならなかった骨の共通の名前（ポーズの骨の順のあと、D の骨）。

    - deltas：骨の、休みの向きからの回転 (B, 4)（rotation_deltas。アーマチュアの中の向き）。並びは bone_names。
    - 回した骨 b の決めた向き = deltas[親] * q_axis(b)（親が無ければ q_axis(b)）。tolerance_deg より離れていれば入れる
      （モデルの拘束がポーズを変えた）。
    - D の骨（足D・ひざD・足首D）は、元の骨（足・ひざ・足首）と同じ向きになるはず。離れていれば入れる
      （D の骨が付いて来ず、脚の形が動かない）。
    """
    names = _as_names(bone_names, "bone_names")
    parent_index = as_array(parents, "parents")
    if parent_index.dtype.kind not in "iu" or parent_index.shape != (len(names),):
        raise ValueError(f"parents は ({len(names)},) の整数の並びにしてください")
    q = as_array(deltas, "deltas")
    if q.dtype.kind not in "iuf" or q.shape != (len(names), 4):
        raise ValueError(f"deltas は ({len(names)}, 4) の並びにしてください（いまの形：{q.shape}）")
    q = np.asarray(q, dtype=np.float64)
    lookup = {name: index for index, name in enumerate(names)}
    found: list[str] = []
    for actual, q_axis in resolved.axis_rotations.items():
        index = lookup.get(actual)
        if index is None:
            raise ValueError(f"骨 {actual!r} が bone_names にありません")
        parent = int(parent_index[index])
        intended = np.asarray(q_axis) if parent < 0 else quat_multiply(q[parent], q_axis)
        if float(quat_angle_deg(intended, q[index])) > tolerance_deg:
            found.append(resolved.canonical[actual])
    by_canonical = bones.map_bones(names).by_canonical
    for d_bone, source in D_BONE_SOURCES.items():
        d_actual, source_actual = by_canonical.get(d_bone), by_canonical.get(source)
        if d_actual is None or source_actual is None:
            continue
        if float(quat_angle_deg(q[lookup[d_actual]], q[lookup[source_actual]])) > tolerance_deg:
            found.append(d_bone)
    return tuple(dict.fromkeys(found))


@dataclass(frozen=True)
class AppliedPose:
    """ポーズをモデルに付けた結果（poses_bpy.apply_pose が返す）。

    pose: ポーズの名前
    posed: 回した骨（共通の名前 → モデルの骨の名前）
    missing: モデルに無かった共通の名前（あれば「足りない」ポーズ）
    overridden: 付けたあと、決めた向きにならなかった骨の共通の名前（モデルの拘束がポーズを変えた・D の骨が付いて来ない）
    center_location: センター に入れた位置の動き（骨のローカル。床に合わせた分を含む。メートル）
    ground_shift: 床に合わせるために、世界の上下に動かした長さ（メートル。合わせなかったときは None）
    grounded_by: 床に合わせた物（"mesh"：メッシュの頂点、"bones"：形を動かす骨の根元と先。合わせなかったときは None）
    leg_length: センター を動かすのに使った脚の長さ（アーマチュアの中。使わなかったときは None）
    """

    pose: str
    posed: dict[str, str] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    overridden: tuple[str, ...] = ()
    center_location: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ground_shift: float | None = None
    grounded_by: str | None = None
    leg_length: float | None = None

    @property
    def partial(self) -> bool:
        """決めたとおりのポーズになっていない（骨が足りない、または決めた向きにならなかった）"""
        return bool(self.missing or self.overridden)

"""A4 関節と足：動きのフレームを通して、関節と足の困った所を見つける（bpy なし・numpy だけ）。

見つける物と、使う関数の流れ：
- ひざ・ひじの逆曲がり：knee_bend・hinge_bend → reverse_bend_frames → summarize_reverse_bend
- 足の床への沈み：sink_depths → summarize_sink
- 足すべり（床に着いている間に、足が横にずれる）：foot_contacts → slide_distances → summarize_slide
- 足が届かないフレーム（IK の目標が、脚の長さより遠い）：unreachable_frames → summarize_unreachable
- 骨が1フレームで跳ぶ：bone_jump_frames → summarize_jumps

決まり（CLAUDE.md 7章）：
- 長さはメートル、角度は度、時間は秒。
- 向きは Blender のまま：Z が上、モデルは -Y を向く（前は -Y、後ろは +Y）、キャラの左は +X。
- 動きの位置は (F, J, 3)（フレーム数, 関節の数, xyz）、関節1つなら (F, 3) の float64。
- 回転（クォータニオン）は Blender の順 (w, x, y, z)。
- details のフレームの番号は、並びの番号（0 から）。シーンのフレーム番号にするときは、呼ぶ側で始まりのフレームを足す。
- 区間（span）は、両端を含む (始まり, 終わり) の組。
- 角度の関数は、向きが決まらないとき（長さがほぼ 0 のベクトルなど）に NaN を返す。NaN は「測れなかった」で、
  まとめ（summarize_*）では灰の理由になる（測れなかった物は緑にしない）。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import numpy as np

from .. import thresholds
from .geometry import as_array, as_positive_int
from .results import Measurement, Signal

# MARK: - 計算のための数字（判定の基準ではない）

# これより短いベクトル（メートル）は、向きが決まらないとみなす（角度は NaN）。
# Blender の座標は float32（約7桁）なので、1m あたり約 0.1µm の誤差がある。それより十分大きく、本物の骨の長さ（1mm 以上）より十分小さい
MIN_DIRECTION_LENGTH = 1e-6
# 軸に垂直な成分が、ベクトルの長さのこの割合以下なら、ベクトルは軸とほぼ同じ向きで、軸まわりの向きが決まらないとみなす（角度は NaN）
PARALLEL_SINE_EPS = 1e-6
# 長さがこれ以下のクォータニオンは、向きを表せないとみなす（ValueError）
QUATERNION_MIN_NORM = 1e-9
# 骨の跳びを調べるとき、一度に計算するクォータニオンの数。1つあたり約 300 バイトなので、約 20MB になる
STEP_CHUNK_QUATERNIONS = 1 << 16

# MARK: - 結果の名前

REVERSE_BEND_NAME = "reverse_bend"
FLOOR_SINK_NAME = "floor_sink"
FOOT_SLIDE_NAME = "foot_slide"
UNREACHABLE_NAME = "unreachable_frames"
BONE_JUMPS_NAME = "bone_jumps"


# MARK: - 入力の確かめ

def _check_real_dtype(array: np.ndarray, name: str) -> None:
    """数（整数・小数）の並びか。真偽・文字・複素数・物（object）は受け付けない"""
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} は数の並びにしてください（いまの型：{array.dtype}）")


def _as_vectors(values: Any, name: str, width: int = 3) -> np.ndarray:
    """最後の軸の長さが width の float64 の並び (..., width) にする。

    NaN は通す（測れない値として、答えも NaN になる）。無限大は ValueError。空の並び（[]）は (0, width) にする。
    """
    array = as_array(values, name)
    _check_real_dtype(array, name)
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 1 and array.size == 0:
        array = array.reshape(0, width)
    if array.ndim == 0 or array.shape[-1] != width:
        raise ValueError(f"{name} は最後の軸の長さが {width} の並び (..., {width}) にしてください（いまの形：{array.shape}）")
    if np.isinf(array).any():
        raise ValueError(f"{name} に無限大が入っています")
    return array


def _as_track(values: Any, name: str) -> np.ndarray:
    """1つの点のフレームごとの位置 (F, 3) の float64 にする。NaN・無限大は ValueError"""
    array = _as_vectors(values, name)
    if array.ndim != 2:
        raise ValueError(f"{name} は (F, 3)（1つの点の、フレームごとの位置）にしてください（いまの形：{array.shape}）")
    if np.isnan(array).any():
        raise ValueError(f"{name} に NaN が入っています")
    return array


def _as_angles(values: Any, name: str) -> np.ndarray:
    """フレームごとの角度 (F,) の float64 にする。NaN は「測れなかったフレーム」として通す。無限大は ValueError"""
    array = as_array(values, name)
    _check_real_dtype(array, name)
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} は1次元の並び (F,) にしてください（いまの形：{array.shape}）")
    if np.isinf(array).any():
        raise ValueError(f"{name} に無限大が入っています")
    return array


def _as_mask(mask: Any, name: str, ndim: int = 1) -> np.ndarray:
    """真偽（bool）の並びにする。0/1 の整数の並びは、番号の並びとまちがえやすいので受け付けない（空の並びはよい）"""
    array = as_array(mask, name)
    if array.dtype.kind != "b":
        if array.size == 0 and array.dtype.kind in "iuf":
            array = array.astype(bool)
        else:
            raise ValueError(f"{name} は真偽（bool）の並びにしてください（いまの型：{array.dtype}）")
    if array.ndim != ndim:
        raise ValueError(f"{name} は {ndim} 次元の真偽の並びにしてください（いまの形：{array.shape}）")
    return array


def _as_real(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    """有限の数（float）にする。真偽（True・False）は数として受け付けない。0 次元の numpy の並びは、中の数として扱う"""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value[()]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} は数にしてください（いまの値：{value!r}）")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} は有限の数にしてください（いまの値：{value!r}）")
    if positive and number <= 0.0:
        raise ValueError(f"{name} は 0 より大きい数にしてください（いまの値：{value!r}）")
    if nonnegative and number < 0.0:
        raise ValueError(f"{name} は 0 以上の数にしてください（いまの値：{value!r}）")
    return number


def _broadcast(names: str, *arrays: np.ndarray) -> tuple[int, ...]:
    """並びの形が numpy の決まりで合わせられるか確かめる。合わないときは ValueError"""
    try:
        return np.broadcast_shapes(*(array.shape for array in arrays))
    except ValueError:
        shapes = "、".join(str(array.shape) for array in arrays)
        raise ValueError(f"{names} の形が合いません（いまの形：{shapes}）") from None


# MARK: - ベクトルの計算

def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """最後の軸（x, y, z）の内積。足す順を決めておく"""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _norm(v: np.ndarray) -> np.ndarray:
    """最後の軸の長さ。hypot を使い、とても大きい・小さい値でもあふれないようにする"""
    return np.hypot(np.hypot(v[..., 0], v[..., 1]), v[..., 2])


def _unit(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """長さ 1 にしたベクトルと、向きが決まるかの印（長さが MIN_DIRECTION_LENGTH より大きい。NaN は False）"""
    length = _norm(v)
    ok = length > MIN_DIRECTION_LENGTH
    return v / np.where(ok, length, 1.0)[..., None], ok


# MARK: - 角度

def angle_at(a: Any, b: Any, c: Any) -> np.ndarray:
    """b での角度（度、0〜180）：(a − b) と (c − b) の間の角度。

    - a・b・c は (..., 3)。形は numpy の決まり（broadcast）で合わせる。答えの形は、合わせた形から最後の軸を除いた物。
    - a か c が b とほぼ同じ場所（MIN_DIRECTION_LENGTH 以下）なら、角度は決まらないので NaN。入力の NaN も NaN になる。
    - 小さい角度でも正しく出るように、acos ではなく atan2（外積の長さ, 内積）で計算する。
    """
    pa, pb, pc = _as_vectors(a, "a"), _as_vectors(b, "b"), _as_vectors(c, "c")
    _broadcast("a・b・c", pa, pb, pc)
    u, u_ok = _unit(pa - pb)
    v, v_ok = _unit(pc - pb)
    angle = np.degrees(np.arctan2(_norm(np.cross(u, v)), _dot(u, v)))
    return np.where(u_ok & v_ok, angle, np.nan)


def _signed_angle(u: np.ndarray, v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """signed_angle の計算（入力は確かめずみ）"""
    n, n_ok = _unit(axis)
    uu, u_ok = _unit(u)
    vv, v_ok = _unit(v)
    # 軸に垂直な成分だけを使う
    up = uu - _dot(uu, n)[..., None] * n
    vp = vv - _dot(vv, n)[..., None] * n
    ok = n_ok & u_ok & v_ok & (_norm(up) > PARALLEL_SINE_EPS) & (_norm(vp) > PARALLEL_SINE_EPS)
    angle = np.degrees(np.arctan2(_dot(np.cross(up, vp), n), _dot(up, vp)))
    # atan2 の答えは [-180, 180]。-180 は 180 にそろえ、(-180, 180] にする
    angle = np.where(angle <= -180.0, angle + 360.0, angle)
    angle = np.minimum(angle, 180.0)
    return np.where(ok, angle, np.nan)


def signed_angle(u: Any, v: Any, axis: Any) -> np.ndarray:
    """u から v への、axis まわりの回転の角度（度、-180 より大きく 180 以下）。

    - 右手の決まり：右手の親指を axis に向けたとき、ほかの指が曲がる向きに回るのが正。
      例：u = +X、v = +Y、axis = +Z なら +90。axis = -Z なら -90。
    - u と v は、axis に垂直な成分だけを使う（axis に沿った成分は無視する）。axis の長さは何でもよい。
    - 形は numpy の決まり（broadcast）で合わせる。答えの形は、合わせた形から最後の軸を除いた物。
    - 向きが決まらないときは NaN：axis・u・v の長さがほぼ 0、u か v が axis とほぼ同じ向き（垂直な成分がほぼ 0）、入力に NaN。
    - ちょうど反対向き（180 度）は、向きが決まらないが、+180 とする。
    """
    pu, pv, pa = _as_vectors(u, "u"), _as_vectors(v, "v"), _as_vectors(axis, "axis")
    _broadcast("u・v・axis", pu, pv, pa)
    return _signed_angle(pu, pv, pa)


# MARK: - ひざ・ひじの逆曲がり

def _hinge_bend(upper: np.ndarray, middle: np.ndarray, lower: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """hinge_bend の計算（入力は確かめずみ）。上の骨の向きを延ばした向きから、下の骨の向きへの、axis まわりの角度"""
    return _signed_angle(middle - upper, lower - middle, axis)


def hinge_bend(upper: Any, middle: Any, lower: Any, axis: Any) -> np.ndarray:
    """ちょうつがいの関節（ひざ・ひじ）の、符号つきの曲がり（度、-180 より大きく 180 以下）。

    - upper → middle が上の骨（太もも・上腕）、middle → lower が下の骨（すね・前腕）。middle が関節。
    - 0 は、まっすぐに伸びている。
    - 正は、axis が決める「ふつうの向き」に曲がっている。右手の決まり：下の骨が、上の骨を延ばした向きから
      +axis まわりに回っていれば正（signed_angle(middle − upper, lower − middle, axis) と同じ）。
    - 負は、逆に曲がっている（逆曲がり）。
    - 骨は、axis に垂直な面に映して測る。骨がその面から外れている（関節が横を向いている）と、3次元で測った骨の間の角度とは
      違う値になる（小さくも大きくも出る。例：曲がりが 90 度より大きいときは、映した角度のほうが大きく出ることがある）。
      上の骨か下の骨が axis とほぼ同じ向きのとき、関節が重なっているときは NaN。
    - ちょうど折りたたんだ形（180 度）は +180 とする。
    - 形は numpy の決まり（broadcast）で合わせる。例：upper・middle・lower が (F, 3)、axis が (3,) か (F, 3)。

    ひじの例：T ポーズの左腕（肩 → ひじが +X）で、前腕を前（-Y）に曲げるのがふつうなら、axis = -Z。
    右腕（-X 向き）なら axis = +Z。上腕が回ると軸も回るので、ひじには上腕の骨の向きから作った軸を渡す。
    """
    pu = _as_vectors(upper, "upper")
    pm = _as_vectors(middle, "middle")
    pl = _as_vectors(lower, "lower")
    pa = _as_vectors(axis, "axis")
    _broadcast("upper・middle・lower・axis", pu, pm, pl, pa)
    return _hinge_bend(pu, pm, pl, pa)


def knee_bend(hip: Any, knee: Any, ankle: Any, lateral: Any) -> np.ndarray:
    """ひざの、符号つきの曲がり（度）。ふつうの曲がりが正、逆曲がりが負、まっすぐが 0。

    - hip・knee・ankle：足・ひざ・足首の位置（MMD 準標準の 左足・左ひざ・左足首 など）。(F, 3) か (3,)。
    - lateral：キャラの左を向くベクトル（例：左足の位置 − 右足の位置）。フレームごと (F, 3) か、全部に同じ (3,)。
      長さは何でもよい（向きだけを使う）。左脚・右脚のどちらにも、同じ「左向き」を渡す。
    - ふつうの曲がり：太ももが下を向いているとき、足首が後ろ（モデルは -Y を向くので +Y）に動く向き。
      これは lateral（+X）まわりの右手の回りなので、hinge_bend(hip, knee, ankle, axis=lateral) が正になる。
      例：腰 (0, 0, 1)、ひざ (0, 0, 0.5)、足首 (0, 0.5, 0.5)（すねが後ろへ水平）なら +90。足首が (0, -0.5, 0.5)（前）なら -90。
    - 体全体が向きを変えても、lateral も一緒に回っていれば、答えは変わらない。
    - 太ももを強くひねる・横に開く姿勢では、腰の左右から作った軸と本当のひざの軸がずれる。
      太ももが lateral に近い向き（脚を横に大きく開く・真横に上げる）ほど、映した太ももが短くなり、符号もあてにならない。
      例：太ももが真横（+X）から 5 度後ろ（+Y）を向き、すねが真下（ふつうに 90 度曲げた形）だと -90（逆曲がりに見える）。
      NaN（測れない）になるのは、太ももが lateral とほぼ完全に同じ向き（PARALLEL_SINE_EPS 以下）のときだけ。
    """
    hp = _as_vectors(hip, "hip")
    kn = _as_vectors(knee, "knee")
    an = _as_vectors(ankle, "ankle")
    la = _as_vectors(lateral, "lateral")
    _broadcast("hip・knee・ankle・lateral", hp, kn, an, la)
    return _hinge_bend(hp, kn, an, la)


def reverse_bend_frames(bend_deg: Any, *, tolerance_deg: float = thresholds.REVERSE_BEND_TOLERANCE_DEG) -> np.ndarray:
    """逆に曲がったフレームの印 (F,) bool：曲がりが -tolerance_deg より小さい（ちょうど -tolerance_deg は数えない）。

    NaN（測れなかったフレーム）は False にする。summarize_reverse_bend では「測れなかった」として灰の理由に数える。
    """
    bend = _as_angles(bend_deg, "bend_deg")
    tolerance = _as_real(tolerance_deg, "tolerance_deg", nonnegative=True)
    measured = ~np.isnan(bend)
    return measured & (np.where(measured, bend, 0.0) < -tolerance)


# MARK: - 足の床への沈み

def sink_depths(points: Any, *, floor_z: float = 0.0) -> np.ndarray:
    """床に沈んだ深さ（メートル）：max(0, floor_z − z)。床より上なら 0。

    points は (..., 3)（例：(F, 3) や (F, J, 3)）。答えは最後の軸を除いた形。NaN の点は NaN（測れなかった）。
    """
    p = _as_vectors(points, "points")
    floor = _as_real(floor_z, "floor_z")
    below = floor - p[..., 2]
    return np.where(below > 0.0, below, np.where(np.isnan(below), np.nan, 0.0))


# MARK: - 足の接地と足すべり

def track_speeds(track: Any, *, fps: float) -> np.ndarray:
    """1つの点の、フレームごとの速さ (F,)（メートル毎秒、3次元）。

    フレームの間の差は中央の差（np.gradient。両端は片側の差）に fps を掛けて、毎秒にする。
    フレームが1つなら速さ 0、フレームが無ければ空。
    """
    p = _as_track(track, "track")
    rate = _as_real(fps, "fps", positive=True)
    if len(p) < 2:
        return np.zeros(len(p))
    return _norm(np.gradient(p, axis=0) * rate)


def foot_contacts(foot: Any, *, fps: float, floor_z: float = 0.0, height: float = thresholds.FOOT_CONTACT_HEIGHT,
                  speed: float = thresholds.FOOT_CONTACT_SPEED) -> np.ndarray:
    """足が床に着いているフレームの印 (F,) bool：床からの高さが height 以下、かつ速さが speed 以下。

    - foot：足の点のフレームごとの位置 (F, 3)。足の裏に近い点（つま先・かかとなど）を渡す。
      足首の骨の位置は床より高い（数 cm）ので、そのまま渡すときは floor_z をその高さに合わせる。
    - 床より下（高さが負）も「着いている」に数える（速さの条件を満たせば）。
    - 速さは track_speeds（中央の差）。足を上げる直前・下ろした直後のフレームは、となりのフレームの動きで速さが出るので、
      着いていないことがある。
    - NaN の位置は ValueError。
    """
    p = _as_track(foot, "foot")
    floor = _as_real(floor_z, "floor_z")
    max_height = _as_real(height, "height", nonnegative=True)
    max_speed = _as_real(speed, "speed", nonnegative=True)
    speeds = track_speeds(p, fps=fps)
    return ((p[:, 2] - floor) <= max_height) & (speeds <= max_speed)


def spans(mask: Any) -> list[tuple[int, int]]:
    """True が続く区間の一覧。両端を含む (始まり, 終わり) の組を、前から順に返す。

    例：[True, True, False, True] → [(0, 1), (3, 3)]。全部 False・空なら []。
    mask は1次元の真偽（bool）の並び。0/1 の整数の並びは ValueError。
    """
    m = _as_mask(mask, "mask")
    if m.size == 0:
        return []
    edges = np.diff(np.concatenate(([False], m, [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def slide_distances(foot: Any, contacts: Any) -> list[tuple[int, int, float]]:
    """床に着いている区間ごとの、足すべりの距離。(始まり, 終わり, 距離) の一覧を、前から順に返す。

    距離は、区間の始まりのフレームの位置から、区間の間にいちばん離れた水平（XY）の距離（メートル）。
    上下（Z）の動きは数えない。1フレームだけの区間は 0。
    foot は (F, 3)、contacts は (F,) bool（foot_contacts の答えなど）。
    """
    p = _as_track(foot, "foot")
    m = _as_mask(contacts, "contacts")
    if len(m) != len(p):
        raise ValueError(f"contacts の長さ（{len(m)}）が foot のフレーム数（{len(p)}）と違います")
    runs = spans(m)
    if not runs:
        return []
    starts = np.array([start for start, _ in runs], dtype=np.int64)
    ends = np.array([end for _, end in runs], dtype=np.int64)
    lengths = ends - starts + 1
    frames = np.flatnonzero(m)  # 区間を前から順につないだ物と同じ並び
    offsets = p[frames, :2] - p[np.repeat(starts, lengths), :2]
    distances = np.hypot(offsets[:, 0], offsets[:, 1])
    first = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    maxima = np.maximum.reduceat(distances, first)
    return [(int(start), int(end), float(distance)) for start, end, distance in zip(starts, ends, maxima)]


# MARK: - 足が届かないフレーム

def _as_chain_length(chain_length: Any, frame_count: int) -> np.ndarray | float:
    """鎖の長さ：0 より大きい有限の数か、フレームごとの (F,) の並び"""
    array = as_array(chain_length, "chain_length")
    if array.ndim == 0:
        return _as_real(chain_length, "chain_length", positive=True)
    _check_real_dtype(array, "chain_length")
    array = np.asarray(array, dtype=np.float64)
    if array.shape != (frame_count,):
        raise ValueError(f"chain_length は数か ({frame_count},) の並びにしてください（いまの形：{array.shape}）")
    if not np.isfinite(array).all() or (array <= 0.0).any():
        raise ValueError("chain_length は 0 より大きい有限の数にしてください")
    return array


def _distances(root: Any, target: Any) -> np.ndarray:
    r = _as_track(root, "root")
    t = _as_track(target, "target")
    if r.shape != t.shape:
        raise ValueError(f"root と target の形が違います（いまの形：{r.shape}、{t.shape}）")
    return _norm(t - r)


def reach_ratios(root: Any, target: Any, chain_length: Any) -> np.ndarray:
    """目標までの距離が、鎖の長さの何倍か (F,)。1 より大きいと、まっすぐ伸ばしても届かない。

    root：鎖の付け根（脚なら 足 の骨の位置）(F, 3)。target：目標（足ＩＫ の位置）(F, 3)。
    chain_length：鎖の長さ（太もも + すね）。数か、フレームごとの (F,)。0 以下は ValueError。
    """
    distances = _distances(root, target)
    return distances / _as_chain_length(chain_length, len(distances))


def unreachable_frames(root: Any, target: Any, chain_length: Any, *,
                       tolerance: float = thresholds.IK_REACH_TOLERANCE) -> np.ndarray:
    """目標に届かないフレームの印 (F,) bool：|target − root| > chain_length × (1 + tolerance)（ちょうどは届く）。

    引数は reach_ratios と同じ。tolerance は割合（0.01 なら 1% まで超えてよい）。
    """
    distances = _distances(root, target)
    length = _as_chain_length(chain_length, len(distances))
    tol = _as_real(tolerance, "tolerance", nonnegative=True)
    return distances > length * (1.0 + tol)


# MARK: - 回転（クォータニオン）と骨の跳び

def _as_quaternions(q: Any, name: str) -> np.ndarray:
    """クォータニオンの並び (..., 4) の float64 にする（順は Blender と同じ w, x, y, z）。NaN・無限大は ValueError"""
    array = as_array(q, name)
    _check_real_dtype(array, name)
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != 4:
        raise ValueError(f"{name} は最後の軸の長さが 4（w, x, y, z）の並びにしてください（いまの形：{array.shape}）")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} に NaN か無限大が入っています")
    return array


def _quat_norms(q: np.ndarray, name: str) -> np.ndarray:
    """クォータニオンの長さ。長さがほぼ 0 の物があれば ValueError"""
    norms = np.hypot(np.hypot(q[..., 0], q[..., 1]), np.hypot(q[..., 2], q[..., 3]))
    if (norms <= QUATERNION_MIN_NORM).any():
        raise ValueError(f"{name} に長さがほぼ 0 のクォータニオンがあります（向きを表せません）")
    return norms


def quat_normalize(q: Any) -> np.ndarray:
    """クォータニオンを長さ 1 にする（順は w, x, y, z）。符号は変えない（q と -q は同じ向き）。長さがほぼ 0 なら ValueError"""
    array = _as_quaternions(q, "q")
    return array / _quat_norms(array, "q")[..., None]


def _relative_angle_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """長さ 1 のクォータニオンどうしの、向きの差の角度（度、0〜180）。

    r = q1⁻¹ q2 の回転の角度を 2·atan2(|r のベクトル部|, |r のスカラー部|) で出す。
    スカラー部の絶対値を使うので q と -q は同じ向きになり、acos と違って小さい角度でも正しい。
    """
    w1, v1 = q1[..., 0], q1[..., 1:]
    w2, v2 = q2[..., 0], q2[..., 1:]
    # 共役 (w1, -v1) と (w2, v2) の積
    scalar = w1 * w2 + _dot(v1, v2)
    vector = w1[..., None] * v2 - w2[..., None] * v1 - np.cross(v1, v2)
    return np.degrees(2.0 * np.arctan2(_norm(vector), np.abs(scalar)))


def quat_angle_deg(q1: Any, q2: Any) -> np.ndarray:
    """2つの向き（クォータニオン、w, x, y, z）の間の回転の角度（度、0〜180）。

    - q と -q は同じ向きとして扱う（差は 0）。長さ 1 でなくてもよい（中で長さ 1 にする）。
    - 形は numpy の決まり（broadcast）で合わせる。答えの形は、合わせた形から最後の軸を除いた物。
    - 長さがほぼ 0、NaN・無限大は ValueError。
    """
    a = _as_quaternions(q1, "q1")
    b = _as_quaternions(q2, "q2")
    _broadcast("q1・q2", a, b)
    a = a / _quat_norms(a, "q1")[..., None]
    b = b / _quat_norms(b, "q2")[..., None]
    return _relative_angle_deg(a, b)


def _frames_per_chunk(bone_count: int, chunk_size: int) -> int:
    """小分け1つに入れるフレームの数（クォータニオンが chunk_size 個ほどになるように。少なくとも1）"""
    return max(1, chunk_size // max(bone_count, 1))


def _as_rotation_track(rotations: Any, chunk_size: int) -> np.ndarray:
    """骨ごとの、フレームごとの回転 (F, B, 4) を確かめる。

    大きな並びの写しを作らないように、型は変えずに返す（float64 にするのは、小分けごとに行う）。
    NaN・無限大、長さがほぼ 0 のクォータニオンは ValueError（フレームが1つでも確かめる）。
    """
    array = as_array(rotations, "rotations")
    _check_real_dtype(array, "rotations")
    if array.ndim != 3 or array.shape[-1] != 4:
        raise ValueError(f"rotations は (F, B, 4)（フレーム数, 骨の数, w x y z）にしてください（いまの形：{array.shape}）")
    step = _frames_per_chunk(array.shape[1], chunk_size)
    for f0 in range(0, len(array), step):
        block = np.asarray(array[f0:f0 + step], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("rotations に NaN か無限大が入っています")
        _quat_norms(block, "rotations")
    return array


def _step_angle_chunks(r: np.ndarray, chunk_size: int) -> Iterator[tuple[int, int, np.ndarray]]:
    """となりのフレームとの角度を、フレームの小分けごとに出す：(始まり, 終わり（含まない）, 角度 (終わり − 始まり, B))"""
    frame_count, bone_count = r.shape[:2]
    step = _frames_per_chunk(bone_count, chunk_size)
    for f0 in range(0, frame_count - 1, step):
        f1 = min(frame_count - 1, f0 + step)
        block = np.asarray(r[f0:f1 + 1], dtype=np.float64)
        block = block / _quat_norms(block, "rotations")[..., None]
        yield f0, f1, _relative_angle_deg(block[:-1], block[1:])


def step_angles_deg(rotations: Any, *, chunk_size: int = STEP_CHUNK_QUATERNIONS) -> np.ndarray:
    """フレーム i と i+1 の間に、骨が回った角度 (F−1, B)（度、0〜180）。

    rotations は (F, B, 4)（フレーム数, 骨の数, w x y z）。フレームが1つ以下なら (0, B)。
    メモリを抑えるため、chunk_size 個のクォータニオンずつ計算する。
    """
    chunk = as_positive_int(chunk_size, "chunk_size")
    r = _as_rotation_track(rotations, chunk)
    result = np.zeros((max(len(r) - 1, 0), r.shape[1]))
    for f0, f1, angles in _step_angle_chunks(r, chunk):
        result[f0:f1] = angles
    return result


def bone_jump_frames(rotations: Any, *, threshold_deg: float = thresholds.BONE_JUMP_DEG,
                     chunk_size: int = STEP_CHUNK_QUATERNIONS) -> np.ndarray:
    """骨が1フレームで跳んだ印 (F−1, B) bool：フレーム i と i+1 の間の角度が threshold_deg より大きい（ちょうどは数えない）。

    - rotations は (F, B, 4)。骨の向きは、アーマチュアの中の向き（親の回転を含む）でも、骨ごとの回転（親に対する）でもよい。
      親を含む向きでは、親が跳ぶと子も跳んだと数える。骨ごとの回転なら、跳んだ骨だけが出る。
    - q と -q は同じ向きなので、符号が入れ替わっただけでは跳びにならない。
    """
    limit = _as_real(threshold_deg, "threshold_deg", nonnegative=True)
    chunk = as_positive_int(chunk_size, "chunk_size")
    r = _as_rotation_track(rotations, chunk)
    result = np.zeros((max(len(r) - 1, 0), r.shape[1]), dtype=bool)
    for f0, f1, angles in _step_angle_chunks(r, chunk):
        result[f0:f1] = angles > limit
    return result


# MARK: - まとめ（信号）

def summarize_reverse_bend(bend_deg: Any, *,
                           tolerance_deg: float = thresholds.REVERSE_BEND_TOLERANCE_DEG) -> Measurement:
    """ひざ・ひじの逆曲がりを、1つの信号にまとめる（名前 "reverse_bend"、単位 "deg"）。

    - bend_deg は knee_bend・hinge_bend の答え (F,)（1つの関節）。NaN は「測れなかったフレーム」。
    - 値は、いちばん深い逆曲がりの角度（いちばん負の曲がりの符号を変えた物）。逆に曲がったフレームが無ければ 0。
    - 赤：tolerance_deg を超えて逆に曲がったフレームが1つでもある（reverse_bend_frames と同じ判定）。
    - 灰：フレームが無い、全部測れなかった、または赤でなく、測れなかったフレームがある（そこに逆曲がりが隠れているかもしれない）。
    - details："reverse_spans"（逆曲がりの区間。全部）、"unmeasured_spans"（測れなかった区間）、数、
      "worst_frame"（いちばん負のフレーム。同じなら前のフレーム）、"min_bend_deg"。
    """
    bend = _as_angles(bend_deg, "bend_deg")
    tolerance = _as_real(tolerance_deg, "tolerance_deg", nonnegative=True)
    if len(bend) == 0:
        return Measurement.unmeasured(REVERSE_BEND_NAME, "deg", "フレームがありません")
    measured = ~np.isnan(bend)
    reverse = reverse_bend_frames(bend, tolerance_deg=tolerance)
    details: dict[str, Any] = {
        "frame_count": len(bend),
        "tolerance_deg": tolerance,
        "reverse_count": int(reverse.sum()),
        "reverse_spans": spans(reverse),
        "unmeasured_count": int((~measured).sum()),
        "unmeasured_spans": spans(~measured),
    }
    if not measured.any():
        return Measurement.unmeasured(REVERSE_BEND_NAME, "deg",
                                      "曲がりを測れたフレームがありません（関節が重なっているか、骨が軸の向きを向いています）",
                                      details)
    worst_frame = int(np.argmin(np.where(measured, bend, np.inf)))
    min_bend = float(bend[worst_frame])
    details["worst_frame"] = worst_frame
    details["min_bend_deg"] = min_bend
    value = max(0.0, -min_bend)
    unmeasured = details["unmeasured_count"] > 0
    if details["reverse_count"]:
        reason = "逆に曲がったフレームがあります"
        if unmeasured:
            reason += "（曲がりを測れなかったフレームもあります）"
        return Measurement(REVERSE_BEND_NAME, value, "deg", Signal.RED, reason, details)
    if unmeasured:
        return Measurement(REVERSE_BEND_NAME, value, "deg", Signal.GRAY,
                           "曲がりを測れなかったフレームがあります（そこに逆曲がりが隠れているかもしれません）", details)
    return Measurement(REVERSE_BEND_NAME, value, "deg", Signal.GREEN, "", details)


def _as_depths(depths: Any) -> np.ndarray:
    """沈みの深さ (F,) か (F, J) にする。NaN は通す。負の値・無限大は ValueError"""
    array = as_array(depths, "depths")
    _check_real_dtype(array, "depths")
    array = np.asarray(array, dtype=np.float64)
    if array.ndim not in (1, 2):
        raise ValueError(f"depths は (F,) か (F, J) にしてください（いまの形：{array.shape}）")
    if np.isinf(array).any():
        raise ValueError("depths に無限大が入っています")
    if (array < 0.0).any():
        raise ValueError("depths に負の値があります（沈みの深さは 0 以上）")
    return array


def summarize_sink(depths: Any, *, tolerance: float = thresholds.FLOOR_SINK_TOLERANCE) -> Measurement:
    """足の床への沈みを、1つの信号にまとめる（名前 "floor_sink"、単位 "m"）。

    - depths は sink_depths の答え：(F,)（点1つ）か (F, J)（足の点がいくつか）。NaN は「測れなかった」。
    - 値は、いちばん深い沈み。
    - 赤：tolerance より深い（ちょうどは赤にしない）。沈みが tolerance 以下なら緑（基準は仮）。
    - 灰：点が無い、全部測れなかった、または赤でなく、測れなかった点がある。
    - details："sink_spans"（どれかの点が tolerance より深いフレームの区間。全部）、"worst_frame"、
      "worst_point"（(F, J) のときの点の番号。(F,) のときは None）、数。
    """
    d = _as_depths(depths)
    tol = _as_real(tolerance, "tolerance", nonnegative=True)
    if d.size == 0:
        return Measurement.unmeasured(FLOOR_SINK_NAME, "m", "点がありません")
    table = d if d.ndim == 2 else d[:, None]
    measured = ~np.isnan(table)
    safe = np.where(measured, table, 0.0)
    sinking = (measured & (safe > tol)).any(axis=1)
    details: dict[str, Any] = {
        "frame_count": table.shape[0],
        "point_count": table.shape[1],
        "tolerance": tol,
        "sink_count": int(sinking.sum()),
        "sink_spans": spans(sinking),
        "unmeasured_count": int((~measured).sum()),
        "unmeasured_frame_spans": spans((~measured).any(axis=1)),
    }
    if not measured.any():
        return Measurement.unmeasured(FLOOR_SINK_NAME, "m", "沈みを測れた点がありません", details)
    flat = int(np.argmax(np.where(measured, table, -np.inf)))
    worst_frame, worst_point = divmod(flat, table.shape[1])
    value = float(table[worst_frame, worst_point])
    details["worst_frame"] = worst_frame
    details["worst_point"] = worst_point if d.ndim == 2 else None
    unmeasured = details["unmeasured_count"] > 0
    if value > tol:
        reason = "足が床に沈んでいます"
        if unmeasured:
            reason += "（測れなかった点もあります）"
        return Measurement(FLOOR_SINK_NAME, value, "m", Signal.RED, reason, details)
    if unmeasured:
        return Measurement(FLOOR_SINK_NAME, value, "m", Signal.GRAY,
                           "沈みを測れなかった点があります（そこに沈みが隠れているかもしれません）", details)
    return Measurement(FLOOR_SINK_NAME, value, "m", Signal.GREEN, "", details)


def _as_slides(slides: Iterable[Sequence[Any]]) -> list[tuple[int, int, float]]:
    """(始まり, 終わり, 距離) の一覧を確かめる。距離の NaN は通す（測れなかった区間）"""
    result: list[tuple[int, int, float]] = []
    for item in slides:
        try:
            start, end, distance = item
        except (TypeError, ValueError):
            raise ValueError(f"slides の1つ1つは (始まり, 終わり, 距離) の組にしてください（いまの値：{item!r}）") from None
        for name, frame in (("始まり", start), ("終わり", end)):
            if isinstance(frame, (bool, np.bool_)) or not isinstance(frame, (int, np.integer)) or frame < 0:
                raise ValueError(f"slides の{name}は 0 以上の整数にしてください（いまの値：{item!r}）")
        if start > end:
            raise ValueError(f"slides の始まりが終わりより後です（いまの値：{item!r}）")
        if isinstance(distance, (bool, np.bool_)) or not isinstance(distance, (int, float, np.integer, np.floating)):
            raise ValueError(f"slides の距離は数にしてください（いまの値：{item!r}）")
        value = float(distance)
        if math.isinf(value) or value < 0.0:
            raise ValueError(f"slides の距離は 0 以上の有限の数にしてください（いまの値：{item!r}）")
        result.append((int(start), int(end), value))
    return result


def summarize_slide(slides: Iterable[Sequence[Any]], *,
                    tolerance: float = thresholds.FOOT_SLIDE_TOLERANCE) -> Measurement:
    """足すべりを、1つの信号にまとめる（名前 "foot_slide"、単位 "m"）。

    - slides は slide_distances の答え（(始まり, 終わり, 距離) の一覧）。距離の NaN は「測れなかった区間」。
    - 値は、いちばん大きい足すべり。
    - 赤：tolerance より大きい（ちょうどは赤にしない）。
    - 灰：着いている区間が1つも無い（足すべりを測れない）、全部測れなかった、または赤でなく、測れなかった区間がある。
    - details："slide_spans"（tolerance を超えた区間 (始まり, 終わり, 距離)。全部）、"worst_span"、区間とフレームの数。
    """
    items = _as_slides(slides)
    tol = _as_real(tolerance, "tolerance", nonnegative=True)
    if not items:
        return Measurement.unmeasured(FOOT_SLIDE_NAME, "m", "床に着いている区間がありません（足すべりを測れません）")
    measured = [item for item in items if not math.isnan(item[2])]
    details: dict[str, Any] = {
        "span_count": len(items),
        "contact_frame_count": sum(end - start + 1 for start, end, _ in items),
        "tolerance": tol,
        "slide_spans": [item for item in measured if item[2] > tol],
        "unmeasured_count": len(items) - len(measured),
    }
    if not measured:
        return Measurement.unmeasured(FOOT_SLIDE_NAME, "m", "足すべりを測れた区間がありません", details)
    worst = max(measured, key=lambda item: item[2])  # 同じ距離なら前の区間
    details["worst_span"] = worst
    value = worst[2]
    unmeasured = details["unmeasured_count"] > 0
    if value > tol:
        reason = "床に着いている間に、足が滑っています"
        if unmeasured:
            reason += "（測れなかった区間もあります）"
        return Measurement(FOOT_SLIDE_NAME, value, "m", Signal.RED, reason, details)
    if unmeasured:
        return Measurement(FOOT_SLIDE_NAME, value, "m", Signal.GRAY,
                           "足すべりを測れなかった区間があります（そこに足すべりが隠れているかもしれません）", details)
    return Measurement(FOOT_SLIDE_NAME, value, "m", Signal.GREEN, "", details)


def summarize_unreachable(mask: Any) -> Measurement:
    """足が届かないフレームを、1つの信号にまとめる（名前 "unreachable_frames"、単位 "count"）。

    - mask は unreachable_frames の答え (F,) bool。値は、届かないフレームの数。
    - ロードマップ（段階1 基準3）：届かないフレームは、全部を一覧に出す。details["unreachable_spans"] に、全部の区間を入れる（省かない）。
    - 黄：届かないフレームが1つでもある（仮。この段階では見つけて一覧にするまで。直すのは M5）。無ければ緑。
    - 灰：フレームが無い。
    """
    m = _as_mask(mask, "mask")
    if m.size == 0:
        return Measurement.unmeasured(UNREACHABLE_NAME, "count", "フレームがありません")
    count = int(m.sum())
    details: dict[str, Any] = {
        "frame_count": len(m),
        "unreachable_count": count,
        "unreachable_spans": spans(m),
    }
    if count:
        return Measurement(UNREACHABLE_NAME, float(count), "count", Signal.YELLOW,
                           "目標に届かないフレームがあります（全部を一覧に出します。黄は仮）", details)
    return Measurement(UNREACHABLE_NAME, 0.0, "count", Signal.GREEN, "", details)


def summarize_jumps(mask: Any, *, bone_names: Sequence[str] | None = None) -> Measurement:
    """骨の跳びを、1つの信号にまとめる（名前 "bone_jumps"、単位 "count"）。

    - mask は bone_jump_frames の答え (F−1, B) bool。行 i は「フレーム i と i+1 の間」。
    - 値は、跳んだ数（フレームの間と骨の組の数）。
    - 赤：1つでも跳んでいる（基準 thresholds.BONE_JUMP_DEG は仮）。無ければ緑。
    - 灰：比べるフレームの間が無い（フレームが1つ以下）、骨が無い。
    - details："jumps"（跳んだ全部の (フレームの間 i, 骨の番号) を、行にした (K, 2) の int64 の並び。
      フレームの間の順、同じなら骨の番号の順。壊れた動きで跳びがとても多くても、メモリが増えすぎないように numpy の並びにする）、
      "jump_spans"（どれかの骨が跳んだ、フレームの間の区間）、
      "bone_jump_counts"（跳んだ骨ごとの (骨, 数)。bone_names があれば骨は名前、無ければ番号）。
    - bone_names を渡すときは、長さを B にした名前の一覧（list・tuple）にする（MMD 準標準の名前など）。1つの文字の並び（str）は受け付けない。
    """
    m = _as_mask(mask, "mask", ndim=2)
    transition_count, bone_count = m.shape
    names: list[str] | None = None
    if bone_names is not None:
        if isinstance(bone_names, str):
            raise ValueError("bone_names は名前の一覧（list・tuple）にしてください（1つの文字の並びは、1文字ずつの名前になってしまいます）")
        names = list(bone_names)
        if len(names) != bone_count or not all(isinstance(name, str) for name in names):
            raise ValueError(f"bone_names は、骨の数（{bone_count}）と同じ長さの文字の一覧にしてください")
    if transition_count == 0:
        return Measurement.unmeasured(BONE_JUMPS_NAME, "count", "比べるフレームがありません（フレームが1つ以下です）")
    if bone_count == 0:
        return Measurement.unmeasured(BONE_JUMPS_NAME, "count", "骨がありません")
    jumps = np.argwhere(m).astype(np.int64, copy=False)  # フレームの間の順、同じなら骨の番号の順
    per_bone = m.sum(axis=0)
    details: dict[str, Any] = {
        "transition_count": transition_count,
        "bone_count": bone_count,
        "jump_count": len(jumps),
        "jumps": jumps,
        "jump_spans": spans(m.any(axis=1)),
        "bone_jump_counts": [(names[bone] if names is not None else int(bone), int(per_bone[bone]))
                             for bone in np.flatnonzero(per_bone)],
        "bone_names": names,
    }
    count = details["jump_count"]
    if count:
        return Measurement(BONE_JUMPS_NAME, float(count), "count", Signal.RED, "1フレームで大きく回った骨があります", details)
    return Measurement(BONE_JUMPS_NAME, 0.0, "count", Signal.GREEN, "", details)

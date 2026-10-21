"""A5 確かめの動き：ポーズごとの確かめと、まとめ・報告の文（bpy なし・numpy だけ）。

1つのポーズを付けた形（並び）から、次の確かめをする。どれも世界の座標（メートル、Z が上）で受け取る。
- "knee_reverse_bend"：ひざの逆曲がり（check_knees）。太ももの骨の回転を使う measure.kinematics.knee_bend_from_thigh で
  左右のひざの曲がりを測り、summarize_reverse_bend でまとめる。逆に曲がりすぎていれば赤、測れなければ灰。
- "floor_sink"：床への沈み（check_floor_sink）。体の頂点が、床（休みの姿勢の体のいちばん低い高さ）より下にあるか。
  立つ・座る姿勢だけで測る。横になる姿勢では、体が床に着くのがふつうなので測らない（出来事を作らない）。
- "cloth_penetration"：服からの体のはみ出し（check_cloth_penetration）。服の頂点が、体の閉じた部品の中にあるか
  （服が体の中へ入る＝体が服から出ている）。measure.geometry.penetration_depths で深さを測り、summarize_penetration で
  まとめる。閉じていない体の部品は、中か外かを決められないので、その近くの服の点は「測れない」（灰）にする。
  見え方（アプリのカメラから見えるか）は、重いので visibility を選んだときだけ調べる（まわり全体のカメラ 62 台）。
  調べないときは、隠れためり込みも数える（多めに赤になる）。
- アプリの動き（motions.json）：MotionLibrary.swift がまだ無く、値から骨の回転への式が分からないので、計算しない。
  推測で埋めずに、灰の出来事として一覧に出す（app_motion_events）。VMD の動きは M1 で足す。

足りないポーズ（モデルに無い骨がある・モデルの拘束で決めた向きにならなかった）は、そのポーズの確かめを灰にする
（赤はそのまま赤。赤 > 灰 の順に重い）。
報告の文は texts（motion 分野）の決まった形に、プログラムが作った数字を入れて作る。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import texts, thresholds
from ..measure import kinematics
from ..measure.geometry import (DEFAULT_CHUNK_PAIRS, PENETRATION_NAME, as_array, as_points, as_triangles,
                                closest_points, penetration_depths, summarize_penetration, weld_exact)
from ..measure.results import Measurement, Signal, worst
from ..measure.visibility import Camera, DepthBuffer, orbit_views, summarize_visible_points
from ..thresholds import APP_ORBIT_DISTANCE_MARGIN
from .poses import POSES, AppliedPose, PoseSpec

__all__ = [
    "APP_MOTIONS_ALL",
    "APP_MOTION_CHECK",
    "APP_MOTION_NAME",
    "APP_MOTION_PREFIX",
    "CHECKS",
    "CLOTH_CHECK",
    "COURSE_NAME",
    "KNEE_CHECK",
    "SIDES",
    "SINK_CHECK",
    "VISIBLE_PENETRATION_NAME",
    "BodyParts",
    "Event",
    "app_motion_events",
    "body_parts",
    "check_cloth_penetration",
    "check_floor_sink",
    "check_knees",
    "check_label",
    "cloth_depths",
    "orbit_cameras",
    "pose_events",
    "pose_label",
    "report_lines",
    "summarize_course",
]

# MARK: - 決まり

CHECKS = ("knee_reverse_bend", "floor_sink", "cloth_penetration")
KNEE_CHECK, SINK_CHECK, CLOTH_CHECK = CHECKS
APP_MOTION_CHECK = "app_motion"
# アプリの動きの名前が分からないとき、まとめて1つの灰の出来事にするときのポーズの名前
APP_MOTIONS_ALL = "app_motions"
# アプリの動きの出来事のポーズの名前の頭（"app:" + 動きの名前）
APP_MOTION_PREFIX = "app:"
# 結果の名前
COURSE_NAME = "pose_course"
APP_MOTION_NAME = "app_motion"
VISIBLE_PENETRATION_NAME = "visible_penetration"
# 左右の並び（キャラから見た左、右）
SIDES = ("L", "R")

_SIGNAL_KEYS = {Signal.GREEN: "motion.signal_green", Signal.YELLOW: "motion.signal_yellow",
                Signal.GRAY: "motion.signal_gray", Signal.RED: "motion.signal_red"}
_POSES_BY_NAME = {pose.name: pose for pose in POSES}


# MARK: - 出来事


@dataclass(frozen=True)
class Event:
    """確かめの出来事1つ：ポーズ（またはアプリの動き）1つの、確かめ1つの結果。

    pose: ポーズの名前（POSES の name）。アプリの動きは "app:<動きの名前>" か "app_motions"（名前が分からないとき）
    check: 確かめの名前（CHECKS のどれか、または "app_motion"）
    measurement: 測った結果（信号・値・理由）
    """

    pose: str
    check: str
    measurement: Measurement

    def __post_init__(self) -> None:
        if not isinstance(self.pose, str) or not self.pose:
            raise ValueError(f"pose は空でない文字にしてください（いまは {self.pose!r}）")
        if self.check not in CHECKS + (APP_MOTION_CHECK,):
            raise ValueError(f"check は {'・'.join(CHECKS + (APP_MOTION_CHECK,))} のどれかにしてください（いまは {self.check!r}）")
        if not isinstance(self.measurement, Measurement):
            raise ValueError(f"measurement は Measurement にしてください（いまは {type(self.measurement).__name__}）")

    @property
    def signal(self) -> Signal:
        return self.measurement.signal


def _join(items: Iterable[str]) -> str:
    return texts.text("motion.list_separator").join(items)


def _with(measurement: Measurement, *, signal: Signal | None = None, reasons: Sequence[str] = (),
          details: Mapping[str, Any] | None = None, reasons_first: bool = False) -> Measurement:
    """測った結果に、理由と details を足した写し（信号を重くするときは signal）"""
    own = [measurement.reason] if measurement.reason else []
    joined = [*reasons, *own] if reasons_first else [*own, *reasons]
    return Measurement(measurement.name, measurement.value, measurement.unit,
                       measurement.signal if signal is None else signal, "。".join(r for r in joined if r),
                       {**measurement.details, **(details or {})})


# MARK: - ひざの逆曲がり


def _as_rows(value: Any, name: str, width: int) -> np.ndarray:
    """(2, width) の float64（左, 右）。NaN は「その側は測れない」として通す。無限大は ValueError"""
    array = as_array(value, name)
    if array.dtype.kind not in "iuf" or array.shape != (len(SIDES), width):
        raise ValueError(f"{name} は ({len(SIDES)}, {width}) の並び（左, 右の順）にしてください（いまの形：{array.shape}）")
    array = np.asarray(array, dtype=np.float64)
    if np.isinf(array).any():
        raise ValueError(f"{name} に無限大が入っています")
    return array


def check_knees(*, rest_hips: Any, rest_knees: Any, hips: Any, knees: Any, ankles: Any, thigh_rotations: Any,
                lateral: Any, missing_bones: Sequence[str] = ()) -> Measurement:
    """左右のひざの逆曲がり（名前 "reverse_bend"、単位 "deg"。measure.kinematics.summarize_reverse_bend の形）。

    - rest_hips・rest_knees：休みの姿勢の、太ももの骨の根元（股関節）とひざの骨の根元 (2, 3)（左, 右の順）。
    - hips・knees・ankles：ポーズを付けたあとの、股関節・ひざ・足首の位置 (2, 3)。
    - thigh_rotations：太ももの骨（足 または 足D）を、休みの向きからポーズの向きへ回す回転 (2, 4)（w, x, y, z。
      位置と同じ座標の向き）。
    - lateral：キャラの左を向く向き (3,)（休みの姿勢の 左足 − 右足 など）。ひざの休みの軸は、これを太ももに
      垂直にした向き（kinematics.knee_rest_axis）。
    - 行に NaN がある側は、骨が無いなどで測れない（灰の理由）。missing_bones には、その足りない骨の名前を渡す
      （理由の文に入る）。
    値は、いちばん深い逆曲がりの角度（逆に曲がっていなければ 0）。details に "bend_deg"（左右の曲がり。ふつうが正）。
    """
    rest_h = _as_rows(rest_hips, "rest_hips", 3)
    rest_k = _as_rows(rest_knees, "rest_knees", 3)
    posed_h = _as_rows(hips, "hips", 3)
    posed_k = _as_rows(knees, "knees", 3)
    posed_a = _as_rows(ankles, "ankles", 3)
    rotations = _as_rows(thigh_rotations, "thigh_rotations", 4)
    side = as_array(lateral, "lateral")
    if side.dtype.kind not in "iuf" or side.shape != (3,) or not np.isfinite(side).all():
        raise ValueError("lateral は3つの有限の数にしてください")
    bends = np.full(len(SIDES), np.nan)
    axis_failed: list[str] = []
    for index, name in enumerate(SIDES):
        rows = (rest_h[index], rest_k[index], posed_h[index], posed_k[index], posed_a[index], rotations[index])
        if any(np.isnan(row).any() for row in rows):
            continue
        try:
            axis = kinematics.knee_rest_axis(rest_h[index], rest_k[index], side)
        except ValueError:
            axis_failed.append(name)
            continue
        bends[index] = float(kinematics.knee_bend_from_thigh(posed_h[index], posed_k[index], posed_a[index],
                                                             rest_axis=axis, thigh_rotation=rotations[index]))
    measurement = kinematics.summarize_reverse_bend(bends)
    reasons = []
    if missing_bones:
        reasons.append(texts.text("motion.reason_no_leg_bones", bones=_join(missing_bones)))
    if axis_failed:
        reasons.append(texts.text("motion.reason_knee_axis", sides=_join(axis_failed)))
    details = {"sides": SIDES, "bend_deg": {name: float(bends[i]) for i, name in enumerate(SIDES)},
               "axis_failed": tuple(axis_failed), "missing_bones": tuple(missing_bones)}
    if "worst_frame" in measurement.details:
        details["worst_side"] = SIDES[measurement.details["worst_frame"]]
    return _with(measurement, reasons=reasons, details=details)


# MARK: - 床への沈み


def check_floor_sink(body_points: Any, *, floor_z: float) -> Measurement:
    """体の頂点の、床への沈み（名前 "floor_sink"、単位 "m"。measure.kinematics.summarize_sink の形）。

    body_points：ポーズを付けた体の頂点 (N, 3)（世界の座標）。floor_z：床の高さ（休みの姿勢の体のいちばん低い高さ）。
    値は、いちばん深い沈み。thresholds.FLOOR_SINK_TOLERANCE より深ければ赤。details["worst_point"] は頂点の番号。
    """
    points = as_points(body_points, "body_points")
    if isinstance(floor_z, bool) or not isinstance(floor_z, (int, float, np.integer, np.floating)) \
            or not math.isfinite(float(floor_z)):
        raise ValueError(f"floor_z は有限の数にしてください（いまは {floor_z!r}）")
    depths = kinematics.sink_depths(points, floor_z=float(floor_z))
    measurement = kinematics.summarize_sink(depths.reshape(1, -1))
    return _with(measurement, details={"floor_z": float(floor_z)})


# MARK: - 服からの体のはみ出し


@dataclass(frozen=True, eq=False)
class BodyParts:
    """体の形を、部品（殻）に分けた物。休みの姿勢で1回だけ作る（ポーズでは頂点の位置だけが変わる）。

    vertex_count: 体の頂点の数（分けた元の並びの数）
    part_vertices: 部品ごとの頂点の番号（元の並びの番号、小さい順）
    part_triangles: 部品ごとの三角形 (M_k, 3)（part_vertices の中の番号に付け直した物）
    closed: 部品ごとの「閉じた形か」(K,) bool：部品のどの辺もちょうど2つの面が使い、面の向きがそろい、体積が正
        （面が外を向く）。閉じた形でないと、中か外かを決められない

    部品（殻）の分け方：辺を共有する2つの面を、次のどちらかのときにつなぐ。
    - 同じ頂点の番号の辺を、ちょうど2つの面が使う（ふつうのつながり）。
    - 座標がまったく同じ頂点を同じ点とみなした辺を、ちょうど2つの面が使う（UV などで分かれた頂点でも、つながる）。
    座標が同じだけの辺を3つ以上の面が使う所（2つの閉じた形が、同じ位置の頂点の輪でくっついた所など）ではつなげない
    ので、くっついた閉じた形は、別々の閉じた部品になる。閉じているかは、座標が同じ頂点を同じ点とみなして調べる。
    """

    vertex_count: int
    part_vertices: tuple[np.ndarray, ...]
    part_triangles: tuple[np.ndarray, ...]
    closed: np.ndarray

    @property
    def part_count(self) -> int:
        return len(self.part_vertices)

    @property
    def closed_count(self) -> int:
        return int(np.count_nonzero(self.closed))

    @property
    def open_count(self) -> int:
        return self.part_count - self.closed_count


def _signed_volumes(vertices: np.ndarray, triangles: np.ndarray, labels: np.ndarray, count: int) -> np.ndarray:
    """部品ごとの体積 (K,)（閉じた形で、面が外を向いていれば正）"""
    a, b, c = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    return np.bincount(labels, weights=np.einsum("ij,ij->i", a, np.cross(b, c)), minlength=count) / 6.0


def _half_edges(welded: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """三角形の向きのある辺（v0→v1、v1→v2、v2→v0）：(辺の番号 (E,)、面の番号 (E,)、小さい番号から大きい番号へ向かうか (E,))。
    同じ頂点どうしの辺（つぶれた三角形の辺）は除く。辺の番号は、向きを無くした辺ごとに同じ数"""
    start = welded.reshape(-1)
    end = welded[:, [1, 2, 0]].reshape(-1)
    face = np.repeat(np.arange(len(welded), dtype=np.int64), 3)
    keep = start != end
    start, end, face = start[keep], end[keep], face[keep]
    size = int(welded.max()) + 1 if welded.size else 1
    key = np.minimum(start, end) * size + np.maximum(start, end)
    return key, face, start < end


def _pairs(key: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """同じ番号がちょうど2回ある所の (1つ目, 2つ目) の位置と、番号ごとの回数（key の並びの位置ごと）"""
    order = np.argsort(key, kind="stable")
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    per_entry = counts[inverse.reshape(-1)]
    sorted_two = order[per_entry[order] == 2]
    return sorted_two[0::2], sorted_two[1::2], per_entry


def _connected_labels(count: int, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """count 個の物を、(first[i], second[i]) の組でつないだ部品の番号（部品の中のいちばん小さい物の番号）"""
    labels = np.arange(count, dtype=np.int64)
    while True:
        low = np.minimum(labels[first], labels[second])
        updated = labels.copy()
        np.minimum.at(updated, first, low)
        np.minimum.at(updated, second, low)
        updated = updated[updated]
        if np.array_equal(updated, labels):
            return labels
        labels = updated


def body_parts(vertices: Any, triangles: Any) -> BodyParts:
    """体の形（休みの姿勢）を部品（殻）に分け、部品ごとに閉じた形かを調べる（BodyParts の説明）"""
    v = as_points(vertices, "vertices")
    t = as_triangles(triangles, len(v))
    if len(t) == 0:
        return BodyParts(vertex_count=len(v), part_vertices=(), part_triangles=(), closed=np.zeros(0, dtype=bool))
    welded = weld_exact(v)[t]
    key, face, forward = _half_edges(welded)
    first, second, _ = _pairs(key)
    raw_key, raw_face, _ = _half_edges(t)
    raw_first, raw_second, _ = _pairs(raw_key)
    roots = _connected_labels(len(t), np.concatenate([face[first], raw_face[raw_first]]),
                              np.concatenate([face[second], raw_face[raw_second]]))
    _, labels = np.unique(roots, return_inverse=True)
    labels = labels.reshape(-1).astype(np.int64)
    count = int(labels.max()) + 1
    # 部品の中で数える：同じ辺でも、部品が違えば別の辺（くっついた閉じた形は、それぞれの中で閉じている）
    part_key = labels[face] * (int(key.max()) + 1) + key
    part_first, part_second, per_entry = _pairs(part_key)
    open_part = np.zeros(count, dtype=bool)
    open_part[labels[face[per_entry != 2]]] = True
    inconsistent = forward[part_first] == forward[part_second]
    open_part[labels[face[part_first[inconsistent]]]] = True
    closed = ~open_part & (_signed_volumes(v, t, labels, count) > 0.0)
    order = np.argsort(labels, kind="stable")
    cuts = np.cumsum(np.bincount(labels, minlength=count))[:-1]
    part_vertices = []
    part_triangles = []
    for rows in np.split(order, cuts):
        tri = t[rows]
        used = np.unique(tri)
        part_vertices.append(used)
        part_triangles.append(np.searchsorted(used, tri).astype(np.int64))
    return BodyParts(vertex_count=len(v), part_vertices=tuple(part_vertices), part_triangles=tuple(part_triangles),
                     closed=closed)


def cloth_depths(cloth_points: Any, body_vertices: Any, parts: BodyParts, *,
                 chunk_size: int = DEFAULT_CHUNK_PAIRS) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """服の点ごとの、体の閉じた部品の中への深さ。返す物：(深さ (P,), 体の表面のいちばん近い点 (P, 3), 数)。

    - 深さ：どれかの閉じた部品の中にあれば、その中でいちばん深い深さ（メートル、正）。
      どの部品の中にも無ければ 0。中か外かを決められない点（閉じていない部品の囲みの箱の中にある、
      符号が決まらない）で、どの閉じた部品の中にも無い点は NaN。
    - 体の表面のいちばん近い点：めり込んだ点から、その部品の表面へいちばん近い点（体が服から出ている所）。
      めり込んでいない点は NaN。
    - 部品の中にある点は、その部品の頂点の囲みの箱（軸にそろった箱）の中にあるので、箱の中の点だけを調べる。
    body_vertices は、parts を作った体の頂点の、いまのポーズの位置（同じ並び・同じ数）。
    """
    points = as_points(cloth_points, "cloth_points")
    vertices = as_points(body_vertices, "body_vertices")
    if not isinstance(parts, BodyParts):
        raise ValueError(f"parts は BodyParts にしてください（いまは {type(parts).__name__}）")
    if len(vertices) != parts.vertex_count:
        raise ValueError(f"body_vertices の数（{len(vertices)}）が、parts の頂点の数（{parts.vertex_count}）と違います")
    count = len(points)
    depth = np.zeros(count)
    surface = np.full((count, 3), np.nan)
    unknown = np.zeros(count, dtype=bool)
    near_open = np.zeros(count, dtype=bool)
    for index in range(parts.part_count):
        part = vertices[parts.part_vertices[index]]
        low, high = part.min(axis=0), part.max(axis=0)
        inside = np.flatnonzero(np.all((points >= low) & (points <= high), axis=1))
        if inside.size == 0:
            continue
        if not parts.closed[index]:
            unknown[inside] = True
            near_open[inside] = True
            continue
        triangles = parts.part_triangles[index]
        found = penetration_depths(points[inside], part, triangles, chunk_size=chunk_size)
        undecided = np.isnan(found)
        unknown[inside[undecided]] = True
        deeper = ~undecided & (found > depth[inside])
        if deeper.any():
            rows = inside[deeper]
            depth[rows] = found[deeper]
            surface[rows] = closest_points(points[rows], part, triangles, chunk_size=chunk_size).points
    penetrating = depth > 0.0
    result = np.where(penetrating, depth, np.where(unknown, np.nan, 0.0))
    info = {
        "point_count": count,
        "penetrating_count": int(penetrating.sum()),
        "undecided_count": int((unknown & ~penetrating).sum()),
        "near_open_part_count": int((near_open & ~penetrating).sum()),
    }
    return result, surface, info


def check_cloth_penetration(cloth_points: Any, body_vertices: Any, parts: BodyParts, *,
                            renders: Mapping[str, tuple[Camera, DepthBuffer]] | None = None,
                            scene_mesh: tuple[Any, Any] | None = None,
                            chunk_size: int = DEFAULT_CHUNK_PAIRS) -> Measurement:
    """服からの体のはみ出し（名前 "penetration_depth"、単位 "m"。measure.geometry.summarize_penetration の形）。

    - 服の頂点が体の閉じた部品の中にあれば、体がそこで服から出ている。深さは cloth_depths。
      基準は thresholds.PENETRATION_YELLOW・PENETRATION_RED。
    - renders（visibility.render_views の答え）と scene_mesh（描いたときと同じ (頂点, 三角形)）を渡すと、
      めり込んだ点の「体の表面のいちばん近い点」（はみ出した体）が、どれかのカメラから見えるかを調べ、
      見えない点は数えない（記録だけ。details["visibility"] に summarize_visible_points の結果）。
      どのカメラでも枠の外の点は、見えるとして数える（見落とさないため）。
    - renders が None なら、全部の点を数える（隠れためり込みも赤になる）。
    - 体の部品が無い・閉じた部品が無いときは灰。閉じていない部品の近くで決められない点があるときは、赤でなければ灰。
    """
    if not isinstance(parts, BodyParts):
        raise ValueError(f"parts は BodyParts にしてください（いまは {type(parts).__name__}）")
    if (renders is None) != (scene_mesh is None):
        raise ValueError("renders と scene_mesh は、両方渡すか、両方渡さないかにしてください")
    base = {"body_part_count": parts.part_count, "closed_part_count": parts.closed_count,
            "open_part_count": parts.open_count, "visibility_checked": renders is not None}
    if parts.part_count == 0:
        return Measurement.unmeasured(PENETRATION_NAME, "m", texts.text("motion.reason_no_body"), base)
    if parts.closed_count == 0:
        return Measurement.unmeasured(PENETRATION_NAME, "m", texts.text("motion.reason_no_closed_body"), base)
    depths, surface, info = cloth_depths(cloth_points, body_vertices, parts, chunk_size=chunk_size)
    visible = None
    seen = None
    if renders is not None:
        visible = np.ones(len(depths), dtype=bool)
        penetrating = np.flatnonzero(depths > 0.0)
        if penetrating.size:
            seen = summarize_visible_points(surface[penetrating], renders, mesh=scene_mesh,
                                            name=VISIBLE_PENETRATION_NAME)
            if "visible" in seen.details:
                visible[penetrating] = seen.details["visible"] | seen.details["unmeasured"]
            # カメラが無いなど、見え方を調べられなかったときは、全部を見える点として数える（見落とさないため）
    measurement = summarize_penetration(depths, visible=visible)
    reasons = []
    if info["near_open_part_count"]:
        reasons.append(texts.text("motion.reason_open_body", count=str(parts.open_count)))
    return _with(measurement, reasons=reasons, details={**base, **info, "visibility": seen})


def orbit_cameras(vertices: Any, **options: Any) -> dict[str, Camera]:
    """モデル全体が写る、まわり全体から見るカメラ（measure.visibility.orbit_views。アプリのカメラの画角・縦長の画面）。

    中心は頂点の囲みの箱の真ん中。距離は、囲みの箱を包む球が画像の短い辺に入る距離 × thresholds.APP_ORBIT_DISTANCE_MARGIN。
    options は orbit_views にそのまま渡す（azimuth_step_deg・elevations_deg・include_poles・height など）。
    """
    points = as_points(vertices, "vertices")
    if len(points) == 0:
        raise ValueError("頂点がありません（カメラを置けません）")
    low, high = points.min(axis=0), points.max(axis=0)
    radius = max(float(np.linalg.norm(high - low)) / 2.0, thresholds.CAMERA_NEAR)
    aspect = float(options.get("aspect", thresholds.APP_SCREEN_ASPECT))
    fov = float(options.get("fov_deg", thresholds.APP_CAMERA_FOV_DEG))
    if not (math.isfinite(aspect) and aspect > 0.0 and math.isfinite(fov) and 0.0 < fov < 180.0):
        raise ValueError(f"aspect と fov_deg を確かめてください（いまは {aspect}・{fov}）")
    # 画角は長い辺に合わせる（fov_axis "auto"）。短い辺の半分の角度で、球が入る距離を決める
    short_half = math.atan(math.tan(math.radians(fov) / 2.0) * min(aspect, 1.0 / aspect))
    distance = radius / math.sin(short_half) * APP_ORBIT_DISTANCE_MARGIN
    return orbit_views((low + high) / 2.0, distance, **options)


# MARK: - ポーズの出来事


def pose_events(pose: PoseSpec | str, measurements: Mapping[str, Measurement],
                applied: AppliedPose | None = None) -> list[Event]:
    """1つのポーズの確かめの結果を、出来事の並びにする（CHECKS の順）。

    applied が「足りない」ポーズ（骨が無い・決めた向きにならなかった）なら、どの確かめも灰にする
    （赤はそのまま。理由の頭に、足りない骨を書く）。details に "partial"・"missing_bones"・"overridden_bones" を足す。
    """
    name = pose.name if isinstance(pose, PoseSpec) else pose
    unknown = [check for check in measurements if check not in CHECKS]
    if unknown:
        raise ValueError(f"知らない確かめの名前があります：{unknown}（{'・'.join(CHECKS)} のどれか）")
    reasons: list[str] = []
    details: dict[str, Any] = {}
    if applied is not None:
        if not isinstance(applied, AppliedPose):
            raise ValueError(f"applied は AppliedPose にしてください（いまは {type(applied).__name__}）")
        if applied.pose != name:
            raise ValueError(f"applied は別のポーズの物です（{applied.pose} と {name}）")
        if applied.missing:
            reasons.append(texts.text("motion.reason_missing_bones", bones=_join(applied.missing)))
        if applied.overridden:
            reasons.append(texts.text("motion.reason_overridden", bones=_join(applied.overridden)))
        details = {"partial": applied.partial, "missing_bones": applied.missing,
                   "overridden_bones": applied.overridden}
    events = []
    for check in CHECKS:
        measurement = measurements.get(check)
        if measurement is None:
            continue
        if not isinstance(measurement, Measurement):
            raise ValueError(f"{check} の結果は Measurement にしてください")
        if reasons:
            measurement = _with(measurement, signal=worst([measurement.signal, Signal.GRAY]), reasons=reasons,
                                details=details, reasons_first=True)
        elif details:
            measurement = _with(measurement, details=details)
        events.append(Event(name, check, measurement))
    return events


def app_motion_events(names: Sequence[str] | None = None) -> list[Event]:
    """アプリの動き（motions.json）の出来事。どれも灰（MotionLibrary.swift が無いので、まだ計算できない）。

    names（動きの名前の並び）を渡すと、動きごとに1つ（ポーズの名前は "app:<名前>"）。None なら、まとめて1つ
    （ポーズの名前は "app_motions"）。名前は、呼ぶ側が motions.json から読んで渡す（ここでは読まない）。
    """
    reason = texts.text("motion.reason_app_motions")
    if names is None:
        return [Event(APP_MOTIONS_ALL, APP_MOTION_CHECK, Measurement.unmeasured(APP_MOTION_NAME, "count", reason))]
    if isinstance(names, (str, bytes)) or not isinstance(names, Iterable):
        raise ValueError("names は動きの名前の並びにしてください")
    values = list(names)
    if not all(isinstance(name, str) and name for name in values):
        raise ValueError("動きの名前は空でない文字にしてください")
    if len(set(values)) != len(values):
        raise ValueError("動きの名前が重なっています")
    return [Event(APP_MOTION_PREFIX + name, APP_MOTION_CHECK,
                  Measurement.unmeasured(APP_MOTION_NAME, "count", reason, {"motion": name})) for name in values]


# MARK: - まとめ・報告


def _as_events(events: Iterable[Event]) -> list[Event]:
    values = list(events)
    if not all(isinstance(event, Event) for event in values):
        raise ValueError("events は Event の並びにしてください")
    return values


def summarize_course(events: Iterable[Event]) -> Measurement:
    """確かめの動き全体のまとめ（名前 "pose_course"、単位 "count"）。

    信号は、いちばん重い出来事の信号（赤 > 灰 > 黄 > 緑）。値は赤の出来事の数。出来事が無ければ灰。
    details：
    - "per_pose"：ポーズの名前 → {"signal": いちばん重い信号, "checks": {確かめ: 信号}}（出てきた順）
    - "red_events"・"gray_events"・"yellow_events"：(ポーズ, 確かめ) の並び
    - "counts"：信号の名前（"green"・"yellow"・"gray"・"red"）→ 出来事の数、"pose_count"：ポーズの数（アプリの動きは数えない）、"event_count"
    """
    values = _as_events(events)
    if not values:
        return Measurement.unmeasured(COURSE_NAME, "count", texts.text("motion.reason_no_events"))
    per_pose: dict[str, dict[str, Any]] = {}
    for event in values:
        entry = per_pose.setdefault(event.pose, {"signal": event.signal, "checks": {}})
        entry["checks"][event.check] = event.signal
        entry["signal"] = max(entry["signal"], event.signal)
    # キーは信号の名前の文字（"green"・"yellow"・"gray"・"red"）。JSON に書けるように（to_data・裏の係の result.json）
    counts = {signal.name.lower(): sum(1 for event in values if event.signal == signal) for signal in Signal}
    pose_count = len({event.pose for event in values if event.check != APP_MOTION_CHECK})
    signal = worst(event.signal for event in values)
    if signal == Signal.GREEN:
        reason = texts.text("motion.summary_all_green", poses=str(pose_count))
    else:
        reason = texts.text("motion.summary_reason", poses=str(pose_count), red=str(counts["red"]),
                            gray=str(counts["gray"]), yellow=str(counts["yellow"]))
    details = {
        "per_pose": per_pose,
        "red_events": [(e.pose, e.check) for e in values if e.signal == Signal.RED],
        "gray_events": [(e.pose, e.check) for e in values if e.signal == Signal.GRAY],
        "yellow_events": [(e.pose, e.check) for e in values if e.signal == Signal.YELLOW],
        "counts": counts,
        "pose_count": pose_count,
        "event_count": len(values),
    }
    return Measurement(COURSE_NAME, float(counts["red"]), "count", signal, reason, details)


def pose_label(pose: str, *, language: str | None = None) -> str:
    """ポーズの名前の文（画面と報告に出す）。アプリの動きは「アプリの動き「名前」」。知らない名前は、そのまま返す"""
    known = _POSES_BY_NAME.get(pose)
    if known is not None and known.text_key in texts.TEXTS:
        return texts.text(known.text_key, language=language)
    if pose == APP_MOTIONS_ALL:
        return texts.text("motion.app_motions_all", language=language)
    if pose.startswith(APP_MOTION_PREFIX):
        return texts.text("motion.app_motion", language=language, name=pose[len(APP_MOTION_PREFIX):])
    return pose


def check_label(check: str, *, language: str | None = None) -> str:
    """確かめの名前の文"""
    if check not in CHECKS + (APP_MOTION_CHECK,):
        raise ValueError(f"知らない確かめの名前です：{check!r}")
    return texts.text(f"motion.check_{check}", language=language)


def _value_text(measurement: Measurement, language: str | None) -> str:
    """値の文。長さは mm（小数1桁）、角度は度（小数1桁）、数は整数"""
    value = measurement.value
    if value is None:
        return texts.text("motion.value_none", language=language)
    if measurement.unit == "m":
        return texts.text("motion.value_mm", language=language, value=f"{value * 1000.0:.1f}")
    if measurement.unit == "deg":
        return texts.text("motion.value_deg", language=language, value=f"{value:.1f}")
    if measurement.unit == "count" and float(value).is_integer():
        return texts.text("motion.value_count", language=language, value=str(int(value)))
    return texts.text("motion.value_other", language=language, value=f"{value:.3g}", unit=measurement.unit)


def report_lines(events: Iterable[Event], *, language: str | None = None) -> list[str]:
    """報告の文の並び：1行目はまとめ（ポーズと出来事の数）、そのあと赤・灰・黄の出来事を1行ずつ（重い順。同じなら出た順）。
    緑の出来事は数だけ（1行目）。赤・灰・黄が無ければ「ありません」の1行"""
    values = _as_events(events)
    counts = {signal: sum(1 for event in values if event.signal == signal) for signal in Signal}
    pose_count = len({event.pose for event in values if event.check != APP_MOTION_CHECK})
    lines = [texts.text("motion.report_header", language=language, poses=str(pose_count), events=str(len(values)),
                        red=str(counts[Signal.RED]), gray=str(counts[Signal.GRAY]),
                        yellow=str(counts[Signal.YELLOW]), green=str(counts[Signal.GREEN]))]
    notable = [event for event in values if event.signal != Signal.GREEN]
    notable.sort(key=lambda event: -int(event.signal))  # 並べかえは安定（同じ信号は出た順）
    for event in notable:
        fields = dict(signal=texts.text(_SIGNAL_KEYS[event.signal], language=language),
                      pose=pose_label(event.pose, language=language),
                      check=check_label(event.check, language=language),
                      value=_value_text(event.measurement, language))
        if event.measurement.reason:
            lines.append(texts.text("motion.report_event_reason", language=language, reason=event.measurement.reason,
                                    **fields))
        else:
            lines.append(texts.text("motion.report_event", language=language, **fields))
    if not notable:
        lines.append(texts.text("motion.report_all_green", language=language))
    return lines

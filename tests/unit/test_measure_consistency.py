"""部品どうしの決まりがそろっているかを確かめる（A4 の測る道具と、A8・A15 をまたぐ点検）。

- 基準の数字は thresholds.py だけに置く（CLAUDE.md 7章）。部品の中に「仮」の数字を残さない。
- 入力がおかしいときは、どの測る道具も日本語の ValueError にする（行の長さがそろわない並びも）。
- 結果（Measurement）の名前は snake_case、単位は決まった物だけ。調べる物が無いときは灰。
- 測れなかった値（NaN）が混ざるときは、どの道具も「赤でないかぎり灰」（測れなかった物は緑にしない。灰は黄より重い）。
- 同じことを計算する所が2か所あるときは、同じ答えになる（辺の一覧・つぶれた三角形・体積・閉じているか・重なった点）。
- 向き（Z が上、モデルは -Y を向く、キャラの左は +X）が、見え方・ひざの曲がり・左右のずれで同じ。
"""

import ast
import inspect
import math
import re
import unittest
from pathlib import Path

import numpy as np

from blender_ai_tools import catalog, provenance, thresholds
from blender_ai_tools.catalog import schema
from blender_ai_tools.export import roundtrip
from blender_ai_tools.measure import (deformation, geometry, kinematics, mesh_check, symmetry, visibility,
                                      weights)
from blender_ai_tools.measure.results import Measurement, Signal
from blender_ai_tools.model import card
from tests import shapes

PACKAGE_DIR = Path(__file__).resolve().parents[2] / "blender_ai_tools"

# results.py の Measurement の説明に書いた単位
UNITS = {"m", "m2", "m3", "deg", "ratio", "count", "px", "s"}
SNAKE_CASE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
JAPANESE = re.compile(r"[぀-ヿ一-鿿]")

# 部品の中から thresholds.py へ移した基準の数字：(読み込む部品, 名前)
MOVED_THRESHOLDS = [
    (catalog, "MAX_TOOL_CALL_LENGTH"),
    (schema, "MAX_SCHEMA_DEPTH"),
    (provenance, "GIT_TIMEOUT_SECONDS"),
    (visibility, "CAMERA_NEAR"),
    (visibility, "CAMERA_FAR"),
    (mesh_check, "LOOSE_PART_MAX_FACES"),
    (deformation, "VOLUME_MIN_RATIO"),
    (roundtrip, "ORDER_MATCH_MIN_FRACTION"),
    (roundtrip, "SKELETON_MATRIX_TOLERANCE"),
    (roundtrip, "POSE_MOVED_TOLERANCE"),
]


def top_level_assignments(path: Path) -> set[str]:
    """ファイルの先頭（関数・クラスの外）で値を入れている名前"""
    names: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(
            node, (ast.AnnAssign, ast.AugAssign)) else []
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def imported_from_thresholds(path: Path) -> set[str]:
    """from ..thresholds import X（相対の import）で読み込んだ名前"""
    names: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ImportFrom) and node.level > 0 and node.module == "thresholds":
            names.update(alias.asname or alias.name for alias in node.names)
    return names


# MARK: - 基準の数字の置き場所

class ThresholdLocationTest(unittest.TestCase):
    def test_no_provisional_markers_left_in_modules(self):
        """「thresholds.py へ移す」の印が、thresholds.py のほかに残っていない"""
        for path in sorted(PACKAGE_DIR.rglob("*.py")):
            if path.name == "thresholds.py":
                continue
            with self.subTest(path=str(path.relative_to(PACKAGE_DIR))):
                self.assertNotIn("thresholds.py へ移す", path.read_text(encoding="utf-8"))

    def test_moved_thresholds_are_read_from_thresholds(self):
        """部品は名前を thresholds から読み込み、自分では数を書かない（同じ名前で値が2つにならない）"""
        for module, name in MOVED_THRESHOLDS:
            path = Path(module.__file__)
            with self.subTest(module=module.__name__, name=name):
                self.assertIn(name, vars(thresholds))
                self.assertNotIn(name, top_level_assignments(path))
                self.assertIn(name, imported_from_thresholds(path))
                self.assertEqual(getattr(module, name), getattr(thresholds, name))

    def test_moved_thresholds_are_marked_provisional(self):
        """thresholds.py で、移した数字のすぐ上の説明に「仮」と書いてある（まだ決まっていない数字）"""
        lines = (PACKAGE_DIR / "thresholds.py").read_text(encoding="utf-8").splitlines()
        for _, name in MOVED_THRESHOLDS:
            with self.subTest(name=name):
                row = next(i for i, line in enumerate(lines) if line.startswith(f"{name} = "))
                comments = []
                for line in reversed(lines[:row]):
                    if not line.startswith("#") or line.startswith("# MARK"):
                        break
                    comments.append(line)
                self.assertTrue(comments, f"{name} の上に説明がありません")
                self.assertIn("仮", "".join(comments))

    def test_values(self):
        """移す前と同じ値（移したことで、ふるまいが変わらない）"""
        self.assertEqual(thresholds.MAX_TOOL_CALL_LENGTH, 65536)
        self.assertEqual(thresholds.MAX_SCHEMA_DEPTH, 32)
        self.assertEqual(thresholds.GIT_TIMEOUT_SECONDS, 10.0)
        self.assertEqual(thresholds.CAMERA_NEAR, 0.01)  # メートル（10mm）
        self.assertEqual(thresholds.CAMERA_FAR, 1000.0)
        self.assertEqual(thresholds.LOOSE_PART_MAX_FACES, 4)
        self.assertEqual(thresholds.VOLUME_MIN_RATIO, 1e-6)
        self.assertEqual(thresholds.ORDER_MATCH_MIN_FRACTION, 0.5)
        self.assertEqual(thresholds.SKELETON_MATRIX_TOLERANCE, 1e-5)
        self.assertEqual(thresholds.POSE_MOVED_TOLERANCE, 1e-6)

    def test_defaults_use_thresholds(self):
        """関数の既定の値が thresholds の数字になっている"""
        def default(function, parameter):
            return inspect.signature(function).parameters[parameter].default

        self.assertEqual(default(catalog.parse_tool_call, "max_length"), thresholds.MAX_TOOL_CALL_LENGTH)
        self.assertEqual(default(mesh_check.loose_parts, "max_faces"), thresholds.LOOSE_PART_MAX_FACES)
        self.assertEqual(default(mesh_check.summarize_loose_parts, "max_faces"), thresholds.LOOSE_PART_MAX_FACES)
        for function in (visibility.look_at, visibility.standard_views):
            with self.subTest(function=function.__name__):
                self.assertEqual(default(function, "near"), thresholds.CAMERA_NEAR)
                self.assertEqual(default(function, "far"), thresholds.CAMERA_FAR)
        camera = visibility.look_at((0.0, -3.0, 1.0), (0.0, 0.0, 1.0))
        self.assertEqual((camera.near, camera.far), (thresholds.CAMERA_NEAR, thresholds.CAMERA_FAR))

    def test_loose_part_threshold_is_inclusive(self):
        """LOOSE_PART_MAX_FACES 個ちょうどの部品は小さい部品、1つ多いと入らない（既定の値で）"""
        big_v, big_t = shapes.icosphere(subdivisions=1)
        limit = thresholds.LOOSE_PART_MAX_FACES
        for faces, expected in ((limit, 1), (limit + 1, 0)):
            with self.subTest(faces=faces):
                small_v, small_t = shapes.grid(nx=faces, ny=1)
                small_t = small_t[:faces] + len(big_v)  # 帯の三角形を faces 個だけ使う
                v = np.vstack([big_v, small_v + [5.0, 0.0, 0.0]])
                t = np.vstack([big_t, small_t])
                self.assertEqual(len(mesh_check.loose_parts(v, t)), expected)

    def test_volume_min_ratio_boundary(self):
        """体積 / 表面積^1.5 が VOLUME_MIN_RATIO 以下の形は測らない（None）。基準をはさむ2つの形で確かめる"""
        v, t = shapes.box()
        area = float(geometry.triangle_areas(v, t).sum())
        volume = deformation.mesh_volume(v, t)
        self.assertIsNotNone(deformation.volume_change_ratio(v, v, t))
        # Z を縮めて、体積 / 表面積^1.5 を基準の下（約 3.5e-8）と上（約 3.5e-6）にする（Z の大きさ h：体積 h、表面積 2 + 4h）
        for scale, expected_none in ((1e-7, True), (1e-5, False)):
            with self.subTest(scale=scale):
                flat = v * [1.0, 1.0, scale]
                ratio = deformation.mesh_volume(flat, t) / float(geometry.triangle_areas(flat, t).sum()) ** 1.5
                self.assertEqual(ratio <= thresholds.VOLUME_MIN_RATIO, expected_none)
                self.assertEqual(deformation.volume_change_ratio(flat, flat, t) is None, expected_none)
        self.assertGreater(volume / area ** 1.5, thresholds.VOLUME_MIN_RATIO)


# MARK: - 入力のエラーは日本語

def ragged() -> list[list[float]]:
    """行の長さがそろわない並び（np.asarray は英語の ValueError を出す）"""
    return [[0.0, 0.0, 0.0], [1.0, 0.0]]


class JapaneseErrorTest(unittest.TestCase):
    def assertJapaneseValueError(self, function, *args, **kwargs):
        with self.assertRaises(ValueError) as caught:
            function(*args, **kwargs)
        message = str(caught.exception)
        self.assertRegex(message, JAPANESE, f"日本語の説明がありません：{message}")
        self.assertNotIn("inhomogeneous", message)

    def test_ragged_inputs_in_every_measure_module(self):
        box_v, box_t = shapes.box()
        table = weights.WeightTable([[0]], [[1.0]])
        camera = visibility.look_at((0.0, -3.0, 0.0), (0.0, 0.0, 0.0), width=8, height=8)
        cases = {
            "geometry.as_points": (geometry.as_points, ragged()),
            "geometry.closest_points": (geometry.closest_points, ragged(), box_v, box_t),
            "geometry.summarize_penetration": (geometry.summarize_penetration, [[0.0], [0.0, 1.0]]),
            "symmetry.asymmetry": (symmetry.asymmetry, ragged()),
            "mesh_check.summarize_mesh": (mesh_check.summarize_mesh, box_v, [[0, 1, 2], [0, 1]]),
            "deformation.displacement": (deformation.displacement, ragged(), ragged()),
            "deformation.unique_edges": (deformation.unique_edges, [[0, 1, 2], [0, 1]]),
            "deformation.summarize_stretch": (deformation.summarize_stretch, [[1.0], [1.0, 2.0]]),
            "deformation.summarize_flips": (deformation.summarize_flips, [[True], [True, False]]),
            "weights.WeightTable": (weights.WeightTable, [[0], [0, 1]], [[1.0], [0.5, 0.5]]),
            "weights.from_dense": (weights.WeightTable.from_dense, [[1.0], [0.5, 0.5]]),
            "weights.linear_blend_skinning": (weights.linear_blend_skinning, ragged(), table, np.eye(4)[None]),
            "weights.bone_matrices": (weights.linear_blend_skinning, [[0.0, 0.0, 0.0]], table,
                                      [np.eye(4).tolist(), [[1.0]]]),
            "kinematics.angle_at": (kinematics.angle_at, ragged(), [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            "kinematics.summarize_reverse_bend": (kinematics.summarize_reverse_bend, [[1.0], [1.0, 2.0]]),
            "kinematics.summarize_sink": (kinematics.summarize_sink, [[0.0], [0.0, 1.0]]),
            "kinematics.spans": (kinematics.spans, [[True], [True, False]]),
            "kinematics.reach_ratios": (kinematics.reach_ratios, [[0.0, 0.0, 0.0]] * 2, [[0.0, 0.0, 1.0]] * 2,
                                        [[1.0], [1.0, 2.0]]),
            "kinematics.quat_angle_deg": (kinematics.quat_angle_deg, [[1.0, 0.0, 0.0, 0.0], [1.0]],
                                          [1.0, 0.0, 0.0, 0.0]),
            "kinematics.step_angles_deg": (kinematics.step_angles_deg, [[[1.0, 0.0, 0.0, 0.0]], [[1.0]]]),
            "visibility.rasterize": (visibility.rasterize, ragged(), box_t, camera),
        }
        for label, (function, *args) in cases.items():
            with self.subTest(function=label):
                self.assertJapaneseValueError(function, *args)

    def test_weights_deform_ragged(self):
        table = weights.WeightTable([[0]], [[1.0]])
        self.assertJapaneseValueError(weights.check, table, deform=[[True], [True, False]])
        self.assertJapaneseValueError(weights.linear_blend_skinning, [[0.0, 0.0, 0.0]], table, np.eye(4)[None],
                                      deform=[[True], [True, False]])

    def test_bool_is_not_a_number_or_count(self):
        """真偽（True・False）は、数・個数・点の並びとして受け付けない（1 と 0 にまちがえない）"""
        v, t = shapes.box()
        cases = {
            "geometry chunk_size": lambda: geometry.closest_points(v, v, t, chunk_size=True),
            "symmetry chunk_size": lambda: symmetry.nearest_distances(v, v, chunk_size=True),
            "mesh_check max_faces": lambda: mesh_check.loose_parts(v, t, max_faces=True),
            "deformation tolerance": lambda: deformation.changed_vertices(v, v, tolerance=False),
            "weights limit": lambda: weights.check(weights.WeightTable([[0]], [[1.0]]), limit=True),
            "kinematics chunk_size": lambda: kinematics.step_angles_deg(np.ones((2, 1, 4)), chunk_size=True),
            "kinematics tolerance_deg": lambda: kinematics.reverse_bend_frames([0.0], tolerance_deg=True),
            "visibility width": lambda: visibility.look_at((0.0, -3.0, 0.0), (0.0, 0.0, 0.0), width=True),
            "bool points": lambda: geometry.as_points(np.ones((2, 3), dtype=bool)),
            "bool rest": lambda: deformation.displacement(np.ones((2, 3), dtype=bool), v[:2]),
            "bool vectors": lambda: kinematics.angle_at([True, False, True], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        }
        for label, call in cases.items():
            with self.subTest(case=label), self.assertRaises(ValueError):
                call()


# MARK: - 結果の名前・単位・灰

# 点より下にある箱（点を隠さない）。renders_for はこれを写す。summarize_visible_points の mesh にも渡す
RENDER_MESH = shapes.box(size=(1.0, 1.0, 1.0), center=(0.0, 0.0, -1.0))


def renders_for(points: np.ndarray) -> dict:
    """点のまわりを写す、小さな画像のカメラ（前から1つ）"""
    v, t = RENDER_MESH
    camera = visibility.look_at((0.0, -3.0, 0.0), (0.0, 0.0, 0.0), width=32, height=32)
    return visibility.render_views(v, t, {"front": camera})


def all_summaries(empty: bool) -> list[tuple[str, Measurement]]:
    """どの summarize も1回ずつ呼ぶ。empty=True のときは、調べる物が空の入力で呼ぶ"""
    v, t = shapes.icosphere(subdivisions=1)
    if empty:
        v0, t0 = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
        table = weights.WeightTable(np.zeros((0, 1), dtype=np.int64), np.zeros((0, 1)))
        return [
            ("penetration", geometry.summarize_penetration([])),
            ("gap", geometry.summarize_gap([])),
            ("asymmetry", symmetry.summarize_asymmetry([])),
            *(("mesh", m) for m in mesh_check.summarize_mesh(v0, t0, closed_required=True)),
            ("changes", deformation.summarize_changes(v0, v0)),
            ("stretch", deformation.summarize_stretch([])),
            ("flips", deformation.summarize_flips(np.zeros(0, dtype=bool))),
            ("volume", deformation.summarize_volume(None)),
            ("spikes", deformation.summarize_spikes([])),
            *(("weights", m) for m in weights.summarize(table)),
            ("reverse_bend", kinematics.summarize_reverse_bend([])),
            ("sink", kinematics.summarize_sink([])),
            ("slide", kinematics.summarize_slide([])),
            ("unreachable", kinematics.summarize_unreachable(np.zeros(0, dtype=bool))),
            ("jumps", kinematics.summarize_jumps(np.zeros((0, 3), dtype=bool))),
            ("visible", visibility.summarize_visible_points(np.zeros((0, 3)), {}, mesh=RENDER_MESH)),
        ]
    table = weights.WeightTable.from_dense(np.tile([[0.5, 0.5]], (len(v), 1)))
    point = np.array([[0.0, 0.0, 0.0]])
    return [
        ("penetration", geometry.summarize_penetration(geometry.penetration_depths(v * 1.5, v, t))),
        ("gap", geometry.summarize_gap(geometry.gaps(v * 1.002, v, t))),
        ("asymmetry", symmetry.summarize_asymmetry(symmetry.asymmetry(v))),
        *(("mesh", m) for m in mesh_check.summarize_mesh(v, t, closed_required=True)),
        ("changes", deformation.summarize_changes(v, v)),
        ("stretch", deformation.summarize_stretch(deformation.edge_length_ratios(v, v, deformation.unique_edges(t)))),
        ("flips", deformation.summarize_flips(*deformation.triangle_flips(v, v, t))),
        ("volume", deformation.summarize_volume(deformation.volume_change_ratio(v, v, t))),
        ("spikes", deformation.summarize_spikes(deformation.spike_scores(v, v, t))),
        *(("weights", m) for m in weights.summarize(table)),
        ("reverse_bend", kinematics.summarize_reverse_bend([10.0, 20.0])),
        ("sink", kinematics.summarize_sink([0.0, 0.0])),
        ("slide", kinematics.summarize_slide([(0, 3, 0.0)])),
        ("unreachable", kinematics.summarize_unreachable(np.array([False, False]))),
        ("jumps", kinematics.summarize_jumps(np.zeros((2, 3), dtype=bool))),
        ("visible", visibility.summarize_visible_points(point, renders_for(point), mesh=RENDER_MESH)),
    ]


# 部品ごとの結果の名前の定数（報告の文の形は、この名前で引く）
NAME_CONSTANTS = {
    geometry.PENETRATION_NAME, geometry.GAP_NAME, symmetry.ASYMMETRY_NAME,
    mesh_check.DUPLICATES_NAME, mesh_check.DEGENERATE_NAME, mesh_check.NON_MANIFOLD_NAME,
    mesh_check.INCONSISTENT_NAME, mesh_check.HOLES_NAME, mesh_check.LOOSE_PARTS_NAME,
    deformation.CHANGES_NAME, deformation.STRETCH_NAME, deformation.FLIPS_NAME, deformation.VOLUME_NAME,
    deformation.SPIKES_NAME,
    weights.OVER_LIMIT_NAME, weights.UNWEIGHTED_NAME, weights.NOT_NORMALIZED_NAME, weights.NEGATIVE_NAME,
    weights.DUPLICATE_NAME,
    kinematics.REVERSE_BEND_NAME, kinematics.FLOOR_SINK_NAME, kinematics.FOOT_SLIDE_NAME,
    kinematics.UNREACHABLE_NAME, kinematics.BONE_JUMPS_NAME, visibility.VISIBLE_POINTS_NAME,
}


class MeasurementConventionsTest(unittest.TestCase):
    def test_names_and_units(self):
        results = all_summaries(empty=False)
        seen = set()
        for label, result in results:
            with self.subTest(summary=label, name=result.name):
                self.assertIsInstance(result, Measurement)
                self.assertRegex(result.name, SNAKE_CASE)
                self.assertIn(result.name, NAME_CONSTANTS)
                self.assertIn(result.unit, UNITS)
                seen.add(result.name)
        self.assertEqual(seen, NAME_CONSTANTS, "名前の定数と、summarize の返す名前が1対1でありません")
        self.assertEqual(len(NAME_CONSTANTS), 25, "名前が重なっています")

    def test_good_samples_are_not_red_or_gray(self):
        """壊れていない形・動き（そろった球、同じ姿勢）では、赤も灰も出ない（見える点は1つあるので赤）"""
        for label, result in all_summaries(empty=False):
            with self.subTest(summary=label, name=result.name):
                if result.name == visibility.VISIBLE_POINTS_NAME:
                    self.assertEqual(result.signal, Signal.RED)
                    continue
                self.assertNotIn(result.signal, (Signal.RED, Signal.GRAY), result.reason)

    def test_empty_inputs_are_gray_with_reason(self):
        """調べる物が無いときは、どれも灰で、理由（日本語）が書いてある（「0 個」で合格にしない）"""
        for label, result in all_summaries(empty=True):
            with self.subTest(summary=label, name=result.name):
                self.assertEqual(result.signal, Signal.GRAY)
                self.assertIsNone(result.value)
                self.assertRegex(result.reason, JAPANESE)

    def test_units_match_quantities(self):
        """長さは m、角度は deg、割合は ratio、数は count（単位の取り違えが無い）"""
        expected = {
            geometry.PENETRATION_NAME: "m", geometry.GAP_NAME: "count", symmetry.ASYMMETRY_NAME: "m",
            deformation.STRETCH_NAME: "ratio", deformation.VOLUME_NAME: "ratio", deformation.SPIKES_NAME: "ratio",
            deformation.FLIPS_NAME: "count", deformation.CHANGES_NAME: "count",
            kinematics.REVERSE_BEND_NAME: "deg", kinematics.FLOOR_SINK_NAME: "m", kinematics.FOOT_SLIDE_NAME: "m",
            kinematics.UNREACHABLE_NAME: "count", kinematics.BONE_JUMPS_NAME: "count",
        }
        units = {result.name: result.unit for _, result in all_summaries(empty=False)}
        for name, unit in expected.items():
            with self.subTest(name=name):
                self.assertEqual(units[name], unit)

    def test_visibility_name_must_be_snake_case(self):
        point = np.array([[0.0, 0.0, 0.0]])
        renders = renders_for(point)
        result = visibility.summarize_visible_points(point, renders, mesh=RENDER_MESH, name="visible_penetration")
        self.assertEqual(result.name, "visible_penetration")
        for bad in ("Visible", "visible points", "見える", "", "_x", 3, None):
            with self.subTest(name=bad), self.assertRaises(ValueError):
                visibility.summarize_visible_points(point, renders, mesh=RENDER_MESH, name=bad)


# E2 往復の確認（export/roundtrip.py）と A1 登録カード（model/card.py）の結果の名前 → 単位
ROUNDTRIP_UNITS = {
    roundtrip.DEVIATION_NAME: "m", roundtrip.COVERAGE_NAME: "count", roundtrip.SKELETON_NAME: "m",
    roundtrip.JOINT_NAMES_NAME: "count", roundtrip.HUMANOID_NAME: "count", roundtrip.SHAPE_KEYS_NAME: "count",
    roundtrip.INFLUENCES_NAME: "count", roundtrip.WEIGHT_SUMS_NAME: "ratio", roundtrip.PHYSICS_NAME: "count",
}
CARD_UNITS = {
    card.SEMI_STANDARD_NAME: "count", card.UNKNOWN_BONES_NAME: "count", card.UNCONFIRMED_PARTS_NAME: "count",
    card.BODY_MEASUREMENTS_NAME: "count", card.SOURCE_TERMS_NAME: "count", card.CARD_CURRENT_NAME: "count",
}


def empty_roundtrip_results() -> list[Measurement]:
    """E2 の確かめを、骨も形も無い書き出しで1回ずつ呼ぶ（rig の bones も空）"""
    model = roundtrip.ExportedModel(joint_paths=(), bind=np.zeros((0, 4, 4)), rest_local=None, meshes=())
    rig = {"version": 1, "bones": {}, "expressions": {}, "originalNames": {"bones": {}, "shapeKeys": {}}}
    return [
        roundtrip.check_coverage(model, {}, {}),
        roundtrip.compare_rest(model, {}, np.zeros((0, 4, 4)), {}),
        roundtrip.check_skeleton(model),
        *roundtrip.check_names(rig, model, [], []),
        *roundtrip.check_influences(model),
        roundtrip.check_no_physics(model, []),
    ]


class ExportAndCardConventionsTest(unittest.TestCase):
    """E2・A1 の結果も、A4 と同じ決まり（snake_case の名前・決まった単位・名前が重ならない・調べる物が無ければ灰）"""

    def test_names_are_snake_case_and_distinct(self):
        groups = {"measure": NAME_CONSTANTS, "roundtrip": set(ROUNDTRIP_UNITS), "card": set(CARD_UNITS)}
        self.assertEqual(len(ROUNDTRIP_UNITS), 9)
        self.assertEqual(len(CARD_UNITS), 6)
        for label, names in groups.items():
            for name in names:
                with self.subTest(group=label, name=name):
                    self.assertRegex(name, SNAKE_CASE)
        self.assertFalse(NAME_CONSTANTS & set(ROUNDTRIP_UNITS))
        self.assertFalse(NAME_CONSTANTS & set(CARD_UNITS))
        self.assertFalse(set(ROUNDTRIP_UNITS) & set(CARD_UNITS))
        for unit in (*ROUNDTRIP_UNITS.values(), *CARD_UNITS.values()):
            self.assertIn(unit, UNITS)

    def test_roundtrip_units_and_empty_inputs(self):
        """E2 の確かめは、骨も形も無いとき「問題 0」の緑にしない（物理の部品と、シェイプキーの名前は除く：
        無いことを確かめる物と、キーも表情も無いモデルで食い違いが無いことを確かめる物）"""
        results = empty_roundtrip_results()
        self.assertEqual({m.name for m in results}, set(ROUNDTRIP_UNITS))
        for result in results:
            with self.subTest(name=result.name):
                self.assertEqual(result.unit, ROUNDTRIP_UNITS[result.name])
                if result.name in (roundtrip.PHYSICS_NAME, roundtrip.SHAPE_KEYS_NAME):
                    self.assertEqual(result.signal, Signal.GREEN)
                    continue
                self.assertEqual(result.signal, Signal.GRAY)
                self.assertIsNone(result.value)
                self.assertRegex(result.reason, JAPANESE)

    def test_card_summary_units(self):
        bones = card.BoneArrays(names=("センター",), parents=np.array([-1]), heads=np.zeros((1, 3)),
                                tails=np.array([[0.0, 0.0, 0.1]]), deform=np.array([True]),
                                matrices=np.eye(4)[None])
        model_card = card.make_card(blend_name="m.blend", armature_name="Arm", bones=bones, meshes=[],
                                    created_at="2026-01-01T00:00:00+00:00", tool_version="0", blender_version="5.2.2")
        results = card.summarize_card(model_card, [])
        self.assertEqual({m.name for m in results}, set(CARD_UNITS))
        for result in results:
            with self.subTest(name=result.name):
                self.assertEqual(result.unit, CARD_UNITS[result.name])
        by_name = {m.name: m for m in results}
        # 部品が無い・測れない値があるときは灰（緑にしない）
        self.assertEqual(by_name[card.UNCONFIRMED_PARTS_NAME].signal, Signal.GRAY)
        self.assertEqual(by_name[card.BODY_MEASUREMENTS_NAME].signal, Signal.GRAY)


# MARK: - 測れなかった値の扱い（赤でないかぎり灰）

class PartialUnmeasuredTest(unittest.TestCase):
    """測れなかった値（NaN）が一部にあるとき、どの道具も同じ決まりにする：
    測れた所が赤なら赤、そうでなければ灰（緑・黄にしない）。理由は日本語で書く。値は測れた所から決めたまま"""

    # (名前, 関数, 緑の入力, 黄の入力（黄が無い道具は None）, 赤の入力)。どの入力にも NaN が1つ入っている
    CASES = [
        ("penetration", geometry.summarize_penetration,
         [0.0, math.nan], [thresholds.PENETRATION_YELLOW, math.nan], [thresholds.PENETRATION_RED, math.nan]),
        ("gap", geometry.summarize_gap,
         [thresholds.mm(2.0), math.nan], [thresholds.mm(9.0), math.nan], None),
        ("stretch", deformation.summarize_stretch,
         [1.0, math.nan], [thresholds.EDGE_STRETCH_YELLOW, math.nan], [thresholds.EDGE_SHRINK_RED, math.nan]),
        ("spikes", deformation.summarize_spikes,
         [0.0, math.nan], [thresholds.SPIKE_YELLOW, math.nan], [thresholds.SPIKE_RED, math.nan]),
        ("reverse_bend", kinematics.summarize_reverse_bend,
         [5.0, math.nan], None, [-thresholds.REVERSE_BEND_TOLERANCE_DEG - 1.0, math.nan]),
        ("sink", kinematics.summarize_sink,
         [0.0, math.nan], None, [thresholds.FLOOR_SINK_TOLERANCE * 2.0, math.nan]),
        ("slide", kinematics.summarize_slide,
         [(0, 2, 0.0), (4, 5, math.nan)], None, [(0, 2, thresholds.FOOT_SLIDE_TOLERANCE * 2.0), (4, 5, math.nan)]),
    ]

    def test_gray_unless_red(self):
        for name, function, green, yellow, red in self.CASES:
            for label, data, expected in (("green", green, Signal.GRAY), ("yellow", yellow, Signal.GRAY),
                                          ("red", red, Signal.RED)):
                if data is None:
                    continue
                with self.subTest(summary=name, measured=label):
                    result = function(data)
                    self.assertEqual(result.signal, expected)
                    self.assertIsNotNone(result.value)
                    self.assertRegex(result.reason, JAPANESE)

    def test_without_nan_the_measured_signal_is_kept(self):
        """NaN を除くと、緑・黄・赤がそのまま出る（上の灰が、NaN だけで決まっていることの確かめ）"""
        for name, function, green, yellow, red in self.CASES:
            for data, expected in ((green, Signal.GREEN), (yellow, Signal.YELLOW), (red, Signal.RED)):
                if data is None:
                    continue
                if name == "slide":
                    clean = [item for item in data if not math.isnan(item[2])]
                else:
                    clean = [value for value in data if not math.isnan(value)]
                with self.subTest(summary=name, expected=expected):
                    self.assertEqual(function(clean).signal, expected)

    def test_flips_with_uncompared_triangles(self):
        """裏返りの点検でも同じ：向きを比べられなかった三角形があれば、裏返りが無くても灰、あれば赤"""
        result = deformation.summarize_flips(np.array([False, False]), np.array([True, False]))
        self.assertEqual(result.signal, Signal.GRAY)
        result = deformation.summarize_flips(np.array([True, False]), np.array([False, True]))
        self.assertEqual(result.signal, Signal.RED)

    def test_flips_with_real_degenerate_triangle(self):
        """休みの形でつぶれた三角形があると、triangle_flips → summarize_flips は灰になる（緑にしない）"""
        v, t = shapes.icosphere(subdivisions=1)
        v = np.vstack([v, [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]]])  # 1本の線に並んだ3点
        t = np.vstack([t, [[len(v) - 3, len(v) - 2, len(v) - 1]]])
        flipped, degenerate = deformation.triangle_flips(v, v, t)
        self.assertEqual(int(degenerate.sum()), 1)
        self.assertEqual(deformation.summarize_flips(flipped, degenerate).signal, Signal.GRAY)

    def test_visibility_unmeasured_outranks_yellow(self):
        """見え方でも同じ：どのカメラでも調べられない点があれば、黄で見える点があっても灰"""
        inside = np.array([[0.0, 0.0, 0.0]])
        outside = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])  # 2つ目は画像の枠の外
        renders = renders_for(inside)
        self.assertEqual(visibility.summarize_visible_points(inside, renders, mesh=RENDER_MESH, signal_if_visible=Signal.YELLOW).signal,
                         Signal.YELLOW)
        self.assertEqual(visibility.summarize_visible_points(outside, renders, mesh=RENDER_MESH, signal_if_visible=Signal.YELLOW).signal,
                         Signal.GRAY)
        self.assertEqual(visibility.summarize_visible_points(outside, renders, mesh=RENDER_MESH).signal, Signal.RED)


# MARK: - 2か所で計算する物が、同じ答えになる

def awkward_meshes() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """つぶれた三角形・同じ頂点を2回使う三角形・3つの面が集まる辺・離れた部品を含む形"""
    sphere_v, sphere_t = shapes.icosphere(subdivisions=1)
    grid_v, grid_t = shapes.grid(nx=3, ny=2)
    fan_v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, -1.0, 0.0], [0.5, 0.0, 1.0]])
    fan_t = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])  # 辺 0-1 を3つの面が使う
    mixed_v = np.vstack([sphere_v, [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]]])
    n = len(sphere_v)
    mixed_t = np.vstack([sphere_t, [[n, n + 1, n + 2], [n, n, n + 1], [3, 3, 3]]])
    return {"sphere": (sphere_v, sphere_t), "grid": (grid_v, grid_t), "fan": (fan_v, fan_t),
            "mixed": (mixed_v, mixed_t), "empty": (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))}


class SharedComputationTest(unittest.TestCase):
    def test_edge_lists_agree(self):
        """deformation.unique_edges と mesh_check.edge_face_counts の辺は、同じ並びの同じ辺"""
        for label, (_, t) in awkward_meshes().items():
            with self.subTest(mesh=label):
                edges, counts = mesh_check.edge_face_counts(t)
                np.testing.assert_array_equal(deformation.unique_edges(t), edges)
                self.assertEqual(deformation.unique_edges(t).dtype, np.int64)
                # 使う回数は、向きのある辺（同じ頂点どうしを除く）の数を足した物
                half_edges = sum(a != b for tri in t.tolist()
                                 for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])))
                self.assertEqual(int(counts.sum()), half_edges)

    def test_degenerate_triangles_agree(self):
        """面積 0 とみなす三角形は、mesh_check と deformation（裏返りで比べない三角形）で同じ。基準ちょうども同じ側"""
        v, t = awkward_meshes()["mixed"]
        for area in (thresholds.DEGENERATE_AREA, float(geometry.triangle_areas(v, t)[0])):
            with self.subTest(area=area):
                expected = mesh_check.degenerate_triangles(v, t, area=area)
                _, degenerate = deformation.triangle_flips(v, v, t, local_rotation=False, degenerate_area=area)
                np.testing.assert_array_equal(np.flatnonzero(degenerate), expected)
        # 基準を三角形0の面積ちょうどにすると、三角形0も入る（両方とも「以下」）
        area0 = float(geometry.triangle_areas(v, t)[0])
        self.assertIn(0, mesh_check.degenerate_triangles(v, t, area=area0).tolist())

    def test_volumes_agree(self):
        """deformation.mesh_volume は、テスト用の signed_volume と同じ（外向きの閉じた形なら正）"""
        for v, t in (shapes.box(size=(1.0, 2.0, 3.0), center=(0.5, -1.0, 2.0)), shapes.icosphere(subdivisions=2)):
            self.assertAlmostEqual(deformation.mesh_volume(v, t), shapes.signed_volume(v, t), places=12)
        v, t = shapes.box(size=(1.0, 2.0, 3.0))
        self.assertAlmostEqual(deformation.mesh_volume(v, t), 6.0, places=12)
        self.assertAlmostEqual(deformation.mesh_volume(v, t[:, [0, 2, 1]]), -6.0, places=12)  # 裏返すと負

    def test_closed_shape_rules_agree(self):
        """体積を測れる形（volume_change_ratio が None でない）は、mesh_check で穴も向きの合わない辺も無い形"""
        v, t = shapes.box()
        flipped_one = t.copy()
        flipped_one[0] = flipped_one[0, [0, 2, 1]]
        cases = {"closed": t, "hole": t[1:], "flipped": flipped_one,
                 "fixed": mesh_check.orient_consistently(v, flipped_one)[0]}
        for label, triangles in cases.items():
            with self.subTest(mesh=label):
                closed = (len(mesh_check.boundary_edges(triangles)) == 0
                          and len(mesh_check.inconsistent_edges(triangles)) == 0)
                ratio = deformation.volume_change_ratio(v, v * 1.1, triangles)
                self.assertEqual(ratio is not None, closed)
                if closed:
                    self.assertAlmostEqual(ratio, 1.1 ** 3 - 1.0, places=12)
        self.assertTrue(np.array_equal(cases["fixed"], t))

    def test_exact_duplicates_agree(self):
        """座標がまったく同じ頂点のまとまり：geometry.weld_exact と mesh_check.duplicate_vertex_groups(距離 0) が同じ"""
        v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                      [1.0, 0.0, 1e-12]])
        groups = mesh_check.duplicate_vertex_groups(v, 0.0)
        welded = geometry.weld_exact(v)
        from_weld = [np.flatnonzero(welded == root) for root in np.unique(welded)]
        from_weld = [group for group in from_weld if len(group) > 1]
        self.assertEqual([g.tolist() for g in groups], [g.tolist() for g in from_weld])
        self.assertEqual([g.tolist() for g in groups], [[0, 2, 3], [1, 4]])

    def test_worst_lists_have_the_same_length_and_order(self):
        """「いちばん悪い物」の一覧は、どの道具も同じ長さ（WORST_LIST_LENGTH）で、悪い順・同じなら番号の小さい順"""
        values = np.array([0.5] * 15 + [2.0])
        result = deformation.summarize_spikes(values)
        self.assertEqual(result.details["worst_indices"].tolist(), [15] + list(range(geometry.WORST_LIST_LENGTH - 1)))
        asym = symmetry.summarize_asymmetry(values * 1e-3)
        self.assertEqual(asym.details["worst_indices"].tolist(), result.details["worst_indices"].tolist())
        shrink = deformation.summarize_stretch(np.array([0.9] * 15 + [0.4]))
        self.assertEqual(shrink.details["worst_shrink_indices"].tolist(),
                         [15] + list(range(geometry.WORST_LIST_LENGTH - 1)))
        self.assertEqual(deformation.WORST_LIST_LENGTH, geometry.WORST_LIST_LENGTH)

    def test_change_tolerance_matches_provenance(self):
        """「再現性の幅ちょうどの差は、変化にしない」が、版の比べ（deformation）と記録の比べ（provenance）で同じ"""
        tol = thresholds.REPRO_TOLERANCE_LENGTH
        before = np.zeros((2, 3))
        for delta, changed in ((tol, False), (np.nextafter(tol, 1.0), True)):
            with self.subTest(delta=delta):
                after = before.copy()
                after[1, 0] = delta
                self.assertEqual(len(deformation.changed_vertices(before, after)) == 1, changed)
                self.assertEqual(deformation.summarize_changes(before, after).value, 1.0 if changed else 0.0)
                differences = provenance.compare_results({"x": 0.0}, {"x": float(delta)}, tolerances={"x": tol})
                self.assertEqual(len(differences) == 1, changed)


# MARK: - 向きの決まり（Z が上、モデルは -Y を向く、キャラの左は +X）

class AxisConventionTest(unittest.TestCase):
    def test_front_camera_and_knee_agree_on_backward(self):
        """前のカメラはモデルの前（-Y）から +Y の向きを見る。ふつうのひざの曲がりでは、足首は後ろ（+Y）へ動く"""
        views = visibility.standard_views((0.0, 0.0, 1.0), 3.0)
        backward = views["front"].forward
        np.testing.assert_allclose(backward, [0.0, 1.0, 0.0], atol=1e-12)
        self.assertLess(views["front"].position[1], 0.0)
        hip, knee = np.array([0.1, 0.0, 1.0]), np.array([0.1, 0.0, 0.5])
        lateral = views["left"].position - views["right"].position  # キャラの左の向き
        normal = kinematics.knee_bend(hip, knee, knee + 0.5 * backward, lateral)
        reverse = kinematics.knee_bend(hip, knee, knee - 0.5 * backward, lateral)
        self.assertAlmostEqual(float(normal), 90.0, places=9)
        self.assertAlmostEqual(float(reverse), -90.0, places=9)

    def test_left_is_plus_x_everywhere(self):
        """キャラの左（+X）：見え方の "left" カメラは +X の側。左右を映すと +X の点は -X（"right" の側）へ行く"""
        views = visibility.standard_views((0.0, 0.0, 0.0), 2.0)
        self.assertGreater(views["left"].position[0], 0.0)
        self.assertLess(views["right"].position[0], 0.0)
        left_point = np.array([[0.3, -0.1, 1.2]])
        mirrored = symmetry.mirror_x(left_point)
        np.testing.assert_array_equal(mirrored, [[-0.3, -0.1, 1.2]])
        # 左だけに出っ張った点は、左右のずれとして見つかる（映す軸が X でなければ、ずれが出ない）
        v, _ = shapes.icosphere(subdivisions=1)
        moved = v.copy()
        right_most = int(np.argmax(v[:, 0]))
        moved[right_most, 0] += 0.01
        self.assertAlmostEqual(float(np.nanmax(symmetry.asymmetry(moved))), 0.01, places=9)
        self.assertEqual(float(np.nanmax(symmetry.asymmetry(v))), 0.0)

    def test_floor_is_z(self):
        """床は Z（上が +Z）：沈みは Z だけで決まり、X・Y の値は関係ない"""
        depths = kinematics.sink_depths([[5.0, -7.0, -0.003], [0.0, 0.0, 0.003], [-0.003, -0.003, 0.0]])
        np.testing.assert_allclose(depths, [0.003, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()

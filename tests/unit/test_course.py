"""A5 確かめの動き：ポーズごとの確かめ・まとめ・報告の文（motion/course.py）を確かめる（bpy なし）"""

import math
import unittest

import numpy as np

from blender_ai_tools import texts, thresholds
from blender_ai_tools.measure.results import Measurement, Signal
from blender_ai_tools.measure.visibility import project, render_views
from blender_ai_tools.motion import course
from blender_ai_tools.motion.course import (APP_MOTIONS_ALL, CHECKS, CLOTH_CHECK, KNEE_CHECK, SINK_CHECK, Event,
                                            app_motion_events, body_parts, check_cloth_penetration, check_floor_sink,
                                            check_knees, cloth_depths, orbit_cameras, pose_events, pose_label,
                                            report_lines, summarize_course)
from blender_ai_tools.motion.poses import AppliedPose, axis_quaternion, pose_by_name
from tests.shapes import box


def rotation_matrix(axis, deg: float) -> np.ndarray:
    n = np.asarray(axis, dtype=np.float64)
    n = n / np.linalg.norm(n)
    t = math.radians(deg)
    k = np.array([[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]])
    return np.eye(3) + math.sin(t) * k + (1.0 - math.cos(t)) * (k @ k)


# MARK: - ひざ


class KneeCheckTest(unittest.TestCase):
    """休みの姿勢：股関節 (±0.1, 0, 1)、ひざ (±0.1, 0, 0.5)。モデルは -Y を向く（すねの先が +Y へ行くのがふつう）"""

    REST_HIPS = np.array([[0.1, 0.0, 1.0], [-0.1, 0.0, 1.0]])
    REST_KNEES = np.array([[0.1, 0.0, 0.5], [-0.1, 0.0, 0.5]])
    IDENTITY = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

    def check(self, ankles, *, knees=None, rotations=None, **extra):
        return check_knees(rest_hips=self.REST_HIPS, rest_knees=self.REST_KNEES, hips=self.REST_HIPS,
                           knees=self.REST_KNEES if knees is None else knees, ankles=ankles,
                           thigh_rotations=self.IDENTITY if rotations is None else rotations,
                           lateral=(1.0, 0.0, 0.0), **extra)

    def shin(self, bend_deg):
        """ひざから、bend_deg 度ふつうに曲げたすねの先（長さ 0.5）"""
        t = math.radians(bend_deg)
        return np.array([0.0, 0.5 * math.sin(t), -0.5 * math.cos(t)])

    def test_normal_bend_is_green(self):
        ankles = self.REST_KNEES + np.array([self.shin(60.0), self.shin(30.0)])
        m = self.check(ankles)
        self.assertEqual(m.signal, Signal.GREEN)
        self.assertEqual(m.name, "reverse_bend")
        self.assertEqual(m.value, 0.0)
        self.assertAlmostEqual(m.details["bend_deg"]["L"], 60.0, places=9)
        self.assertAlmostEqual(m.details["bend_deg"]["R"], 30.0, places=9)

    def test_reverse_bend_is_red(self):
        ankles = self.REST_KNEES + np.array([self.shin(-10.0), self.shin(20.0)])
        m = self.check(ankles)
        self.assertEqual(m.signal, Signal.RED)
        self.assertAlmostEqual(m.value, 10.0, places=9)
        self.assertEqual(m.details["worst_side"], "L")
        self.assertEqual(m.details["reverse_count"], 1)

    def test_within_tolerance_is_green(self):
        slight = thresholds.REVERSE_BEND_TOLERANCE_DEG - 0.5
        m = self.check(self.REST_KNEES + np.array([self.shin(-slight), self.shin(0.0)]))
        self.assertEqual(m.signal, Signal.GREEN)
        self.assertAlmostEqual(m.value, slight, places=9)

    def test_thigh_rotation_carries_the_knee_axis(self):
        """太ももを横に 90 度上げた（+X を向く）脚でも、太ももの回転でひざの軸が決まり、ふつうの曲がりは正になる"""
        lift = axis_quaternion((0.0, -1.0, 0.0), 90.0)  # 左の太ももを横へ上げる
        knees = np.array([[0.6, 0.0, 1.0], [-0.1, 0.0, 0.5]])
        # 軸は +X（休み）→ +Z。すねは太ももの延長（+X）から +Z のまわりに 90 度：+Y（後ろ）
        ankles = np.array([[0.6, 0.5, 1.0], self.REST_KNEES[1] + self.shin(0.0)])
        rotations = np.array([lift, [1.0, 0.0, 0.0, 0.0]])
        m = self.check(ankles, knees=knees, rotations=rotations)
        self.assertEqual(m.signal, Signal.GREEN)
        self.assertAlmostEqual(m.details["bend_deg"]["L"], 90.0, places=9)
        # 太ももの回転を使わない（休みの軸のまま）と、軸が太ももと同じ向きになって測れない（灰）
        wrong = self.check(ankles, knees=knees, rotations=self.IDENTITY)
        self.assertEqual(wrong.signal, Signal.GRAY)

    def test_missing_side_is_gray(self):
        ankles = self.REST_KNEES + np.array([self.shin(40.0), self.shin(40.0)])
        rest_hips = self.REST_HIPS.copy()
        rest_hips[1] = np.nan
        m = check_knees(rest_hips=rest_hips, rest_knees=self.REST_KNEES, hips=self.REST_HIPS, knees=self.REST_KNEES,
                        ankles=ankles, thigh_rotations=self.IDENTITY, lateral=(1, 0, 0), missing_bones=("右足",))
        self.assertEqual(m.signal, Signal.GRAY)
        self.assertTrue(math.isnan(m.details["bend_deg"]["R"]))
        self.assertIn(texts.text("motion.reason_no_leg_bones", bones="右足"), m.reason)

    def test_undetermined_axis_is_gray(self):
        rest_knees = np.array([[0.6, 0.0, 1.0], [-0.1, 0.0, 0.5]])  # 休みの姿勢で、左の太ももが真横を向く
        m = check_knees(rest_hips=self.REST_HIPS, rest_knees=rest_knees, hips=self.REST_HIPS, knees=rest_knees,
                        ankles=rest_knees + [0.0, 0.0, -0.5], thigh_rotations=self.IDENTITY, lateral=(1, 0, 0))
        self.assertEqual(m.signal, Signal.GRAY)
        self.assertEqual(m.details["axis_failed"], ("L",))
        self.assertIn(texts.text("motion.reason_knee_axis", sides="L"), m.reason)

    def test_bad_input(self):
        ankles = self.REST_KNEES.copy()
        with self.assertRaises(ValueError):
            self.check(ankles[:1])
        with self.assertRaises(ValueError):
            self.check(np.full((2, 3), np.inf))
        with self.assertRaises(ValueError):
            check_knees(rest_hips=self.REST_HIPS, rest_knees=self.REST_KNEES, hips=self.REST_HIPS,
                        knees=self.REST_KNEES, ankles=ankles, thigh_rotations=self.IDENTITY, lateral=(np.nan, 0, 0))


# MARK: - 床への沈み


class FloorSinkTest(unittest.TestCase):
    def test_deepest_point(self):
        points = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.001], [0.2, 0.0, -0.003], [0.0, 0.0, 0.5]])
        m = check_floor_sink(points, floor_z=0.0)
        self.assertEqual(m.signal, Signal.RED)
        self.assertEqual(m.name, "floor_sink")
        self.assertAlmostEqual(m.value, 0.003, places=12)
        self.assertEqual(m.details["worst_point"], 2)
        self.assertEqual(m.details["floor_z"], 0.0)

    def test_floor_height_and_tolerance(self):
        tolerance = thresholds.FLOOR_SINK_TOLERANCE
        points = np.array([[0.0, 0.0, -tolerance], [0.0, 0.0, 0.2]])
        m = check_floor_sink(points, floor_z=0.0)
        self.assertEqual(m.signal, Signal.GREEN)  # ちょうど基準の深さは赤にしない
        self.assertEqual(m.value, tolerance)
        # 床の高さを上げると、同じ点がもっと沈む
        raised = check_floor_sink(points, floor_z=0.1)
        self.assertEqual(raised.signal, Signal.RED)
        self.assertAlmostEqual(raised.value, 0.1 + tolerance, places=12)
        self.assertEqual(check_floor_sink(points + [0.0, 0.0, 1.0], floor_z=0.0).value, 0.0)

    def test_empty_and_bad(self):
        self.assertEqual(check_floor_sink(np.zeros((0, 3)), floor_z=0.0).signal, Signal.GRAY)
        for bad in (math.nan, True, "0"):
            with self.assertRaises(ValueError):
                check_floor_sink(np.zeros((1, 3)), floor_z=bad)


# MARK: - 服からの体のはみ出し


def open_box(**options):
    vertices, triangles = box(**options)
    return vertices, triangles[2:]  # 下の面を外す


class BodyPartsTest(unittest.TestCase):
    def test_closed_box(self):
        parts = body_parts(*box(size=(2.0, 2.0, 2.0)))
        self.assertEqual(parts.part_count, 1)
        self.assertEqual(parts.closed.tolist(), [True])
        self.assertEqual(parts.vertex_count, 8)
        self.assertEqual(parts.part_vertices[0].tolist(), list(range(8)))

    def test_open_and_inverted_boxes_are_not_closed(self):
        self.assertEqual(body_parts(*open_box()).closed.tolist(), [False])
        vertices, triangles = box()
        self.assertEqual(body_parts(vertices, triangles[:, ::-1]).closed.tolist(), [False])
        flipped = triangles.copy()
        flipped[0] = flipped[0, ::-1]  # 1つの面だけ向きが違う
        self.assertEqual(body_parts(vertices, flipped).closed.tolist(), [False])

    def test_two_boxes_glued_on_a_face_stay_two_closed_parts(self):
        """同じ位置の頂点の輪（面）でくっついた2つの閉じた形は、別々の閉じた部品"""
        a_vertices, a_triangles = box(center=(0.0, 0.0, 0.0))
        b_vertices, b_triangles = box(center=(1.0, 0.0, 0.0))
        parts = body_parts(np.concatenate([a_vertices, b_vertices]), np.concatenate([a_triangles, b_triangles + 8]))
        self.assertEqual(parts.part_count, 2)
        self.assertEqual(parts.closed.tolist(), [True, True])

    def test_split_vertices_are_joined_by_position(self):
        """UV などで頂点が分かれていても（同じ座標の別の頂点）、1つの閉じた部品になる"""
        vertices, triangles = box()
        extra = vertices[[4, 5, 6, 7]]
        split = triangles.copy()
        split[2:4] = np.array([[8, 9, 10], [8, 10, 11]])  # 上の面だけ、別の頂点を使う
        parts = body_parts(np.concatenate([vertices, extra]), split)
        self.assertEqual(parts.part_count, 1)
        self.assertEqual(parts.closed.tolist(), [True])

    def test_empty(self):
        parts = body_parts(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
        self.assertEqual(parts.part_count, 0)


class ClothPenetrationTest(unittest.TestCase):
    """体：中心 0、一辺 2 の閉じた箱（面は x・y・z = ±1）"""

    def setUp(self):
        self.body, self.triangles = box(size=(2.0, 2.0, 2.0))
        self.parts = body_parts(self.body, self.triangles)
        self.cloth = np.array([[0.0, 0.0, 0.9], [0.0, 0.0, 1.5], [0.5, 0.2, 0.0]])

    def test_depths_and_surface_points(self):
        depths, surface, info = cloth_depths(self.cloth, self.body, self.parts)
        np.testing.assert_allclose(depths, [0.1, 0.0, 0.5], atol=1e-12)
        np.testing.assert_allclose(surface[0], [0.0, 0.0, 1.0], atol=1e-12)
        self.assertTrue(np.isnan(surface[1]).all())
        np.testing.assert_allclose(surface[2], [1.0, 0.2, 0.0], atol=1e-12)
        self.assertEqual(info["penetrating_count"], 2)

    def test_posed_body_vertices_move_the_parts(self):
        moved = self.body + [0.0, 0.0, 10.0]
        depths, _, _ = cloth_depths(self.cloth, moved, self.parts)
        np.testing.assert_allclose(depths, 0.0)
        with self.assertRaises(ValueError):
            cloth_depths(self.cloth, moved[:4], self.parts)

    def test_open_part_makes_nearby_points_undecided(self):
        """閉じた箱（x = 5 のまわり）と、下の面の無い箱（原点のまわり）"""
        open_vertices, open_triangles = open_box(size=(2.0, 2.0, 2.0))
        vertices = np.concatenate([self.body + [5.0, 0.0, 0.0], open_vertices])
        triangles = np.concatenate([self.triangles, open_triangles + 8])
        parts = body_parts(vertices, triangles)
        self.assertEqual(parts.closed.tolist(), [True, False])
        cloth = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 3.0], [5.0, 0.0, 0.8]])
        depths, _, info = cloth_depths(cloth, vertices, parts)
        self.assertTrue(math.isnan(depths[0]))  # 閉じていない箱の中か外か、決められない
        self.assertEqual(depths[1], 0.0)
        self.assertAlmostEqual(depths[2], 0.2, places=12)
        self.assertEqual(info["near_open_part_count"], 1)
        m = check_cloth_penetration(cloth[:2], vertices, parts)
        self.assertEqual(m.signal, Signal.GRAY)
        self.assertIn(texts.text("motion.reason_open_body", count="1"), m.reason)
        # 赤のめり込みがあれば、決められない点があっても赤
        self.assertEqual(check_cloth_penetration(cloth, vertices, parts).signal, Signal.RED)

    def test_signal(self):
        m = check_cloth_penetration(self.cloth, self.body, self.parts)
        self.assertEqual(m.signal, Signal.RED)
        self.assertEqual(m.name, "penetration_depth")
        self.assertAlmostEqual(m.value, 0.5, places=12)
        self.assertEqual(m.details["closed_part_count"], 1)
        self.assertFalse(m.details["visibility_checked"])
        shallow = check_cloth_penetration(np.array([[0.0, 0.0, 1.0 - thresholds.PENETRATION_YELLOW * 1.5]]),
                                          self.body, self.parts)
        self.assertEqual(shallow.signal, Signal.YELLOW)
        outside = check_cloth_penetration(np.array([[0.0, 0.0, 1.2]]), self.body, self.parts)
        self.assertEqual(outside.signal, Signal.GREEN)
        self.assertEqual(outside.value, 0.0)

    def test_no_body_or_no_closed_part_is_gray(self):
        empty = body_parts(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
        m = check_cloth_penetration(self.cloth, np.zeros((0, 3)), empty)
        self.assertEqual(m.signal, Signal.GRAY)
        self.assertEqual(m.reason, texts.text("motion.reason_no_body"))
        vertices, triangles = open_box()
        m = check_cloth_penetration(self.cloth, vertices, body_parts(vertices, triangles))
        self.assertEqual(m.reason, texts.text("motion.reason_no_closed_body"))

    def test_visibility_counts_only_visible_problems(self):
        """はみ出した体の表面（いちばん近い点）が、どれかのカメラから見えるときだけ数える"""
        cloth = np.array([[0.0, 0.0, 0.9]])  # 上の面（z = 1）の近くで、体の中へ 0.1 入った服の点
        options = dict(azimuth_step_deg=90.0, elevations_deg=(0.0,), include_poles=True, height=48)
        visible_cameras = orbit_cameras(self.body, **options)
        renders = render_views(self.body, self.triangles, visible_cameras)
        m = check_cloth_penetration(cloth, self.body, self.parts, renders=renders, scene_mesh=(self.body,
                                                                                                 self.triangles))
        self.assertEqual(m.signal, Signal.RED)
        self.assertAlmostEqual(m.value, 0.1, places=12)
        self.assertTrue(m.details["visibility_checked"])
        self.assertEqual(m.details["visibility"].value, 1.0)
        # 体のまわりを大きな閉じた箱で包むと、はみ出した所はどのカメラからも見えない（記録だけ）
        cover, cover_triangles = box(size=(3.0, 3.0, 3.0))
        scene = (np.concatenate([self.body, cover]), np.concatenate([self.triangles, cover_triangles + 8]))
        renders = render_views(scene[0], scene[1], orbit_cameras(scene[0], **options))
        two = np.array([[0.0, 0.0, 0.9], [0.0, 0.0, 1.2]])  # 2つ目は体の外（めり込んでいない）
        hidden = check_cloth_penetration(two, self.body, self.parts, renders=renders, scene_mesh=scene)
        self.assertEqual(hidden.signal, Signal.GREEN)
        self.assertEqual(hidden.value, 0.0)
        self.assertEqual(hidden.details["penetrating_hidden_count"], 1)
        self.assertEqual(hidden.details["penetrating_visible_count"], 0)
        with self.assertRaises(ValueError):
            check_cloth_penetration(cloth, self.body, self.parts, renders=renders)

    def test_orbit_cameras_frame_the_model(self):
        vertices, _ = box(size=(0.5, 0.4, 1.6), center=(0.0, 0.0, 0.8))
        cameras = orbit_cameras(vertices)
        self.assertEqual(len(cameras), 62)
        radius = np.linalg.norm([0.5, 0.4, 1.6]) / 2.0
        short_half = math.atan(math.tan(math.radians(thresholds.APP_CAMERA_FOV_DEG) / 2.0)
                               * thresholds.APP_SCREEN_ASPECT)
        expected = radius / math.sin(short_half) * thresholds.APP_ORBIT_DISTANCE_MARGIN
        for name, camera in cameras.items():
            with self.subTest(camera=name):
                self.assertAlmostEqual(float(np.linalg.norm(camera.position - [0.0, 0.0, 0.8])), expected, places=9)
                xy, _, in_front = project(vertices, camera)
                self.assertTrue(in_front.all())
                self.assertTrue(((xy[:, 0] >= 0) & (xy[:, 0] < camera.width)
                                 & (xy[:, 1] >= 0) & (xy[:, 1] < camera.height)).all())
        with self.assertRaises(ValueError):
            orbit_cameras(np.zeros((0, 3)))


# MARK: - 出来事・まとめ・報告


def measured(name="reverse_bend", value=0.0, unit="deg", signal=Signal.GREEN, reason=""):
    return Measurement(name, value, unit, signal, reason)


class PoseEventsTest(unittest.TestCase):
    def test_order_and_pass_through(self):
        events = pose_events(pose_by_name("bow"), {
            CLOTH_CHECK: measured("penetration_depth", 0.0, "m"),
            KNEE_CHECK: measured(),
            SINK_CHECK: measured("floor_sink", 0.0, "m"),
        })
        self.assertEqual([e.check for e in events], list(CHECKS))
        self.assertEqual({e.pose for e in events}, {"bow"})
        self.assertTrue(all(e.signal == Signal.GREEN for e in events))

    def test_partial_pose_turns_gray_but_keeps_red(self):
        applied = AppliedPose(pose="clap", missing=("左ひじ",))
        events = pose_events("clap", {KNEE_CHECK: measured(),
                                      CLOTH_CHECK: measured("penetration_depth", 0.01, "m", Signal.RED, "深い")},
                             applied)
        knee, cloth = events
        self.assertEqual(knee.signal, Signal.GRAY)
        self.assertEqual(knee.measurement.reason, texts.text("motion.reason_missing_bones", bones="左ひじ"))
        self.assertEqual(knee.measurement.value, 0.0)
        self.assertTrue(knee.measurement.details["partial"])
        self.assertEqual(knee.measurement.details["missing_bones"], ("左ひじ",))
        self.assertEqual(cloth.signal, Signal.RED)
        self.assertTrue(cloth.measurement.reason.endswith("深い"))

    def test_overridden_pose_is_gray(self):
        applied = AppliedPose(pose="squat", overridden=("左ひざ", "右ひざ"))
        (event,) = pose_events("squat", {KNEE_CHECK: measured()}, applied)
        self.assertEqual(event.signal, Signal.GRAY)
        self.assertIn(texts.text("motion.reason_overridden", bones=texts.text("motion.list_separator").join(
            ["左ひざ", "右ひざ"])), event.measurement.reason)

    def test_complete_pose_keeps_signal(self):
        (event,) = pose_events("squat", {KNEE_CHECK: measured()}, AppliedPose(pose="squat"))
        self.assertEqual(event.signal, Signal.GREEN)
        self.assertFalse(event.measurement.details["partial"])

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            pose_events("bow", {"other": measured()})
        with self.assertRaises(ValueError):
            pose_events("bow", {KNEE_CHECK: measured()}, AppliedPose(pose="squat"))
        with self.assertRaises(ValueError):
            pose_events("bow", {KNEE_CHECK: 1.0})
        with self.assertRaises(ValueError):
            Event("", KNEE_CHECK, measured())
        with self.assertRaises(ValueError):
            Event("bow", "other", measured())


class AppMotionEventsTest(unittest.TestCase):
    def test_all_gray_with_reason(self):
        (event,) = app_motion_events()
        self.assertEqual(event.pose, APP_MOTIONS_ALL)
        self.assertEqual(event.signal, Signal.GRAY)
        self.assertEqual(event.measurement.reason, texts.text("motion.reason_app_motions"))
        self.assertIsNone(event.measurement.value)

    def test_named_motions(self):
        events = app_motion_events(["idle", "wave"])
        self.assertEqual([e.pose for e in events], ["app:idle", "app:wave"])
        self.assertTrue(all(e.signal == Signal.GRAY for e in events))
        self.assertEqual(events[1].measurement.details["motion"], "wave")
        self.assertEqual(app_motion_events([]), [])
        for bad in (["a", "a"], [""], "idle", [1]):
            with self.assertRaises(ValueError):
                app_motion_events(bad)


class SummaryAndReportTest(unittest.TestCase):
    def tearDown(self):
        texts.set_language("ja")

    def events(self):
        return [
            Event("bow", KNEE_CHECK, measured()),
            Event("bow", SINK_CHECK, measured("floor_sink", 0.0, "m")),
            Event("squat", KNEE_CHECK, measured(value=12.5, signal=Signal.RED, reason="逆")),
            Event("squat", CLOTH_CHECK, measured("penetration_depth", 0.00123, "m", Signal.YELLOW)),
            *app_motion_events(),
        ]

    def test_summary(self):
        m = summarize_course(self.events())
        self.assertEqual(m.name, "pose_course")
        self.assertEqual(m.signal, Signal.RED)
        self.assertEqual(m.value, 1.0)
        self.assertEqual(m.details["pose_count"], 2)  # アプリの動きは数えない
        self.assertEqual(m.details["event_count"], 5)
        self.assertEqual(m.details["red_events"], [("squat", KNEE_CHECK)])
        self.assertEqual(m.details["gray_events"], [(APP_MOTIONS_ALL, "app_motion")])
        self.assertEqual(m.details["per_pose"]["bow"]["signal"], Signal.GREEN)
        self.assertEqual(m.details["per_pose"]["squat"]["checks"], {KNEE_CHECK: Signal.RED, CLOTH_CHECK: Signal.YELLOW})
        self.assertEqual(m.reason, texts.text("motion.summary_reason", poses="2", red="1", gray="1", yellow="1"))

    def test_all_green_and_empty(self):
        green = summarize_course([Event("bow", KNEE_CHECK, measured())])
        self.assertEqual(green.signal, Signal.GREEN)
        self.assertEqual(green.value, 0.0)
        self.assertEqual(green.reason, texts.text("motion.summary_all_green", poses="1"))
        empty = summarize_course([])
        self.assertEqual(empty.signal, Signal.GRAY)
        self.assertIsNone(empty.value)
        with self.assertRaises(ValueError):
            summarize_course([measured()])

    def test_report_lines_japanese(self):
        lines = report_lines(self.events(), language="ja")
        self.assertEqual(lines, [
            "確かめの動き：ポーズ 2 個、出来事 5 件（赤 1・灰 1・黄 1・緑 2）",
            "赤　しゃがむ：ひざの逆曲がり　12.5 度（逆）",
            "灰　アプリの動き（全部）：アプリの動き　値なし（" + texts.text("motion.reason_app_motions") + "）",
            "黄　しゃがむ：服からの体のはみ出し　1.2 mm",
        ])

    def test_report_lines_english(self):
        lines = report_lines(self.events()[:4], language="en")
        self.assertEqual(lines, [
            "Pose course: 2 pose(s), 4 event(s) (red 1, gray 0, yellow 1, green 2)",
            "Red  Squatting: Reverse knee bend  12.5° (逆)",
            "Yellow  Squatting: Body poking through the clothes  1.2 mm",
        ])

    def test_report_all_green(self):
        lines = report_lines([Event("bow", KNEE_CHECK, measured())], language="ja")
        self.assertEqual(lines[-1], texts.text("motion.report_all_green", language="ja"))
        self.assertEqual(len(lines), 2)

    def test_labels(self):
        self.assertEqual(pose_label("seiza", language="ja"), "正座")
        self.assertEqual(pose_label("seiza", language="en"), "Seiza (kneeling)")
        self.assertEqual(pose_label("app:wave", language="ja"), "アプリの動き「wave」")
        self.assertEqual(pose_label("custom_pose"), "custom_pose")
        self.assertEqual(course.check_label(SINK_CHECK, language="en"), "Sinking into the floor")
        with self.assertRaises(ValueError):
            course.check_label("other")

    def test_language_follows_current_setting(self):
        texts.set_language("en")
        self.assertEqual(pose_label("bow"), "Bowing")


if __name__ == "__main__":
    unittest.main()

"""A5 確かめの動き：いまのシーンのモデルでポーズを1つずつ確かめる所（motion/course_bpy.py）を確かめる（Blender の中）"""

import dataclasses
import math
import unittest
from unittest import mock

import bpy
import numpy as np

from blender_ai_tools import texts, thresholds
from blender_ai_tools.measure.results import Signal
from blender_ai_tools.model import card_bpy, extract_bpy
from blender_ai_tools.model.card import set_part_kind
from blender_ai_tools.motion import course, course_bpy, poses
from blender_ai_tools.motion.course import APP_MOTIONS_ALL, CLOTH_CHECK, KNEE_CHECK, SINK_CHECK
from blender_ai_tools.motion.poses import POSES, pose_by_name
from tests.fixtures import mmd_model

REST_KNEE_BEND = math.degrees(math.atan2(0.012, 0.34) + math.atan2(0.012, 0.325))
# 軽い見え方の設定（テストの時間を短くする）：水平に 90 度ごと 4 台と、真上・真下
LIGHT_ORBIT = {"azimuth_step_deg": 90.0, "elevations_deg": (0.0,), "include_poles": True, "height": 96}


def by_key(events):
    return {(event.pose, event.check): event for event in events}


def uses_bone(pose, bone):
    return any(name == bone for name, _, _ in pose.rotations)


class CourseTestCase(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_homefile(use_empty=True)
        self.info = mmd_model.build_model()
        self.armature = bpy.data.objects[self.info["armature"]]
        self.body = bpy.data.objects[mmd_model.BODY_NAME]
        self.skirt = bpy.data.objects[mmd_model.SKIRT_NAME]


class RunCourseTest(CourseTestCase):
    def test_events_for_every_pose_and_app_motions(self):
        events = course_bpy.run_course()
        lying = [pose for pose in POSES if pose.lying]
        self.assertEqual(len(events), 3 * len(POSES) - len(lying) + 1)
        expected = []
        for pose in POSES:
            expected += [(pose.name, KNEE_CHECK)] + ([] if pose.lying else [(pose.name, SINK_CHECK)])
            expected += [(pose.name, CLOTH_CHECK)]
        expected.append((APP_MOTIONS_ALL, "app_motion"))
        self.assertEqual([(e.pose, e.check) for e in events], expected)
        app = events[-1]
        self.assertEqual(app.signal, Signal.GRAY)
        self.assertEqual(app.measurement.reason, texts.text("motion.reason_app_motions"))

    def test_knees_bend_normally_and_nothing_sinks(self):
        events = course_bpy.run_course()
        for event in events:
            pose = poses.pose_by_name(event.pose) if event.check != "app_motion" else None
            if event.check == KNEE_CHECK:
                with self.subTest(pose=event.pose, check="knee"):
                    self.assertEqual(event.signal, Signal.GREEN)
                    self.assertEqual(event.measurement.value, 0.0)
                    for side, prefix in (("L", "左"), ("R", "右")):
                        applied = sum(a for b, _, a in pose.rotations if b == f"{prefix}ひざ")
                        self.assertAlmostEqual(event.measurement.details["bend_deg"][side], REST_KNEE_BEND + applied,
                                               delta=0.01)
            if event.check == SINK_CHECK:
                with self.subTest(pose=event.pose, check="sink"):
                    self.assertEqual(event.signal, Signal.GREEN)
                    self.assertLessEqual(event.measurement.value, thresholds.REPRO_TOLERANCE_LENGTH)
                    self.assertAlmostEqual(event.measurement.details["floor_z"], 0.0, places=6)

    def test_cloth_penetration_found_where_the_legs_go_through_the_skirt(self):
        events = by_key(course_bpy.run_course(checks=[CLOTH_CHECK]))
        for name in ("head_down", "arms_up", "wave", "lying_back"):
            with self.subTest(pose=name):
                event = events[(name, CLOTH_CHECK)]
                self.assertEqual(event.signal, Signal.GREEN)
                self.assertEqual(event.measurement.value, 0.0)
                self.assertEqual(event.measurement.details["closed_part_count"], len(mmd_model.BODY_PARTS))
                self.assertEqual(event.measurement.details["open_part_count"], 0)
                self.assertFalse(event.measurement.details["visibility_checked"])
        for name in ("leg_front_left", "leg_front_right", "chair_sit", "squat"):
            with self.subTest(pose=name):
                event = events[(name, CLOTH_CHECK)]
                self.assertEqual(event.signal, Signal.RED)
                self.assertGreater(event.measurement.value, 0.01)
        # 左右の鏡のポーズは、同じ深さ
        self.assertAlmostEqual(events[("leg_front_left", CLOTH_CHECK)].measurement.value,
                               events[("leg_front_right", CLOTH_CHECK)].measurement.value, places=6)

    def test_summary_and_report(self):
        events = course_bpy.run_course()
        summary = course.summarize_course(events)
        red = [e for e in events if e.signal == Signal.RED]
        self.assertEqual(summary.signal, Signal.RED)
        self.assertEqual(summary.value, float(len(red)))
        self.assertEqual(summary.details["pose_count"], len(POSES))
        lines = course.report_lines(events, language="ja")
        self.assertEqual(lines[0], texts.text(
            "motion.report_header", language="ja", poses=str(len(POSES)), events=str(len(events)), red=str(len(red)),
            gray=str(sum(e.signal == Signal.GRAY for e in events)), yellow=str(sum(e.signal == Signal.YELLOW
                                                                                    for e in events)),
            green=str(sum(e.signal == Signal.GREEN for e in events))))
        self.assertEqual(len(lines), 1 + sum(e.signal != Signal.GREEN for e in events))

    def test_app_motion_names(self):
        events = course_bpy.run_course(poses=[pose_by_name("bow")], app_motions=["idle", "wave"])
        self.assertEqual([e.pose for e in events[-2:]], ["app:idle", "app:wave"])
        self.assertTrue(all(e.signal == Signal.GRAY for e in events[-2:]))

    def test_selected_checks_only(self):
        events = course_bpy.run_course(poses=[pose_by_name("bow"), pose_by_name("lying_back")], checks=[SINK_CHECK])
        self.assertEqual([(e.pose, e.check) for e in events], [("bow", SINK_CHECK), (APP_MOTIONS_ALL, "app_motion")])

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            course_bpy.run_course(checks=["other"])
        with self.assertRaises(ValueError):
            course_bpy.run_course(checks=SINK_CHECK)
        with self.assertRaises(ValueError):
            course_bpy.run_course(poses=[pose_by_name("bow"), pose_by_name("bow")])
        with self.assertRaises(ValueError):
            course_bpy.run_course(poses=["bow"])
        with self.assertRaises(ValueError):
            course_bpy.run_course(visibility="yes")
        with self.assertRaises(ValueError):
            course_bpy.run_course(card={"parts": []})
        other = bpy.data.scenes.new("Other")
        with self.assertRaises(ValueError):
            course_bpy.run_course(other)


class VisibilityTest(CourseTestCase):
    def test_visible_problems_are_counted(self):
        events = by_key(course_bpy.run_course(poses=[pose_by_name("leg_front_left"), pose_by_name("arms_up")],
                                              checks=[CLOTH_CHECK], visibility=True, orbit=LIGHT_ORBIT))
        leg = events[("leg_front_left", CLOTH_CHECK)].measurement
        self.assertTrue(leg.details["visibility_checked"])
        self.assertEqual(leg.signal, Signal.RED)
        self.assertGreaterEqual(leg.details["visibility"].value, 1.0)
        self.assertEqual(leg.details["penetrating_visible_count"], int(leg.details["visibility"].value))
        arms = events[("arms_up", CLOTH_CHECK)].measurement
        self.assertEqual(arms.signal, Signal.GREEN)
        self.assertIsNone(arms.details["visibility"])  # めり込みが無ければ、見え方は調べない


class StateTest(CourseTestCase):
    def snapshot(self):
        return {pb.name: (tuple(pb.location), tuple(pb.rotation_quaternion), tuple(pb.scale))
                for pb in self.armature.pose.bones}

    def prepare(self):
        """ユーザーの作業中の状態：フレーム・ポーズ・IK・選択・アクティブなオブジェクト・ポーズモード"""
        bpy.context.scene.frame_set(3)
        self.armature.pose.bones["頭"].rotation_quaternion = (0.9, 0.1, 0.2, 0.3)
        self.armature.pose.bones["センター"].location = (0.0, 0.01, 0.02)
        ik = self.armature.pose.bones["左ひざ"].constraints.new("IK")
        ik.target = self.armature
        ik.subtarget = "左足ＩＫ"
        ik.chain_count = 2
        for obj in bpy.context.view_layer.objects:
            obj.select_set(obj == self.skirt)
        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.mode_set(mode="POSE")
        bpy.context.view_layer.update()
        self.selected = [obj.name for obj in bpy.context.view_layer.objects if obj.select_get()]
        self.assertIn(self.skirt.name, self.selected)
        return ik

    def assert_restored(self, ik, before, wrists):
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(ik.mute)
        self.assertEqual(bpy.context.scene.frame_current, 3)
        self.assertEqual(self.armature.mode, "POSE")
        self.assertIs(bpy.context.view_layer.objects.active, self.armature)
        self.assertEqual([obj.name for obj in bpy.context.view_layer.objects if obj.select_get()], self.selected)
        np.testing.assert_allclose(extract_bpy.joint_positions(self.armature, ["左手首", "右手首"]), wrists, atol=1e-9)

    def test_state_is_restored(self):
        ik = self.prepare()
        before = self.snapshot()
        wrists = extract_bpy.joint_positions(self.armature, ["左手首", "右手首"])
        events = course_bpy.run_course(poses=[pose_by_name("knee_bend_left"), pose_by_name("arms_up")])
        self.assertEqual(by_key(events)[("knee_bend_left", KNEE_CHECK)].signal, Signal.GREEN)
        self.assert_restored(ik, before, wrists)
        bpy.ops.object.mode_set(mode="OBJECT")

    def test_state_is_restored_after_an_error(self):
        ik = self.prepare()
        before = self.snapshot()
        wrists = extract_bpy.joint_positions(self.armature, ["左手首", "右手首"])
        with mock.patch.object(course, "check_floor_sink", side_effect=RuntimeError("途中で止まる")):
            with self.assertRaises(RuntimeError):
                course_bpy.run_course(poses=[pose_by_name("squat")])
        self.assert_restored(ik, before, wrists)
        bpy.ops.object.mode_set(mode="OBJECT")

    def test_rest_position_is_restored(self):
        self.armature.data.pose_position = "REST"
        events = course_bpy.run_course(poses=[pose_by_name("knee_bend_left")], checks=[KNEE_CHECK])
        self.assertEqual(events[0].signal, Signal.GREEN)
        self.assertAlmostEqual(events[0].measurement.details["bend_deg"]["L"], REST_KNEE_BEND + 130.0, delta=0.01)
        self.assertEqual(self.armature.data.pose_position, "REST")

    def test_failed_pose_is_gray_and_the_course_goes_on(self):
        with mock.patch.object(course, "check_floor_sink", side_effect=ValueError("読めない")):
            events = course_bpy.run_course(poses=[pose_by_name("bow"), pose_by_name("arms_up")])
        pose_events = [e for e in events if e.check != "app_motion"]
        self.assertEqual(len(pose_events), 6)
        reason = texts.text("motion.reason_pose_failed", error="読めない")
        for event in pose_events:
            self.assertEqual(event.signal, Signal.GRAY)
            self.assertEqual(event.measurement.reason, reason)


class ModelVariantsTest(CourseTestCase):
    def test_missing_bone_makes_poses_gray(self):
        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.mode_set(mode="EDIT")
        self.armature.data.edit_bones.remove(self.armature.data.edit_bones["左ひじ"])
        bpy.ops.object.mode_set(mode="OBJECT")
        events = course_bpy.run_course()
        needing = {pose.name for pose in POSES if uses_bone(pose, "左ひじ")}
        self.assertIn("elbows_bend", needing)
        reason = texts.text("motion.reason_missing_bones", bones="左ひじ")
        for event in events:
            if event.check == "app_motion":
                continue
            with self.subTest(pose=event.pose, check=event.check):
                if event.pose in needing:
                    self.assertIn(event.signal, (Signal.GRAY, Signal.RED))
                    self.assertTrue(event.measurement.reason.startswith(reason))
                    self.assertEqual(event.measurement.details["missing_bones"], ("左ひじ",))
                elif event.check == KNEE_CHECK:
                    self.assertEqual(event.signal, Signal.GREEN)
        elbows = by_key(events)
        self.assertEqual({elbows[("elbows_bend", check)].signal for check in course.CHECKS}, {Signal.GRAY})

    def test_card_kinds_are_used(self):
        card = card_bpy.build_card()
        not_cloth = set_part_kind(card, mmd_model.SKIRT_NAME, "accessory")
        events = course_bpy.run_course(poses=[pose_by_name("leg_front_left")], checks=[CLOTH_CHECK], card=not_cloth)
        self.assertEqual(events[0].signal, Signal.GRAY)
        self.assertEqual(events[0].measurement.reason, texts.text("motion.reason_no_cloth"))
        # シーンに置いたカードも使う
        card_bpy.save_card(not_cloth)
        events = course_bpy.run_course(poses=[pose_by_name("leg_front_left")], checks=[CLOTH_CHECK])
        self.assertEqual(events[0].measurement.reason, texts.text("motion.reason_no_cloth"))
        # ほかのアーマチュアのカードは使わない（名前から推測する：Skirt は服）
        events = course_bpy.run_course(poses=[pose_by_name("leg_front_left")], checks=[CLOTH_CHECK],
                                       card=dataclasses.replace(not_cloth, armature_name="Other"))
        self.assertEqual(events[0].signal, Signal.RED)

    def test_materials_split_one_mesh_into_body_and_cloth(self):
        """1つのメッシュに肌とスカートの材質がある（MMD のモデルに多い）：材質ごとに体と服に分ける"""
        separate = by_key(course_bpy.run_course(poses=[pose_by_name("leg_front_left")]))
        skin = bpy.data.materials.new("肌")
        cloth = bpy.data.materials.new("スカート")
        self.body.data.materials.append(skin)
        self.skirt.data.materials.append(cloth)
        for obj in bpy.context.view_layer.objects:
            obj.select_set(obj in (self.body, self.skirt))
        bpy.context.view_layer.objects.active = self.body
        bpy.ops.object.join()
        joined = bpy.data.objects[mmd_model.BODY_NAME]
        self.assertEqual([slot.material.name for slot in joined.material_slots], ["肌", "スカート"])
        self.assertIsNone(bpy.data.objects.get(mmd_model.SKIRT_NAME))
        together = by_key(course_bpy.run_course(poses=[pose_by_name("leg_front_left")]))
        for check in course.CHECKS:
            with self.subTest(check=check):
                self.assertEqual(together[("leg_front_left", check)].signal,
                                 separate[("leg_front_left", check)].signal)
                self.assertAlmostEqual(together[("leg_front_left", check)].measurement.value,
                                       separate[("leg_front_left", check)].measurement.value, places=9)
        self.assertEqual(together[("leg_front_left", CLOTH_CHECK)].signal, Signal.RED)

    def test_unreadable_meshes_make_checks_gray(self):
        self.skirt.hide_viewport = True
        events = by_key(course_bpy.run_course(poses=[pose_by_name("bow")]))
        self.assertEqual(events[("bow", CLOTH_CHECK)].signal, Signal.GRAY)
        self.assertEqual(events[("bow", CLOTH_CHECK)].measurement.reason,
                         texts.text("motion.reason_unreadable_meshes", names=mmd_model.SKIRT_NAME))
        self.assertEqual(events[("bow", SINK_CHECK)].signal, Signal.GREEN)
        self.body.hide_viewport = True
        events = by_key(course_bpy.run_course(poses=[pose_by_name("bow")]))
        reason = texts.text("motion.reason_unreadable_meshes", names=mmd_model.BODY_NAME)
        self.assertEqual(events[("bow", SINK_CHECK)].measurement.reason, reason)
        self.assertEqual(events[("bow", CLOTH_CHECK)].measurement.reason, reason)
        self.assertEqual(events[("bow", KNEE_CHECK)].signal, Signal.GREEN)  # 骨だけで測れる
        self.assertTrue(self.body.hide_viewport)

    def test_no_body_part_is_gray(self):
        self.body.name = "Thing"  # 名前から種類が分からない
        events = by_key(course_bpy.run_course(poses=[pose_by_name("squat")]))
        reason = texts.text("motion.reason_no_body")
        self.assertEqual(events[("squat", SINK_CHECK)].measurement.reason, reason)
        self.assertEqual(events[("squat", CLOTH_CHECK)].measurement.reason, reason)
        self.assertEqual(events[("squat", KNEE_CHECK)].signal, Signal.GREEN)

    def test_leg_length_from_card_is_converted_to_armature_space(self):
        self.armature.scale = (2.0, 2.0, 2.0)
        bpy.context.view_layer.update()
        card = card_bpy.build_card()
        rest = extract_bpy.bone_rest(self.armature, space="armature")
        from_bones = poses.leg_length_from_bones(rest.names, rest.heads)
        self.assertAlmostEqual(course_bpy._leg_length(card, self.armature), from_bones, places=6)
        self.assertIsNone(course_bpy._leg_length(None, self.armature))
        events = course_bpy.run_course(poses=[pose_by_name("squat")], card=card)
        self.assertTrue(all(e.signal != Signal.GRAY for e in events if e.check == KNEE_CHECK))


if __name__ == "__main__":
    unittest.main()

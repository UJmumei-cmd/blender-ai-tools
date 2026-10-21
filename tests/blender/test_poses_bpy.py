"""A5 確かめの動き：ポーズをモデルに付ける所（motion/poses_bpy.py）を、テスト用のモデルで確かめる（Blender の中）"""

import math
import unittest

import bpy
import numpy as np

from blender_ai_tools import thresholds
from blender_ai_tools.measure import kinematics
from blender_ai_tools.model import extract_bpy
from blender_ai_tools.motion import poses, poses_bpy
from blender_ai_tools.motion.poses import POSES, pose_by_name
from tests.fixtures import mmd_model

# 足の裏で立つポーズ（いちばん低い所が足になる）
FOOT_POSES = ("tiptoe", "walk", "run", "squat", "chair_sit", "hands_on_knees", "sword_stance")
# 休みの姿勢のひざの曲がり（テスト用のモデルは、ひざが少し前にある）：股関節 (0.08, 0, 0.74)、ひざ (0.08, -0.012, 0.40)、
# 足首 (0.08, 0, 0.075)
REST_KNEE_BEND = math.degrees(math.atan2(0.012, 0.34) + math.atan2(0.012, 0.325))


def knee_rotation_total(pose, side: str) -> float:
    """ポーズがひざに付ける角度の合計（ひざは +X のまわりにだけ回す）"""
    return sum(angle for bone, _, angle in pose.rotations if bone == f"{side}ひざ")


class PoseBpyTestCase(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_homefile(use_empty=True)
        self.info = mmd_model.build_model()
        self.armature = bpy.data.objects[self.info["armature"]]
        self.body = bpy.data.objects[mmd_model.BODY_NAME]
        self.parts = self.info["parts"][mmd_model.BODY_NAME]

    def part_points(self, *names):
        points = extract_bpy.mesh_evaluated(self.body)
        return np.concatenate([points[slice(*self.parts[name])] for name in names])

    def knee_bends(self):
        """いまのポーズの左右のひざの曲がり（位置だけで測る。D の骨。ふつうが正）"""
        rest = extract_bpy.bone_rest(self.armature)
        lateral = rest.heads[rest.index("左足D")] - rest.heads[rest.index("右足D")]
        result = {}
        for side in ("左", "右"):
            hip, knee, ankle = extract_bpy.joint_positions(self.armature, [f"{side}足D", f"{side}ひざD",
                                                                          f"{side}足首D"])[0]
            result[side] = float(kinematics.knee_bend(hip, knee, ankle, lateral))
        return result


class ApplyPoseTest(PoseBpyTestCase):
    def test_every_pose_applies_completely(self):
        with poses_bpy.posing(self.armature):
            for pose in POSES:
                with self.subTest(pose=pose.name):
                    applied = poses_bpy.apply_pose(self.armature, pose, ground_meshes=[self.body])
                    self.assertEqual(applied.missing, ())
                    self.assertEqual(applied.overridden, ())
                    self.assertFalse(applied.partial)
                    self.assertEqual(set(applied.posed), {bone for bone, _, _ in pose.rotations})
                    self.assertEqual(applied.grounded_by, "mesh" if pose.ground else None)

    def test_arms_up_puts_both_hands_above_the_head(self):
        with poses_bpy.posing(self.armature):
            poses_bpy.apply_pose(self.armature, pose_by_name("arms_up"))
            wrists = extract_bpy.joint_positions(self.armature, ["左手首", "右手首"])[0]
            hands = self.part_points("hand_L", "hand_R")
        top_of_head = self.info["height"]
        self.assertGreater(wrists[:, 2].min(), top_of_head)
        self.assertGreater(hands[:, 2].min(), top_of_head)
        # 左右そろって上がる（左右の鏡）
        np.testing.assert_allclose(wrists[0], wrists[1] * [-1.0, 1.0, 1.0], atol=1e-6)

    def test_knee_poses_bend_knees_the_normal_way(self):
        knee_poses = [pose for pose in POSES if knee_rotation_total(pose, "左") or knee_rotation_total(pose, "右")]
        self.assertGreaterEqual(len(knee_poses), 10)
        with poses_bpy.posing(self.armature):
            for pose in knee_poses:
                poses_bpy.apply_pose(self.armature, pose, ground_meshes=[self.body])
                bends = self.knee_bends() if not pose.lying else None
                rotations = extract_bpy.pose_rotations(self.armature)[0]
                rest = extract_bpy.bone_rest(self.armature)
                for side in ("左", "右"):
                    with self.subTest(pose=pose.name, side=side):
                        # 太ももの回転を使う測り方は、どのポーズでも、休みの曲がり + 付けた角度
                        hip, knee, ankle = extract_bpy.joint_positions(
                            self.armature, [f"{side}足D", f"{side}ひざD", f"{side}足首D"])[0]
                        axis = kinematics.knee_rest_axis(rest.heads[rest.index(f"{side}足D")],
                                                         rest.heads[rest.index(f"{side}ひざD")], (1.0, 0.0, 0.0))
                        bend = float(kinematics.knee_bend_from_thigh(
                            hip, knee, ankle, rest_axis=axis, thigh_rotation=rotations[rest.index(f"{side}足D")]))
                        expected = REST_KNEE_BEND + knee_rotation_total(pose, side)
                        self.assertAlmostEqual(bend, expected, delta=0.01)
                        self.assertGreater(bend, 0.0)
                        if bends is not None and pose.name not in ("squat",):
                            # 脚を横に開かないポーズでは、位置だけで測っても同じ
                            self.assertAlmostEqual(bends[side], expected, delta=0.01)

    def test_ground_poses_touch_the_floor(self):
        floor = float(extract_bpy.mesh_evaluated(self.body)[:, 2].min())
        self.assertAlmostEqual(floor, 0.0, places=6)
        with poses_bpy.posing(self.armature):
            for pose in POSES:
                if not pose.ground:
                    continue
                with self.subTest(pose=pose.name):
                    applied = poses_bpy.apply_pose(self.armature, pose, ground_meshes=[self.body], floor_z=floor)
                    lowest = float(extract_bpy.mesh_evaluated(self.body)[:, 2].min())
                    self.assertAlmostEqual(lowest, floor, delta=thresholds.REPRO_TOLERANCE_LENGTH)
                    self.assertIsNotNone(applied.ground_shift)
                    if pose.name in FOOT_POSES:
                        feet = self.part_points("foot_L", "foot_R")
                        self.assertLessEqual(abs(float(feet[:, 2].min()) - floor), thresholds.FLOOR_SINK_TOLERANCE)

    def test_grounding_with_vertex_mask_and_default_floor(self):
        """floor_z を渡さなければ、休みの姿勢の同じ頂点のいちばん低い高さ。印を付けた頂点だけで合わせる"""
        head = np.zeros(len(self.body.data.vertices), dtype=bool)
        head[slice(*self.parts["head"])] = True
        rest_head_low = float(extract_bpy.mesh_evaluated(self.body)[head][:, 2].min())
        with poses_bpy.posing(self.armature):
            poses_bpy.apply_pose(self.armature, pose_by_name("seiza"), ground_meshes=[self.body],
                                 ground_vertex_masks=[head])
            posed = extract_bpy.mesh_evaluated(self.body)
        self.assertAlmostEqual(float(posed[head][:, 2].min()), rest_head_low, delta=thresholds.REPRO_TOLERANCE_LENGTH)
        with poses_bpy.posing(self.armature):
            with self.assertRaises(ValueError):
                poses_bpy.apply_pose(self.armature, pose_by_name("seiza"), ground_meshes=[self.body],
                                     ground_vertex_masks=[head[:-1]])

    def test_grounding_by_bones(self):
        rest = extract_bpy.bone_rest(self.armature)
        deform = np.flatnonzero(rest.deform)
        rest_low = min(float(rest.heads[deform][:, 2].min()), float(rest.tails[deform][:, 2].min()))
        with poses_bpy.posing(self.armature):
            applied = poses_bpy.apply_pose(self.armature, pose_by_name("squat"))
            self.assertEqual(applied.grounded_by, "bones")
            posed = extract_bpy.pose_matrices(self.armature)[0][deform]
            lengths = np.linalg.norm(rest.tails[deform] - rest.heads[deform], axis=1)
            tails = posed[:, :3, 3] + posed[:, :3, 1] * lengths[:, None]
            low = min(float(posed[:, 2, 3].min()), float(tails[:, 2].min()))
        self.assertAlmostEqual(low, rest_low, delta=1e-6)

    def test_grounding_by_bones_on_a_moved_and_scaled_armature(self):
        """メッシュ無しの床合わせ（形を動かす骨の根元と先）は、アーマチュアの大きさが 1 でなくても、休みの高さに合わせる。
        骨の先は、ポーズの行列（世界）に、アーマチュアの中の骨の長さを当てた物（大きさを二重に掛けない）"""
        self.armature.location = (0.0, 0.0, 0.3)
        self.armature.scale = (2.0, 2.0, 2.0)
        bpy.context.view_layer.update()
        rest_world = extract_bpy.bone_rest(self.armature)
        rest_arm = extract_bpy.bone_rest(self.armature, space="armature")
        deform = np.flatnonzero(rest_world.deform)
        rest_low = min(float(rest_world.heads[deform][:, 2].min()), float(rest_world.tails[deform][:, 2].min()))
        self.assertAlmostEqual(rest_low, 0.3 + 2.0 * 0.01, places=6)  # 左足首D の先（つま先、z = 0.01）が、いちばん低い
        lengths = np.linalg.norm(rest_arm.tails[deform] - rest_arm.heads[deform], axis=1)
        with poses_bpy.posing(self.armature):
            applied = poses_bpy.apply_pose(self.armature, pose_by_name("squat"))
            self.assertEqual(applied.grounded_by, "bones")
            posed = extract_bpy.pose_matrices(self.armature)[0][deform]
            tails = posed[:, :3, 3] + posed[:, :3, 1] * lengths[:, None]
            low = min(float(posed[:, 2, 3].min()), float(tails[:, 2].min()))
        self.assertAlmostEqual(low, rest_low, delta=1e-6)
        self.assertLess(applied.ground_shift, 0.0)

    def test_center_without_local_location_is_grounded_and_offset(self):
        """センター の「ローカルの位置」（use_local_location）が切れていても、床に合わせる量と座る量は同じになる。
        Blender は、切れている骨の位置を親のポーズの向きで当てる（根の骨ならアーマチュアの向き）"""
        self.armature.data.bones["センター"].use_local_location = False
        floor = float(extract_bpy.mesh_evaluated(self.body)[:, 2].min())
        rest = extract_bpy.bone_rest(self.armature, space="armature")
        rest_center = rest.heads[rest.index("センター")]
        with poses_bpy.posing(self.armature):
            applied = poses_bpy.apply_pose(self.armature, pose_by_name("squat"), ground_meshes=[self.body],
                                           floor_z=floor)
            lowest = float(extract_bpy.mesh_evaluated(self.body)[:, 2].min())
            self.assertAlmostEqual(lowest, floor, delta=thresholds.REPRO_TOLERANCE_LENGTH)
            self.assertLess(applied.ground_shift, 0.0)
            applied = poses_bpy.apply_pose(self.armature, pose_by_name("chair_sit"))
            moved = extract_bpy.joint_positions(self.armature, ["センター"])[0][0] - rest_center
            np.testing.assert_allclose(moved, [0.0, 0.0, -0.45 * applied.leg_length + applied.ground_shift], atol=1e-6)
        # 根の骨（全ての親）の位置は、アーマチュアの向きのまま当たる：横になる姿勢の床合わせも同じ
        self.armature.data.bones["全ての親"].use_local_location = False
        with poses_bpy.posing(self.armature):
            poses_bpy.apply_pose(self.armature, pose_by_name("lying_back"), ground_meshes=[self.body], floor_z=floor)
            lowest = float(extract_bpy.mesh_evaluated(self.body)[:, 2].min())
        self.assertAlmostEqual(lowest, floor, delta=thresholds.REPRO_TOLERANCE_LENGTH)

    def test_lying_poses_lay_the_body_down(self):
        """横になる姿勢：頭と足の高さがほぼ同じになり、あおむけでは顔が上を向き、横向きではキャラの左が下になる"""
        def centers(*names):
            return [self.part_points(*group).mean(axis=0) for group in names]

        with poses_bpy.posing(self.armature):
            poses_bpy.apply_pose(self.armature, pose_by_name("lying_back"), ground_meshes=[self.body])
            head, nose, feet = centers(("head",), ("nose",), ("foot_L", "foot_R"))
            self.assertLess(abs(float(head[2] - feet[2])), 0.1)  # 体が横になっている
            self.assertGreater(float(nose[2]), float(head[2]))  # 顔が上を向く
            self.assertGreater(float(head[1] - feet[1]), 1.0)  # 頭は +Y の側（前（-Y）が上を向くように回した）
            poses_bpy.apply_pose(self.armature, pose_by_name("lying_side"), ground_meshes=[self.body])
            head, nose, feet, hand_left, hand_right = centers(("head",), ("nose",), ("foot_L", "foot_R"), ("hand_L",),
                                                              ("hand_R",))
            self.assertLess(abs(float(head[2] - feet[2])), 0.05)
            self.assertGreater(float(head[0] - feet[0]), 1.0)  # 頭は +X の側（キャラの左が下を向くように回した）
            self.assertLess(float(hand_left[2]), float(hand_right[2]))  # 左手が下、右手が上
            self.assertLess(float(nose[1]), float(head[1]))  # 前は -Y のまま

    def test_center_offset_uses_leg_length(self):
        rest = extract_bpy.bone_rest(self.armature, space="armature")
        leg = poses.leg_length_from_bones(rest.names, rest.heads)
        self.assertAlmostEqual(leg, math.hypot(0.012, 0.34) + math.hypot(0.012, 0.325), places=6)
        rest_center = rest.heads[rest.index("センター")]
        with poses_bpy.posing(self.armature):
            applied = poses_bpy.apply_pose(self.armature, pose_by_name("chair_sit"))
            moved = extract_bpy.joint_positions(self.armature, ["センター"])[0][0] - rest_center
            self.assertAlmostEqual(applied.leg_length, leg, places=9)
            # 床に合わせる分は上下だけ。前後・左右は center_offset_legs × 脚の長さのまま
            np.testing.assert_allclose(moved, [0.0, 0.0, -0.45 * leg + applied.ground_shift], atol=1e-6)
            # センター の休みの向き（Y 軸が上）を通して、骨のローカルの位置にしている
            local = rest.matrices[rest.index("センター"), :3, :3].T @ moved
            np.testing.assert_allclose(applied.center_location, local, atol=1e-6)
            # leg_length を渡すと、それを使う
            applied = poses_bpy.apply_pose(self.armature, pose_by_name("squat"), leg_length=1.0)
            moved = extract_bpy.joint_positions(self.armature, ["センター"])[0][0] - rest_center
            self.assertEqual(applied.leg_length, 1.0)
            np.testing.assert_allclose(moved, [0.0, 0.25, -0.5 + applied.ground_shift], atol=1e-6)

    def test_roll_does_not_change_the_pose(self):
        """骨のロールを変えても、ポーズの形（手の位置）は同じ"""
        with poses_bpy.posing(self.armature):
            poses_bpy.apply_pose(self.armature, pose_by_name("wave"))
            before = extract_bpy.joint_positions(self.armature, ["右手首", "右ひじ"])[0]
        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.mode_set(mode="EDIT")
        for name in ("右腕", "右腕捩", "右ひじ"):
            self.armature.data.edit_bones[name].roll = math.radians(73.0)
        bpy.ops.object.mode_set(mode="OBJECT")
        with poses_bpy.posing(self.armature):
            poses_bpy.apply_pose(self.armature, pose_by_name("wave"))
            after = extract_bpy.joint_positions(self.armature, ["右手首", "右ひじ"])[0]
        np.testing.assert_allclose(after, before, atol=1e-5)
        self.assertGreater(before[0, 2], 1.3)  # 手を振る：右の手首は肩より高い

    def test_moved_and_scaled_armature(self):
        self.armature.location = (1.0, -2.0, 0.5)
        self.armature.scale = (0.5, 0.5, 0.5)
        bpy.context.view_layer.update()
        floor = float(extract_bpy.mesh_evaluated(self.body)[:, 2].min())
        self.assertAlmostEqual(floor, 0.5, places=6)
        with poses_bpy.posing(self.armature):
            applied = poses_bpy.apply_pose(self.armature, pose_by_name("seiza"), ground_meshes=[self.body],
                                           floor_z=floor)
            lowest = float(extract_bpy.mesh_evaluated(self.body)[:, 2].min())
        self.assertAlmostEqual(lowest, floor, delta=thresholds.REPRO_TOLERANCE_LENGTH)
        self.assertLess(applied.ground_shift, 0.0)

    def test_missing_bone_makes_the_pose_partial(self):
        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.mode_set(mode="EDIT")
        self.armature.data.edit_bones.remove(self.armature.data.edit_bones["左ひじ"])
        bpy.ops.object.mode_set(mode="OBJECT")
        with poses_bpy.posing(self.armature):
            applied = poses_bpy.apply_pose(self.armature, pose_by_name("elbows_bend"))
            self.assertEqual(applied.missing, ("左ひじ",))
            self.assertTrue(applied.partial)
            self.assertEqual(set(applied.posed), {"右ひじ"})
            self.assertEqual(poses_bpy.apply_pose(self.armature, pose_by_name("head_down")).missing, ())


class ConstraintTest(PoseBpyTestCase):
    def add_ik(self):
        """mmd_tools のモデルのように、ひざに IK を付ける（目標は 足ＩＫ、骨2本）"""
        constraint = self.armature.pose.bones["左ひざ"].constraints.new("IK")
        constraint.target = self.armature
        constraint.subtarget = "左足ＩＫ"
        constraint.chain_count = 2
        limit = self.armature.pose.bones["左ひざ"].constraints.new("LIMIT_ROTATION")
        limit.name = "mmd_ik_limit_override"
        limit.use_limit_x = True
        limit.min_x = limit.max_x = 0.0
        return constraint, limit

    def test_ik_is_muted_only_while_posing(self):
        ik, limit = self.add_ik()
        self.assertEqual(poses_bpy.fighting_constraints(self.armature),
                         [("左ひざ", "IK"), ("左ひざ", "mmd_ik_limit_override")])
        with self.assertRaises(ValueError):
            poses_bpy.apply_pose(self.armature, pose_by_name("knee_bend_left"))
        with poses_bpy.posing(self.armature):
            self.assertTrue(ik.mute)
            self.assertTrue(limit.mute)
            applied = poses_bpy.apply_pose(self.armature, pose_by_name("knee_bend_left"))
            self.assertEqual(applied.overridden, ())
            self.assertAlmostEqual(self.knee_bends()["左"], REST_KNEE_BEND + 130.0, delta=0.01)
        self.assertFalse(ik.mute)
        self.assertFalse(limit.mute)

    def test_already_muted_constraint_stays_muted(self):
        ik, _ = self.add_ik()
        ik.mute = True
        with poses_bpy.posing(self.armature):
            pass
        self.assertTrue(ik.mute)

    def test_other_constraint_reports_overridden(self):
        limit = self.armature.pose.bones["左ひじ"].constraints.new("LIMIT_ROTATION")
        limit.use_limit_x = limit.use_limit_y = limit.use_limit_z = True
        limit.owner_space = "LOCAL"  # 休みの向きから回せない（最小・最大とも 0）
        with poses_bpy.posing(self.armature):
            applied = poses_bpy.apply_pose(self.armature, pose_by_name("elbows_bend"))
        self.assertEqual(applied.overridden, ("左ひじ",))
        self.assertTrue(applied.partial)

    def test_d_bone_not_following_is_reported(self):
        pose_bone = self.armature.pose.bones["左ひざD"]
        pose_bone.constraints.remove(pose_bone.constraints[mmd_model.COPY_ROTATION_NAME])
        with poses_bpy.posing(self.armature):
            applied = poses_bpy.apply_pose(self.armature, pose_by_name("knee_bend_left"))
        # ひざD が曲がらないので、その子の 足首D も 足首 と違う向きになる
        self.assertEqual(applied.overridden, ("左ひざD", "左足首D"))


class PosingStateTest(PoseBpyTestCase):
    def snapshot(self):
        return {pb.name: (tuple(pb.location), tuple(pb.rotation_quaternion), tuple(pb.scale))
                for pb in self.armature.pose.bones}

    def test_restores_pose_position_and_values_even_on_error(self):
        self.armature.pose.bones["頭"].rotation_quaternion = (0.9, 0.1, 0.2, 0.3)
        self.armature.pose.bones["センター"].location = (0.0, 0.01, 0.02)
        self.armature.data.pose_position = "REST"
        bpy.context.scene.frame_set(7)
        before = self.snapshot()
        with self.assertRaises(RuntimeError):
            with poses_bpy.posing(self.armature):
                self.assertEqual(self.armature.data.pose_position, "POSE")
                poses_bpy.apply_pose(self.armature, pose_by_name("squat"), ground_meshes=[self.body])
                raise RuntimeError("途中で止める")
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.armature.data.pose_position, "REST")
        self.assertEqual(bpy.context.scene.frame_current, 7)
        self.assertEqual(self.armature.mode, "OBJECT")

    def test_apply_pose_outside_posing_in_rest_position_is_rejected(self):
        self.armature.data.pose_position = "REST"
        with self.assertRaises(ValueError):
            poses_bpy.apply_pose(self.armature, pose_by_name("bow"))

    def test_edit_mode_is_rejected_and_kept(self):
        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.mode_set(mode="EDIT")
        with self.assertRaises(ValueError):
            with poses_bpy.posing(self.armature):
                pass
        self.assertEqual(self.armature.mode, "EDIT")
        bpy.ops.object.mode_set(mode="OBJECT")

    def test_pose_mode_is_kept(self):
        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.mode_set(mode="POSE")
        with poses_bpy.posing(self.armature):
            poses_bpy.apply_pose(self.armature, pose_by_name("bow"))
        self.assertEqual(self.armature.mode, "POSE")
        bpy.ops.object.mode_set(mode="OBJECT")

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            with poses_bpy.posing(self.body):
                pass
        with poses_bpy.posing(self.armature):
            with self.assertRaises(ValueError):
                poses_bpy.apply_pose(self.armature, "bow")
            with self.assertRaises(ValueError):
                poses_bpy.apply_pose(self.armature, pose_by_name("seiza"), ground_meshes=[self.body],
                                     floor_z=math.nan)


if __name__ == "__main__":
    unittest.main()

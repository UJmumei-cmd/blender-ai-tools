"""A5 確かめの動き：ポーズの並び（motion/poses.py）を確かめる（bpy なし）"""

import math
import unittest

import numpy as np

from blender_ai_tools import bones, texts
from blender_ai_tools.motion import poses
from blender_ai_tools.motion.poses import (CENTER_BONE, JOINT_LIMITS_DEG, POSES, ROOT_BONE, AppliedPose, PoseSpec,
                                           axis_quaternion, find_overridden, leg_length_from_bones, local_rotation,
                                           pose_by_name, quat_multiply, resolve_pose, rotation_deltas)

# MARK: - 答え合わせ用の計算（poses とは別に書いた物）


def rotation_matrix(axis, deg: float) -> np.ndarray:
    """axis まわりに deg 度回す行列（ロドリゲスの式。右手の決まり）"""
    n = np.asarray(axis, dtype=np.float64)
    n = n / np.linalg.norm(n)
    t = math.radians(deg)
    k = np.array([[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]])
    return np.eye(3) + math.sin(t) * k + (1.0 - math.cos(t)) * (k @ k)


def quat_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def mirror_rotation(rotation):
    """左右を鏡に映した回転（名前は左右を入れかえ、軸は (x, -y, -z)）"""
    bone, axis, angle = rotation
    other = bones.mirror_name(bone)
    return (bone if other is None else other, (axis[0], -axis[1], -axis[2]), angle)


def rounded(rotations):
    return sorted((bone, tuple(round(v, 9) for v in axis), round(angle, 9)) for bone, axis, angle in rotations)


def left_name(bone: str) -> str:
    """右の骨は左の名前にする（JOINT_LIMITS_DEG を引くため）"""
    canonical = bones.canonical_bone(bone)
    if canonical.side == "R":
        return bones.mirror_name(bone)
    return bone


# MARK: - ポーズの並び


class PoseCatalogTest(unittest.TestCase):
    def test_about_forty_poses_with_unique_names(self):
        names = [pose.name for pose in POSES]
        self.assertEqual(len(names), 44)
        self.assertEqual(len(set(names)), len(names))
        self.assertTrue(35 <= len(POSES) <= 50)

    def test_categories(self):
        by_category = {category: {pose.name for pose in POSES if pose.category == category}
                       for category in poses.CATEGORIES}
        self.assertEqual(by_category["lying"], {"lying_back", "lying_side"})
        # ロードマップ・KICKOFF のふだんの動き（両手を上げる・腕を組む・手をたたく・ほおに手・あごに手・おじぎ・振り返る・
        # 歩く・走る・しゃがむ・いすの端に座る・正座・ひざに手・手を振る・刀を構える）
        self.assertEqual(by_category["daily"], {
            "arms_up", "arms_crossed", "clap", "hands_on_cheeks", "hand_on_chin", "bow", "look_back", "walk", "run",
            "squat", "chair_sit", "seiza", "hands_on_knees", "wave", "sword_stance"})
        self.assertEqual(len(by_category["rom"]), 27)

    def test_texts_exist_in_both_languages(self):
        for pose in POSES:
            with self.subTest(pose=pose.name):
                self.assertEqual(pose.text_key, f"motion.pose_{pose.name}")
                self.assertIn(pose.text_key, texts.TEXTS)
                ja = texts.text(pose.text_key, language="ja")
                en = texts.text(pose.text_key, language="en")
                self.assertTrue(ja and en)
                self.assertNotEqual(ja, en)
        labels = [texts.text(pose.text_key, language="ja") for pose in POSES]
        self.assertEqual(len(set(labels)), len(labels), "ポーズの日本語の名前が重なっています")
        self.assertEqual(texts.problems(), [])

    def test_bones_are_canonical_names(self):
        canonical = set(bones.canonical_names())
        for pose in POSES:
            for bone, axis, angle in pose.rotations:
                with self.subTest(pose=pose.name, bone=bone):
                    self.assertIn(bone, canonical)

    def test_axes_are_unit_and_angles_positive(self):
        for pose in POSES:
            for bone, axis, angle in pose.rotations:
                with self.subTest(pose=pose.name, bone=bone):
                    self.assertAlmostEqual(float(np.linalg.norm(axis)), 1.0, places=12)
                    self.assertGreater(angle, 0.0)
                    self.assertLessEqual(angle, 180.0)

    def test_angles_stay_within_human_range(self):
        """骨1本に付ける角度の合計が、JOINT_LIMITS_DEG（ふつうの人の動く範囲の目安）の中"""
        for pose in POSES:
            totals: dict[str, float] = {}
            for bone, _, angle in pose.rotations:
                totals[bone] = totals.get(bone, 0.0) + angle
            for bone, total in totals.items():
                with self.subTest(pose=pose.name, bone=bone):
                    self.assertIn(left_name(bone), JOINT_LIMITS_DEG, "範囲の決まりが無い骨は回さない")
                    self.assertLessEqual(total, JOINT_LIMITS_DEG[left_name(bone)])

    def test_knees_and_elbows_only_bend_the_normal_way(self):
        """ひざは、すねの先が後ろ（+Y）へ行く向きにだけ曲げる。ひじは、前腕が前（-Y）へ行く向きにだけ曲げる"""
        arm_down = math.radians(37.5)  # テスト用のモデルの A ポーズ
        forearm = {"L": np.array([math.cos(arm_down), 0.0, -math.sin(arm_down)]),
                   "R": np.array([-math.cos(arm_down), 0.0, -math.sin(arm_down)])}
        knees = elbows = 0
        for pose in POSES:
            for bone, axis, angle in pose.rotations:
                canonical = bones.canonical_bone(bone)
                small = rotation_matrix(axis, 5.0)
                if bone in ("左ひざ", "右ひざ"):
                    knees += 1
                    with self.subTest(pose=pose.name, bone=bone):
                        self.assertEqual(axis, (1.0, 0.0, 0.0))
                        self.assertGreater((small @ np.array([0.0, 0.0, -1.0]))[1], 0.0)
                if bone in ("左ひじ", "右ひじ"):
                    elbows += 1
                    with self.subTest(pose=pose.name, bone=bone):
                        self.assertLess((small @ forearm[canonical.side])[1], 0.0)
        self.assertGreaterEqual(knees, 10)
        self.assertGreaterEqual(elbows, 10)

    def test_left_right_pairs_are_mirrors(self):
        pairs = [(pose.name, pose.name[:-len("_left")] + "_right") for pose in POSES if pose.name.endswith("_left")]
        self.assertEqual([left for left, _ in pairs],
                         ["head_turn_left", "head_tilt_left", "torso_side_left", "torso_twist_left", "leg_front_left",
                          "leg_side_left", "leg_back_left", "knee_bend_left"])
        for left, right in pairs:
            with self.subTest(pose=left):
                self.assertEqual(rounded(pose_by_name(right).rotations),
                                 rounded(mirror_rotation(r) for r in pose_by_name(left).rotations))

    def test_symmetric_poses_are_symmetric(self):
        for name in ("shoulders_up", "arms_down", "arms_back", "elbows_bend", "forearms_twist", "fists", "tiptoe",
                     "arms_up", "clap", "hands_on_cheeks", "bow", "squat", "chair_sit", "seiza", "hands_on_knees",
                     "lying_back", "head_down", "head_up", "torso_forward", "torso_back"):
            pose = pose_by_name(name)
            with self.subTest(pose=name):
                self.assertEqual(rounded(pose.rotations), rounded(mirror_rotation(r) for r in pose.rotations))

    def test_ground_lying_and_center_flags(self):
        ground = {pose.name for pose in POSES if pose.ground}
        self.assertEqual(ground, {"tiptoe", "walk", "run", "squat", "chair_sit", "seiza", "hands_on_knees",
                                  "sword_stance", "lying_back", "lying_side"})
        self.assertEqual({pose.name for pose in POSES if pose.lying}, {"lying_back", "lying_side"})
        offsets = {pose.name: pose.center_offset_legs for pose in POSES if any(pose.center_offset_legs)}
        self.assertEqual(set(offsets), {"squat", "chair_sit", "seiza"})
        for name, offset in offsets.items():
            with self.subTest(pose=name):
                self.assertLess(offset[2], 0.0, "座る・しゃがむ姿勢では センター が下がる")
        for pose in POSES:
            rotates_root = any(bone == ROOT_BONE for bone, _, _ in pose.rotations)
            self.assertEqual(rotates_root, pose.lying)

    def test_lying_poses_turn_the_body(self):
        back = dict((b, (a, g)) for b, a, g in pose_by_name("lying_back").rotations)[ROOT_BONE]
        # あおむけ：前（-Y）が上（+Z）を向く
        np.testing.assert_allclose(rotation_matrix(*back) @ [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], atol=1e-12)
        side = dict((b, (a, g)) for b, a, g in pose_by_name("lying_side").rotations)[ROOT_BONE]
        # 左を下にした横向き：キャラの左（+X）が下（-Z）を向く
        np.testing.assert_allclose(rotation_matrix(*side) @ [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], atol=1e-12)

    def test_bones_property_adds_center_when_needed(self):
        self.assertEqual(pose_by_name("head_down").bones, ("首", "頭"))
        self.assertEqual(pose_by_name("tiptoe").bones[-1], CENTER_BONE)
        self.assertEqual(pose_by_name("squat").bones[-1], CENTER_BONE)
        self.assertFalse(pose_by_name("wave").needs_center)
        self.assertTrue(pose_by_name("seiza").needs_center)

    def test_pose_by_name(self):
        self.assertIs(pose_by_name("seiza"), POSES[[p.name for p in POSES].index("seiza")])
        with self.assertRaises(ValueError):
            pose_by_name("nothing")

    def test_axis_rotations_compose_in_order(self):
        """同じ骨の回転は、書いた順に当てる（あとの回転も、親に付いた休みの向きの軸）：q2 * q1"""
        pose = pose_by_name("arms_back")
        q = pose.axis_rotations()["左腕"]
        expected = rotation_matrix((1.0, 0.0, 0.0), 40.0) @ rotation_matrix((0.0, 1.0, 0.0), 40.0)
        np.testing.assert_allclose(quat_matrix(q), expected, atol=1e-12)
        # 順を逆にした物とは違う（順を取り違えるまちがいを見つける）
        wrong = rotation_matrix((0.0, 1.0, 0.0), 40.0) @ rotation_matrix((1.0, 0.0, 0.0), 40.0)
        self.assertGreater(float(np.abs(quat_matrix(q) - wrong).max()), 0.1)


class PoseSpecValidationTest(unittest.TestCase):
    def make(self, **changes):
        values = dict(name="test_pose", text_key="motion.pose_test_pose", category="rom",
                      rotations=(("左腕", (0.0, -2.0, 0.0), 30.0),))
        values.update(changes)
        return PoseSpec(**values)

    def test_normalizes_axes_and_keeps_order(self):
        pose = self.make(rotations=[("左腕", [0, -2, 0], 30), ("左腕", (1.0, 0.0, 0.0), 10.0)])
        self.assertEqual(pose.rotations, (("左腕", (0.0, -1.0, 0.0), 30.0), ("左腕", (1.0, 0.0, 0.0), 10.0)))
        self.assertIsInstance(pose.rotations, tuple)

    def test_rejects_bad_values(self):
        cases = {
            "name": dict(name="Bad Name", text_key="motion.pose_Bad Name"),
            "text_key": dict(text_key="motion.other"),
            "category": dict(category="sport"),
            "not canonical": dict(rotations=(("LeftArm", (0, 1, 0), 10.0),)),
            "lr form": dict(rotations=(("腕.L", (0, 1, 0), 10.0),)),
            "zero axis": dict(rotations=(("左腕", (0, 0, 0), 10.0),)),
            "short axis": dict(rotations=(("左腕", (0, 1), 10.0),)),
            "nan axis": dict(rotations=(("左腕", (math.nan, 1, 0), 10.0),)),
            "zero angle": dict(rotations=(("左腕", (0, 1, 0), 0.0),)),
            "negative angle": dict(rotations=(("左腕", (0, 1, 0), -10.0),)),
            "big angle": dict(rotations=(("左腕", (0, 1, 0), 190.0),)),
            "bool angle": dict(rotations=(("左腕", (0, 1, 0), True),)),
            "pair": dict(rotations=(("左腕", (0, 1, 0)),)),
            "string rotations": dict(rotations="左腕"),
            "offset": dict(center_offset_legs=(0.0, math.nan, 0.0)),
            "offset shape": dict(center_offset_legs=(0.0, 0.0)),
            "ground": dict(ground=1),
            "root in rom": dict(rotations=((ROOT_BONE, (1, 0, 0), 90.0),)),
            "lying without root": dict(category="lying", lying=True),
            "lying flag only": dict(lying=True, rotations=((ROOT_BONE, (1, 0, 0), 90.0),)),
        }
        for label, change in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    self.make(**change)

    def test_lying_pose_is_accepted(self):
        pose = self.make(category="lying", lying=True, rotations=((ROOT_BONE, (1, 0, 0), 90.0),))
        self.assertTrue(pose.lying)


# MARK: - 回転


class LocalRotationTest(unittest.TestCase):
    def test_identity_rest_gives_axis_rotation(self):
        q = local_rotation(np.eye(4), (0.0, 0.0, 2.0), 90.0)
        np.testing.assert_allclose(q, (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)), atol=1e-12)

    def test_known_value(self):
        """休みの向きが Z のまわりに 90 度：軸 +X は、骨のローカルでは -Y"""
        rest = np.eye(4)
        rest[:3, :3] = rotation_matrix((0.0, 0.0, 1.0), 90.0)
        q = local_rotation(rest, (1.0, 0.0, 0.0), 90.0)
        half = math.sqrt(0.5)
        np.testing.assert_allclose(q, (half, 0.0, -half, 0.0), atol=1e-12)

    def test_roll_independence(self):
        """同じ向きの骨で、ロール（骨の Y 軸まわりのねじれ）だけが違っても、アーマチュアの中での回り方は同じ"""
        direction = np.array([math.cos(math.radians(37.5)), 0.0, -math.sin(math.radians(37.5))])
        # 骨の Y 軸を direction にする休みの向き（ロール 0）
        y = direction
        x = np.cross(y, [0.0, 0.0, 1.0])
        x /= np.linalg.norm(x)
        z = np.cross(x, y)
        base = np.column_stack([x, y, z])
        results = []
        for roll in (0.0, 45.0, 137.0, -90.0):
            rest = np.eye(4)
            rest[:3, :3] = rotation_matrix(direction, roll) @ base
            rest[:3, 3] = (0.15, 0.0, 1.13)
            np.testing.assert_allclose(rest[:3, 1], direction, atol=1e-12)
            q = local_rotation(rest, (0.0, -1.0, 0.0), 120.0)
            posed = rest[:3, :3] @ quat_matrix(q)
            # 回したあとの骨の向き（Y 軸）と、アーマチュアの中での回転
            results.append((posed[:, 1], posed @ rest[:3, :3].T))
        expected_direction = rotation_matrix((0.0, -1.0, 0.0), 120.0) @ direction
        for direction_after, world_rotation in results:
            np.testing.assert_allclose(direction_after, expected_direction, atol=1e-12)
            np.testing.assert_allclose(world_rotation, rotation_matrix((0.0, -1.0, 0.0), 120.0), atol=1e-12)
        # 腕は 37.5 度下から 120 度上がって、水平から 82.5 度上を向く
        self.assertAlmostEqual(math.degrees(math.asin(expected_direction[2])), 82.5, places=9)

    def test_scale_and_translation_are_ignored(self):
        rest = np.eye(4)
        rest[:3, :3] = 2.0 * rotation_matrix((0.0, 1.0, 0.0), 30.0)
        rest[:3, 3] = (1.0, 2.0, 3.0)
        unscaled = np.eye(3)
        unscaled[:] = rotation_matrix((0.0, 1.0, 0.0), 30.0)
        np.testing.assert_allclose(local_rotation(rest, (1, 0, 0), 40.0), local_rotation(unscaled, (1, 0, 0), 40.0),
                                   atol=1e-12)

    def test_w_is_not_negative(self):
        for angle in (10.0, 179.0, 180.0):
            self.assertGreaterEqual(local_rotation(np.eye(3), (0, 0, 1), angle)[0], 0.0)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            local_rotation(np.eye(2), (1, 0, 0), 10.0)
        with self.assertRaises(ValueError):
            local_rotation(np.eye(4), (0, 0, 0), 10.0)
        with self.assertRaises(ValueError):
            local_rotation(np.eye(4), (1, 0, 0), math.nan)
        with self.assertRaises(ValueError):
            local_rotation(np.eye(4), (1, 0, 0), True)
        with self.assertRaises(ValueError):
            local_rotation(np.full((4, 4), math.nan), (1, 0, 0), 10.0)

    def test_quaternion_helpers(self):
        q1 = axis_quaternion((0, 0, 1), 90.0)
        q2 = axis_quaternion((1, 0, 0), 90.0)
        np.testing.assert_allclose(quat_matrix(quat_multiply(q2, q1)),
                                   rotation_matrix((1, 0, 0), 90.0) @ rotation_matrix((0, 0, 1), 90.0), atol=1e-12)

    def test_rotation_deltas(self):
        rest = np.stack([np.eye(4), np.eye(4)])
        rest[1, :3, :3] = 3.0 * rotation_matrix((0, 0, 1), 90.0)
        posed = rest.copy()
        posed[0, :3, :3] = rotation_matrix((0, 0, 1), 30.0)
        posed[1, :3, :3] = 3.0 * rotation_matrix((1, 0, 0), 20.0) @ rotation_matrix((0, 0, 1), 90.0)
        deltas = rotation_deltas(rest, posed)
        np.testing.assert_allclose(deltas[0], axis_quaternion((0, 0, 1), 30.0), atol=1e-12)
        np.testing.assert_allclose(deltas[1], axis_quaternion((1, 0, 0), 20.0), atol=1e-12)
        self.assertEqual(rotation_deltas(np.zeros((0, 4, 4)), np.zeros((0, 4, 4))).shape, (0, 4))
        with self.assertRaises(ValueError):
            rotation_deltas(rest, posed[:1])
        with self.assertRaises(ValueError):
            rotation_deltas(np.eye(4), np.eye(4))


# MARK: - モデルの骨に当てる


def skeleton(names, rolls=None):
    """休みの行列（ロールを Z まわりの回転で入れる）"""
    matrices = np.repeat(np.eye(4)[None], len(names), axis=0)
    for index, roll in (rolls or {}).items():
        matrices[index, :3, :3] = rotation_matrix((0.0, 0.0, 1.0), roll)
    return matrices


class ResolvePoseTest(unittest.TestCase):
    def test_maps_names_and_reports_missing(self):
        pose = pose_by_name("clap")
        names = ["腕.L", "腕.R", "ひじ.L", "センター"]  # mmd_tools の左右の書き方。右ひじが無い
        resolved = resolve_pose(pose, names, skeleton(names))
        self.assertEqual(set(resolved.rotations), {"腕.L", "腕.R", "ひじ.L"})
        self.assertEqual(resolved.canonical["腕.L"], "左腕")
        self.assertEqual(resolved.missing, ("右ひじ",))
        self.assertEqual(resolved.center, "センター")
        np.testing.assert_allclose(resolved.axis_rotations["腕.L"], pose.axis_rotations()["左腕"], atol=1e-12)
        # 休みの向きが単位行列なら、ローカルの回転はアーマチュアの中の回転と同じ
        np.testing.assert_allclose(resolved.rotations["腕.L"], pose.axis_rotations()["左腕"], atol=1e-12)

    def test_local_rotation_uses_each_bone_rest(self):
        pose = pose_by_name("arms_up")
        names = ["左腕", "右腕", "左肩", "右肩"]
        resolved = resolve_pose(pose, names, skeleton(names, {0: 90.0}))
        rest = rotation_matrix((0.0, 0.0, 1.0), 90.0)
        posed = rest @ quat_matrix(resolved.rotations["左腕"])
        np.testing.assert_allclose(posed @ rest.T, quat_matrix(pose.axis_rotations()["左腕"]), atol=1e-12)

    def test_center_offset(self):
        pose = pose_by_name("squat")  # center_offset_legs = (0, 0.25, -0.5)
        names = ["センター"]
        resolved = resolve_pose(pose, names, skeleton(names), leg_length=0.8)
        np.testing.assert_allclose(resolved.center_location, (0.0, 0.2, -0.4), atol=1e-12)
        # センター の休みの向きが Z のまわりに 90 度なら、骨のローカルでは (0.2, 0, -0.4)
        rotated = resolve_pose(pose, names, skeleton(names, {0: 90.0}), leg_length=0.8)
        np.testing.assert_allclose(rotated.center_location, (0.2, 0.0, -0.4), atol=1e-12)
        self.assertEqual(resolve_pose(pose_by_name("arms_up"), names, skeleton(names)).center_location,
                         (0.0, 0.0, 0.0))

    def test_center_offset_needs_leg_length(self):
        pose = pose_by_name("squat")
        legs = ["センター", "左足", "左ひざ", "左足首", "右足", "右ひざ", "右足首"]
        with self.assertRaises(ValueError):
            resolve_pose(pose, legs, skeleton(legs))
        for bad in (0.0, -1.0, math.nan, True, "1"):
            with self.subTest(leg_length=bad):
                with self.assertRaises(ValueError):
                    resolve_pose(pose, legs, skeleton(legs), leg_length=bad)
        # 脚の骨が無いモデルでは、センター を動かせないので、足りない骨として数える
        resolved = resolve_pose(pose, ["センター"], skeleton(["センター"]))
        self.assertEqual(resolved.center_location, (0.0, 0.0, 0.0))
        self.assertEqual(set(resolved.missing), {"左足", "左ひざ", "左足首", "右足", "右ひざ", "右足首", "左腕", "右腕",
                                                 "上半身"})

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            resolve_pose("arms_up", ["左腕"], skeleton(["左腕"]))
        with self.assertRaises(ValueError):
            resolve_pose(pose_by_name("arms_up"), ["左腕"], skeleton(["左腕", "右腕"]))
        with self.assertRaises(ValueError):
            resolve_pose(pose_by_name("arms_up"), "左腕", skeleton(["左腕"]))


class LegLengthTest(unittest.TestCase):
    HEADS = {"左足": (0.08, 0.0, 0.74), "左ひざ": (0.08, -0.012, 0.40), "左足首": (0.08, 0.0, 0.075)}

    def expected(self) -> float:
        return math.hypot(0.012, 0.34) + math.hypot(0.012, 0.325)

    def test_visible_bones(self):
        names = list(self.HEADS) + ["右足", "右ひざ", "右足首"]
        heads = [self.HEADS[n] for n in self.HEADS] + [(-x, y, z) for x, y, z in self.HEADS.values()]
        self.assertAlmostEqual(leg_length_from_bones(names, heads), self.expected(), places=12)

    def test_d_bones_and_one_side(self):
        names = ["左足D", "左ひざD", "左足首D"]
        self.assertAlmostEqual(leg_length_from_bones(names, list(self.HEADS.values())), self.expected(), places=12)

    def test_missing_and_bad(self):
        self.assertIsNone(leg_length_from_bones(["左足", "左ひざ"], [(0, 0, 1), (0, 0, 0.5)]))
        with self.assertRaises(ValueError):
            leg_length_from_bones(["左足"], [(0, 0, 1), (0, 0, 0)])


class FindOverriddenTest(unittest.TestCase):
    NAMES = ["センター", "左足", "左ひざ", "左足D", "左ひざD"]
    PARENTS = np.array([-1, 0, 1, 0, 3])

    def setUp(self):
        self.pose = PoseSpec(name="knee", text_key="motion.pose_knee", category="rom",
                             rotations=(("左足", (-1.0, 0.0, 0.0), 30.0), ("左ひざ", (1.0, 0.0, 0.0), 60.0)))
        self.resolved = resolve_pose(self.pose, self.NAMES, skeleton(self.NAMES))
        flex = axis_quaternion((-1, 0, 0), 30.0)
        knee = quat_multiply(flex, axis_quaternion((1, 0, 0), 60.0))
        self.deltas = np.array([[1.0, 0, 0, 0], flex, knee, flex, knee])

    def test_followed(self):
        self.assertEqual(find_overridden(self.resolved, self.NAMES, self.PARENTS, self.deltas), ())

    def test_bone_kept_by_constraint(self):
        deltas = self.deltas.copy()
        deltas[2] = deltas[4] = deltas[1]  # ひざが曲がらなかった（IK などに戻された。ひざD も ひざ について来る）
        self.assertEqual(find_overridden(self.resolved, self.NAMES, self.PARENTS, deltas), ("左ひざ",))

    def test_d_bone_not_following(self):
        deltas = self.deltas.copy()
        deltas[4] = deltas[3]  # ひざD が ひざ について来ない
        self.assertEqual(find_overridden(self.resolved, self.NAMES, self.PARENTS, deltas), ("左ひざD",))

    def test_tolerance(self):
        deltas = self.deltas.copy()
        deltas[2] = quat_multiply(deltas[2], axis_quaternion((1, 0, 0), 0.5))
        self.assertEqual(find_overridden(self.resolved, self.NAMES, self.PARENTS, deltas), ())
        deltas[2] = quat_multiply(self.deltas[2], axis_quaternion((1, 0, 0), 1.5))
        self.assertEqual(find_overridden(self.resolved, self.NAMES, self.PARENTS, deltas), ("左ひざ", "左ひざD"))

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            find_overridden(self.resolved, self.NAMES, self.PARENTS[:2], self.deltas)
        with self.assertRaises(ValueError):
            find_overridden(self.resolved, self.NAMES, self.PARENTS, self.deltas[:, :3])

    def test_applied_pose_partial(self):
        self.assertFalse(AppliedPose(pose="a").partial)
        self.assertTrue(AppliedPose(pose="a", missing=("左ひじ",)).partial)
        self.assertTrue(AppliedPose(pose="a", overridden=("左ひざ",)).partial)


if __name__ == "__main__":
    unittest.main()

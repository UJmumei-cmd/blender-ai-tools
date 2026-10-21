"""A4 関節と足（kinematics）を確かめる：逆曲がり・床への沈み・足すべり・届かないフレーム・骨の跳び"""

import math
import tracemalloc
import unittest

import numpy as np

from blender_ai_tools import thresholds
from blender_ai_tools.measure import kinematics as km
from blender_ai_tools.measure.results import Signal

RNG_SEED = 20261010


# MARK: - テスト用の道具（kinematics とは別に書いた、答え合わせ用の計算）

def rotation_matrix(axis, deg: float) -> np.ndarray:
    """axis まわりに deg 度回す行列（ロドリゲスの式。右手の決まり）"""
    n = np.asarray(axis, dtype=np.float64)
    n = n / np.linalg.norm(n)
    t = math.radians(deg)
    k = np.array([[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]])
    return np.eye(3) + math.sin(t) * k + (1.0 - math.cos(t)) * (k @ k)


def rotate(points, axis, deg: float) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ rotation_matrix(axis, deg).T


def quat(axis, deg: float) -> np.ndarray:
    """axis まわりに deg 度回すクォータニオン (w, x, y, z)"""
    n = np.asarray(axis, dtype=np.float64)
    n = n / np.linalg.norm(n)
    half = math.radians(deg) / 2.0
    return np.concatenate(([math.cos(half)], math.sin(half) * n))


def quat_mul(a, b) -> np.ndarray:
    """ハミルトンの積 a·b（w, x, y, z）"""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_matrix(q) -> np.ndarray:
    """長さ 1 のクォータニオンの回転行列"""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def leg(bend_deg: float, *, thigh: float = 0.5, shin: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """立った脚（モデルは -Y 向き）。ひざを bend_deg 度曲げる。正なら足首が後ろ（+Y）、負なら前（-Y）"""
    hip = np.array([0.0, 0.0, 1.0])
    knee = hip + np.array([0.0, 0.0, -thigh])
    t = math.radians(bend_deg)
    ankle = knee + shin * np.array([0.0, math.sin(t), -math.cos(t)])
    return hip, knee, ankle


def legs(bends) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """フレームごとの脚 (F, 3) を3つ"""
    parts = [leg(b) for b in bends]
    return tuple(np.array([p[i] for p in parts]) for i in range(3))


LEFT = np.array([1.0, 0.0, 0.0])  # キャラの左（+X）


def random_unit_quaternions(rng: np.random.Generator, count: int) -> np.ndarray:
    q = rng.normal(size=(count, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


# MARK: - 角度

class AngleAtTest(unittest.TestCase):
    def test_exact_angles(self):
        origin = [0.0, 0.0, 0.0]
        self.assertAlmostEqual(float(km.angle_at([1, 0, 0], origin, [0, 1, 0])), 90.0, places=12)
        self.assertAlmostEqual(float(km.angle_at([-1, 0, 0], origin, [2, 0, 0])), 180.0, places=12)
        self.assertAlmostEqual(float(km.angle_at([1, 0, 0], origin, [3, 0, 0])), 0.0, places=12)
        # 正三角形の角は 60
        self.assertAlmostEqual(float(km.angle_at([0, 0, 0], [1, 0, 0], [0.5, math.sqrt(3) / 2, 0])), 60.0, places=12)

    def test_angle_is_measured_at_b_with_offsets_and_lengths(self):
        b = np.array([2.0, -3.0, 5.0])
        a = b + 0.2 * np.array([1.0, 0.0, 0.0])
        c = b + 7.0 * np.array([1.0, 1.0, 0.0])
        self.assertAlmostEqual(float(km.angle_at(a, b, c)), 45.0, places=12)
        # 角は b で測る（a で測ると違う答えになる）
        self.assertAlmostEqual(float(km.angle_at(c, b, a)), 45.0, places=12)
        self.assertNotAlmostEqual(float(km.angle_at(b, a, c)), 45.0, places=3)

    def test_tiny_angle_is_accurate(self):
        # acos(内積) では 0 になってしまう小ささ
        angle = float(km.angle_at([1, 0, 0], [0, 0, 0], [1, 1e-8, 0]))
        self.assertAlmostEqual(angle / math.degrees(1e-8), 1.0, places=9)

    def test_broadcast_over_frames(self):
        t = np.radians([0.0, 30.0, 90.0, 150.0, 180.0])
        c = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)
        result = km.angle_at([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], c)
        self.assertEqual(result.shape, (5,))
        np.testing.assert_allclose(result, [0.0, 30.0, 90.0, 150.0, 180.0], atol=1e-12)

    def test_degenerate_and_nan_give_nan(self):
        origin = [0.0, 0.0, 0.0]
        self.assertTrue(math.isnan(float(km.angle_at(origin, origin, [1, 0, 0]))))
        self.assertTrue(math.isnan(float(km.angle_at([1, 0, 0], origin, origin))))
        self.assertTrue(math.isnan(float(km.angle_at([1, 0, 0], origin, [math.nan, 0, 0]))))
        # 1µm より短い腕は向きが決まらない。それより長ければ測る
        self.assertTrue(math.isnan(float(km.angle_at([1e-7, 0, 0], origin, [0, 1, 0]))))
        self.assertAlmostEqual(float(km.angle_at([1e-5, 0, 0], origin, [0, 1, 0])), 90.0, places=9)

    def test_empty(self):
        self.assertEqual(km.angle_at(np.zeros((0, 3)), [0, 0, 0], np.zeros((0, 3))).shape, (0,))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            km.angle_at([1, 0], [0, 0], [0, 1])  # 2次元の点
        with self.assertRaises(ValueError):
            km.angle_at([math.inf, 0, 0], [0, 0, 0], [0, 1, 0])
        with self.assertRaises(ValueError):
            km.angle_at([True, False, False], [0, 0, 0], [0, 1, 0])  # 真偽は数ではない
        with self.assertRaises(ValueError):
            km.angle_at(np.zeros((2, 3)), np.zeros((3, 3)), np.zeros((2, 3)))  # 形が合わない
        with self.assertRaises(ValueError):
            km.angle_at(["a", "b", "c"], [0, 0, 0], [0, 1, 0])


class SignedAngleTest(unittest.TestCase):
    def test_sign_follows_right_hand_rule_and_axis(self):
        x, y = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
        self.assertAlmostEqual(float(km.signed_angle(x, y, [0, 0, 1])), 90.0, places=12)
        self.assertAlmostEqual(float(km.signed_angle(x, y, [0, 0, -1])), -90.0, places=12)
        self.assertAlmostEqual(float(km.signed_angle(y, x, [0, 0, 1])), -90.0, places=12)
        # 軸の長さは関係ない
        self.assertAlmostEqual(float(km.signed_angle(x, y, [0, 0, 7.5])), 90.0, places=12)

    def test_components_along_axis_are_ignored(self):
        self.assertAlmostEqual(float(km.signed_angle([1, 0, 5], [0, 1, -3], [0, 0, 1])), 90.0, places=12)
        self.assertAlmostEqual(float(km.signed_angle([2, 0, 0], [1, 1, 100], [0, 0, 1])), 45.0, places=12)

    def test_known_rotations_about_random_axes(self):
        rng = np.random.default_rng(RNG_SEED)
        for angle in (-179.0, -120.0, -45.0, -0.5, 0.0, 0.5, 30.0, 90.0, 135.0, 179.0):
            axis = rng.normal(size=3)
            u = rng.normal(size=3)
            v = rotation_matrix(axis, angle) @ u * 2.5  # 長さを変えても同じ
            with self.subTest(angle=angle):
                self.assertAlmostEqual(float(km.signed_angle(u, v, axis)), angle, places=9)

    def test_range_is_minus_180_exclusive_to_180_inclusive(self):
        self.assertEqual(float(km.signed_angle([1, 0, 0], [-1, 0, 0], [0, 0, 1])), 180.0)
        self.assertEqual(float(km.signed_angle([1, 0, 0], [-1, -0.0, 0], [0, 0, 1])), 180.0)
        # 本当は -180 + 6e-299 度。double では -180 に丸まるので、+180 にそろえる
        self.assertEqual(float(km.signed_angle([1, 0, 0], [-1, -1e-300, 0], [0, 0, 1])), 180.0)
        rng = np.random.default_rng(RNG_SEED)
        result = km.signed_angle(rng.normal(size=(2000, 3)), rng.normal(size=(2000, 3)), rng.normal(size=(2000, 3)))
        self.assertTrue(np.all(result > -180.0))
        self.assertTrue(np.all(result <= 180.0))

    def test_degenerate_gives_nan(self):
        x, y, z = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
        self.assertTrue(math.isnan(float(km.signed_angle(x, y, [0, 0, 0]))))  # 軸が無い
        self.assertTrue(math.isnan(float(km.signed_angle(z, y, z))))  # u が軸の向き
        self.assertTrue(math.isnan(float(km.signed_angle(x, [0, 0, -2], z))))  # v が軸の逆向き
        self.assertTrue(math.isnan(float(km.signed_angle(x, [0, 0, 0], z))))  # v が無い

    def test_broadcast(self):
        t = np.radians([10.0, -20.0, 170.0])
        v = np.stack([np.cos(t), np.sin(t), np.zeros(3)], axis=1)
        np.testing.assert_allclose(km.signed_angle([1, 0, 0], v, [0, 0, 1]), [10.0, -20.0, 170.0], atol=1e-12)


# MARK: - ひざ・ひじの逆曲がり

class KneeBendTest(unittest.TestCase):
    def test_straight_leg_is_zero(self):
        self.assertEqual(float(km.knee_bend(*leg(0.0), LEFT)), 0.0)

    def test_normal_bend_is_positive(self):
        hip, knee, ankle = leg(90.0)
        np.testing.assert_allclose(ankle, [0.0, 0.5, 0.5], atol=1e-15)  # 足首は後ろ（+Y）
        self.assertAlmostEqual(float(km.knee_bend(hip, knee, ankle, LEFT)), 90.0, places=12)
        self.assertAlmostEqual(float(km.knee_bend(*leg(30.0), LEFT)), 30.0, places=12)

    def test_reverse_bend_is_negative(self):
        hip, knee, ankle = leg(-90.0)
        np.testing.assert_allclose(ankle, [0.0, -0.5, 0.5], atol=1e-15)  # 足首は前（-Y）
        self.assertAlmostEqual(float(km.knee_bend(hip, knee, ankle, LEFT)), -90.0, places=12)
        self.assertAlmostEqual(float(km.knee_bend(*leg(-5.0), LEFT)), -5.0, places=12)

    def test_lateral_from_hips_works_for_both_legs(self):
        left_hip, right_hip = np.array([0.1, 0.0, 1.0]), np.array([-0.1, 0.0, 1.0])
        lateral = left_hip - right_hip  # キャラの左（+X）
        for hip in (left_hip, right_hip):
            knee = hip + [0.0, 0.0, -0.45]
            behind = knee + [0.0, 0.3, -0.3]  # 後ろへ 45 度
            in_front = knee + [0.0, -0.3, -0.3]  # 前へ 45 度
            self.assertAlmostEqual(float(km.knee_bend(hip, knee, behind, lateral)), 45.0, places=12)
            self.assertAlmostEqual(float(km.knee_bend(hip, knee, in_front, lateral)), -45.0, places=12)

    def test_frames_and_flipped_lateral(self):
        bends = [-30.0, -5.0, 0.0, 10.0, 45.0, 90.0, 135.0, 170.0]
        hip, knee, ankle = legs(bends)
        np.testing.assert_allclose(km.knee_bend(hip, knee, ankle, LEFT), bends, atol=1e-9)
        # 軸を逆（キャラの右）にすると符号が変わる
        np.testing.assert_allclose(km.knee_bend(hip, knee, ankle, -LEFT), [-b for b in bends], atol=1e-9)
        # 軸はフレームごとでもよい
        lateral = np.tile(LEFT * 0.2, (len(bends), 1))
        np.testing.assert_allclose(km.knee_bend(hip, knee, ankle, lateral), bends, atol=1e-9)

    def test_invariant_when_pose_and_lateral_turn_together(self):
        bends = [-30.0, -5.0, 0.0, 10.0, 90.0, 150.0]
        hip, knee, ankle = legs(bends)
        expected = km.knee_bend(hip, knee, ankle, LEFT)
        for deg in (90.0, -90.0, 37.0, 180.0):
            with self.subTest(turn=deg):
                turned = [rotate(p, [0, 0, 1], deg) for p in (hip, knee, ankle)]
                result = km.knee_bend(*turned, rotate(LEFT, [0, 0, 1], deg))
                np.testing.assert_allclose(result, expected, atol=1e-9)
        # 90 度回すと、左は +Y になる（モデルは +X を向く）
        np.testing.assert_allclose(rotate(LEFT, [0, 0, 1], 90.0), [0.0, 1.0, 0.0], atol=1e-15)
        # 前に倒れる・横に動く（どんな回転と移動でも）同じ
        rng = np.random.default_rng(RNG_SEED)
        axis, deg, shift = rng.normal(size=3), 63.0, np.array([0.3, -1.2, 0.05])
        moved = [rotate(p, axis, deg) + shift for p in (hip, knee, ankle)]
        np.testing.assert_allclose(km.knee_bend(*moved, rotate(LEFT, axis, deg)), expected, atol=1e-9)

    def test_measured_in_plane_perpendicular_to_lateral(self):
        # すねが横（+X）に外れても、横に垂直な面（YZ）に映した角度になる：映した向きは後ろ 45 度
        hip, knee = np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.5])
        ankle = knee + [0.2, 0.3, -0.3]
        self.assertAlmostEqual(float(km.knee_bend(hip, knee, ankle, LEFT)), 45.0, places=12)

    def test_wide_stance_is_not_called_reverse(self):
        """太ももが真横に近い（脚を真横に大きく開く）と、位置だけではひざの軸が決まらないので、測らない（灰）。
        ふつうに 90 度曲げた形を、逆曲がり（赤）とまちがえない"""
        bends = []
        for yaw_deg in (-5.0, 5.0):
            t = math.radians(yaw_deg)
            hip = np.array([0.1, 0.0, 1.0])
            knee = hip + 0.45 * np.array([math.cos(t), math.sin(t), 0.0])  # 真横（+X）から yaw_deg 度。正は後ろ（+Y）
            ankle = knee + [0.0, 0.0, -0.45]
            with self.subTest(yaw_deg=yaw_deg):
                self.assertAlmostEqual(float(km.angle_at(hip, knee, ankle)), 90.0, places=9)  # 3次元ではどちらも 90 度
                bend = float(km.knee_bend(hip, knee, ankle, LEFT))
                self.assertTrue(math.isnan(bend))
                bends.append(bend)
        result = km.summarize_reverse_bend(bends)
        self.assertNotEqual(result.signal, Signal.RED)
        self.assertEqual(result.signal, Signal.GRAY)

    def test_abducted_leg_keeps_true_bend(self):
        """脚を横に開いても（ねじらなければ）、ひざの曲がりは本当の値になる。開きすぎは測らない"""
        limit = thresholds.KNEE_AXIS_MIN_ANGLE_DEG
        for abduct in (20.0, 45.0, 90.0 - limit - 1.0):
            for bend in (-30.0, 0.0, 45.0, 90.0):
                hip, knee, ankle = leg(bend)
                # 左脚を外（キャラの左 +X）へ開く：-Y まわりに回す
                posed = [hip + rotate(point - hip, [0, -1, 0], abduct) for point in (hip, knee, ankle)]
                with self.subTest(abduct=abduct, bend=bend):
                    self.assertGreater(posed[1][0], hip[0])  # ひざは外（+X）へ出ている
                    self.assertAlmostEqual(float(km.knee_bend(*posed, LEFT)), bend, places=9)
        hip, knee, ankle = leg(90.0)
        too_wide = [hip + rotate(point - hip, [0, -1, 0], 90.0 - limit + 1.0) for point in (hip, knee, ankle)]
        self.assertTrue(math.isnan(float(km.knee_bend(*too_wide, LEFT))))

    def test_flexed_and_abducted_leg_keeps_sign(self):
        """前に上げて横に開いた脚（ガニ股のしゃがみなど）でも、ふつうの曲がりは正、逆曲がりは負のまま"""
        for flex in (30.0, 60.0, 90.0):
            for abduct in (20.0, 40.0):
                for bend, positive in ((60.0, True), (110.0, True), (-20.0, False)):
                    hip, knee, ankle = leg(bend)
                    # 太ももを前（-Y）へ上げる：+X まわりに回す（ひざは前へ）。そのあと外へ開く
                    flexed = [hip + rotate(point - hip, [1, 0, 0], -flex) for point in (hip, knee, ankle)]
                    posed = [hip + rotate(point - hip, [0, -1, 0], abduct) for point in flexed]
                    with self.subTest(flex=flex, abduct=abduct, bend=bend):
                        self.assertLess(posed[1][1], hip[1])  # ひざは前（-Y）
                        result = float(km.knee_bend(*posed, LEFT))
                        self.assertEqual(result > 0.0, positive, result)

    def test_rotation_method_handles_twist_and_side_splits(self):
        """太ももの骨の回転を使えば、真横に上げてねじった脚でも、正しい曲がりになる"""
        rest_hip, rest_knee, _ = leg(0.0)
        rest_axis = km.knee_rest_axis(rest_hip, rest_knee, LEFT)
        np.testing.assert_allclose(rest_axis, [1.0, 0.0, 0.0], atol=1e-15)
        abduct = quat([0, -1, 0], 90.0)  # 太ももを真横（+X）へ
        twist = quat([1, 0, 0], 90.0)  # 真横に上げたあと、太ももの向き（+X）まわりにねじる
        for rotation in (abduct, quat_mul(twist, abduct)):
            for bend in (-30.0, 0.0, 45.0, 90.0):
                hip, knee, ankle = leg(bend)
                posed = [hip + km.quat_rotate(rotation, point - hip) for point in (hip, knee, ankle)]
                with self.subTest(rotation=rotation.tolist(), bend=bend):
                    np.testing.assert_allclose((posed[1] - hip) / 0.5, [1.0, 0.0, 0.0], atol=1e-12)  # 太ももは真横
                    measured = float(km.knee_bend_from_thigh(*posed, rest_axis=rest_axis, thigh_rotation=rotation))
                    self.assertAlmostEqual(measured, bend, places=9)
                    if bend != 0.0:
                        # 位置だけでは測らない（灰）。逆曲がりとまちがえない
                        self.assertTrue(math.isnan(float(km.knee_bend(*posed, LEFT))))

    def test_rotation_method_per_frame(self):
        bends = [10.0, 45.0, -15.0]
        hip, knee, ankle = legs(bends)
        rotations = np.array([quat([0, -1, 0], 30.0 * i) for i in range(3)])
        posed = [np.array([h + km.quat_rotate(r, x - h) for h, x, r in zip(hip, part, rotations)])
                 for part in (hip, knee, ankle)]
        result = km.knee_bend_from_thigh(*posed, rest_axis=[1.0, 0.0, 0.0], thigh_rotation=rotations)
        np.testing.assert_allclose(result, bends, atol=1e-9)

    def test_knee_rest_axis_and_limits(self):
        with self.assertRaises(ValueError):
            km.knee_rest_axis([0, 0, 1], [0.5, 0, 1], LEFT)  # 太ももが真横
        for bad in (90.0, -1.0, math.nan):
            with self.subTest(min_angle_deg=bad), self.assertRaises(ValueError):
                km.knee_bend(*leg(30.0), LEFT, min_angle_deg=bad)
        # 0 にすると、ほぼ真横でも測る（前の測り方と同じ）
        hip = np.array([0.1, 0.0, 1.0])
        knee = hip + 0.45 * np.array([math.cos(math.radians(30.0)), math.sin(math.radians(30.0)), 0.0])
        self.assertFalse(math.isnan(float(km.knee_bend(hip, knee, knee + [0, 0, -0.45], LEFT, min_angle_deg=0.0))))


    def test_degenerate_gives_nan(self):
        # 太ももが横（lateral の向き）を向く：ひざの軸まわりの向きが決まらない
        hip, knee = np.array([0.0, 0.0, 0.5]), np.array([0.5, 0.0, 0.5])
        self.assertTrue(math.isnan(float(km.knee_bend(hip, knee, [0.5, 0.3, 0.2], LEFT))))
        # ひざと足首が重なる
        self.assertTrue(math.isnan(float(km.knee_bend([0, 0, 1], [0, 0, 0.5], [0, 0, 0.5], LEFT))))
        # lateral が無い
        self.assertTrue(math.isnan(float(km.knee_bend(*leg(30.0), [0, 0, 0]))))

    def test_invalid_shapes(self):
        hip, knee, ankle = legs([0.0, 10.0, 20.0])
        with self.assertRaises(ValueError):
            km.knee_bend(hip, knee, ankle[:2], LEFT)
        with self.assertRaises(ValueError):
            km.knee_bend(hip, knee, ankle, [1.0, 0.0])


class HingeBendTest(unittest.TestCase):
    def test_elbow_example_from_docstring(self):
        # T ポーズの左腕（+X）。前腕を前（-Y）に曲げるのがふつう → 軸は -Z
        shoulder, elbow = np.array([0.2, 0.0, 1.4]), np.array([0.45, 0.0, 1.4])
        forward = elbow + [0.0, -0.25, 0.0]
        backward = elbow + [0.0, 0.25, 0.0]
        self.assertAlmostEqual(float(km.hinge_bend(shoulder, elbow, forward, [0, 0, -1])), 90.0, places=12)
        self.assertAlmostEqual(float(km.hinge_bend(shoulder, elbow, backward, [0, 0, -1])), -90.0, places=12)
        # 右腕（-X）は軸が +Z
        r_shoulder, r_elbow = np.array([-0.2, 0.0, 1.4]), np.array([-0.45, 0.0, 1.4])
        self.assertAlmostEqual(float(km.hinge_bend(r_shoulder, r_elbow, r_elbow + [0, -0.25, 0], [0, 0, 1])), 90.0,
                               places=12)

    def test_same_as_knee_bend_with_lateral_axis(self):
        hip, knee, ankle = legs([-20.0, 0.0, 60.0])
        np.testing.assert_array_equal(km.hinge_bend(hip, knee, ankle, LEFT), km.knee_bend(hip, knee, ankle, LEFT))

    def test_folded_limb_is_plus_180(self):
        self.assertEqual(float(km.hinge_bend([0, 0, 1], [0, 0, 0.5], [0, 0, 1], LEFT)), 180.0)


class ReverseBendFramesTest(unittest.TestCase):
    def test_tolerance_boundary_and_nan(self):
        tol = thresholds.REVERSE_BEND_TOLERANCE_DEG
        bends = [0.0, -tol, -tol - 1e-9, -10.0, math.nan, 5.0, -tol + 1e-9]
        np.testing.assert_array_equal(km.reverse_bend_frames(bends), [False, False, True, True, False, False, False])

    def test_custom_tolerance(self):
        np.testing.assert_array_equal(km.reverse_bend_frames([-1e-9, 0.0, -0.0, 3.0], tolerance_deg=0.0),
                                      [True, False, False, False])
        np.testing.assert_array_equal(km.reverse_bend_frames([-9.0, -11.0], tolerance_deg=10),
                                      [False, True])

    def test_empty(self):
        result = km.reverse_bend_frames([])
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.dtype, bool)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            km.reverse_bend_frames([[1.0, 2.0]])
        with self.assertRaises(ValueError):
            km.reverse_bend_frames([True, False])
        with self.assertRaises(ValueError):
            km.reverse_bend_frames([-math.inf])
        with self.assertRaises(ValueError):
            km.reverse_bend_frames([0.0], tolerance_deg=-1.0)
        with self.assertRaises(ValueError):
            km.reverse_bend_frames([0.0], tolerance_deg=True)
        with self.assertRaises(ValueError):
            km.reverse_bend_frames([0.0], tolerance_deg=math.nan)


# MARK: - 足の床への沈み

class SinkDepthsTest(unittest.TestCase):
    def test_depths(self):
        points = [[0.0, 0.0, 0.01], [1.0, 2.0, 0.0], [0.0, 0.0, -0.003], [0.0, 0.0, math.nan], [5.0, 5.0, -1.5]]
        result = km.sink_depths(points)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.003, math.nan, 1.5], atol=1e-15)
        self.assertFalse(np.signbit(result[:2]).any())  # -0.0 にしない

    def test_floor_height(self):
        result = km.sink_depths([[0.0, 0.0, 0.05], [0.0, 0.0, 0.2]], floor_z=0.1)
        np.testing.assert_allclose(result, [0.05, 0.0], atol=1e-15)

    def test_shapes(self):
        points = np.zeros((4, 2, 3))
        points[2, 1, 2] = -0.004
        result = km.sink_depths(points)
        self.assertEqual(result.shape, (4, 2))
        self.assertAlmostEqual(result[2, 1], 0.004, places=15)
        self.assertEqual(result.sum(), result[2, 1])
        self.assertEqual(km.sink_depths(np.zeros((0, 3))).shape, (0,))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            km.sink_depths([[0.0, 0.0]])
        with self.assertRaises(ValueError):
            km.sink_depths([[0.0, 0.0, 0.0]], floor_z=math.nan)
        with self.assertRaises(ValueError):
            km.sink_depths([[0.0, 0.0, 0.0]], floor_z=False)


# MARK: - 足の接地と足すべり

def step_cycle() -> np.ndarray:
    """30 フレーム：0〜9 は床に着く、10〜19 は上げて前（+X）へ、20〜29 は新しい場所に着く"""
    foot = np.zeros((30, 3))
    for i in range(10, 20):
        foot[i] = [0.03 * (i - 9), 0.0, 0.1]
    foot[20:] = [0.33, 0.0, 0.0]
    return foot


class TrackSpeedsTest(unittest.TestCase):
    def test_central_differences_times_fps(self):
        track = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.3, 0.0, 0.0], [0.3, 0.4, 0.0]])
        # 両端は片側の差、中は (次 − 前) / 2。fps を掛けて毎秒にする
        np.testing.assert_allclose(km.track_speeds(track, fps=10), [1.0, 1.5, np.hypot(1.0, 2.0), 4.0], atol=1e-12)

    def test_short_tracks(self):
        np.testing.assert_array_equal(km.track_speeds([[1.0, 2.0, 3.0]], fps=30), [0.0])
        self.assertEqual(km.track_speeds(np.zeros((0, 3)), fps=30).shape, (0,))

    def test_vertical_motion_counts(self):
        # 速さは3次元。上下（Z）だけの動きも数える（水平 XY だけで測ると 0 になってしまう）
        rising = np.stack([np.zeros(4), np.zeros(4), 0.005 * np.arange(4)], axis=1)  # 1フレームに 5mm 上がる
        np.testing.assert_allclose(km.track_speeds(rising, fps=30), [0.15] * 4, atol=1e-12)
        # 高さは 15mm まで（FOOT_CONTACT_HEIGHT 以下）だが、0.15 m/s で上がっているので着いていない
        self.assertLessEqual(rising[:, 2].max(), thresholds.FOOT_CONTACT_HEIGHT)
        self.assertGreater(0.15, thresholds.FOOT_CONTACT_SPEED)
        self.assertFalse(km.foot_contacts(rising, fps=30).any())


class FootContactsTest(unittest.TestCase):
    def test_step_cycle_exact_frames(self):
        contacts = km.foot_contacts(step_cycle(), fps=30)
        # 9 と 20 は床の上だが、となりのフレームの動きで速さが出る（中央の差）
        expected = np.zeros(30, dtype=bool)
        expected[0:9] = True
        expected[21:30] = True
        np.testing.assert_array_equal(contacts, expected)
        self.assertEqual(km.spans(contacts), [(0, 8), (21, 29)])

    def test_speed_scales_with_fps(self):
        # とても遅い再生なら、速さは小さくなり、9 と 20 も着いている
        contacts = km.foot_contacts(step_cycle(), fps=0.1)
        self.assertEqual(km.spans(contacts), [(0, 9), (20, 29)])

    def test_both_height_and_speed_are_needed(self):
        hovering = np.tile([0.0, 0.0, 0.03], (5, 1))  # 止まっているが 3cm 上
        self.assertFalse(km.foot_contacts(hovering, fps=30).any())
        sliding = np.stack([0.1 * np.arange(5), np.zeros(5), np.zeros(5)], axis=1)  # 床の上だが 3 m/s
        self.assertFalse(km.foot_contacts(sliding, fps=30).any())
        below = np.tile([0.0, 0.0, -0.05], (5, 1))  # 床より下で止まっている
        self.assertTrue(km.foot_contacts(below, fps=30).all())

    def test_boundaries_are_inclusive(self):
        at_height = np.tile([0.0, 0.0, thresholds.FOOT_CONTACT_HEIGHT], (3, 1))
        self.assertTrue(km.foot_contacts(at_height, fps=30).all())
        above = np.tile([0.0, 0.0, thresholds.FOOT_CONTACT_HEIGHT * 1.001], (3, 1))
        self.assertFalse(km.foot_contacts(above, fps=30).any())
        # 1フレームに 0.125m、fps 4 なら、ちょうど 0.5 m/s
        moving = np.stack([0.125 * np.arange(4), np.zeros(4), np.zeros(4)], axis=1)
        self.assertTrue(km.foot_contacts(moving, fps=4, speed=0.5).all())
        self.assertFalse(km.foot_contacts(moving, fps=4, speed=0.4999).any())
        # 基準が 0 でも、床の上で止まっていれば着いている（両端を含む）
        self.assertTrue(km.foot_contacts(np.zeros((3, 3)), fps=30, height=0.0, speed=0.0).all())

    def test_floor_height(self):
        raised = step_cycle() + [0.0, 0.0, 1.0]
        np.testing.assert_array_equal(km.foot_contacts(raised, fps=30, floor_z=1.0),
                                      km.foot_contacts(step_cycle(), fps=30))
        self.assertFalse(km.foot_contacts(raised, fps=30).any())

    def test_single_and_empty(self):
        np.testing.assert_array_equal(km.foot_contacts([[0.0, 0.0, 0.0]], fps=30), [True])
        np.testing.assert_array_equal(km.foot_contacts([[0.0, 0.0, 0.5]], fps=30), [False])
        result = km.foot_contacts(np.zeros((0, 3)), fps=30)
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.dtype, bool)

    def test_invalid(self):
        foot = step_cycle()
        for bad_fps in (0, -30, True, math.nan, math.inf, "30"):
            with self.subTest(fps=bad_fps), self.assertRaises(ValueError):
                km.foot_contacts(foot, fps=bad_fps)
        bad = foot.copy()
        bad[3, 2] = math.nan
        with self.assertRaises(ValueError):
            km.foot_contacts(bad, fps=30)
        with self.assertRaises(ValueError):
            km.foot_contacts(foot[:, :2], fps=30)
        with self.assertRaises(ValueError):
            km.foot_contacts(foot[:, None, :], fps=30)  # (F, J, 3) は1つの点ではない
        with self.assertRaises(ValueError):
            km.foot_contacts(foot, fps=30, height=-0.01)


class SpansTest(unittest.TestCase):
    def test_cases(self):
        cases = [
            ([], []),
            ([False], []),
            ([True], [(0, 0)]),
            ([False] * 4, []),
            ([True] * 5, [(0, 4)]),
            ([True, False, True, False, True], [(0, 0), (2, 2), (4, 4)]),
            ([True, True, False, False, True], [(0, 1), (4, 4)]),
            ([False, True, True, True, False], [(1, 3)]),
        ]
        for mask, expected in cases:
            with self.subTest(mask=mask):
                result = km.spans(np.array(mask, dtype=bool))
                self.assertEqual(result, expected)
                self.assertTrue(all(type(v) is int for pair in result for v in pair))

    def test_round_trip_random(self):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(50):
            mask = rng.random(int(rng.integers(0, 40))) < 0.5
            rebuilt = np.zeros(len(mask), dtype=bool)
            previous_end = -2
            for start, end in km.spans(mask):
                self.assertLessEqual(start, end)
                self.assertGreater(start, previous_end + 1)  # 区間は離れていて、前から順
                rebuilt[start:end + 1] = True
                previous_end = end
            np.testing.assert_array_equal(rebuilt, mask)

    def test_rejects_integers(self):
        with self.assertRaises(ValueError):
            km.spans([1, 0, 1])
        with self.assertRaises(ValueError):
            km.spans(np.zeros((2, 2), dtype=bool))


class SlideDistancesTest(unittest.TestCase):
    def test_planted_foot_drifting_one_centimetre(self):
        foot = np.zeros((10, 3))
        foot[:, 0] = np.linspace(0.0, 0.01, 10)
        result = km.slide_distances(foot, np.ones(10, dtype=bool))
        self.assertEqual(len(result), 1)
        start, end, distance = result[0]
        self.assertEqual((start, end), (0, 9))
        self.assertAlmostEqual(distance, 0.01, places=15)
        # 30fps では 1cm / 0.3秒 ≈ 0.033 m/s なので、foot_contacts でも着いている
        contacts = km.foot_contacts(foot, fps=30)
        self.assertTrue(contacts.all())
        self.assertEqual(km.summarize_slide(km.slide_distances(foot, contacts)).signal, Signal.RED)

    def test_each_span_measures_from_its_own_start(self):
        foot = np.zeros((10, 3))
        foot[0:5, 0] = [0.0, 0.001, 0.002, 0.003, 0.004]
        foot[5:7] = [[0.5, 0.0, 0.2], [0.9, 0.0, 0.2]]
        foot[7:10, 0] = [1.0, 1.002, 1.001]
        contacts = np.array([True] * 5 + [False] * 2 + [True] * 3)
        result = km.slide_distances(foot, contacts)
        self.assertEqual([(s, e) for s, e, _ in result], [(0, 4), (7, 9)])
        np.testing.assert_allclose([d for _, _, d in result], [0.004, 0.002], atol=1e-12)

    def test_maximum_not_final_and_horizontal_only(self):
        out_and_back = np.array([[0.0, 0.0, 0.0], [0.003, 0.004, 0.0], [0.0, 0.0, 0.0]])
        self.assertAlmostEqual(km.slide_distances(out_and_back, np.ones(3, dtype=bool))[0][2], 0.005, places=15)
        lifting = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.05], [0.0, 0.0, 0.01]])
        self.assertEqual(km.slide_distances(lifting, np.ones(3, dtype=bool))[0][2], 0.0)

    def test_single_frame_and_no_contact(self):
        foot = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        self.assertEqual(km.slide_distances(foot, np.array([False, True, False])), [(1, 1, 0.0)])
        self.assertEqual(km.slide_distances(foot, np.zeros(3, dtype=bool)), [])
        self.assertEqual(km.slide_distances(np.zeros((0, 3)), np.zeros(0, dtype=bool)), [])

    def test_invalid(self):
        foot = np.zeros((4, 3))
        with self.assertRaises(ValueError):
            km.slide_distances(foot, np.ones(3, dtype=bool))
        with self.assertRaises(ValueError):
            km.slide_distances(foot, [1, 1, 1, 1])
        bad = foot.copy()
        bad[1, 0] = math.nan
        with self.assertRaises(ValueError):
            km.slide_distances(bad, np.ones(4, dtype=bool))


# MARK: - 足が届かないフレーム

class UnreachableFramesTest(unittest.TestCase):
    def test_boundary_with_exact_numbers(self):
        root = np.zeros((4, 3))
        target = np.array([[1.0, 0.0, 0.0], [1.25, 0.0, 0.0], [1.25 + 1e-12, 0.0, 0.0], [2.0, 0.0, 0.0]])
        np.testing.assert_array_equal(km.unreachable_frames(root, target, 1.0, tolerance=0.25),
                                      [False, False, True, True])
        np.testing.assert_allclose(km.reach_ratios(root, target, 1.0), [1.0, 1.25, 1.25, 2.0])

    def test_default_tolerance(self):
        length = 0.8
        limit = length * (1.0 + thresholds.IK_REACH_TOLERANCE)
        root = np.tile([0.1, -0.2, 0.9], (2, 1))
        direction = np.array([0.0, 0.6, -0.8])  # 長さ 1
        target = root + np.outer([limit * (1 - 1e-9), limit * (1 + 1e-9)], direction)
        np.testing.assert_array_equal(km.unreachable_frames(root, target, length), [False, True])

    def test_distance_is_three_dimensional_and_length_per_frame(self):
        root = np.zeros((3, 3))
        target = np.array([[0.0, 0.0, 1.1], [0.6, 0.8, 0.0], [0.0, 0.0, -0.5]])
        np.testing.assert_array_equal(km.unreachable_frames(root, target, 1.0, tolerance=0.0), [True, False, False])
        np.testing.assert_array_equal(km.unreachable_frames(root, target, [1.2, 0.9, 0.4], tolerance=0.0),
                                      [False, True, True])

    def test_invalid(self):
        root, target = np.zeros((2, 3)), np.ones((2, 3))
        for bad in (0.0, -1.0, True, math.nan, [1.0, 0.0], [1.0, 1.0, 1.0], [[1.0, 1.0]]):
            with self.subTest(chain_length=bad), self.assertRaises(ValueError):
                km.unreachable_frames(root, target, bad)
        with self.assertRaises(ValueError):
            km.unreachable_frames(root, np.ones((3, 3)), 1.0)
        with self.assertRaises(ValueError):
            km.unreachable_frames(root, target, 1.0, tolerance=-0.1)
        bad_target = target.copy()
        bad_target[0, 0] = math.nan
        with self.assertRaises(ValueError):
            km.unreachable_frames(root, bad_target, 1.0)
        # 0次元の numpy の数はよい
        np.testing.assert_array_equal(km.unreachable_frames(root, target, np.float64(2.0)), [False, False])
        np.testing.assert_array_equal(km.unreachable_frames(root, target, np.array(1.0)), [True, True])


# MARK: - 回転（クォータニオン）と骨の跳び

class QuaternionTest(unittest.TestCase):
    def test_normalize(self):
        np.testing.assert_allclose(km.quat_normalize([2.0, 0.0, 0.0, 0.0]), [1.0, 0.0, 0.0, 0.0])
        q = km.quat_normalize([[0.0, 3.0, 0.0, 4.0], [-1.0, -1.0, -1.0, -1.0]])
        np.testing.assert_allclose(q, [[0.0, 0.6, 0.0, 0.8], [-0.5, -0.5, -0.5, -0.5]])  # 符号は変えない
        with self.assertRaises(ValueError):
            km.quat_normalize([0.0, 0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            km.quat_normalize([1.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            km.quat_normalize([1.0, math.nan, 0.0, 0.0])

    def test_q_and_minus_q_are_the_same(self):
        rng = np.random.default_rng(RNG_SEED)
        q = random_unit_quaternions(rng, 20)
        np.testing.assert_array_equal(km.quat_angle_deg(q, -q), np.zeros(20))
        self.assertEqual(float(km.quat_angle_deg([1, 0, 0, 0], [-1, 0, 0, 0])), 0.0)

    def test_known_angles(self):
        identity = [1.0, 0.0, 0.0, 0.0]
        self.assertAlmostEqual(float(km.quat_angle_deg(identity, quat([0, 0, 1], 90.0))), 90.0, places=12)
        self.assertAlmostEqual(float(km.quat_angle_deg(identity, [0.0, 0.0, 0.0, 1.0])), 180.0, places=12)
        # 270 度回すのは、逆に 90 度回すのと同じ向き
        self.assertAlmostEqual(float(km.quat_angle_deg(identity, quat([1, 2, 3], 270.0))), 90.0, places=12)
        rng = np.random.default_rng(RNG_SEED)
        for angle in (1.0, 45.0, 90.0, 120.0, 179.0):
            q1 = random_unit_quaternions(rng, 1)[0]
            r = quat(rng.normal(size=3), angle)
            with self.subTest(angle=angle):
                self.assertAlmostEqual(float(km.quat_angle_deg(q1, quat_mul(q1, r))), angle, places=9)
                self.assertAlmostEqual(float(km.quat_angle_deg(q1, quat_mul(r, q1))), angle, places=9)

    def test_agrees_with_rotation_matrices(self):
        rng = np.random.default_rng(RNG_SEED)
        q1, q2 = random_unit_quaternions(rng, 200), random_unit_quaternions(rng, 200)
        result = km.quat_angle_deg(q1, q2)
        for a, b, angle in zip(q1, q2, result):
            cosine = (np.trace(quat_matrix(a).T @ quat_matrix(b)) - 1.0) / 2.0
            expected = math.degrees(math.acos(min(1.0, max(-1.0, cosine))))
            if 1.0 < expected < 179.0:  # acos が正確な所だけで比べる
                self.assertAlmostEqual(angle, expected, places=6)
        self.assertTrue(np.all((result >= 0.0) & (result <= 180.0)))
        np.testing.assert_allclose(km.quat_angle_deg(q2, q1), result, atol=1e-12)  # 入れ替えても同じ

    def test_tiny_angles_are_accurate(self):
        # acos を使うと、1e-4 度では約 1e-4 の割合でずれる
        tiny = 1e-4
        rng = np.random.default_rng(RNG_SEED)
        self.assertLess(abs(float(km.quat_angle_deg([1, 0, 0, 0], quat([0, 1, 0], tiny))) - tiny), tiny * 1e-9)
        for _ in range(20):
            q1 = random_unit_quaternions(rng, 1)[0]
            q2 = quat_mul(q1, quat(rng.normal(size=3), tiny))
            self.assertLess(abs(float(km.quat_angle_deg(q1, q2)) - tiny), tiny * 1e-7)

    def test_not_normalized_inputs(self):
        q1 = quat([1, 0, 0], 10.0)
        q2 = quat([1, 0, 0], 50.0)
        self.assertAlmostEqual(float(km.quat_angle_deg(3.0 * q1, 0.25 * q2)), 40.0, places=12)
        self.assertAlmostEqual(float(km.quat_angle_deg(-7.0 * q1, 0.25 * q2)), 40.0, places=12)

    def test_broadcast_and_invalid(self):
        targets = np.array([quat([0, 0, 1], a) for a in (0.0, 30.0, 60.0)])
        np.testing.assert_allclose(km.quat_angle_deg([1, 0, 0, 0], targets), [0.0, 30.0, 60.0], atol=1e-12)
        with self.assertRaises(ValueError):
            km.quat_angle_deg([0, 0, 0, 0], [1, 0, 0, 0])
        with self.assertRaises(ValueError):
            km.quat_angle_deg(np.ones((2, 4)), np.ones((3, 4)))
        with self.assertRaises(ValueError):
            km.quat_angle_deg([True, False, False, False], [1, 0, 0, 0])
        with self.assertRaises(ValueError):
            km.quat_angle_deg([1, 0, 0, math.inf], [1, 0, 0, 0])


def jump_sequence() -> np.ndarray:
    """5 フレーム、3 本の骨。骨 1 だけが、フレーム 2 と 3 の間で 40 度跳ぶ"""
    rotations = np.zeros((5, 3, 4))
    # 骨 0：動かない。フレーム 3 だけ符号が逆（同じ向き）
    rotations[:, 0] = [1.0, 0.0, 0.0, 0.0]
    rotations[3, 0] = [-1.0, 0.0, 0.0, 0.0]
    # 骨 1：Z まわりに 0, 5, 10, 50, 55 度
    for i, angle in enumerate((0.0, 5.0, 10.0, 50.0, 55.0)):
        rotations[i, 1] = quat([0, 0, 1], angle)
    # 骨 2：X まわりに 29.9 度ずつ（基準の 30 度より少し小さい）
    for i in range(5):
        rotations[i, 2] = quat([1, 0, 0], 29.9 * i)
    return rotations


class BoneJumpTest(unittest.TestCase):
    def test_one_jump(self):
        mask = km.bone_jump_frames(jump_sequence())
        expected = np.zeros((4, 3), dtype=bool)
        expected[2, 1] = True
        np.testing.assert_array_equal(mask, expected)
        steps = km.step_angles_deg(jump_sequence())
        np.testing.assert_allclose(steps[:, 1], [5.0, 5.0, 40.0, 5.0], atol=1e-9)
        np.testing.assert_allclose(steps[:, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(steps[:, 2], 29.9, atol=1e-9)

    def test_threshold_and_large_rotations(self):
        rotations = np.array([[quat([0, 1, 0], 0.0)], [quat([0, 1, 0], 30.1)], [quat([0, 1, 0], 220.1)]])
        # 30.1 は跳ぶ。190 度回すのは、逆に 170 度回すのと同じ
        np.testing.assert_array_equal(km.bone_jump_frames(rotations)[:, 0], [True, True])
        np.testing.assert_allclose(km.step_angles_deg(rotations)[:, 0], [30.1, 170.0], atol=1e-9)
        np.testing.assert_array_equal(km.bone_jump_frames(rotations, threshold_deg=175.0)[:, 0], [False, False])
        np.testing.assert_array_equal(km.bone_jump_frames(rotations, threshold_deg=165)[:, 0], [False, True])

    def test_threshold_is_strict(self):
        # 「超えたら跳ぶ」：ちょうど基準の角度は跳びにしない。0 度と 180 度は double でちょうど出る
        flip = np.array([[[1.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 1.0]]])  # Z まわりにちょうど 180 度
        self.assertEqual(float(km.step_angles_deg(flip)[0, 0]), 180.0)
        np.testing.assert_array_equal(km.bone_jump_frames(flip, threshold_deg=180.0), [[False]])
        np.testing.assert_array_equal(km.bone_jump_frames(flip, threshold_deg=179.999), [[True]])
        still = np.tile([1.0, 0.0, 0.0, 0.0], (3, 2, 1))  # 動かない骨は、基準が 0 でも跳ばない
        np.testing.assert_array_equal(km.bone_jump_frames(still, threshold_deg=0.0), np.zeros((2, 2), dtype=bool))

    def test_not_normalized_and_chunks(self):
        rng = np.random.default_rng(RNG_SEED)
        rotations = random_unit_quaternions(rng, 23 * 5).reshape(23, 5, 4) * rng.uniform(0.5, 3.0, size=(23, 5, 1))
        expected = km.quat_angle_deg(rotations[:-1], rotations[1:])
        for chunk in (1, 4, 7, 1000):
            with self.subTest(chunk=chunk):
                np.testing.assert_allclose(km.step_angles_deg(rotations, chunk_size=chunk), expected, atol=1e-12)
                np.testing.assert_array_equal(km.bone_jump_frames(rotations, threshold_deg=90.0, chunk_size=chunk),
                                              expected > 90.0)

    def test_float32_input(self):
        # Blender から取り出す並びは float32 のことが多い
        rotations = jump_sequence().astype(np.float32)
        np.testing.assert_array_equal(km.bone_jump_frames(rotations), km.bone_jump_frames(jump_sequence()))
        np.testing.assert_allclose(km.step_angles_deg(rotations, chunk_size=2)[:, 1], [5.0, 5.0, 40.0, 5.0], atol=1e-4)

    def test_short_sequences(self):
        self.assertEqual(km.bone_jump_frames(np.zeros((0, 3, 4))).shape, (0, 3))
        self.assertEqual(km.bone_jump_frames(np.tile([1.0, 0, 0, 0], (1, 3, 1))).shape, (0, 3))
        self.assertEqual(km.step_angles_deg(np.tile([1.0, 0, 0, 0], (1, 3, 1))).shape, (0, 3))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            km.bone_jump_frames(np.tile([1.0, 0, 0, 0], (3, 1)))  # (F, 4) は骨の軸が無い
        with self.assertRaises(ValueError):
            km.bone_jump_frames(np.zeros((1, 2, 4)))  # 長さ 0 のクォータニオンは、フレームが1つでも見つける
        broken = jump_sequence()
        broken[4, 2, 1] = math.nan  # NaN を黙って「跳んでいない」にしない
        for bad in (broken, np.where(np.isnan(broken), math.inf, broken)):
            with self.assertRaises(ValueError):
                km.bone_jump_frames(bad)
            with self.assertRaises(ValueError):
                km.step_angles_deg(bad, chunk_size=3)
        with self.assertRaises(ValueError):
            km.bone_jump_frames(jump_sequence(), threshold_deg=True)
        with self.assertRaises(ValueError):
            km.bone_jump_frames(jump_sequence(), chunk_size=0)
        with self.assertRaises(ValueError):
            km.bone_jump_frames(jump_sequence(), chunk_size=True)


# MARK: - まとめ（信号）

class SummarizeReverseBendTest(unittest.TestCase):
    def test_no_reverse_is_green_with_zero(self):
        m = km.summarize_reverse_bend([10.0, 20.0, 30.0])
        self.assertEqual((m.name, m.unit, m.signal, m.value), ("reverse_bend", "deg", Signal.GREEN, 0.0))
        self.assertEqual(m.details["reverse_spans"], [])

    def test_within_tolerance_is_green(self):
        m = km.summarize_reverse_bend([-1.5, 10.0, -thresholds.REVERSE_BEND_TOLERANCE_DEG])
        self.assertEqual(m.signal, Signal.GREEN)
        self.assertEqual(m.value, thresholds.REVERSE_BEND_TOLERANCE_DEG)  # 正の数で返す
        self.assertEqual(m.details["worst_frame"], 2)

    def test_reverse_is_red_with_spans(self):
        m = km.summarize_reverse_bend([-1.0, -5.0, -6.0, 3.0, -10.0, -10.0])
        self.assertEqual(m.signal, Signal.RED)
        self.assertEqual(m.value, 10.0)
        self.assertEqual(m.details["reverse_spans"], [(1, 2), (4, 5)])
        self.assertEqual(m.details["reverse_count"], 4)
        self.assertEqual(m.details["worst_frame"], 4)  # 同じなら前のフレーム
        self.assertEqual(m.details["min_bend_deg"], -10.0)

    def test_custom_tolerance_matches_frames(self):
        bends = [-3.0, -4.0]
        m = km.summarize_reverse_bend(bends, tolerance_deg=4.0)
        self.assertEqual(m.signal, Signal.GREEN)
        self.assertEqual(m.details["reverse_spans"], km.spans(km.reverse_bend_frames(bends, tolerance_deg=4.0)))

    def test_unmeasured(self):
        for bends in ([], [math.nan, math.nan]):
            with self.subTest(bends=bends):
                m = km.summarize_reverse_bend(bends)
                self.assertEqual(m.signal, Signal.GRAY)
                self.assertIsNone(m.value)
                self.assertTrue(m.reason)
        partial = km.summarize_reverse_bend([10.0, math.nan, math.nan, 20.0])
        self.assertEqual(partial.signal, Signal.GRAY)  # 測れなかったフレームがあれば緑にしない
        self.assertEqual(partial.value, 0.0)
        self.assertEqual(partial.details["unmeasured_spans"], [(1, 2)])
        with_red = km.summarize_reverse_bend([math.nan, -30.0])
        self.assertEqual(with_red.signal, Signal.RED)
        self.assertEqual(with_red.details["worst_frame"], 1)

    def test_end_to_end_with_knee_bend(self):
        hip, knee, ankle = legs([0.0, 20.0, 40.0, -8.0, -8.0, 0.0])
        m = km.summarize_reverse_bend(km.knee_bend(hip, knee, ankle, LEFT))
        self.assertEqual(m.signal, Signal.RED)
        self.assertAlmostEqual(m.value, 8.0, places=9)
        self.assertEqual(m.details["reverse_spans"], [(3, 4)])


class SummarizeSinkTest(unittest.TestCase):
    def test_red_beyond_tolerance(self):
        m = km.summarize_sink([0.0, 0.001, 0.0025, 0.0])
        self.assertEqual((m.name, m.unit, m.signal), ("floor_sink", "m", Signal.RED))
        self.assertEqual(m.value, 0.0025)
        self.assertEqual(m.details["sink_spans"], [(2, 2)])
        self.assertEqual(m.details["worst_frame"], 2)
        self.assertIsNone(m.details["worst_point"])

    def test_exactly_tolerance_and_zero_are_green(self):
        exact = km.summarize_sink([0.0, thresholds.FLOOR_SINK_TOLERANCE])
        self.assertEqual(exact.signal, Signal.GREEN)
        # 一覧も、信号と同じ判定（ちょうどは沈みに数えない）
        self.assertEqual((exact.details["sink_count"], exact.details["sink_spans"]), (0, []))
        m = km.summarize_sink(np.zeros(5))
        self.assertEqual((m.signal, m.value), (Signal.GREEN, 0.0))

    def test_several_points(self):
        depths = np.zeros((6, 2))
        depths[1, 0] = 0.003
        depths[2, 0] = 0.004
        depths[4, 1] = 0.005
        m = km.summarize_sink(depths)
        self.assertEqual(m.signal, Signal.RED)
        self.assertEqual((m.value, m.details["worst_frame"], m.details["worst_point"]), (0.005, 4, 1))
        self.assertEqual(m.details["sink_spans"], [(1, 2), (4, 4)])

    def test_unmeasured(self):
        for depths in ([], np.zeros((3, 0)), [math.nan, math.nan]):
            with self.subTest(depths=depths):
                m = km.summarize_sink(depths)
                self.assertEqual(m.signal, Signal.GRAY)
                self.assertIsNone(m.value)
        partial = km.summarize_sink([0.0, math.nan, 0.001])
        self.assertEqual((partial.signal, partial.value), (Signal.GRAY, 0.001))
        self.assertEqual(km.summarize_sink([math.nan, 0.01]).signal, Signal.RED)

    def test_end_to_end_and_invalid(self):
        foot = np.zeros((5, 3))
        foot[3, 2] = -0.006
        m = km.summarize_sink(km.sink_depths(foot))
        self.assertEqual(m.signal, Signal.RED)
        self.assertAlmostEqual(m.value, 0.006, places=15)
        with self.assertRaises(ValueError):
            km.summarize_sink([-0.001])
        with self.assertRaises(ValueError):
            km.summarize_sink([math.inf])
        with self.assertRaises(ValueError):
            km.summarize_sink(np.zeros((2, 2, 2)))


class FloorGlideTest(unittest.TestCase):
    """床の上のすべり：床の高さのまま、上下に動かずに、横へ速く動いた距離"""

    FPS = 30.0

    def track(self, xs, zs=None):
        xs = np.asarray(xs, dtype=np.float64)
        zs = np.zeros_like(xs) if zs is None else np.asarray(zs, dtype=np.float64)
        return np.stack([xs, np.zeros_like(xs), zs], axis=1)

    def test_fast_glide_on_the_floor_is_counted(self):
        step = thresholds.FOOT_CONTACT_SPEED * 4.0 / self.FPS  # 速さの基準の4倍
        xs = np.concatenate([np.zeros(5), step * np.arange(1, 11), np.full(5, step * 10)])
        glides = km.floor_glides(self.track(xs), fps=self.FPS)
        self.assertEqual(len(glides), 1)
        start, end, distance = glides[0]
        self.assertEqual((start, end), (0, len(xs) - 1))
        self.assertAlmostEqual(distance, step * 10)
        self.assertEqual(km.summarize_glide(glides).signal, Signal.RED)

    def test_slow_slide_is_left_to_foot_slide(self):
        """ゆっくりのすべりは foot_slide が数えるので、ここでは 0（同じすべりを2回数えない）"""
        step = thresholds.FOOT_CONTACT_SPEED * 0.5 / self.FPS
        glides = km.floor_glides(self.track(step * np.arange(60)), fps=self.FPS)
        self.assertEqual([round(d, 12) for _, _, d in glides], [0.0])
        self.assertEqual(km.summarize_glide(glides).signal, Signal.GREEN)

    def test_lifting_the_foot_is_not_a_glide(self):
        """足を上げ下ろしするフレームは上下に動くので区間に入らない。空中の速い動きも数えない"""
        frames = 40
        xs = np.linspace(0.0, 0.5, frames)
        zs = np.concatenate([np.zeros(10), np.linspace(0.0, 0.1, 10), np.linspace(0.1, 0.0, 10), np.zeros(10)])
        xs[:10] = 0.0
        xs[30:] = xs[29]
        glides = km.floor_glides(self.track(xs, zs), fps=self.FPS)
        self.assertTrue(glides)
        self.assertLessEqual(max(d for _, _, d in glides), thresholds.FOOT_SLIDE_TOLERANCE)

    def test_short_runs_and_no_floor(self):
        self.assertEqual(km.floor_glides(self.track([0.0, 1.0], [0.5, 0.5]), fps=self.FPS), [])
        self.assertEqual(km.summarize_glide([]).signal, Signal.GRAY)
        self.assertEqual(km.floor_glides(self.track([0.0]), fps=self.FPS), [])
        two = km.floor_glides(self.track([0.0, 0.0]), fps=self.FPS, min_frames=3)
        self.assertEqual(two, [])

    def test_bad_input(self):
        for bad in (0, True, 1.5, -1):
            with self.subTest(min_frames=bad), self.assertRaises(ValueError):
                km.floor_glides(self.track([0.0, 0.0, 0.0]), fps=self.FPS, min_frames=bad)
        with self.assertRaises(ValueError):
            km.floor_glides(self.track([0.0, 0.0]), fps=0.0)


class SummarizeSlideTest(unittest.TestCase):
    def test_red_and_green(self):
        m = km.summarize_slide([(0, 9, 0.001), (20, 29, 0.006)])
        self.assertEqual((m.name, m.unit, m.signal, m.value), ("foot_slide", "m", Signal.RED, 0.006))
        self.assertEqual(m.details["slide_spans"], [(20, 29, 0.006)])
        self.assertEqual(m.details["worst_span"], (20, 29, 0.006))
        self.assertEqual(m.details["contact_frame_count"], 20)
        exact = km.summarize_slide([(0, 3, thresholds.FOOT_SLIDE_TOLERANCE)])
        self.assertEqual(exact.signal, Signal.GREEN)
        self.assertEqual(exact.details["slide_spans"], [])  # 一覧も、信号と同じ判定（ちょうどは数えない）

    def test_unmeasured(self):
        m = km.summarize_slide([])
        self.assertEqual(m.signal, Signal.GRAY)
        self.assertIsNone(m.value)
        self.assertTrue(m.reason)
        self.assertEqual(km.summarize_slide([(0, 1, math.nan)]).signal, Signal.GRAY)
        self.assertEqual(km.summarize_slide([(0, 1, math.nan), (3, 4, 0.001)]).signal, Signal.GRAY)
        self.assertEqual(km.summarize_slide([(0, 1, math.nan), (3, 4, 0.1)]).signal, Signal.RED)

    def test_invalid(self):
        for bad in ([(0, 1, -0.001)], [(3, 1, 0.0)], [(True, 1, 0.0)], [(0, 1)], [(0, 1.5, 0.0)], [(0, 1, math.inf)],
                    [5]):
            with self.subTest(slides=bad), self.assertRaises(ValueError):
                km.summarize_slide(bad)

    def test_end_to_end_step_cycle(self):
        foot = step_cycle()
        foot[21:, 1] = np.linspace(0.0, 0.002, 9)  # 2 回目の着地で 2mm 滑る
        slides = km.slide_distances(foot, km.foot_contacts(foot, fps=30))
        self.assertEqual([(s, e) for s, e, _ in slides], [(0, 8), (21, 29)])
        m = km.summarize_slide(slides)
        self.assertEqual(m.signal, Signal.GREEN)
        self.assertAlmostEqual(m.value, 0.002, places=15)


class SummarizeUnreachableTest(unittest.TestCase):
    def test_every_span_is_listed(self):
        mask = np.zeros(100, dtype=bool)
        mask[::3] = True  # 34 個の区間（省かずに全部）
        m = km.summarize_unreachable(mask)
        self.assertEqual((m.name, m.unit, m.signal, m.value), ("unreachable_frames", "count", Signal.YELLOW, 34.0))
        self.assertEqual(m.details["unreachable_spans"], [(i, i) for i in range(0, 100, 3)])

    def test_green_gray_and_invalid(self):
        m = km.summarize_unreachable(np.zeros(4, dtype=bool))
        self.assertEqual((m.signal, m.value), (Signal.GREEN, 0.0))
        self.assertEqual(km.summarize_unreachable([]).signal, Signal.GRAY)
        with self.assertRaises(ValueError):
            km.summarize_unreachable([0, 1, 1])

    def test_end_to_end(self):
        root = np.zeros((5, 3))
        target = np.array([[0, 0, -0.5], [0, 0, -0.9], [0, 0, -0.95], [0, 0, -0.7], [0, 0, -1.2]], dtype=float)
        m = km.summarize_unreachable(km.unreachable_frames(root, target, 0.85))
        self.assertEqual(m.details["unreachable_spans"], [(1, 2), (4, 4)])
        self.assertEqual(m.value, 3.0)


class SummarizeJumpsTest(unittest.TestCase):
    def assert_jumps(self, jumps, expected):
        """details["jumps"] は (K, 2) の int64 の並び。行は (フレームの間 i, 骨の番号)"""
        self.assertIsInstance(jumps, np.ndarray)
        self.assertEqual(jumps.dtype, np.int64)
        self.assertEqual(jumps.shape, (len(expected), 2))
        self.assertEqual(jumps.tolist(), [list(pair) for pair in expected])

    def test_one_jump_is_red(self):
        m = km.summarize_jumps(km.bone_jump_frames(jump_sequence()), bone_names=["センター", "左腕", "左ひじ"])
        self.assertEqual((m.name, m.unit, m.signal, m.value), ("bone_jumps", "count", Signal.RED, 1.0))
        self.assert_jumps(m.details["jumps"], [(2, 1)])
        self.assertEqual(m.details["jump_spans"], [(2, 2)])
        self.assertEqual(m.details["bone_jump_counts"], [("左腕", 1)])

    def test_counts_and_order(self):
        mask = np.zeros((6, 3), dtype=bool)
        mask[4, 0] = mask[1, 2] = mask[1, 0] = mask[2, 2] = True
        m = km.summarize_jumps(mask)
        self.assertEqual(m.value, 4.0)
        self.assert_jumps(m.details["jumps"], [(1, 0), (1, 2), (2, 2), (4, 0)])
        self.assertEqual(m.details["jump_spans"], [(1, 2), (4, 4)])
        self.assertEqual(m.details["bone_jump_counts"], [(0, 2), (2, 2)])

    def test_green_gray_and_invalid(self):
        green = km.summarize_jumps(np.zeros((3, 2), dtype=bool))
        self.assertEqual(green.signal, Signal.GREEN)
        self.assert_jumps(green.details["jumps"], [])
        self.assertEqual(km.summarize_jumps(np.zeros((0, 2), dtype=bool)).signal, Signal.GRAY)
        self.assertEqual(km.summarize_jumps(np.zeros((3, 0), dtype=bool)).signal, Signal.GRAY)
        with self.assertRaises(ValueError):
            km.summarize_jumps(np.zeros(3, dtype=bool))
        with self.assertRaises(ValueError):
            km.summarize_jumps(np.zeros((3, 2), dtype=int))
        with self.assertRaises(ValueError):
            km.summarize_jumps(np.zeros((3, 2), dtype=bool), bone_names=["左腕"])

    def test_bone_names_must_not_be_one_string(self):
        # 文字の並び "左ひじ" は、長さが骨の数と同じでも、名前の一覧ではない（"左"・"ひ"・"じ" にしてしまう）
        mask = np.ones((2, 3), dtype=bool)
        with self.assertRaises(ValueError):
            km.summarize_jumps(mask, bone_names="左ひじ")
        m = km.summarize_jumps(mask, bone_names=("左腕", "左ひじ", "左手首"))  # タプルはよい
        self.assertEqual(m.details["bone_jump_counts"], [("左腕", 2), ("左ひじ", 2), ("左手首", 2)])

    def test_many_jumps_use_little_memory(self):
        # 壊れた動き（全部の骨が全部のフレームで跳ぶ）でも、跳び1つあたりのメモリを小さく保つ。
        # Python の (i, 骨) の組の一覧だと、1つあたり約 100 バイト（5分・300本の骨で 300MB をこえる）
        mask = np.ones((1000, 120), dtype=bool)
        tracemalloc.start()
        try:
            m = km.summarize_jumps(mask)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(m.value, float(mask.size))
        self.assertEqual(m.details["jumps"].shape, (mask.size, 2))
        self.assertLess(peak / mask.size, 48.0)


class QuatRotateTest(unittest.TestCase):
    def test_matches_rotation_matrix(self):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(20):
            axis, deg, v = rng.normal(size=3), float(rng.uniform(-180, 180)), rng.normal(size=3)
            np.testing.assert_allclose(km.quat_rotate(quat(axis, deg), v), rotate(v, axis, deg), atol=1e-12)

    def test_examples_and_sign(self):
        q = quat([0, 0, 1], 90.0)
        np.testing.assert_allclose(km.quat_rotate(q, [1, 0, 0]), [0, 1, 0], atol=1e-15)
        np.testing.assert_allclose(km.quat_rotate(-q, [1, 0, 0]), [0, 1, 0], atol=1e-15)  # q と -q は同じ回転
        np.testing.assert_allclose(km.quat_rotate(3.0 * q, [1, 0, 0]), [0, 1, 0], atol=1e-15)  # 長さ 1 でなくてよい
        np.testing.assert_allclose(km.quat_rotate([q, q], [[1, 0, 0], [0, 1, 0]]), [[0, 1, 0], [-1, 0, 0]], atol=1e-15)
        with self.assertRaises(ValueError):
            km.quat_rotate([0, 0, 0, 0], [1, 0, 0])


if __name__ == "__main__":
    unittest.main()

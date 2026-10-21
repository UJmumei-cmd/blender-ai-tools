"""A4 測る道具の自己テスト（ロードマップ段階1 基準1。CLAUDE.md 10章「できあがりの基準」1）。

「わざと壊した見本（約20種）を全部見つける。壊していない見本では赤が0。同じ入力を2回測ると、決めた幅の中で同じ数字になる。」

見本は tests/fixtures/broken_samples.py が作る（bpy なし）。ここでは、次のことを確かめる。
- 壊していない見本：どの測る道具でも赤が出ない。全部を測れる（灰も出ない）。
- わざと壊した見本：見つけるはずの測る物（finds）が、決めた信号（赤か黄）になり、値が 0 より大きい。
  ほかの測る物の信号は、壊していない見本と同じ（一緒に変わってよい物は also に書き、理由をここと見本の説明に書く）。
- 測れない見本（GRAY_CASES）：灰になり、理由が書いてある。赤にはならない（ひざの逆曲がりは、まちがえて赤にしない）。
- 何も変えない書き換え（HARMLESS）：どの信号も、どの値も変わらない。
- 同じ入力を2回測る・小分けの仕方を変える・乱数の種を変える（前の版の小さな差だけが変わる）：値は単位ごとの再現性の幅
  （m → REPRO_TOLERANCE_LENGTH、deg → REPRO_TOLERANCE_ANGLE_DEG、ratio → REPRO_TOLERANCE_RATIO、count → ぴったり）の中で同じ。
"""

import unittest
from dataclasses import fields

import numpy as np

from blender_ai_tools import thresholds
from blender_ai_tools.measure import deformation, geometry, kinematics, mesh_check, symmetry, weights
from blender_ai_tools.measure.results import Measurement, Signal, worst
from tests.fixtures import broken_samples
from tests.fixtures.broken_samples import BREAKAGES, GRAY_CASES, HARMLESS, KNOWN_MISSES, clean_sample, measure_sample

# 単位ごとの再現性の幅（これより小さい差は、同じ数字とみなす）。count はぴったり同じでなければならない
UNIT_TOLERANCE = {
    "m": thresholds.REPRO_TOLERANCE_LENGTH,
    "deg": thresholds.REPRO_TOLERANCE_ANGLE_DEG,
    "ratio": thresholds.REPRO_TOLERANCE_RATIO,
    "count": 0.0,
}

# measure_sample で測る、A4 の測る物の名前（visibility の visible_points は、画像を描くので重く、ここでは測らない）
MEASURED_NAMES = {
    mesh_check.DUPLICATES_NAME, mesh_check.DEGENERATE_NAME, mesh_check.NON_MANIFOLD_NAME,
    mesh_check.INCONSISTENT_NAME, mesh_check.HOLES_NAME, mesh_check.LOOSE_PARTS_NAME,
    symmetry.ASYMMETRY_NAME, geometry.PENETRATION_NAME, geometry.GAP_NAME,
    weights.OVER_LIMIT_NAME, weights.UNWEIGHTED_NAME, weights.NOT_NORMALIZED_NAME, weights.NEGATIVE_NAME,
    weights.DUPLICATE_NAME,
    deformation.STRETCH_NAME, deformation.FLIPS_NAME, deformation.VOLUME_NAME, deformation.SPIKES_NAME,
    deformation.CHANGES_NAME,
    kinematics.REVERSE_BEND_NAME, kinematics.FLOOR_SINK_NAME, kinematics.FOOT_SLIDE_NAME,
    kinematics.UNREACHABLE_NAME, kinematics.BONE_JUMPS_NAME,
}

# わざと壊した見本の、一緒に変わってよい測る物と、その理由（見本の also と、ぴったり同じ組でなければならない）
SIDE_EFFECT_REASONS = {
    ("split_seam", mesh_check.HOLES_NAME):
        "頂点の番号の上で切れ目ができ、縁の辺の輪になる（重なった点をまとめると消える。PMX の UV の切れ目と同じ）",
    ("split_seam", deformation.VOLUME_NAME):
        "縁の辺があり閉じた形ではないので、体積の変わり方を測れない（灰）",
    ("degenerate_triangle", deformation.FLIPS_NAME):
        "面積 0 の三角形は向きを比べられないので、裏返りの確かめが灰になる（つぶれた三角形を直すと消える）",
    ("flipped_triangle", deformation.VOLUME_NAME):
        "向きの合わない辺は行きと帰りがそろわず、閉じた形ではないので、体積の変わり方を測れない（灰）",
    ("hole", deformation.VOLUME_NAME):
        "穴があり閉じた形ではないので、体積の変わり方を測れない（灰）",
    ("cloth_inside", geometry.GAP_NAME):
        "体の中の服の点は、すき間が 0（近すぎる）にも数えられる",
    ("folded_pose", deformation.STRETCH_NAME):
        "1つの頂点で面を折ると、まわりの辺が必ず大きく伸び縮みする",
    ("folded_pose", deformation.SPIKES_NAME):
        "1つの頂点で面を折ると、その頂点がまわりから大きくずれる（とげ）",
}

# ロードマップの「約20種」
MIN_BREAKAGES = 20


def same_number(test: unittest.TestCase, first: Measurement, second: Measurement, where: str) -> None:
    """2つの結果が、信号も、単位ごとの再現性の幅の中の値も同じか"""
    test.assertEqual(first.unit, second.unit, where)
    test.assertIn(first.unit, UNIT_TOLERANCE, f"{where}：再現性の幅が決まっていない単位 {first.unit}")
    test.assertEqual(first.signal, second.signal, where)
    if first.value is None or second.value is None:
        test.assertEqual(first.value, second.value, where)
        return
    test.assertLessEqual(abs(first.value - second.value), UNIT_TOLERANCE[first.unit],
                         f"{where}：{first.value} と {second.value}")


def by_name(results: dict[str, Measurement], name: str) -> dict[str, Measurement]:
    return {key: measurement for key, measurement in results.items() if measurement.name == name}


def poke_through_depth(sample: broken_samples.Sample) -> float:
    """体の頂点が、服の面の外へ出ている、いちばん深い長さ（測る道具を使わずに計算する）。

    体の頂点ごとに、服の三角形の面に垂直に下ろした足が、その三角形の中にあるときだけ、面からの高さ
    （外向きの法線の側が正）を求める。服は閉じた殻で外を向くので、正なら体が服の外へ出ている。
    """
    corners = sample.cloth_vertices[sample.cloth_triangles]  # (M, 3, 3)
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    points = sample.body_vertices[:, None, :]  # (N, 1, 3)
    height = ((points - a) * normal).sum(axis=2)  # (N, M)
    foot = points - height[..., None] * normal
    # 足が三角形の中か（3本の辺のどれについても、内側にある）
    inside = np.ones(height.shape, dtype=bool)
    for p, q in ((a, b), (b, c), (c, a)):
        inside &= (np.cross(q - p, foot - p) * normal).sum(axis=2) >= 0.0
    return float(np.where(inside, height, 0.0).max(initial=0.0))


class SelfCheckBase(unittest.TestCase):
    """壊していない見本と、その測った結果を1回だけ作る（どのテストでも同じ物を使う）"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.clean = clean_sample()
        cls.baseline = measure_sample(cls.clean)

    def assert_signals(self, results: dict[str, Measurement], name: str, expected: Signal, label: str) -> None:
        """name の測る物が expected の信号になり（場所が2つ以上ある物は、どれか1つ）、ほかの場所は壊していない見本のまま"""
        found = by_name(results, name)
        self.assertTrue(found, f"{label}：{name} を測っていません")
        hits = [key for key, measurement in found.items() if measurement.signal == expected]
        self.assertTrue(hits, f"{label}：{name} が {expected.label} になりません（"
                              + "、".join(f"{key}={m.signal.label}" for key, m in found.items()) + "）")
        for key, measurement in found.items():
            if measurement.signal != expected:
                self.assertEqual(measurement.signal, self.baseline[key].signal, f"{label}：{key}")
        if expected == Signal.GRAY:
            for key in hits:
                self.assertTrue(found[key].reason, f"{label}：{key} が灰なのに理由がありません")
        else:
            changed = [key for key in hits if found[key].value is not None and found[key].value > 0.0]
            self.assertTrue(changed, f"{label}：{name} の値が 0 より大きくなりません")


# MARK: - 壊していない見本

class CleanSampleTest(SelfCheckBase):
    def test_no_red(self):
        """壊していない見本では、どの測る道具でも赤が出ない"""
        for key, measurement in self.baseline.items():
            with self.subTest(measurement=key):
                self.assertNotEqual(measurement.signal, Signal.RED, measurement.reason)

    def test_fully_measurable(self):
        """壊していない見本は、全部を測れる（灰が出ない）。もし灰があれば、理由が書いてあること"""
        for key, measurement in self.baseline.items():
            with self.subTest(measurement=key):
                if measurement.signal == Signal.GRAY:
                    self.assertTrue(measurement.reason)
                self.assertNotEqual(measurement.signal, Signal.GRAY, measurement.reason)
                self.assertIsNotNone(measurement.value)

    def test_all_green(self):
        """黄も出ない（どの基準からも離れた、壊していない見本）"""
        self.assertEqual(worst(m.signal for m in self.baseline.values()), Signal.GREEN,
                         {key: m.signal.label for key, m in self.baseline.items() if m.signal != Signal.GREEN})

    def test_measures_every_a4_measure(self):
        """A4 の測る物（見え方を除く）を全部測る。名前は各部品の定数と同じ"""
        self.assertEqual({m.name for m in self.baseline.values()}, MEASURED_NAMES)
        for key, measurement in self.baseline.items():
            with self.subTest(measurement=key):
                self.assertTrue(key.endswith("/" + measurement.name))
                self.assertIn(measurement.unit, UNIT_TOLERANCE)

    def test_measures_something(self):
        """「0 個」で合格にしていない：調べた物の数が 0 でない"""
        details = {key: m.details for key, m in self.baseline.items()}
        self.assertGreater(details["pose/edge_stretch"]["edge_count"], 0)
        self.assertGreater(details["cloth/penetration_depth"]["point_count"], 0)
        self.assertGreater(details["weights/unweighted_vertices"]["vertex_count"], 0)
        self.assertGreater(details["version/unexpected_changes"]["allowed_changed_count"], 0)
        for side in broken_samples.SIDES:
            with self.subTest(side=side):
                self.assertGreater(details[f"{side}/foot_slide"]["span_count"], 0)
                self.assertEqual(details[f"{side}/reverse_bend"]["unmeasured_count"], 0)

    def test_sample_is_well_formed(self):
        """見本そのものの決まり：体は閉じて向きがそろい、ウェイトは (N, K) で空きは -1・0、合計 1、1点4本まで"""
        sample = self.clean
        self.assertEqual(sample.bone_ids.shape[1], thresholds.MAX_BONE_INFLUENCES)
        empty = sample.bone_ids == -1
        self.assertTrue(empty.any(), "空きの枠（-1）が無い見本では、空きの扱いを確かめられません")
        self.assertTrue((sample.bone_weights[empty] == 0.0).all())
        sums = sample.bone_weights.sum(axis=1)
        self.assertLessEqual(float(np.abs(sums - 1.0).max()), thresholds.WEIGHT_SUM_TOLERANCE)
        self.assertEqual(len(mesh_check.boundary_edges(sample.body_triangles)), 0)
        self.assertEqual(len(mesh_check.inconsistent_edges(sample.body_triangles)), 0)
        gap = geometry.gaps(sample.cloth_vertices, sample.body_vertices, sample.body_triangles)
        self.assertTrue(((gap >= thresholds.CLOTH_GAP_MIN) & (gap <= thresholds.CLOTH_GAP_MAX)).all())
        self.assertEqual(sample.seed, thresholds.DEFAULT_SEED)

    def test_pose_really_deforms(self):
        """ポーズで本当に形が変わる（形のくずれの確かめが、動かない形だけで合格していない）。
        縮む辺のいちばん小さい割合は、見本の見込み（CLEAN_MIN_EDGE_RATIO。shrunk_pose が使う）より大きい"""
        stretch = self.baseline["pose/edge_stretch"].details
        self.assertGreater(stretch["max_ratio"], 1.0 + thresholds.REPRO_TOLERANCE_RATIO)
        self.assertLess(stretch["min_ratio"], 1.0 - thresholds.REPRO_TOLERANCE_RATIO)
        self.assertGreater(stretch["min_ratio"], broken_samples.CLEAN_MIN_EDGE_RATIO)

    def test_arrays_are_read_only(self):
        """見本の並びは書き換えられない（壊すときは、必ず写しを作る）"""
        with self.assertRaises(ValueError):
            self.clean.body_vertices[0, 0] = 1.0
        with self.assertRaises(ValueError):
            self.clean.bone_weights[0, 0] = 0.0


# MARK: - わざと壊した見本

class BreakageTest(SelfCheckBase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.broken = {name: measure_sample(breakage.apply(cls.clean)) for name, breakage in BREAKAGES.items()}

    def test_about_twenty_kinds(self):
        self.assertGreaterEqual(len(BREAKAGES), MIN_BREAKAGES)
        for name, breakage in BREAKAGES.items():
            with self.subTest(breakage=name):
                self.assertIn(breakage.expected, (Signal.RED, Signal.YELLOW))
                self.assertTrue(breakage.finds)
                self.assertTrue(breakage.apply.__doc__, "壊し方の説明（docstring）を書いてください")

    def test_every_measure_has_a_breakage(self):
        """A4 の測る物（見え方を除く）は、どれも、わざと壊した見本のどれかで見つかる"""
        self.assertEqual({name for breakage in BREAKAGES.values() for name in breakage.finds}, MEASURED_NAMES)

    def test_every_breakage_is_found(self):
        """見つけるはずの測る物が、決めた信号になり、値（数）が 0 より大きい"""
        for name, breakage in BREAKAGES.items():
            with self.subTest(breakage=name):
                for measure in breakage.finds:
                    self.assert_signals(self.broken[name], measure, breakage.expected, name)

    def test_side_effects_are_declared_with_reasons(self):
        """一緒に変わってよい測る物（also）は、ここ（SIDE_EFFECT_REASONS）にも理由があり、見本の説明にも名前が書いてある"""
        declared = {(name, measure) for name, breakage in BREAKAGES.items() for measure in breakage.also}
        self.assertEqual(declared, set(SIDE_EFFECT_REASONS))
        for (name, measure), reason in SIDE_EFFECT_REASONS.items():
            with self.subTest(breakage=name, measure=measure):
                self.assertTrue(reason)
                self.assertIn(measure, BREAKAGES[name].apply.__doc__)
                self.assertNotIn(measure, BREAKAGES[name].finds)

    def test_side_effects_happen_as_declared(self):
        """一緒に変わってよいと書いた物は、書いた信号になる（書いた理由が、いまも正しい）"""
        for name, breakage in BREAKAGES.items():
            for measure, signal in breakage.also.items():
                with self.subTest(breakage=name, measure=measure, reason=SIDE_EFFECT_REASONS[(name, measure)]):
                    self.assert_signals(self.broken[name], measure, signal, name)

    def test_only_declared_measures_change(self):
        """finds と also のほかの測る物は、信号が変わらない（赤・黄・灰が増えない）。数（count）は値も変わらない"""
        for name, breakage in BREAKAGES.items():
            touched = set(breakage.finds) | set(breakage.also)
            for key, measurement in self.broken[name].items():
                if measurement.name in touched:
                    continue
                with self.subTest(breakage=name, measurement=key):
                    before = self.baseline[key]
                    self.assertEqual(measurement.signal, before.signal, measurement.reason)
                    if measurement.unit == "count":
                        self.assertEqual(measurement.value, before.value)

    def test_breaking_does_not_touch_the_clean_sample(self):
        """壊しても、もとの見本は変わらない（新しく作った見本と、ぴったり同じ）"""
        fresh = clean_sample()
        for name, breakage in BREAKAGES.items():
            breakage.apply(self.clean)
        for item in fields(broken_samples.Sample):
            with self.subTest(field=item.name):
                np.testing.assert_array_equal(np.asarray(getattr(self.clean, item.name)),
                                              np.asarray(getattr(fresh, item.name)))


# MARK: - 測れない見本・何も変えない書き換え

class GrayCaseTest(SelfCheckBase):
    def test_unmeasurable_is_gray_not_red(self):
        """測れない見本は灰（理由つき）。どこにも赤は出ない（ひざの逆曲がりを、まちがえて赤にしない）"""
        for name, case in GRAY_CASES.items():
            results = measure_sample(case.apply(self.clean))
            with self.subTest(case=name):
                self.assertEqual(case.expected, Signal.GRAY)
                for measure in case.finds:
                    self.assert_signals(results, measure, Signal.GRAY, name)
                for key, measurement in results.items():
                    self.assertNotEqual(measurement.signal, Signal.RED, f"{key}：{measurement.reason}")
                    if measurement.name not in case.finds:
                        self.assertEqual(measurement.signal, self.baseline[key].signal, key)


class HarmlessEditTest(SelfCheckBase):
    def test_signals_and_values_do_not_change(self):
        """形も動きも変えない書き換えでは、どの信号も、どの値も（再現性の幅の中で）変わらない"""
        for name, edit in HARMLESS.items():
            results = measure_sample(edit(self.clean))
            for key, measurement in results.items():
                with self.subTest(edit=name, measurement=key):
                    same_number(self, measurement, self.baseline[key], f"{name}/{key}")


# MARK: - いまの測り方では見つからない壊れ方

class KnownMissTest(SelfCheckBase):
    """KNOWN_MISSES：見つかるのが正しいが、いまの測る道具では見つからない壊れ方（報告に書いた。直したら BREAKAGES へ移し、
    expectedFailure を外す。外し忘れると、思いがけず通ったテストとして、全体が失敗になる）"""

    def test_cases_are_really_broken(self):
        """見本は本当に壊れていて（測る道具とは別の計算で確かめる）、ほかの測る物はどれも変わらない"""
        skating = KNOWN_MISSES["skating_foot"].apply(self.clean)
        sole = skating.soles[:, 0]
        stance = slice(0, broken_samples.SWING_FRAMES)
        self.assertLessEqual(float(sole[stance, 2].max()) - skating.floor_z, thresholds.FOOT_CONTACT_HEIGHT)
        glide = float(np.hypot(*(sole[stance][-1, :2] - sole[stance][0, :2])))
        self.assertGreater(glide, thresholds.FOOT_SLIDE_TOLERANCE)

        coarse = KNOWN_MISSES["coarse_cloth"].apply(self.clean)
        self.assertGreater(poke_through_depth(coarse), thresholds.PENETRATION_RED)
        self.assertEqual(len(mesh_check.boundary_edges(coarse.cloth_triangles)), 0)
        self.assertEqual(len(mesh_check.inconsistent_edges(coarse.cloth_triangles)), 0)

        for name, case in KNOWN_MISSES.items():
            results = measure_sample(case.apply(self.clean))
            for key, measurement in results.items():
                with self.subTest(case=name, measurement=key):
                    self.assertEqual(measurement.signal, self.baseline[key].signal, measurement.reason)

    @unittest.expectedFailure
    def test_fast_foot_glide_is_found(self):
        """床の高さのまま速く滑る足：foot_contacts が「遅い」フレームだけを着いているとみなすので、いまは足すべりが緑"""
        case = KNOWN_MISSES["skating_foot"]
        self.assert_signals(measure_sample(case.apply(self.clean)), case.finds[0], case.expected, "skating_foot")

    @unittest.expectedFailure
    def test_body_through_coarse_cloth_is_found(self):
        """粗い服の面のまん中を突き抜ける体：服の頂点だけを調べるので、いまはめり込みが緑"""
        case = KNOWN_MISSES["coarse_cloth"]
        self.assert_signals(measure_sample(case.apply(self.clean)), case.finds[0], case.expected, "coarse_cloth")


# MARK: - 再現性

class ReproducibilityTest(SelfCheckBase):
    def assert_same(self, first: dict[str, Measurement], second: dict[str, Measurement], label: str) -> None:
        self.assertEqual(set(first), set(second))
        for key in first:
            with self.subTest(case=label, measurement=key):
                same_number(self, first[key], second[key], f"{label}/{key}")

    def test_same_input_twice(self):
        """同じ入力（作り直した見本）を2回測ると、決めた幅の中で同じ数字になる"""
        self.assert_same(self.baseline, measure_sample(clean_sample()), "clean")

    def test_broken_samples_twice(self):
        """わざと壊した見本も、作り直して2回測ると、決めた幅の中で同じ数字になる"""
        for name, breakage in BREAKAGES.items():
            first = measure_sample(breakage.apply(clean_sample()))
            second = measure_sample(breakage.apply(clean_sample()))
            self.assert_same(first, second, name)

    def test_chunk_size_does_not_change_numbers(self):
        """小分けにして計算する道具は、小分けの大きさを変えても同じ数字になる。
        61（素数）では、三角形 320・点 162・フレーム 60 のどれも、半端な小分けになる（1 のように小さいと、遅すぎる）"""
        for chunk in (61, 1000):
            with self.subTest(chunk=chunk):
                self.assert_same(self.baseline, measure_sample(self.clean, chunk_size=chunk), f"chunk={chunk}")

    def test_seed_only_changes_sub_tolerance_noise(self):
        """乱数の種を変えると、前の版の小さな差（再現性の幅より小さい）だけが変わり、どの数字も変わらない"""
        other = clean_sample(seed=thresholds.DEFAULT_SEED + 1)
        self.assertEqual(other.seed, thresholds.DEFAULT_SEED + 1)
        self.assertFalse(np.array_equal(other.previous_body_vertices, self.clean.previous_body_vertices))
        self.assert_same(self.baseline, measure_sample(other), "seed")

    def test_weight_numbers_within_weight_tolerance(self):
        """ウェイトの数字（重みの合計のずれ・いちばん小さい重み）は、REPRO_TOLERANCE_WEIGHT の中で同じ"""
        cases = [("clean", self.clean, clean_sample())]
        for name in ("not_normalized", "negative_weight"):
            cases.append((name, BREAKAGES[name].apply(self.clean), BREAKAGES[name].apply(clean_sample())))
        for label, first, second in cases:
            with self.subTest(case=label):
                np.testing.assert_allclose(weights.sums(first.weight_table(), deform=first.deform_bones),
                                           weights.sums(second.weight_table(), deform=second.deform_bones),
                                           rtol=0.0, atol=thresholds.REPRO_TOLERANCE_WEIGHT)
                one = {m.name: m.details for m in weights.summarize(first.weight_table(), deform=first.deform_bones)}
                two = {m.name: m.details for m in weights.summarize(second.weight_table(), deform=second.deform_bones)}
                for measure, key in ((weights.NOT_NORMALIZED_NAME, "max_deviation"), (weights.NEGATIVE_NAME, "min_weight")):
                    self.assertLessEqual(abs(one[measure][key] - two[measure][key]), thresholds.REPRO_TOLERANCE_WEIGHT)


if __name__ == "__main__":
    unittest.main()

"""A4 スキンのウェイト（measure/weights.py）を確かめる"""

import math
import unittest
from unittest import mock

import numpy as np

from blender_ai_tools import thresholds
from blender_ai_tools.measure import weights as w
from blender_ai_tools.measure.results import Signal
from blender_ai_tools.measure.weights import WeightTable

EPS = thresholds.WEIGHT_EPSILON
TOL = thresholds.WEIGHT_SUM_TOLERANCE


# MARK: - 比べるための、頂点ごとの素直な実装

def reference_fix(ids_row, weights_row, *, limit, deform, epsilon=EPS, tolerance=TOL):
    """normalize_and_limit の決まりを、頂点1つずつ、そのまま書いた物。返す物は（骨の番号の並び、重みの並び、変わったか）"""
    width = len(ids_row)
    merged: dict[int, float] = {}
    others: list[tuple[int, float]] = []
    seen: set[int] = set()
    duplicate = negative = False
    raw_sum = 0.0
    for bone, weight in zip(ids_row, weights_row):
        bone = int(bone)
        if bone < 0:
            continue
        if deform is not None and not deform[bone]:
            others.append((bone, float(weight)))
            continue
        duplicate |= bone in seen
        seen.add(bone)
        negative |= weight < 0
        raw_sum += weight
        merged[bone] = merged.get(bone, 0.0) + max(float(weight), 0.0)
    influences = {bone: weight for bone, weight in merged.items() if weight > epsilon}
    unchanged = (list(ids_row), list(weights_row), False)
    if not influences:
        return unchanged
    valid = len(influences) <= limit and abs(raw_sum - 1.0) <= tolerance and not duplicate and not negative
    if valid:
        return unchanged
    kept = sorted(influences.items(), key=lambda item: (-item[1], item[0]))[:limit]
    total = sum(weight for _, weight in kept)
    entries = [(bone, weight / total) for bone, weight in kept] + others
    entries += [(-1, 0.0)] * (width - len(entries))
    return [bone for bone, _ in entries], [weight for _, weight in entries], True


def random_table(seed: int, count: int = 400, width: int = 7, bones: int = 9, dyadic: bool = True):
    """でたらめな表。同じ重み（引き分け）・同じ骨の2回目・負の重み・ごく小さい重み・正しい頂点を混ぜる"""
    rng = np.random.default_rng(seed)
    ids = rng.integers(-1, bones, size=(count, width))
    if dyadic:
        # 1/16 の倍数（足し算に丸めが出ないので、引き分けがそのまま残る）
        weights = rng.integers(-2, 9, size=(count, width)) / 16.0
        tiny = rng.random((count, width)) < 0.05
        weights[tiny] = 2.0 ** -21  # epsilon より小さい（2 のべきなので、足しても丸めが出ない）
    else:
        weights = rng.random((count, width)) * 1.2 - 0.1
    weights[ids < 0] = 0.0
    # 4分の1の頂点は、正しい頂点（骨は1〜4本、重複なし、合計 1）にする
    for row in range(0, count, 4):
        used = rng.choice(bones, size=rng.integers(1, 5), replace=False)
        parts = rng.integers(1, 5, size=len(used)).astype(float)
        if dyadic:
            parts = np.zeros(len(used))
            parts[0] = 1.0  # 1/16 の倍数で合計 1 にする：まず 1 を置き、半分ずつ分ける
            for k in range(1, len(used)):
                parts[k] = parts[k - 1] / 2.0
                parts[k - 1] -= parts[k]
        ids[row] = -1
        weights[row] = 0.0
        ids[row, :len(used)] = used
        weights[row, :len(used)] = parts / parts.sum()
    deform = rng.random(bones) < 0.75
    return WeightTable(ids, weights), deform


# MARK: - 表

class WeightTableTest(unittest.TestCase):
    def test_from_lists(self):
        table = WeightTable.from_lists([[(2, 0.5), (0, 0.5)], [], [(1, 1.0)]])
        np.testing.assert_array_equal(table.bone_ids, [[2, 0], [-1, -1], [1, -1]])
        np.testing.assert_array_equal(table.weights, [[0.5, 0.5], [0.0, 0.0], [1.0, 0.0]])
        self.assertEqual(table.bone_ids.dtype, np.int64)
        self.assertEqual(table.weights.dtype, np.float64)
        self.assertEqual((table.vertex_count, table.width), (3, 2))

    def test_from_lists_empty(self):
        self.assertEqual(WeightTable.from_lists([]).bone_ids.shape, (0, 0))
        self.assertEqual(WeightTable.from_lists([[], []]).bone_ids.shape, (2, 0))
        table = WeightTable.from_lists([[(np.int64(3), np.float32(0.5))]])  # numpy の数も受け付ける
        self.assertEqual((table.bone_ids[0, 0], table.weights[0, 0]), (3, 0.5))

    def test_from_lists_rejects_bad_entries(self):
        for bad in ([[(True, 1.0)]], [[(0, True)]], [[(-1, 0.0)]], [[(1.0, 0.5)]], [[(0, math.nan)]],
                    [[(0, 0.5, 1)]], [[(0,)]], [[0.5]], ["ab"], [[("a", 1.0)]], "abc",
                    [[(0, 10 ** 400)]]):  # float にできない大きな整数（OverflowError ではなく ValueError）
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                WeightTable.from_lists(bad)

    def test_dense_round_trip(self):
        dense = np.array([
            [0.0, 0.25, 0.0, 0.75],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, -0.5, 0.3, 0.2],
            [0.0, EPS / 2.0, 0.6, 0.0],
        ])
        table = WeightTable.from_dense(dense)
        self.assertEqual(table.width, 4)
        np.testing.assert_array_equal(table.bone_ids[0], [1, 3, -1, -1])  # 骨の番号の小さい順
        np.testing.assert_array_equal(table.bone_ids[3], [2, -1, -1, -1])  # ごく小さい重みは入れない
        expected = dense.copy()
        expected[3, 1] = 0.0
        np.testing.assert_array_equal(table.to_dense(4), expected)
        again = WeightTable.from_dense(table.to_dense(4))
        np.testing.assert_array_equal(again.bone_ids, table.bone_ids)
        np.testing.assert_array_equal(again.weights, table.weights)

    def test_lists_dense_lists_round_trip(self):
        table = WeightTable.from_lists([[(0, 0.4), (3, 0.6)], [(1, 1.0)], [(0, 0.1), (2, 0.2), (3, 0.7)]])
        again = WeightTable.from_dense(table.to_dense(5))
        np.testing.assert_array_equal(again.bone_ids, table.bone_ids)
        np.testing.assert_array_equal(again.weights, table.weights)

    def test_from_dense_epsilon_and_width(self):
        table = WeightTable.from_dense(np.array([[0.1, 0.2, 0.3]]), epsilon=0.2)
        np.testing.assert_array_equal(table.bone_ids, [[2]])
        self.assertEqual(WeightTable.from_dense(np.zeros((3, 5))).bone_ids.shape, (3, 0))
        self.assertEqual(WeightTable.from_dense(np.zeros((0, 5))).bone_ids.shape, (0, 0))
        with self.assertRaises(ValueError):
            WeightTable.from_dense(np.array([[math.nan]]))
        with self.assertRaises(ValueError):
            WeightTable.from_dense(np.array([[True]]))
        with self.assertRaises(ValueError):
            WeightTable.from_dense(np.array([1.0, 0.0]))

    def test_to_dense_sums_duplicates(self):
        table = WeightTable.from_lists([[(1, 0.25), (1, 0.5), (0, 0.25)]])
        np.testing.assert_array_equal(table.to_dense(3), [[0.25, 0.75, 0.0]])
        with self.assertRaises(ValueError):
            table.to_dense(1)
        with self.assertRaises(ValueError):
            table.to_dense(True)
        with self.assertRaises(ValueError):
            table.to_dense(-1)
        self.assertEqual(WeightTable.from_lists([[], []]).to_dense(0).shape, (2, 0))

    def test_validation(self):
        good_ids = np.array([[0, -1]])
        good_weights = np.array([[1.0, 0.0]])
        bad_cases = [
            (np.array([[0, -1, 2]]), good_weights),  # 形が違う
            (np.array([[0, -2]]), good_weights),  # -1 より小さい
            (good_ids, np.array([[1.0, 0.5]])),  # 空きの枠に重み
            (good_ids, np.array([[math.nan, 0.0]])),
            (good_ids, np.array([[math.inf, 0.0]])),
            (np.array([[True, False]]), good_weights),
            (np.array([[0.0, -1.0]]), good_weights),  # 小数の番号
            (good_ids, np.array([[True, False]])),
            (np.array([0, -1]), np.array([1.0, 0.0])),  # 1次元
        ]
        for ids, weights in bad_cases:
            with self.subTest(ids=ids, weights=weights), self.assertRaises(ValueError):
                WeightTable(ids, weights)
        table = WeightTable(good_ids, good_weights)
        self.assertEqual(table.bone_ids.dtype, np.int64)
        WeightTable(np.array([[0, 1]]), np.array([[-0.5, 1.5]]))  # 負の重みは表としては受け付ける

    def test_arrays_are_copied_and_read_only(self):
        ids = np.array([[0, 1]])
        weights = np.array([[0.5, 0.5]])
        table = WeightTable(ids, weights)
        ids[0, 0] = 5
        weights[0, 0] = 9.0
        self.assertEqual(table.bone_ids[0, 0], 0)
        self.assertEqual(table.weights[0, 0], 0.5)
        with self.assertRaises(ValueError):
            table.weights[0, 0] = 1.0
        with self.assertRaises(ValueError):
            table.bone_ids[0, 0] = 1


# MARK: - 数える・点検

class CountTest(unittest.TestCase):
    def test_sums(self):
        table = WeightTable.from_lists([[(0, 0.5), (1, 0.25), (2, -0.25)], [(1, 1.0)]])
        np.testing.assert_array_equal(w.sums(table), [0.5, 1.0])
        np.testing.assert_array_equal(w.sums(table, deform=np.array([True, False, True])), [0.25, 0.0])

    def test_influence_counts(self):
        table = WeightTable.from_lists([
            [(0, 0.3), (0, 0.3), (1, 0.4)],  # 同じ骨は1本
            [(0, EPS), (1, np.nextafter(EPS, 1.0))],  # ちょうど epsilon は数えない
            [(0, -0.5), (1, 1.5)],  # 負は数えない
            [(0, 0.25), (1, 0.25), (2, 0.25), (3, 0.25)],
            [],
        ])
        np.testing.assert_array_equal(w.influence_counts(table), [2, 1, 1, 4, 0])
        self.assertEqual(w.influence_counts(table).dtype, np.int64)
        deform = np.array([True, False, True, True])
        np.testing.assert_array_equal(w.influence_counts(table, deform=deform), [1, 0, 0, 3, 0])
        np.testing.assert_array_equal(w.influence_counts(table, epsilon=0.3), [2, 0, 1, 0, 0])

    def test_duplicates_merge_before_epsilon(self):
        table = WeightTable.from_lists([[(0, 0.6 * EPS), (0, 0.6 * EPS)]])
        self.assertEqual(w.influence_counts(table)[0], 1)

    def test_check_masks(self):
        table = WeightTable.from_lists([
            [(0, 1.0)],  # 0：正しい
            [],  # 1：どの骨にも付いていない
            [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2), (4, 0.2)],  # 2：5本
            [(0, 0.5), (1, 0.4)],  # 3：合計 0.9
            [(0, 1.1), (1, -0.1)],  # 4：負（合計は 1）
            [(2, 0.5), (2, 0.5)],  # 5：同じ骨が2回（合計は 1）
            [(5, 1.0), (0, 0.0)],  # 6：形を動かさない骨だけ → 付いていない
            [(0, 1.0 + 0.9 * TOL)],  # 7：幅の中
            [(0, -1.0)],  # 8：負だけ → 付いていない、負
        ])
        deform = np.array([True, True, True, True, True, False])
        flags = w.check(table, deform=deform)
        self.assertEqual(tuple(flags), w.CHECK_KEYS)
        expected = {
            "unweighted": [1, 6, 8],
            "over_limit": [2],
            "not_normalized": [3],
            "negative": [4, 8],
            "duplicate_ids": [5],
        }
        for key, rows in expected.items():
            with self.subTest(key=key):
                self.assertEqual(flags[key].dtype, bool)
                np.testing.assert_array_equal(np.flatnonzero(flags[key]), rows)

    def test_check_limit_and_sum_boundary(self):
        table = WeightTable.from_lists([[(0, 0.5), (1, 0.5)], [(0, 1.0 + 2 * TOL)]])
        np.testing.assert_array_equal(w.check(table, limit=1)["over_limit"], [True, False])
        np.testing.assert_array_equal(w.check(table)["not_normalized"], [False, True])
        np.testing.assert_array_equal(w.check(table, tolerance=3 * TOL)["not_normalized"], [False, False])

    def test_sum_tolerance_boundary_is_exclusive(self):
        """合計のずれが、ちょうど tolerance なら正しい（直さない）。わずかに超えると not_normalized（直す）。
        2 のべきの重みなので、足し算に丸めが出ない"""
        table = WeightTable.from_lists([[(0, 1.0), (1, 0.25)], [(0, 1.0), (1, 0.25 + 2.0 ** -20)]])
        flags = w.check(table, tolerance=0.25)
        np.testing.assert_array_equal(flags["not_normalized"], [False, True])
        fixed, changed = w.normalize_and_limit(table, tolerance=0.25)
        np.testing.assert_array_equal(changed, [False, True])
        np.testing.assert_array_equal(fixed.weights[0], table.weights[0])

    def test_non_deform_bones_are_not_checked(self):
        table = WeightTable.from_lists([[(0, 1.0), (1, -0.5), (1, 0.25), (2, 0.1), (3, 0.1), (4, 0.1), (5, 0.1)]])
        deform = np.array([True, False, False, False, False, False])
        flags = w.check(table, deform=deform)
        self.assertFalse(any(flag[0] for flag in flags.values()))

    def test_bad_arguments(self):
        table = WeightTable.from_lists([[(3, 1.0)]])
        with self.assertRaises(ValueError):
            w.check(table, deform=np.array([True, True]))  # 番号 3 に届かない
        with self.assertRaises(ValueError):
            w.check(table, deform=np.array([1, 1, 1, 1]))
        with self.assertRaises(ValueError):
            w.check(table, limit=0)
        with self.assertRaises(ValueError):
            w.check(table, limit=True)
        with self.assertRaises(ValueError):
            w.check(table, limit=2.0)
        with self.assertRaises(ValueError):
            w.influence_counts(table, epsilon=-1.0)
        with self.assertRaises(ValueError):
            w.check((np.array([[0]]), np.array([[1.0]])))


# MARK: - 安全な直し

class NormalizeAndLimitTest(unittest.TestCase):
    def assert_matches_reference(self, table, deform, limit, *, exact):
        fixed, changed = w.normalize_and_limit(table, limit=limit, deform=deform)
        self.assertEqual(fixed.bone_ids.shape, table.bone_ids.shape)
        for row in range(table.vertex_count):
            ids, weights, did_change = reference_fix(table.bone_ids[row], table.weights[row], limit=limit,
                                                     deform=deform)
            self.assertEqual(fixed.bone_ids[row].tolist(), ids, f"頂点 {row}")
            if exact:
                self.assertEqual(fixed.weights[row].tolist(), weights, f"頂点 {row}")
            else:
                np.testing.assert_allclose(fixed.weights[row], weights, rtol=1e-12, atol=1e-15)
            self.assertEqual(bool(changed[row]), did_change, f"頂点 {row}")
        return fixed, changed

    def test_matches_reference_with_ties(self):
        for seed in range(4):
            for limit in (1, 2, 4):
                with self.subTest(seed=seed, limit=limit):
                    table, deform = random_table(seed)
                    self.assert_matches_reference(table, deform, limit, exact=True)
                    self.assert_matches_reference(table, None, limit, exact=True)

    def test_matches_reference_random_floats(self):
        table, deform = random_table(11, dyadic=False)
        self.assert_matches_reference(table, deform, 4, exact=False)

    def test_random_tables_cover_every_case(self):
        """でたらめな表に、確かめたい場合が全部入っていること"""
        table, deform = random_table(0)
        flags = w.check(table, deform=deform)
        for key in w.CHECK_KEYS:
            self.assertTrue(flags[key].any(), key)
        valid = ~np.logical_or.reduce([flags[key] for key in w.CHECK_KEYS])
        self.assertTrue(valid.any())

    def test_tie_at_the_cut(self):
        table = WeightTable.from_lists([[(3, 0.25), (1, 0.25), (7, 0.125), (2, 0.25), (5, 0.125)]])
        fixed, changed = w.normalize_and_limit(table, limit=4)
        np.testing.assert_array_equal(fixed.bone_ids, [[1, 2, 3, 5, -1]])
        np.testing.assert_allclose(fixed.weights, [[2 / 7, 2 / 7, 2 / 7, 1 / 7, 0.0]], rtol=1e-15)
        self.assertTrue(changed[0])

    def test_keeps_largest(self):
        table = WeightTable.from_lists([[(0, 0.05), (1, 0.4), (2, 0.1), (3, 0.3), (4, 0.15)]])
        fixed, _ = w.normalize_and_limit(table, limit=2)
        np.testing.assert_array_equal(fixed.bone_ids, [[1, 3, -1, -1, -1]])
        np.testing.assert_allclose(fixed.weights[0, :2], [0.4 / 0.7, 0.3 / 0.7], rtol=1e-15)

    def test_valid_vertices_are_untouched(self):
        rows = [
            [(4, 0.5), (1, 0.5)],  # 番号の順でなくても、正しければ並べ替えない
            [(0, 0.25), (1, 0.25), (2, 0.25), (3, 0.2505)],  # 合計 1.0005（幅の中）
            [(2, 1.0), (3, 0.0)],  # 重み 0 の枠も残す
            [(6, 0.3), (2, 0.7), (5, -2.0)],  # 形を動かさない骨（5・6）は数えない
        ]
        table = WeightTable.from_lists(rows)
        deform = np.array([True, True, True, True, True, False, False])
        fixed, changed = w.normalize_and_limit(table, deform=deform)
        self.assertFalse(changed[:3].any())
        np.testing.assert_array_equal(fixed.bone_ids[:3], table.bone_ids[:3])
        np.testing.assert_array_equal(fixed.weights[:3], table.weights[:3])
        # 頂点 3：形を動かす骨は 2 だけで重み 0.7 → 合計 1 に直す。形を動かさない骨は元の順で後ろへ
        self.assertTrue(changed[3])
        np.testing.assert_array_equal(fixed.bone_ids[3], [2, 6, 5, -1])
        np.testing.assert_array_equal(fixed.weights[3], [1.0, 0.3, -2.0, 0.0])

    def test_unweighted_vertices_are_untouched(self):
        table = WeightTable.from_lists([[(0, -0.5)], [(1, EPS)], [], [(2, 1.0)]])
        deform = np.array([True, True, False])
        fixed, changed = w.normalize_and_limit(table, deform=deform)
        self.assertFalse(changed.any())
        np.testing.assert_array_equal(fixed.weights, table.weights)
        flags = w.check(fixed, deform=deform)
        np.testing.assert_array_equal(flags["unweighted"], [True, True, True, True])

    def test_duplicates_and_negatives(self):
        table = WeightTable.from_lists([[(1, 0.25), (0, -0.5), (1, 0.25), (2, 0.5)]])
        fixed, changed = w.normalize_and_limit(table)
        np.testing.assert_array_equal(fixed.bone_ids, [[1, 2, -1, -1]])  # 0.5 と 0.5 → 番号の小さい順
        np.testing.assert_array_equal(fixed.weights, [[0.5, 0.5, 0.0, 0.0]])
        self.assertTrue(changed[0])

    def test_after_fix_everything_passes_and_fix_is_idempotent(self):
        table, deform = random_table(5, dyadic=False)
        fixed, changed = w.normalize_and_limit(table, deform=deform)
        flags = w.check(fixed, deform=deform)
        weighted = ~flags["unweighted"]
        for key in ("over_limit", "not_normalized", "negative", "duplicate_ids"):
            self.assertFalse((flags[key] & weighted).any(), key)
        np.testing.assert_allclose(w.sums(fixed, deform=deform)[weighted], 1.0, atol=1e-12)
        self.assertTrue((w.influence_counts(fixed, deform=deform) <= thresholds.MAX_BONE_INFLUENCES).all())
        again, changed_again = w.normalize_and_limit(fixed, deform=deform)
        self.assertFalse(changed_again.any())
        np.testing.assert_array_equal(again.weights, fixed.weights)
        before = w.check(table, deform=deform)
        should_fix = ~before["unweighted"] & (before["over_limit"] | before["not_normalized"] | before["negative"]
                                              | before["duplicate_ids"])
        np.testing.assert_array_equal(changed, should_fix)

    def test_small_blocks_give_the_same_answer(self):
        """頂点のかたまりの大きさを変えても、答えは同じ（かたまりのつなぎ目のまちがいを見つける）"""
        table, deform = random_table(6, count=101)
        fixed, changed = w.normalize_and_limit(table, deform=deform)
        flags = w.check(table, deform=deform)
        counts = w.influence_counts(table, deform=deform)
        totals = w.sums(table, deform=deform)
        summary = w.summarize(table, deform=deform)
        with mock.patch.object(w, "CHUNK_SIZE", 8):
            small_fixed, small_changed = w.normalize_and_limit(table, deform=deform)
            small_flags = w.check(table, deform=deform)
            np.testing.assert_array_equal(w.influence_counts(table, deform=deform), counts)
            np.testing.assert_array_equal(w.sums(table, deform=deform), totals)
            small_summary = w.summarize(table, deform=deform)
        np.testing.assert_array_equal(small_fixed.bone_ids, fixed.bone_ids)
        np.testing.assert_array_equal(small_fixed.weights, fixed.weights)
        np.testing.assert_array_equal(small_changed, changed)
        self.assertTrue(changed.any() and not changed.all())
        for key in w.CHECK_KEYS:
            np.testing.assert_array_equal(small_flags[key], flags[key])
        for big, small in zip(summary, small_summary):
            self.assertEqual((small.name, small.value, small.signal), (big.name, big.value, big.signal))
            self.assertEqual(small.details["indices"].tolist(), big.details["indices"].tolist())
        self.assertEqual(small_summary[3].details["min_weight"], summary[3].details["min_weight"])

    def test_empty_tables(self):
        for table in (WeightTable.from_lists([]), WeightTable.from_lists([[], []])):
            fixed, changed = w.normalize_and_limit(table)
            self.assertEqual(fixed.bone_ids.shape, table.bone_ids.shape)
            self.assertFalse(changed.any())

    def test_overflowing_sum_raises_instead_of_zero_weights(self):
        """足すと無限大になる重み。黙って重みを全部 0（どの骨にも付いていない点）にせず、ValueError にする"""
        for rows in ([[(0, 1e308), (1, 1e308)]], [[(0, 1e308), (0, 1e308), (1, 0.5)]]):
            table = WeightTable.from_lists(rows)
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                w.normalize_and_limit(table)

    def test_input_is_not_modified(self):
        table, deform = random_table(2)
        ids, weights = table.bone_ids.copy(), table.weights.copy()
        w.normalize_and_limit(table, deform=deform)
        np.testing.assert_array_equal(table.bone_ids, ids)
        np.testing.assert_array_equal(table.weights, weights)


# MARK: - 骨で形を動かす

def translation(offset) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = offset
    return matrix


class LinearBlendSkinningTest(unittest.TestCase):
    def setUp(self):
        self.rest = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, 0.5, 0.25]])

    def test_identity_returns_rest(self):
        table = WeightTable.from_lists([[(0, 1.0)], [(0, 0.5), (1, 0.5)], [(1, 0.25), (2, 0.75)]])
        result = w.linear_blend_skinning(self.rest, table, np.stack([np.eye(4)] * 3))
        np.testing.assert_array_equal(result, self.rest)

    def test_half_and_half_translation_is_average(self):
        table = WeightTable.from_lists([[(0, 0.5), (1, 0.5)]] * 3)
        matrices = np.stack([translation([1.0, 0.0, 0.0]), translation([0.0, 0.0, 2.0])])
        result = w.linear_blend_skinning(self.rest, table, matrices)
        np.testing.assert_allclose(result, self.rest + [0.5, 0.0, 1.0], rtol=0, atol=1e-15)

    def test_rotation_and_translation(self):
        angle = math.radians(90.0)
        matrix = np.eye(4)
        matrix[:3, :3] = [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0],
                          [0.0, 0.0, 1.0]]
        matrix[:3, 3] = [0.0, 0.0, 1.0]
        table = WeightTable.from_lists([[(1, 1.0)]] * 3)
        result = w.linear_blend_skinning(self.rest, table, np.stack([np.eye(4), matrix]))
        expected = np.column_stack([-self.rest[:, 1], self.rest[:, 0], self.rest[:, 2] + 1.0])
        np.testing.assert_allclose(result, expected, atol=1e-15)

    def test_weights_are_used_as_given(self):
        table = WeightTable.from_lists([[(0, 0.5)], [], [(0, 2.0)]])
        result = w.linear_blend_skinning(self.rest, table, np.stack([translation([1.0, 1.0, 1.0])]))
        np.testing.assert_array_equal(result[0], [0.5, 0.5, 0.5])
        np.testing.assert_array_equal(result[1], [0.0, 0.0, 0.0])  # 重みが無い頂点は (0, 0, 0)
        np.testing.assert_array_equal(result[2], 2.0 * (self.rest[2] + 1.0))

    def test_deform_mask_and_empty_slots(self):
        table = WeightTable(np.array([[0, -1, 1]] * 3), np.array([[1.0, 0.0, 1.0]] * 3))
        matrices = np.stack([translation([0.0, 1.0, 0.0]), translation([5.0, 0.0, 0.0])])
        result = w.linear_blend_skinning(self.rest, table, matrices, deform=np.array([True, False]))
        np.testing.assert_array_equal(result, self.rest + [0.0, 1.0, 0.0])

    def test_chunks_give_the_same_answer(self):
        rng = np.random.default_rng(8)
        rest = rng.normal(size=(50, 3))
        table, _ = random_table(3, count=50, width=5, bones=6, dyadic=False)
        matrices = np.stack([translation(rng.normal(size=3)) for _ in range(6)])
        matrices[:, :3, :3] = rng.normal(size=(6, 3, 3))
        whole = w.linear_blend_skinning(rest, table, matrices)
        np.testing.assert_array_equal(w.linear_blend_skinning(rest, table, matrices, chunk_size=7), whole)
        # 頂点ごとに、素直に計算した物と同じ
        for row in range(len(rest)):
            expected = np.zeros(3)
            for bone, weight in zip(table.bone_ids[row], table.weights[row]):
                if bone >= 0:
                    expected += weight * (matrices[bone] @ np.append(rest[row], 1.0))[:3]
            np.testing.assert_allclose(whole[row], expected, rtol=1e-12, atol=1e-12)

    def test_bad_inputs(self):
        table = WeightTable.from_lists([[(2, 1.0)]] * 3)
        with self.assertRaises(ValueError):
            w.linear_blend_skinning(self.rest, table, np.stack([np.eye(4)] * 2))  # 番号 2 が無い
        transposed = translation([1.0, 2.0, 3.0]).T
        with self.assertRaises(ValueError):
            w.linear_blend_skinning(self.rest, table, np.stack([np.eye(4), np.eye(4), transposed]))
        with self.assertRaises(ValueError):
            w.linear_blend_skinning(self.rest[:2], table, np.stack([np.eye(4)] * 3))
        with self.assertRaises(ValueError):
            w.linear_blend_skinning(self.rest, table, np.eye(4))
        with self.assertRaises(ValueError):
            w.linear_blend_skinning(self.rest, table, np.stack([np.eye(4)] * 3), deform=np.array([True]))
        with self.assertRaises(ValueError):
            w.linear_blend_skinning(self.rest, table, np.stack([np.eye(4)] * 3), chunk_size=0)
        nan_matrix = np.eye(4)
        nan_matrix[0, 0] = math.nan
        with self.assertRaises(ValueError):
            w.linear_blend_skinning(self.rest, table, np.stack([np.eye(4), np.eye(4), nan_matrix]))

    def test_empty(self):
        result = w.linear_blend_skinning(np.zeros((0, 3)), WeightTable.from_lists([]), np.zeros((0, 4, 4)))
        self.assertEqual(result.shape, (0, 3))


# MARK: - 信号

class SummarizeTest(unittest.TestCase):
    def signals(self, results):
        return {result.name: (result.signal, result.value) for result in results}

    def test_all_green(self):
        table = WeightTable.from_lists([[(0, 1.0)], [(0, 0.5), (1, 0.5)]])
        results = w.summarize(table)
        self.assertEqual([result.name for result in results],
                         ["bone_influences_over_limit", "unweighted_vertices", "weights_not_normalized",
                          "negative_weights", "duplicate_bone_ids"])
        for result in results:
            self.assertEqual((result.signal, result.value, result.unit), (Signal.GREEN, 0.0, "count"))

    def test_signals(self):
        table = WeightTable.from_lists([
            [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2), (4, 0.2)],
            [],
            [],
            [(0, 0.5)],
            [(0, 1.5), (1, -0.5)],
            [(2, 0.5), (2, 0.5)],
        ])
        signals = self.signals(w.summarize(table))
        self.assertEqual(signals["bone_influences_over_limit"], (Signal.RED, 1.0))
        self.assertEqual(signals["unweighted_vertices"], (Signal.RED, 2.0))
        self.assertEqual(signals["weights_not_normalized"], (Signal.YELLOW, 1.0))
        self.assertEqual(signals["negative_weights"], (Signal.RED, 1.0))
        self.assertEqual(signals["duplicate_bone_ids"], (Signal.YELLOW, 1.0))
        details = {result.name: result.details for result in w.summarize(table)}
        np.testing.assert_array_equal(details["unweighted_vertices"]["indices"], [1, 2])
        self.assertEqual(details["bone_influences_over_limit"]["max_influences"], 5)
        self.assertEqual(details["negative_weights"]["min_weight"], -0.5)
        self.assertAlmostEqual(details["weights_not_normalized"]["max_deviation"], 0.5)

    def test_limit_and_deform_are_used(self):
        table = WeightTable.from_lists([[(0, 0.5), (1, 0.5)]])
        self.assertEqual(self.signals(w.summarize(table, limit=1))["bone_influences_over_limit"][0], Signal.RED)
        signals = self.signals(w.summarize(table, deform=np.array([True, False])))
        self.assertEqual(signals["weights_not_normalized"], (Signal.YELLOW, 1.0))

    def test_summary_after_fix(self):
        table, deform = random_table(1)
        fixed, _ = w.normalize_and_limit(table, deform=deform)
        signals = self.signals(w.summarize(fixed, deform=deform))
        self.assertEqual(signals["bone_influences_over_limit"][0], Signal.GREEN)
        self.assertEqual(signals["weights_not_normalized"][0], Signal.GREEN)
        self.assertEqual(signals["duplicate_bone_ids"][0], Signal.GREEN)

    def test_empty_is_gray(self):
        for result in w.summarize(WeightTable.from_lists([])):
            self.assertEqual(result.signal, Signal.GRAY)
            self.assertTrue(result.reason)


if __name__ == "__main__":
    unittest.main()

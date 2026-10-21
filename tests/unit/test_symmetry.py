"""A4 左右のずれ（measure/symmetry.py）を確かめる"""

import math
import unittest

import numpy as np

from blender_ai_tools import thresholds
from blender_ai_tools.measure.results import Signal
from blender_ai_tools.measure.symmetry import asymmetry, mirror_x, nearest_distances, summarize_asymmetry
from tests import shapes


class MirrorTest(unittest.TestCase):
    def test_negates_only_x_and_keeps_input(self):
        points = np.array([[0.25, -1.0, 2.0], [-3.0, 0.5, 0.0]])
        before = points.copy()
        np.testing.assert_array_equal(mirror_x(points), [[-0.25, -1.0, 2.0], [3.0, 0.5, 0.0]])
        np.testing.assert_array_equal(points, before)
        self.assertEqual(mirror_x([]).shape, (0, 3))
        with self.assertRaises(ValueError):
            mirror_x([[np.nan, 0.0, 0.0]])


class NearestDistancesTest(unittest.TestCase):
    def naive(self, query, reference):
        distances, indices = [], []
        for q in query:
            row = [math.dist(q, r) for r in reference]
            best = min(range(len(row)), key=lambda i: (row[i], i))
            distances.append(row[best])
            indices.append(best)
        return np.array(distances), np.array(indices)

    def test_matches_naive_loop(self):
        rng = np.random.default_rng(2024)
        query = rng.normal(size=(120, 3))
        reference = rng.normal(size=(90, 3))
        expected_d, expected_i = self.naive(query, reference)
        for chunk in (1, 7, 89, 90, 91, 10 ** 6):
            with self.subTest(chunk=chunk):
                distances, indices = nearest_distances(query, reference, chunk_size=chunk)
                np.testing.assert_allclose(distances, expected_d, rtol=1e-14, atol=0)
                np.testing.assert_array_equal(indices, expected_i)
                self.assertEqual(indices.dtype, np.int64)

    def test_ties_choose_lowest_index(self):
        reference = np.array([[5.0, 0, 0], [1.0, 0, 0], [-1.0, 0, 0], [1.0, 0, 0]])
        for chunk in (1, 2, 3, 100):
            distances, indices = nearest_distances([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], reference, chunk_size=chunk)
            np.testing.assert_array_equal(indices, [1, 1])
            np.testing.assert_array_equal(distances, [1.0, 0.0])

    def test_single_reference_and_empty(self):
        distances, indices = nearest_distances([[0, 0, 0], [3, 4, 0]], [[0, 0, 0]])
        np.testing.assert_array_equal(distances, [0.0, 5.0])
        np.testing.assert_array_equal(indices, [0, 0])
        distances, indices = nearest_distances(np.zeros((0, 3)), [[0, 0, 0]])
        self.assertEqual(distances.shape, (0,))
        self.assertEqual(indices.dtype, np.int64)
        with self.assertRaises(ValueError):
            nearest_distances([[0, 0, 0]], np.zeros((0, 3)))
        with self.assertRaises(ValueError):
            nearest_distances([[0, 0, 0]], [[0, 0, 0]], chunk_size=True)
        with self.assertRaises(ValueError):
            nearest_distances([[0, 0]], [[0, 0, 0]])


class AsymmetryTest(unittest.TestCase):
    def test_symmetric_box_is_zero(self):
        vertices, _ = shapes.box(size=(1.0, 2.0, 3.0), center=(0.0, 0.5, -1.0))
        np.testing.assert_array_equal(asymmetry(vertices), np.zeros(8))

    def test_box_off_center_is_not_symmetric(self):
        # X にずらした箱は、左右が合わない（Y・Z にずらしても合う）。軸をまちがえると、ここで分かる
        vertices, _ = shapes.box(size=(1.0, 1.0, 1.0), center=(0.1, 0.0, 0.0))
        np.testing.assert_allclose(asymmetry(vertices), np.full(8, 0.2), atol=1e-15)
        vertices, _ = shapes.box(size=(1.0, 1.0, 1.0), center=(0.0, 0.3, 0.7))
        np.testing.assert_array_equal(asymmetry(vertices), np.zeros(8))

    def test_points_on_mirror_plane_are_zero(self):
        np.testing.assert_array_equal(asymmetry([[0.0, 1.0, 2.0], [0.0, -1.0, 0.5]]), [0.0, 0.0])

    def test_shifted_vertex(self):
        vertices, _ = shapes.box()
        vertices[1, 0] += 0.002  # 頂点 1（+X 側）だけを外へ 2mm
        result = asymmetry(vertices)
        expected = np.zeros(8)
        expected[[0, 1]] = 0.002  # 頂点 1 と、その相手の頂点 0
        np.testing.assert_allclose(result, expected, atol=1e-15)
        summary = summarize_asymmetry(result)
        self.assertEqual(summary.signal, Signal.YELLOW)
        self.assertAlmostEqual(summary.value, 0.002)
        np.testing.assert_array_equal(summary.details["worst_indices"], [0, 1])

    def test_exclude_intentional_asymmetry(self):
        vertices, _ = shapes.box()
        ponytail = np.vstack([vertices, [[0.8, 0.0, 0.2]]])  # 片側だけの髪
        without = asymmetry(ponytail)
        self.assertAlmostEqual(without[8], math.dist((-0.8, 0.0, 0.2), (-0.5, -0.5, 0.5)))
        self.assertEqual(summarize_asymmetry(without).signal, Signal.RED)
        exclude = np.zeros(9, dtype=bool)
        exclude[8] = True
        result = asymmetry(ponytail, exclude=exclude)
        self.assertTrue(math.isnan(result[8]))
        np.testing.assert_array_equal(result[:8], np.zeros(8))
        summary = summarize_asymmetry(result)
        self.assertEqual(summary.signal, Signal.GREEN)
        self.assertEqual(summary.details["excluded_count"], 1)

    def test_excluded_vertex_is_not_a_target(self):
        # 頂点 1 を除くと、頂点 0 の相手がいなくなる（除いた頂点を相手に使っていないこと）
        vertices, _ = shapes.box()
        exclude = np.zeros(8, dtype=bool)
        exclude[1] = True
        result = asymmetry(vertices, exclude=exclude)
        self.assertAlmostEqual(result[0], 1.0)
        self.assertTrue(math.isnan(result[1]))

    def test_all_excluded_and_invalid(self):
        self.assertTrue(np.all(np.isnan(asymmetry([[1.0, 0, 0]], exclude=np.array([True])))))
        self.assertEqual(asymmetry(np.zeros((0, 3))).shape, (0,))
        with self.assertRaises(ValueError):
            asymmetry([[1.0, 0, 0]], exclude=np.array([1]))  # 0/1 の整数は受け付けない
        with self.assertRaises(ValueError):
            asymmetry([[1.0, 0, 0]], exclude=np.array([True, False]))


class SummarizeAsymmetryTest(unittest.TestCase):
    Y, R = thresholds.ASYMMETRY_YELLOW, thresholds.ASYMMETRY_RED

    def test_thresholds(self):
        result = summarize_asymmetry([0.0, np.nextafter(self.Y, 0)])
        self.assertEqual(result.name, "asymmetry")
        self.assertEqual(result.unit, "m")
        self.assertEqual(result.signal, Signal.GREEN)
        self.assertEqual(summarize_asymmetry([self.Y]).signal, Signal.YELLOW)
        self.assertEqual(summarize_asymmetry([self.R, np.nan]).signal, Signal.RED)

    def test_counts_and_worst(self):
        values = [self.R, 0.0, self.Y, np.nan, 2 * self.R, self.R]
        result = summarize_asymmetry(values)
        self.assertEqual(result.value, 2 * self.R)
        self.assertEqual(result.details["over_yellow_count"], 4)
        self.assertEqual(result.details["over_red_count"], 3)
        self.assertEqual(result.details["excluded_count"], 1)
        np.testing.assert_array_equal(result.details["worst_indices"], [4, 0, 5, 2])

    def test_gray_and_invalid(self):
        self.assertEqual(summarize_asymmetry([]).signal, Signal.GRAY)
        all_nan = summarize_asymmetry([np.nan, np.nan])
        self.assertEqual(all_nan.signal, Signal.GRAY)
        self.assertTrue(all_nan.reason)
        for bad in ([-0.1], [np.inf], [[0.1]], ["x"]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                summarize_asymmetry(bad)


if __name__ == "__main__":
    unittest.main()

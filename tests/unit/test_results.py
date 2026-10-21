"""測った結果の共通の形（信号）を確かめる"""

import math
import unittest

from blender_ai_tools.measure.results import Measurement, Signal, classify, worst


class SignalTest(unittest.TestCase):
    def test_worst_order(self):
        self.assertEqual(worst([Signal.GREEN, Signal.YELLOW]), Signal.YELLOW)
        self.assertEqual(worst([Signal.YELLOW, Signal.GRAY]), Signal.GRAY)
        self.assertEqual(worst([Signal.GRAY, Signal.RED, Signal.GREEN]), Signal.RED)
        self.assertEqual(worst([]), Signal.GRAY)

    def test_labels(self):
        self.assertEqual([s.label for s in Signal], ["緑", "黄", "灰", "赤"])

    def test_classify_higher_is_worse(self):
        self.assertEqual(classify(0.0, yellow=1.0, red=2.0), Signal.GREEN)
        self.assertEqual(classify(1.0, yellow=1.0, red=2.0), Signal.YELLOW)
        self.assertEqual(classify(2.0, yellow=1.0, red=2.0), Signal.RED)
        self.assertEqual(classify(5.0, red=2.0), Signal.RED)
        self.assertEqual(classify(1.5, red=2.0), Signal.GREEN)

    def test_classify_lower_is_worse(self):
        self.assertEqual(classify(0.9, yellow=0.7, red=0.5, higher_is_worse=False), Signal.GREEN)
        self.assertEqual(classify(0.6, yellow=0.7, red=0.5, higher_is_worse=False), Signal.YELLOW)
        self.assertEqual(classify(0.5, yellow=0.7, red=0.5, higher_is_worse=False), Signal.RED)

    def test_missing_is_gray(self):
        self.assertEqual(classify(None, red=1.0), Signal.GRAY)
        self.assertEqual(classify(math.nan, red=1.0), Signal.GRAY)


class MeasurementTest(unittest.TestCase):
    def test_missing_value_must_be_gray(self):
        with self.assertRaises(ValueError):
            Measurement("x", None, "m", Signal.GREEN)
        with self.assertRaises(ValueError):
            Measurement("x", math.nan, "m", Signal.RED)

    def test_gray_needs_reason(self):
        with self.assertRaises(ValueError):
            Measurement("x", None, "m", Signal.GRAY)
        self.assertEqual(Measurement.unmeasured("x", "m", "理由").signal, Signal.GRAY)

    def test_from_value(self):
        self.assertEqual(Measurement.from_value("x", 3.0, "m", red=2.0).signal, Signal.RED)
        gray = Measurement.from_value("x", None, "m", red=2.0)
        self.assertEqual(gray.signal, Signal.GRAY)
        self.assertTrue(gray.reason)


class MeasurementDataTest(unittest.TestCase):
    """JSON にする形（to_data / from_data）"""

    def test_round_trip(self):
        original = Measurement("penetration_depth", 0.0012, "m", Signal.RED, "深い",
                               {"count": 3, "worst": [1, 2], "nested": {"ok": True, "none": None}})
        data = original.to_data()
        self.assertEqual(data["signal"], "red")
        self.assertEqual(data["value"], 0.0012)
        self.assertEqual(Measurement.from_data(data), original)
        self.assertEqual(Measurement.from_data(Measurement.unmeasured("x", "count", "理由").to_data()).signal,
                         Signal.GRAY)

    def test_numpy_and_tuples_become_plain(self):
        import numpy as np

        details = {"mask": np.array([True, False]), "ids": np.arange(3, dtype=np.int64), "pair": (1.5, 2.5),
                   "scalar": np.float64(0.25), "nan": math.nan, "signal": Signal.YELLOW}
        data = Measurement("x", 1.0, "m", Signal.GREEN, details=details).to_data()
        self.assertEqual(data["details"], {"mask": [True, False], "ids": [0, 1, 2], "pair": [1.5, 2.5],
                                           "scalar": 0.25, "nan": None, "signal": "yellow"})
        self.assertEqual(Measurement("x", 1.0, "m", Signal.GREEN, details=details).to_data(include_details=False)["details"], {})

    def test_unconvertible_details_raise(self):
        with self.assertRaises(TypeError):
            Measurement("x", 1.0, "m", Signal.GREEN, details={"bad": {1, 2}}).to_data()
        with self.assertRaises(TypeError):
            Measurement("x", 1.0, "m", Signal.GREEN, details={1: "キーが数"}).to_data()

    def test_from_data_rejects_bad_shapes(self):
        good = Measurement("x", 1.0, "m", Signal.GREEN).to_data()
        for bad in ([], {**good, "signal": "blue"}, {**good, "signal": 3}, {**good, "extra": 1},
                    {**good, "value": "1"}, {**good, "value": True}, {**good, "details": []},
                    {**good, "value": None}):  # 値が無いのに緑は、Measurement 自身が断る
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Measurement.from_data(bad)

if __name__ == "__main__":
    unittest.main()

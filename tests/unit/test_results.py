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


if __name__ == "__main__":
    unittest.main()

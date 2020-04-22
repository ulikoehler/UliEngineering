#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from UliEngineering.Physics.Pressure import *
from UliEngineering.EngineerIO import auto_format
import unittest
import math

class TestPressureConversion(unittest.TestCase):
    def test_pascal_to_bar(self):
        self.assertAlmostEqual(pascal_to_bar(0.), 0)
        self.assertAlmostEqual(pascal_to_bar(1.), 1/100000)
        self.assertAlmostEqual(pascal_to_bar(5.), 5/100000)
        self.assertAlmostEqual(pascal_to_bar(100000), 1)

    def test_bar_to_pascal(self):
        self.assertAlmostEqual(bar_to_pascal(0.), 0)
        self.assertAlmostEqual(bar_to_pascal(1.), 100000)
        self.assertAlmostEqual(bar_to_pascal(5.), 500000)
        self.assertAlmostEqual(bar_to_pascal(0.00001), 1)
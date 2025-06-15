#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from UliEngineering.Physics.Pressure import *
import unittest

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
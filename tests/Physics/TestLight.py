#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from UliEngineering.Physics.Light import *
from UliEngineering.EngineerIO import auto_format
import unittest

class TestJohnsonNyquistNoise(unittest.TestCase):
    def test_lumen_to_candela_by_apex_angle(self):
        v = lumen_to_candela_by_apex_angle("25 lm", "120°")
        assert_approx_equal(v, 7.9577471546, significant=5)
        self.assertEqual(auto_format(lumen_to_candela_by_apex_angle, "25 lm", "120°"), "7.96 cd")


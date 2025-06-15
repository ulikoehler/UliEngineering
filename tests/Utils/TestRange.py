#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from UliEngineering.Utils.Range import *
import unittest

class TestValueRange(unittest.TestCase):
    def testConstructorUnit(self):
        vr = ValueRange(-1.0, 1.0, "A")
        self.assertEqual(str(vr), "ValueRange('-1.000 A', '1.000 A')")
        assert_approx_equal(vr.min, -1.0)
        assert_approx_equal(vr.max, 1.0)
        self.assertEqual(vr.unit, "A")

    def testConstructorNoUnit(self):
        vr = ValueRange(-1.0, 1.0)
        self.assertEqual(str(vr), "ValueRange('-1.000', '1.000')")
        assert_approx_equal(vr.min, -1.0)
        assert_approx_equal(vr.max, 1.0)
        self.assertEqual(vr.unit, None)


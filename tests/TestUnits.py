#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from parameterized import parameterized
from UliEngineering.Units import *
import unittest

class TestSubUnit(unittest.TestCase):
    def test_eq(self):
        self.assertEqual(SubUnit.parse("V"), SubUnit("V"))
        self.assertEqual(SubUnit.parse("V"), SubUnit("V", 1))
        self.assertEqual(SubUnit.parse("V^2"), SubUnit("V", 2))
        self.assertEqual(SubUnit.parse("V²"), SubUnit("V", 2))
        self.assertEqual(SubUnit.parse("V^3"), SubUnit("V", 3))
        # Not equal: Unit
        self.assertNotEqual(SubUnit.parse("V"), SubUnit("A"))
        self.assertNotEqual(SubUnit.parse("V²"), SubUnit("A", 2))
        # Not equal: power
        self.assertNotEqual(SubUnit.parse("V^2"), SubUnit("V"))
        self.assertNotEqual(SubUnit.parse("V^2"), SubUnit("V", 3))
        self.assertNotEqual(SubUnit.parse("V²"), SubUnit("V"))
        self.assertNotEqual(SubUnit.parse("V²"), SubUnit("V", 1))
        self.assertNotEqual(SubUnit.parse("V²"), SubUnit("V", 3))
        self.assertNotEqual(SubUnit.parse("V^3"), SubUnit("V"))
        self.assertNotEqual(SubUnit.parse("V^3"), SubUnit("V", 2))
        self.assertNotEqual(SubUnit.parse("V^3"), SubUnit("V", 4))

class TestUnit(unittest.TestCase):
    def test_construct(self):
        u = Unit("V")
        self.assertEqual(u.numerator, [SubUnit("V")])
        self.assertEqual(u.denominator, [])
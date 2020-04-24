#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from parameterized import parameterized
from UliEngineering.Quantity import *
from UliEngineering.Units import UnknownUnitInContextException
import unittest

class TestQuantity(unittest.TestCase):
    def test_construct(self):
        q = Quantity(1.23, "V")
        self.assertEqual(q.value, 1.23)
        self.assertEqual(q.unit, "V")
        self.assertEqual(q.__repr__(), "1.23 V")

    def test_parse(self):
        "Test whether a Quantity can be parsed from a string"
        q = Quantity("1.23 mV")
        self.assertEqual(q.value, 1.23e-3)
        self.assertEqual(q.unit, "V")
        self.assertEqual(q.__repr__(), "1.23 mV")

    def test_abs(self):
        self.assertEqual(abs(Quantity("-1.23 mV")), Quantity("1.23 mV"))

    def test_equality(self):
        "Test whether a Quantity can be parsed from a string"
        q = Quantity("1.23 mV")
        self.assertEqual(q, q) # Self-identity equal
        self.assertEqual(q, Quantity("1.23 mV")) # Self-non-identity equal
        self.assertEqual(q, Quantity(1.23e-3, "V")) # Self-non-identity equal, alt constructor
        self.assertEqual(q, 1.23e-3)
        self.assertEqual(q, "1.23 mV")
        # Inequality
        self.assertNotEqual(q, Quantity("2.34 mV"))
        self.assertNotEqual(q, Quantity(2.34e-3, "V"))
        self.assertNotEqual(q, 2.34e-3)
        self.assertNotEqual(q, -1.23e-3)
        self.assertNotEqual(q, "2.34 mV")


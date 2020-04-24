#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from parameterized import parameterized
from UliEngineering.Quantity import *
from UliEngineering.Units import UnknownUnitInContextException, InvalidUnitCombinationException
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
        self.assertTrue(q != Quantity("2.34 mV")) # Direct non equality
        self.assertNotEqual(q, Quantity("2.34 mV"))
        self.assertNotEqual(q, Quantity(2.34e-3, "V"))
        self.assertNotEqual(q, 2.34e-3)
        self.assertNotEqual(q, -1.23e-3)
        self.assertNotEqual(q, "2.34 mV")

    def test_numerical_operators(self):
        "Test whether a Quantity can be parsed from a string"
        # Quantity-Quantity comparisons
        self.assertLess(Quantity("1.23 mV"), Quantity("2.34 mV"))
        self.assertLessEqual(Quantity("1.23 mV"), Quantity("2.34 mV"))
        self.assertLessEqual(Quantity("1.23 mV"), Quantity("1.23 mV"))
        self.assertGreater(Quantity("9.23 mV"), Quantity("2.34 mV"))
        self.assertGreaterEqual(Quantity("9.23 mV"), Quantity("2.34 mV"))
        self.assertGreaterEqual(Quantity("9.23 mV"), Quantity("9.23 mV"))
        # Quantity-number comparisons
        self.assertLess(Quantity("1.23 mV"), 2.34e-3)
        self.assertLessEqual(Quantity("1.23 mV"), 2.34e-3)
        self.assertLessEqual(Quantity("1.23 mV"), 1.23e-3)
        self.assertGreater(Quantity("9.23 mV"), 1.23e-3)
        self.assertGreaterEqual(Quantity("9.23 mV"), 2.34e-3)
        self.assertGreaterEqual(Quantity("9.23 mV"), 9.23e-3)

    @parameterized.expand([
        ("1.23 A", ),
        ("99 °C", ),
    ])
    def test_numerical_operators_invalid_combination(self, arg):
        with self.assertRaises(InvalidUnitCombinationException):
            Quantity("1.23 V") < Quantity(arg)
        with self.assertRaises(InvalidUnitCombinationException):
            Quantity("1.23 V") > Quantity(arg)
        with self.assertRaises(InvalidUnitCombinationException):
            Quantity("1.23 V") <= Quantity(arg)
        with self.assertRaises(InvalidUnitCombinationException):
            Quantity("1.23 V") >= Quantity(arg)
        # Inverted order
        with self.assertRaises(InvalidUnitCombinationException):
            Quantity(arg) < Quantity("1.23 V") 
        with self.assertRaises(InvalidUnitCombinationException):
            Quantity(arg) > Quantity("1.23 V") 
        with self.assertRaises(InvalidUnitCombinationException):
            Quantity(arg) <= Quantity("1.23 V") 
        with self.assertRaises(InvalidUnitCombinationException):
            Quantity(arg) >= Quantity("1.23 V") 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from nose.tools import assert_equal, assert_true, raises
from parameterized import parameterized
from UliEngineering.Length import *
from UliEngineering.Units import UnknownUnitInContextException

class TestLengths(object):
    def test_length_normalization(self):
        assert_approx_equal(normalize_length(1.0), 1.0)
        assert_approx_equal(normalize_length("1.0 m"), 1.0)
        assert_approx_equal(normalize_length("1.0 meter"), 1.0)
        assert_approx_equal(normalize_length("5.0 m"), 5.0)
        assert_approx_equal(normalize_length("5.0 meters"), 5.0)
        assert_approx_equal(normalize_length("3 m"), 3)
        assert_approx_equal(normalize_length("1.0 mm"), 1e-3)
        assert_approx_equal(normalize_length("1.0 nm"), 1e-9) # NO, NOT nautical mile!
        assert_approx_equal(normalize_length("2.0 mil"), 2*2.54e-5)
        assert_approx_equal(normalize_length("1.0 in"), 0.0254)
        assert_approx_equal(normalize_length("1.0 \""), 0.0254)
        assert_approx_equal(normalize_length("1.0 inch"), 0.0254)
        assert_approx_equal(normalize_length("1.0 inches"), 0.0254)
        assert_approx_equal(normalize_length("3 ft"), 0.9144)
        assert_approx_equal(normalize_length("3 foot"), 0.9144)
        assert_approx_equal(normalize_length("3 feet"), 0.9144)
        assert_approx_equal(normalize_length("7 yd"), 6.4008)
        assert_approx_equal(normalize_length("7 yard"), 6.4008)
        assert_approx_equal(normalize_length("3.7 mile"), 5954.5728)
        assert_approx_equal(normalize_length("3.7 miles"), 5954.5728)
        assert_approx_equal(normalize_length("0.77 nautical mile"), 1426.04)
        assert_approx_equal(normalize_length("0.77 nautical miles"), 1426.04)
        assert_approx_equal(normalize_length("18 pt"), 0.00635)
        assert_approx_equal(normalize_length("18 point"), 0.00635)
        assert_approx_equal(normalize_length("18 points"), 0.00635)
        assert_approx_equal(normalize_length("1.2 ly"), 1.135287656709696e+16)
        assert_approx_equal(normalize_length("1.2 lightyear"), 1.135287656709696e+16)
        assert_approx_equal(normalize_length("1.2 light year"), 1.135287656709696e+16)
        assert_approx_equal(normalize_length("1.2 lightyears"), 1.135287656709696e+16)
        assert_approx_equal(normalize_length("1.2 light years"), 1.135287656709696e+16)
        assert_approx_equal(normalize_length("1.2 M light years"), 1.135287656709696e+22)
        assert_approx_equal(normalize_length("1.2 kly"), 1.135287656709696e+19)
        assert_approx_equal(normalize_length("1.2 Mly"), 1.135287656709696e+22)
        assert_approx_equal(normalize_length("5.5 AU"), 149597870700*5.5)
        assert_approx_equal(normalize_length("5.5 AUs"), 149597870700*5.5)
        assert_approx_equal(normalize_length("5.5 au"), 149597870700*5.5)
        assert_approx_equal(normalize_length("9.15 pc"), 2.8233949868947424e+17)
        assert_approx_equal(normalize_length("9.15 kpc"), 2.8233949868947424e+20)
        assert_approx_equal(normalize_length("3.33 Å"), 3.33e-10)
        assert_approx_equal(normalize_length("3.33 Angstrom"), 3.33e-10)
        assert_approx_equal(normalize_length("3.33 angstrom"), 3.33e-10)

    @parameterized([
        ("1A"),
        ("xaz"),
        ("yxard"),
    ])
    @raises(ValueError)
    def test_invalid_unit(self, unit):
        normalize_length("6.6 {}".format(unit))

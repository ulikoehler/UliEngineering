#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_allclose, assert_approx_equal
from UliEngineering.Physics.Mass import normalize_mass_grams
import unittest


class TestMassNormalization(unittest.TestCase):
    def test_normalize_mass_grams(self):
        assert_approx_equal(normalize_mass_grams("500 g"), 500.0)
        assert_approx_equal(normalize_mass_grams("0.5 kg"), 500.0)
        assert_approx_equal(normalize_mass_grams("1000 mg"), 1.0)

    def test_normalize_mass_iterables(self):
        assert_allclose(normalize_mass_grams(["500 g", "0.5 kg"]), [500.0, 500.0])
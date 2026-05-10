#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_allclose, assert_approx_equal
from UliEngineering.Physics.Density import normalize_density_kg_per_m3
import unittest


class TestDensityNormalization(unittest.TestCase):
    def test_normalize_density(self):
        assert_approx_equal(normalize_density_kg_per_m3("1000 kg/m^3"), 1000.0)
        assert_approx_equal(normalize_density_kg_per_m3("1 g/cm^3"), 1000.0)
        assert_approx_equal(normalize_density_kg_per_m3("1000 g/L"), 1000.0)

    def test_normalize_density_iterables(self):
        assert_allclose(normalize_density_kg_per_m3(["1 g/cm^3", "2 g/cm^3"]), [1000.0, 2000.0])
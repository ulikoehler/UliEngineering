#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from parameterized import parameterized
from UliEngineering.EngineerIO.Volume import *
import unittest
import numpy as np
import scipy.constants

from UliEngineering.Units import UnknownUnitInContextException

class TestVolume(unittest.TestCase):
    def setUp(self):
        self.volume_io = EngineerVolumeIO()

    def test_volume_normalization_basic(self):
        # Basic numeric values
        assert_approx_equal(normalize_volume(1.0), 1.0)
        assert_approx_equal(normalize_volume(5.0), 5.0)
        assert_approx_equal(normalize_volume(3), 3)
        
        # Test with class instance
        assert_approx_equal(self.volume_io.normalize_volume(1.0), 1.0)
        assert_approx_equal(self.volume_io.normalize_volume(5.0), 5.0)
        assert_approx_equal(self.volume_io.normalize_volume(3), 3)
        
        # None handling
        self.assertIsNone(normalize_volume(None))

    def test_volume_normalization_metric_units(self):
        # Test SI prefixes with cubic meters
        assert_approx_equal(normalize_volume("1 km³"), 1e9)
        assert_approx_equal(self.volume_io.normalize_volume("1 km³"), 1e9)
        assert_approx_equal(normalize_volume("1 mm³"), 1e-9)
        assert_approx_equal(self.volume_io.normalize_volume("1 mm³"), 1e-9)
        assert_approx_equal(normalize_volume("1 cm³"), 1e-6)
        assert_approx_equal(self.volume_io.normalize_volume("1 cm³"), 1e-6)
        assert_approx_equal(normalize_volume("1 dm³"), 1e-3)
        assert_approx_equal(self.volume_io.normalize_volume("1 dm³"), 1e-3)

    def test_volume_normalization_imperial_units(self):
        # Imperial units
        assert_approx_equal(normalize_volume("1 in³"), scipy.constants.inch**3)
        assert_approx_equal(self.volume_io.normalize_volume("1 in³"), scipy.constants.inch**3)
        assert_approx_equal(normalize_volume("1 ft³"), scipy.constants.foot**3)
        assert_approx_equal(self.volume_io.normalize_volume("1 ft³"), scipy.constants.foot**3)
        assert_approx_equal(normalize_volume("1 yd³"), scipy.constants.yard**3)
        assert_approx_equal(self.volume_io.normalize_volume("1 yd³"), scipy.constants.yard**3)
        
        # Test alias forms
        assert_approx_equal(normalize_volume("1 cubic inch"), scipy.constants.inch**3)
        assert_approx_equal(self.volume_io.normalize_volume("1 cubic foot"), scipy.constants.foot**3)
        assert_approx_equal(normalize_volume("1 cu ft"), scipy.constants.foot**3)

    def test_volume_normalization_liquid_units(self):
        # Liquid volume units
        assert_approx_equal(normalize_volume("1 L"), 0.001)
        assert_approx_equal(self.volume_io.normalize_volume("1 L"), 0.001)
        assert_approx_equal(normalize_volume("1 liter"), 0.001)
        assert_approx_equal(self.volume_io.normalize_volume("1 litre"), 0.001)
        assert_approx_equal(normalize_volume("1 gal"), 0.003785411784)
        assert_approx_equal(self.volume_io.normalize_volume("1 gallon"), 0.003785411784)
        assert_approx_equal(normalize_volume("1 qt"), 0.000946352946)
        assert_approx_equal(self.volume_io.normalize_volume("1 quart"), 0.000946352946)
        assert_approx_equal(normalize_volume("1 pt"), 0.000473176473)
        assert_approx_equal(self.volume_io.normalize_volume("1 pint"), 0.000473176473)
        assert_approx_equal(normalize_volume("1 fl oz"), 2.95735296875e-05)
        assert_approx_equal(self.volume_io.normalize_volume("1 fluid ounce"), 2.95735296875e-05)

    def test_volume_normalization_scientific_units(self):
        # Scientific units
        assert_approx_equal(normalize_volume("1.0 Å³"), 1e-30)
        assert_approx_equal(self.volume_io.normalize_volume("1.0 Å³"), 1e-30)
        assert_approx_equal(normalize_volume("1.0 angstrom cubed"), 1e-30)
        assert_approx_equal(self.volume_io.normalize_volume("1.0 A^3"), 1e-30)
        
        # Bohr radius cubed
        bohr_volume = (scipy.constants.physical_constants['Bohr radius'][0])**3
        assert_approx_equal(normalize_volume("1.0 bohr³"), bohr_volume)
        assert_approx_equal(self.volume_io.normalize_volume("1.0 atomic unit of volume"), bohr_volume)

    def test_volume_normalization_astronomical_units(self):
        # Astronomical units
        au_volume = (scipy.constants.au)**3
        assert_approx_equal(normalize_volume("1.0 AU³"), au_volume)
        assert_approx_equal(self.volume_io.normalize_volume("1.0 astronomical unit cubed"), au_volume)
        
        pc_volume = (scipy.constants.parsec)**3
        assert_approx_equal(normalize_volume("1.0 pc³"), pc_volume)
        assert_approx_equal(self.volume_io.normalize_volume("1.0 parsec cubed"), pc_volume)
        
        ly_volume = (scipy.constants.c * scipy.constants.Julian_year)**3
        assert_approx_equal(normalize_volume("1.0 ly³"), ly_volume)
        assert_approx_equal(self.volume_io.normalize_volume("1.0 light year cubed"), ly_volume)

    def test_volume_normalization_prefixed_liquid_units(self):
        # Test SI prefixes with liters
        assert_approx_equal(normalize_volume("1 mL"), 1e-6)
        assert_approx_equal(self.volume_io.normalize_volume("1 mL"), 1e-6)
        assert_approx_equal(normalize_volume("1 µL"), 1e-9)
        assert_approx_equal(self.volume_io.normalize_volume("1 µL"), 1e-9)
        assert_approx_equal(normalize_volume("1 kL"), 1.0)
        assert_approx_equal(self.volume_io.normalize_volume("1 kL"), 1.0)

    def test_volume_normalization_lists(self):
        # Test list input
        input_list = ["1 m³", "1000 cm³", "1 ft³"]
        result = normalize_volume(input_list)
        result_class = self.volume_io.normalize_volume(input_list)
        expected = [1.0, 0.001, scipy.constants.foot**3]
        for i, val in enumerate(expected):
            assert_approx_equal(result[i], val)
            assert_approx_equal(result_class[i], val)

    def test_volume_normalization_numpy_arrays(self):
        # Test numpy array input
        input_array = np.array(["1 m³", "1000 cm³", "1 ft³"])
        result = normalize_volume(input_array)
        result_class = self.volume_io.normalize_volume(input_array)
        expected = np.array([1.0, 0.001, scipy.constants.foot**3])
        np.testing.assert_array_almost_equal(result, expected)
        np.testing.assert_array_almost_equal(result_class, expected)

    def test_convert_volume_to_cubic_meters_basic(self):
        # Basic conversions
        assert_approx_equal(convert_volume_to_cubic_meters(1.0, "m³"), 1.0)
        assert_approx_equal(self.volume_io.convert_volume_to_cubic_meters(1.0, "m³"), 1.0)
        assert_approx_equal(convert_volume_to_cubic_meters(1.0, "cubic meter"), 1.0)
        assert_approx_equal(self.volume_io.convert_volume_to_cubic_meters(1.0, "cubic meter"), 1.0)
        assert_approx_equal(convert_volume_to_cubic_meters(5.0, "m³"), 5.0)
        assert_approx_equal(self.volume_io.convert_volume_to_cubic_meters(5.0, "m³"), 5.0)
        assert_approx_equal(convert_volume_to_cubic_meters(3, "m³"), 3)
        assert_approx_equal(self.volume_io.convert_volume_to_cubic_meters(3, "m³"), 3)

    def test_convert_volume_to_cubic_meters_metric(self):
        # Metric units
        assert_approx_equal(convert_volume_to_cubic_meters(1.0, "mm³"), 1e-9)
        assert_approx_equal(self.volume_io.convert_volume_to_cubic_meters(1.0, "mm³"), 1e-9)
        assert_approx_equal(convert_volume_to_cubic_meters(1.0, "cm³"), 1e-6)
        assert_approx_equal(self.volume_io.convert_volume_to_cubic_meters(1.0, "cm³"), 1e-6)
        assert_approx_equal(convert_volume_to_cubic_meters(1.0, "dm³"), 1e-3)
        assert_approx_equal(self.volume_io.convert_volume_to_cubic_meters(1.0, "dm³"), 1e-3)
        assert_approx_equal(convert_volume_to_cubic_meters(1.0, "km³"), 1e9)
        assert_approx_equal(self.volume_io.convert_volume_to_cubic_meters(1.0, "km³"), 1e9)
        assert_approx_equal(convert_volume_to_cubic_meters(1000000000, "mm³"), 1.0)
        assert_approx_equal(self.volume_io.convert_volume_to_cubic_meters(1000000000, "mm³"), 1.0)
        assert_approx_equal(convert_volume_to_cubic_meters(1000000, "cm³"), 1.0)
        assert_approx_equal(self.volume_io.convert_volume_to_cubic_meters(1000000, "cm³"), 1.0)

    def test_convert_volume_to_cubic_meters_liquid(self):
        # Liquid units
        assert_approx_equal(convert_volume_to_cubic_meters(1.0, "L"), 0.001)
        assert_approx_equal(self.volume_io.convert_volume_to_cubic_meters(1.0, "L"), 0.001)
        assert_approx_equal(convert_volume_to_cubic_meters(1000.0, "L"), 1.0)
        assert_approx_equal(self.volume_io.convert_volume_to_cubic_meters(1000.0, "L"), 1.0)
        assert_approx_equal(convert_volume_to_cubic_meters(1.0, "gal"), 0.003785411784)
        assert_approx_equal(self.volume_io.convert_volume_to_cubic_meters(1.0, "gal"), 0.003785411784)

    def test_special_alias_cc(self):
        # Test the special "cc" alias for cubic centimeters
        assert_approx_equal(normalize_volume("1 cc"), 1e-6)
        assert_approx_equal(self.volume_io.normalize_volume("1 cc"), 1e-6)
        assert_approx_equal(normalize_volume("500 cc"), 0.0005)
        assert_approx_equal(self.volume_io.normalize_volume("500 cc"), 0.0005)

    def test_oil_and_gas_units(self):
        # Oil and gas industry units
        assert_approx_equal(normalize_volume("1 bbl"), 0.158987294928)
        assert_approx_equal(self.volume_io.normalize_volume("1 barrel"), 0.158987294928)
        assert_approx_equal(normalize_volume("42 bbl"), 42 * 0.158987294928)
        assert_approx_equal(self.volume_io.normalize_volume("42 barrels"), 42 * 0.158987294928)

    def test_kitchen_units(self):
        # Kitchen/cooking units
        assert_approx_equal(normalize_volume("1 cup"), 0.0002365882365)
        assert_approx_equal(self.volume_io.normalize_volume("1 cup"), 0.0002365882365)
        assert_approx_equal(normalize_volume("1 tbsp"), 1.47867648437e-05)
        assert_approx_equal(self.volume_io.normalize_volume("1 tablespoon"), 1.47867648437e-05)
        assert_approx_equal(normalize_volume("1 tsp"), 4.92892161458e-06)
        assert_approx_equal(self.volume_io.normalize_volume("1 teaspoon"), 4.92892161458e-06)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from parameterized import parameterized
from UliEngineering.EngineerIO.Length import *
import unittest

class TestLength(unittest.TestCase):
    def setUp(self):
        self.length_io = EngineerLengthIO()

    def test_length_normalization(self):
        # Test with functions
        assert_approx_equal(normalize_length(1.0), 1.0)
        assert_approx_equal(normalize_length("1.0 m"), 1.0)
        assert_approx_equal(normalize_length("1.0 meter"), 1.0)
        
        # Test with class instance
        assert_approx_equal(self.length_io.normalize_length(1.0), 1.0)
        assert_approx_equal(self.length_io.normalize_length("1.0 m"), 1.0)
        assert_approx_equal(self.length_io.normalize_length("1.0 meter"), 1.0)
        
        assert_approx_equal(normalize_length("5.0 m"), 5.0)
        assert_approx_equal(normalize_length("5.0 meters"), 5.0)
        assert_approx_equal(normalize_length("3 m"), 3)
        assert_approx_equal(normalize_length("1.0 mm"), 1e-3)
        assert_approx_equal(normalize_length("1.0 cm"), 1e-2)
        assert_approx_equal(normalize_length("2.5 dm"), 0.25)
        assert_approx_equal(normalize_length("1.5 km"), 1500.0)
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
    
    def test_convert_length_to_meters(self):
        # Test with functions
        assert_approx_equal(convert_length_to_meters(1.0, "m"), 1.0)
        assert_approx_equal(convert_length_to_meters(1.0, "meter"), 1.0)
        
        # Test with class instance
        assert_approx_equal(self.length_io.convert_length_to_meters(1.0, "m"), 1.0)
        assert_approx_equal(self.length_io.convert_length_to_meters(1.0, "meter"), 1.0)
        
        assert_approx_equal(convert_length_to_meters(5.0, "m"), 5.0)
        assert_approx_equal(convert_length_to_meters(5.0, "meters"), 5.0)
        assert_approx_equal(convert_length_to_meters(3, "m"), 3)
        assert_approx_equal(convert_length_to_meters(1.0, "mm"), 1e-3)
        assert_approx_equal(convert_length_to_meters(1.0, "cm"), 1e-2)
        assert_approx_equal(convert_length_to_meters(2.5, "dm"), 0.25)
        assert_approx_equal(convert_length_to_meters(1.5, "km"), 1500.0)
        assert_approx_equal(convert_length_to_meters(1.0, "nm"), 1e-9) # NO, NOT nautical mile!
        assert_approx_equal(convert_length_to_meters(2.0, "mil"), 2*2.54e-5)
        assert_approx_equal(convert_length_to_meters(1.0, "in"), 0.0254)
        assert_approx_equal(convert_length_to_meters(1.0, "\""), 0.0254)
        assert_approx_equal(convert_length_to_meters(1.0, "inch"), 0.0254)
        assert_approx_equal(convert_length_to_meters(1.0, "inches"), 0.0254)
        assert_approx_equal(convert_length_to_meters(3, "ft"), 0.9144)
        assert_approx_equal(convert_length_to_meters(3, "foot"), 0.9144)
        assert_approx_equal(convert_length_to_meters(3, "feet"), 0.9144)
        assert_approx_equal(convert_length_to_meters(7, "yd"), 6.4008)
        assert_approx_equal(convert_length_to_meters(7, "yard"), 6.4008)
        assert_approx_equal(convert_length_to_meters(3.7, "mile"), 5954.5728)
        assert_approx_equal(convert_length_to_meters(3.7, "miles"), 5954.5728)
        assert_approx_equal(convert_length_to_meters(0.77, "nautical mile"), 1426.04)
        assert_approx_equal(convert_length_to_meters(0.77, "nautical miles"), 1426.04)
        assert_approx_equal(convert_length_to_meters(18, "pt"), 0.00635)
        assert_approx_equal(convert_length_to_meters(18, "point"), 0.00635)
        assert_approx_equal(convert_length_to_meters(18, "points"), 0.00635)
        assert_approx_equal(convert_length_to_meters(1.2, "ly"), 1.135287656709696e+16)
        assert_approx_equal(convert_length_to_meters(1.2, "lightyear"), 1.135287656709696e+16)
        assert_approx_equal(convert_length_to_meters(1.2, "light year"), 1.135287656709696e+16)
        assert_approx_equal(convert_length_to_meters(1.2, "lightyears"), 1.135287656709696e+16)
        assert_approx_equal(convert_length_to_meters(1.2, "light years"), 1.135287656709696e+16)
        assert_approx_equal(convert_length_to_meters(1.2, "M light years"), 1.135287656709696e+22)
        assert_approx_equal(convert_length_to_meters(1.2, "kly"), 1.135287656709696e+19)
        assert_approx_equal(convert_length_to_meters(1.2, "Mly"), 1.135287656709696e+22)
        assert_approx_equal(convert_length_to_meters(5.5, "AU"), 149597870700*5.5)
        assert_approx_equal(convert_length_to_meters(5.5, "AUs"), 149597870700*5.5)
        assert_approx_equal(convert_length_to_meters(5.5, "au"), 149597870700*5.5)
        assert_approx_equal(convert_length_to_meters(9.15, "pc"), 2.8233949868947424e+17)
        assert_approx_equal(convert_length_to_meters(9.15, "kpc"), 2.8233949868947424e+20)
        assert_approx_equal(convert_length_to_meters(3.33, "Å"), 3.33e-10)
        assert_approx_equal(convert_length_to_meters(3.33, "Angstrom"), 3.33e-10)
        assert_approx_equal(convert_length_to_meters(3.33, "angstrom"), 3.33e-10)

    def test_class_instance_vs_function_consistency(self):
        """Test that class methods and functions produce identical results"""
        test_cases = [
            "1.0 m", "100 mm", "1 ft", "1 inch", "1 ly", "1 AU"
        ]
        
        for case in test_cases:
            with self.subTest(case=case):
                function_result = normalize_length(case)
                class_result = self.length_io.normalize_length(case)
                assert_approx_equal(function_result, class_result, 
                                  err_msg=f"Results differ for {case}")

    def test_class_convert_consistency(self):
        """Test that class convert method and function produce identical results"""
        test_cases = [
            (1.0, "m"), (100.0, "mm"), (1.0, "ft"), (1.0, "inch")
        ]
        
        for value, unit in test_cases:
            with self.subTest(value=value, unit=unit):
                function_result = convert_length_to_meters(value, unit)
                class_result = self.length_io.convert_length_to_meters(value, unit)
                assert_approx_equal(function_result, class_result, 
                                  err_msg=f"Results differ for {value} {unit}")

    @parameterized.expand([
        ("1A",),
        ("xaz",),
        ("yxard",),
    ])
    def test_invalid_unit(self, unit):
        with self.assertRaises(ValueError):
            normalize_length("6.6 {}".format(unit))
        with self.assertRaises(ValueError):
            self.length_io.normalize_length("6.6 {}".format(unit))

    def test_none_handling(self):
        """Test handling of None input"""
        with self.assertRaises(ValueError):
            normalize_length(None)

    def test_list_input(self):
        """Test list input handling"""
        input_list = ["1 m", "100 mm", "1 ft"]
        result_func = normalize_length(input_list)
        result_class = self.length_io.normalize_length(input_list)
        
        expected = [1.0, 0.1, 0.3048]
        for i, val in enumerate(expected):
            assert_approx_equal(result_func[i], val)
            assert_approx_equal(result_class[i], val)

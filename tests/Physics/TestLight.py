#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from UliEngineering.Physics.Light import (
    lumen_to_candela_by_apex_angle,
    normalize_luminous_flux, LuminousFluxLumen,
    normalize_angle_degrees, AngleDegrees
)
from UliEngineering.EngineerIO import auto_format
import unittest

class TestLight(unittest.TestCase):
    def test_lumen_to_candela_by_apex_angle(self):
        v = lumen_to_candela_by_apex_angle("25 lm", "120°")
        assert_approx_equal(v, 7.9577471546, significant=5)
        self.assertEqual(auto_format(lumen_to_candela_by_apex_angle, "25 lm", "120°"), "7.96 cd")

    def test_type_annotations_exist(self):
        """Test that the new type annotations are available."""
        self.assertIsNotNone(LuminousFluxLumen)
        self.assertIsNotNone(AngleDegrees)

    def test_normalize_luminous_flux_various_units(self):
        """Test normalize_luminous_flux with various unit inputs."""
        test_cases = [
            ("1 lm", 1.0),
            ("1 lumen", 1.0),
            ("1 lumens", 1.0),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_luminous_flux(input_val)
                assert_approx_equal(result, expected)

    def test_normalize_angle_degrees_various_units(self):
        """Test normalize_angle_degrees with various unit inputs."""
        test_cases = [
            ("1 °", 1.0),
            ("1 deg", 1.0),
            ("1 degree", 1.0),
            ("1 degrees", 1.0),
            ("1 rad", 57.29577951308232),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_angle_degrees(input_val)
                assert_approx_equal(result, expected)

    def test_lumen_to_candela_various_units(self):
        """Test lumen_to_candela_by_apex_angle with various unit inputs."""
        # Test with different luminous flux units
        v1 = lumen_to_candela_by_apex_angle("25 lumen", "120°")
        v2 = lumen_to_candela_by_apex_angle("25 lm", "120°")
        assert_approx_equal(v1, v2)

        # Test with different angle units
        v3 = lumen_to_candela_by_apex_angle("25 lm", "120 deg")
        v4 = lumen_to_candela_by_apex_angle("25 lm", "120°")
        assert_approx_equal(v3, v4)


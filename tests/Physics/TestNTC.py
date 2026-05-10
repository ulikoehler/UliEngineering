#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from UliEngineering.Physics.NTC import (
    ntc_resistance, ntc_resistances,
    normalize_resistance, ResistanceOhm,
    TemperatureKelvin
)
import unittest

class TestNTC(unittest.TestCase):
    def test_ntc_resistance(self):
        # Values arbitrarily from Murata NCP15WB473D03RC
        assert_approx_equal(ntc_resistance("47k", "4050K", "25°C"), 47000)
        assert_approx_equal(ntc_resistance("47k", "4050K", "0°C"), 162942.79)
        assert_approx_equal(ntc_resistance("47k", "4050K", "-18°C"), 463773.791)
        assert_approx_equal(ntc_resistance("47k", "4050K", "5°C"), 124819.66)
        assert_approx_equal(ntc_resistance("47k", "4050K", "60°C"), 11280.407)

    def test_ntc_resistances(self):
        # Currently mostly test if it runs
        _, values = ntc_resistances("47k", "4050K")

    def test_type_annotations_exist(self):
        """Test that the new type annotations are available."""
        self.assertIsNotNone(ResistanceOhm)
        self.assertIsNotNone(TemperatureKelvin)

    def test_normalize_resistance_various_units(self):
        """Test normalize_resistance with various unit inputs."""
        test_cases = [
            ("1 Ω", 1.0),
            ("1 Ohm", 1.0),
            ("1 ohm", 1.0),
            ("1 kΩ", 1000.0),
            ("1 MΩ", 1e6),
            ("1 GΩ", 1e9),
            ("1 k", 1000.0),
            ("1 M", 1e6),
            ("1 G", 1e9),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_resistance(input_val)
                assert_approx_equal(result, expected)

    def test_ntc_resistance_various_units(self):
        """Test ntc_resistance with various resistance units."""
        # Test with different resistance units
        r1 = ntc_resistance("47k", "4050K", "25°C")
        r2 = ntc_resistance("47000 Ω", "4050K", "25°C")
        assert_approx_equal(r1, r2)

        # Test with k shorthand
        r3 = ntc_resistance("47k", "4050K", "25°C")
        r4 = ntc_resistance("47 k", "4050K", "25°C")
        assert_approx_equal(r3, r4)

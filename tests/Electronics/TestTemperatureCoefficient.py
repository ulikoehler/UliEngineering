#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from UliEngineering.Electronics.TemperatureCoefficient import *
from UliEngineering.Utils.Range import ValueRange
import unittest
import numpy as np

class TestTemperatureCoefficient(unittest.TestCase):
    def test_value_range_over_temperature_zero(self):
        # Test with simple ppm input
        self.assertEqual(str(value_range_over_temperature("1 kΩ", "0 ppm")),
            str(ValueRange(1000, 1000, "Ω"))
        )

    def test_value_range_over_temperature1(self):
        # Test with simple ppm input
        # Wolfram Alpha verified: "1kOhm-(65*100ppm*1kOhm)" (0.9935 rounded to 0.994), "1kOhm+(60*100ppm*1kOhm)"
        self.assertEqual(str(value_range_over_temperature("1 kΩ", "100 ppm")),
            str(ValueRange(994, 1006, "Ω"))
        )

    def test_value_range_over_temperature_percent(self):
        # Wolfram Alpha verified: "(1-(1%*65))*1.00", "(1+(1%*60))*1.00"
        self.assertEqual(str(value_range_over_temperature("1 kΩ", "1 %")),
            str(ValueRange(350, 1600, "Ω"))
        )
        # Test with slightly different percentage input (not enough to change the result after formatting)
        self.assertEqual(str(value_range_over_temperature("1 kΩ", "1.006 %")),
            str(ValueRange(346, 1604, "Ω"))
        )
    
    def test_value_range_over_temperature_tolerance(self):
        # Test with +- the same ppm input#
        # Wolfram Alpha verified: "(1+100ppm*60)*1.0*101%", "(1-(100ppm*65))*1.00*99%"
        self.assertEqual(str(value_range_over_temperature("1 kΩ", "100 ppm", tolerance="1%")),
            str(ValueRange(984, 1016, "Ω"))
        )

    def test_value_range_over_temperature_custom_temp_range(self):
        # Test with custom temperature range
        result = value_range_over_temperature("1 kΩ", "100 ppm", tmin="0 °C", tmax="50 °C")
        self.assertEqual(str(result), str(ValueRange(997.5, 1002.5, "Ω")))
        
    def test_value_range_over_temperature_custom_tref(self):
        # Test with custom reference temperature
        result = value_range_over_temperature("1 kΩ", "100 ppm", tref="20 °C")
        self.assertEqual(str(result), str(ValueRange(993.5, 1006.5, "Ω")))

    def test_value_range_over_temperature_negative_coefficient(self):
        # Test with negative temperature coefficient
        result = value_range_over_temperature("1 kΩ", "-100 ppm")
        self.assertEqual(str(result), str(ValueRange(994, 1006, "Ω")))

    def test_value_range_over_temperature_kelvin_input(self):
        # Test with Kelvin temperature inputs
        result = value_range_over_temperature("1 kΩ", "100 ppm", tmin="233 K", tmax="358 K", tref="298 K")
        self.assertEqual(str(result), str(ValueRange(994, 1006, "Ω")))

    def test_value_range_over_temperature_numeric_coefficient(self):
        # Test with numeric coefficient (not string)
        result = value_range_over_temperature("1 kΩ", 100e-6)
        self.assertEqual(str(result), str(ValueRange(994, 1006, "Ω")))

class TestValueAtTemperature(unittest.TestCase):
    def test_value_at_temperature_basic(self):
        # Ref temp => zero difference
        assert_approx_equal(value_at_temperature("1 kΩ", "25 °C", "100 ppm"), 1000.0)
        # delta T = 10° => 10 * 100 ppm
        assert_approx_equal(value_at_temperature("1 kΩ", "35 °C", "100 ppm"), 1001.)
        # delta T = -10° => -10 * 100 ppm
        assert_approx_equal(value_at_temperature("1 kΩ", "15 °C", "100 ppm"), 999.)

    def test_value_at_temperature_different_units(self):
        # Test with different units for nominal value
        assert_approx_equal(value_at_temperature("1000 Ω", "35 °C", "100 ppm"), 1001.)
        assert_approx_equal(value_at_temperature("1 MΩ", "35 °C", "100 ppm"), 1001000.)
        assert_approx_equal(value_at_temperature("1 mΩ", "35 °C", "100 ppm"), 0.001001)

    def test_value_at_temperature_kelvin_input(self):
        # Test with Kelvin temperature inputs
        assert_approx_equal(value_at_temperature("1 kΩ", "298 K", "100 ppm", tref="298 K"), 1000.0)
        assert_approx_equal(value_at_temperature("1 kΩ", "308 K", "100 ppm", tref="298 K"), 1001.0)
        assert_approx_equal(value_at_temperature("1 kΩ", "288 K", "100 ppm", tref="298 K"), 999.0)

    def test_value_at_temperature_numeric_inputs(self):
        # Test with numeric temperature inputs (interpreted as °C)
        assert_approx_equal(value_at_temperature(1000, 25, 100e-6), 1000.0)
        assert_approx_equal(value_at_temperature(1000, 35, 100e-6), 1001.0)
        assert_approx_equal(value_at_temperature(1000, 15, 100e-6), 999.0)

    def test_value_at_temperature_percent_coefficient(self):
        # Test with percentage coefficient
        assert_approx_equal(value_at_temperature("1 kΩ", "35 °C", "0.01 %"), 1001.0)
        assert_approx_equal(value_at_temperature("1 kΩ", "15 °C", "0.01 %"), 999.0)

    def test_value_at_temperature_negative_coefficient(self):
        # Test with negative temperature coefficient
        assert_approx_equal(value_at_temperature("1 kΩ", "35 °C", "-100 ppm"), 999.0)
        assert_approx_equal(value_at_temperature("1 kΩ", "15 °C", "-100 ppm"), 1001.0)

    def test_value_at_temperature_custom_tref(self):
        # Test with custom reference temperature
        assert_approx_equal(value_at_temperature("1 kΩ", "30 °C", "100 ppm", tref="20 °C"), 1001.0)
        assert_approx_equal(value_at_temperature("1 kΩ", "10 °C", "100 ppm", tref="20 °C"), 999.0)

    def test_value_at_temperature_extreme_temperatures(self):
        # Test with extreme temperatures
        assert_approx_equal(value_at_temperature("1 kΩ", "-40 °C", "100 ppm"), 993.5)
        assert_approx_equal(value_at_temperature("1 kΩ", "125 °C", "100 ppm"), 1010.0)

    def test_value_at_temperature_zero_coefficient(self):
        # Test with zero temperature coefficient
        assert_approx_equal(value_at_temperature("1 kΩ", "35 °C", "0 ppm"), 1000.0)
        assert_approx_equal(value_at_temperature("1 kΩ", "15 °C", "0 ppm"), 1000.0)
        assert_approx_equal(value_at_temperature("1 kΩ", "-40 °C", "0 ppm"), 1000.0)

    def test_value_at_temperature_large_coefficient(self):
        # Test with large temperature coefficient
        assert_approx_equal(value_at_temperature("1 kΩ", "35 °C", "1000 ppm"), 1010.0)
        assert_approx_equal(value_at_temperature("1 kΩ", "35 °C", "1 %"), 1100.0)

    def test_value_at_temperature_arrays(self):
        # Test with numpy arrays if supported
        temperatures = np.array([15, 25, 35])
        expected = np.array([999.0, 1000.0, 1001.0])
        
        results = []
        for temp in temperatures:
            results.append(value_at_temperature("1 kΩ", temp, "100 ppm"))
        
        assert_allclose(results, expected, rtol=1e-10)

    def test_value_at_temperature_realistic_scenarios(self):
        # Test realistic component scenarios
        
        # Precision resistor: 25 ppm/°C, -40°C to +85°C
        r_cold = value_at_temperature("10 kΩ", "-40 °C", "25 ppm")
        r_hot = value_at_temperature("10 kΩ", "85 °C", "25 ppm")
        self.assertAlmostEqual(r_cold, 9983.75, places=2)
        self.assertAlmostEqual(r_hot, 10015.0, places=1)
        
        # Crystal oscillator: -20 ppm/°C
        f_cold = value_at_temperature("32.768 kHz", "0 °C", "-20 ppm")
        f_hot = value_at_temperature("32.768 kHz", "50 °C", "-20 ppm")
        self.assertAlmostEqual(f_cold, 32.784384e3, places=6)
        self.assertAlmostEqual(f_hot, 32.751616e3, places=6)
        
        # Capacitor: +150 ppm/°C
        c_cold = value_at_temperature("100 nF", "-25 °C", "150 ppm")
        c_hot = value_at_temperature("100 nF", "75 °C", "150 ppm")
        self.assertAlmostEqual(c_cold * 1e9, 99.25, places=3)  # Convert to nF
        self.assertAlmostEqual(c_hot * 1e9, 100.75, places=2)   # Convert to nF

class TestTemperatureCoefficientEdgeCases(unittest.TestCase):
    def test_invalid_inputs(self):
        # Test error handling for invalid inputs
        with self.assertRaises(ValueError):
            value_range_over_temperature(None, "100 ppm")
            
        # Test with empty string should work (might be interpreted as 0)
        # This depends on the normalize function behavior
        
    def test_extreme_coefficients(self):
        # Test with very large coefficients
        result = value_at_temperature("1 kΩ", "35 °C", "10 %")
        self.assertAlmostEqual(result, 2000.0, places=1)
        
        # Test with very small coefficients
        result = value_at_temperature("1 kΩ", "35 °C", "1 ppb")
        self.assertAlmostEqual(result, 1000.00001, places=8)

    def test_coefficient_units_consistency(self):
        # Test that different ways of expressing the same coefficient give same results
        result_ppm = value_at_temperature("1 kΩ", "35 °C", "100 ppm")
        result_percent = value_at_temperature("1 kΩ", "35 °C", "0.01 %")
        result_numeric = value_at_temperature("1 kΩ", "35 °C", 100e-6)
        
        assert_approx_equal(result_ppm, result_percent)
        assert_approx_equal(result_ppm, result_numeric)

    def test_temperature_range_boundary_conditions(self):
        # Test at exactly the reference temperature
        result = value_range_over_temperature("1 kΩ", "100 ppm", tmin="25 °C", tmax="25 °C")
        self.assertEqual(str(result), str(ValueRange(1000, 1000, "Ω")))
        
        # Test with inverted temperature range (tmin > tmax)
        result = value_range_over_temperature("1 kΩ", "100 ppm", tmin="85 °C", tmax="-40 °C")
        # Should still work correctly (min/max will be computed properly)
        self.assertIsInstance(result, ValueRange)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_allclose, assert_approx_equal
from scipy.constants import elementary_charge, k as boltzmann_k
from UliEngineering.Electronics.Diode import (
    DiodeModel,
    ShockleyDiodeModel,
    SimpleDiodeModel,
    diode_thermal_voltage,
    normalize_diode_model,
    shockley_diode_current,
    shockley_diode_power,
    shockley_diode_saturation_current,
    shockley_diode_small_signal_resistance,
    shockley_diode_voltage,
    normalize_power, PowerW,
)
from UliEngineering.EngineerIO import auto_format
import numpy as np
import unittest


class TestDiode(unittest.TestCase):
    def test_diode_model_base_class(self):
        model = DiodeModel()
        with self.assertRaises(NotImplementedError):
            model.minimum_series_voltage()
        with self.assertRaises(NotImplementedError):
            model.forward_voltage(1e-3)
        with self.assertRaises(NotImplementedError):
            model.series_current(1.0, 1000.0)
        with self.assertRaises(NotImplementedError):
            model.series_current_integral(1.0, 1000.0)

    def test_normalize_diode_model(self):
        default_model = normalize_diode_model(None)
        self.assertIsInstance(default_model, SimpleDiodeModel)
        self.assertAlmostEqual(default_model.forward_voltage_drop, 0.0, places=12)

        normalized = normalize_diode_model("700mV")
        self.assertIsInstance(normalized, SimpleDiodeModel)
        self.assertAlmostEqual(normalized.forward_voltage_drop, 0.7, places=12)

        shockley = ShockleyDiodeModel("1pA")
        self.assertIs(normalize_diode_model(shockley), shockley)

    def test_simple_diode_model(self):
        model = SimpleDiodeModel("700mV")
        self.assertAlmostEqual(model.minimum_series_voltage(), 0.7, places=12)
        self.assertAlmostEqual(model.forward_voltage("1mA"), 0.7, places=12)
        self.assertEqual(model.forward_voltage(0), 0.0)
        self.assertAlmostEqual(model.series_current(2.7, 1000.0), 0.002, places=12)
        self.assertEqual(model.series_current(0.7, 1000.0), 0.0)
        self.assertFalse(np.isfinite(model.series_current_integral(0.7, 1000.0)))

    def test_shockley_diode_model(self):
        model = ShockleyDiodeModel("1pA", ideality_factor=1.5, temperature="35°C")
        expected_voltage = shockley_diode_voltage("2mA", "1pA", ideality_factor=1.5, temperature="35°C")
        self.assertEqual(model.minimum_series_voltage(), 0.0)
        self.assertAlmostEqual(model.forward_voltage("2mA"), expected_voltage, places=12)
        total_voltage = 1.0
        resistance = 470.0
        current = model.series_current(total_voltage, resistance)
        residual = total_voltage - (current * resistance + model.forward_voltage(current))
        self.assertAlmostEqual(residual, 0.0, places=12)

    def test_diode_thermal_voltage(self):
        expected = boltzmann_k * 298.15 / elementary_charge
        assert_approx_equal(diode_thermal_voltage("25°C"), expected)
        assert_approx_equal(diode_thermal_voltage("298.15K"), expected)

    def test_diode_thermal_voltage_auto_format(self):
        self.assertEqual(auto_format(diode_thermal_voltage, "25°C"), "25.7 mV")

    def test_shockley_diode_current(self):
        voltage = 0.7
        saturation_current = 1e-12
        thermal_voltage = boltzmann_k * 298.15 / elementary_charge
        expected = saturation_current * np.expm1(voltage / thermal_voltage)
        assert_approx_equal(shockley_diode_current(voltage, saturation_current), expected)
        assert_approx_equal(shockley_diode_current("700mV", "1pA"), expected)

    def test_shockley_diode_current_reverse_bias(self):
        current = shockley_diode_current("-100mV", "1pA")
        self.assertLess(current, 0)
        self.assertGreater(current, -1e-12)

    def test_shockley_diode_current_numpy_arrays(self):
        voltages = np.asarray([0.6, 0.7])
        currents = shockley_diode_current(voltages, 1e-12)
        expected = 1e-12 * np.expm1(voltages / diode_thermal_voltage("25°C"))
        assert_allclose(currents, expected)

    def test_shockley_diode_current_invalid_inputs(self):
        with self.assertRaises(ValueError):
            shockley_diode_current("700mV", 0)
        with self.assertRaises(ValueError):
            shockley_diode_current("700mV", "1pA", ideality_factor=0)

    def test_shockley_diode_voltage(self):
        current = 1e-3
        saturation_current = 1e-12
        thermal_voltage = boltzmann_k * 298.15 / elementary_charge
        expected = thermal_voltage * np.log1p(current / saturation_current)
        assert_approx_equal(shockley_diode_voltage(current, saturation_current), expected)
        assert_approx_equal(shockley_diode_voltage("1mA", "1pA"), expected)

    def test_shockley_diode_voltage_round_trip(self):
        original_voltage = 0.68
        saturation_current = 2e-12
        ideality_factor = 1.8
        temperature = "50°C"
        current = shockley_diode_current(original_voltage, saturation_current, ideality_factor, temperature)
        recovered_voltage = shockley_diode_voltage(current, saturation_current, ideality_factor, temperature)
        self.assertAlmostEqual(recovered_voltage, original_voltage, places=12)

    def test_shockley_diode_voltage_invalid_current(self):
        with self.assertRaises(ValueError):
            shockley_diode_voltage(-1e-12, 1e-12)

    def test_shockley_diode_saturation_current(self):
        saturation_current = 5e-12
        voltage = 0.65
        current = shockley_diode_current(voltage, saturation_current, ideality_factor=1.4, temperature="35°C")
        recovered = shockley_diode_saturation_current(voltage, current, ideality_factor=1.4, temperature="35°C")
        self.assertAlmostEqual(recovered, saturation_current, places=18)

    def test_shockley_diode_saturation_current_reverse_bias(self):
        saturation_current = shockley_diode_saturation_current("-100mV", shockley_diode_current("-100mV", "1pA"))
        self.assertAlmostEqual(saturation_current, 1e-12, places=18)

    def test_shockley_diode_saturation_current_invalid_voltage(self):
        with self.assertRaises(ValueError):
            shockley_diode_saturation_current(0, 1e-3)

    def test_shockley_diode_small_signal_resistance(self):
        expected = diode_thermal_voltage("25°C") / 1e-3
        assert_approx_equal(shockley_diode_small_signal_resistance("1mA"), expected)
        self.assertEqual(auto_format(shockley_diode_small_signal_resistance, "1mA"), "25.7 Ω")

    def test_shockley_diode_small_signal_resistance_zero_current(self):
        self.assertTrue(np.isinf(shockley_diode_small_signal_resistance(0)))

    def test_shockley_diode_power(self):
        voltage = 0.7
        saturation_current = 1e-12
        expected = voltage * shockley_diode_current(voltage, saturation_current)
        assert_approx_equal(shockley_diode_power(voltage, saturation_current), expected)
        self.assertEqual(auto_format(shockley_diode_power, "700mV", "1pA"), "476 mW")

    def test_shockley_diode_temperature_dependency(self):
        cold_voltage = shockley_diode_voltage("1mA", "1pA", temperature="0°C")
        hot_voltage = shockley_diode_voltage("1mA", "1pA", temperature="100°C")
        self.assertLess(cold_voltage, hot_voltage)

class TestNormalizeFunctions(unittest.TestCase):
    def test_type_annotations_exist(self):
        """Test that the new type annotations are available."""
        self.assertIsNotNone(PowerW)

    def test_normalize_power_various_units(self):
        """Test normalize_power with various unit inputs."""
        test_cases = [
            ("1 W", 1.0),
            ("1 mW", 1e-3),
            ("1 µW", 1e-6),
            ("1 kW", 1e3),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_power(input_val)
                self.assertAlmostEqual(result, expected)

    def test_diode_functions_various_units(self):
        """Test diode functions with various unit inputs."""
        # Test shockley_diode_current with different units
        i1 = shockley_diode_current("1 V", "1 A")
        i2 = shockley_diode_current("1000 mV", "1000 mA")
        self.assertAlmostEqual(i1, i2)

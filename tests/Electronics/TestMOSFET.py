#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from UliEngineering.Electronics.MOSFET import mosfet_gate_charge_losses, mosfet_gate_capacitance_from_gate_charge, normalize_charge, ChargeC
from UliEngineering.Electronics.Diode import VoltageV
from UliEngineering.Electronics.Filter import FrequencyHz
import unittest

class TestLEDSeriesResistors(unittest.TestCase):
    def test_mosfet_gate_charge_losses(self):
        # Example verified at http://www.elektronik-kompendium.de/sites/bau/1109111.htm
        # Also verified at https://www.digikey.com/en/resources/conversion-calculators/conversion-calculator-led-series-resistor
        assert_approx_equal(mosfet_gate_charge_losses(39.0e-9, 10, 300e3), 0.117)
        assert_approx_equal(mosfet_gate_charge_losses("39nC", "10V", "300 kHz"), 0.117)

    def test_mosfet_gate_capacitance_from_gate_charge(self):
        assert_approx_equal(mosfet_gate_capacitance_from_gate_charge(39.0e-9, 10), 3.9e-9)
        assert_approx_equal(mosfet_gate_capacitance_from_gate_charge("39nC", "10V"), 3.9e-9)

class TestNormalizeFunctions(unittest.TestCase):
    def test_type_annotations_exist(self):
        """Test that the new type annotations are available"""
        self.assertIsNotNone(ChargeC)
        self.assertIsNotNone(VoltageV)
        self.assertIsNotNone(FrequencyHz)

    def test_normalize_charge_various_units(self):
        """Test normalize_charge with various unit inputs"""
        test_cases = [
            ("1 C", 1.0),
            ("1 mC", 1e-3),
            ("1 µC", 1e-6),
            ("1 nC", 1e-9),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_charge(input_val)
                self.assertAlmostEqual(result, expected)

    def test_mosfet_functions_various_units(self):
        """Test MOSFET functions with various unit inputs"""
        # Test with different unit representations
        p1 = mosfet_gate_charge_losses("39 nC", "10 V", "300 kHz")
        p2 = mosfet_gate_charge_losses("0.039 µC", "10000 mV", "0.3 MHz")
        self.assertAlmostEqual(p1, p2)

        # Test capacitance with different units
        c1 = mosfet_gate_capacitance_from_gate_charge("39 nC", "10 V")
        c2 = mosfet_gate_capacitance_from_gate_charge("0.039 µC", "10000 mV")
        self.assertAlmostEqual(c1, c2)

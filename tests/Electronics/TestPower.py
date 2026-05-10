#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest

from numpy.testing import assert_allclose

from UliEngineering.Electronics.Power import current_by_power, power_by_current_and_voltage
from UliEngineering.Electronics.Diode import PowerW, CurrentA, VoltageV


class TestPower(unittest.TestCase):
    def test_current_by_power(self):
        assert_allclose(current_by_power(25), 25 / 230, atol=1e-15)
        assert_allclose(current_by_power(25, 230), 25 / 230, atol=1e-15)
        assert_allclose(current_by_power(25, 100), 25 / 100, atol=1e-15)
        assert_allclose(current_by_power("25 W", "100 V"), 25 / 100, atol=1e-15)

    def test_power_by_current_and_voltage(self):
        assert_allclose(power_by_current_and_voltage(1, 10), 1*10, atol=1e-15)
        assert_allclose(power_by_current_and_voltage(1, 230), 1 * 230, atol=1e-15)
        assert_allclose(power_by_current_and_voltage(0.2, 230), 0.2 * 230, atol=1e-15)
        assert_allclose(power_by_current_and_voltage(0.2), 0.2 * 230, atol=1e-15)
        assert_allclose(power_by_current_and_voltage("0.2 A", "230 V"), 0.2 * 230, atol=1e-15)
        assert_allclose(power_by_current_and_voltage(0.2, "230 V"), 0.2 * 230, atol=1e-15)

class TestAnnotatedTypes(unittest.TestCase):
    def test_type_annotations_exist(self):
        """Test that the type annotations are available"""
        self.assertIsNotNone(PowerW)
        self.assertIsNotNone(CurrentA)
        self.assertIsNotNone(VoltageV)

    def test_power_functions_various_units(self):
        """Test power functions with various unit inputs"""
        # Test current_by_power with different unit representations
        i1 = current_by_power("25 W", "100 V")
        i2 = current_by_power("25000 mW", "100000 mV")
        assert_allclose(i1, i2)

        # Test power_by_current_and_voltage with different units
        p1 = power_by_current_and_voltage("0.2 A", "230 V")
        p2 = power_by_current_and_voltage("200 mA", "230000 mV")
        assert_allclose(p1, p2)


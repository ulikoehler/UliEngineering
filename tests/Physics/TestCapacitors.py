#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from UliEngineering.Electronics.Capacitors import *
from UliEngineering.EngineerIO import auto_format
import numpy as np
import unittest

class TestCapacitors(unittest.TestCase):
    def test_capacitor_energy(self):
        assert_approx_equal(capacitor_energy("1.5 F", "5.0 V"), 18.75)
        assert_approx_equal(capacitor_energy("1.5 F", "0.0 V"), 0.0)
        self.assertEqual(auto_format(capacitor_energy, "100 mF", "1.2 V"), "72.0 mJ")

    def test_capacitor_charge(self):
        assert_approx_equal(capacitor_charge("1.5 F", "5.0 V"), 7.5)
        assert_approx_equal(capacitor_charge("1.5 F", "0.0 V"), 0.0)
        self.assertEqual(auto_format(capacitor_charge, "1.5 F", "5.0 V"), "7.50 C")

    def test_numpy_arrays(self):
        l = np.asarray([1.5, 0.1])
        assert_allclose(capacitor_energy(l, 5.0), [18.75, 1.25])

    def test_capacitor_lifetime(self):
        self.assertAlmostEqual(capacitor_lifetime(105, "2000 h", "105 °C", A=10), 2000)
        self.assertAlmostEqual(capacitor_lifetime(115, "2000 h", "105 °C", A=10), 1000)
        self.assertAlmostEqual(capacitor_lifetime(95, "2000 h", "105 °C", A=10), 4000)
        
    def test_capacitor_constant_current_discharge_time(self):
        # verified using https://www.circuits.dk/calculator_capacitor_discharge.htm
        # with Vcapmax=10, Vcapmin=1e-9, size=100e-6, ESR=1e-9, Imax=1000uA
        capacitance = 100e-6  # 100 uF
        voltage = 10  # 10 V
        current = 1e-3  # 1 mA
        expected_time = 1 # seconds

        # Charge should be the same
        discharge_time = capacitor_constant_current_discharge_time(capacitance, voltage, current)
        self.assertAlmostEqual(discharge_time, expected_time, places=2)
        
        charge_time = capacitor_constant_current_discharge_time(capacitance, voltage, current)
        self.assertAlmostEqual(charge_time, expected_time, places=2)
        
        # Check with keyword arguments
        discharge_time = capacitor_constant_current_discharge_time(capacitance=capacitance, initial_voltage=voltage, current=current)
        self.assertAlmostEqual(discharge_time, expected_time, places=2)

        # Check with nonzero target voltage (half voltage => half time)
        charge_time = capacitor_constant_current_discharge_time(capacitance, voltage, current, target_voltage=voltage/2)
        self.assertAlmostEqual(charge_time, expected_time/2, places=2)
    
    def test_capacitor_voltage_by_energy(self):
        # Basic test with zero starting voltage
        capacitance = 1.5  # F
        voltage = 5.0     # V
        energy = capacitor_energy(capacitance, voltage)
        calculated_voltage = capacitor_voltage_by_energy(capacitance, energy)
        self.assertAlmostEqual(calculated_voltage, voltage, places=10)
        
        # Test with non-zero starting voltage
        starting_voltage = 2.0  # V
        # Calculate additional energy needed to reach target voltage
        additional_energy = capacitor_energy(capacitance, voltage) - capacitor_energy(capacitance, starting_voltage)
        calculated_voltage = capacitor_voltage_by_energy(capacitance, additional_energy, starting_voltage)
        self.assertAlmostEqual(calculated_voltage, voltage, places=10)
        
        # Test with engineering notation
        energy = capacitor_energy("100 mF", "5.0 V")
        calculated_voltage = capacitor_voltage_by_energy("100 mF", energy)
        self.assertAlmostEqual(calculated_voltage, 5.0, places=10)
        
        # Test with zero energy (should return starting_voltage)
        calculated_voltage = capacitor_voltage_by_energy(capacitance, 0, "3V")
        self.assertAlmostEqual(calculated_voltage, 3.0, places=10)
        
        # Test with auto_format
        self.assertEqual(auto_format(capacitor_voltage_by_energy, "1.5 F", "18.75 J"), "5.00 V")
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

class TestCapacitorCapacitanceByEnergy(unittest.TestCase):
    def test_basic_functionality_zero_starting_voltage(self):
        """Test basic capacitance calculation with zero starting voltage"""
        energy = 18.75  # J
        voltage = 5.0   # V
        expected_capacitance = 1.5  # F
        calculated_capacitance = capacitor_capacitance_by_energy(energy, voltage)
        self.assertAlmostEqual(calculated_capacitance, expected_capacitance, places=10)

    def test_non_zero_starting_voltage(self):
        """Test capacitance calculation with non-zero starting voltage"""
        starting_voltage = "2.0 V"
        final_voltage = "5.0 V"
        capacitance = 1.5       # F
        # Calculate energy difference between final and starting voltage
        energy_diff = capacitor_energy(capacitance, final_voltage) - capacitor_energy(capacitance, starting_voltage)
        calculated_capacitance = capacitor_capacitance_by_energy(energy_diff, final_voltage, starting_voltage)
        self.assertAlmostEqual(calculated_capacitance, capacitance, places=10)

    def test_engineering_notation(self):
        """Test with engineering notation units"""
        # Test with mJ and V
        calculated_capacitance = capacitor_capacitance_by_energy("1.25 mJ", "5.0 V")
        self.assertAlmostEqual(calculated_capacitance, 0.1e-3, places=12)  # 0.1 mF
        
        # Test with all string parameters
        calculated_capacitance = capacitor_capacitance_by_energy("72.0 mJ", "1.2 V", "0V")
        self.assertAlmostEqual(calculated_capacitance, 0.1, places=10)  # 100 mF

    def test_consistency_with_capacitor_energy(self):
        """Test mathematical consistency with capacitor_energy function"""
        test_capacitance = 2.2e-6  # 2.2 µF
        test_voltage = 12.0        # V
        test_energy = capacitor_energy(test_capacitance, test_voltage)
        calculated_capacitance = capacitor_capacitance_by_energy(test_energy, test_voltage)
        self.assertAlmostEqual(calculated_capacitance, test_capacitance, places=12)

    def test_numpy_arrays(self):
        """Test with numpy arrays"""
        energies = np.array([18.75, 1.25])  # J
        voltages = np.array([5.0, 5.0])     # V
        expected_capacitances = np.array([1.5, 0.1])  # F
        calculated_capacitances = capacitor_capacitance_by_energy(energies, voltages)
        assert_allclose(calculated_capacitances, expected_capacitances, rtol=1e-10)
        
        # Test with mixed arrays and scalars
        calculated_capacitances = capacitor_capacitance_by_energy(energies, 5.0)
        assert_allclose(calculated_capacitances, expected_capacitances, rtol=1e-10)

    def test_edge_cases(self):
        """Test edge cases with very small and large values"""
        # Very small energy
        small_energy = 1e-12  # pJ
        voltage = 1.0         # V
        calculated_capacitance = capacitor_capacitance_by_energy(small_energy, voltage)
        self.assertAlmostEqual(calculated_capacitance, 2e-12, places=15)  # 2 pF
        
        # Large values
        large_energy = 1000  # J
        voltage = 100        # V
        calculated_capacitance = capacitor_capacitance_by_energy(large_energy, voltage)
        self.assertAlmostEqual(calculated_capacitance, 0.2, places=10)  # 200 mF

    def test_starting_voltage_effects(self):
        """Test how different starting voltages affect required capacitance"""
        energy = 10  # J
        final_voltage = 10  # V
        
        # Case 1: Starting from 0V
        cap1 = capacitor_capacitance_by_energy(energy, final_voltage, "0V")
        
        # Case 2: Starting from 5V (should require larger capacitance for same energy addition)
        cap2 = capacitor_capacitance_by_energy(energy, final_voltage, "5V")
        
        # cap2 should be larger than cap1
        self.assertGreater(cap2, cap1)
        
        # Verify the math: for starting voltage = 5V, final = 10V
        # Energy = 0.5 * C * (10² - 5²) = 0.5 * C * 75 = 37.5 * C
        # So C = 10 / 37.5 = 0.2667 F
        expected_cap2 = 10.0 / 37.5
        self.assertAlmostEqual(cap2, expected_cap2, places=10)

    def test_auto_format_functionality(self):
        """Test auto_format integration"""
        self.assertEqual(auto_format(capacitor_capacitance_by_energy, "18.75 J", "5.0 V"), "1.50 F")
        self.assertEqual(auto_format(capacitor_capacitance_by_energy, "1.25 mJ", "5.0 V"), "100 µF")

    def test_boundary_conditions(self):
        """Test boundary conditions and numerical stability"""
        # Zero energy case (should give infinite capacitance when starting voltage equals final voltage)
        calculated_capacitance = capacitor_capacitance_by_energy(0, "5V", "5V")
        # This should actually raise a division by zero error or return inf
        self.assertTrue(np.isinf(calculated_capacitance) or np.isnan(calculated_capacitance))
        
        # Test with very close voltages (numerical stability)
        energy = 1e-6  # µJ
        final_voltage = "1.0001 V"
        starting_voltage = "1.0000 V"
        calculated_capacitance = capacitor_capacitance_by_energy(energy, final_voltage, starting_voltage)
        # Should be finite and positive
        self.assertTrue(np.isfinite(calculated_capacitance))
        self.assertGreater(calculated_capacitance, 0)
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

    def test_round_trip_with_capacitor_energy(self):
        """Test round-trip calculations between capacitor_energy and capacitor_capacitance_by_energy"""
        # Test various capacitance and voltage combinations
        test_cases = [
            (1e-6, 5.0),    # 1 µF, 5V
            (100e-6, 12.0), # 100 µF, 12V
            (1e-3, 3.3),    # 1 mF, 3.3V
            (0.47, 24.0),   # 470 mF, 24V
            (2.2, 1.5),     # 2.2 F, 1.5V
        ]
        
        for original_capacitance, voltage in test_cases:
            with self.subTest(capacitance=original_capacitance, voltage=voltage):
                # Forward: capacitance + voltage -> energy
                energy = capacitor_energy(original_capacitance, voltage)
                
                # Backward: energy + voltage -> capacitance
                calculated_capacitance = capacitor_capacitance_by_energy(energy, voltage)
                
                # Should match original capacitance
                self.assertAlmostEqual(calculated_capacitance, original_capacitance, places=12)

    def test_round_trip_with_starting_voltage(self):
        """Test round-trip calculations with non-zero starting voltages"""
        test_cases = [
            (100e-6, 2.0, 10.0),  # 100 µF, 2V->10V
            (1e-3, 1.0, 5.0),     # 1 mF, 1V->5V
            (0.22, 3.0, 12.0),    # 220 mF, 3V->12V
        ]
        
        for original_capacitance, start_voltage, end_voltage in test_cases:
            with self.subTest(capacitance=original_capacitance, start_v=start_voltage, end_v=end_voltage):
                # Calculate energy difference
                end_energy = capacitor_energy(original_capacitance, end_voltage)
                start_energy = capacitor_energy(original_capacitance, start_voltage)
                energy_diff = end_energy - start_energy
                
                # Calculate capacitance from energy difference
                calculated_capacitance = capacitor_capacitance_by_energy(
                    energy_diff, f"{end_voltage}V", f"{start_voltage}V"
                )
                
                # Should match original capacitance
                self.assertAlmostEqual(calculated_capacitance, original_capacitance, places=12)

    def test_energy_calculation_verification(self):
        """Test that calculated capacitance produces correct energy when used with capacitor_energy"""
        test_energies = [1e-9, 1e-6, 1e-3, 1.0, 100.0]  # nJ to 100J
        test_voltages = [1.0, 3.3, 5.0, 12.0, 24.0]     # Various voltages
        
        for energy in test_energies:
            for voltage in test_voltages:
                with self.subTest(energy=energy, voltage=voltage):
                    # Calculate required capacitance
                    capacitance = capacitor_capacitance_by_energy(energy, voltage)
                    
                    # Verify that this capacitance at this voltage gives the expected energy
                    calculated_energy = capacitor_energy(capacitance, voltage)
                    
                    # Should match original energy
                    self.assertAlmostEqual(calculated_energy, energy, places=12)

    def test_energy_verification_with_starting_voltage(self):
        """Test energy calculations with starting voltages match expected results"""
        capacitance = 100e-6  # 100 µF
        start_voltage = 2.0   # V
        end_voltage = 8.0     # V
        
        # Calculate actual energy difference
        actual_energy_diff = (
            capacitor_energy(capacitance, end_voltage) - 
            capacitor_energy(capacitance, start_voltage)
        )
        
        # Calculate required capacitance for this energy difference
        calculated_capacitance = capacitor_capacitance_by_energy(
            actual_energy_diff, f"{end_voltage}V", f"{start_voltage}V"
        )
        
        # Should match original capacitance
        self.assertAlmostEqual(calculated_capacitance, capacitance, places=12)
        
        # Verify the energy calculation is correct
        # Energy difference should be 0.5 * C * (V_end² - V_start²)
        expected_energy_diff = 0.5 * capacitance * (end_voltage**2 - start_voltage**2)
        self.assertAlmostEqual(actual_energy_diff, expected_energy_diff, places=12)

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
        with np.errstate(divide='ignore', invalid='ignore'):
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

class TestCapacitorChargingEnergy(unittest.TestCase):
    """Comprehensive tests for capacitor_charging_energy function"""
    
    def test_basic_functionality_zero_starting_voltage(self):
        """Test basic energy calculation from 0V to target voltage"""
        capacitance = 1.5  # F
        end_voltage = 5.0  # V
        # Energy from 0V to 5V should be same as total stored energy
        expected_energy = capacitor_energy(capacitance, end_voltage)
        calculated_energy = capacitor_charging_energy(capacitance, end_voltage)
        self.assertAlmostEqual(calculated_energy, expected_energy, places=10)
        
        # Test specific values
        self.assertAlmostEqual(calculated_energy, 18.75, places=10)  # 0.5 * 1.5 * 25 = 18.75 J

    def test_non_zero_starting_voltage(self):
        """Test energy calculation with non-zero starting voltage"""
        capacitance = "1.0 F"  # F
        starting_voltage = "2.0 V"  # V
        end_voltage = "6.0 V"  # V
        
        # Manual calculation: 0.5 * 1.0 * (6² - 2²) = 0.5 * (36 - 4) = 16 J
        expected_energy = 0.5 * 1.0 * (6.0**2 - 2.0**2)
        calculated_energy = capacitor_charging_energy(capacitance, end_voltage, starting_voltage)
        self.assertAlmostEqual(calculated_energy, expected_energy, places=10)
        self.assertAlmostEqual(calculated_energy, 16.0, places=10)

    def test_engineering_notation(self):
        """Test with engineering notation units"""
        # Test with mF and V
        calculated_energy = capacitor_charging_energy("100 mF", "1.2 V")
        expected_energy = 0.5 * 0.1 * (1.2**2)  # 0.072 J = 72 mJ
        self.assertAlmostEqual(calculated_energy, expected_energy, places=12)
        
        # Test with µF and kV
        calculated_energy = capacitor_charging_energy("10 µF", "1 kV", "500 V")
        expected_energy = 0.5 * 10e-6 * (1000**2 - 500**2)  # 3.75 J
        self.assertAlmostEqual(calculated_energy, 3.75, places=10)

    def test_consistency_with_capacitor_energy(self):
        """Test mathematical consistency with capacitor_energy function"""
        test_cases = [
            ("100 µF", "12.0 V", "0.0 V"),    # 100 µF, 0V->12V
            ("1 mF", "5.0 V", "2.0 V"),       # 1 mF, 2V->5V  
            ("0.47 F", "24.0 V", "12.0 V"),   # 470 mF, 12V->24V
            ("2.2 F", "1.5 V", "0.5 V"),      # 2.2 F, 0.5V->1.5V
        ]
        
        for capacitance, end_voltage, start_voltage in test_cases:
            with self.subTest(capacitance=capacitance, end_v=end_voltage, start_v=start_voltage):
                # Calculate using capacitor_charging_energy
                charging_energy = capacitor_charging_energy(capacitance, end_voltage, start_voltage)
                
                # Calculate manually using capacitor_energy difference
                end_energy = capacitor_energy(capacitance, end_voltage)
                start_energy = capacitor_energy(capacitance, start_voltage)
                expected_energy = end_energy - start_energy
                
                self.assertAlmostEqual(charging_energy, expected_energy, places=12)

    def test_zero_energy_cases(self):
        """Test cases where no energy is required"""
        capacitance = 1.0  # F
        voltage = 5.0     # V
        
        # Same starting and ending voltage should require zero energy
        energy = capacitor_charging_energy(capacitance, voltage, voltage)
        self.assertAlmostEqual(energy, 0.0, places=12)
        
        # Test with string voltages
        energy = capacitor_charging_energy("1.5 F", "10 V", "10 V")
        self.assertAlmostEqual(energy, 0.0, places=12)

    def test_negative_energy_discharge(self):
        """Test that discharging (higher start voltage) gives negative energy"""
        capacitance = 1.0  # F
        starting_voltage = 10.0  # V
        end_voltage = 5.0       # V
        
        # Should be negative since we're going from higher to lower voltage
        energy = capacitor_charging_energy(capacitance, end_voltage, starting_voltage)
        self.assertLess(energy, 0)
        
        # Manual calculation: 0.5 * 1.0 * (5² - 10²) = 0.5 * (25 - 100) = -37.5 J
        expected_energy = 0.5 * capacitance * (end_voltage**2 - starting_voltage**2)
        self.assertAlmostEqual(energy, expected_energy, places=10)
        self.assertAlmostEqual(energy, -37.5, places=10)

    def test_numpy_arrays(self):
        """Test with numpy arrays"""
        capacitances = np.array([1.0, 2.0, 0.5])  # F
        end_voltages = np.array([5.0, 3.0, 4.0])  # V
        start_voltages = np.array([0.0, 1.0, 2.0]) # V
        
        # Expected energies: [12.5, 8.0, 4.0] J
        expected_energies = 0.5 * capacitances * (end_voltages**2 - start_voltages**2)
        calculated_energies = capacitor_charging_energy(capacitances, end_voltages, start_voltages)
        
        assert_allclose(calculated_energies, expected_energies, rtol=1e-10)
        
        # Test with mixed arrays and scalars
        calculated_energies = capacitor_charging_energy(1.0, end_voltages, 0.0)
        expected_energies = 0.5 * 1.0 * end_voltages**2
        assert_allclose(calculated_energies, expected_energies, rtol=1e-10)

    def test_edge_cases_small_values(self):
        """Test edge cases with very small values"""
        # Very small capacitance
        small_capacitance = 1e-12  # pF
        voltage_change = 1.0       # V
        energy = capacitor_charging_energy(small_capacitance, voltage_change)
        expected_energy = 0.5 * small_capacitance * voltage_change**2
        self.assertAlmostEqual(energy, expected_energy, places=15)
        self.assertAlmostEqual(energy, 0.5e-12, places=15)  # 0.5 pJ

    def test_edge_cases_large_values(self):
        """Test edge cases with large values"""
        # Large capacitance and voltage
        large_capacitance = 100  # F (supercapacitor range)
        high_voltage = 1000     # V
        energy = capacitor_charging_energy(large_capacitance, high_voltage)
        expected_energy = 0.5 * large_capacitance * high_voltage**2
        self.assertAlmostEqual(energy, expected_energy, places=5)
        self.assertAlmostEqual(energy, 50e6, places=5)  # 50 MJ

    def test_symmetry_properties(self):
        """Test symmetry properties of the function"""
        capacitance = 1.0  # F
        voltage_a = 3.0   # V
        voltage_b = 7.0   # V
        
        # Energy to go from A to B should be negative of energy to go from B to A
        energy_a_to_b = capacitor_charging_energy(capacitance, voltage_b, voltage_a)
        energy_b_to_a = capacitor_charging_energy(capacitance, voltage_a, voltage_b)
        
        self.assertAlmostEqual(energy_a_to_b, -energy_b_to_a, places=12)

    def test_additivity_property(self):
        """Test that energy is additive for multi-step charging"""
        capacitance = 1.0  # F
        voltage_start = 1.0  # V
        voltage_middle = 4.0 # V  
        voltage_end = 8.0    # V
        
        # Energy from start to end in one step
        direct_energy = capacitor_charging_energy(capacitance, voltage_end, voltage_start)
        
        # Energy from start to middle, then middle to end
        step1_energy = capacitor_charging_energy(capacitance, voltage_middle, voltage_start)
        step2_energy = capacitor_charging_energy(capacitance, voltage_end, voltage_middle)
        total_step_energy = step1_energy + step2_energy
        
        self.assertAlmostEqual(direct_energy, total_step_energy, places=12)

    def test_real_world_scenarios(self):
        """Test realistic capacitor charging scenarios"""
        # Smartphone camera flash capacitor
        flash_cap = 100e-6  # 100 µF
        flash_voltage = 300  # V
        flash_energy = capacitor_charging_energy(flash_cap, flash_voltage)
        self.assertAlmostEqual(flash_energy, 4.5, places=10)  # 4.5 J
        
        # Car audio system capacitor
        audio_cap = 1.0     # 1 F
        audio_voltage = 14.4 # V (car electrical system)
        audio_energy = capacitor_charging_energy(audio_cap, audio_voltage)
        self.assertAlmostEqual(audio_energy, 103.68, places=8)  # ~104 J
        
        # Defibrillator capacitor
        defib_cap = 32e-6   # 32 µF
        defib_voltage = 5000 # V
        defib_energy = capacitor_charging_energy(defib_cap, defib_voltage)
        self.assertAlmostEqual(defib_energy, 400, places=5)  # 400 J

    def test_partial_charging_scenarios(self):
        """Test partial charging from non-zero starting voltages"""
        capacitance = 470e-6  # 470 µF
        
        # Charging from 50% to 100% of rated voltage
        rated_voltage = 25    # V
        half_voltage = 12.5   # V
        
        partial_energy = capacitor_charging_energy(capacitance, rated_voltage, half_voltage)
        full_energy = capacitor_charging_energy(capacitance, rated_voltage)
        half_energy = capacitor_charging_energy(capacitance, half_voltage)
        
        # Partial energy should equal full energy minus energy already stored
        self.assertAlmostEqual(partial_energy, full_energy - half_energy, places=12)
        
        # Calculate expected value: 0.5 * 470e-6 * (25² - 12.5²) = 0.11 J
        expected_partial = 0.5 * capacitance * (rated_voltage**2 - half_voltage**2)
        self.assertAlmostEqual(partial_energy, expected_partial, places=12)

    def test_auto_format_functionality(self):
        """Test auto_format integration"""
        # Test various formats
        result = auto_format(capacitor_charging_energy, "1.5 F", "5.0 V")
        self.assertEqual(result, "18.8 J")
        
        result = auto_format(capacitor_charging_energy, "100 mF", "1.2 V")
        self.assertEqual(result, "72.0 mJ")
        
        result = auto_format(capacitor_charging_energy, "10 µF", "100 V", "50 V")
        # 0.5 * 10e-6 * (100² - 50²) = 37.5 mJ
        self.assertEqual(result, "37.5 mJ")

    def test_mathematical_formulas(self):
        """Test against known mathematical formulas"""
        test_cases = [
            # (capacitance, end_voltage, start_voltage, expected_energy)
            (1e-6, 10, 0, 50e-6),        # 1µF, 0->10V: 50µJ
            (100e-6, 5, 3, 800e-6),      # 100µF, 3->5V: 800µJ  
            (0.001, 12, 8, 40e-3),       # 1mF, 8->12V: 40mJ
            (1.0, 10, 6, 32),            # 1F, 6->10V: 32J
        ]
        
        for capacitance, end_v, start_v, expected in test_cases:
            with self.subTest(C=capacitance, V_end=end_v, V_start=start_v):
                calculated = capacitor_charging_energy(capacitance, end_v, start_v)
                self.assertAlmostEqual(calculated, expected, places=10)

    def test_error_handling_and_boundary_conditions(self):
        """Test boundary conditions and numerical stability"""
        # Very small voltage differences
        capacitance = 1.0
        v1 = 1.000000000
        v2 = 1.000000001
        
        energy = capacitor_charging_energy(capacitance, v2, v1)
        # Should be very small but finite
        self.assertTrue(np.isfinite(energy))
        self.assertGreater(energy, 0)
        self.assertLess(energy, 1e-6)  # Should be very small
        
        # Test with zero capacitance
        zero_energy = capacitor_charging_energy(0, 10, 5)
        self.assertAlmostEqual(zero_energy, 0.0, places=12)

    def test_units_consistency(self):
        """Test that function works correctly with various unit combinations"""
        # All combinations should give same result when properly converted
        base_result = capacitor_charging_energy(1e-3, 5.0, 0.0)  # 1mF, 5V
        
        # Test with different unit representations
        result1 = capacitor_charging_energy("1 mF", "5 V", "0 V")
        result2 = capacitor_charging_energy("1000 µF", "5000 mV", "0 mV")
        result3 = capacitor_charging_energy("0.001 F", "5.0 V")
        
        self.assertAlmostEqual(base_result, result1, places=10)
        self.assertAlmostEqual(base_result, result2, places=10)
        self.assertAlmostEqual(base_result, result3, places=10)
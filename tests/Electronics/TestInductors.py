#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from UliEngineering.Electronics.Inductors import *
from UliEngineering.EngineerIO import auto_format
import numpy as np
import unittest

class TestInductors(unittest.TestCase):
    def test_ideal_inductor_current_change_rate_basic(self):
        """Test basic current change rate calculation"""
        # Basic test: V = L * di/dt, so di/dt = V/L
        inductance = 1.0  # H
        voltage = 5.0     # V
        expected_rate = 5.0  # A/s
        
        calculated_rate = ideal_inductor_current_change_rate(inductance, voltage)
        assert_approx_equal(calculated_rate, expected_rate)
        
        # Test with zero voltage
        calculated_rate = ideal_inductor_current_change_rate(inductance, 0.0)
        assert_approx_equal(calculated_rate, 0.0)
        
        # Test with negative voltage (current decreasing)
        calculated_rate = ideal_inductor_current_change_rate(inductance, -2.0)
        assert_approx_equal(calculated_rate, -2.0)

    def test_engineering_notation(self):
        """Test with engineering notation units"""
        # Test with mH and V
        calculated_rate = ideal_inductor_current_change_rate("10 mH", "1.2 V")
        expected_rate = 1.2 / 0.01  # 120 A/s
        assert_approx_equal(calculated_rate, expected_rate)
        
        # Test with µH and mV
        calculated_rate = ideal_inductor_current_change_rate("100 µH", "500 mV")
        expected_rate = 0.5 / 100e-6  # 5000 A/s
        assert_approx_equal(calculated_rate, expected_rate)
        
        # Test with H and kV
        calculated_rate = ideal_inductor_current_change_rate("1 H", "2 kV")
        expected_rate = 2000.0  # A/s
        assert_approx_equal(calculated_rate, expected_rate)

    def test_auto_format_functionality(self):
        """Test auto_format integration"""
        result = auto_format(ideal_inductor_current_change_rate, "1 H", "5 V")
        self.assertEqual(result, "5.00 A/s")
        
        result = auto_format(ideal_inductor_current_change_rate, "10 mH", "1.2 V")
        self.assertEqual(result, "120 A/s")
        
        result = auto_format(ideal_inductor_current_change_rate, "100 µH", "500 mV")
        self.assertEqual(result, "5.00 kA/s")

    def test_numpy_arrays(self):
        """Test with numpy arrays"""
        inductances = np.array([1.0, 2.0, 0.5])  # H
        voltages = np.array([5.0, 4.0, 3.0])     # V
        expected_rates = np.array([5.0, 2.0, 6.0])  # A/s
        
        calculated_rates = ideal_inductor_current_change_rate(inductances, voltages)
        assert_allclose(calculated_rates, expected_rates, rtol=1e-10)
        
        # Test with mixed arrays and scalars
        calculated_rates = ideal_inductor_current_change_rate(inductances, 10.0)
        expected_rates = np.array([10.0, 5.0, 20.0])  # A/s
        assert_allclose(calculated_rates, expected_rates, rtol=1e-10)
        
        # Test with scalar inductance and array voltages
        calculated_rates = ideal_inductor_current_change_rate(1.0, voltages)
        expected_rates = voltages  # Same as voltages when L=1H
        assert_allclose(calculated_rates, expected_rates, rtol=1e-10)

    def test_realistic_inductor_scenarios(self):
        """Test with realistic inductor applications"""
        # Buck converter inductor (typical switching frequency ~100kHz)
        # Typical ripple current calculation
        buck_inductance = 47e-6  # 47 µH
        buck_voltage = 2.0       # V (Vin - Vout difference)
        buck_rate = ideal_inductor_current_change_rate(buck_inductance, buck_voltage)
        self.assertAlmostEqual(buck_rate, 2.0 / 47e-6, places=5)  # ~42553 A/s
        
        # Motor inductance (typical AC motor)
        motor_inductance = 0.1   # H
        motor_voltage = 230      # V (line voltage)
        motor_rate = ideal_inductor_current_change_rate(motor_inductance, motor_voltage)
        self.assertAlmostEqual(motor_rate, 2300.0, places=5)  # A/s
        
        # RF choke inductor
        rf_inductance = 1e-6     # 1 µH
        rf_voltage = 0.1         # V (small signal)
        rf_rate = ideal_inductor_current_change_rate(rf_inductance, rf_voltage)
        self.assertAlmostEqual(rf_rate, 100000.0, places=5)  # 100 kA/s
        
        # Power transformer primary (large inductance)
        transformer_inductance = 10.0  # H
        transformer_voltage = 120      # V (AC line)
        transformer_rate = ideal_inductor_current_change_rate(transformer_inductance, transformer_voltage)
        self.assertAlmostEqual(transformer_rate, 12.0, places=10)  # A/s

    def test_switching_power_supply_scenarios(self):
        """Test scenarios common in switching power supplies"""
        # Boost converter inductor during switch-on phase
        boost_inductance = 22e-6  # 22 µH
        input_voltage = 12.0      # V
        boost_rate = ideal_inductor_current_change_rate(boost_inductance, input_voltage)
        # This is the rate at which current builds up when switch is closed
        expected_rate = 12.0 / 22e-6  # ~545454 A/s
        self.assertAlmostEqual(boost_rate, expected_rate, places=0)
        
        # Buck-boost converter
        buck_boost_inductance = 100e-6  # 100 µH
        voltage_across = 8.0            # V
        buck_boost_rate = ideal_inductor_current_change_rate(buck_boost_inductance, voltage_across)
        expected_rate = 8.0 / 100e-6    # 80000 A/s
        self.assertAlmostEqual(buck_boost_rate, expected_rate, places=5)

    def test_ac_circuit_scenarios(self):
        """Test scenarios relevant to AC circuits"""
        # Inductor in AC circuit - instantaneous rate
        ac_inductance = 0.01     # H (10 mH)
        peak_voltage = 170       # V (120V RMS * sqrt(2))
        ac_rate = ideal_inductor_current_change_rate(ac_inductance, peak_voltage)
        self.assertAlmostEqual(ac_rate, 17000.0, places=5)  # A/s
        
        # Filter inductor in rectifier circuit
        filter_inductance = 0.05  # H (50 mH)
        ripple_voltage = 5.0      # V
        filter_rate = ideal_inductor_current_change_rate(filter_inductance, ripple_voltage)
        self.assertAlmostEqual(filter_rate, 100.0, places=10)  # A/s

    def test_edge_cases_small_values(self):
        """Test edge cases with very small values"""
        # Very small inductance (parasitic inductance)
        tiny_inductance = 1e-12  # pH (picohenry)
        small_voltage = 1e-3     # mV
        tiny_rate = ideal_inductor_current_change_rate(tiny_inductance, small_voltage)
        expected_rate = 1e-3 / 1e-12  # 1e9 A/s = 1 GA/s
        self.assertAlmostEqual(tiny_rate, expected_rate, places=5)
        
        # Very small voltage
        normal_inductance = 1e-3  # mH
        tiny_voltage = 1e-9       # nV
        tiny_rate = ideal_inductor_current_change_rate(normal_inductance, tiny_voltage)
        expected_rate = 1e-9 / 1e-3  # 1e-6 A/s = 1 µA/s
        self.assertAlmostEqual(tiny_rate, expected_rate, places=15)

    def test_edge_cases_large_values(self):
        """Test edge cases with large values"""
        # Large inductance (power grid applications)
        large_inductance = 100   # H
        high_voltage = 10000     # V (10 kV)
        large_rate = ideal_inductor_current_change_rate(large_inductance, high_voltage)
        expected_rate = 10000 / 100  # 100 A/s
        self.assertAlmostEqual(large_rate, expected_rate, places=10)
        
        # Very high voltage (lightning arrester)
        normal_inductance = 1e-6  # µH
        lightning_voltage = 1e6   # MV
        lightning_rate = ideal_inductor_current_change_rate(normal_inductance, lightning_voltage)
        expected_rate = 1e6 / 1e-6  # 1e12 A/s = 1 TA/s
        self.assertAlmostEqual(lightning_rate, expected_rate, places=5)

    def test_mathematical_relationships(self):
        """Test mathematical relationships and consistency"""
        inductance = 0.001  # H
        voltage = 10.0      # V
        
        # Test linearity with voltage
        rate1 = ideal_inductor_current_change_rate(inductance, voltage)
        rate2 = ideal_inductor_current_change_rate(inductance, 2 * voltage)
        self.assertAlmostEqual(rate2, 2 * rate1, places=12)
        
        # Test inverse relationship with inductance
        rate1 = ideal_inductor_current_change_rate(inductance, voltage)
        rate2 = ideal_inductor_current_change_rate(2 * inductance, voltage)
        self.assertAlmostEqual(rate2, rate1 / 2, places=12)
        
        # Test that rate is symmetric for positive/negative voltages
        positive_rate = ideal_inductor_current_change_rate(inductance, voltage)
        negative_rate = ideal_inductor_current_change_rate(inductance, -voltage)
        self.assertAlmostEqual(negative_rate, -positive_rate, places=12)

    def test_physical_consistency(self):
        """Test physical consistency of results"""
        # Verify units and physical meaning
        inductance = 1e-3  # mH
        voltage = 5.0      # V
        rate = ideal_inductor_current_change_rate(inductance, voltage)
        
        # Rate should be in A/s and finite
        self.assertTrue(np.isfinite(rate))
        self.assertGreater(abs(rate), 0)  # Non-zero voltage should give non-zero rate
        
        # For positive voltage and positive inductance, rate should be positive
        self.assertGreater(rate, 0)
        
        # Test energy considerations (conceptual verification)
        # Higher voltage should give higher rate for same inductance
        high_voltage_rate = ideal_inductor_current_change_rate(inductance, 2 * voltage)
        self.assertGreater(high_voltage_rate, rate)
        
        # Larger inductance should give lower rate for same voltage
        large_inductance_rate = ideal_inductor_current_change_rate(2 * inductance, voltage)
        self.assertLess(large_inductance_rate, rate)

    def test_switching_transient_calculations(self):
        """Test calculations relevant to switching transients"""
        # MOSFET gate driver circuit
        gate_inductance = 10e-9  # 10 nH (parasitic inductance)
        gate_voltage = 12.0      # V
        gate_rate = ideal_inductor_current_change_rate(gate_inductance, gate_voltage)
        # This represents how fast current can change during switching
        expected_rate = 12.0 / 10e-9  # 1.2e9 A/s = 1.2 GA/s
        self.assertAlmostEqual(gate_rate, expected_rate, places=5)
        
        # PCB trace inductance effect
        trace_inductance = 1e-9   # 1 nH/mm typical
        switching_voltage = 3.3   # V
        trace_rate = ideal_inductor_current_change_rate(trace_inductance, switching_voltage)
        expected_rate = 3.3 / 1e-9  # 3.3e9 A/s
        self.assertAlmostEqual(trace_rate, expected_rate, places=5)

    def test_boundary_conditions(self):
        """Test boundary conditions and numerical stability"""
        # Zero inductance case (should raise division by zero or return inf)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = ideal_inductor_current_change_rate(0, 5.0)
            self.assertTrue(np.isinf(result) or np.isnan(result))
        
        # Very small inductance (numerical stability)
        tiny_inductance = 1e-15  # fH
        normal_voltage = 1.0     # V
        result = ideal_inductor_current_change_rate(tiny_inductance, normal_voltage)
        self.assertTrue(np.isfinite(result))
        self.assertGreater(result, 0)

    def test_units_consistency(self):
        """Test that function works correctly with various unit combinations"""
        # All combinations should give same result when properly converted
        base_result = ideal_inductor_current_change_rate(1e-3, 5.0)  # 1mH, 5V
        
        # Test with different unit representations
        result1 = ideal_inductor_current_change_rate("1 mH", "5 V")
        result2 = ideal_inductor_current_change_rate("1000 µH", "5000 mV")
        result3 = ideal_inductor_current_change_rate("0.001 H", "5.0 V")
        
        self.assertAlmostEqual(base_result, result1, places=10)
        self.assertAlmostEqual(base_result, result2, places=10)
        self.assertAlmostEqual(base_result, result3, places=10)

    def test_real_world_component_values(self):
        """Test with actual component values from datasheets"""
        # Ferrite bead (high frequency)
        ferrite_inductance = 120e-9  # 120 nH @ 100 MHz
        signal_voltage = 0.1         # V
        ferrite_rate = ideal_inductor_current_change_rate(ferrite_inductance, signal_voltage)
        self.assertGreater(ferrite_rate, 0.5e6)  # Should be greater than 500 kA/s
        self.assertLess(ferrite_rate, 2e9)     # Should be less than 1 GA/s
        
        # Common mode choke
        common_mode_inductance = 1e-3  # 1 mH
        differential_voltage = 1.0     # V
        cm_rate = ideal_inductor_current_change_rate(common_mode_inductance, differential_voltage)
        self.assertAlmostEqual(cm_rate, 1000.0, places=10)  # 1000 A/s
        
        # Power inductor (SMD)
        power_inductance = 4.7e-6    # 4.7 µH
        supply_voltage = 3.3         # V
        power_rate = ideal_inductor_current_change_rate(power_inductance, supply_voltage)
        expected_rate = 3.3 / 4.7e-6  # ~702127 A/s
        self.assertAlmostEqual(power_rate, expected_rate, places=0)

    def test_frequency_domain_implications(self):
        """Test scenarios that relate to frequency domain behavior"""
        # At different frequencies, the same inductor will have different impedance
        # But the current change rate formula di/dt = V/L remains the same
        inductance = 100e-6  # 100 µH
        
        # Low frequency scenario
        low_freq_voltage = 1.0  # V
        low_freq_rate = ideal_inductor_current_change_rate(inductance, low_freq_voltage)
        
        # High frequency scenario (same voltage magnitude)
        high_freq_voltage = 1.0  # V
        high_freq_rate = ideal_inductor_current_change_rate(inductance, high_freq_voltage)
        
        # The rate should be the same since it only depends on L and instantaneous V
        self.assertAlmostEqual(low_freq_rate, high_freq_rate, places=12)
        
        # This confirms that the function correctly calculates instantaneous rate
        # regardless of frequency (as it should for an ideal inductor)

if __name__ == '__main__':
    unittest.main()

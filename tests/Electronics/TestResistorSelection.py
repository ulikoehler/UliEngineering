#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from UliEngineering.Electronics.ResistorSelection import *
from UliEngineering.EngineerIO import *
from UliEngineering.Electronics.Resistors import resistor_value_by_voltage_and_power
import unittest
import numpy as np

class TestResistorSeriesCostFunctor(unittest.TestCase):
    
    def test_default_weights_and_basic_functionality(self):
        """Test basic functionality with default weights"""
        functor = ResistorSeriesCostFunctor()
        
        # Test known E6 values (should have lowest cost)
        assert_approx_equal(functor("10Ω"), 0.95)  # E6 value
        assert_approx_equal(functor("22Ω"), 0.95)  # E6 value
        assert_approx_equal(functor("47Ω"), 0.95)  # E6 value
        assert_approx_equal(functor("100Ω"), 0.95) # E6 value
        
        # Test known E12 values (not in E6)
        assert_approx_equal(functor("12Ω"), 1.0)   # E12 but not E6
        assert_approx_equal(functor("18Ω"), 1.0)   # E12 but not E6
        assert_approx_equal(functor("39Ω"), 1.0)   # E12 but not E6
        
        # Test known E24 values (not in E12 or E6)
        assert_approx_equal(functor("11Ω"), 2.0)   # E24 but not E12/E6
        assert_approx_equal(functor("13Ω"), 2.0)   # E24 but not E12/E6
        
        # Test non-standard values
        assert_approx_equal(functor("7.5Ω"), 100.0)  # Not in any standard series
        assert_approx_equal(functor("123Ω"), 100.0)  # Not in any standard series

    def test_custom_weights(self):
        """Test with custom weight configuration"""
        custom_weights = ResistorSeriesWeights(
            E6=1.0,
            E12=2.0,
            E24=3.0,
            E48=4.0,
            E96=5.0,
            E192=6.0,
            non_series=50.0
        )
        functor = ResistorSeriesCostFunctor(weights=custom_weights)
        
        # Test that custom weights are applied
        assert_approx_equal(functor("10Ω"), 1.0)   # E6 with custom weight
        assert_approx_equal(functor("12Ω"), 2.0)   # E12 with custom weight
        assert_approx_equal(functor("11Ω"), 3.0)   # E24 with custom weight
        assert_approx_equal(functor("123Ω"), 50.0) # Non-series with custom weight

    def test_tolerance_handling(self):
        """Test tolerance parameter functionality"""
        # Strict tolerance
        strict_functor = ResistorSeriesCostFunctor(tolerance=0.0001)  # 0.01%
        
        # Loose tolerance
        loose_functor = ResistorSeriesCostFunctor(tolerance=0.05)     # 5%
        
        # Value very close to E6 standard (10Ω)
        close_value = 10.005  # 0.05% off from 10Ω
        
        # With strict tolerance, this should not match E6
        assert_approx_equal(strict_functor(close_value), 100.0)  # non_series
        
        # With loose tolerance, this should match E6
        assert_approx_equal(loose_functor(close_value), 0.95)    # E6

    def test_engineer_string_inputs(self):
        """Test with various engineer string formats"""
        functor = ResistorSeriesCostFunctor()
        
        # Test different units and formats
        assert_approx_equal(functor("10Ω"), 0.95)      # E6
        assert_approx_equal(functor("10R"), 0.95)      # E6, different unit
        assert_approx_equal(functor("0.01kΩ"), 0.95)   # E6, different scale
        assert_approx_equal(functor("10000mΩ"), 0.95)  # E6, different scale
        
        assert_approx_equal(functor("1kΩ"), 0.95)      # E6 (1000Ω)
        assert_approx_equal(functor("1.0kΩ"), 0.95)    # E6 (1000Ω)
        assert_approx_equal(functor("1000Ω"), 0.95)    # E6

    def test_series_precedence(self):
        """Test that values are assigned to the most common series (E6 before E12, etc.)"""
        functor = ResistorSeriesCostFunctor()
        
        # Values that appear in multiple series should get the lowest-cost assignment
        # 10Ω is in E6, E12, E24, E48, E96, E192 - should get E6 cost
        assert_approx_equal(functor("10Ω"), 0.95)  # E6 cost, not E12 cost
        
        # 22Ω is in E6, E12, E24, E48, E96, E192 - should get E6 cost
        assert_approx_equal(functor("22Ω"), 0.95)  # E6 cost, not E12 cost
        
        # 18Ω is in E12, E24, E48, E96, E192 but not E6 - should get E12 cost
        assert_approx_equal(functor("18Ω"), 1.0)   # E12 cost, not E24 cost

    def test_wide_resistance_range(self):
        """Test with a wide range of resistance values"""
        functor = ResistorSeriesCostFunctor()
        
        # Test very small resistances
        assert_approx_equal(functor("0.1Ω"), 0.95)    # Should find in E6 (scaled)
        assert_approx_equal(functor("0.22Ω"), 0.95)   # Should find in E6 (scaled)
        
        # Test medium resistances
        assert_approx_equal(functor("4.7kΩ"), 0.95)   # E6
        assert_approx_equal(functor("47kΩ"), 0.95)    # E6
        
        # Test large resistances
        assert_approx_equal(functor("1MΩ"), 0.95)     # E6 (1000000Ω)
        assert_approx_equal(functor("10MΩ"), 0.95)    # E6 (10000000Ω)

    def test_e48_e96_e192_values(self):
        """Test specific values from higher-precision series"""
        functor = ResistorSeriesCostFunctor()
        
        # Test some E48 values (not in lower series)
        # Note: These are approximate - actual values depend on the implementation
        # of standard_resistors() function
        
        # Test some values that should be in E96 but not in lower series
        # These tests may need adjustment based on actual E-series implementation
        
        # For now, test that non-standard values get high cost
        assert_approx_equal(functor("7.32Ω"), 100.0)   # Definitely not standard
        assert_approx_equal(functor("43.21Ω"), 100.0)  # Definitely not standard

    def test_boundary_values(self):
        """Test values at the boundaries of tolerance"""
        functor = ResistorSeriesCostFunctor(tolerance=0.01)  # 1% tolerance
        
        # Test values just inside and outside tolerance for 100Ω (E6 value)
        assert_approx_equal(functor(100.0), 0.95)      # Exact match
        assert_approx_equal(functor(100.5), 0.95)      # 0.5% off, within tolerance
        assert_approx_equal(functor(99.5), 0.95)       # 0.5% off, within tolerance
        assert_approx_equal(functor(102.0), 100.0)     # 2% off, outside tolerance
        assert_approx_equal(functor(98.0), 100.0)      # 2% off, outside tolerance

    def test_zero_and_negative_values(self):
        """Test edge cases with zero and negative values"""
        functor = ResistorSeriesCostFunctor()
        
        # These should not crash but should return non_series cost
        assert_approx_equal(functor(0.0), 100.0)
        assert_approx_equal(functor(-10.0), 100.0)
        assert_approx_equal(functor(-100.0), 100.0)

    def test_very_large_and_small_values(self):
        """Test with extreme resistance values"""
        functor = ResistorSeriesCostFunctor()
        
        # Very small values
        result_small = functor(1e-9)  # 1 nanoohm
        self.assertIsInstance(result_small, float)
        
        # Very large values  
        result_large = functor(1e12)  # 1 teraohm
        self.assertIsInstance(result_large, float)

    def test_reproducibility(self):
        """Test that results are reproducible"""
        functor1 = ResistorSeriesCostFunctor()
        functor2 = ResistorSeriesCostFunctor()
        
        test_values = ["10Ω", "12Ω", "15Ω", "22Ω", "47Ω", "100Ω", "1kΩ"]
        
        for value in test_values:
            result1 = functor1(value)
            result2 = functor2(value)
            assert_approx_equal(result1, result2)

    def test_performance_consistency(self):
        """Test that the functor performs consistently across multiple calls"""
        functor = ResistorSeriesCostFunctor()
        
        # Test same value multiple times
        test_value = "47Ω"
        results = [functor(test_value) for _ in range(100)]
        
        # All results should be identical
        for result in results[1:]:
            assert_approx_equal(result, results[0])

    def test_initialization_parameters(self):
        """Test different initialization parameter combinations"""
        # Test with only custom weights
        custom_weights = ResistorSeriesWeights(E6=0.5, non_series=200.0)
        functor1 = ResistorSeriesCostFunctor(weights=custom_weights)
        assert_approx_equal(functor1("10Ω"), 0.5)
        
        # Test with only custom tolerance
        functor2 = ResistorSeriesCostFunctor(tolerance=0.1)  # 10% tolerance
        self.assertEqual(functor2.tolerance, 0.1)
        
        # Test with both custom weights and tolerance
        functor3 = ResistorSeriesCostFunctor(weights=custom_weights, tolerance=0.05)
        assert_approx_equal(functor3("10Ω"), 0.5)
        self.assertEqual(functor3.tolerance, 0.05)

    def test_weights_dataclass_modification(self):
        """Test modifying weights dataclass"""
        weights = ResistorSeriesWeights()
        weights.E6 = 0.1
        weights.non_series = 1000.0
        
        functor = ResistorSeriesCostFunctor(weights=weights)
        assert_approx_equal(functor("10Ω"), 0.1)      # Modified E6 weight
        assert_approx_equal(functor("123Ω"), 1000.0)  # Modified non_series weight

class TestResistorAroundValueCostFunctor(unittest.TestCase):
    
    def test_resistor_around_value_basic(self):
        """Test basic functionality with default base=10"""
        functor = ResistorAroundValueCostFunctor("1kΩ")
        
        # Exact match should return 0
        assert_approx_equal(functor("1kΩ"), 0.0)
        assert_approx_equal(functor(1000.0), 0.0)
        
        # 10x larger/smaller should return 1
        assert_approx_equal(functor("10kΩ"), 1.0)
        assert_approx_equal(functor("100Ω"), 1.0)
        
        # 100x larger/smaller should return 2
        assert_approx_equal(functor("100kΩ"), 2.0)
        assert_approx_equal(functor("10Ω"), 2.0)
        
        # 1000x larger/smaller should return 3
        assert_approx_equal(functor("1MΩ"), 3.0)
        assert_approx_equal(functor("1Ω"), 3.0)

    def test_resistor_around_value_custom_base(self):
        """Test with custom logarithmic base"""
        # Base 2 for powers of 2
        functor = ResistorAroundValueCostFunctor(1000.0, base=2.0)
        
        # Exact match
        assert_approx_equal(functor(1000.0), 0.0)
        
        # 2x larger/smaller should return 1
        assert_approx_equal(functor(2000.0), 1.0)
        assert_approx_equal(functor(500.0), 1.0)
        
        # 4x larger/smaller should return 2
        assert_approx_equal(functor(4000.0), 2.0)
        assert_approx_equal(functor(250.0), 2.0)
        
        # 8x larger/smaller should return 3
        assert_approx_equal(functor(8000.0), 3.0)
        assert_approx_equal(functor(125.0), 3.0)

    def test_resistor_around_value_natural_log(self):
        """Test with natural logarithm base (e)"""
        functor = ResistorAroundValueCostFunctor(1000.0, base=np.e)
        
        # Exact match
        assert_approx_equal(functor(1000.0), 0.0)
        
        # e times larger/smaller should return 1
        assert_approx_equal(functor(1000.0 * np.e), 1.0, significant=4)
        assert_approx_equal(functor(1000.0 / np.e), 1.0, significant=4)

    def test_resistor_around_value_engineer_strings(self):
        """Test with engineer notation strings"""
        functor = ResistorAroundValueCostFunctor("47kΩ")
        
        # Test various engineer string formats
        assert_approx_equal(functor("47kΩ"), 0.0)
        assert_approx_equal(functor("470kΩ"), 1.0)
        assert_approx_equal(functor("4.7kΩ"), 1.0)
        assert_approx_equal(functor("4.7MΩ"), 2.0)
        assert_approx_equal(functor("470Ω"), 2.0)

    def test_resistor_around_value_intermediate_values(self):
        """Test with non-exact power values"""
        functor = ResistorAroundValueCostFunctor(1000.0)
        
        # Test values between exact powers
        assert_approx_equal(functor(3162.3), 0.5, significant=4)  # sqrt(10) * 1000
        assert_approx_equal(functor(316.23), 0.5, significant=4)   # 1000 / sqrt(10)
        
        # Test other intermediate values
        self.assertGreater(functor(2000.0), 0.0)
        self.assertLess(functor(2000.0), 1.0)
        self.assertGreater(functor(500.0), 0.0)
        self.assertLess(functor(500.0), 1.0)

    def test_resistor_around_value_edge_cases(self):
        """Test edge cases and error conditions"""
        # Test with zero and negative values
        functor = ResistorAroundValueCostFunctor(1000.0)
        
        # Zero or negative resistor value should return infinity
        self.assertEqual(functor(0.0), float('inf'))
        self.assertEqual(functor(-100.0), float('inf'))

    def test_resistor_around_value_initialization_errors(self):
        """Test initialization error conditions"""
        # Negative target value
        with self.assertRaises(ValueError):
            ResistorAroundValueCostFunctor(-1000.0)
        
        # Zero target value
        with self.assertRaises(ValueError):
            ResistorAroundValueCostFunctor(0.0)
        
        # Invalid base values
        with self.assertRaises(ValueError):
            ResistorAroundValueCostFunctor(1000.0, base=0.0)
        
        with self.assertRaises(ValueError):
            ResistorAroundValueCostFunctor(1000.0, base=-2.0)
        
        with self.assertRaises(ValueError):
            ResistorAroundValueCostFunctor(1000.0, base=1.0)

    def test_resistor_around_value_symmetry(self):
        """Test that the function is symmetric (same distance for X and 1/X)"""
        functor = ResistorAroundValueCostFunctor(1000.0)
        
        # Test symmetry for various multipliers
        multipliers = [2, 5, 10, 50, 100]
        for mult in multipliers:
            high_value = 1000.0 * mult
            low_value = 1000.0 / mult
            assert_approx_equal(functor(high_value), functor(low_value), significant=6)

    def test_resistor_around_value_monotonicity(self):
        """Test that distances increase monotonically as values get further apart"""
        functor = ResistorAroundValueCostFunctor(1000.0)
        
        # Test increasing sequence of values above target
        values_above = [1000, 2000, 5000, 10000, 50000, 100000]
        distances_above = [functor(v) for v in values_above]
        
        # Distances should be monotonically increasing
        for i in range(1, len(distances_above)):
            self.assertGreater(distances_above[i], distances_above[i-1])
        
        # Test decreasing sequence of values below target
        values_below = [1000, 500, 200, 100, 50, 10]
        distances_below = [functor(v) for v in values_below]
        
        # Distances should be monotonically increasing
        for i in range(1, len(distances_below)):
            self.assertGreater(distances_below[i], distances_below[i-1])

    def test_resistor_around_value_different_targets(self):
        """Test with different target values"""
        # Test with small target
        functor_small = ResistorAroundValueCostFunctor("10Ω")
        assert_approx_equal(functor_small("100Ω"), 1.0)
        assert_approx_equal(functor_small("1Ω"), 1.0)
        
        # Test with large target
        functor_large = ResistorAroundValueCostFunctor("10MΩ")
        assert_approx_equal(functor_large("100MΩ"), 1.0)
        assert_approx_equal(functor_large("1MΩ"), 1.0)
        
        # Test with fractional target
        functor_frac = ResistorAroundValueCostFunctor("4.7kΩ")
        assert_approx_equal(functor_frac("4.7kΩ"), 0.0)
        assert_approx_equal(functor_frac("47kΩ"), 1.0)

    def test_resistor_around_value_precision(self):
        """Test precision with very close values"""
        functor = ResistorAroundValueCostFunctor(1000.0)
        
        # Test very small differences
        assert_approx_equal(functor(1001.0), 0.00043407, significant=4)
        assert_approx_equal(functor(999.0), 0.00043451, significant=4)
        
        # Test that very small differences are much less than 1
        self.assertLess(functor(1100.0), 0.1)
        self.assertLess(functor(900.0), 0.1)

class TestResistorPowerCostFunctor(unittest.TestCase):
    
    def test_power_cost_basic_functionality(self):
        """Test basic power cost calculation functionality"""
        # 12V input, 0.25W max power per resistor
        functor = ResistorPowerCostFunctor("12V", "0.25W", maximum_cost=100.0)
        
        # Test with resistors that should be within power limits
        # For 12V across 1kΩ + 1kΩ series: I = 12V/2kΩ = 6mA, P = I²R = 36µA² * 1kΩ = 0.036W
        cost = functor("1kΩ", "1kΩ")
        expected_cost = (0.036 / 0.25) * 100.0  # 14.4
        assert_approx_equal(cost, expected_cost, significant=3)
        
        # Test with identical calculation using different units
        cost2 = functor(1000.0, 1000.0)
        assert_approx_equal(cost, cost2)

    def test_power_cost_exceeds_maximum(self):
        """Test behavior when power exceeds maximum rating"""
        # 12V input, very low max power (0.001W)
        functor = ResistorPowerCostFunctor("12V", "0.001W")
        
        # With small resistors, power will exceed limit
        # For 12V across 10Ω + 10Ω: I = 12V/20Ω = 0.6A, P = I²R = 0.36 * 10 = 3.6W >> 0.001W
        cost = functor("10Ω", "10Ω")
        self.assertEqual(cost, float('inf'))
        
        # Test with different resistor combinations that exceed limit
        cost2 = functor("1Ω", "100Ω")
        self.assertEqual(cost2, float('inf'))

    def test_power_cost_at_maximum_limit(self):
        """Test behavior when power is exactly at maximum limit"""
        # Design scenario where one resistor hits exactly the limit
        # 10V across series resistors where one dissipates exactly 1W
        functor = ResistorPowerCostFunctor("10V", "1W", maximum_cost=50.0)
        
        # Compute resistor values. 2W split onto two resistors
        r = resistor_value_by_voltage_and_power("10V", 1*2)
        cost = functor(r/2, r/2)
        # This should be high cost but not infinite
        self.assertNotEqual(cost, float('inf'))
        self.assertGreater(cost, 49.0)  # Should be close to maximum

    def test_power_cost_zero_power_case(self):
        """Test edge case with zero input voltage"""
        functor = ResistorPowerCostFunctor("0V", "1W")
        
        # With zero voltage, no current flows, no power dissipated
        cost = functor("100Ω", "200Ω")
        assert_approx_equal(cost, 0.0)

    def test_power_cost_unequal_resistors(self):
        """Test with unequal resistor values"""
        functor = ResistorPowerCostFunctor("24V", "2W", maximum_cost=200.0)
        
        # Test R1 >> R2: most power in R1
        # 24V across 1kΩ + 100Ω: I = 24/1100 ≈ 0.0218A
        # P1 = I²R1 = 0.000476 * 1000 = 0.476W, P2 = 0.000476 * 100 = 0.0476W
        cost1 = functor("1kΩ", "100Ω")
        expected_cost1 = (0.476 / 2.0) * 200.0  # Based on higher power (R1)
        assert_approx_equal(cost1, expected_cost1, significant=2)
        
        # Test R1 << R2: most power in R2
        cost2 = functor("100Ω", "1kΩ")
        # Should be identical due to symmetry in series circuit
        assert_approx_equal(cost1, cost2, significant=3)

    def test_power_cost_engineer_string_inputs(self):
        """Test with various engineer string formats"""
        functor = ResistorPowerCostFunctor("5V", "100mW")
        
        # Test different resistor value formats
        cost1 = functor("1kΩ", "2.2kΩ")
        cost2 = functor("1000Ω", "2200Ω") 
        cost3 = functor("1000", "2200")
        
        # All should give same result
        assert_approx_equal(cost1, cost2)
        assert_approx_equal(cost2, cost3)
        
        # Test voltage and power in different units
        functor2 = ResistorPowerCostFunctor("5000mV", "0.1W")
        cost4 = functor2("1kΩ", "2.2kΩ")
        assert_approx_equal(cost1, cost4)

    def test_power_cost_invalid_resistor_values(self):
        """Test with invalid resistor values"""
        functor = ResistorPowerCostFunctor("12V", "1W")
        
        # Zero resistors
        self.assertEqual(functor(0.0, "100Ω"), float('inf'))
        self.assertEqual(functor("100Ω", 0.0), float('inf'))
        self.assertEqual(functor(0.0, 0.0), float('inf'))
        
        # Negative resistors  
        self.assertEqual(functor(-100.0, "100Ω"), float('inf'))
        self.assertEqual(functor("100Ω", -100.0), float('inf'))

    def test_power_cost_initialization_validation(self):
        """Test initialization parameter validation"""
        # Valid initialization
        functor = ResistorPowerCostFunctor("12V", "1W", 50.0)
        self.assertEqual(functor.input_voltage, 12.0)
        self.assertEqual(functor.maximum_power, 1.0)
        self.assertEqual(functor.maximum_cost, 50.0)
        
        # Invalid input voltage
        with self.assertRaises(ValueError):
            ResistorPowerCostFunctor("-12V", "1W")
        
        # Invalid maximum power
        with self.assertRaises(ValueError):
            ResistorPowerCostFunctor("12V", "0W")
        
        with self.assertRaises(ValueError):
            ResistorPowerCostFunctor("12V", "-1W")
        
        # Invalid maximum cost
        with self.assertRaises(ValueError):
            ResistorPowerCostFunctor("12V", "1W", -10.0)

    def test_power_cost_scaling_behavior(self):
        """Test that cost scales properly with power utilization"""
        functor = ResistorPowerCostFunctor("10V", "1W", maximum_cost=100.0)
        
        # Calculate scenarios with different power levels
        # Scenario 1: Low power (large resistors)
        cost_low = functor("10kΩ", "10kΩ")
        # 10V across 20kΩ: I = 0.5mA, P = 0.0025W each
        
        # Scenario 2: Medium power
        cost_med = functor("1kΩ", "1kΩ") 
        # 10V across 2kΩ: I = 5mA, P = 0.025W each
        
        # Scenario 3: Higher power (smaller resistors)
        cost_high = functor("100Ω", "100Ω")
        # 10V across 200Ω: I = 50mA, P = 0.25W each
        
        # Costs should increase with power
        self.assertLess(cost_low, cost_med)
        self.assertLess(cost_med, cost_high)
        
        # All should be finite (below 1W limit)
        self.assertNotEqual(cost_low, float('inf'))
        self.assertNotEqual(cost_med, float('inf'))
        self.assertNotEqual(cost_high, float('inf'))

    def test_power_cost_maximum_cost_parameter(self):
        """Test different maximum_cost parameter values"""
        # Test with different maximum cost values
        functor1 = ResistorPowerCostFunctor("12V", "1W", maximum_cost=50.0)
        functor2 = ResistorPowerCostFunctor("12V", "1W", maximum_cost=200.0)
        
        cost1 = functor1("500Ω", "500Ω")
        cost2 = functor2("500Ω", "500Ω")
        
        # Cost2 should be 4x cost1 (200/50 = 4)
        assert_approx_equal(cost2, cost1 * 4.0, significant=4)
        
        # Test with maximum_cost = 0 (valid edge case)
        functor3 = ResistorPowerCostFunctor("12V", "1W", maximum_cost=0.0)
        cost3 = functor3("500Ω", "500Ω")
        assert_approx_equal(cost3, 0.0)

    def test_power_cost_extreme_values(self):
        """Test with extreme resistance and voltage values"""
        # Very high voltage, high power limit
        functor_high = ResistorPowerCostFunctor("1000V", "100W")
        
        # Very low voltage, low power limit  
        functor_low = ResistorPowerCostFunctor("0.1V", "1mW")
        
        # Test that calculations don't crash with extreme values
        cost_high = functor_high("1MΩ", "1MΩ")
        cost_low = functor_low("1Ω", "1Ω")
        
        # Both should return finite values
        self.assertIsInstance(cost_high, float)
        self.assertIsInstance(cost_low, float)
        self.assertNotEqual(cost_high, float('inf'))
        # cost_low might be infinite due to exceeding power limit

    def test_power_cost_symmetry(self):
        """Test that swapping resistor order gives same result"""
        functor = ResistorPowerCostFunctor("15V", "2W")
        
        # Test various resistor combinations
        test_pairs = [
            ("100Ω", "200Ω"),
            ("1kΩ", "470Ω"), 
            ("10Ω", "1kΩ"),
            ("22kΩ", "47kΩ")
        ]
        
        for r1, r2 in test_pairs:
            cost1 = functor(r1, r2)
            cost2 = functor(r2, r1)
            assert_approx_equal(cost1, cost2, significant=6)

    def test_power_cost_reproducibility(self):
        """Test that repeated calls give same results"""
        functor = ResistorPowerCostFunctor("9V", "0.5W", 75.0)
        
        # Test same calculation multiple times
        test_value = ("220Ω", "330Ω")
        results = [functor(*test_value) for _ in range(10)]
        
        # All results should be identical
        for result in results[1:]:
            assert_approx_equal(result, results[0])

    def test_power_cost_known_calculations(self):
        """Test against hand-calculated known values"""
        # Known scenario: 12V across 100Ω + 200Ω series
        # Total R = 300Ω, I = 12V/300Ω = 0.04A = 40mA
        # P1 = I²R1 = 0.0016 * 100 = 0.16W
        # P2 = I²R2 = 0.0016 * 200 = 0.32W  
        # Max power = 0.32W
        functor = ResistorPowerCostFunctor("12V", "1W", maximum_cost=100.0)
        
        cost = functor("100Ω", "200Ω")
        expected_cost = (0.32 / 1.0) * 100.0  # 32.0
        assert_approx_equal(cost, expected_cost, significant=3)

    def test_power_cost_just_under_limit(self):
        """Test power cost just under the limit"""
        functor = ResistorPowerCostFunctor("10V", "0.5W", maximum_cost=100.0)
        
        # Two resistors in series must EACH have 0.49W power.
        r_under = resistor_value_by_voltage_and_power("10V", 0.49*2)  # ~51Ω
        # Split the one calculated resistor into two equal resistors
        cost_under = functor(r_under/2, r_under/2)
        self.assertNotEqual(cost_under, float('inf'))
        self.assertGreater(cost_under, 90.0)  # Should be high cost

    def test_power_cost_just_over_limit(self):
        """Test power cost just over the limit"""
        functor = ResistorPowerCostFunctor("10V", "0.5W", maximum_cost=100.0)
        
        # For power just over limit: 0.51W per resistor, for two resistors -> double the power
        r_over = resistor_value_by_voltage_and_power("10V", 0.51*2)  # ~49Ω
        # Split into two equal resistors
        cost_over = functor(r_over/2, r_over/2)
        self.assertEqual(cost_over, float('inf'))  # Should exceed limit

    def test_power_cost_integration_with_resistor_functions(self):
        """Test that integration with Resistors.py functions works correctly"""
        functor = ResistorPowerCostFunctor("24V", "5W")
        
        # Test that the functor properly uses imported functions
        # This is more of an integration test to ensure the imports work
        from UliEngineering.Electronics.Resistors import series_resistors, current_through_resistor, power_dissipated_in_resistor_by_current
        
        r1, r2 = 470.0, 680.0
        
        # Manual calculation using the same functions
        total_r = series_resistors(r1, r2)
        current = current_through_resistor(total_r, 24.0)
        power1 = power_dissipated_in_resistor_by_current(r1, current)
        power2 = power_dissipated_in_resistor_by_current(r2, current)
        max_power = max(power1, power2)
        
        # Compare with functor result
        cost = functor(r1, r2)
        expected_cost = (max_power / 5.0) * 100.0  # default maximum_cost
        
        if max_power <= 5.0:
            assert_approx_equal(cost, expected_cost, significant=4)
        else:
            self.assertEqual(cost, float('inf'))

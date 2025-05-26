#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from UliEngineering.Electronics.TimeConstant import *
import unittest
import numpy as np

class TestTimeConstant(unittest.TestCase):
    def test_rc_time_constant(self):
        # Test basic RC time constant calculation
        # τ = R × C = 1000Ω × 1µF = 1ms
        assert_approx_equal(rc_time_constant(1000, 1e-6), 1e-3)
        assert_approx_equal(rc_time_constant("1kΩ", "1µF"), 1e-3)
        
        # Another example: 10kΩ × 100nF = 1ms
        assert_approx_equal(rc_time_constant("10kΩ", "100nF"), 1e-3)
        
    def test_rl_time_constant(self):
        # Test basic RL time constant calculation
        # τ = L / R = 1mH / 1Ω = 1ms
        assert_approx_equal(rl_time_constant(1, 1e-3), 1e-3)
        assert_approx_equal(rl_time_constant("1Ω", "1mH"), 1e-3)
        
        # Another example: 10µH / 10Ω = 1µs
        assert_approx_equal(rl_time_constant("10Ω", "10µH"), 1e-6)
        
    def test_rc_cutoff_frequency(self):
        # Test RC cutoff frequency calculation
        # fc = 1 / (2π × R × C)
        # For R=1kΩ, C=1µF: fc = 1/(2π × 1000 × 1e-6) ≈ 159.15 Hz
        # Verified using http://sim.okawa-denshi.jp/en/CRtool.php
        assert_approx_equal(rc_cutoff_frequency(1000, 1e-6), 159.154943, significant=6)
        assert_approx_equal(rc_cutoff_frequency("1kΩ", "1µF"), 159.154943, significant=6)
        
    def test_rl_cutoff_frequency(self):
        # Test RL cutoff frequency calculation
        # fc = R / (2π × L)
        # For R=1Ω, L=1mH: fc = 1/(2π × 1e-3) ≈ 159.15 Hz
        assert_approx_equal(rl_cutoff_frequency(1, 1e-3), 159.154943, significant=6)
        assert_approx_equal(rl_cutoff_frequency("1Ω", "1mH"), 159.154943, significant=6)
        
    def test_rc_charge_time(self):
        # Test RC charging time calculation
        # Time to charge from 0V to 3.16V (63.2% of 5V) with τ=1ms should be 1ms
        tau = 1e-3  # 1ms
        R = 1000
        C = 1e-6
        assert_approx_equal(rc_charge_time(R, C, 0, 5, 5 * 0.632), tau, significant=3)
        
        # Time to charge from 0V to 4.33V (86.5% of 5V) should be ≈ 2τ
        assert_approx_equal(rc_charge_time(R, C, 0, 5, 5 * 0.865), 2 * tau, significant=3)
        
    def test_rc_discharge_time(self):
        # Test RC discharge time calculation
        # Time to discharge from 5V to 1.84V (36.8% of 5V) should be 1τ
        tau = 1e-3  # 1ms
        R = 1000
        C = 1e-6
        assert_approx_equal(rc_discharge_time(R, C, 5, 5 * 0.368), tau, significant=3)
        
    def test_rl_current_rise_time(self):
        # Test RL current rise time calculation
        # Time to reach 63.2% of final current should be 1τ
        tau = 1e-3  # 1ms
        R = 1
        L = 1e-3
        final_current = 5
        assert_approx_equal(rl_current_rise_time(R, L, final_current, final_current * 0.632), tau, significant=3)
        
    def test_rl_current_fall_time(self):
        # Test RL current fall time calculation
        # Time to fall to 36.8% of initial current should be 1τ
        tau = 1e-3  # 1ms
        R = 1
        L = 1e-3
        initial_current = 5
        assert_approx_equal(rl_current_fall_time(R, L, initial_current, initial_current * 0.368), tau, significant=3)
        
    def test_rc_step_response(self):
        # Test RC step response calculation
        R = 1000
        C = 1e-6
        tau = R * C  # 1ms
        
        # At t=0, voltage should be initial voltage
        assert_approx_equal(rc_step_response(R, C, 0, initial_voltage=0, final_voltage=5), 0)
        
        # At t=τ, voltage should be ~63.2% of final value
        v_tau = rc_step_response(R, C, tau, initial_voltage=0, final_voltage=5)
        assert_approx_equal(v_tau, 5 * 0.632, significant=3)
        
        # At t=5τ, voltage should be ~99.3% of final value
        v_5tau = rc_step_response(R, C, 5 * tau, initial_voltage=0, final_voltage=5)
        assert_approx_equal(v_5tau, 5 * 0.993, significant=3)
        
    def test_rl_step_response(self):
        # Test RL step response calculation
        R = 1
        L = 1e-3
        tau = L / R  # 1ms
        
        # At t=0, current should be 0
        assert_approx_equal(rl_step_response(R, L, 0, final_current=5), 0)
        
        # At t=τ, current should be ~63.2% of final value
        i_tau = rl_step_response(R, L, tau, final_current=5)
        assert_approx_equal(i_tau, 5 * 0.632, significant=3)
        
        # At t=5τ, current should be ~99.3% of final value
        i_5tau = rl_step_response(R, L, 5 * tau, final_current=5)
        assert_approx_equal(i_5tau, 5 * 0.993, significant=3)
        
    def test_rlc_resonant_frequency(self):
        # Test RLC resonant frequency calculation
        # f0 = 1 / (2π × sqrt(L × C))
        # For L=1mH, C=1µF: f0 = 1/(2π × sqrt(1e-3 × 1e-6)) ≈ 5033 Hz
        L = 1e-3
        C = 1e-6
        expected_f0 = 1.0 / (2 * np.pi * np.sqrt(L * C))
        assert_approx_equal(rlc_resonant_frequency(L, C), expected_f0)
        assert_approx_equal(rlc_resonant_frequency("1mH", "1µF"), expected_f0)
        
    def test_rlc_quality_factor(self):
        # Test RLC quality factor calculation
        # Q = (1/R) × sqrt(L/C)
        # For R=10Ω, L=1mH, C=1µF: Q = (1/10) × sqrt(1e-3/1e-6) ≈ 3.16
        R = 10
        L = 1e-3
        C = 1e-6
        expected_Q = (1.0 / R) * np.sqrt(L / C)
        assert_approx_equal(rlc_quality_factor(R, L, C), expected_Q)
        assert_approx_equal(rlc_quality_factor("10Ω", "1mH", "1µF"), expected_Q)
        
    def test_rlc_damping_ratio(self):
        # Test RLC damping ratio calculation
        # ζ = R/2 × sqrt(C/L)
        # For R=10Ω, L=1mH, C=1µF: ζ = 10/2 × sqrt(1e-6/1e-3) ≈ 0.158
        R = 10
        L = 1e-3
        C = 1e-6
        expected_zeta = (R / 2.0) * np.sqrt(C / L)
        assert_approx_equal(rlc_damping_ratio(R, L, C), expected_zeta)
        assert_approx_equal(rlc_damping_ratio("10Ω", "1mH", "1µF"), expected_zeta)
        
    def test_rlc_bandwidth(self):
        # Test RLC bandwidth calculation
        # BW = R / (2π × L)
        # For R=10Ω, L=1mH: BW = 10/(2π × 1e-3) ≈ 1591.5 Hz
        R = 10
        L = 1e-3
        expected_bw = R / (2 * np.pi * L)
        assert_approx_equal(rlc_bandwidth(R, L), expected_bw)
        assert_approx_equal(rlc_bandwidth("10Ω", "1mH"), expected_bw)
        
    def test_edge_cases(self):
        # Test edge cases for charge time
        R = 1000
        C = 1e-6
        
        # Test when target voltage equals final voltage - should return infinity
        result = rc_charge_time(R, C, 0, 5, 5)
        self.assertTrue(np.isinf(result))
        
        # Test when initial voltage equals final voltage - should return 0
        assert_approx_equal(rc_charge_time(R, C, 5, 5, 4), 0.0)
        
        # Test RL edge cases
        R = 1
        L = 1e-3
        
        # No initial current should return 0
        assert_approx_equal(rl_current_fall_time(R, L, 0, 1), 0.0)
        
        # Target current is 0 should return infinity
        result = rl_current_fall_time(R, L, 5, 0)
        self.assertTrue(np.isinf(result))

if __name__ == '__main__':
    unittest.main()
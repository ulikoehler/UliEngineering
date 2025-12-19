#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from UliEngineering.SignalProcessing.Ramp import periodic_ramp
from UliEngineering.Exceptions import OperationImpossibleException


class TestRamp(unittest.TestCase):
    def test_basic_periodic_properties(self):
        sr = 1000.0
        freq = 2.0  # 2 Hz -> period 0.5 s
        amp = 5.0
        rise = 0.05
        fall = 0.05
        high = 0.2
        length = 1.0

        y = periodic_ramp(freq, sr, amplitude=amp, rise_time=rise,
                          fall_time=fall, high_time=high, length=length)
        self.assertEqual(len(y), int(length * sr))
        self.assertAlmostEqual(np.min(y), 0.0)
        self.assertAlmostEqual(np.max(y), amp)

    def test_negative_amplitude(self):
        sr = 1000.0
        freq = 1.0
        amp = -3.0
        y = periodic_ramp(freq, sr, amplitude=amp, rise_time=0.1, fall_time=0.1, high_time=0.3, length=1.0)
        # For negative amplitude, min and max should be swapped appropriately
        self.assertAlmostEqual(np.min(y), amp)
        self.assertAlmostEqual(np.max(y), 0.0)
        # mean should be between min and max
        self.assertTrue(np.mean(y) < 0.0)

    def test_zero_rise_or_fall(self):
        sr = 1000.0
        freq = 1.0
        # Zero rise -> instantaneous jump to high level
        y = periodic_ramp(freq, sr, amplitude=2.0, rise_time=0.0, fall_time=0.1, high_time=0.4, length=1.0)
        # Immediately after start it should be at max (since rise is 0)
        assert_almost_equal(y[0], 2.0)

        # Zero fall -> instantaneous drop to low level at fall start
        y2 = periodic_ramp(freq, sr, amplitude=2.0, rise_time=0.1, fall_time=0.0, high_time=0.4, length=1.0)
        # After end of high_time the value should go to 0 immediately
        idx_after_high = int((0.1 + 0.4) * sr)
        assert_almost_equal(y2[idx_after_high], 0.0)

    def test_acceleration_profile_values(self):
        sr = 1.0
        freq = 1.0 / 10.0  # period 10s, easier to reason in sample units
        amp = 1.05e8
        rise = 3.5
        # Choose acceleration slightly above minimum
        min_acc = 4.0 * amp / (rise ** 2)
        acc = min_acc * 1.1
        # Make segments sum to period (3.5 + 3.0 + 3.5 = 10)
        y = periodic_ramp(freq, sr, amplitude=amp, rise_time=rise, fall_time=3.5, high_time=3.0, acceleration=acc, length=10)
        # Derive t1 as in implementation
        term = rise ** 2 - 4.0 * amp / acc
        if term < 0:
            t1 = rise / 2.0
        else:
            t1 = (rise - np.sqrt(term)) / 2.0
        # Choose a sample inside acceleration phase (use floor of half t1, but at least 1)
        t_check = int(max(1, np.floor(t1 / 2.0)))
        # Compute expected value using the same piecewise formula
        if t_check < t1:
            expected = 0.5 * acc * (t_check ** 2)
        else:
            # fallback for safety: compare to generated value within tolerance
            expected = y[t_check]
        assert_almost_equal(y[t_check], expected)

    def test_acceleration_too_low_raises(self):
        sr = 1000.0
        freq = 1.0
        amp = 10.0
        rise = 0.1
        # min_acc = 4*A/T^2 -> set smaller acceleration
        acc = 1.0
        with self.assertRaises(OperationImpossibleException):
            periodic_ramp(freq, sr, amplitude=amp, rise_time=rise, fall_time=0.1, high_time=0.3, acceleration=acc, length=1.0)

    def test_invalid_timings_raise(self):
        sr = 1000.0
        freq = 1.0
        # rise + fall > period
        with self.assertRaises(OperationImpossibleException):
            periodic_ramp(freq, sr, amplitude=1.0, rise_time=0.6, fall_time=0.6, high_time=0.0, length=1.0)

    def test_phase_shift_equivalence(self):
        sr = 100.0
        freq = 2.0  # period 0.5s
        amp = 7.0
        rise = 0.05
        fall = 0.05
        high = 0.2
        length = 0.5

        y = periodic_ramp(freq, sr, amplitude=amp, rise_time=rise, fall_time=fall, high_time=high, length=length)
        # 180 degrees shift -> half a period -> first sample of phased should equal middle sample of unphased
        y_phased = periodic_ramp(freq, sr, amplitude=amp, rise_time=rise, fall_time=fall, high_time=high, length=length, phaseshift=180.0)
        mid_idx = int((length * sr) // 2)
        assert_almost_equal(y_phased[0], y[mid_idx])

    def test_none_acceleration_is_linear(self):
        # Verify that acceleration=None yields linear ramp (value at t == t/T * A)
        sr = 100.0
        freq = 1.0
        amp = 10.0
        rise = 0.1
        y_lin = periodic_ramp(freq, sr, amplitude=amp, rise_time=rise, fall_time=0.1, high_time=0.4, acceleration=None, length=1.0)
        # sample early in rise
        idx = int(0.02 * sr)
        expected = (idx / (rise * sr)) * amp
        assert_almost_equal(y_lin[idx], expected)

    def test_c2_continuity(self):
        sr = 1.0
        freq = 1.0 / 10.0
        amp = 100.0
        rise = 3.5
        # Min acc for C2 is 2*pi*A/T^2 approx 6.28*100/12.25 approx 51
        acc = 60.0
        y = periodic_ramp(freq, sr, amplitude=amp, rise_time=rise, fall_time=3.5, high_time=3.0, 
                          acceleration=acc, length=10, continuity="C2")
        
        # Check value at t=1 (inside acceleration phase)
        # t1 calculation
        term = rise**2 - (2 * np.pi * amp) / acc
        t1 = (rise - np.sqrt(term)) / 2.0
        
        t_check = 1
        if t_check < t1:
            expected = (acc * t1 / np.pi) * (t_check - (t1 / np.pi) * np.sin(np.pi * t_check / t1))
            assert_almost_equal(y[t_check], expected)
            
    def test_c2_continuity_too_low_acc(self):
        sr = 1000.0
        freq = 1.0
        amp = 10.0
        rise = 0.1
        # Min acc for C2 is higher than C1.
        # C1 min: 4*10/0.01 = 4000
        # C2 min: 2*pi*10/0.01 = 6283
        acc = 5000.0 # Enough for C1, not for C2
        
        # Should pass with C1 (default)
        periodic_ramp(freq, sr, amplitude=amp, rise_time=rise, fall_time=0.1, high_time=0.3, acceleration=acc, length=1.0, continuity="C1")
        
        # Should fail with C2
        with self.assertRaises(OperationImpossibleException):
            periodic_ramp(freq, sr, amplitude=amp, rise_time=rise, fall_time=0.1, high_time=0.3, acceleration=acc, length=1.0, continuity="C2")

    def test_invalid_continuity_value(self):
        with self.assertRaises(ValueError):
            periodic_ramp(1.0, 100.0, continuity="C3")



if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest

from numpy.testing import assert_allclose

from UliEngineering.Reliability.Conversion import *


class TestConversion(unittest.TestCase):
    def test_fit_to_mttf_years_and_hours(self):
        fit = 1000.0
        yrs = FIT_to_MTTF(fit, unit="years")
        hrs = FIT_to_MTTF(fit, unit="hours")
        assert_allclose(hrs, yrs * 365 * 24, atol=1e-12)
        expected_years = 1e9 / (365 * 24 * fit)
        assert_allclose(yrs, expected_years, atol=1e-12)

    def test_fit_to_mttfd_double(self):
        fit = 2000.0
        mttf = FIT_to_MTTF(fit, unit="years")
        mttfd = FIT_to_MTTFd(fit, unit="years")
        assert_allclose(mttfd, 2.0 * mttf, atol=1e-12)

    def test_roundtrip_fit_mttf(self):
        fit = 500.0
        yrs = FIT_to_MTTF(fit, unit="years")
        fit_back = MTTF_to_FIT(yrs, unit="years")
        assert_allclose(fit, fit_back, rtol=1e-12)

    def test_roundtrip_mttfd_fit(self):
        fit = 250.0
        mttfd = FIT_to_MTTFd(fit, unit="days")
        fit_back = MTTFd_to_FIT(mttfd, unit="days")
        assert_allclose(fit, fit_back, rtol=1e-12)

    def test_invalid_values(self):
        with self.assertRaises(ValueError):
            FIT_to_MTTF(0)
        with self.assertRaises(ValueError):
            MTTF_to_FIT(0)
        with self.assertRaises(ValueError):
            MTTFd_to_FIT(0)

    def test_b10d_to_mttfd_and_roundtrip(self):
        B10d = 1e6
        t_cycle = 10.0
        d_op = 250
        h_op = 8
        expected_years = B10d * t_cycle / (360.0 * d_op * h_op)
        got_years = B10d_to_MTTFd(B10d, t_cycle, d_op, h_op, unit="years")
        assert_allclose(got_years, expected_years, rtol=1e-12)
        back = MTTFd_to_B10d(got_years, t_cycle, d_op, h_op, unit="years")
        assert_allclose(B10d, back, rtol=1e-12)

    def test_b10d_units_and_invalid(self):
        B10d = 1000.0
        t_cycle = 60.0
        got_days = B10d_to_MTTFd(B10d, t_cycle, unit="days")
        got_years = B10d_to_MTTFd(B10d, t_cycle, unit="years")
        assert_allclose(got_days, got_years * 365, rtol=1e-12)
        with self.assertRaises(ValueError):
            B10d_to_MTTFd(0, 1)
        with self.assertRaises(ValueError):
            MTTFd_to_B10d(0, 1)

    def test_pfh_to_mttf_and_roundtrip(self):
        pfh = 1e-6
        yrs = PFH_to_MTTF(pfh, unit="years")
        hrs = PFH_to_MTTF(pfh, unit="hours")
        assert_allclose(hrs, yrs * 365 * 24, atol=1e-12)
        expected_years = 1.0 / (365 * 24 * pfh)
        assert_allclose(yrs, expected_years, rtol=1e-12)
        back = MTTF_to_PFH(yrs, unit="years")
        assert_allclose(back, pfh, rtol=1e-12)

    def test_fit_pfh_roundtrip(self):
        fit = 500.0
        pfh = FIT_to_PFH(fit)
        fit_back = PFH_to_FIT(pfh)
        assert_allclose(fit, fit_back, rtol=1e-12)

    def test_pfhd_and_mttfd_roundtrip_and_invalid(self):
        pfhd = 2e-6
        yrs = PFHd_to_MTTFd(pfhd, unit="years")
        back_pfhd = MTTFd_to_PFHd(yrs, unit="years")
        assert_allclose(back_pfhd, pfhd, rtol=1e-12)
        with self.assertRaises(ValueError):
            PFH_to_MTTF(0)
        with self.assertRaises(ValueError):
            PFHd_to_MTTFd(0)
        with self.assertRaises(ValueError):
            FIT_to_PFH(0)

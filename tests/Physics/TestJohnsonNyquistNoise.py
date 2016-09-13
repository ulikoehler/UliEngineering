#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from nose.tools import assert_equal
from UliEngineering.Physics.JohnsonNyquistNoise import *
from UliEngineering.EngineerIO import auto_format

class TestJohnsonNyquistNoise(object):
    def test_johnson_nyquist_noise_current(self):
        v = johnson_nyquist_noise_current("20 MΩ", "10000 ΔHz", "20 °C")
        assert_approx_equal(v, 2.84512e-12, significant=5)
        assert_equal(auto_format(johnson_nyquist_noise_current, "20 MΩ", "10000 ΔHz", "20 °C"), "2.85 pA")

    def test_johnson_nyquist_noise_voltage(self):
        v = johnson_nyquist_noise_voltage("20 MΩ", "10000 ΔHz", "20 °C")
        assert_equal(auto_format(johnson_nyquist_noise_voltage, "20 MΩ", "10000 ΔHz", "20 °C"), "56.9 µV")
        assert_approx_equal(v, 56.9025e-6, significant=5)

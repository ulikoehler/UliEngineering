#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from nose.tools import assert_equal
from UliEngineering.Electronics.OpAmp import *
from UliEngineering.EngineerIO import autoFormat

class TestOpAmp(object):
    def test_summing_amplifier_noninv(self):
        v = summing_amplifier_noninv("2.5V", "500mV", "1kΩ", "1kΩ", "1kΩ", "1kΩ")
        assert_approx_equal(v, 3.0)
        v = summing_amplifier_noninv(2.5, 0.5, 1e3, 1e3, 300, 1e3)
        assert_approx_equal(v, 6.5)
        assert_equal(autoFormat(summing_amplifier_noninv, 2.5, 0.5, 1e3, 1e3, 300, 1e3), "6.50 V")

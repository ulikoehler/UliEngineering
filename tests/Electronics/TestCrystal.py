#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from nose.tools import assert_equal
from UliEngineering.Electronics.Crystal import *
from UliEngineering.EngineerIO import auto_format
import numpy as np

class TestCrystal(object):
    def test_load_capacitor(self):
        # Example from https://blog.adafruit.com/2012/01/24/choosing-the-right-crystal-and-caps-for-your-design/
        assert_equal(auto_format(load_capacitors, "6 pF", cpin="3 pF", cstray="2pF"), '5.00 pF')

    def test_actual_load_capacitance(self):
        assert_equal(auto_format(actual_load_capacitance, "5 pF", cpin="3 pF", cstray="2pF"), '6.00 pF')

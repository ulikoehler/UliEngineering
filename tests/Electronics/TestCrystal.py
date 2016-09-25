#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from nose.tools import assert_equal
from UliEngineering.Electronics.Crystal import *
from UliEngineering.EngineerIO import auto_format
import numpy as np

class TestNoiseDensity(object):
    def test_load_capacitory(self):
        # Example from https://blog.adafruit.com/2012/01/24/choosing-the-right-crystal-and-caps-for-your-design/
        assert_approx_equal(load_capacitors(8e-12, 3e-12), 10e-12)
        assert_approx_equal(load_capacitors("8 pF", "3 pF"), 10e-12)

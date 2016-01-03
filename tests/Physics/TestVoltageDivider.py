#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from nose.tools import assert_equal
from UliEngineering.Physics.VoltageDivider import *

class TestNoiseDensity(object):
    def test_unloadedVoltageDividerRatio(self):
        assert_approx_equal(unloadedVoltageDividerRatio(1000.0, 1000.0), 0.5)

    def test_loadedVoltageDividerRatio(self):
        assert_approx_equal(loadedVoltageDividerRatio(1000.0, 1000.0, 1e60), 0.5)
        assert_approx_equal(loadedVoltageDividerRatio(1000.0, 1000.0, 1000.0), 0.6666666666666666)
        assert_approx_equal(loadedVoltageDividerRatio("1kΩ", "1kΩ", "10 MΩ"), 0.500024998)

    def test_computeTopResistor(self):
        assert_approx_equal(computeTopResistor(1000.0, 0.5), 1000.0)

    def test_computeBottomResistor(self):
        assert_approx_equal(computeBottomResistor(1000.0, 0.5), 1000.0)



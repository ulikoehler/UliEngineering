#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from nose.tools import assert_equal, assert_true
from UliEngineering.Electronics.Hysteresis import *

class TestHysteresis(object):
    def test_hysteresis_thresholds(self):
        # 1e300: Near-infinite resistor should not affect ratio
        assert_allclose(hysteresis_threshold_ratios(1e3, 1e3, 1e300), (0.5, 0.5))
        assert_allclose(hysteresis_threshold_voltages(1e3, 1e3, 1e300, 5.0), (2.5, 2.5))
        assert_allclose(hysteresis_threshold_factors(1e3, 1e3, 1e300), (1.0, 1.0))
        # More realistic values
        assert_allclose(hysteresis_threshold_ratios(1e3, 1e3, 1e3), (0.3333333333, 0.6666666666))
        assert_allclose(hysteresis_threshold_voltages(1e3, 1e3, 1e3, 5.0), (0.3333333333*5., 0.6666666666*5.))
        assert_allclose(hysteresis_threshold_factors(1e3, 1e3, 1e3), (0.3333333333/.5, 0.6666666666/.5))

    def test_hysteresis_opendrain(self):
        # 1e300: Near-infinite resistor should not affect ratio
        assert_allclose(hysteresis_threshold_ratios_opendrain(1e3, 1e3, 1e300), (0.5, 0.5))
        assert_allclose(hysteresis_threshold_voltages_opendrain(1e3, 1e3, 1e300, 5.0), (2.5, 2.5))
        # More realistic values
        assert_allclose(hysteresis_threshold_ratios_opendrain(1e3, 1e3, 1e3), (0.3333333333, 0.5))
        assert_allclose(hysteresis_threshold_voltages_opendrain(1e3, 1e3, 1e3, 5.0), (0.3333333333*5., 0.5*5.))

    def test_hysteresis_resistor(self):
        assert_allclose(hysteresis_resistor(1e3, 1e3, 0.1), 4500)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose, raises
from nose.tools import assert_equal
from UliEngineering.Electronics.PowerFactor import *
from UliEngineering.EngineerIO import auto_format
import numpy as np

class TestPowerFactor(object):
    def test_power_factor_by_phase_angle(self):
        assert_approx_equal(power_factor_by_phase_angle(0.0), 1.0)
        assert_approx_equal(power_factor_by_phase_angle("0"), 1.0)
        assert_approx_equal(power_factor_by_phase_angle("0°"), 1.0)
        assert_approx_equal(power_factor_by_phase_angle("0°", unit="deg"), 1.0)
        assert_approx_equal(power_factor_by_phase_angle("0°", unit="degrees"), 1.0)
        assert_approx_equal(power_factor_by_phase_angle("0°", unit="rad"), 1.0)
        assert_approx_equal(power_factor_by_phase_angle("0°", unit="radiant"), 1.0)
        assert_approx_equal(power_factor_by_phase_angle("90°"), 0.0)
        assert_approx_equal(power_factor_by_phase_angle("90°", unit="deg"), 0.0)
        assert_approx_equal(power_factor_by_phase_angle(90), 1.0)
        assert_approx_equal(power_factor_by_phase_angle(90+360), 1.0)
        assert_approx_equal(power_factor_by_phase_angle(90-360), 1.0)
        assert_approx_equal(power_factor_by_phase_angle(np.pi/2., unit="rad"), 1.0)

    @raises(ValueError)
    def test_power_factor_by_phase_angle_bad_unit(self):
        power_factor_by_phase_angle(unit="nosuchunit")


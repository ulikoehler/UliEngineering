#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose, raises
from nose.tools import assert_equal
from UliEngineering.Electronics.PowerFactor import *
from UliEngineering.EngineerIO import auto_format
import numpy as np
import unittest

class TestPowerFactor(unittest.TestCase):
    def test_power_factor_by_phase_angle(self):
        assert_allclose(power_factor_by_phase_angle(0.0), 1.0, atol=1e-15)
        assert_allclose(power_factor_by_phase_angle("0"), 1.0, atol=1e-15)
        assert_allclose(power_factor_by_phase_angle("0°"), 1.0, atol=1e-15)
        assert_allclose(power_factor_by_phase_angle("0°", unit="deg"), 1.0, atol=1e-15)
        assert_allclose(power_factor_by_phase_angle("0°", unit="degrees"), 1.0, atol=1e-15)
        assert_allclose(power_factor_by_phase_angle("0°", unit="rad"), 1.0, atol=1e-15)
        assert_allclose(power_factor_by_phase_angle("0°", unit="radiant"), 1.0, atol=1e-15)
        assert_allclose(power_factor_by_phase_angle("90°"), 0.0, atol=1e-15)
        assert_allclose(power_factor_by_phase_angle("90°", unit="deg"), 0.0, atol=1e-15)
        assert_allclose(power_factor_by_phase_angle(90), 0.0, atol=1e-15)
        assert_allclose(power_factor_by_phase_angle(90+360), 0.0, atol=1e-15)
        assert_allclose(power_factor_by_phase_angle(90-360), 0.0, atol=1e-15)
        assert_allclose(power_factor_by_phase_angle(np.pi/2., unit="rad"), 0.0, atol=1e-15)

    @raises(ValueError)
    def test_power_factor_by_phase_angle_bad_unit(self):
        power_factor_by_phase_angle(unit="nosuchunit")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose, raises
from UliEngineering.Electronics.Power import *
from UliEngineering.EngineerIO import auto_format
import numpy as np
import unittest

class TestPower(unittest.TestCase):
    def test_current_by_power(self):
        assert_allclose(current_by_power(25), 25 / 230, atol=1e-15)
        assert_allclose(current_by_power(25, 230), 25 / 230, atol=1e-15)
        assert_allclose(current_by_power(25, 100), 25 / 100, atol=1e-15)
        assert_allclose(current_by_power("25 W", "100 V"), 25 / 100, atol=1e-15)

    def test_power_by_current_and_voltage(self):
        assert_allclose(power_by_current_and_voltage(1, 10), 1*10, atol=1e-15)
        assert_allclose(power_by_current_and_voltage(1, 230), 1 * 230, atol=1e-15)
        assert_allclose(power_by_current_and_voltage(0.2, 230), 0.2 * 230, atol=1e-15)
        assert_allclose(power_by_current_and_voltage(0.2), 0.2 * 230, atol=1e-15)
        assert_allclose(power_by_current_and_voltage("0.2 A", "230 V"), 0.2 * 230, atol=1e-15)


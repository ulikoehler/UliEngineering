#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_allclose
from UliEngineering.Math.Decibel import *
from parameterized import parameterized
import functools
import numpy as np
import unittest
import pytest

class TestDecibel(unittest.TestCase):
    def test_ratio_to_db(self):
        assert_allclose(12, ratio_to_dB(4, factor=dBFactor.Field), 0.05)
        assert_allclose(6, ratio_to_dB(2, factor=dBFactor.Field), 0.05)
        assert_allclose(-6, ratio_to_dB(0.5, factor=dBFactor.Field), 0.05)
        assert_allclose(-12, ratio_to_dB(0.25, factor=dBFactor.Field), 0.05)

        assert_allclose(6, ratio_to_dB(4, factor=dBFactor.Power), 0.05)
        assert_allclose(3, ratio_to_dB(2, factor=dBFactor.Power), 0.05)
        assert_allclose(-3, ratio_to_dB(0.5, factor=dBFactor.Power), 0.05)
        assert_allclose(-6, ratio_to_dB(0.25, factor=dBFactor.Power), 0.05)

    @pytest.mark.filterwarnings("ignore:divide by zero encountered in log10")
    def test_ratio_to_dB_infinite(self):
        self.assertEqual(-np.inf, ratio_to_dB(0))
        self.assertEqual(-np.inf, ratio_to_dB(0, factor=dBFactor.Power))
        self.assertEqual(-np.inf, ratio_to_dB(-5))
        self.assertEqual(-np.inf, ratio_to_dB(-5, factor=dBFactor.Power))

    def test_value_to_db(self):
        # Test v0 = 1
        assert_allclose(12, value_to_dB(4, 1, factor=dBFactor.Field), 0.05)
        assert_allclose(6, value_to_dB(2, 1, factor=dBFactor.Field), 0.05)
        assert_allclose(-6, value_to_dB(0.5, 1, factor=dBFactor.Field), 0.05)
        assert_allclose(-12, value_to_dB(0.25, 1, factor=dBFactor.Field), 0.05)

        assert_allclose(6, value_to_dB(4, 1, factor=dBFactor.Power), 0.05)
        assert_allclose(3, value_to_dB(2, 1, factor=dBFactor.Power), 0.05)
        assert_allclose(-3, value_to_dB(0.5, 1, factor=dBFactor.Power), 0.05)
        assert_allclose(-6, value_to_dB(0.25, 1, factor=dBFactor.Power), 0.05)
        # Test v0 != 1
        assert_allclose(6, value_to_dB(4, 2, factor=dBFactor.Field), 0.05)
        assert_allclose(0, value_to_dB(2, 2, factor=dBFactor.Field), 0.05)
        assert_allclose(-6, value_to_dB(1, 2, factor=dBFactor.Field), 0.05)
        assert_allclose(-12, value_to_dB(0.5, 2, factor=dBFactor.Field), 0.05)
        assert_allclose(-18, value_to_dB(0.25, 2, factor=dBFactor.Field), 0.05)

        assert_allclose(3, value_to_dB(4, 2, factor=dBFactor.Power), 0.05)
        assert_allclose(0, value_to_dB(2, 2, factor=dBFactor.Power), 0.05)
        assert_allclose(-3, value_to_dB(1, 2, factor=dBFactor.Power), 0.05)
        assert_allclose(-6, value_to_dB(0.5, 2, factor=dBFactor.Power), 0.05)
        assert_allclose(-9, value_to_dB(0.25, 2, factor=dBFactor.Power), 0.05)
        # Test string
        assert_allclose(6, value_to_dB("4 V", "2 V", factor=dBFactor.Field), 0.05)
    
    def test_dB_to_ratio(self):
        self.assertAlmostEqual(dB_to_ratio(0), 1.0)
        self.assertAlmostEqual(dB_to_ratio(20), 10.0)
        self.assertAlmostEqual(dB_to_ratio(-20), 0.1)
        self.assertAlmostEqual(dB_to_ratio(20, factor=dBFactor.Power), 100.0)
        self.assertAlmostEqual(dB_to_ratio(-20, factor=dBFactor.Power), 0.01)

    def test_dB_to_value(self):
        self.assertAlmostEqual(dB_to_value(0), 1.0)
        self.assertAlmostEqual(dB_to_value(20), 10.0)
        self.assertAlmostEqual(dB_to_value(-20), 0.1)
        self.assertAlmostEqual(dB_to_value(20, v0=100), 1000.0)
        self.assertAlmostEqual(dB_to_value(-20, v0=100), 10.0)

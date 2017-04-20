#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_allclose
from nose.tools import assert_equal, assert_tuple_equal, assert_is_none, assert_true, assert_false, raises, assert_in, assert_not_in
from UliEngineering.Math.Decibel import *
from nose_parameterized import parameterized
import functools
import numpy as np

class TestDecibel(object):
    def test_ratio_to_db(self):
        assert_allclose(12, ratio_to_db_field(4), 0.05)
        assert_allclose(6, ratio_to_db_field(2), 0.05)
        assert_allclose(-6, ratio_to_db_field(0.5), 0.05)
        assert_allclose(-12, ratio_to_db_field(0.25), 0.05)

        assert_allclose(6, ratio_to_db_power(4), 0.05)
        assert_allclose(3, ratio_to_db_power(2), 0.05)
        assert_allclose(-3, ratio_to_db_power(0.5), 0.05)
        assert_allclose(-6, ratio_to_db_power(0.25), 0.05)

    def test_value_to_db(self):
        # Test v0 = 1
        assert_allclose(12, value_to_db_field(4, 1), 0.05)
        assert_allclose(6, value_to_db_field(2, 1), 0.05)
        assert_allclose(-6, value_to_db_field(0.5, 1), 0.05)
        assert_allclose(-12, value_to_db_field(0.25, 1), 0.05)

        assert_allclose(6, value_to_db_power(4, 1), 0.05)
        assert_allclose(3, value_to_db_power(2, 1), 0.05)
        assert_allclose(-3, value_to_db_power(0.5, 1), 0.05)
        assert_allclose(-6, value_to_db_power(0.25, 1), 0.05)
        # Test v0 != 1
        assert_allclose(6, value_to_db_field(4, 2), 0.05)
        assert_allclose(0, value_to_db_field(2, 2), 0.05)
        assert_allclose(-6, value_to_db_field(1, 2), 0.05)
        assert_allclose(-12, value_to_db_field(0.5, 2), 0.05)
        assert_allclose(-18, value_to_db_field(0.25, 2), 0.05)

        assert_allclose(3, value_to_db_power(4, 2), 0.05)
        assert_allclose(0, value_to_db_power(2, 2), 0.05)
        assert_allclose(-3, value_to_db_power(1, 2), 0.05)
        assert_allclose(-6, value_to_db_power(0.5, 2), 0.05)
        assert_allclose(-9, value_to_db_power(0.25, 2), 0.05)
        # Test string
        assert_allclose(6, value_to_db_field("4 V", "2 V"), 0.05)

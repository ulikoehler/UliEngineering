#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from nose.tools import assert_equal, assert_true
from UliEngineering.Electronics.Resistors import *
from UliEngineering.EngineerIO import formatValue

class TestResistors(object):
    def test_parallel_resistors(self):
        assert_approx_equal(parallel_resistors(1000.0, 1000.0), 500.0)
        assert_approx_equal(parallel_resistors(1000.0, 1000.0, 500.0), 250.0)
        assert_approx_equal(parallel_resistors("1kΩ", "1kΩ"), 500.0)

    def test_serial_resistors(self):
        assert_approx_equal(serial_resistors(1000.0, 1000.0), 2000.0)
        assert_approx_equal(serial_resistors(1000.0, 1000.0, 500.0), 2500.0)
        assert_approx_equal(serial_resistors("1kΩ", "1kΩ"), 2000.0)

    def test_standard_resistors(self):
        assert_true(len(list(standard_resistors())) > 500)

    def test_find_nearest_resistor(self):
        assert_equal(nearest_resistor(5000, sequence=e48), 5110.0)
        assert_equal(nearest_resistor(4998), 4990.0)

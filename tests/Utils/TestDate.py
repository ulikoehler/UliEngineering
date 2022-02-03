#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from subprocess import check_output
from UliEngineering.Utils.Date import *
import unittest
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
import numpy as np

class TestDate(unittest.TestCase):
    def test_number_of_days_in_month(self):
        self.assertEqual(number_of_days_in_month(2019, 1), 31)
        self.assertEqual(number_of_days_in_month(2019, 2), 28)

    def test_all_dates_in_year(self):
        # Wolfram Alpha: number of days in 2019
        self.assertEqual(len(list(all_dates_in_year(2019))), 365)
        # Wolfram Alpha: number of days in 2020
        self.assertEqual(len(list(all_dates_in_year(2020))), 366)

class TestGenerateDays(unittest.TestCase):
    def test_empty(self):
        assert_array_equal(generate_days(0), np.ndarray([], 'datetime64[us]'))
    def test_simple(self):
        assert_array_equal(generate_days(5, 2022, 1, 1),
            np.asarray([
                np.datetime64('2022-01-01T00:00:00.000000'),
                np.datetime64('2022-01-02T00:00:00.000000'),
                np.datetime64('2022-01-03T00:00:00.000000'),
                np.datetime64('2022-01-04T00:00:00.000000'),
                np.datetime64('2022-01-05T00:00:00.000000')
            ], dtype='datetime64[us]'))
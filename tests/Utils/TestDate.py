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

class TestExtractMonths(unittest.TestCase):
    def test_empty(self):
        assert_array_equal(extract_months(np.asarray([])), np.ndarray([], int))
    def test_simple(self):
        assert_array_equal(extract_months(generate_days(100, 2022, 1, 1)),
            np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
                3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        )

class TestExtractYears(unittest.TestCase):
    def test_empty(self):
        assert_array_equal(extract_years(np.asarray([])), np.ndarray([], int))
    def test_simple(self):
        assert_array_equal(extract_years(generate_days(5, 2021, 12, 30)),
            np.asarray([2021, 2021, 2022, 2022, 2022])
        )

class TestExtractDayOfMonth(unittest.TestCase):
    def test_empty(self):
        assert_array_equal(extract_day_of_month(np.asarray([])), np.ndarray([], int))
    def test_simple(self):
        assert_array_equal(extract_day_of_month(generate_days(5, 2021, 12, 30)),
            np.asarray([30, 31, 1, 2, 3])
        )

class TestExtractDayOfWeek(unittest.TestCase):
    def test_empty(self):
        assert_array_equal(extract_day_of_week(np.asarray([])), np.ndarray([], int))
    def test_simple(self):
        assert_array_equal(extract_day_of_week(generate_days(5, 2021, 12, 30)),
            np.asarray([4, 5, 6, 7, 1]) # 30th Dec 2012 was a thursday. Manually verified.
        )

class TestIsFirstDayOfMonth(unittest.TestCase):
    def test_empty(self):
        assert_array_equal(is_first_day_of_month(np.asarray([])), np.ndarray([], bool))
    def test_simple(self):
        assert_array_equal(is_first_day_of_month(generate_days(5, 2021, 12, 30)),
            np.asarray([False, False, True, False, False])
        )

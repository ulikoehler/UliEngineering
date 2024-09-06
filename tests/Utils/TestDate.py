#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from datetime import datetime
import re
import numpy as np
from numpy.testing import (assert_array_equal)
from UliEngineering.Utils.Date import *


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

class TestGenerateMonths(unittest.TestCase):
    def test_empty(self):
        assert_array_equal(generate_months(0), np.ndarray([], 'datetime64[us]'))
    def test_simple(self):
        assert_array_equal(generate_months(5, 2022, 1, 1),
            np.asarray([
                np.datetime64('2022-01-01T00:00:00.000000'),
                np.datetime64('2022-02-01T00:00:00.000000'),
                np.datetime64('2022-03-01T00:00:00.000000'),
                np.datetime64('2022-04-01T00:00:00.000000'),
                np.datetime64('2022-05-01T00:00:00.000000')
            ], dtype='datetime64[us]'))
    def test_year_wrap(self):
        assert_array_equal(generate_months(5, 2022, 10, 1),
            np.asarray([
                np.datetime64('2022-10-01T00:00:00.000000'),
                np.datetime64('2022-11-01T00:00:00.000000'),
                np.datetime64('2022-12-01T00:00:00.000000'),
                np.datetime64('2023-01-01T00:00:00.000000'),
                np.datetime64('2023-02-01T00:00:00.000000')
            ], dtype='datetime64[us]'))

class TestGenerateYears(unittest.TestCase):
    def test_empty(self):
        assert_array_equal(generate_years(0), np.ndarray([], 'datetime64[us]'))
    def test_simple(self):
        assert_array_equal(generate_years(5, 2022, 1, 1),
            np.asarray([
                np.datetime64('2022-01-01T00:00:00.000000'),
                np.datetime64('2023-01-01T00:00:00.000000'),
                np.datetime64('2024-01-01T00:00:00.000000'),
                np.datetime64('2025-01-01T00:00:00.000000'),
                np.datetime64('2026-01-01T00:00:00.000000')
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

class TestIsFirstDayOfWeek(unittest.TestCase):
    def test_empty(self):
        assert_array_equal(is_first_day_of_week(np.asarray([])), np.ndarray([], bool))
    def test_simple(self):
        assert_array_equal(is_first_day_of_week(generate_days(6, 2021, 12, 30)),
            np.asarray([False, False, False, False, True, False])
        )

class TestIsMonthChange(unittest.TestCase):
    def test_empty(self):
        assert_array_equal(is_month_change(np.asarray([])), np.ndarray([], bool))
    def test_simple(self):
        assert_array_equal(is_month_change(generate_days(6, 2021, 12, 30)),
            np.asarray([False, False, True, False, False, False])
        )
        # To differentiate from is_month_change
        assert_array_equal(is_month_change(generate_days(6, 2021, 1, 30)),
            np.asarray([False, False, True, False, False, False])
        )

class TestIsYearChange(unittest.TestCase):
    def test_empty(self):
        assert_array_equal(is_year_change(np.asarray([])), np.ndarray([], bool))
    def test_simple(self):
        assert_array_equal(is_year_change(generate_days(6, 2021, 12, 30)),
            np.asarray([False, False, True, False, False, False])
        )
        # To differentiate from is_month_change
        assert_array_equal(is_year_change(generate_days(6, 2021, 1, 30)),
            np.asarray([False, False, False, False, False, False])
        )

class TestYieldMinutesSeconds(unittest.TestCase):
    def test_yield_hours_on_day(self):
        results = list(yield_hours_on_day(2022, 6, 15))
        # Check how many results we generate
        self.assertEqual(len(results), 24)
        # Check if its all the correct day
        for result in results:
            self.assertIsInstance(result, datetime)
            self.assertEqual(result.year, 2022)
            self.assertEqual(result.month, 6)
            self.assertEqual(result.day, 15)
            # Minutes & seconds should be always set to 0
            self.assertEqual(result.minute, 0)
            self.assertEqual(result.second, 0)

    def test_yield_minutes_on_day(self):
        results = list(yield_minutes_on_day(2022, 6, 15))
        # Check how many results we generate
        self.assertEqual(len(results), 24*60)
        # Check if its all the correct day
        for result in results:
            self.assertIsInstance(result, datetime)
            self.assertEqual(result.year, 2022)
            self.assertEqual(result.month, 6)
            self.assertEqual(result.day, 15)
            # Seconds should be always set to 0
            self.assertEqual(result.second, 0)

    def test_yield_seconds_on_day(self):
        results = list(yield_seconds_on_day(2022, 6, 15))
        # Check how many results we generate
        self.assertEqual(len(results), 24*60*60)
        # Check if its all the correct day
        for result in results:
            self.assertIsInstance(result, datetime)
            self.assertEqual(result.year, 2022)
            self.assertEqual(result.month, 6)
            self.assertEqual(result.day, 15)

class TestGenerateDatetimeFilename(unittest.TestCase):
    def test_defaults(self):
        filename = generate_datetime_filename()
        self.assertTrue(re.match(r"data-\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{6}.csv", filename), msg=filename)

    def test_defaults_no_fractional(self):
        filename = generate_datetime_filename()
        self.assertTrue(re.match(r"data-\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{6}.csv", filename), msg=filename)
    
    def test_custom_label(self):
        filename = generate_datetime_filename(label="qdata")
        self.assertTrue(re.match(r"qdata-\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{6}.csv", filename), msg=filename)

    def test_no_label(self):
        filename = generate_datetime_filename(label=None)
        self.assertTrue(re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{6}.csv", filename), msg=filename)

    def test_custom_extension(self):
        filename = generate_datetime_filename(extension="txt")
        self.assertTrue(re.match(r"data-\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{6}.txt", filename), msg=filename)

    def test_custom_datetime(self):
        dt = datetime(year=2000, month=1, day=2, hour=3, minute=4, second=5, microsecond=789)
        filename = generate_datetime_filename(dt=dt)
        self.assertEqual("data-2000-01-02_03-04-05-000789.csv", filename)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_allclose, assert_approx_equal
from UliEngineering.EngineerIO.Timespan import EngineerTimespanIO, normalize_timespan, TimespanSeconds
import numpy as np
import unittest

class TestEngineerTimespanIO(unittest.TestCase):
    def setUp(self):
        self.io = EngineerTimespanIO.instance()

    def test_float_int_input(self):
        assert_approx_equal(self.io.normalize_timespan(1), 1)
        assert_approx_equal(self.io.normalize_timespan(1.25), 1.25)
    
    def test_numpy_input(self):
        assert_approx_equal(self.io.normalize_timespan(np.float64(1)), 1)
        assert_approx_equal(self.io.normalize_timespan(np.float64(1.25)), 1.25)

    def test_numpy_2d_input(self):
        assert_allclose(self.io.normalize_timespan(np.asarray([[1, 2], [3, 4]])), [[1,2], [3,4]])
        assert_allclose(self.io.normalize_timespan(np.asarray([[1.25, 2.25], [3.25, 4.25]])), [[1.25,2.25], [3.25,4.25]])

    def test_numpy_array_input(self):
        print(self.io.normalize_timespan(np.asarray([1, 2, 3])))
        assert_allclose(self.io.normalize_timespan(np.asarray([1, 2, 3])), [1,2,3])
        assert_allclose(self.io.normalize_timespan(np.asarray([1.25, 2.25, 3.25])), [1.25,2.25,3.25])

    def test_list_input_1d(self):
        assert_allclose(self.io.normalize_timespan([1, 2, 3]), [1,2,3])
        assert_allclose(self.io.normalize_timespan([1.25, 2.25, 3.25]), [1.25,2.25,3.25])

    def test_list_input_2d(self):
        assert_allclose(self.io.normalize_timespan([[1, 2], [3, 4]]), [[1,2], [3,4]])
        assert_allclose(self.io.normalize_timespan([[1.25, 2.25], [3.25, 4.25]]), [[1.25,2.25], [3.25,4.25]])

    def test_semicomplete_decimal_input(self):
        assert_approx_equal(self.io.normalize_timespan("1."), 1.)
        assert_approx_equal(self.io.normalize_timespan("1.s"), 1.)
        assert_approx_equal(self.io.normalize_timespan(".0 s"), 0.0)
        assert_approx_equal(self.io.normalize_timespan("1.h"), 3600)

    def test_normalize_timespan(self):
        # Test behavior with no unit
        assert_approx_equal(self.io.normalize_timespan("1.25"), 1.25)
        assert_approx_equal(self.io.normalize_timespan("1.25 s"), 1.25)
        assert_approx_equal(self.io.normalize_timespan("1.25 min"), 1.25 * 60)
        assert_approx_equal(self.io.normalize_timespan("1.25 h"), 1.25 * 3600)
        assert_approx_equal(self.io.normalize_timespan("1.25 d"), 1.25 * 86400)
        assert_approx_equal(self.io.normalize_timespan("1.25 w"), 1.25 * 86400 * 7)
        assert_approx_equal(self.io.normalize_timespan("1.25 months"), 1.25*31556952/12)
        assert_approx_equal(self.io.normalize_timespan("1.25 y"), 1.25*31556952)
        # Test negative values
        assert_approx_equal(self.io.normalize_timespan("-1.25"), -1.25)
        assert_approx_equal(self.io.normalize_timespan("-1.25 s"), -1.25)
        assert_approx_equal(self.io.normalize_timespan("-1.25 min"), -1.25 * 60)
        assert_approx_equal(self.io.normalize_timespan("-1.25 h"), -1.25 * 3600)

    def test_normalize_timespan_subseconds(self):
        # Test behavior with no unit
        assert_approx_equal(self.io.normalize_timespan("1.25 ms"), 1.25e-3)
        assert_approx_equal(self.io.normalize_timespan("1.25 µs"), 1.25e-6)
        assert_approx_equal(self.io.normalize_timespan("1.25 ns"), 1.25e-9)
        assert_approx_equal(self.io.normalize_timespan("1.25 ps"), 1.25e-12)
        assert_approx_equal(self.io.normalize_timespan("1.25 fs"), 1.25e-15)
        assert_approx_equal(self.io.normalize_timespan("1.25 as"), 1.25e-18)
        # Test negative values
        assert_approx_equal(self.io.normalize_timespan("-1.25 ms"), -1.25e-3)
        assert_approx_equal(self.io.normalize_timespan("-1.25 µs"), -1.25e-6)
        assert_approx_equal(self.io.normalize_timespan("-1.25 ns"), -1.25e-9)
        assert_approx_equal(self.io.normalize_timespan("-1.25 ps"), -1.25e-12)
        assert_approx_equal(self.io.normalize_timespan("-1.25 fs"), -1.25e-15)
        assert_approx_equal(self.io.normalize_timespan("-1.25 as"), -1.25e-18)

    def test_type_annotation_exists(self):
        """Test that the new type annotation is available"""
        self.assertIsNotNone(TimespanSeconds)

    def test_timespan_type_various_units(self):
        """Test TimespanSeconds type with various unit inputs"""
        test_cases = [
            ("1 s", 1.0),
            ("1 min", 60.0),
            ("1 h", 3600.0),
            ("1 d", 86400.0),
            ("1 ms", 1e-3),
            ("1 µs", 1e-6),
            ("1 ns", 1e-9),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_timespan(input_val)
                assert_approx_equal(result, expected)

    def test_normalize_timespan_function(self):
        """Test the normalize_timespan function directly"""
        assert_approx_equal(normalize_timespan("1 s"), 1.0)
        assert_approx_equal(normalize_timespan("1 min"), 60.0)
        assert_approx_equal(normalize_timespan("1 h"), 3600.0)
        assert_approx_equal(normalize_timespan("1 ms"), 1e-3)

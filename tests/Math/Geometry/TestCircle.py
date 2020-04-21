#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
from UliEngineering.Math.Geometry.Circle import *
from parameterized import parameterized
import concurrent.futures
import numpy as np
import datetime
import unittest


class TestCircle(unittest.TestCase):
    @parameterized.expand([
        (0.0, ),
        (1.0, ),
        (4.0, ),
        (3.125, ),
    ])
    def test_circle_area(self, radius):
        assert_approx_equal(circle_area(radius), np.pi*(radius**2))
        assert_approx_equal(circle_area(f"{radius}"), np.pi*(radius**2))

    @parameterized.expand([
        (0.0, ),
        (1.0, ),
        (4.0, ),
        (3.125, ),
    ])
    def test_circle_circumference(self, radius):
        assert_approx_equal(circle_circumference(radius), 2*np.pi*radius)
        assert_approx_equal(circle_circumference(f"{radius}"), 2*np.pi*radius)


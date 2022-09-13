#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from UliEngineering.Math.Geometry.Sphere import *
from parameterized import parameterized
import numpy as np
import unittest
from UliEngineering.EngineerIO import normalize_numeric, Unit


class TestSphere(unittest.TestCase):
    @parameterized.expand([
        (0.0, ),
        (1.0, ),
        (4.0, ),
        (3.125, ),
        (55.55, ),
    ])
    def test_sphere_volume(self, radius):
        volume = 4/3*np.pi*(radius**3)
        assert_approx_equal(sphere_volume_by_radius(radius), volume)
        assert_approx_equal(sphere_volume_by_diameter(radius*2), volume)
        assert_approx_equal(sphere_volume_by_radius(f"{radius}"), volume)
        assert_approx_equal(sphere_volume_by_diameter(radius*2), volume)

    @parameterized.expand([
        (0.0, ),
        (1.0, ),
        (4.0, ),
        (3.125, ),
        (55.55, ),
    ])
    def test_sphere_surface_area(self, radius):
        area = 4*np.pi*(radius**2)
        assert_approx_equal(sphere_surface_area_by_radius(radius), area)
        assert_approx_equal(sphere_surface_area_by_diameter(radius*2), area)
        assert_approx_equal(sphere_surface_area_by_radius(f"{radius}"), area)
        assert_approx_equal(sphere_surface_area_by_diameter(radius*2), area)



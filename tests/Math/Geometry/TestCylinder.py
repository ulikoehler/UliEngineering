#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from UliEngineering.Math.Geometry.Cylinder import *
from parameterized import parameterized
import concurrent.futures
import numpy as np
import datetime
import unittest
import math

class TestCylinder(unittest.TestCase):
    @parameterized.expand([
        # Wolfram Alpha as a reference
        (0.0, 3.0, 0.0),
        (3.0, 0.0, 0.0),
        (1.0, 1.0, math.pi), # "volume of cylinder radius 1 height 1"
        (4.0, 2.5, 125.664),
        (3.125, 44.15, 1354.51), # "volume of cylinder radius 3.125 height 44.15"
    ])
    def test_cylinder_volume(self, radius, height, expected):
        self.assertAlmostEqual(cylinder_volume(radius, height), expected, delta=.025)
        self.assertAlmostEqual(cylinder_volume(f"{radius}", f"{height}"), expected, delta=.025)

    @parameterized.expand([
        # Wolfram Alpha as a reference
        (0.0, 3.0, 0.0),
        (3.0, 0.0, 0.0),
        (1.0, 1.0, 2*math.pi), # "lateral surface area of cylinder radius 1 height 1"
        (4.0, 2.5, 62.8139),
        (3.125, 44.15, 866.883), # "lateral surface area of cylinder radius 3.125 height 44.15"
    ])
    def test_cylinder_side_surface_area(self, radius, height, expected):
        self.assertAlmostEqual(cylinder_side_surface_area(radius, height), expected, delta=.025)
        self.assertAlmostEqual(cylinder_side_surface_area(f"{radius}", f"{height}"), expected, delta=.025)

    @parameterized.expand([
        # Wolfram Alpha as a reference
        (0.0, 3.0, 0.0),
        (3.0, 0.0, 56.549),
        (1.0, 1.0, 4*math.pi), # "lateral surface area of cylinder radius 1 height 1"
        (4.0, 2.5, 163.363),
        (3.125, 44.15, 928.242), # "lateral surface area of cylinder radius 3.125 height 44.15"
    ])
    def test_cylinder_surface_area(self, radius, height, expected):
        self.assertAlmostEqual(cylinder_surface_area(radius, height), expected, delta=.025)
        self.assertAlmostEqual(cylinder_surface_area(f"{radius}", f"{height}"), expected, delta=.025)


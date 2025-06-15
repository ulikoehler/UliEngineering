#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from UliEngineering.Math.Geometry.Cylinder import *
from parameterized import parameterized
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

class TestHollowCylinder(unittest.TestCase):
    @parameterized.expand([
        # Wolfram Alpha as a reference
        # The following are not actually hollow (inner_radius=0.0)
        (0.0, 0.0, 3.0, 0.0),
        (3.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0, math.pi), # "volume of cylinder radius 1 height 1"
        (4.0, 0.0, 2.5, 125.664),
        (3.125, 0.0, 44.15, 1354.51), # "volume of cylinder radius 3.125 height 44.15"
        # The following are actually hollow
        (0.0, 0.0, 3.0, 0.0),
        (3.0, 1.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 0.),
        (1.0, 0.5, 1.0, math.pi - 0.785398),
        (4.0, 1.0, 2.5, 125.664-7.85398),
        (3.125, 2.25, 44.15, 1354.51-702.175),
    ])
    def test_hollow_cylinder_volume(self, outer_radius, inner_radius, height, expected):
        self.assertAlmostEqual(hollow_cylinder_volume(outer_radius, inner_radius, height), expected, delta=.025)
        self.assertAlmostEqual(hollow_cylinder_volume(f"{outer_radius}", f"{inner_radius}", f"{height}"), expected, delta=.025)
   

    @parameterized.expand([
        # Wolfram Alpha as a reference
        # The following are not actually hollow (inner_radius=0.0)
        (3.125, 0.0, 44.15, 1354.51),
        # The following are actually hollow
        (1.0, 0.5, 1.0, math.pi - 0.785398),
        (4.0, 1.0, 2.5, 125.664-7.85398),
        (3.125, 2.25, 44.15, 1354.51-702.175),
    ])
    def test_hollow_cylinder_inner_radius_by_volume(self, outer_radius, inner_radius, height, volume):
        # NOTE: This uses the hollow_cylinder_volume() test cases except the ones that yield 0 volume
        self.assertAlmostEqual(hollow_cylinder_inner_radius_by_volume(outer_radius, volume, height), inner_radius, delta=.025)
        self.assertAlmostEqual(hollow_cylinder_inner_radius_by_volume(f"{outer_radius}", f"{volume}", f"{height}"), inner_radius, delta=.025)
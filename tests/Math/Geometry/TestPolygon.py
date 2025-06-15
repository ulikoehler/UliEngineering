#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_allclose
from UliEngineering.Math.Geometry.Polygon import *
from parameterized import parameterized
import numpy as np
import unittest

class TestPolygon(unittest.TestCase):
    def test_polygon_lines(self):
        coords = np.asarray([[0, 1],
                             [1, 2],
                             [2, 3]])
        closed = np.asarray([[[2, 3], [0, 1]],
                             [[0, 1], [1, 2]],
                             [[1, 2], [2, 3]]])
        opened = np.asarray([[[0, 1], [1, 2]],
                             [[1, 2], [2, 3]]])
        assert_allclose(closed, polygon_lines(coords, closed=True))
        assert_allclose(opened, polygon_lines(coords, closed=False))

    def test_polygon_area_triangle(self):
        # Triangle test
        coords = np.asarray([[0, 0],
                             [1, 1],
                             [1, 0]])
        assert_allclose(0.5*1*1, polygon_area(coords))

    def test_polygon_area_square(self):
        coords = np.asarray([[0, 0],
                             [1, 0],
                             [1, 1],
                             [0, 1]])
        assert_allclose(1, polygon_area(coords))

    def test_polygon_area_rectangle(self):
        # Triangle test
        coords = np.asarray([[0, 0],
                             [2, 0],
                             [2, 1],
                             [0, 1]])
        assert_allclose(2, polygon_area(coords))

    @parameterized.expand([
        (np.zeros(5),),
        (np.zeros((5,5)),),
        (np.zeros((5,5)),)
    ])
    def test_polygon_area_rectangle_zeros(self, arr):
        with self.assertRaises(ValueError):
            polygon_area(arr)

    @parameterized.expand([
        (np.zeros(5),),
        (np.zeros((5,5)),),
        (np.zeros((5,5)),)
    ])
    def test_polygon_lines_rectangle(self, arr):
        with self.assertRaises(ValueError):
            polygon_lines(arr)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_allclose
from nose.tools import assert_equal, assert_tuple_equal, assert_is_none, assert_true, assert_false, raises, assert_in, assert_not_in
from UliEngineering.Math.Geometry import *
from nose_parameterized import parameterized
import functools
import numpy as np

class TestGeometry(object):
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
                    
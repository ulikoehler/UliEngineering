#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_allclose
from nose.tools import assert_equal, assert_tuple_equal, assert_is_none, assert_true, assert_false, raises, assert_in, assert_not_in
from UliEngineering.Math.Coordinates import *
from nose_parameterized import parameterized
import functools
import numpy as np

class TestBoundingBox(object):
    def test_bbox(self):
        coords = [(6.74219, -53.57835),
                  (6.74952, -53.57241),
                  (6.75652, -53.56289),
                  (6.74756, -53.56598),
                  (6.73462, -53.57518)]
        coords = np.asarray(coords)
        bbox = BoundingBox(coords)
        assert_allclose(bbox.minx, 6.73462)
        assert_allclose(bbox.maxx, 6.75652)
        assert_allclose(bbox.miny, -53.57835)
        assert_allclose(bbox.maxy, -53.56289)
        assert_allclose(bbox.width, 6.75652 - 6.73462)
        assert_allclose(bbox.height, -53.56289 - -53.57835)
        assert_in("BoundingBox(", bbox.__repr__())

    @parameterized([(np.zeros((0,2)),),
                    (np.zeros((2,3)),),
                    (np.zeros((2,2,2)),)
                    ])
    @raises(ValueError)
    def test_invalid_bbox_input(self, arr):
        BoundingBox(arr)

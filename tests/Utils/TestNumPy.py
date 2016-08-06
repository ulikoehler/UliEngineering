#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
from subprocess import check_output
from nose.tools import assert_equal, assert_true, raises, assert_greater
from UliEngineering.Utils.NumPy import *
import numpy as np

class TestDynamicArrayResize(object):
    def test_numpy_resize_insert(self):
        arr = np.zeros(3)
        # Insert within bounds
        arr = numpy_resize_insert(arr, 1, 0)
        arr = numpy_resize_insert(arr, 2, 1)
        arr = numpy_resize_insert(arr, 3, 2)
        assert_equal(arr.size, 3)
        assert_array_equal(arr, [1, 2, 3])
        # Resize
        arr = numpy_resize_insert(arr, 4, 3)
        assert_equal(arr.size, 1003) # 3 orig size + min growth = 1000
        # Remove extra size
        arr = np.resize(arr, 4)
        assert_array_equal(arr, [1, 2, 3, 4])


class TestInvertBijection(object):
    def testSimple(self):
        assert_allclose([0, 1, 2, 3], invert_bijection(np.arange(4)))
        assert_allclose([2, 0, 1, 3], invert_bijection([1, 2, 0, 3]))
        assert_allclose([1, 0, 2, 3], invert_bijection([1, 0, 2, 3]))

    def testEmpty(self):
        assert_equal((0,), invert_bijection([]).shape)

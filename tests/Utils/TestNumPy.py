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


class TestApplyPairwise1D(object):
    def testSimple(self):
        assert_allclose([[0,0,0], [1,2,3], [2,4,6]],
                        apply_pairwise_1d(np.arange(3), np.arange(1,4), lambda a, b: a * b))


class TestNgrams(object):

    def test_ngrams1(self):
        inp = np.arange(5) # 0..4
        closed = np.asarray([[0,1],[1,2],[2,3],[3,4],[4,0]])
        opened = np.asarray([[0,1],[1,2],[2,3],[3,4]])
        print(np.asarray(list(ngrams(inp, 2, closed=True))))
        print(closed)
        assert_allclose(closed, np.asarray(list(ngrams(inp, 2, closed=True))))
        assert_allclose(opened, np.asarray(list(ngrams(inp, 2, closed=False))))

    def test_ngrams2(self):
        inp = np.asarray([[0, 1], [1, 2],  [2, 3]])
        closed = np.asarray([[[0, 1], [1, 2]],
                             [[1, 2], [2, 3]],
                             [[2, 3], [0, 1]]])
        opened = np.asarray([[[0, 1], [1, 2]],
                             [[1, 2], [2, 3]]])
        print(np.asarray(list(ngrams(inp, 2, closed=True))))
        print(closed)
        assert_allclose(closed, np.asarray(list(ngrams(inp, 2, closed=True))))
        assert_allclose(opened, np.asarray(list(ngrams(inp, 2, closed=False))))

        
    def test_ngrams2(self):
        inp = np.asarray([[0, 1], [1, 2],  [2, 3]])
        closed = np.asarray([[[0, 1], [1, 2]],
                             [[1, 2], [2, 3]],
                             [[2, 3], [0, 1]]])
        opened = np.asarray([[[0, 1], [1, 2]],
                             [[1, 2], [2, 3]]])
        print(np.asarray(list(ngrams(inp, 2, closed=True))))
        print(closed)
        assert_allclose(closed, np.asarray(list(ngrams(inp, 2, closed=True))))
        assert_allclose(opened, np.asarray(list(ngrams(inp, 2, closed=False))))

class TestPivotSplit(object):
    def test_pivot_split(self):
        assert_equal([[0,1],[2,3,4,5]], list(split_by_pivot([0,1,2,3,4,5], [2])))
        assert_equal([[0,1],[2,3],[4,5]], list(split_by_pivot([0,1,2,3,4,5], [2,4])))
        assert_equal([[],[0,1],[2,3],[4,5]], list(split_by_pivot([0,1,2,3,4,5], [0,2,4])))

class TestDatetime64Now(object):
    def test_datetime64_now(self):
        assert_equal(type(datetime64_now()), np.datetime64)
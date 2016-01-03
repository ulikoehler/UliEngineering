#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from nose.tools import assert_equal, assert_true, raises
from UliEngineering.Utils.JSON import *

class TestNumpyEncoder(object):
    def testNDArrayEncoding(self):
        arr = np.asarray([1, 2, 3, 5])
        s = json.dumps(arr, cls=NumPyEncoder)
        assert_equal(s, "[1, 2, 3, 5]")

    def testNDMultidimensionalArrayEncoding(self):
        arr = np.asarray([[1, 1], [2, 2], [3, 3], [5, 5]])
        s = json.dumps(arr, cls=NumPyEncoder)
        assert_equal(s, "[[1, 1], [2, 2], [3, 3], [5, 5]]")

    def testNPScalarEncoding(self):
        arr = [np.int64(75),
               np.int64(31)]
        print(arr)
        assert_true(isinstance(arr[0], np.generic))
        s = json.dumps(arr, cls=NumPyEncoder)
        assert_equal(s, "[75, 31]")

    def testOtherEncoding(self):
        arr = {"a": "b"}
        assert_equal(json.dumps(arr, cls=NumPyEncoder), '{"a": "b"}')
        assert_equal(json.dumps("gaa", cls=NumPyEncoder), '"gaa"')
        assert_equal(json.dumps(None, cls=NumPyEncoder), 'null')

    @raises(TypeError)
    def test_invalid_encoding(self):
        json.dumps(set([1,2,3]), cls=NumPyEncoder)

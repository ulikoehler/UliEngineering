#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from nose.tools import assert_equal
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
    def testNDOtherEncoding(self):
        arr = np.asarray({"a": "b"})
        s = json.dumps(arr, cls=NumPyEncoder)
        assert_equal(s, '{"a": "b"}')


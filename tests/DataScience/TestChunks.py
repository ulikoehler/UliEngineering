#!/usr/bin/env python3
from nose.tools import assert_equal, assert_true, raises
import numpy as np
from numpy.testing import assert_array_equal, assert_array_less
from UliEngineering.DataScience.Chunks import fixedSizeChunkGenerator, evaluateGeneratorFunction

class TestChunkGeneration(object):
    def __init__(self):
        self.data1 = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.data2 = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    def test_fixedSizeChunkGenerator(self):
        # Odd-sized array
        vals = evaluateGeneratorFunction(fixedSizeChunkGenerator(self.data1, 3, 3), as_list=True)
        expected = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert_array_equal(vals, expected)
        # Even-sized array
        vals = evaluateGeneratorFunction(fixedSizeChunkGenerator(self.data2, 3, 3), as_list=True)
        expected = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        assert_array_equal(vals, expected)
        # Array which is too long
        vals = evaluateGeneratorFunction(fixedSizeChunkGenerator(self.data2, 25, 3), as_list=True)
        expected = np.asarray([])
        assert_array_equal(vals, expected)
    def test_fixedSizeChunkGenerator_perform_copy(self):
        d1 = self.data1.copy()
        d2 = self.data1.copy()
        # perform_copy=True should be the default.
        vals = evaluateGeneratorFunction(fixedSizeChunkGenerator(d1, 3, 10), as_list=True)
        vals[0][0] = -1000  # Change value ...
        assert_array_equal(d1, d2)  # ... should not have any effect
        # Do the same without copy
        vals = evaluateGeneratorFunction(fixedSizeChunkGenerator(d1, 3, 10, perform_copy=False), as_list=True)
        vals[0][0] = -1000  # Change value ...
        d2[0] = -1000  # Should make d2 equal to d1
        assert_array_equal(d1, d2)  # ... should have an effect


    @raises
    def test_fixedSizeChunkGenerator_invalid1(self):
        fixedSizeChunkGenerator(None, 3, 3)
    @raises
    def test_fixedSizeChunkGenerator_invalid2(self):
        fixedSizeChunkGenerator(self.data1, 3, 0)
    @raises
    def test_fixedSizeChunkGenerator_invalid3(self):
        fixedSizeChunkGenerator(self.data1, 0, 3)

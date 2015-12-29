#!/usr/bin/env python3
from nose.tools import assert_equal, assert_true, raises
import numpy as np
from numpy.testing import assert_array_equal
from UliEngineering.DataScience.Chunks import fixedSizeChunkGenerator, evaluateGeneratorFunction

class TestChunkGeneration(object):
    def __init__(self):
        self.data1 = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.data2 = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    def test_fixedSizeChunkGenerator(self):
        # Odd-sized array
        vals = evaluateGeneratorFunction(fixedSizeChunkGenerator(self.data1, 3, 3), as_list=True)
        expected = np.asarray([[1, 2, 3],  [4, 5, 6],  [7, 8, 9]])
        assert_array_equal(vals, expected)
        # Even-sized array
        vals = evaluateGeneratorFunction(fixedSizeChunkGenerator(self.data2, 3, 3), as_list=True)
        expected = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        assert_array_equal(vals, expected)
    @raises
    def test_fixedSizeChunkGenerator_invalid1(self):
        fixedSizeChunkGenerator(None, 3, 3)
    @raises
    def test_fixedSizeChunkGenerator_invalid2(self):
        fixedSizeChunkGenerator(self.data1, 3, 0)
    @raises
    def test_fixedSizeChunkGenerator_invalid3(self):
        fixedSizeChunkGenerator(self.data1, 0, 3)

#!/usr/bin/env python3
import numpy as np
from nose.tools import assert_equal, assert_true, raises
from numpy.testing import assert_array_equal, assert_array_less
from UliEngineering.SignalProcessing.Chunks import *

class TestChunkGeneration(object):
    def __init__(self):
        self.data1 = np.arange(1, 11)
        self.data2 = np.arange(1, 13)

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

    def test_randomSampleChunkGenerator(self):
        vals = np.vstack((evaluateGeneratorFunction(
            randomSampleChunkGenerator(self.data1, 3, 2), as_list=True)))
        assert_equal(vals.shape, (2, 3))
        assert_true((vals < 10).all())
        assert_true((vals >= 0).all())


    @raises
    def test_fixedSizeChunkGenerator_invalid1(self):
        fixedSizeChunkGenerator(None, 3, 3)
    @raises
    def test_fixedSizeChunkGenerator_invalid2(self):
        fixedSizeChunkGenerator(self.data1, 3, 0)
    @raises
    def test_fixedSizeChunkGenerator_invalid3(self):
        fixedSizeChunkGenerator(self.data1, 0, 3)

class TestReshapedChunks(object):
    def test_reshaped_chunks(self):
        arr = np.arange(10)
        # 1
        chunks = reshapedChunks(arr, 2)
        assert_equal(chunks.shape, (5, 2))
        assert_array_equal(chunks, np.asarray([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]))
        # 2
        chunks = reshapedChunks(arr, 4)
        assert_equal(chunks.shape, (2, 4))
        assert_array_equal(chunks, np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]]))

    def test_empty(self):
        empty = np.asarray([])
        assert_array_equal(reshapedChunks(empty, 4), empty)


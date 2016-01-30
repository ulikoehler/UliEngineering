#!/usr/bin/env python3
import numpy as np
from toolz import functoolz
from nose.tools import assert_equal, assert_true, raises
from numpy.testing import assert_array_equal, assert_allclose
from UliEngineering.SignalProcessing.Chunks import *

class TestChunkGeneration(object):
    def __init__(self):
        self.data1 = np.arange(1, 11)
        self.data2 = np.arange(1, 13)

    def test_overlapping_chunks(self):
        # Odd-sized array
        vals = overlapping_chunks(self.data1, 3, 3)
        assert_array_equal(vals.as_array(), [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert_equal(len(vals), 3)
        assert_array_equal(vals.as_array(), [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # Even-sized array
        vals = overlapping_chunks(self.data2, 3, 3)
        assert_equal(len(vals), 4)
        assert_array_equal(vals.as_array(), [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        # Array which is too long
        vals = overlapping_chunks(self.data2, 25, 3)
        assert_array_equal(vals.as_array(), [])
        # Test apply function
        cg = overlapping_chunks(self.data1, 3, 3)
        cg.apply(functoolz.identity)
        cg.apply(functoolz.identity)
        cg.apply(np.square)
        expected = np.square(np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        assert_array_equal(cg.as_array(), expected)


    def test_overlapping_chunks_copy(self):
        d1 = self.data1.copy()
        d2 = self.data1.copy()
        # copy=True should be the default.
        vals = overlapping_chunks(d1, 3, 10, copy=True)(0)
        vals[0] = -1000  # Change value ...
        assert_array_equal(d1, d2)  # ... should not have any effect
        # Do the same without copy
        vals = overlapping_chunks(d1, 3, 10)(0)
        vals[0] = -1000  # Change value ...
        d2[0] = -1000  # Should make d2 equal to d1
        assert_array_equal(d1, d2)  # ... should have had an effect

    def test_randomSampleChunkGenerator(self):
        vals = random_sample_chunks(self.data1, 3, 2).as_array()
        assert_equal(vals.shape, (2, 3))
        assert_true((vals <= 10).all())
        assert_true((vals >= 0).all())

    def test_randomSampleChunkGeneratorNonoverlapping(self):
        vals = random_sample_chunks_nonverlapping(self.data1, 3, 2).as_array()
        assert_equal(vals.shape, (2, 3))
        assert_true((vals <= 10).all())
        assert_true((vals >= 0).all())

    @raises
    def test_fixedSizeChunkGenerator_invalid1(self):
        overlapping_chunks(None, 3, 3)
    @raises
    def test_fixedSizeChunkGenerator_invalid2(self):
        overlapping_chunks(self.data1, 3, 0)
    @raises
    def test_fixedSizeChunkGenerator_invalid3(self):
        overlapping_chunks(self.data1, 0, 3)


class TestReshapedChunks(object):
    def test_reshaped_chunks(self):
        arr = np.arange(10)
        # 1
        chunks = reshaped_chunks(arr, 2)
        assert_equal(chunks.shape, (5, 2))
        assert_array_equal(chunks, np.asarray([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]))
        # 2
        chunks = reshaped_chunks(arr, 4)
        assert_equal(chunks.shape, (2, 4))
        assert_array_equal(chunks, np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]]))

    def test_empty(self):
        empty = np.asarray([])
        assert_array_equal(reshaped_chunks(empty, 4), empty)


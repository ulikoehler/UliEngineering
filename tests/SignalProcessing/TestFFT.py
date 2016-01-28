#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from nose.tools import assert_equal, assert_true, raises, assert_less
from UliEngineering.SignalProcessing.FFT import *
from UliEngineering.SignalProcessing.Chunks import *
from nose_parameterized import parameterized
import concurrent.futures
import numpy as np

executor = concurrent.futures.ThreadPoolExecutor(4)

class TestFFT(object):
    def testBasicFFT(self):
        rand = np.random.random(1000) * 5.0 + 1.0 # +1: Artifical DC artifacts
        x, y = computeFFT(rand, 10.0)
        assert_equal(x.shape, (rand.shape[0] / 2, ))
        assert_equal(y.shape, (rand.shape[0] / 2, ))
        assert_equal(x.shape, y.shape)
        # Test if artifacts can be cut
        origLength = x.shape[0]
        x2, y2 = cutFFTDCArtifacts(x, y)
        assert_less(x2.shape[0], origLength)
        # Check if we can also pass a tuple
        x3, y3 = cutFFTDCArtifacts((x, y))
        assert_allclose(x2, x3)
        assert_allclose(y2, y3)

    def testCutFFTDCArtifacts(self):
        x = np.linspace(100, 199, 100) # Must not be equal to array index (so we check the fn doesnt just return indices)
        # Generate down/up slope; minimum at 10
        y = np.linspace(0, 100, 100)
        y[:10] = np.linspace(100, 0, 10)
        # Insert artificial peak
        assert_equal(cutFFTDCArtifacts(x, y, return_idx=True), 10)

    def testCutDCArtifactsNoMinimum(self):
        # No minimum --> should return original array
        x = np.linspace(100, 1, 100)
        y = np.linspace(500, 5, 100)
        x2, y2 = cutFFTDCArtifacts(x, y)
        assert_equal(x2.shape, x.shape)
        assert_equal(y2.shape, y.shape)
        assert_allclose(x2, x)
        assert_allclose(y2, y)
        # Check returned index
        assert_equal(cutFFTDCArtifacts(x, y, return_idx=True), 0)

    def testCutFFTDCArtifactsMulti(self):
        x = np.linspace(100, 199, 100) # Must not be equal to array index (so we check the fn doesnt just return indices)
        # Generate down/up slope; minimum at 10
        y1 = np.linspace(0, 100, 100)
        y1[:10] = np.linspace(100, 0, 10)
        y2 = np.linspace(0, 100, 100)
        y2[:15] = np.linspace(100, 0, 15)
        # Insert artificial peak
        assert_equal(cutFFTDCArtifactsMulti(x, [y1], return_idx=True), 10)
        assert_equal(cutFFTDCArtifactsMulti(x, [y1, y2], return_idx=True), 15)
        # Check actual return value
        xr, yrs = cutFFTDCArtifactsMulti(x, [y1, y2])
        yr1 = yrs[0]
        yr2 = yrs[1]
        assert_allclose(xr, x[15:])
        assert_allclose(yr1, y1[15:])
        assert_allclose(yr2, y2[15:])

    def testDominantFrequency(self):
        x = np.linspace(100, 199, 100) # Must not be equal to array index (so we check the fn doesnt just return indices)
        y = np.random.random(100)
        # Insert artificial peak
        y[32] = 8.0
        assert_equal(dominantFrequency(x, y), 132)
        # Check if we can also pass a tuple
        assert_equal(dominantFrequency((x, y)), 132)
        # Check with frequency range
        assert_equal(dominantFrequency(x, y, low=100.0, high=140.0), 132)

    @parameterized.expand([
        ("With DC", False),
        ("Without DC", True),
    ])
    def testParallelFFTReduce(self, name, removeDC):
        d = np.random.random(1000)
        y, nchunks = fixedSizeChunkGenerator(d, 100, 5)
        # Just test if it actually runs
        x, y = parallelFFTReduce(executor, y, nchunks, 10.0, 100, removeDC=removeDC)
        if removeDC:
            assert_equal(x.shape[0], y.shape[0])
        else:  # With DC
            assert_equal(x.shape[0], 50)
            assert_equal(y.shape[0], 50)
        assert_equal(x.shape, y.shape)

    def testSimpleParallelFFTReduce(self):
        d = np.random.random(1000)
        # Just test if it actually runs
        x, y = simpleParallelFFTReduce(d, 100.0, 100)
        assert_equal(x.shape[0], 50)
        assert_equal(y.shape[0], 50)

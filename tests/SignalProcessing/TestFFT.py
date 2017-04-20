#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from nose.tools import assert_equal, assert_true, raises, assert_less, assert_almost_equal
from UliEngineering.SignalProcessing.FFT import *
from UliEngineering.SignalProcessing.Chunks import *
from nose_parameterized import parameterized
import concurrent.futures
import numpy as np

class TestFFT(object):
    def testBasicFFT(self):
        rand = np.random.random(1000) * 5.0 + 1.0 # +1: Artifical DC artifacts
        x, y = compute_fft(rand, 10.0)
        assert_equal(x.shape, (rand.shape[0] / 2, ))
        assert_equal(y.shape, (rand.shape[0] / 2, ))
        assert_equal(x.shape, y.shape)
        # Test if artifacts can be cut
        origLength = x.shape[0]
        x2, y2 = fft_cut_dc_artifacts(x, y)
        assert_less(x2.shape[0], origLength)
        # Check if we can also pass a tuple
        x3, y3 = fft_cut_dc_artifacts((x, y))
        assert_allclose(x2, x3)
        assert_allclose(y2, y3)

    def testCutFFTDCArtifacts(self):
        x = np.linspace(100, 199, 100) # Must not be equal to array index (so we check the fn doesnt just return indices)
        # Generate down/up slope; minimum at 10
        y = np.linspace(0, 100, 100)
        y[:10] = np.linspace(100, 0, 10)
        # Insert artificial peak
        assert_equal(fft_cut_dc_artifacts(x, y, return_idx=True), 10)

    def testCutDCArtifactsNoMinimum(self):
        # No minimum --> should return original array
        x = np.linspace(100, 1, 100)
        y = np.linspace(500, 5, 100)
        x2, y2 = fft_cut_dc_artifacts(x, y)
        assert_equal(x2.shape, x.shape)
        assert_equal(y2.shape, y.shape)
        assert_allclose(x2, x)
        assert_allclose(y2, y)
        # Check returned index
        assert_equal(fft_cut_dc_artifacts(x, y, return_idx=True), 0)

    def testCutFFTDCArtifactsMulti(self):
        x = np.linspace(100, 199, 100) # Must not be equal to array index (so we check the fn doesnt just return indices)
        # Generate down/up slope; minimum at 10
        y1 = np.linspace(0, 100, 100)
        y1[:10] = np.linspace(100, 0, 10)
        y2 = np.linspace(0, 100, 100)
        y2[:15] = np.linspace(100, 0, 15)
        # Insert artificial peak
        assert_equal(fft_cut_dc_artifacts_multi(x, [y1], return_idx=True), 10)
        assert_equal(fft_cut_dc_artifacts_multi(x, [y1, y2], return_idx=True), 15)
        # Check actual return value
        xr, yrs = fft_cut_dc_artifacts_multi(x, [y1, y2])
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
        assert_equal(dominant_frequency(x, y), 132)
        # Check if we can also pass a tuple
        assert_equal(dominant_frequency((x, y)), 132)
        # Check with frequency range
        assert_equal(dominant_frequency(x, y, low=100.0, high=140.0), 132)

    @parameterized.expand([
        (1., 1.0),
        (10.234, 1.0),
        (0.01, 1.0),
        (10000., 1.0),
        (1., 30.0),
        (10.234, 30.0),
        (0.01, 30.0),
        (10000., 30.0),
    ])
    def testFFTAmplitudeIntegral(self, amplitude, length):
        """FFT amplitude integral should be equal to ptp value of a sine wave"""
        sine = generate_sinewave(10.0, 100.0, amplitude, length)
        fftx, ffty = compute_fft(sine, 100.0)
        # Number of decimals must depend on value, so we need to divide here
        assert_almost_equal(np.sum(ffty) / amplitude, 1.0, 2)

    @raises(ValueError)
    def test_fft_empty_chunks(self):
        cg = ChunkGenerator(lambda _: [], 0)
        parallel_fft_reduce(cg, None, None)

    @parameterized.expand([
        ("With DC", False),
        ("Without DC", True),
    ])
    def testParallelFFTReduce(self, name, removeDC):
        d = np.random.random(1000)
        chunkgen = overlapping_chunks(d, 100, 5)
        # Just test if it actually runs
        x, y = parallel_fft_reduce(chunkgen, 10.0, 100, removeDC=removeDC)
        if removeDC:
            assert_equal(x.shape[0], y.shape[0])
        else:  # With DC
            assert_equal(x.shape[0], 50)
            assert_equal(y.shape[0], 50)
        assert_equal(x.shape, y.shape)

    def testSimpleParallelFFTReduce(self):
        d = np.random.random(1000)
        # Just test if it actually runs
        x, y = simple_parallel_fft_reduce(d, 100.0, 100)
        assert_equal(x.shape[0], 50)
        assert_equal(y.shape[0], 50)

    def testGenerateSinewave(self):
        sw = generate_sinewave(25., 400.0, 1.0, 10.)
        fftx, ffty = compute_fft(sw, 400.)
        df = dominant_frequency(fftx, ffty)
        assert_true(abs(df - 25.0) < 0.1)

    def testAmplitudeIntegral(self):
        fx = np.arange(4)
        fy = np.asarray([2, 3, 4, 5])
        assert_almost_equal(amplitude_integral(fx, fy), sum(fy) / 3.)
        assert_almost_equal(amplitude_integral(fx, fy, low=1., high=2.01), 3 + 4)

    @raises(ValueError)
    def test_too_small_fft(self):
        d = np.random.random(10)
        # Just test if it actually runs
        x, y = simple_parallel_fft_reduce(d, 1000.0, 100)

    def test_find_closest_frequency(self):
        fftx = np.asarray([1,2,3,4,5])
        ffty = fftx * 2
        assert_equal((1, 2), find_closest_frequency(fftx, ffty, 0.))

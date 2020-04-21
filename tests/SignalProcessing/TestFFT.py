#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from nose.tools import assert_equal, assert_true, raises, assert_less, assert_almost_equal
from UliEngineering.SignalProcessing.FFT import *
from UliEngineering.SignalProcessing.Simulation import *
from UliEngineering.SignalProcessing.Chunks import *
from parameterized import parameterized
import concurrent.futures
import numpy as np
import numpy.random
import unittest

class TestFFT(unittest.TestCase):
    def testBasicFFT(self):
        rand = np.random.random_sample(1000) * 5.0 + 1.0 # +1: Artifical DC artifacts
        fft = compute_fft(rand, 10.0)
        assert_equal(fft.frequencies.shape[0], rand.shape[0] / 2)
        assert_equal(fft.amplitudes.shape[0], rand.shape[0] / 2)
        assert_equal(fft.frequencies.shape, fft.amplitudes.shape)
        # Test if artifacts can be cut
        origLength = fft.frequencies.shape[0]
        fft2 = fft.cut_dc_artifacts()
        assert_less(fft2.frequencies.shape[0], origLength)

    def testCutFFTDCArtifacts(self):
        x = np.linspace(100, 199, 100) # Must not be equal to array index (so we check the fn doesnt just return indices)
        # Generate down/up slope; minimum at 10
        y = np.linspace(0, 100, 100)
        y[:10] = np.linspace(100, 0, 10)
        fft = FFT(x, y)
        # Insert artificial peak
        assert_equal(fft.cut_dc_artifacts(return_idx=True), 10)

    def testCutDCArtifactsNoMinimum(self):
        # No minimum --> should return original array
        x = np.linspace(100, 1, 100)
        y = np.linspace(500, 5, 100)
        fft = FFT(x, y).cut_dc_artifacts()
        assert_equal(fft.frequencies.shape, x.shape)
        assert_equal(fft.amplitudes.shape, y.shape)
        assert_allclose(fft.frequencies, x)
        assert_allclose(fft.amplitudes, y)
        # Check returned index
        assert_equal(FFT(x, y).cut_dc_artifacts(return_idx=True), 0)

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
        y = np.random.random_sample(100)
        # Insert artificial peak
        y[32] = 8.0
        fft = FFT(x, y)
        assert_equal(fft.dominant_frequency(), 132)
        # Check with frequency range
        assert_equal(fft.dominant_frequency(low=100.0, high=140.0), 132)

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
        sine = sine_wave(10.0, 100.0, amplitude, length)
        fft = compute_fft(sine, 100.0)
        # Number of decimals must depend on value, so we need to divide here
        assert_almost_equal(np.sum(fft.amplitudes) / amplitude, 1.0, 2)

    @raises(ValueError)
    def test_fft_empty_chunks(self):
        cg = ChunkGenerator(lambda _: [], 0)
        parallel_fft_reduce(cg, None, None)

    @parameterized.expand([
        ("With DC", False),
        ("Without DC", True),
    ])
    def testParallelFFTReduce(self, name, removeDC):
        d = np.random.random_sample(1000)
        chunkgen = overlapping_chunks(d, 100, 5)
        # Just test if it actually runs
        fft = parallel_fft_reduce(chunkgen, 10.0, 100, removeDC=removeDC)
        if removeDC:
            assert_equal(fft.frequencies.shape[0], fft.amplitudes.shape[0])
        else:  # With DC
            assert_equal(fft.frequencies.shape[0], 50)
            assert_equal(fft.amplitudes.shape[0], 50)
        assert_equal(fft.frequencies.shape, fft.amplitudes.shape)

    def testSimpleParallelFFTReduce(self):
        d = np.random.random_sample(1000)
        # Just test if it actually runs
        fft = simple_parallel_fft_reduce(d, 100.0, 100)
        assert_equal(fft.frequencies.shape[0], 50)
        assert_equal(fft.amplitudes.shape[0], 50)

    def testAmplitudeIntegral(self):
        fft = FFT(np.arange(4), np.asarray([2, 3, 4, 5]), None)
        assert_almost_equal(fft.amplitude_integral(), sum(fft.amplitudes) / 3.)
        assert_almost_equal(fft.amplitude_integral(low=1., high=2.01), 3 + 4)

    @raises(ValueError)
    def test_too_small_fft(self):
        d = np.random.random_sample(10)
        # Just test if it actually runs
        x, y = simple_parallel_fft_reduce(d, 1000.0, 100)


class TestClosestFrequency(unittest.TestCase):
    def __init__(self):
        pass

    def test_find_closest_frequency(self):
        fftx = np.asarray([1,2,3,4,5])
        fft = FFT(fftx, fftx * 2)
        assert_equal(1, fft.closest_frequency(0.))

    def test_find_closest_value_noangle(self):
        fftx = np.asarray([1,2,3,4,5])
        fft = FFT(fftx, fftx * 2)
        assert_equal((1, 2, None), fft.closest_value(0.))

    def test_find_closest_value_angle(self):
        fftx = np.asarray([1,2,3,4,5])
        fft = FFT(fftx, fftx * 2, fftx * 3)
        assert_equal((1, 2, 3), fft.closest_value(0.))

class TestFFTSelectFrequencyRange(unittest.TestCase):
    def testGeneric(self):
        arr = np.arange(0.0, 10.0)
        result = FFT(arr, arr + 1.0, None)[1.0:5.5]
        desired = np.asarray([2.0, 3.0, 4.0, 5.0, 6.0])
        assert_allclose(result.frequencies, desired - 1.0)
        assert_allclose(result.amplitudes, desired)

    def testNoneLimit(self):
        arr = np.arange(0.0, 10.0)
        # Low = None
        result = FFT(arr, arr + 1.0, None)[:5.0]
        desired = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        assert_allclose(result.frequencies, desired - 1.0)
        assert_allclose(result.amplitudes, desired)
        # High = None
        result = FFT(arr, arr + 1.0, None)[5.0:]
        desired = np.asarray([6.0, 7.0, 8.0, 9.0, 10.0])
        assert_allclose(result.frequencies, desired - 1.0)
        assert_allclose(result.amplitudes, desired)

    def testTupleUnpacking(self):
        "Test tuple unpacking for FFT inlining"
        arr = np.arange(0.0, 10.0)
        result = FFT(arr, arr + 1.0, None)[1.0:5.5]
        desired = np.asarray([2.0, 3.0, 4.0, 5.0, 6.0])
        assert_allclose(result.frequencies, desired - 1.0)
        assert_allclose(result.amplitudes, desired)

#!/usr/bin/env python3
import numpy as np
from nose.tools import assert_equal, assert_true, raises, assert_raises
from numpy.testing import assert_allclose, assert_approx_equal
from UliEngineering.SignalProcessing.Resampling import *
import unittest


class TestBSplineResampling(unittest.TestCase):
    def __init__(self):
        self.x = np.arange(100)
        self.y = np.square(self.x)

    def testDiscard(self):
        x = np.arange(10)
        assert_allclose(resample_discard(x, 2), [0, 2, 4, 6, 8])
        assert_allclose(resample_discard(x, 3), [0, 3, 6, 9])

class TestSignalSamplerate(unittest.TestCase):
    def __init__(self):
        # 100% equal sample rate
        self.tequal = np.asarray([
            '2019-02-01T12:00:00.100000000',
            '2019-02-01T12:00:00.200000000',
            '2019-02-01T12:00:00.300000000',
            '2019-02-01T12:00:00.400000000',
            '2019-02-01T12:00:00.500000000',
            '2019-02-01T12:00:00.600000000',
            '2019-02-01T12:00:00.700000000',
            '2019-02-01T12:00:00.800000000',
            '2019-02-01T12:00:00.900000000',
        ], dtype='datetime64[ns]')
        # Almost 100% equal sample rate
        self.talmostequal = np.asarray([
            '2019-02-01T12:00:00.100000000',
            '2019-02-01T12:00:00.200000000',
            '2019-02-01T12:00:00.300000000',
            '2019-02-01T12:00:00.400000000',
            '2019-02-01T12:00:00.500000000',
            '2019-02-01T12:00:00.600000000',
            '2019-02-01T12:00:00.700000000',
            '2019-02-01T12:00:00.800000000',
            '2019-02-01T12:00:00.900000100',
        ], dtype='datetime64[ns]')
        # Jittery sample rate
        self.tunequal = np.asarray([
            '2019-02-01T12:00:00.103000000',
            '2019-02-01T12:00:00.205000000',
            '2019-02-01T12:00:00.301000000',
            '2019-02-01T12:00:00.403000000',
            '2019-02-01T12:00:00.502000000',
            '2019-02-01T12:00:00.606000000',
            '2019-02-01T12:00:00.701000000',
            '2019-02-01T12:00:00.802000000',
            '2019-02-01T12:00:00.900000000',
        ], dtype='datetime64[ns]')

    def testSignalSamplerate(self):
        assert_approx_equal(signal_samplerate(self.tunequal, ignore_percentile=3), 10.03344)
        assert_approx_equal(signal_samplerate(self.talmostequal, ignore_percentile=3), 10.0)
        assert_approx_equal(signal_samplerate(self.tequal, ignore_percentile=3), 10.0)

class TestParallelResampling(unittest.TestCase):
    def __init__(self):
        self.x = np.arange(100)
        self.y = np.square(self.x)

    def testSimpleCall(self):
        # Check if a simple call does not raise any exceptions
        print("foo")
        parallel_resample(self.x, self.y, 10.0, time_factor=1.0)

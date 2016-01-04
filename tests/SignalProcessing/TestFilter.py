#!/usr/bin/env python3
import numpy as np
from nose.tools import assert_equal, assert_true, raises
from numpy.testing import assert_array_equal, assert_array_less
from UliEngineering.SignalProcessing.Filter import *
from nose_parameterized import parameterized

class TestFilter(object):

    def __init__(self):
        self.d = np.random.random(1000)

    @parameterized([
        ("lowpass", 1.0),
        ("highpass", 1.0),
        ("bandpass", [1.0, 2.0]),
        ("bandstop", [1.0, 2.0]),
    ])
    def testBasicFilter(self, btype, frequencies):
        filt = SignalFilter(100.0, frequencies)
        filt.iir(order=3, btype=btype)
        d2 = filt(self.d)
        assert_equal(self.d.shape, d2.shape)

    @parameterized([
        ("butter",),
        ("cheby1",),
        ("cheby2",),
        ("ellip",),
        ("bessel",),
    ])
    def testFilterTypes(self, ftype):
        filt = SignalFilter(100.0, [1.0, 2.0])
        filt.iir(order=3, ftype=ftype)
        d2 = filt(self.d)
        assert_equal(self.d.shape, d2.shape)

    def testFrequencyResponse(self):
        filt = SignalFilter(100.0, [1.0, 2.0])
        filt.iir(order=3)
        filt.frequency_response()


    @raises(NotComputedException)
    def testUninitializedFilter1(self):
        filt = SignalFilter(100.0, 1.0)
        filt.is_stable()

    @raises(NotComputedException)
    def testUninitializedFilter2(self):
        filt = SignalFilter(100.0, 1.0)
        filt(self.d)

    @raises(FilterUnstableError)
    def testUnstableFilter(self):
        filt = SignalFilter(100.0, [1.0, 2.0])
        filt.iir(order=100, rp=1e-12)

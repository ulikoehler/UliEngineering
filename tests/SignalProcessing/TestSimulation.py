#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from nose.tools import assert_equal, assert_true, raises, assert_less, assert_almost_equal
from UliEngineering.SignalProcessing.Simulation import *
from UliEngineering.SignalProcessing.FFT import *
from UliEngineering.SignalProcessing.Chunks import *
from nose_parameterized import parameterized
import concurrent.futures
import numpy as np

class TestGenerateSinewave(object):
    def testGenerateSinewave(self):
        sw = generate_sinewave(25., 400.0, 1.0, 10.)
        fftx, ffty = compute_fft(sw, 400.)
        df = dominant_frequency(fftx, ffty)
        assert_true(abs(df - 25.0) < 0.1)

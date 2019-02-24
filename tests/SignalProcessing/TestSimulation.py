#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from nose.tools import assert_equal, assert_true, raises, assert_less, assert_almost_equal
from UliEngineering.SignalProcessing.Simulation import *
from UliEngineering.SignalProcessing.FFT import *
from UliEngineering.SignalProcessing.Chunks import *
from parameterized import parameterized
import concurrent.futures
import numpy as np

class TestGenerateSinewave(object):
    @parameterized.expand([
        (1.,),
        (2.,),
        (3.,),
        (4.,),
        (5.,),
        (6.,),
        (7.,),
        (8.,),
        (9.,),
        (10.,),
        (10.234,),
        (11.,),
    ])
    def testPhaseShift(self, frequency):
        """Test if 0/360/720° phase shift matches, and 180/540° matches as wel"""
        sw0 = sine_wave(frequency, 1000.0, amplitude=1., length=5.0, phaseshift=0.0)
        sw180 = sine_wave(frequency, 1000.0, amplitude=1., length=5.0, phaseshift=180.0)
        sw360 = sine_wave(frequency, 1000.0, amplitude=1., length=5.0, phaseshift=360.0)
        sw540 = sine_wave(frequency, 1000.0, amplitude=1., length=5.0, phaseshift=540.0)
        sw720 = sine_wave(frequency, 1000.0, amplitude=1., length=5.0, phaseshift=720.0)
        # Test in-phase
        assert_allclose(sw0, sw360, atol=1e-7)
        assert_allclose(sw0, sw720, atol=1e-7)
        assert_allclose(sw180, sw540, atol=1e-7)
        # Test out-of-phase
        assert_allclose(sw0, -sw180, atol=1e-7)

    def testOffset(self):
        sw1 = sine_wave(25., 400.0, 1.0, 10.)
        sw2 = sine_wave(25., 400.0, 1.0, 10., offset=2.5)
        assert_allclose(sw1, sw2 - 2.5, atol=1e-7)



class TestGenerateWaves(object):
    @parameterized.expand([
        (sine_wave,),
        (cosine_wave,),
        (square_wave,),
        (triangle_wave,),
        (sawtooth,),
        (inverse_sawtooth,),
    ])
    def testByFFT(self, fn):
        """Test sine_wave by computing FFT dominant frequency"""
        sw = fn(25., 400.0, 1.0, 10.)
        fft = compute_fft(sw, 400.)
        df = fft.dominant_frequency()
        assert_true(abs(df - 25.0) < 0.1)

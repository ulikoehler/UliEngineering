#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from nose.tools import assert_equal, assert_true, raises, assert_less
from UliEngineering.SignalProcessing.FFT import *
import numpy as np

class TestFFT(object):
    def testBasicFFT(self):
        rand = np.random.random(1000) * 5.0 + 1.0 # +1: Artifical DC artifacts
        x, y = computeFFT(rand, 10.0)
        assert_equal(x.shape, (rand.shape[0] / 2, ))
        assert_equal(y.shape, (rand.shape[0] / 2, ))
        assert_equal(x.shape, y.shape)
        # Test if artifacts can be cut
        origLength = x.shape[0]
        x, y = cutFFTDCArtifacts(x, y)
        assert_less(x.shape[0], origLength)

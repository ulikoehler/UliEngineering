#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from nose.tools import assert_equal, assert_true, raises
from UliEngineering.SignalProcessing.FFT import *
import numpy as np

class TestFFT(object):
    def testBasicFFT(self):
        rand = np.random.random(1000)
        x, y = computeFFT(rand, 10.0)
        assert_equal(x.shape, (rand.shape[0] / 2, ))
        assert_equal(y.shape, (rand.shape[0] / 2, ))
        assert_equal(x.shape, y.shape)

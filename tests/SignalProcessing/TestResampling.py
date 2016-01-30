#!/usr/bin/env python3
import numpy as np
from nose.tools import assert_equal, assert_true, raises
from numpy.testing import assert_allclose
from UliEngineering.SignalProcessing.Resampling import *

class TestResampling(object):
    def testDiscard(self):
        x = np.arange(10)
        assert_allclose(resample_discard(x, 2), [0, 2, 4, 6, 8])
        assert_allclose(resample_discard(x, 3), [0, 3, 6, 9])

    def testBSplineResampler(self):
        x = np.arange(100)
        y = np.square(x)
        y2 = BSplineResampler(x, y, time_factor=1).resample_to(1)

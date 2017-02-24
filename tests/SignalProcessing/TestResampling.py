#!/usr/bin/env python3
import numpy as np
from nose.tools import assert_equal, assert_true, raises, assert_raises
from numpy.testing import assert_allclose
from UliEngineering.SignalProcessing.Resampling import *


class TestBSplineResampling(object):
    def __init__(self):
        self.x = np.arange(100)
        self.y = np.square(self.x)

    def testDiscard(self):
        x = np.arange(10)
        assert_allclose(resample_discard(x, 2), [0, 2, 4, 6, 8])
        assert_allclose(resample_discard(x, 3), [0, 3, 6, 9])

class TestParallelResampling(object):
    def __init__(self):
        self.x = np.arange(100)
        self.y = np.square(self.x)

    def testSimpleCall(self):
        # Check if a simple call does not raise any exceptions
        print("foo")
        parallel_resample(self.x, self.y, 10.0, time_factor=1.0)

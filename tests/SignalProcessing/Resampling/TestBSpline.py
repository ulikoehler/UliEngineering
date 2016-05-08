#!/usr/bin/env python3
import numpy as np
from nose.tools import assert_equal, assert_true, raises, assert_raises
from numpy.testing import assert_allclose
from UliEngineering.SignalProcessing.Resampling.BSpline import *


class TestBSplineResampling(object):
    def __init__(self):
        self.x = np.arange(100)
        self.y = np.square(self.x)

    def testDiscard(self):
        x = np.arange(10)
        assert_allclose(resample_discard(x, 2), [0, 2, 4, 6, 8])
        assert_allclose(resample_discard(x, 3), [0, 3, 6, 9])

    def testBSplineResampler(self):
        y2 = BSplineResampler(self.x, self.y, time_factor=1).resample_to(1)

    def testResampledFilteredXYView(self):
        # TODO improve test
        y2 = ResampledFilteredXYView(self.x, self.y, 1.0, 1.0)
        y2[30:80]
        assert_equal(y2.shape, self.y.shape)
        assert_raises(TypeError, lambda: y2[1])
        assert_raises(TypeError, lambda: y2[self])

    def testResampledFilteredView(self):
        # TODO improve test
        y2 = ResampledFilteredView(self.x, self.y, 1.0, 1.0)
        y2[30:50]
        assert_equal(y2.shape, self.y.shape)
        assert_raises(TypeError, lambda: y2[1])
        assert_raises(TypeError, lambda: y2[self])

    def testResampledFilteredViewYOnlyDecorator(self):
        # TODO improve test
        y2 = ResampledFilteredXYView(self.x, self.y, 1.0, 1.0)
        y3 = ResampledFilteredViewYOnlyDecorator(y2)
        y3[30:80]
        assert_equal(y3.shape, self.y.shape)
        assert_raises(TypeError, lambda: y3[1])
        assert_raises(TypeError, lambda: y3[self])


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
from nose.tools import assert_equal, assert_true, raises, assert_less, assert_is_none, assert_raises
from UliEngineering.SignalProcessing.Utils import *
from nose_parameterized import parameterized
import concurrent.futures
import numpy as np
import datetime

unstairMethods = [
    ("left",),
    ("middle",),
    ("right",),
    ("reduce",),
]

class TestUnstair(object):
    @parameterized(unstairMethods)
    def testNoReduction(self, method):
        # Test if unstair returns the original array for a non-step function
        x = np.arange(10)
        y = np.square(x)
        xres, yres = unstair(x, y, method=method)
        assert_array_equal(xres, x)
        assert_array_equal(yres, y)

    def testSimpleLeft(self):
        y = np.asarray([0, 0, 0, 1, 2, 2, 2, 3, 4, 5, 5])
        x = np.arange(y.size)
        xexpected = [0, 3, 4, 7, 8, 9, 10]
        yexpected = y[xexpected]
        xres, yres = unstair(x, y, method="left")
        assert_array_equal(xres, xexpected)
        assert_array_equal(yres, yexpected)

    def testSimpleRight(self):
        y = np.asarray([0, 0, 0, 1, 2, 2, 2, 3, 4, 5, 5])
        x = np.arange(y.size)
        xexpected = [0, 2, 3, 6, 7, 8, 10]
        yexpected = y[xexpected]
        xres, yres = unstair(x, y, method="right")
        assert_array_equal(xres, xexpected)
        assert_array_equal(yres, yexpected)

    def testSimpleMiddle(self):
        y = np.asarray([0, 0, 0, 1, 2, 2, 2, 3, 4, 5, 5])
        x = np.arange(y.size)
        xexpected = [0, 1, 3, 5, 7, 8, 10]
        yexpected = y[xexpected]
        xres, yres = unstair(x, y, method="middle")
        assert_array_equal(xres, xexpected)
        assert_array_equal(yres, yexpected)

    def testSimpleReduce(self):
        y = np.asarray([0, 0, 0, 1, 2, 2, 2, 3, 4, 5, 5])
        x = np.arange(y.size)
        xexpected = [0, 2, 3, 4, 6, 7, 8, 9, 10]
        yexpected = y[xexpected]
        xres, yres = unstair(x, y, method="reduce")
        assert_array_equal(xres, xexpected)
        assert_array_equal(yres, yexpected)

    @parameterized(unstairMethods)
    def testSine(self, method):
        # Test with a rounded sine function. Data should be reduced
        sx = np.arange(1000) * .02
        rsine = np.round(np.sin(sx) * 10.) / 10.
        rx, ry = unstair(sx, rsine, method=method)
        assert_less(rx.size, sx.size)
        assert_less(ry.size, rsine.size)

class TestOptimumPolyfit(object):
    def testBasic(self):
        x = np.linspace(-100., 100., 10000)
        y = np.square(x)
        poly, deg, score = optimum_polyfit(x, y)
        assert_less(score, 1e-10)
        assert_equal(np.max(np.abs(y - poly(x))), score)

    def testRandom(self):
        x = np.linspace(-100., 100., 1000)
        y = np.random.random(x.size)
        poly, deg, score = optimum_polyfit(x, y)


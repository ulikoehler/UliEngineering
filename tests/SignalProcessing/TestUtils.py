#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
from UliEngineering.SignalProcessing.Utils import *
from parameterized import parameterized
import concurrent.futures
import numpy as np
import datetime
import unittest

unstairMethods = [
    ("left",),
    ("middle",),
    ("right",),
    ("reduce",),
]

class TestRemoveMean(unittest.TestCase):
    def testRemoveMean(self):
        assert_allclose(remove_mean([]), [])
        assert_allclose(remove_mean([1.0, 2.0, 3.0]), [-1.0, 0.0, 1.0])

class TestRMS(unittest.TestCase):
    def testRMS(self):
        assert_allclose(rms([]), [])
        assert_allclose(rms([1.0, 2.0, 3.0]), np.sqrt(np.mean([1*1, 2*2, 3*3])))

class TestPeakToPeak(unittest.TestCase):
    def testPeakToPeak(self):
        assert_allclose(peak_to_peak(None), 0.0)
        assert_allclose(peak_to_peak([]), 0.0)
        assert_allclose(peak_to_peak([0.0]), 0.0)
        assert_allclose(peak_to_peak([1.0]), 0.0)
        assert_allclose(peak_to_peak([1.0, 1.0]), 0.0)
        assert_allclose(peak_to_peak([1.0, 2.0]), 1.0)
        assert_allclose(peak_to_peak([2.0, 1.0]), 1.0)
        assert_allclose(peak_to_peak([0, 1, 3, -3, 0, 5, 0.7, 0.9]), 8)
        assert_allclose(peak_to_peak(np.asarray([])), 0.0)
        assert_allclose(peak_to_peak(np.asarray([0.0])), 0.0)
        assert_allclose(peak_to_peak(np.asarray([1.0])), 0.0)
        assert_allclose(peak_to_peak(np.asarray([1.0, 1.0])), 0.0)
        assert_allclose(peak_to_peak(np.asarray([1.0, 2.0])), 1.0)
        assert_allclose(peak_to_peak(np.asarray([2.0, 1.0])), 1.0)
        assert_allclose(peak_to_peak(np.asarray([0, 1, 3, -3, 0, 5, 0.7, 0.9])), 8)

class TestUnstair(unittest.TestCase):
    @parameterized.expand(unstairMethods)
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

    @parameterized.expand(unstairMethods)
    def testSine(self, method):
        # Test with a rounded sine function. Data should be reduced
        sx = np.arange(1000) * .02
        rsine = np.round(np.sin(sx) * 10.) / 10.
        rx, ry = unstair(sx, rsine, method=method)
        assert_less(rx.size, sx.size)
        assert_less(ry.size, rsine.size)

class TestOptimumPolyfit(unittest.TestCase):
    def testBasic(self):
        x = np.linspace(-100., 100., 10000)
        y = np.square(x)
        poly, deg, score = optimum_polyfit(x, y)
        assert_less(score, 1e-10)
        self.assertEqual(np.max(np.abs(y - poly(x))), score)

    def testRandom(self):
        x = np.linspace(-100., 100., 1000)
        y = np.random.random_sample(x.size)
        poly, deg, score = optimum_polyfit(x, y)


class TestLinSpace(unittest.TestCase):
    @parameterized.expand([
        (0.0, 100.0, 101, True),
        (0.0, 100.0, 202, True),
        (0.0, 100.0, 735, True),
        (0.0, 200.0, 101, True),
        (12.5, 202.3, 101, True),
        (0.0, 100.0, 101, False),
        (0.0, 100.0, 202, False),
        (0.0, 100.0, 735, False),
        (0.0, 200.0, 101, False),
        (12.5, 202.3, 101, False),
    ])
    def testBasic(self, start, end, n, endpoint):
        params = (start, end, n)
        spc = LinRange(*params, endpoint=endpoint)
        linspc = np.linspace(*params, endpoint=endpoint)
        self.assertEqual(len(spc), params[2])
        self.assertEqual(len(spc), linspc.size)
        self.assertEqual((len(spc),), linspc.shape)
        assert_allclose(spc[:], linspc)
        # Test samplerate
        assert_approx_equal(spc.samplerate(), (n - 1 if endpoint else n) / (end - start))
        # Test some slice
        istart, iend = len(spc) // 3, len(spc) // 2
        assert_allclose(spc[istart:iend], linspc[istart:iend])
        # Test negative indices
        assert_allclose(spc[-istart], linspc[-istart])
        # Test mid
        self.assertEqual(spc.mid, (start + end) / 2.)
        # Test view
        assert_allclose(spc.view(0, None).size, linspc.size)
        assert_allclose(spc.view(0, None)[:], linspc)

    def test_equal(self):
        l1 = LinRange(0., 100., 100, endpoint=False)
        l2 = LinRange(0., 100., 100, endpoint=False)
        l3 = LinRange(0., 100., 100, endpoint=True)
        self.assertTrue(l1 == l2)
        self.assertTrue(l2 == l1)
        self.assertFalse(l3 == l1)
        self.assertFalse(l3 == l2)

    def test_repr(self):
        l = LinRange(0., 100., 100, endpoint=False)
        self.assertEqual("LinRange(0.0, 100.0, 1.0)", str(l))
        l = LinRange(0., 100., 100, endpoint=False, dtype=np.int)
        self.assertEqual("LinRange(0.0, 100.0, 1.0, dtype=int)", str(l))

    def testDtype(self):
        lin1 = LinRange(0.0, 100.0, 101)
        assert_is_instance(lin1, LinRange)
        assert_is_instance(lin1.view(0, 5), LinRange)

class TestAggregate(unittest.TestCase):
    def test_aggregate(self):
        self.assertEqual([("a", 1), ("b", 1), ("c", 1)], list(aggregate("abc")))
        self.assertEqual([], list(aggregate("")))
        self.assertEqual([("a", 2), ("b", 1), ("c", 2), ("d", 1)],
            list(aggregate("aabccd")))

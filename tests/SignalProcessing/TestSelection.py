#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from nose.tools import assert_equal, assert_true, raises, assert_less, assert_is_none, assert_raises
from UliEngineering.SignalProcessing.Selection import *
from nose_parameterized import parameterized
import concurrent.futures
import numpy as np
import datetime

executor = concurrent.futures.ThreadPoolExecutor(4)

class TestIntInterval(object):
    def testAdd(self):
        assert_equal(IntInterval(1, 10) + 5, (6, 15))
        assert_equal(5 + IntInterval(1, 10), (6, 15))

    def testSub(self):
        assert_equal(IntInterval(6, 15) - 5, (1, 10))
        assert_equal(5 - IntInterval(6, 15), (-1, -10))

    def testMultiplyDivide(self):
        # == 1
        assert_equal(IntInterval(1, 10) * 1, (1, 10))
        assert_equal(IntInterval(1, 10) * 1.0, (1, 10))
        # > 1
        assert_equal(IntInterval(5, 10) * 2, (3, 12))
        assert_equal(IntInterval(5, 10) * 3, (0, 15))
        assert_equal(IntInterval(5, 10) * 5, (-5, 20))
        assert_equal(IntInterval(5, 10) * 3.0, (0, 15))
        # < 1
        assert_equal(IntInterval(5, 10) * 0.5, (6, 9))
        assert_equal(IntInterval(5, 10) * 0.1, (7, 8))
        assert_equal(IntInterval(5, 10) * 1e-6, (7, 8))
        # == 0
        assert_equal(IntInterval(5, 10) * 0.0, (7, 7))
        # Division, assumed to be implemented via multiplication
        assert_equal(IntInterval(5, 10) / (1. / 3.), (0, 15))

    @parameterized([
        (lambda iv: 'b' - iv,),
        (lambda iv: iv + 'b',),
        (lambda iv: 'b' + iv,),
        (lambda iv: iv * 'b',),
        (lambda iv: 'b' * iv,),
        (lambda iv: iv / 'b',),
    ])
    @raises(TypeError)
    def testInvalidArithmetic(self, lamb):
        iv = IntInterval(5, 10)
        lamb(iv)

    def testCall(self):
        x = np.arange(100)
        assert_allclose(IntInterval(6, 15)(x), np.arange(6, 15))
        # Multiple args
        y = np.arange(1, 100)
        assert_allclose(IntInterval(6, 15)(x, y), (np.arange(6, 15), np.arange(7, 16)))

    @raises(TypeError)
    def testCallNoArgs(self):
        IntInterval(6, 15)()

    def testLen(self):
        assert_equal(len(IntInterval(6, 15)), 15 - 6)

    @raises
    def testInvalidAdd(self):
        IntInterval(1, 10) + self


class TestSelectByDatetime(object):
    def testGeneric(self):
        now = datetime.datetime.now()
        # Test out of range
        assert_equal(selectByDatetime(np.arange(10), now), 10)
        # Test in range
        ts = now.timestamp()
        seconds50 = datetime.timedelta(seconds=50)
        r = np.arange(ts, ts + 100)
        assert_equal(selectByDatetime(r, now + seconds50), 50)
        # Test with string
        assert_equal(selectByDatetime(np.arange(10), "2015-01-29 23:59:01"), 10)
        # Test with string w/ microseconds
        assert_equal(selectByDatetime(np.arange(10), "2015-01-29 23:59:01.000001"), 10)
        # Test with around set
        assert_equal(selectByDatetime(np.arange(10), now, around=5), (5, 15))

    @parameterized.expand([
        ("whatever"),
        ("2015-01-2923:59:01"),  # Missing space
        ("2015-01-29 23:59:01,000001"),  # Comma instead of point
        ("2015-01-29 23:59:01.00001"),  # Not enough digits
        ("2015-01-29 23:59:01.0000001"),  # Too many digits
        ("2015-01-29 23:59:01.00000a1"),  # a is not a digit
    ])
    @raises
    def testInvalidStringFormat(self, str):
        selectByDatetime(np.arange(10), "2015-01-29 23:59:01")

    @raises
    def testNoneArray(self, str):
        selectByDatetime(None, "2015-01-29 23:59:01")

    @raises
    def testNoneTimestamp(self, str):
        selectByDatetime(np.arange(10), None)

class TestSelectFrequencyRange(object):
    def testGeneric(self):
        arr = np.arange(0.0, 10.0)
        result = selectFrequencyRange(arr, arr + 1.0, 1.0, 5.5)
        desired = np.asarray([2.0, 3.0, 4.0, 5.0, 6.0])
        assert_allclose(result, (desired - 1.0, desired))

    def testNoneLimit(self):
        arr = np.arange(0.0, 10.0)
        # Low = None
        result = selectFrequencyRange(arr, arr + 1.0, high=5.0)
        desired = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        assert_allclose(result, (desired - 1.0, desired))
        # High = None
        result = selectFrequencyRange(arr, arr + 1.0, low=5.0)
        desired = np.asarray([6.0, 7.0, 8.0, 9.0, 10.0])
        assert_allclose(result, (desired - 1.0, desired))

    def testTupleUnpacking(self):
        "Test tuple unpacking for FFT inlining"
        arr = np.arange(0.0, 10.0)
        result = selectFrequencyRange((arr, arr + 1.0), low=1.0, high=5.5)
        desired = np.asarray([2.0, 3.0, 4.0, 5.0, 6.0])
        assert_allclose(result, (desired - 1.0, desired))


class TestFindSortedExtrema(object):
    def testGreater(self):
        x = np.arange(10)
        y = np.zeros(10)
        y[2] = 1.0  # Not the largest maximum
        y[6] = 5.0

        assert_allclose(findSortedExtrema(x, y), [[6.0, 5.0], [2.0, 1.0]])

    def testLess(self):
        x = np.arange(10)
        y = np.zeros(10)
        y[2] = -1.0  # Not the largest maximum
        y[6] = -5.0

        assert_allclose(findSortedExtrema(x, y, comparator=np.less),
                        [[6.0, -5.0], [2.0, -1.0]])

    @raises(ValueError)
    def testInvalidComparator(self):
        findSortedExtrema(None, None, comparator=map)

class TestSelectByThreshold(object):

    def testGreater(self):
        x = np.linspace(100, 199, 100)
        y = np.random.random(100)
        y[32] = 8.0
        y[92] = 5.5
        y[98] = 4.5
        assert_allclose(selectByThreshold(x, y, 5.0), [[132., 8.], [192., 5.5]])

    @raises(ValueError)
    def testInvalidComparator(self):
        selectByThreshold(None, None, 1.0, comparator=map)


class TestFindTrueRuns(object):
    def testSimple(self):
        x = np.full(25, False)
        x[4:9] = True
        x[14:21] = True
        assert_allclose(findTrueRuns(x), [[4, 9], [14, 21]])

    def testNumericComparator(self):
        # Generate test data
        x = np.zeros(25)
        x[4:9] = 1.0
        x[5] = 2.0
        x[14:21] = 1.0
        x[20] = 3.0
        ranges = findTrueRuns(x > 0.5)
        assert_allclose(ranges, [[4, 9], [14, 21]])

    def testEdges(self):
        x = np.full(25, False)
        x[0:9] = True
        x[14:24] = True
        assert_allclose(findTrueRuns(x), [[0, 9], [14, 24]])

    def testNone(self):
        x = np.full(25, False)
        assert_allclose(findTrueRuns(x), np.zeros((0, 2)))

    def testEmpty(self):
        x = np.full(0, False)
        assert_allclose(findTrueRuns(x), np.zeros((0, 2)))

class TestShrinkRanges(object):
    def testSimple(self):
        # Generate test data
        x = np.zeros(25)
        x[4:9] = 1.0
        x[5] = 2.0
        x[14:21] = 1.0
        x[20] = 3.0
        x[22] = 4.0
        ranges = findTrueRuns(x > 0.5)
        # Run shrinker
        assert_allclose(shrinkRanges(ranges, x), [5, 20, 22])

    @raises(KeyError)
    def testInvalidFunction(self):
        shrinkRanges(np.zeros(5), None, "foobar")


class TestRandomSelection(object):
    def testBasic(self):
        # Just test if it does not raise
        selectRandomSlice(100, 10)
        selectRandomSlice(np.arange(100), 10)
        # Test array which is just large enough
        selectRandomSlice(np.arange(10), 10)

    @parameterized.expand([
        ("number", 5),
        ("numpy array", np.arange(5))
    ])
    @raises(ValueError)
    def testTooSmall(self, _, arr):
        "Test if arrays which are too small are handled correctly"
        selectRandomSlice(arr, 10)


class TestFindNearestIdx(object):
    def testBasic(self):
        assert_equal(findNearestIdx(np.arange(10), 5.0), 5)
        assert_equal(findNearestIdx(np.arange(3, 13), 5.0), 2)
        assert_equal(findNearestIdx(np.arange(3, 13), 5.2), 2)
        assert_equal(findNearestIdx(np.arange(3, 13), 5.4), 2)
        assert_equal(findNearestIdx(np.arange(3, 13), 5.5), 2)
        assert_equal(findNearestIdx(np.arange(3, 13), 5.500001), 3)
        assert_equal(findNearestIdx(np.arange(3, 13), 5.6), 3)

class TestGeneratorCount(object):
    @parameterized([
        (True,), (False,)
    ])
    def testBasic(self, generator):
        lst = np.arange(5)
        gc = GeneratorCounter((i for i in lst) if generator else lst)
        assert_equal(len(gc), 0)  # No iterations yet
        assert_equal(list(gc), [0, 1, 2, 3, 4])
        assert_equal(len(gc), len(lst))
        # Test reiter
        if not generator:
            gc.reiter(reset_count=False)
            assert_equal(len(gc), len(lst))
            assert_equal(list(gc), list(lst))
            assert_equal(len(gc), len(lst) * 2)
            # With count reset
            gc.reiter(reset_count=True)
            assert_equal(len(gc), 0)
            assert_equal(list(gc), list(lst))
            assert_equal(len(gc), len(lst))


class TestMajorityVote(object):
    def testBasic(self):
        lst = [1, 2, 3, 3]
        res = majority_vote_all(lst)
        # Check mv_all
        assert_true(res == [(3, 0.5), (2, 0.25), (1, 0.25)] or res == [(3, 0.5), (1, 0.25), (2, 0.25)])
        assert_equal(majority_vote(lst), 3)
        assert_is_none(majority_vote([]))


class TestExtractByReference(object):
    def testBasic(self):
        fx = np.arange(100)
        fy = np.square(fx)
        ref = np.linspace(20.0, 50.0, 1000)
        resX, resY = extract_by_reference(fx, fy, ref)
        assert_allclose(resX, fx[20:50])
        assert_allclose(resY, fy[20:50])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
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

    def testRangeArrayToIntIntervals(self):
        arr = np.asarray([[41, 60], [0, 30]])
        res = IntInterval.from_ranges(arr)
        assert_equal(res, [IntInterval(41, 60), IntInterval(0, 30)])


    def testIntIntervalsToRangeArray(self):
        arr = np.asarray([[41, 60], [0, 30]])
        intervals = [IntInterval(41, 60), IntInterval(0, 30)]
        assert_allclose(IntInterval.to_ranges(intervals), arr)


class TestSelectByDatetime(object):
    def testGeneric(self):
        now = datetime.datetime.now()
        # Test out of range
        assert_equal(select_by_datetime(np.arange(10), now), 10)
        # Test in range
        ts = now.timestamp()
        seconds50 = datetime.timedelta(seconds=50)
        r = np.arange(ts, ts + 100)
        assert_equal(select_by_datetime(r, now + seconds50), 50)
        # Test with string
        assert_equal(select_by_datetime(np.arange(10), "2015-01-29 23:59:01"), 10)
        # Test with string w/ microseconds
        assert_equal(select_by_datetime(np.arange(10), "2015-01-29 23:59:01.000001"), 10)
        # Test with around set
        assert_equal(select_by_datetime(np.arange(10), now, around=5), (5, 15))

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
        select_by_datetime(np.arange(10), "2015-01-29 23:59:01")

    @raises
    def testNoneArray(self, str):
        select_by_datetime(None, "2015-01-29 23:59:01")

    @raises
    def testNoneTimestamp(self, str):
        select_by_datetime(np.arange(10), None)

class Testfind_sorted_extrema(object):
    def testGreater(self):
        x = np.arange(10)
        y = np.zeros(10)
        y[2] = 1.0  # Not the largest maximum
        y[6] = 5.0

        assert_allclose(find_sorted_extrema(x, y), [[6.0, 5.0], [2.0, 1.0]])

    def testLess(self):
        x = np.arange(10)
        y = np.zeros(10)
        y[2] = -1.0  # Not the largest maximum
        y[6] = -5.0

        assert_allclose(find_sorted_extrema(x, y, comparator=np.less),
                        [[6.0, -5.0], [2.0, -1.0]])

    @raises(ValueError)
    def testInvalidComparator(self):
        find_sorted_extrema(None, None, comparator=map)

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


class TestFindRuns(object):
    def testSimple(self):
        x = np.full(25, False, np.bool)
        x[4:9] = True
        x[14:21] = True
        # Test find_true_runs
        result = find_true_runs(x)
        assert_allclose(result, [[4, 8], [14, 20]])
        assert_equal(result.dtype, np.int)
        # Test findFalseRuns
        result = find_false_runs(x)
        assert_allclose(result, [[0, 3], [9, 13], [21, 24]])
        assert_equal(result.dtype, np.int)

    def testSize1(self):
        # Generate test data
        x = np.zeros(25)
        x[5] = 2.0
        ranges = find_true_runs(x > 0.5)
        assert_allclose(ranges, [[5, 5]])
        # First element
        x = np.zeros(25)
        x[0] = 2.0
        ranges = find_true_runs(x > 0.5)
        assert_allclose(ranges, [[0, 0]])

    def testStart(self):
        # Generate test data
        x = np.full(25, False, np.bool)
        x[:3] = True
        ranges = find_true_runs(x > 0.5)
        assert_allclose(ranges, [[0, 2]])

    def testEnd(self):
        # Generate test data
        x = np.full(25, False, np.bool)
        # One
        x[24] = True
        ranges = find_true_runs(x > 0.5)
        assert_allclose(ranges, [[24, 24]])
        # Multiple
        x[22:24] = True
        ranges = find_true_runs(x > 0.5)
        assert_allclose(ranges, [[22, 24]])

    def testNumericComparator(self):
        # Generate test data
        x = np.zeros(25)
        x[4:9] = 1.0
        x[5] = 2.0
        x[14:21] = 1.0
        x[20] = 3.0
        ranges = find_true_runs(x > 0.5)
        assert_allclose(ranges, [[4, 8], [14, 20]])

    def testEdges(self):
        "Test range with both edges at once"
        x = np.full(25, False)
        x[0:9] = True
        x[14:24] = True
        assert_allclose(find_true_runs(x), [[0, 8], [14, 23]])

    def testFull(self):
        "Test range with all elements True"
        x = np.full(25, True)
        assert_allclose(find_true_runs(x), [[0, 24]])

    def testNone(self):
        x = np.full(25, False)
        assert_allclose(find_true_runs(x), np.zeros((0, 2)))

    def testEmpty(self):
        x = np.full(0, False)
        assert_allclose(find_true_runs(x), np.zeros((0, 2)))


class TestShrinkRanges(object):
    def __init__(self):
        self.x = np.zeros(25)
        self.x[4:9] = 1.0
        self.x[5] = 2.0
        self.x[6] = 0.51
        self.x[14:21] = 1.0
        self.x[17] = 0.6
        self.x[20] = 3.0
        self.x[22] = 4.0

    def testSimple(self):
        """Test simple (non data-aware) methods"""
        # Generate test data. Tuned so there are definitive min/max values
        ranges = find_true_runs(self.x > 0.5)
        # Min selector
        result = shrink_ranges(ranges, "min")
        assert_allclose(result, [4, 14, 22])
        assert_equal(result.dtype, np.int)
        # Max selector
        result = shrink_ranges(ranges, "max")
        assert_allclose(result, [8, 20, 22])
        assert_equal(result.dtype, np.int)
        # Middle selector
        result = shrink_ranges(ranges, "middle")
        assert_allclose(result, [6, 17, 22])
        assert_equal(result.dtype, np.int)

    def testComplex(self):
        """Test data-aware methods"""
        # Test data is tuned to
        ranges = find_true_runs(self.x > 0.5)
        # Min Y selector
        result = shrink_ranges(ranges, "miny", y=self.x)
        assert_allclose(result, [6, 17, 22])
        assert_equal(result.dtype, np.int)
        # Max Y selector
        result = shrink_ranges(ranges, "maxy", y=self.x)
        assert_allclose(result, [5, 20, 22])
        assert_equal(result.dtype, np.int)

    @raises(KeyError)
    def testInvalidFunction(self):
        shrink_ranges(np.zeros(5), "foobar")


class TestRandomSelection(object):
    def testBasic(self):
        # Just test if it does not raise
        random_slice(100, 10)
        random_slice(np.arange(100), 10)
        # Test array which is just large enough
        random_slice(np.arange(10), 10)

    @parameterized.expand([
        ("number", 5),
        ("numpy array", np.arange(5))
    ])
    @raises(ValueError)
    def testTooSmall(self, _, arr):
        "Test if arrays which are too small are handled correctly"
        random_slice(arr, 10)


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


class TestSelectRanges(object):
    def testBasic(self):
        arr = np.arange(10)
        ranges = np.asarray([[3, 5], [7, 8]])
        result = list(select_ranges(ranges, arr))
        assert_array_equal(result[0], [3, 4])
        assert_array_equal(result[1], [7])

class TestMultiSelect(object):
    def testBasic(self):
        assert_equal([4, 2, 6], multiselect([1, 2, 3, 4, 5, 6], [3, 1, 5]))

    def testEmpty(self):
        assert_equal([], multiselect([], []))

    def testShort(self):
        assert_equal([1], multiselect([1], [0]))

    @raises(IndexError)
    def testBadIdx(self):
        assert_equal([1], multiselect([1], [3]))

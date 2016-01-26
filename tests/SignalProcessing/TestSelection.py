#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from nose.tools import assert_equal, assert_true, raises, assert_less
from UliEngineering.SignalProcessing.Selection import *
from nose_parameterized import parameterized
import concurrent.futures
import numpy as np
import datetime

executor = concurrent.futures.ThreadPoolExecutor(4)

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

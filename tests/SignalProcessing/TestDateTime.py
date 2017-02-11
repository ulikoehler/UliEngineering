#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
from nose.tools import assert_equal, assert_true, raises, assert_less, assert_is_none, assert_raises
from UliEngineering.SignalProcessing.DateTime import *
from nose_parameterized import parameterized
import concurrent.futures
import numpy as np
import datetime

class TestSpliceDate(object):
    def test_simple(self):
        d1 = datetime.datetime(2016, 1, 1, 12, 32, 15, microsecond=123456)
        d2 = datetime.datetime(1905, 1, 1, 14, 11, 25, microsecond=52)
        dres = datetime.datetime(2016, 1, 1, 14, 11, 25, microsecond=52)
        assert_equal(dres, splice_date(d1, d2))


class TestAutoStrptime(object):
    def test_formats(self):
        #%Y-%m-%d %H:%M:%S.%f
        assert_equal(datetime.datetime(2016, 2, 1, 15, 2, 11, 50), auto_strptime("2016-02-01 15:02:11.000050"))
        assert_equal(datetime.datetime(2016, 2, 1, 15, 2, 11, 50), auto_strptime(" 2016-02-01 15:02:11.000050"))
        assert_equal(datetime.datetime(2016, 2, 1, 15, 2, 11, 50), auto_strptime("2016-02-01 15:02:11.000050 "))
        #%H:%M:%S.%f
        assert_equal(datetime.datetime(1900, 1, 1, 15, 2, 11, 50), auto_strptime("15:02:11.000050"))
        #%Y-%m-%d
        assert_equal(datetime.datetime(2016, 2, 1), auto_strptime("2016-02-01"))
        assert_equal(datetime.datetime(2016, 2, 1), auto_strptime("2016-02-01 "))
        #%H:%M:%S
        assert_equal(datetime.datetime(1900, 1, 1, 15, 2, 11), auto_strptime("15:02:11"))
        assert_equal(datetime.datetime(1900, 1, 1, 15, 2, 11), auto_strptime("15:02:11 "))
        assert_equal(datetime.datetime(1900, 1, 1, 15, 2, 11), auto_strptime(" 15:02:11"))
        #%Y-%m-%d %H
        assert_equal(datetime.datetime(2016, 2, 1, 15, 0, 0), auto_strptime("2016-02-01 15"))
        assert_equal(datetime.datetime(2016, 2, 1, 15, 0, 0), auto_strptime(" 2016-02-01 15"))
        assert_equal(datetime.datetime(2016, 2, 1, 15, 0, 0), auto_strptime("2016-02-01 15"))
        #%Y-%m-%d %H
        assert_equal(datetime.datetime(2016, 2, 1, 15, 2, 0), auto_strptime("2016-02-01 15:02"))
        assert_equal(datetime.datetime(2016, 2, 1, 15, 2, 0), auto_strptime("2016-02-01 15:02 "))
        assert_equal(datetime.datetime(2016, 2, 1, 15, 2, 0), auto_strptime(" 2016-02-01 15:02"))
        #%Y-%m-%d %H:%M:%S
        assert_equal(datetime.datetime(2016, 2, 1, 15, 2, 11), auto_strptime("2016-02-01 15:02:11"))
        assert_equal(datetime.datetime(2016, 2, 1, 15, 2, 11), auto_strptime("2016-02-01 15:02:11 "))
        assert_equal(datetime.datetime(2016, 2, 1, 15, 2, 11), auto_strptime(" 2016-02-01 15:02:11"))

    def test_examples(self):
        assert_equal(datetime.datetime(2016, 7, 21, 00, 00, 00), auto_strptime("2016-07-21 00:00:00"))
        assert_equal(datetime.datetime(2016, 7, 21, 3, 00, 00), auto_strptime("2016-07-21 03:00:00"))
        assert_equal(datetime.datetime(2016, 9, 1, 10, 00, 00), auto_strptime("2016-09-01 10"))

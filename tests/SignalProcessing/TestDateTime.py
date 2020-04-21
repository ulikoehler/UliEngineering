#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
from nose.tools import self.assertEqual, assert_true, raises, assert_less, assert_is_none, assert_raises
from UliEngineering.SignalProcessing.DateTime import *
from parameterized import parameterized
import concurrent.futures
import numpy as np
import datetime
import unittest

class TestSpliceDate(unittest.TestCase):
    def test_simple(self):
        d1 = datetime.datetime(2016, 1, 1, 12, 32, 15, microsecond=123456)
        d2 = datetime.datetime(1905, 1, 1, 14, 11, 25, microsecond=52)
        dres = datetime.datetime(2016, 1, 1, 14, 11, 25, microsecond=52)
        self.assertEqual(dres, splice_date(d1, d2))


class TestAutoStrptime(unittest.TestCase):
    def test_formats(self):
        #%Y-%m-%d %H:%M:%S.%f
        self.assertEqual(datetime.datetime(2016, 2, 1, 15, 2, 11, 50), auto_strptime("2016-02-01 15:02:11.000050"))
        self.assertEqual(datetime.datetime(2016, 2, 1, 15, 2, 11, 50), auto_strptime(" 2016-02-01 15:02:11.000050"))
        self.assertEqual(datetime.datetime(2016, 2, 1, 15, 2, 11, 50), auto_strptime("2016-02-01 15:02:11.000050 "))
        #%H:%M:%S.%f
        self.assertEqual(datetime.datetime(1900, 1, 1, 15, 2, 11, 50), auto_strptime("15:02:11.000050"))
        #%Y-%m-%d
        self.assertEqual(datetime.datetime(2016, 2, 1), auto_strptime("2016-02-01"))
        self.assertEqual(datetime.datetime(2016, 2, 1), auto_strptime("2016-02-01 "))
        #%H:%M:%S
        self.assertEqual(datetime.datetime(1900, 1, 1, 15, 2, 11), auto_strptime("15:02:11"))
        self.assertEqual(datetime.datetime(1900, 1, 1, 15, 2, 11), auto_strptime("15:02:11 "))
        self.assertEqual(datetime.datetime(1900, 1, 1, 15, 2, 11), auto_strptime(" 15:02:11"))
        #%Y-%m-%d %H
        self.assertEqual(datetime.datetime(2016, 2, 1, 15, 0, 0), auto_strptime("2016-02-01 15"))
        self.assertEqual(datetime.datetime(2016, 2, 1, 15, 0, 0), auto_strptime(" 2016-02-01 15"))
        self.assertEqual(datetime.datetime(2016, 2, 1, 15, 0, 0), auto_strptime("2016-02-01 15"))
        #%Y-%m-%d %H
        self.assertEqual(datetime.datetime(2016, 2, 1, 15, 2, 0), auto_strptime("2016-02-01 15:02"))
        self.assertEqual(datetime.datetime(2016, 2, 1, 15, 2, 0), auto_strptime("2016-02-01 15:02 "))
        self.assertEqual(datetime.datetime(2016, 2, 1, 15, 2, 0), auto_strptime(" 2016-02-01 15:02"))
        #%Y-%m-%d %H:%M:%S
        self.assertEqual(datetime.datetime(2016, 2, 1, 15, 2, 11), auto_strptime("2016-02-01 15:02:11"))
        self.assertEqual(datetime.datetime(2016, 2, 1, 15, 2, 11), auto_strptime("2016-02-01 15:02:11 "))
        self.assertEqual(datetime.datetime(2016, 2, 1, 15, 2, 11), auto_strptime(" 2016-02-01 15:02:11"))

    def test_examples(self):
        self.assertEqual(datetime.datetime(2016, 7, 21, 00, 00, 00), auto_strptime("2016-07-21 00:00:00"))
        self.assertEqual(datetime.datetime(2016, 7, 21, 3, 00, 00), auto_strptime("2016-07-21 03:00:00"))
        self.assertEqual(datetime.datetime(2016, 9, 1, 10, 00, 00), auto_strptime("2016-09-01 10"))

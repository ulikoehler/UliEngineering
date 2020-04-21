#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
from nose.tools import assert_equal, assert_true, assert_false, raises, assert_less, assert_is_none, assert_raises, assert_is_instance
from UliEngineering.SignalProcessing.Correlation import *
from parameterized import parameterized
import concurrent.futures
import numpy as np
import datetime
import unittest

class TestCorrelation(unittest.TestCase):
    def testAutocorrelation(self):
        # Just test no exceptions
        autocorrelate(np.random.random_sample(1000))
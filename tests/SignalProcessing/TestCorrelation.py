#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
from nose.tools import assert_equal, assert_true, assert_false, raises, assert_less, assert_is_none, assert_raises, assert_is_instance
from UliEngineering.SignalProcessing.Weight import *
from nose_parameterized import parameterized
import concurrent.futures
import numpy as np
import datetime

class TestWeighHalves(object):
    def testWeighHalves(self):
        # Empty array
        assert_allclose(weigh_halves(np.zeros(0)), (0, 0))
        # Even array size
        assert_allclose(weigh_halves(np.arange(4)), (1., 5.))
        # Odd array size
        assert_allclose(weigh_halves(np.arange(5)), (2., 8.))

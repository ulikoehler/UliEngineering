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

class TestWeightSymmetry(object):
    def testWeightSymmetry(self):
        assert_approx_equal(weight_symmetry(0.5, 0.5), 1.0)
        assert_approx_equal(weight_symmetry(0.02, 0.98), 0.04)
        assert_approx_equal(weight_symmetry(0.0, 1.0), 0.0)
        # Independence of scale
        assert_approx_equal(weight_symmetry(0.02*13, 0.98*13), 0.04)
        # Example usecase
        assert_approx_equal(weight_symmetry(*weigh_halves(np.arange(4))), 1/3.)
        assert_approx_equal(weight_symmetry(*weigh_halves(np.arange(5))), 0.4)

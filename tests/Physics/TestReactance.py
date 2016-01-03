#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from nose.tools import assert_equal
from UliEngineering.Physics.Reactance import *
import numpy as np

class TestNoiseDensity(object):
    def test_capacitive_reactance(self):
        assert_approx_equal(capacitive_reactance("100 pF", "3.2 MHz"), 497.3592)
        assert_approx_equal(capacitive_reactance(100e-12, 3.2e6), 497.3592)

    def test_inductive_reactance(self):
        assert_approx_equal(inductive_reactance("100 µH", "3.2 MHz"), 2010.619)
        assert_approx_equal(inductive_reactance(100e-6, 3.2e6), 2010.619)

    def test_numpy_arrays(self):
        l = np.asarray([100e-6, 200e-6])
        assert_allclose(inductive_reactance(l, 3.2e6), [2010.6193, 4021.23859659])

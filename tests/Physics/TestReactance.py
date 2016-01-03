#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from nose.tools import assert_equal
from UliEngineering.Physics.Reactance import *

class TestNoiseDensity(object):
    def test_capacitive_reactance(self):
        assert_approx_equal(capacitive_reactance("100 pF", "3.2 MHz"), 497.3592)

    def test_inductive_reactance(self):
        assert_approx_equal(inductive_reactance("100 µH", "3.2 MHz"), 2010.619)

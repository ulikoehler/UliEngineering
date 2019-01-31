#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from nose.tools import assert_equal
from UliEngineering.Physics.RF import *
from UliEngineering.EngineerIO import auto_format
import numpy as np

class TestRF(object):
    def test_quality_facotr(self):
        assert_approx_equal(quality_factor("8.000 MHz", "1 kHz"), 8000.0)
        assert_approx_equal(quality_factor("8.000 MHz", "1 MHz"), 8.0)

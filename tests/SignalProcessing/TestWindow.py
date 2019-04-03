#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
from nose.tools import assert_equal, assert_true, assert_false, raises, assert_less, assert_is_none, assert_raises, assert_is_instance
from UliEngineering.SignalProcessing.Window import *
from parameterized import parameterized
import concurrent.futures
import numpy as np
import datetime

class TestWindow(object):
    def testWindowFunctor(self):
        data = np.random.random(1000) 
        ftor = WindowFunctor(len(data), "blackman")
        # normal
        result = ftor(data)
        assert_allclose(result, data * np.blackman(1000))
        # inplace
        result = ftor(data, inplace=True)
        assert_allclose(result, data)

    
    def testWindowFunctor(self):
        data = np.random.random(1000) 
        # normal
        result = create_and_apply_window(data, "blackman")
        assert_allclose(result, data * np.blackman(1000))
        # inplace
        result = create_and_apply_window(data, inplace=True)
        assert_allclose(result, data)

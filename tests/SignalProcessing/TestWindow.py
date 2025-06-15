#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_allclose
from UliEngineering.SignalProcessing.Window import *
import numpy as np
import unittest

class TestWindow(unittest.TestCase):
    def testWindowFunctor(self):
        data = np.random.random_sample(1000) 
        ftor = WindowFunctor(len(data), "blackman")
        # normal
        result = ftor(data)
        assert_allclose(result, data * np.blackman(1000))
        # inplace
        result = ftor(data, inplace=True)
        assert_allclose(result, data)

    def testWindow(self):
        data = np.random.random_sample(1000) 
        # normal
        result = create_and_apply_window(data, "blackman")
        assert_allclose(result, data * np.blackman(1000))
        # inplace
        result = create_and_apply_window(data, inplace=True)
        assert_allclose(result, data)

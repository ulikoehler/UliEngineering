#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from UliEngineering.SignalProcessing.Correlation import *
import numpy as np
import unittest

class TestCorrelation(unittest.TestCase):
    def testAutocorrelation(self):
        # Just test no exceptions
        autocorrelate(np.random.random_sample(1000))
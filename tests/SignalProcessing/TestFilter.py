#!/usr/bin/env python3
import numpy as np
from nose.tools import assert_equal, assert_true, raises
from numpy.testing import assert_array_equal, assert_array_less
from UliEngineering.SignalProcessing.Filter import *

class TestFilter(object):

    def __init__(self):
        self.d = np.random.random(1000)

    def testBasicFilter(self):
        filt = SignalFilter(100.0, [1.0, 2.0])
        filt.iir(order=3)
        filt(self.d)

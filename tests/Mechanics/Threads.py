#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from nose.tools import assert_equal
from UliEngineering.Mechanics.Threads import *
import unittest

class TestThreads(unittest.TestCase):
    def test_thread_params(self):
        assert_equal(threads["M3"].outer_diameter, 41)

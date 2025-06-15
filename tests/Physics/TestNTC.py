#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from UliEngineering.Physics.NTC import *
from UliEngineering.Exceptions import *
import unittest

class TestNTC(unittest.TestCase):
    def test_ntc_resistance(self):
        # Values arbitrarily from Murata NCP15WB473D03RC
        assert_approx_equal(ntc_resistance("47k", "4050K", "25°C"), 47000)
        assert_approx_equal(ntc_resistance("47k", "4050K", "0°C"), 162942.79)
        assert_approx_equal(ntc_resistance("47k", "4050K", "-18°C"), 463773.791)
        assert_approx_equal(ntc_resistance("47k", "4050K", "5°C"), 124819.66)
        assert_approx_equal(ntc_resistance("47k", "4050K", "60°C"), 11280.407)

    def test_ntc_resistances(self):
        # Currently mostly test if it runs
        ts, values = ntc_resistances("47k", "4050K")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from UliEngineering.Physics.MagneticResonance import *
from UliEngineering.EngineerIO import auto_format
import unittest

class TestLarmorFrequency(unittest.TestCase):
    def test_larmor_frequency_h1(self):
        self.assertAlmostEqual(larmor_frequency(0., nucleus_larmor_frequency=NucleusLarmorFrequency.H1), 0)
        self.assertAlmostEqual(larmor_frequency(1., nucleus_larmor_frequency=NucleusLarmorFrequency.H1), 42.57638474e6)
        self.assertAlmostEqual(larmor_frequency(2.2, nucleus_larmor_frequency=NucleusLarmorFrequency.H1), 2.2*42.57638474e6)
        # H1 hould be the standard value
        self.assertAlmostEqual(larmor_frequency(0.), 0)

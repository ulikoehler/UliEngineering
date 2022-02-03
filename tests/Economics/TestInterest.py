#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from subprocess import check_output
from UliEngineering.Economics.Interest import *
import unittest
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
import numpy as np

class TestEquivalentInterest(unittest.TestCase):
    def test_yearly_interest_to_equivalent_monthly_interest(self):
        self.assertAlmostEqual(interest_apply_multiple_times(
            yearly_interest_to_equivalent_monthly_interest(0), 1), 0.0)
        self.assertAlmostEqual(interest_apply_multiple_times(
            yearly_interest_to_equivalent_monthly_interest(0.22), 12), 0.22)
        self.assertAlmostEqual(interest_apply_multiple_times(
            yearly_interest_to_equivalent_monthly_interest(0.33), 12), 0.33)

    def test_yearly_interest_to_equivalent_daily_interest(self):
        self.assertAlmostEqual(interest_apply_multiple_times(
            yearly_interest_to_equivalent_daily_interest(0), 1), 0.0)
        self.assertAlmostEqual(interest_apply_multiple_times(
            yearly_interest_to_equivalent_daily_interest(0.22), 365.25), 0.22)
        self.assertAlmostEqual(interest_apply_multiple_times(
            yearly_interest_to_equivalent_daily_interest(0.33), 365.25), 0.33)

class TestInterestApplyMultipleTimes(unittest.TestCase):
    def test_interest_apply_multiple_times(self):
        self.assertAlmostEqual(interest_apply_multiple_times(0.22, 0), 0)
        self.assertAlmostEqual(interest_apply_multiple_times(0.22, 1), 0.22)
        self.assertAlmostEqual(interest_apply_multiple_times(0.22, 2), 1.22**2-1)

        self.assertAlmostEqual(interest_apply_multiple_times(0.33, 0), 0)
        self.assertAlmostEqual(interest_apply_multiple_times(0.33, 1), 0.33)
        self.assertAlmostEqual(interest_apply_multiple_times(0.33, 2), 1.33**2-1)


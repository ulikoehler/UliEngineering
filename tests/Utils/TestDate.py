#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from subprocess import check_output
from UliEngineering.Utils.Date import *
import unittest

class TestDate(unittest.TestCase):
    def test_number_of_days_in_month(self):
        self.assertEqual(number_of_days_in_month(2019, 1), 31)
        self.assertEqual(number_of_days_in_month(2019, 2), 28)

    def test_all_dates_in_year(self):
        # Wolfram Alpha: number of days in 2019
        self.assertEqual(len(list(all_dates_in_year(2019))), 365)
        # Wolfram Alpha: number of days in 2020
        self.assertEqual(len(list(all_dates_in_year(2020))), 366)

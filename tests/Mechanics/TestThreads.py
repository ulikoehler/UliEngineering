#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from UliEngineering.Mechanics.Threads import *
import unittest

class TestThreads(unittest.TestCase):
    def test_thread_params(self):
        self.assertEqual(threads["M3"].outer_diameter, 3.0)

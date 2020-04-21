#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
from UliEngineering.Utils.ZIP import *
from UliEngineering.Utils.Temporary import *
import unittest

class TestFileUtils(unittest.TestCase):
    def setUp(self):
        self.tmp = AutoDeleteTempfileGenerator()

    def create_zip_from_directory(self):
        pass #TODO
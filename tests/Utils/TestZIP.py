#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
from nose.tools import assert_equal, assert_true, assert_false
from UliEngineering.Utils.ZIP import *
from UliEngineering.Utils.Temporary import *

class TestFileUtils(object):
    def __init__(self):
        self.tmp = AutoDeleteTempfileGenerator()

    def create_zip_from_directory(self):
        pass #TODO
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from UliEngineering.Utils.Temporary import AutoDeleteTempfileGenerator
import unittest

class TestFileUtils(unittest.TestCase):
    def setUp(self):
        self.tmp = AutoDeleteTempfileGenerator()

    def create_zip_from_directory(self):
        pass #TODO
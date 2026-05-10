#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from UliEngineering.Optoelectronics.MPPC import pixel_capacitance_from_terminal_capacitance
from UliEngineering.EngineerIO.Types import NormalizableArgument
from UliEngineering.Electronics.Capacitors import CapacitanceFarad
import unittest

class TestMPPC(unittest.TestCase):
    def test_pixel_capacitance_from_terminal_capacitance(self):
        # Test with numeric inputs
        result = pixel_capacitance_from_terminal_capacitance(900e-12, 14331)
        assert_approx_equal(result, 900e-12 / 14331)

        # Test with string inputs
        result = pixel_capacitance_from_terminal_capacitance("900pF", "14331")
        assert_approx_equal(result, 900e-12 / 14331)

class TestAnnotatedTypes(unittest.TestCase):
    def test_type_annotations_exist(self):
        """Test that the type annotations are available."""
        self.assertIsNotNone(NormalizableArgument)
        self.assertIsNotNone(CapacitanceFarad)

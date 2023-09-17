#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from UliEngineering.Electronics.MOSFET import *
from UliEngineering.Exceptions import OperationImpossibleException
from UliEngineering.EngineerIO import auto_format
import unittest

class TestLEDSeriesResistors(unittest.TestCase):
    def test_mosfet_gate_charge_losses(self):
        # Example verified at http://www.elektronik-kompendium.de/sites/bau/1109111.htm
        # Also verified at https://www.digikey.com/en/resources/conversion-calculators/conversion-calculator-led-series-resistor
        assert_approx_equal(mosfet_gate_charge_losses(39.0e-9, 10, 300e3), 0.117)
        assert_approx_equal(mosfet_gate_charge_losses("39nC", "10V", "300 kHz"), 0.117)
        
    def test_mosfet_gate_capacitance_from_gate_charge(self):
        assert_approx_equal(mosfet_gate_capacitance_from_gate_charge(39.0e-9, 10), 3.9e-9)
        assert_approx_equal(mosfet_gate_capacitance_from_gate_charge("39nC", "10V"), 3.9e-9)

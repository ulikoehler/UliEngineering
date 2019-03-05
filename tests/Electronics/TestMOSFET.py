#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose
from nose.tools import raises
from UliEngineering.Electronics.MOSFET import *
from UliEngineering.Exceptions import OperationImpossibleException
from UliEngineering.EngineerIO import auto_format

class TestLEDSeriesResistors(object):
    def test_mosfet_gate_charge_losses(self):
        # Example verified at http://www.elektronik-kompendium.de/sites/bau/1109111.htm
        # Also verified at https://www.digikey.com/en/resources/conversion-calculators/conversion-calculator-led-series-resistor
        assert_approx_equal(mosfet_gate_charge_losses(39.0e-9, 10, 300e3), 0.117)
        assert_approx_equal(mosfet_gate_charge_losses("39nC", "10V", "300 kHz"), 0.117)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_allclose
from UliEngineering.Electronics.Hysteresis import hysteresis_threshold_ratios, hysteresis_threshold_voltages, hysteresis_threshold_factors, hysteresis_threshold_factors_opendrain, hysteresis_threshold_voltages_opendrain, hysteresis_threshold_ratios_opendrain, hysteresis_resistor
from UliEngineering.Electronics.Diode import ResistanceOhm, VoltageV
import unittest

class TestHysteresis(unittest.TestCase):
    def test_hysteresis_thresholds(self):
        # 1e300: Near-infinite resistor should not affect ratio
        assert_allclose(hysteresis_threshold_ratios(1e3, 1e3, 1e300), (0.5, 0.5))
        assert_allclose(hysteresis_threshold_voltages(1e3, 1e3, 1e300, 5.0), (2.5, 2.5))
        assert_allclose(hysteresis_threshold_factors(1e3, 1e3, 1e300), (1.0, 1.0))
        # More realistic values
        assert_allclose(hysteresis_threshold_ratios(1e3, 1e3, 1e3), (0.3333333333, 0.6666666666))
        assert_allclose(hysteresis_threshold_voltages(1e3, 1e3, 1e3, 5.0), (0.3333333333*5., 0.6666666666*5.))
        assert_allclose(hysteresis_threshold_factors(1e3, 1e3, 1e3), (0.3333333333/.5, 0.6666666666/.5))

    def test_hysteresis_opendrain(self):
        # 1e300: Near-infinite resistor should not affect ratio
        assert_allclose(hysteresis_threshold_factors_opendrain(1e3, 1e3, 1e300), (1.0, 1.0))
        assert_allclose(hysteresis_threshold_voltages_opendrain(1e3, 1e3, 1e300, 5.0), (2.5, 2.5))
        assert_allclose(hysteresis_threshold_ratios_opendrain(1e3, 1e3, 1e300), (0.5, 0.5))
        # More realistic values
        assert_allclose(hysteresis_threshold_factors_opendrain(1e3, 1e3, 1e3), (0.3333333333/.5, 1.))
        assert_allclose(hysteresis_threshold_ratios_opendrain(1e3, 1e3, 1e3), (0.3333333333, 0.5))
        assert_allclose(hysteresis_threshold_voltages_opendrain(1e3, 1e3, 1e3, 5.0), (0.3333333333*5., 0.5*5.))

    def test_hysteresis_resistor(self):
        assert_allclose(hysteresis_resistor(1e3, 1e3, 0.1), 4500)

class TestAnnotatedTypes(unittest.TestCase):
    def test_type_annotations_exist(self):
        """Test that the type annotations are available."""
        self.assertIsNotNone(ResistanceOhm)
        self.assertIsNotNone(VoltageV)

    def test_hysteresis_functions_various_units(self):
        """Test hysteresis functions with various unit inputs."""
        # Test hysteresis_threshold_ratios with different resistance units
        ratios1 = hysteresis_threshold_ratios("1 kΩ", "1 kΩ", "1 kΩ")
        ratios2 = hysteresis_threshold_ratios("1000 ohm", "1000 ohm", "1000 ohm")
        assert_allclose(ratios1, ratios2)

        # Test hysteresis_threshold_voltages with different units
        volts1 = hysteresis_threshold_voltages("1 kΩ", "1 kΩ", "1 kΩ", "5 V")
        volts2 = hysteresis_threshold_voltages("1000 ohm", "1000 ohm", "1000 ohm", "5000 mV")
        assert_allclose(volts1, volts2)

        # Test hysteresis_resistor with different units
        r1 = hysteresis_resistor("1 kΩ", "1 kΩ", 0.1)
        r2 = hysteresis_resistor("1000 ohm", "1000 ohm", 0.1)
        assert_allclose(r1, r2)

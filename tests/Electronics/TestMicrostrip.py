#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_allclose
from UliEngineering.Electronics.Microstrip import *
from UliEngineering.EngineerIO import *
import unittest
import numpy as np

class TestMicrostrip(unittest.TestCase):
    def test_microstrip_width_roundtrip(self):
        """
        Verify that microstrip_width() is the inverse of microstrip_impedance()
        for a wide range of parameters.
        """
        # Test ranges
        impedances = np.linspace(20, 150, 20)  # 20 to 150 Ohms
        heights = ["100 um", "200 um", "500 um", "1.6 mm"]
        permittivities = [2.2, 4.0, 4.8, 10.0] # PTFE, SiO2, FR4, Alumina
        thicknesses = ["18 um", "35 um", "70 um"]

        for Z0 in impedances:
            for h in heights:
                for er in permittivities:
                    for t in thicknesses:
                        # Calculate width for given impedance
                        w = microstrip_width(Z0, h=h, t=t, e_r=er)
                        
                        # Calculate impedance back from width
                        Z0_calc = microstrip_impedance(w, h=h, t=t, e_r=er)
                        
                        # Check if they match
                        # We use a relatively loose tolerance because the formulas 
                        # might have some numerical instability or approximation errors
                        # but they should be reasonably close.
                        # The solver tolerance is 1e-9, so we expect good agreement.
                        assert_allclose(Z0_calc, Z0, rtol=0.01, 
                                        err_msg=f"Failed for Z0={Z0}, h={h}, er={er}, t={t}, w={w}")

    def test_microstrip_width_specific_values(self):
        """
        Test specific known values or sanity checks
        """
        # 50 Ohm on standard FR4 (1.6mm height, 35um copper)
        # Width should be around 3mm (roughly 2*h for FR4)
        w_50 = microstrip_width(50.0, h="1.6 mm", t="35 um", e_r=4.8)
        self.assertTrue(2.5e-3 < w_50 < 3.5e-3, f"Width for 50 Ohm FR4 seems off: {w_50}")

        # 100 Ohm trace should be thinner than 50 Ohm trace
        w_100 = microstrip_width(100.0, h="1.6 mm", t="35 um", e_r=4.8)
        self.assertLess(w_100, w_50)

    def test_microstrip_width_input_formats(self):
        """
        Test that string inputs work correctly (handled by decorators)
        """
        w1 = microstrip_width(50, h="1.6mm", t="35um", e_r=4.8)
        w2 = microstrip_width("50 Ohm", h=1.6e-3, t=35e-6, e_r=4.8)
        assert_allclose(w1, w2)


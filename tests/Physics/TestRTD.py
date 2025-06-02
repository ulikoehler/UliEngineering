#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_less
from UliEngineering.Physics.RTD import *
from UliEngineering.Exceptions import *
import functools
import numpy as np
import unittest

class TestRTD(unittest.TestCase):
    def test_ptx_resistance(self):
        # Reference values from http://pavitronic.dk/eng/pt1000val.html
        # Other reference values (4 significant digits): http://grundpraktikum.physik.uni-saarland.de/scripts/Platin_Widerstandsthermometer.pdf
        ptassert = functools.partial(assert_approx_equal, significant=6)
        ptassert4 = functools.partial(assert_approx_equal, significant=5)
        # Test PT1000
        ptassert4(pt1000_resistance("-200 °C"), 185.2)
        ptassert4(pt1000_resistance("-100 °C"), 602.55)
        ptassert(pt1000_resistance("-50 °C"), 803.063)
        ptassert(pt1000_resistance("-20 °C"), 921.599)
        ptassert(pt1000_resistance("0 °C"), 1000.000)
        ptassert(pt1000_resistance("10 °C"), 1039.025)
        ptassert(pt1000_resistance("20 °C"), 1077.935)
        ptassert(pt1000_resistance("30 °C"), 1116.729)
        ptassert(pt1000_resistance("40 °C"), 1155.408)
        ptassert(pt1000_resistance("50 °C"), 1193.971)
        ptassert(pt1000_resistance("60 °C"), 1232.419)
        ptassert(pt1000_resistance("70 °C"), 1270.751)
        ptassert(pt1000_resistance("80 °C"), 1308.968)
        ptassert(pt1000_resistance("90 °C"), 1347.069)
        ptassert(pt1000_resistance("100 °C"), 1385.055)
        ptassert(pt1000_resistance("150 °C"), 1573.251)
        ptassert(pt1000_resistance("200 °C"), 1758.560)
        ptassert(pt1000_resistance("300 °C"), 2120.515)
        ptassert(pt1000_resistance("400 °C"), 2470.920)
        ptassert(pt1000_resistance("500 °C"), 2809.775)
        ptassert(pt1000_resistance("600 °C"), 3137.080)
        # Test PT100
        ptassert(pt100_resistance("-50 °C"), 80.3063)
        ptassert(pt100_resistance("-20 °C"), 92.1599)
        ptassert(pt100_resistance("0 °C"), 100.0000)
        ptassert(pt100_resistance("10 °C"), 103.9025)
        ptassert(pt100_resistance("20 °C"), 107.7935)
        ptassert(pt100_resistance("30 °C"), 111.6729)
        ptassert(pt100_resistance("40 °C"), 115.5408)
        ptassert(pt100_resistance("50 °C"), 119.3971)
        ptassert(pt100_resistance("60 °C"), 123.2419)
        ptassert(pt100_resistance("70 °C"), 127.0751)
        ptassert(pt100_resistance("80 °C"), 130.8968)
        ptassert(pt100_resistance("90 °C"), 134.7069)
        ptassert(pt100_resistance("100 °C"), 138.5055)
        ptassert(pt100_resistance("150 °C"), 157.3251)
        ptassert(pt100_resistance("200 °C"), 175.8560)
        ptassert(pt100_resistance("300 °C"), 212.0515)
        ptassert(pt100_resistance("400 °C"), 247.0920)
        ptassert(pt100_resistance("500 °C"), 280.9775)
        ptassert(pt100_resistance("600 °C"), 313.7080)

    def test_ptx_temperature(self):
        # Reference values from http://pavitronic.dk/eng/pt1000val.html
        tempassert = functools.partial(assert_approx_equal, significant=5)
        # Test PT1000
        tempassert(pt1000_temperature("185.2 Ω"), -200.0)
        tempassert(pt1000_temperature("602.55 Ω"), -100.0)
        tempassert(pt1000_temperature("803.063 Ω"), -50.0)
        tempassert(pt1000_temperature("921.599 Ω"), -20.0)
        tempassert(pt1000_temperature("1000.000 Ω"), 0.0)
        tempassert(pt1000_temperature("1039.025 Ω"), 10.0)
        tempassert(pt1000_temperature("1077.935 Ω"), 20.0)
        tempassert(pt1000_temperature("1116.729 Ω"), 30.0)
        tempassert(pt1000_temperature("1155.408 Ω"), 40.0)
        tempassert(pt1000_temperature("1193.971 Ω"), 50.0)
        tempassert(pt1000_temperature("1232.419 Ω"), 60.0)
        tempassert(pt1000_temperature("1270.751 Ω"), 70.0)
        tempassert(pt1000_temperature("1308.968 Ω"), 80.0)
        tempassert(pt1000_temperature("1347.069 Ω"), 90.0)
        tempassert(pt1000_temperature("1385.055 Ω"), 100.0)
        tempassert(pt1000_temperature("1573.251 Ω"), 150.0)
        tempassert(pt1000_temperature("1758.560 Ω"), 200.0)
        tempassert(pt1000_temperature("2120.515 Ω"), 300.0)
        tempassert(pt1000_temperature("2470.920 Ω"), 400.0)
        tempassert(pt1000_temperature("2809.775 Ω"), 500.0)
        tempassert(pt1000_temperature("3137.080 Ω"), 600.0)
        # Test PT100
        tempassert(pt100_temperature("18.52 Ω"), -200.0)
        tempassert(pt100_temperature("60.255 Ω"), -100.0)
        tempassert(pt100_temperature("80.3063 Ω"), -50.0)
        tempassert(pt100_temperature("92.1599 Ω"), -20.0)
        tempassert(pt100_temperature("100.0000 Ω"), 0.0)
        tempassert(pt100_temperature("103.9025 Ω"), 10.0)
        tempassert(pt100_temperature("107.7935 Ω"), 20.0)
        tempassert(pt100_temperature("111.6729 Ω"), 30.0)
        tempassert(pt100_temperature("115.5408 Ω"), 40.0)
        tempassert(pt100_temperature("119.3971 Ω"), 50.0)
        tempassert(pt100_temperature("123.2419 Ω"), 60.0)
        tempassert(pt100_temperature("127.0751 Ω"), 70.0)
        tempassert(pt100_temperature("130.8968 Ω"), 80.0)
        tempassert(pt100_temperature("134.7069 Ω"), 90.0)
        tempassert(pt100_temperature("138.5055 Ω"), 100.0)
        tempassert(pt100_temperature("157.3251 Ω"), 150.0)
        tempassert(pt100_temperature("175.8560 Ω"), 200.0)
        tempassert(pt100_temperature("212.0515 Ω"), 300.0)
        tempassert(pt100_temperature("247.0920 Ω"), 400.0)
        tempassert(pt100_temperature("280.9775 Ω"), 500.0)
        tempassert(pt100_temperature("313.7080 Ω"), 600.0)

    def test_temperature_numpy_array(self):
        tempassert = functools.partial(assert_allclose, rtol=1e-3)
        data = np.asarray([602.55, 1000.000, 1385.055])
        expected = np.asarray([-100.0, 0.0, 100.0])
        tempassert(pt1000_temperature(data), expected)
        
    def test_resistane_numpy_array(self):
        tempassert = functools.partial(assert_allclose, rtol=1e-3)
        data = np.asarray([-100.0, 0.0, 100.0])
        expected = np.asarray([602.55, 1000.000, 1385.055])
        tempassert(pt1000_resistance(data), expected)

class TestRTDPolynomialComputation(unittest.TestCase):
    def test_optimize_check_polynomial(self):
        # Basically the code from techoverflow.net without matplotlib
        temp = np.linspace(-200.0, 0.0, 1000)
        x, y, pkdev = check_correction_polynomial_quality(1000.0, temp, poly=noCorrection)
        self.assertEqual(x.shape, temp.shape)
        self.assertEqual(y.shape, temp.shape)
        # Peak deviation w/o correction should be around 2.4 .. 2.5 °C
        assert_allclose(pkdev, 2.45, rtol=0.05)
        # Correct
        mypoly = compute_correction_polynomial(1000.0, n=5000)
        self.assertTrue(isinstance(mypoly, np.poly1d))
        x, y, pkdev = check_correction_polynomial_quality(1000.0, temp, poly=mypoly)
        # pkdev should not exceed 0.1 m°C
        assert_array_less([pkdev], [1e-4])
        assert_array_less(y, np.full(y.shape, 1e-4))

    def test_nonstandard_r0(self):
        # Check if ptx_temperature() runs correctly with poly=None with nonstandard r0
        ptx_temperature(1234.0, 1155.1) 

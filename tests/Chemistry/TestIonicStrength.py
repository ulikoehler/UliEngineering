#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from numpy.testing import assert_approx_equal
from UliEngineering.Chemistry.IonicStrength import (
    ionic_strength,
    ionic_strength_from_pairs,
    ionic_strength_monovalent,
)


class TestIonicStrength(unittest.TestCase):
    def test_nacl(self):
        # NaCl 0.1 M: I = 0.5*(0.1*1 + 0.1*1) = 0.1
        I = ionic_strength([0.1, 0.1], [1, -1])
        assert_approx_equal(I, 0.1, significant=5)

    def test_cacl2(self):
        # CaCl2 0.1 M: Ca2+ 0.1M, Cl- 0.2M
        # I = 0.5*(0.1*4 + 0.2*1) = 0.5*(0.4+0.2) = 0.3
        I = ionic_strength([0.1, 0.2], [2, -1])
        assert_approx_equal(I, 0.3, significant=5)

    def test_mgso4(self):
        # MgSO4 0.1 M: Mg2+ 0.1M, SO4^2- 0.1M
        # I = 0.5*(0.1*4 + 0.1*4) = 0.4
        I = ionic_strength([0.1, 0.1], [2, -2])
        assert_approx_equal(I, 0.4, significant=5)

    def test_zero(self):
        I = ionic_strength([0, 0], [1, -1])
        assert_approx_equal(I, 0.0, significant=5)

    def test_mixed_electrolyte(self):
        # 0.1 M NaCl + 0.05 M CaCl2:
        # Na+ 0.1, Ca2+ 0.05, Cl- 0.2
        # I = 0.5*(0.1*1 + 0.05*4 + 0.2*1) = 0.5*(0.1+0.2+0.2) = 0.25
        I = ionic_strength([0.1, 0.05, 0.2], [1, 2, -1])
        assert_approx_equal(I, 0.25, significant=5)


class TestIonicStrengthFromPairs(unittest.TestCase):
    def test_nacl(self):
        I = ionic_strength_from_pairs([(0.1, 1), (0.1, -1)])
        assert_approx_equal(I, 0.1, significant=5)

    def test_cacl2(self):
        I = ionic_strength_from_pairs([(0.1, 2), (0.2, -1)])
        assert_approx_equal(I, 0.3, significant=5)


class TestIonicStrengthMonovalent(unittest.TestCase):
    def test_basic(self):
        assert_approx_equal(ionic_strength_monovalent(0.1), 0.1, significant=5)

    def test_high(self):
        assert_approx_equal(ionic_strength_monovalent(1.0), 1.0, significant=5)


if __name__ == '__main__':
    unittest.main()

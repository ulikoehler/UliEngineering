#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from UliEngineering.Chemistry.PREN import (
    pren,
    pren_w,
    COMMON_STEEL_COMPOSITIONS,
)


class TestPREN(unittest.TestCase):
    def test_pren_basic(self):
        """Test basic PREN calculation"""
        result = pren(Cr=18.0, Mo=2.1, N=0.05)
        expected = 18.0 + 3.3 * 2.1 + 16.0 * 0.05
        self.assertAlmostEqual(result, expected, places=6)

    def test_pren_zero_molybdenum(self):
        """Test PREN with zero molybdenum."""
        result = pren(Cr=18.0, Mo=0.0, N=0.05)
        expected = 18.0 + 16.0 * 0.05
        self.assertAlmostEqual(result, expected, places=6)

    def test_pren_zero_nitrogen(self):
        """Test PREN with zero nitrogen."""
        result = pren(Cr=18.0, Mo=2.1, N=0.0)
        expected = 18.0 + 3.3 * 2.1
        self.assertAlmostEqual(result, expected, places=6)

    def test_pren_w_basic(self):
        """Test PREN with tungsten"""
        result = pren_w(Cr=15.5, Mo=16.0, N=0.0, W=3.75)
        expected = 15.5 + 3.3 * (16.0 + 0.5 * 3.75) + 16.0 * 0.0
        self.assertAlmostEqual(result, expected, places=6)

    def test_pren_w_zero_tungsten(self):
        """Test PREN with zero tungsten"""
        result = pren_w(Cr=18.0, Mo=2.1, N=0.05, W=0.0)
        expected = 18.0 + 3.3 * 2.1 + 16.0 * 0.05
        self.assertAlmostEqual(result, expected, places=6)

    def test_common_steel_compositions(self):
        """Test that common steel compositions dictionary exists"""
        self.assertIsInstance(COMMON_STEEL_COMPOSITIONS, dict)
        self.assertIn("304", COMMON_STEEL_COMPOSITIONS)
        self.assertIn("316", COMMON_STEEL_COMPOSITIONS)

    def test_pren_from_composition(self):
        """Test PREN calculation using common steel composition"""
        comp = COMMON_STEEL_COMPOSITIONS["316"]
        result = pren(Cr=comp["Cr"], Mo=comp["Mo"], N=comp["N"])
        self.assertIsInstance(result, float)
        self.assertGreater(result, 20)  # 316 should have PREN > 20

    def test_pren_w_from_composition(self):
        """Test PREN with tungsten using common steel composition."""
        comp = COMMON_STEEL_COMPOSITIONS["AlloyC276"]
        result = pren_w(Cr=comp["Cr"], Mo=comp["Mo"], N=comp["N"], W=comp["W"])
        self.assertIsInstance(result, float)
        self.assertGreater(result, 50)  # Alloy C276 should have high PREN


if __name__ == "__main__":
    unittest.main()

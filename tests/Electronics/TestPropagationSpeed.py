#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from UliEngineering.Electronics.PropagationSpeed import propagation_speed, propagation_delay
import scipy.constants
import unittest


class TestPropagationSpeed(unittest.TestCase):
    def test_propagation_speed_vacuum(self):
        self.assertEqual(propagation_speed(1.0, 1.0), scipy.constants.c)

    def test_propagation_speed_dielectric(self):
        # Relative permittivity of 4 -> speed should be c/2 (mu_r=1)
        self.assertEqual(propagation_speed(4.0, 1.0), scipy.constants.c / 2.0)

    def test_propagation_delay_vacuum(self):
        # 1 m in vacuum should be 1 / c seconds
        self.assertAlmostEqual(propagation_delay(1.0, e_r=1.0, mu_r=1.0), 1.0 / scipy.constants.c, places=12)

    def test_propagation_delay_dielectric(self):
        # 1 m in dielectric with e_r=4 should be 2 / c seconds
        self.assertAlmostEqual(propagation_delay(1.0, e_r=4.0, mu_r=1.0), 2.0 / scipy.constants.c, places=12)

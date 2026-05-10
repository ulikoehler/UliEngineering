#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprehensive unit tests for UliEngineering.Physics.Viscosity."""

import unittest
import numpy as np
from numpy.testing import assert_approx_equal

from UliEngineering.Physics.Viscosity import (
    AndradeConstants,
    VFTConstants,
    SutherlandConstants,
    SwindellsConstants,
    KestinConstants,
    BinghamConstants,
    LiquidViscosityData,
    GasViscosityData,
    CommonLiquids,
    CommonGases,
    andrade_viscosity,
    vft_viscosity,
    sutherland_gas_viscosity,
    swindells_viscosity,
    kestin_viscosity,
    bingham_stress,
    poiseuille_flow_rate,
    kinematic_viscosity,
    stokes_drag,
    reynolds_number,
    arrhenius_mixing_viscosity,
    normalize_dynamic_viscosity, DynamicViscosityPas,
    normalize_density, DensityKgM3,
    normalize_length, LengthMeter,
    normalize_pressure, PressurePascal,
    normalize_velocity, VelocityMS,
    normalize_shear_rate, ShearRate,
)


class TestAndradeViscosity(unittest.TestCase):
    def test_scalar_positive(self):
        c = AndradeConstants(name="Test", A=1.0, B=1000.0)
        eta = andrade_viscosity(500.0, c)
        expected = 1.0 * np.exp(1000.0 / 500.0)
        assert_approx_equal(eta, expected, significant=8)

    def test_array_positive(self):
        c = AndradeConstants(name="Test", A=1.0, B=1000.0)
        T = np.array([500.0, 250.0])
        eta = andrade_viscosity(T, c)
        expected = np.array([np.exp(2.0), np.exp(4.0)])
        np.testing.assert_allclose(eta, expected, rtol=1e-10)

    def test_zero_temperature_raises(self):
        c = AndradeConstants(name="Test", A=1.0, B=1000.0)
        with self.assertRaises(ValueError):
            andrade_viscosity(0.0, c)

    def test_negative_temperature_raises(self):
        c = AndradeConstants(name="Test", A=1.0, B=1000.0)
        with self.assertRaises(ValueError):
            andrade_viscosity(-100.0, c)

    def test_array_with_negative_raises(self):
        c = AndradeConstants(name="Test", A=1.0, B=1000.0)
        with self.assertRaises(ValueError):
            andrade_viscosity(np.array([500.0, -1.0]), c)

    def test_default_constants(self):
        eta = andrade_viscosity(293.15)
        self.assertGreater(eta, 0.0)
        self.assertIsInstance(eta, (float, np.floating))

    def test_very_small_temperature(self):
        c = AndradeConstants(name="Test", A=1.0, B=1.0)
        eta = andrade_viscosity(1e-6, c)
        expected = np.exp(1e6)
        self.assertAlmostEqual(eta, expected, delta=expected * 1e-6)

    def test_large_temperature(self):
        c = AndradeConstants(name="Test", A=1.0, B=1.0)
        eta = andrade_viscosity(1e6, c)
        expected = np.exp(1e-6)
        assert_approx_equal(eta, expected, significant=8)


class TestVFTViscosity(unittest.TestCase):
    def test_scalar_above_T0(self):
        c = VFTConstants(name="Test", A=1.0, B=500.0, T0=100.0)
        eta = vft_viscosity(200.0, c)
        expected = np.exp(500.0 / 100.0)
        assert_approx_equal(eta, expected, significant=8)

    def test_array_above_T0(self):
        c = VFTConstants(name="Test", A=1.0, B=500.0, T0=100.0)
        T = np.array([200.0, 300.0])
        eta = vft_viscosity(T, c)
        expected = np.array([np.exp(5.0), np.exp(2.5)])
        np.testing.assert_allclose(eta, expected, rtol=1e-10)

    def test_exactly_T0_raises(self):
        c = VFTConstants(name="Test", A=1.0, B=500.0, T0=100.0)
        with self.assertRaises(ValueError):
            vft_viscosity(100.0, c)

    def test_below_T0_raises(self):
        c = VFTConstants(name="Test", A=1.0, B=500.0, T0=100.0)
        with self.assertRaises(ValueError):
            vft_viscosity(50.0, c)

    def test_default_constants(self):
        eta = vft_viscosity(300.0)
        self.assertGreater(eta, 0.0)

    def test_array_with_one_invalid_raises(self):
        c = VFTConstants(name="Test", A=1.0, B=500.0, T0=100.0)
        with self.assertRaises(ValueError):
            vft_viscosity(np.array([200.0, 50.0]), c)


class TestSutherlandGasViscosity(unittest.TestCase):
    def test_scalar_at_T0(self):
        c = SutherlandConstants(name="Test", mu0=1.0e-5, T0=273.15, C=111.0)
        mu = sutherland_gas_viscosity(273.15, c)
        assert_approx_equal(mu, 1.0e-5, significant=8)

    def test_scalar_above_T0(self):
        c = SutherlandConstants(name="Test", mu0=1.0e-5, T0=273.15, C=111.0)
        mu = sutherland_gas_viscosity(373.15, c)
        expected = 1.0e-5 * (273.15 + 111.0) / (373.15 + 111.0) * (373.15 / 273.15) ** 1.5
        assert_approx_equal(mu, expected, significant=8)

    def test_array(self):
        c = SutherlandConstants(name="Test", mu0=1.0e-5, T0=273.15, C=111.0)
        T = np.array([273.15, 373.15])
        mu = sutherland_gas_viscosity(T, c)
        expected = np.array([
            1.0e-5,
            1.0e-5 * (273.15 + 111.0) / (373.15 + 111.0) * (373.15 / 273.15) ** 1.5
        ])
        np.testing.assert_allclose(mu, expected, rtol=1e-10)

    def test_zero_temperature_raises(self):
        c = SutherlandConstants(name="Test", mu0=1.0e-5, T0=273.15, C=111.0)
        with self.assertRaises(ValueError):
            sutherland_gas_viscosity(0.0, c)

    def test_negative_temperature_raises(self):
        c = SutherlandConstants(name="Test", mu0=1.0e-5, T0=273.15, C=111.0)
        with self.assertRaises(ValueError):
            sutherland_gas_viscosity(-10.0, c)

    def test_default_constants(self):
        mu = sutherland_gas_viscosity(300.0)
        self.assertGreater(mu, 0.0)

    def test_air_known_value(self):
        mu = sutherland_gas_viscosity(300.0, CommonGases.Air.sutherland)
        self.assertGreater(mu, 1.5e-5)
        self.assertLess(mu, 2.0e-5)


class TestSwindellsViscosity(unittest.TestCase):
    def test_scalar_at_T_ref(self):
        c = SwindellsConstants(name="Test", eta_ref=1.0e-3, T_ref=293.15, a=1.5, b=-150.0)
        eta = swindells_viscosity(293.15, c)
        assert_approx_equal(eta, 1.0e-3, significant=8)

    def test_scalar_above_T_ref(self):
        c = SwindellsConstants(name="Test", eta_ref=1.0e-3, T_ref=293.15, a=1.5, b=-150.0)
        eta = swindells_viscosity(373.15, c)
        exponent = -1.5 * (373.15 - 293.15) / (373.15 - 150.0)
        expected = 1.0e-3 * 10.0 ** exponent
        assert_approx_equal(eta, expected, significant=8)

    def test_array(self):
        c = SwindellsConstants(name="Test", eta_ref=1.0e-3, T_ref=293.15, a=1.5, b=-150.0)
        T = np.array([293.15, 373.15])
        eta = swindells_viscosity(T, c)
        exponents = -1.5 * (T - 293.15) / (T - 150.0)
        expected = 1.0e-3 * np.power(10.0, exponents)
        np.testing.assert_allclose(eta, expected, rtol=1e-10)

    def test_default_constants(self):
        eta = swindells_viscosity(293.15)
        self.assertGreater(eta, 0.0)

    def test_water_at_273(self):
        eta = swindells_viscosity(273.15, CommonLiquids.Water.swindells)
        self.assertGreater(eta, 1.0e-3)
        self.assertLess(eta, 2.0e-3)


class TestKestinViscosity(unittest.TestCase):
    def test_scalar_above_C(self):
        c = KestinConstants(name="Test", A=1.0, B=500.0, C=100.0)
        eta = kestin_viscosity(200.0, c)
        expected = np.exp(500.0 / 100.0)
        assert_approx_equal(eta, expected, significant=8)

    def test_array_above_C(self):
        c = KestinConstants(name="Test", A=1.0, B=500.0, C=100.0)
        T = np.array([200.0, 300.0])
        eta = kestin_viscosity(T, c)
        expected = np.array([np.exp(5.0), np.exp(2.5)])
        np.testing.assert_allclose(eta, expected, rtol=1e-10)

    def test_exactly_C_raises(self):
        c = KestinConstants(name="Test", A=1.0, B=500.0, C=100.0)
        with self.assertRaises(ValueError):
            kestin_viscosity(100.0, c)

    def test_below_C_raises(self):
        c = KestinConstants(name="Test", A=1.0, B=500.0, C=100.0)
        with self.assertRaises(ValueError):
            kestin_viscosity(50.0, c)

    def test_default_constants(self):
        eta = kestin_viscosity(300.0)
        self.assertGreater(eta, 0.0)

    def test_array_with_one_invalid_raises(self):
        c = KestinConstants(name="Test", A=1.0, B=500.0, C=100.0)
        with self.assertRaises(ValueError):
            kestin_viscosity(np.array([200.0, 50.0]), c)


class TestBinghamStress(unittest.TestCase):
    def test_scalar_zero_shear(self):
        c = BinghamConstants(name="Test", tau0=10.0, mu_p=0.5)
        tau = bingham_stress(0.0, c)
        assert_approx_equal(tau, 10.0, significant=8)

    def test_scalar_positive_shear(self):
        c = BinghamConstants(name="Test", tau0=10.0, mu_p=0.5)
        tau = bingham_stress(4.0, c)
        assert_approx_equal(tau, 12.0, significant=8)

    def test_array(self):
        c = BinghamConstants(name="Test", tau0=10.0, mu_p=0.5)
        gamma = np.array([0.0, 4.0, 10.0])
        tau = bingham_stress(gamma, c)
        expected = np.array([10.0, 12.0, 15.0])
        np.testing.assert_allclose(tau, expected, rtol=1e-10)

    def test_negative_shear_raises(self):
        c = BinghamConstants(name="Test", tau0=10.0, mu_p=0.5)
        with self.assertRaises(ValueError):
            bingham_stress(-1.0, c)

    def test_default_constants(self):
        tau = bingham_stress(0.0)
        assert_approx_equal(tau, 10.0, significant=8)

    def test_default_with_shear(self):
        tau = bingham_stress(10.0)
        assert_approx_equal(tau, 11.0, significant=8)

    def test_array_with_negative_raises(self):
        c = BinghamConstants(name="Test", tau0=10.0, mu_p=0.5)
        with self.assertRaises(ValueError):
            bingham_stress(np.array([0.0, -1.0]), c)


class TestPoiseuilleFlowRate(unittest.TestCase):
    def test_basic_scalar(self):
        Q = poiseuille_flow_rate(0.01, 1000.0, 1.0, 0.001)
        expected = np.pi * (0.01 ** 4) * 1000.0 / (8.0 * 0.001 * 1.0)
        assert_approx_equal(Q, expected, significant=8)

    def test_array(self):
        r = np.array([0.01, 0.02])
        Q = poiseuille_flow_rate(r, 1000.0, 1.0, 0.001)
        expected = np.pi * r ** 4 * 1000.0 / (8.0 * 0.001 * 1.0)
        np.testing.assert_allclose(Q, expected, rtol=1e-10)

    def test_zero_radius_raises(self):
        with self.assertRaises(ValueError):
            poiseuille_flow_rate(0.0, 1000.0, 1.0, 0.001)

    def test_negative_radius_raises(self):
        with self.assertRaises(ValueError):
            poiseuille_flow_rate(-0.01, 1000.0, 1.0, 0.001)

    def test_zero_length_raises(self):
        with self.assertRaises(ValueError):
            poiseuille_flow_rate(0.01, 1000.0, 0.0, 0.001)

    def test_negative_length_raises(self):
        with self.assertRaises(ValueError):
            poiseuille_flow_rate(0.01, 1000.0, -1.0, 0.001)

    def test_zero_viscosity_raises(self):
        with self.assertRaises(ValueError):
            poiseuille_flow_rate(0.01, 1000.0, 1.0, 0.0)

    def test_negative_viscosity_raises(self):
        with self.assertRaises(ValueError):
            poiseuille_flow_rate(0.01, 1000.0, 1.0, -0.001)

    def test_array_mixed_invalid_radius_raises(self):
        with self.assertRaises(ValueError):
            poiseuille_flow_rate(np.array([0.01, 0.0]), 1000.0, 1.0, 0.001)

    def test_large_radius(self):
        Q = poiseuille_flow_rate(1.0, 1.0, 1.0, 1.0)
        expected = np.pi / 8.0
        assert_approx_equal(Q, expected, significant=8)

    def test_very_small_radius(self):
        Q = poiseuille_flow_rate(1e-6, 1e6, 1.0, 1.0)
        expected = np.pi * 1e-24 * 1e6 / 8.0
        assert_approx_equal(Q, expected, significant=8)


class TestKinematicViscosity(unittest.TestCase):
    def test_scalar(self):
        nu = kinematic_viscosity(0.001, 1000.0)
        assert_approx_equal(nu, 1.0e-6, significant=8)

    def test_array(self):
        eta = np.array([0.001, 0.002])
        rho = np.array([1000.0, 500.0])
        nu = kinematic_viscosity(eta, rho)
        expected = np.array([1.0e-6, 4.0e-6])
        np.testing.assert_allclose(nu, expected, rtol=1e-10)

    def test_zero_density_raises(self):
        with self.assertRaises(ValueError):
            kinematic_viscosity(0.001, 0.0)

    def test_negative_density_raises(self):
        with self.assertRaises(ValueError):
            kinematic_viscosity(0.001, -500.0)

    def test_array_with_zero_density_raises(self):
        with self.assertRaises(ValueError):
            kinematic_viscosity(np.array([0.001, 0.002]), np.array([1000.0, 0.0]))

    def test_very_small_density(self):
        nu = kinematic_viscosity(0.001, 1e-6)
        expected = 1000.0
        assert_approx_equal(nu, expected, significant=8)

    def test_large_density(self):
        nu = kinematic_viscosity(0.001, 1e6)
        expected = 1.0e-9
        assert_approx_equal(nu, expected, significant=8)


class TestStokesDrag(unittest.TestCase):
    def test_scalar(self):
        F = stokes_drag(1e-6, 0.01, 0.001)
        expected = 6.0 * np.pi * 0.001 * 1e-6 * 0.01
        assert_approx_equal(F, expected, significant=8)

    def test_array(self):
        r = np.array([1e-6, 2e-6])
        v = np.array([0.01, 0.02])
        eta = np.array([0.001, 0.002])
        F = stokes_drag(r, v, eta)
        expected = 6.0 * np.pi * eta * r * v
        np.testing.assert_allclose(F, expected, rtol=1e-10)

    def test_zero_radius(self):
        F = stokes_drag(0.0, 1.0, 0.001)
        assert_approx_equal(F, 0.0, significant=8)

    def test_negative_radius_raises(self):
        with self.assertRaises(ValueError):
            stokes_drag(-1e-6, 0.01, 0.001)

    def test_negative_viscosity_raises(self):
        with self.assertRaises(ValueError):
            stokes_drag(1e-6, 0.01, -0.001)

    def test_zero_viscosity(self):
        F = stokes_drag(1e-6, 0.01, 0.0)
        assert_approx_equal(F, 0.0, significant=8)

    def test_negative_velocity(self):
        F = stokes_drag(1e-6, -0.01, 0.001)
        expected = -6.0 * np.pi * 0.001 * 1e-6 * 0.01
        assert_approx_equal(F, expected, significant=8)

    def test_large_values(self):
        F = stokes_drag(1.0, 10.0, 1.0)
        expected = 60.0 * np.pi
        assert_approx_equal(F, expected, significant=8)


class TestReynoldsNumber(unittest.TestCase):
    def test_scalar(self):
        Re = reynolds_number(1000.0, 1.0, 0.1, 0.001)
        expected = 1000.0 * 1.0 * 0.1 / 0.001
        assert_approx_equal(Re, expected, significant=8)

    def test_array(self):
        rho = np.array([1000.0, 500.0])
        v = np.array([1.0, 2.0])
        L = np.array([0.1, 0.2])
        eta = np.array([0.001, 0.002])
        Re = reynolds_number(rho, v, L, eta)
        expected = rho * v * L / eta
        np.testing.assert_allclose(Re, expected, rtol=1e-10)

    def test_zero_density_raises(self):
        with self.assertRaises(ValueError):
            reynolds_number(0.0, 1.0, 0.1, 0.001)

    def test_negative_density_raises(self):
        with self.assertRaises(ValueError):
            reynolds_number(-1000.0, 1.0, 0.1, 0.001)

    def test_zero_length_raises(self):
        with self.assertRaises(ValueError):
            reynolds_number(1000.0, 1.0, 0.0, 0.001)

    def test_negative_length_raises(self):
        with self.assertRaises(ValueError):
            reynolds_number(1000.0, 1.0, -0.1, 0.001)

    def test_zero_viscosity_raises(self):
        with self.assertRaises(ValueError):
            reynolds_number(1000.0, 1.0, 0.1, 0.0)

    def test_negative_viscosity_raises(self):
        with self.assertRaises(ValueError):
            reynolds_number(1000.0, 1.0, 0.1, -0.001)

    def test_laminar_flow(self):
        Re = reynolds_number(1000.0, 0.01, 0.01, 0.001)
        self.assertLess(Re, 2300.0)

    def test_turbulent_flow(self):
        Re = reynolds_number(1000.0, 10.0, 1.0, 0.001)
        self.assertGreater(Re, 4000.0)

    def test_zero_velocity(self):
        Re = reynolds_number(1000.0, 0.0, 0.1, 0.001)
        assert_approx_equal(Re, 0.0, significant=8)

    def test_negative_velocity(self):
        Re = reynolds_number(1000.0, -1.0, 0.1, 0.001)
        expected = -100000.0
        assert_approx_equal(Re, expected, significant=8)


class TestArrheniusMixingViscosity(unittest.TestCase):
    def test_equal_fractions(self):
        eta = arrhenius_mixing_viscosity([0.5, 0.5], [1.0e-3, 2.0e-3])
        expected = np.exp(0.5 * np.log(1.0e-3) + 0.5 * np.log(2.0e-3))
        assert_approx_equal(eta, expected, significant=8)

    def test_pure_component(self):
        eta = arrhenius_mixing_viscosity([1.0, 0.0], [1.0e-3, 2.0e-3])
        assert_approx_equal(eta, 1.0e-3, significant=8)

    def test_three_components(self):
        eta = arrhenius_mixing_viscosity(
            [0.2, 0.3, 0.5], [1.0e-3, 2.0e-3, 3.0e-3]
        )
        expected = np.exp(
            0.2 * np.log(1.0e-3) + 0.3 * np.log(2.0e-3) + 0.5 * np.log(3.0e-3)
        )
        assert_approx_equal(eta, expected, significant=8)

    def test_fractions_not_sum_to_one_raises(self):
        with self.assertRaises(ValueError):
            arrhenius_mixing_viscosity([0.5, 0.4], [1.0e-3, 2.0e-3])

    def test_negative_fraction_raises(self):
        with self.assertRaises(ValueError):
            arrhenius_mixing_viscosity([-0.1, 1.1], [1.0e-3, 2.0e-3])

    def test_zero_viscosity_raises(self):
        with self.assertRaises(ValueError):
            arrhenius_mixing_viscosity([0.5, 0.5], [0.0, 2.0e-3])

    def test_negative_viscosity_raises(self):
        with self.assertRaises(ValueError):
            arrhenius_mixing_viscosity([0.5, 0.5], [-1.0e-3, 2.0e-3])

    def test_empty_arrays_raises(self):
        with self.assertRaises(ValueError):
            arrhenius_mixing_viscosity([], [])

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            arrhenius_mixing_viscosity([0.5, 0.5], [1.0e-3])

    def test_unity_sum_with_tolerance(self):
        eta = arrhenius_mixing_viscosity([0.333333, 0.666667], [1.0e-3, 2.0e-3])
        expected = np.exp(
            0.333333 * np.log(1.0e-3) + 0.666667 * np.log(2.0e-3)
        )
        assert_approx_equal(eta, expected, significant=6)


class TestDataclassesAndConstants(unittest.TestCase):
    def test_andrade_constants_immutable(self):
        c = AndradeConstants(name="Test", A=1.0, B=100.0)
        with self.assertRaises(AttributeError):
            c.A = 2.0

    def test_vft_constants_immutable(self):
        c = VFTConstants(name="Test", A=1.0, B=100.0, T0=50.0)
        with self.assertRaises(AttributeError):
            c.T0 = 60.0

    def test_common_liquids_water(self):
        water = CommonLiquids.Water
        self.assertEqual(water.name, "Water")
        self.assertIsNotNone(water.andrade)
        self.assertIsNotNone(water.vft)
        self.assertIsNotNone(water.swindells)
        self.assertIsNotNone(water.kestin)
        self.assertIsNotNone(water.density_kg_m3)

    def test_common_liquids_ethanol(self):
        ethanol = CommonLiquids.Ethanol
        self.assertEqual(ethanol.name, "Ethanol")
        self.assertIsNotNone(ethanol.andrade)
        self.assertIsNotNone(ethanol.vft)

    def test_common_gases_air(self):
        air = CommonGases.Air
        self.assertEqual(air.name, "Air")
        self.assertIsNotNone(air.sutherland)
        self.assertIsNotNone(air.density_kg_m3)

    def test_common_gases_helium(self):
        helium = CommonGases.Helium
        self.assertEqual(helium.name, "Helium")
        self.assertIsNotNone(helium.sutherland)

    def test_all_common_liquids_have_name(self):
        for attr_name in dir(CommonLiquids):
            if not attr_name.startswith("_"):
                liquid = getattr(CommonLiquids, attr_name)
                self.assertIsInstance(liquid, LiquidViscosityData)
                self.assertTrue(liquid.name)

    def test_all_common_gases_have_name(self):
        for attr_name in dir(CommonGases):
            if not attr_name.startswith("_"):
                gas = getattr(CommonGases, attr_name)
                self.assertIsInstance(gas, GasViscosityData)
                self.assertTrue(gas.name)

    def test_liquid_viscosity_data_defaults(self):
        data = LiquidViscosityData(name="Empty")
        self.assertIsNone(data.andrade)
        self.assertIsNone(data.vft)
        self.assertIsNone(data.swindells)
        self.assertIsNone(data.kestin)
        self.assertIsNone(data.density_kg_m3)

    def test_gas_viscosity_data_defaults(self):
        data = GasViscosityData(name="Empty")
        self.assertIsNone(data.sutherland)
        self.assertIsNone(data.density_kg_m3)

    def test_bingham_defaults(self):
        c = BinghamConstants(name="Test", tau0=5.0, mu_p=0.2)
        self.assertEqual(c.tau0, 5.0)
        self.assertEqual(c.mu_p, 0.2)


class TestCrossConsistency(unittest.TestCase):
    def test_kinematic_from_common_liquids_water(self):
        water = CommonLiquids.Water
        eta = andrade_viscosity(293.15, water.andrade)
        nu = kinematic_viscosity(eta, water.density_kg_m3)
        self.assertGreater(nu, 0.0)

    def test_reynolds_from_common_gases_air(self):
        air = CommonGases.Air
        mu = sutherland_gas_viscosity(300.0, air.sutherland)
        Re = reynolds_number(air.density_kg_m3, 10.0, 0.1, mu)
        self.assertGreater(Re, 0.0)

    def test_poiseuille_with_water_viscosity(self):
        water = CommonLiquids.Water
        eta = andrade_viscosity(293.15, water.andrade)
        Q = poiseuille_flow_rate(0.01, 1000.0, 1.0, eta)
        self.assertGreater(Q, 0.0)

    def test_stokes_with_air_viscosity(self):
        air = CommonGases.Air
        mu = sutherland_gas_viscosity(300.0, air.sutherland)
        F = stokes_drag(1e-6, 0.01, mu)
        self.assertGreater(F, 0.0)

    def test_bingham_with_custom_constants(self):
        c = BinghamConstants(name="Drilling mud", tau0=20.0, mu_p=0.3)
        tau = bingham_stress(50.0, c)
        self.assertEqual(tau, 35.0)

    def test_arrhenius_mixing_symmetric(self):
        eta = arrhenius_mixing_viscosity([0.5, 0.5], [1.0e-3, 1.0e-3])
        assert_approx_equal(eta, 1.0e-3, significant=8)


class TestNumpyBroadcasting(unittest.TestCase):
    def test_andrade_broadcast(self):
        c = AndradeConstants(name="Test", A=1.0, B=1000.0)
        T = np.linspace(300, 400, 5)
        eta = andrade_viscosity(T, c)
        self.assertEqual(eta.shape, (5,))

    def test_sutherland_broadcast(self):
        c = SutherlandConstants(name="Test", mu0=1.0e-5, T0=273.15, C=111.0)
        T = np.linspace(250, 350, 10)
        mu = sutherland_gas_viscosity(T, c)
        self.assertEqual(mu.shape, (10,))

    def test_poiseuille_broadcast(self):
        r = np.array([0.01, 0.02, 0.03])
        Q = poiseuille_flow_rate(r, 1000.0, 1.0, 0.001)
        self.assertEqual(Q.shape, (3,))

    def test_reynolds_broadcast(self):
        rho = np.array([1000.0, 500.0, 2000.0])
        v = np.array([1.0, 2.0, 3.0])
        L = np.array([0.1, 0.2, 0.3])
        eta = np.array([0.001, 0.002, 0.003])
        Re = reynolds_number(rho, v, L, eta)
        # numpy broadcasting: (3,),(3,),(3,),(3,) -> (3,)
        self.assertEqual(Re.shape, (3,))
        self.assertEqual(len(Re), 3)


class TestEdgeCasesAndSpecialValues(unittest.TestCase):
    def test_andrade_very_large_B(self):
        c = AndradeConstants(name="Test", A=1.0, B=1e6)
        eta = andrade_viscosity(1.0, c)
        self.assertAlmostEqual(eta, np.exp(1e6), delta=eta * 1e-6)

    def test_andrade_very_small_B(self):
        c = AndradeConstants(name="Test", A=1.0, B=1e-6)
        eta = andrade_viscosity(1.0, c)
        assert_approx_equal(eta, np.exp(1e-6), significant=8)

    def test_vft_very_close_to_T0(self):
        c = VFTConstants(name="Test", A=1.0, B=1.0, T0=100.0)
        eta = vft_viscosity(100.0 + 1e-6, c)
        expected = np.exp(1.0 / 1e-6)
        self.assertAlmostEqual(eta, expected, delta=expected * 1e-6)

    def test_sutherland_at_very_high_T(self):
        c = SutherlandConstants(name="Test", mu0=1.0e-5, T0=273.15, C=111.0)
        mu = sutherland_gas_viscosity(10000.0, c)
        self.assertGreater(mu, 0.0)

    def test_poiseuille_very_small_radius(self):
        Q = poiseuille_flow_rate(1e-9, 1e6, 1.0, 1.0)
        expected = np.pi * 1e-36 * 1e6 / 8.0
        assert_approx_equal(Q, expected, significant=8)

    def test_stokes_zero_all(self):
        F = stokes_drag(0.0, 0.0, 0.0)
        assert_approx_equal(F, 0.0, significant=8)

    def test_reynolds_very_large(self):
        Re = reynolds_number(1e6, 1e3, 1.0, 1e-6)
        assert_approx_equal(Re, 1e15, significant=8)

    def test_reynolds_very_small(self):
        Re = reynolds_number(1.0, 1e-3, 1e-3, 1.0)
        assert_approx_equal(Re, 1e-6, significant=8)

    def test_arrhenius_single_component(self):
        eta = arrhenius_mixing_viscosity([1.0], [5.0e-3])
        assert_approx_equal(eta, 5.0e-3, significant=8)

    def test_arrhenius_many_components(self):
        n = 20
        fractions = np.ones(n) / n
        viscosities = np.linspace(1e-4, 1e-2, n)
        eta = arrhenius_mixing_viscosity(fractions, viscosities)
        expected = np.exp(np.mean(np.log(viscosities)))
        assert_approx_equal(eta, expected, significant=8)


class TestNormalizeFunctions(unittest.TestCase):
    def test_type_annotations_exist(self):
        """Test that the new type annotations are available."""
        self.assertIsNotNone(DynamicViscosityPas)
        self.assertIsNotNone(DensityKgM3)
        self.assertIsNotNone(LengthMeter)
        self.assertIsNotNone(PressurePascal)
        self.assertIsNotNone(VelocityMS)
        self.assertIsNotNone(ShearRate)

    def test_normalize_dynamic_viscosity_various_units(self):
        """Test normalize_dynamic_viscosity with various unit inputs"""
        test_cases = [
            ("1 Pa·s", 1.0),
            ("1 Pa s", 1.0),
            ("1 Pas", 1.0),
            ("1 mPa·s", 1e-3),
            ("1 cP", 1e-3),
            ("1 P", 0.1),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_dynamic_viscosity(input_val)
                assert_approx_equal(result, expected)

    def test_normalize_density_various_units(self):
        """Test normalize_density with various unit inputs"""
        test_cases = [
            ("1 kg/m³", 1.0),
            ("1 kg/m3", 1.0),
            ("1 kg/m^3", 1.0),
            ("1 g/cm³", 1000.0),
            ("1 g/cm3", 1000.0),
            ("1 g/L", 1.0),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_density(input_val)
                assert_approx_equal(result, expected)

    def test_normalize_length_various_units(self):
        """Test normalize_length with various unit inputs"""
        test_cases = [
            ("1 m", 1.0),
            ("1 mm", 1e-3),
            ("1 cm", 1e-2),
            ("1 km", 1e3),
            ("1 µm", 1e-6),
            ("1 nm", 1e-9),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_length(input_val)
                assert_approx_equal(result, expected)

    def test_normalize_pressure_various_units(self):
        """Test normalize_pressure with various unit inputs"""
        test_cases = [
            ("1 Pa", 1.0),
            ("1 kPa", 1e3),
            ("1 MPa", 1e6),
            ("1 bar", 1e5),
            ("1 mbar", 100),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_pressure(input_val)
                assert_approx_equal(result, expected)

    def test_normalize_velocity_various_units(self):
        """Test normalize_velocity with various unit inputs."""
        test_cases = [
            ("1 m/s", 1.0),
            ("1 km/h", 0.2777777777777778),
            ("1 mph", 0.44704),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_velocity(input_val)
                assert_approx_equal(result, expected)

    def test_normalize_shear_rate_various_units(self):
        """Test normalize_shear_rate with various unit inputs."""
        test_cases = [
            ("1 s⁻¹", 1.0),
            ("1/s", 1.0),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_shear_rate(input_val)
                assert_approx_equal(result, expected)

    def test_viscosity_functions_various_units(self):
        """Test viscosity functions with various unit inputs"""
        # Test kinematic_viscosity with different viscosity units
        nu1 = kinematic_viscosity("1 mPa·s", "1000 kg/m³")
        nu2 = kinematic_viscosity("0.001 Pa·s", "1000 kg/m³")
        assert_approx_equal(nu1, nu2)

        # Test poiseuille_flow_rate with different length units
        Q1 = poiseuille_flow_rate("10 mm", "1 kPa", "1 m", "1 Pa·s")
        Q2 = poiseuille_flow_rate("0.01 m", "1000 Pa", "1 m", "1 Pa·s")
        assert_approx_equal(Q1, Q2)


if __name__ == '__main__':
    unittest.main()

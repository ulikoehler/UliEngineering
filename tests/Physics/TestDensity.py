#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_approx_equal

from UliEngineering.Physics.Density import normalize_density_kg_per_m3, DensityKgPerM3
from UliEngineering.Units import InvalidUnitInContextException


class TestDensityNormalization(unittest.TestCase):
    def test_si_units(self):
        """Test SI base units and unicode variants."""
        assert_approx_equal(normalize_density_kg_per_m3("1 kg/m^3"), 1.0)
        assert_approx_equal(normalize_density_kg_per_m3("1 kg/m3"), 1.0)
        assert_approx_equal(normalize_density_kg_per_m3("1 kg/m³"), 1.0)
        assert_approx_equal(normalize_density_kg_per_m3("1000 kg/m^3"), 1000.0)
        assert_approx_equal(normalize_density_kg_per_m3("2.5 kg/m³"), 2.5)

    def test_chemistry_units(self):
        """Test common chemistry/lab density units."""
        # g/cm³ variants
        assert_approx_equal(normalize_density_kg_per_m3("1 g/cm^3"), 1000.0)
        assert_approx_equal(normalize_density_kg_per_m3("2 g/cm^3"), 2000.0)
        assert_approx_equal(normalize_density_kg_per_m3("2.7 g/cm^3"), 2700.0)
        assert_approx_equal(normalize_density_kg_per_m3("1 g/cm3"), 1000.0)
        assert_approx_equal(normalize_density_kg_per_m3("1 g/cm³"), 1000.0)
        # kg/L variants
        assert_approx_equal(normalize_density_kg_per_m3("1 kg/L"), 1000.0)
        assert_approx_equal(normalize_density_kg_per_m3("1 kg/l"), 1000.0)
        assert_approx_equal(normalize_density_kg_per_m3("2.5 kg/L"), 2500.0)
        # g/mL variants
        assert_approx_equal(normalize_density_kg_per_m3("1 g/mL"), 1000.0)
        assert_approx_equal(normalize_density_kg_per_m3("1 g/ml"), 1000.0)
        # g/L variants
        assert_approx_equal(normalize_density_kg_per_m3("1 g/L"), 1.0)
        assert_approx_equal(normalize_density_kg_per_m3("1000 g/L"), 1000.0)
        assert_approx_equal(normalize_density_kg_per_m3("1 g/l"), 1.0)

    def test_imperial_units(self):
        """Test imperial density units."""
        # lb/ft³
        assert_approx_equal(normalize_density_kg_per_m3("1 lb/ft³"), 16.0184634)
        assert_approx_equal(normalize_density_kg_per_m3("1 lb/ft3"), 16.0184634)
        # lb/in³
        assert_approx_equal(normalize_density_kg_per_m3("1 lb/in³"), 27679.9047)
        assert_approx_equal(normalize_density_kg_per_m3("1 lb/in3"), 27679.9047)
        # lb/gal (US)
        assert_approx_equal(normalize_density_kg_per_m3("1 lb/gal"), 119.826427)
        # oz/in³
        assert_approx_equal(normalize_density_kg_per_m3("1 oz/in³"), 1729.994)
        assert_approx_equal(normalize_density_kg_per_m3("1 oz/in3"), 1729.994)

    def test_metric_ton_units(self):
        """Test tonne per cubic meter units."""
        assert_approx_equal(normalize_density_kg_per_m3("1 t/m³"), 1000.0)
        assert_approx_equal(normalize_density_kg_per_m3("1 t/m^3"), 1000.0)
        assert_approx_equal(normalize_density_kg_per_m3("1 t/m3"), 1000.0)
        assert_approx_equal(normalize_density_kg_per_m3("2.5 t/m³"), 2500.0)

    def test_bare_numbers(self):
        """Test that bare numbers pass through unchanged."""
        assert_approx_equal(normalize_density_kg_per_m3(1.0), 1.0)
        assert_approx_equal(normalize_density_kg_per_m3(1000), 1000.0)
        assert_approx_equal(normalize_density_kg_per_m3(0.5), 0.5)

    def test_numpy_array_input(self):
        """Test numpy array input."""
        arr = np.array(["1 g/cm^3", "2 kg/m^3"])
        result = normalize_density_kg_per_m3(arr)
        assert_allclose(result, [1000.0, 2.0])

    def test_list_input(self):
        """Test list input."""
        result = normalize_density_kg_per_m3(["1 g/cm^3", "2 g/cm^3"])
        assert_allclose(result, [1000.0, 2000.0])
        result = normalize_density_kg_per_m3(["1 kg/m^3", "2 g/L"])
        assert_allclose(result, [1.0, 2.0])

    def test_tuple_input(self):
        """Test tuple input."""
        result = normalize_density_kg_per_m3(("1 g/cm^3", "2 kg/m^3"))
        assert_allclose(result, [1000.0, 2.0])

    def test_bytes_input(self):
        """Test bytes input."""
        assert_approx_equal(normalize_density_kg_per_m3(b"1 kg/m^3"), 1.0)
        assert_approx_equal(normalize_density_kg_per_m3(b"1 g/cm^3"), 1000.0)

    def test_invalid_unit_raises(self):
        """Test that invalid units raise InvalidUnitInContextException."""
        with self.assertRaises(InvalidUnitInContextException):
            normalize_density_kg_per_m3("1 lb/m^3")
        with self.assertRaises(InvalidUnitInContextException):
            normalize_density_kg_per_m3("1 g/mm^3")

    def test_missing_numeric_part_raises(self):
        """Test that missing numeric part raises ValueError."""
        with self.assertRaises(ValueError):
            normalize_density_kg_per_m3("kg/m^3")

    def test_none_raises(self):
        """Test that None raises ValueError."""
        with self.assertRaises(ValueError):
            normalize_density_kg_per_m3(None)

    def test_type_annotation_exists(self):
        """Test that the type annotation is available."""
        self.assertIsNotNone(DensityKgPerM3)

    def test_comprehensive_unit_cases(self):
        """Test all supported units in one parameterized block."""
        test_cases = [
            # SI
            ("1 kg/m^3", 1.0),
            ("1 kg/m3", 1.0),
            ("1 kg/m³", 1.0),
            # Chemistry
            ("1 g/cm^3", 1000.0),
            ("1 g/cm3", 1000.0),
            ("1 g/cm³", 1000.0),
            ("1 kg/L", 1000.0),
            ("1 kg/l", 1000.0),
            ("1 g/mL", 1000.0),
            ("1 g/ml", 1000.0),
            ("1 g/L", 1.0),
            ("1 g/l", 1.0),
            # Imperial
            ("1 lb/ft³", 16.0184634),
            ("1 lb/ft3", 16.0184634),
            ("1 lb/in³", 27679.9047),
            ("1 lb/in3", 27679.9047),
            ("1 lb/gal", 119.826427),
            ("1 oz/in³", 1729.994),
            ("1 oz/in3", 1729.994),
            # Tonne
            ("1 t/m³", 1000.0),
            ("1 t/m^3", 1000.0),
            ("1 t/m3", 1000.0),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                result = normalize_density_kg_per_m3(input_val)
                assert_approx_equal(result, expected)


class TestDensityIntegration(unittest.TestCase):
    """Integration tests for density normalization in downstream functions."""

    def test_cylinder_weight_with_density_units(self):
        """Test that cylinder weight functions accept density with units."""
        from UliEngineering.Math.Geometry.Cylinder import (
            cylinder_weight_by_diameter,
            cylinder_weight_by_radius,
            cylinder_weight_by_cross_sectional_area,
        )

        # radius=1mm, length=1mm, density=2700 kg/m^3 (aluminium)
        # volume = pi * (1)^2 * 1 = pi mm^3
        # weight = volume * density = pi * 2700
        expected = np.pi * 2700.0

        # With SI density unit
        w = cylinder_weight_by_radius(1.0, 1.0, "2700 kg/m^3")
        self.assertAlmostEqual(w, expected, delta=0.1)

        # With g/cm^3 (2.7 g/cm^3 = 2700 kg/m^3)
        w = cylinder_weight_by_radius(1.0, 1.0, "2.7 g/cm^3")
        self.assertAlmostEqual(w, expected, delta=0.1)

        # With kg/L (2.7 kg/L = 2700 kg/m^3)
        w = cylinder_weight_by_radius(1.0, 1.0, "2.7 kg/L")
        self.assertAlmostEqual(w, expected, delta=0.1)

        # diameter=2mm, length=1mm => same volume and weight
        w = cylinder_weight_by_diameter(2.0, 1.0, "2.7 g/cm^3")
        self.assertAlmostEqual(w, expected, delta=0.1)

        # cross-sectional area = pi * r^2 = pi mm^2
        w = cylinder_weight_by_cross_sectional_area(np.pi, 1.0, "2.7 g/cm^3")
        self.assertAlmostEqual(w, expected, delta=0.1)

    def test_viscosity_functions_with_density_units(self):
        """Test that viscosity functions accept density with units."""
        from UliEngineering.Physics.Viscosity import (
            kinematic_viscosity,
            reynolds_number,
        )

        # kinematic_viscosity(0.001 Pa·s, 1000 kg/m³) => 1e-6 m²/s
        nu = kinematic_viscosity(0.001, "1 g/cm^3")
        self.assertAlmostEqual(nu, 1e-6, delta=1e-12)

        nu = kinematic_viscosity(0.001, "1 kg/L")
        self.assertAlmostEqual(nu, 1e-6, delta=1e-12)

        # Reynolds number: Re = rho * v * L / eta
        Re = reynolds_number("1 g/cm^3", 1.0, 0.1, 0.001)
        self.assertAlmostEqual(Re, 100000.0, delta=0.1)


if __name__ == '__main__':
    unittest.main()
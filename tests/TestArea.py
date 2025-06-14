#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from parameterized import parameterized
from UliEngineering.Area import *
import unittest
import numpy as np
import scipy.constants

from UliEngineering.Units import UnknownUnitInContextException

class TestArea(unittest.TestCase):
    def test_area_normalization_basic(self):
        # Basic numeric values
        assert_approx_equal(normalize_area(1.0), 1.0)
        assert_approx_equal(normalize_area(5.0), 5.0)
        assert_approx_equal(normalize_area(3), 3)
        
        # None handling
        self.assertIsNone(normalize_area(None))
    
    def test_area_normalization_metric_units(self):
        # Square meters (base unit)
        assert_approx_equal(normalize_area("1.0 m²"), 1.0)
        assert_approx_equal(normalize_area("1.0 m^2"), 1.0)
        assert_approx_equal(normalize_area("1.0 square meter"), 1.0)
        assert_approx_equal(normalize_area("1.0 square meters"), 1.0)
        assert_approx_equal(normalize_area("1.0 sq m"), 1.0)
        assert_approx_equal(normalize_area("1.0 sqm"), 1.0)
        assert_approx_equal(normalize_area("5.0 m²"), 5.0)
        assert_approx_equal(normalize_area("3 m²"), 3)
        
        # Square millimeters - all variations
        assert_approx_equal(normalize_area("1.0 mm²"), 1e-6)
        assert_approx_equal(normalize_area("1.0 mm^2"), 1e-6)
        assert_approx_equal(normalize_area("1.0 square millimeter"), 1e-6)
        assert_approx_equal(normalize_area("1.0 square millimeters"), 1e-6)
        assert_approx_equal(normalize_area("1.0 square mm"), 1e-6)
        assert_approx_equal(normalize_area("1.0 sq mm"), 1e-6)
        assert_approx_equal(normalize_area("1.0 mm sq"), 1e-6)
        assert_approx_equal(normalize_area("1.0 mm squared"), 1e-6)
        assert_approx_equal(normalize_area("1.0 millimeter squared"), 1e-6)
        assert_approx_equal(normalize_area("1000000 mm²"), 1.0)  # 1 m² = 1,000,000 mm²
        
        # Square centimeters - all variations
        assert_approx_equal(normalize_area("1.0 cm²"), 1e-4)
        assert_approx_equal(normalize_area("1.0 cm^2"), 1e-4)
        assert_approx_equal(normalize_area("1.0 square centimeter"), 1e-4)
        assert_approx_equal(normalize_area("1.0 square centimeters"), 1e-4)
        assert_approx_equal(normalize_area("1.0 square cm"), 1e-4)
        assert_approx_equal(normalize_area("1.0 sq cm"), 1e-4)
        assert_approx_equal(normalize_area("1.0 cm sq"), 1e-4)
        assert_approx_equal(normalize_area("1.0 cm squared"), 1e-4)
        assert_approx_equal(normalize_area("1.0 centimeter squared"), 1e-4)
        assert_approx_equal(normalize_area("1.0 centimeters squared"), 1e-4)
        assert_approx_equal(normalize_area("10000 cm²"), 1.0)  # 1 m² = 10,000 cm²
        
        # Square decimeters - all variations
        assert_approx_equal(normalize_area("1.0 dm²"), 1e-2)
        assert_approx_equal(normalize_area("1.0 dm^2"), 1e-2)
        assert_approx_equal(normalize_area("1.0 square decimeter"), 1e-2)
        assert_approx_equal(normalize_area("1.0 square decimeters"), 1e-2)
        assert_approx_equal(normalize_area("1.0 square dm"), 1e-2)
        assert_approx_equal(normalize_area("1.0 sq dm"), 1e-2)
        assert_approx_equal(normalize_area("1.0 dm sq"), 1e-2)
        assert_approx_equal(normalize_area("1.0 dm squared"), 1e-2)
        assert_approx_equal(normalize_area("1.0 decimeter squared"), 1e-2)
        assert_approx_equal(normalize_area("1.0 decimeters squared"), 1e-2)
        assert_approx_equal(normalize_area("100 dm²"), 1.0)  # 1 m² = 100 dm²
        
        # Square micrometers - all variations
        assert_approx_equal(normalize_area("1.0 µm²"), 1e-12)
        assert_approx_equal(normalize_area("1.0 µm^2"), 1e-12)
        assert_approx_equal(normalize_area("1.0 um²"), 1e-12)
        assert_approx_equal(normalize_area("1.0 um^2"), 1e-12)
        assert_approx_equal(normalize_area("1.0 square micrometer"), 1e-12)
        assert_approx_equal(normalize_area("1.0 square micrometers"), 1e-12)
        assert_approx_equal(normalize_area("1.0 square µm"), 1e-12)
        assert_approx_equal(normalize_area("1.0 sq µm"), 1e-12)
        assert_approx_equal(normalize_area("1.0 sq um"), 1e-12)
        assert_approx_equal(normalize_area("1.0 um sq"), 1e-12)
        assert_approx_equal(normalize_area("1.0 µm sq"), 1e-12)
        assert_approx_equal(normalize_area("1.0 µm squared"), 1e-12)
        assert_approx_equal(normalize_area("1.0 micrometer squared"), 1e-12)
        assert_approx_equal(normalize_area("1.0 micrometers squared"), 1e-12)
        
        # Square nanometers - all variations
        assert_approx_equal(normalize_area("1.0 nm²"), 1e-18)
        assert_approx_equal(normalize_area("1.0 nm^2"), 1e-18)
        assert_approx_equal(normalize_area("1.0 square nanometer"), 1e-18)
        assert_approx_equal(normalize_area("1.0 square nanometers"), 1e-18)
        assert_approx_equal(normalize_area("1.0 square nm"), 1e-18)
        assert_approx_equal(normalize_area("1.0 sq nm"), 1e-18)
        assert_approx_equal(normalize_area("1.0 nm sq"), 1e-18)
        assert_approx_equal(normalize_area("1.0 nm squared"), 1e-18)
        assert_approx_equal(normalize_area("1.0 nanometers squared"), 1e-18)
        
        # Square kilometers - all variations
        assert_approx_equal(normalize_area("1.0 km²"), 1e6)
        assert_approx_equal(normalize_area("1.0 km^2"), 1e6)
        assert_approx_equal(normalize_area("1.0 square kilometer"), 1e6)
        assert_approx_equal(normalize_area("1.0 square kilometers"), 1e6)
        assert_approx_equal(normalize_area("1.0 square km"), 1e6)
        assert_approx_equal(normalize_area("1.0 sq km"), 1e6)
        assert_approx_equal(normalize_area("1.0 km squared"), 1e6)
        assert_approx_equal(normalize_area("1.0 km sq"), 1e6)
        assert_approx_equal(normalize_area("1.0 kilometers sq"), 1e6)
        assert_approx_equal(normalize_area("1.0 kilometers squared"), 1e6)
        assert_approx_equal(normalize_area("0.000001 km²"), 1.0)  # 1 m² = 0.000001 km²

    def test_area_normalization_imperial_units(self):
        # Square inches
        expected_sq_inch = scipy.constants.inch**2
        assert_approx_equal(normalize_area("1.0 in²"), expected_sq_inch)
        assert_approx_equal(normalize_area("1.0 in^2"), expected_sq_inch)
        assert_approx_equal(normalize_area("1.0 square inch"), expected_sq_inch)
        assert_approx_equal(normalize_area("1.0 square inches"), expected_sq_inch)
        assert_approx_equal(normalize_area("1.0 sq in"), expected_sq_inch)
        
        # Square feet
        expected_sq_foot = scipy.constants.foot**2
        assert_approx_equal(normalize_area("1.0 ft²"), expected_sq_foot)
        assert_approx_equal(normalize_area("1.0 ft^2"), expected_sq_foot)
        assert_approx_equal(normalize_area("1.0 square foot"), expected_sq_foot)
        assert_approx_equal(normalize_area("1.0 square feet"), expected_sq_foot)
        assert_approx_equal(normalize_area("1.0 sq ft"), expected_sq_foot)
        assert_approx_equal(normalize_area("144 in²"), expected_sq_foot, significant=6)  # 1 ft² = 144 in²
        
        # Square yards
        expected_sq_yard = scipy.constants.yard**2
        assert_approx_equal(normalize_area("1.0 yd²"), expected_sq_yard)
        assert_approx_equal(normalize_area("1.0 yd^2"), expected_sq_yard)
        assert_approx_equal(normalize_area("1.0 square yard"), expected_sq_yard)
        assert_approx_equal(normalize_area("1.0 square yards"), expected_sq_yard)
        assert_approx_equal(normalize_area("1.0 sq yd"), expected_sq_yard)
        assert_approx_equal(normalize_area("9 ft²"), expected_sq_yard, significant=6)  # 1 yd² = 9 ft²

    def test_area_normalization_agricultural_units(self):
        # Acres
        assert_approx_equal(normalize_area("1.0 acre"), 4046.8564224)
        assert_approx_equal(normalize_area("1.0 acres"), 4046.8564224)
        assert_approx_equal(normalize_area("2.5 acres"), 2.5 * 4046.8564224)
        
        # Hectares
        assert_approx_equal(normalize_area("1.0 hectare"), 10000.0)
        assert_approx_equal(normalize_area("1.0 hectares"), 10000.0)
        assert_approx_equal(normalize_area("1.0 ha"), 10000.0)
        assert_approx_equal(normalize_area("2.5 hectares"), 25000.0)
        
        # Ares
        assert_approx_equal(normalize_area("1.0 are"), 100.0)
        assert_approx_equal(normalize_area("1.0 ares"), 100.0)
        assert_approx_equal(normalize_area("100 ares"), 10000.0)  # 100 ares = 1 hectare

    def test_area_normalization_scientific_units(self):
        # Barns (nuclear physics unit)
        assert_approx_equal(normalize_area("1.0 barn"), 1e-28)
        assert_approx_equal(normalize_area("1.0 barns"), 1e-28)
        assert_approx_equal(normalize_area("1.0 b"), 1e-28)
        assert_approx_equal(normalize_area("1000 barn"), 1e-25)

    def test_area_normalization_prefixed_units(self):
        # Test SI prefixes with square meters
        assert_approx_equal(normalize_area("1 km²"), 1e6)
        assert_approx_equal(normalize_area("1 mm²"), 1e-6)
        assert_approx_equal(normalize_area("1 cm²"), 1e-4)
        
        # Test with barn (microbarns, millibarns, etc.)
        assert_approx_equal(normalize_area("1 µbarn"), 1e-34)
        assert_approx_equal(normalize_area("1 mbarn"), 1e-31)

    def test_area_normalization_lists(self):
        # Test list input
        input_list = ["1 m²", "100 cm²", "1 ft²"]
        result = normalize_area(input_list)
        expected = [1.0, 0.01, scipy.constants.foot**2]
        for i, val in enumerate(expected):
            assert_approx_equal(result[i], val)

    def test_area_normalization_numpy_arrays(self):
        # Test numpy array input
        input_array = np.array(["1 m²", "100 cm²", "1 ft²"])
        result = normalize_area(input_array)
        expected = np.array([1.0, 0.01, scipy.constants.foot**2])
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_area_to_square_meters_basic(self):
        # Basic conversions
        assert_approx_equal(convert_area_to_square_meters(1.0, "m²"), 1.0)
        assert_approx_equal(convert_area_to_square_meters(1.0, "square meter"), 1.0)
        assert_approx_equal(convert_area_to_square_meters(5.0, "m²"), 5.0)
        assert_approx_equal(convert_area_to_square_meters(3, "m²"), 3)

    def test_convert_area_to_square_meters_metric(self):
        # Metric units
        assert_approx_equal(convert_area_to_square_meters(1.0, "mm²"), 1e-6)
        assert_approx_equal(convert_area_to_square_meters(1.0, "cm²"), 1e-4)
        assert_approx_equal(convert_area_to_square_meters(1.0, "dm²"), 1e-2)
        assert_approx_equal(convert_area_to_square_meters(1.0, "km²"), 1e6)
        assert_approx_equal(convert_area_to_square_meters(1000000, "mm²"), 1.0)
        assert_approx_equal(convert_area_to_square_meters(10000, "cm²"), 1.0)
        assert_approx_equal(convert_area_to_square_meters(100, "dm²"), 1.0)
        assert_approx_equal(convert_area_to_square_meters(0.000001, "km²"), 1.0)

    def test_convert_area_to_square_meters_imperial(self):
        # Imperial units
        expected_sq_inch = scipy.constants.inch**2
        expected_sq_foot = scipy.constants.foot**2
        expected_sq_yard = scipy.constants.yard**2
        
        assert_approx_equal(convert_area_to_square_meters(1.0, "in²"), expected_sq_inch)
        assert_approx_equal(convert_area_to_square_meters(1.0, "square inch"), expected_sq_inch)
        assert_approx_equal(convert_area_to_square_meters(1.0, "ft²"), expected_sq_foot)
        assert_approx_equal(convert_area_to_square_meters(1.0, "square foot"), expected_sq_foot)
        assert_approx_equal(convert_area_to_square_meters(1.0, "yd²"), expected_sq_yard)
        assert_approx_equal(convert_area_to_square_meters(1.0, "square yard"), expected_sq_yard)

    def test_convert_area_to_square_meters_agricultural(self):
        # Agricultural units
        assert_approx_equal(convert_area_to_square_meters(1.0, "acre"), 4046.8564224)
        assert_approx_equal(convert_area_to_square_meters(1.0, "hectare"), 10000.0)
        assert_approx_equal(convert_area_to_square_meters(1.0, "ha"), 10000.0)
        assert_approx_equal(convert_area_to_square_meters(1.0, "are"), 100.0)

    def test_convert_area_to_square_meters_scientific(self):
        # Scientific units
        assert_approx_equal(convert_area_to_square_meters(1.0, "barn"), 1e-28)
        assert_approx_equal(convert_area_to_square_meters(1.0, "b"), 1e-28)

    def test_real_world_conversions(self):
        # Real-world examples
        # A standard soccer field is approximately 100m x 70m = 7000 m²
        assert_approx_equal(normalize_area("7000 m²"), 7000.0)
        assert_approx_equal(normalize_area("0.7 hectares"), 7000.0)
        
        # A typical house lot might be 0.25 acres
        house_lot_m2 = 0.25 * 4046.8564224
        assert_approx_equal(normalize_area("0.25 acres"), house_lot_m2)
        
        # A sheet of A4 paper is approximately 210mm x 297mm ≈ 623.7 cm²
        a4_area = 21.0 * 29.7  # cm²
        # Verified using Wolfram Alpha: "area of a4 sheet of paper in m²"
        assert_approx_equal(normalize_area(f"{a4_area} cm²"), 0.06237)

    def test_edge_cases(self):
        # Test very small and very large values
        assert_approx_equal(normalize_area("1e-10 m²"), 1e-10)
        assert_approx_equal(normalize_area("1e10 m²"), 1e10)
        
        # Test decimal values
        assert_approx_equal(normalize_area("3.14159 m²"), 3.14159)
        assert_approx_equal(normalize_area("0.5 hectares"), 5000.0)

    @parameterized.expand([
        ("1A",),
        ("xyz",),
        ("invalid_unit",),
        ("square invalid",),
    ])
    def test_invalid_units(self, unit):
        with self.assertRaises((ValueError, UnknownUnitInContextException)):
            normalize_area(f"6.6 {unit}")

    @parameterized.expand([
        ("1A",),
        ("xyz",),
        ("invalid_unit",),
        ("square invalid",),
    ])
    def test_convert_invalid_units(self, unit):
        with self.assertRaises((ValueError, UnknownUnitInContextException)):
            convert_area_to_square_meters(6.6, unit)

    def test_empty_string(self):
        # Test empty and whitespace strings
        with self.assertRaises(ValueError):
            normalize_area("")
        with self.assertRaises(ValueError):
            normalize_area("   ")

    def test_consistency_between_functions(self):
        # Test that both functions give the same results
        test_cases = [
            (1.0, "m²"),
            (100.0, "cm²"),
            (1.0, "ft²"),
            (1.0, "acre"),
            (2.5, "hectares"),
            (1000.0, "barn")
        ]
        
        for value, unit in test_cases:
            result1 = normalize_area(f"{value} {unit}")
            result2 = convert_area_to_square_meters(value, unit)
            assert_approx_equal(result1, result2, 
                              err_msg=f"Inconsistent results for {value} {unit}")

    def test_all_area_unit_variations_comprehensive(self):
        """Test all area unit variations from _area_units() to ensure complete coverage"""
        # Test all variations that should be equivalent to 1e-6 m² (square millimeters)
        mm_units = [
            "mm²", "mm^2", "square millimeter", "square millimeters", 
            "square mm", "sq mm", "mm sq", "mm squared", "millimeter squared"
        ]
        for unit in mm_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), 1e-6,
                              err_msg=f"Failed for unit: {unit}")
        
        # Test all variations that should be equivalent to 1e-4 m² (square centimeters)
        cm_units = [
            "cm²", "cm^2", "square centimeter", "square centimeters",
            "square cm", "sq cm", "cm sq", "cm squared", 
            "centimeter squared", "centimeters squared"
        ]
        for unit in cm_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), 1e-4,
                              err_msg=f"Failed for unit: {unit}")
        
        # Test all variations that should be equivalent to 1e-2 m² (square decimeters)
        dm_units = [
            "dm²", "dm^2", "square decimeter", "square decimeters",
            "square dm", "sq dm", "dm sq", "dm squared",
            "decimeter squared", "decimeters squared"
        ]
        for unit in dm_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), 1e-2,
                              err_msg=f"Failed for unit: {unit}")
        
        # Test all variations that should be equivalent to 1.0 m² (square meters)
        m_units = [
            "m²", "m^2", "square meter", "square meters", "sq m", "sqm"
        ]
        for unit in m_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), 1.0,
                              err_msg=f"Failed for unit: {unit}")
        
        # Test all variations that should be equivalent to 1e-12 m² (square micrometers)
        um_units = [
            "µm²", "µm^2", "um²", "um^2", "square micrometer", "square micrometers",
            "square µm", "sq µm", "sq um", "um sq", "µm sq", "µm squared",
            "micrometer squared", "micrometers squared"
        ]
        for unit in um_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), 1e-12,
                              err_msg=f"Failed for unit: {unit}")
        
        # Test all variations that should be equivalent to 1e-18 m² (square nanometers)
        nm_units = [
            "nm²", "nm^2", "square nanometer", "square nanometers",
            "square nm", "sq nm", "nm sq", "nm squared", "nanometers squared"
        ]
        for unit in nm_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), 1e-18,
                              err_msg=f"Failed for unit: {unit}")
        
        # Test all variations that should be equivalent to 1e6 m² (square kilometers)
        km_units = [
            "km²", "km^2", "square kilometer", "square kilometers",
            "square km", "sq km", "km squared", "km sq", 
            "kilometers sq", "kilometers squared"
        ]
        for unit in km_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), 1e6,
                              err_msg=f"Failed for unit: {unit}")
        
        # Test all imperial unit variations
        inch_expected = scipy.constants.inch**2
        inch_units = ["in²", "in^2", "square inch", "square inches", "sq in"]
        for unit in inch_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), inch_expected,
                              err_msg=f"Failed for unit: {unit}")
        
        foot_expected = scipy.constants.foot**2
        foot_units = ["ft²", "ft^2", "square foot", "square feet", "sq ft"]
        for unit in foot_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), foot_expected,
                              err_msg=f"Failed for unit: {unit}")
        
        yard_expected = scipy.constants.yard**2
        yard_units = ["yd²", "yd^2", "square yard", "square yards", "sq yd"]
        for unit in yard_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), yard_expected,
                              err_msg=f"Failed for unit: {unit}")
        
        # Test agricultural units
        acre_units = ["acre", "acres"]
        for unit in acre_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), 4046.8564224,
                              err_msg=f"Failed for unit: {unit}")
        
        hectare_units = ["hectare", "hectares", "ha"]
        for unit in hectare_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), 10000.0,
                              err_msg=f"Failed for unit: {unit}")
        
        are_units = ["are", "ares"]
        for unit in are_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), 100.0,
                              err_msg=f"Failed for unit: {unit}")
        
        # Test scientific units
        barn_units = ["barn", "barns", "b"]
        for unit in barn_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), 1e-28,
                              err_msg=f"Failed for unit: {unit}")
        
        # Test prefixed barn units
        assert_approx_equal(normalize_area("1.0 µbarn"), 1e-34)
        assert_approx_equal(normalize_area("1.0 mbarn"), 1e-31)

    def test_all_area_unit_variations(self):
        """Test all area unit variations to ensure complete coverage"""
        # Test all variations that should be equivalent to 1e-6 m² (square millimeters)
        mm_units = [
            "mm²", "mm^2", "square millimeter", "square millimeters", 
            "square mm", "sq mm", "mm sq", "mm squared", "millimeter squared"
        ]
        for unit in mm_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), 1e-6,
                              err_msg=f"Failed for unit: {unit}")
        
        # Test all variations that should be equivalent to 1e-4 m² (square centimeters)
        cm_units = [
            "cm²", "cm^2", "square centimeter", "square centimeters",
            "square cm", "sq cm", "cm sq", "cm squared", 
            "centimeter squared", "centimeters squared"
        ]
        for unit in cm_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), 1e-4,
                              err_msg=f"Failed for unit: {unit}")
        
        # Test all variations that should be equivalent to 1e-2 m² (square decimeters)
        dm_units = [
            "dm²", "dm^2", "square decimeter", "square decimeters",
            "square dm", "sq dm", "dm sq", "dm squared",
            "decimeter squared", "decimeters squared"
        ]
        for unit in dm_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), 1e-2,
                              err_msg=f"Failed for unit: {unit}")
        
        # Test all variations that should be equivalent to 1.0 m² (square meters)
        m_units = [
            "m²", "m^2", "square meter", "square meters", "sq m", "sqm"
        ]
        for unit in m_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), 1.0,
                              err_msg=f"Failed for unit: {unit}")
        
        # Test all variations that should be equivalent to 1e-12 m² (square micrometers)
        um_units = [
            "µm²", "µm^2", "um²", "um^2", "square micrometer", "square micrometers",
            "square µm", "sq µm", "sq um", "um sq", "µm sq", "µm squared",
            "micrometer squared", "micrometers squared"
        ]
        for unit in um_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), 1e-12,
                              err_msg=f"Failed for unit: {unit}")
        
        # Test all variations that should be equivalent to 1e-18 m² (square nanometers)
        nm_units = [
            "nm²", "nm^2", "square nanometer", "square nanometers",
            "square nm", "sq nm", "nm sq", "nm squared", "nanometers squared"
        ]
        for unit in nm_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), 1e-18,
                              err_msg=f"Failed for unit: {unit}")
        
        # Test all variations that should be equivalent to 1e6 m² (square kilometers)
        km_units = [
            "km²", "km^2", "square kilometer", "square kilometers",
            "square km", "sq km", "km squared", "km sq", 
            "kilometers sq", "kilometers squared"
        ]
        for unit in km_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), 1e6,
                              err_msg=f"Failed for unit: {unit}")
        
        # Test all imperial unit variations
        inch_expected = scipy.constants.inch**2
        inch_units = ["in²", "in^2", "square inch", "square inches", "sq in"]
        for unit in inch_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), inch_expected,
                              err_msg=f"Failed for unit: {unit}")
        
        foot_expected = scipy.constants.foot**2
        foot_units = ["ft²", "ft^2", "square foot", "square feet", "sq ft"]
        for unit in foot_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), foot_expected,
                              err_msg=f"Failed for unit: {unit}")
        
        yard_expected = scipy.constants.yard**2
        yard_units = ["yd²", "yd^2", "square yard", "square yards", "sq yd"]
        for unit in yard_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), yard_expected,
                              err_msg=f"Failed for unit: {unit}")
        
        # Test agricultural units
        acre_units = ["acre", "acres"]
        for unit in acre_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), 4046.8564224,
                              err_msg=f"Failed for unit: {unit}")
        
        hectare_units = ["hectare", "hectares", "ha"]
        for unit in hectare_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), 10000.0,
                              err_msg=f"Failed for unit: {unit}")
        
        are_units = ["are", "ares"]
        for unit in are_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), 100.0,
                              err_msg=f"Failed for unit: {unit}")
        
        # Test scientific units
        barn_units = ["barn", "barns", "b"]
        for unit in barn_units:
            assert_approx_equal(normalize_area(f"1.0 {unit}"), 1e-28,
                              err_msg=f"Failed for unit: {unit}")
        
        # Test prefixed barn units
        assert_approx_equal(normalize_area("1.0 µbarn"), 1e-34)
        assert_approx_equal(normalize_area("1.0 mbarn"), 1e-31)

if __name__ == '__main__':
    unittest.main()
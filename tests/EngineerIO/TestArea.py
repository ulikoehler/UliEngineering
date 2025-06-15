#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from parameterized import parameterized
from UliEngineering.EngineerIO.Area import *
import unittest
import numpy as np
import scipy.constants

from UliEngineering.Units import UnknownUnitInContextException

class TestArea(unittest.TestCase):
    def setUp(self):
        self.area_io = EngineerAreaIO()

    def test_area_normalization_basic(self):
        # Basic numeric values
        assert_approx_equal(normalize_area(1.0), 1.0)
        assert_approx_equal(normalize_area(5.0), 5.0)
        assert_approx_equal(normalize_area(3), 3)
        
        # Test with class instance
        assert_approx_equal(self.area_io.normalize_area(1.0), 1.0)
        assert_approx_equal(self.area_io.normalize_area(5.0), 5.0)
        assert_approx_equal(self.area_io.normalize_area(3), 3)
        
        # None handling
        self.assertIsNone(normalize_area(None))
        self.assertIsNone(self.area_io.normalize_area(None))
    
    def test_area_normalization_metric_units(self):
        # Square meters (base unit)
        assert_approx_equal(normalize_area("1.0 m²"), 1.0)
        assert_approx_equal(self.area_io.normalize_area("1.0 m²"), 1.0)
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
        assert_approx_equal(self.area_io.normalize_area("1.0 in²"), expected_sq_inch)
        assert_approx_equal(normalize_area("1.0 in^2"), expected_sq_inch)
        assert_approx_equal(normalize_area("1.0 square inch"), expected_sq_inch)
        assert_approx_equal(normalize_area("1.0 square inches"), expected_sq_inch)
        assert_approx_equal(normalize_area("1.0 sq in"), expected_sq_inch)
        
        # Square feet
        expected_sq_foot = scipy.constants.foot**2
        assert_approx_equal(normalize_area("1.0 ft²"), expected_sq_foot)
        assert_approx_equal(self.area_io.normalize_area("1.0 ft²"), expected_sq_foot)
        assert_approx_equal(normalize_area("1.0 ft^2"), expected_sq_foot)
        assert_approx_equal(normalize_area("1.0 square foot"), expected_sq_foot)
        assert_approx_equal(normalize_area("1.0 square feet"), expected_sq_foot)
        assert_approx_equal(normalize_area("1.0 sq ft"), expected_sq_foot)
        assert_approx_equal(normalize_area("144 in²"), expected_sq_foot, significant=6)  # 1 ft² = 144 in²
        
        # Square yards
        expected_sq_yard = scipy.constants.yard**2
        assert_approx_equal(normalize_area("1.0 yd²"), expected_sq_yard)
        assert_approx_equal(self.area_io.normalize_area("1.0 yd²"), expected_sq_yard)
        assert_approx_equal(normalize_area("1.0 yd^2"), expected_sq_yard)
        assert_approx_equal(normalize_area("1.0 square yard"), expected_sq_yard)
        assert_approx_equal(normalize_area("1.0 square yards"), expected_sq_yard)
        assert_approx_equal(normalize_area("1.0 sq yd"), expected_sq_yard)
        assert_approx_equal(normalize_area("9 ft²"), expected_sq_yard, significant=6)  # 1 yd² = 9 ft²

    def test_area_normalization_agricultural_units(self):
        # Acres
        assert_approx_equal(normalize_area("1.0 acre"), 4046.8564224)
        assert_approx_equal(self.area_io.normalize_area("1.0 acre"), 4046.8564224)
        assert_approx_equal(normalize_area("1.0 acres"), 4046.8564224)
        assert_approx_equal(self.area_io.normalize_area("1.0 acres"), 4046.8564224)
        assert_approx_equal(normalize_area("2.5 acres"), 2.5 * 4046.8564224)
        assert_approx_equal(self.area_io.normalize_area("2.5 acres"), 2.5 * 4046.8564224)
        
        # Hectares
        assert_approx_equal(normalize_area("1.0 hectare"), 10000.0)
        assert_approx_equal(self.area_io.normalize_area("1.0 hectare"), 10000.0)
        assert_approx_equal(normalize_area("1.0 hectares"), 10000.0)
        assert_approx_equal(self.area_io.normalize_area("1.0 hectares"), 10000.0)
        assert_approx_equal(normalize_area("1.0 ha"), 10000.0)
        assert_approx_equal(self.area_io.normalize_area("1.0 ha"), 10000.0)
        assert_approx_equal(normalize_area("2.5 hectares"), 25000.0)
        assert_approx_equal(self.area_io.normalize_area("2.5 hectares"), 25000.0)
        
        # Ares
        assert_approx_equal(normalize_area("1.0 are"), 100.0)
        assert_approx_equal(self.area_io.normalize_area("1.0 are"), 100.0)
        assert_approx_equal(normalize_area("1.0 ares"), 100.0)
        assert_approx_equal(self.area_io.normalize_area("1.0 ares"), 100.0)
        assert_approx_equal(normalize_area("100 ares"), 10000.0)  # 100 ares = 1 hectare

    def test_area_normalization_scientific_units(self):
        # Barns (nuclear physics unit)
        assert_approx_equal(normalize_area("1.0 barn"), 1e-28)
        assert_approx_equal(self.area_io.normalize_area("1.0 barn"), 1e-28)
        assert_approx_equal(normalize_area("1.0 barns"), 1e-28)
        assert_approx_equal(self.area_io.normalize_area("1.0 barns"), 1e-28)
        assert_approx_equal(normalize_area("1.0 b"), 1e-28)
        assert_approx_equal(self.area_io.normalize_area("1.0 b"), 1e-28)
        assert_approx_equal(normalize_area("1000 barn"), 1e-25)
        assert_approx_equal(self.area_io.normalize_area("1000 barn"), 1e-25)

    def test_area_normalization_prefixed_units(self):
        # Test SI prefixes with square meters
        assert_approx_equal(normalize_area("1 km²"), 1e6)
        assert_approx_equal(self.area_io.normalize_area("1 km²"), 1e6)
        assert_approx_equal(normalize_area("1 mm²"), 1e-6)
        assert_approx_equal(self.area_io.normalize_area("1 mm²"), 1e-6)
        assert_approx_equal(normalize_area("1 cm²"), 1e-4)
        assert_approx_equal(self.area_io.normalize_area("1 cm²"), 1e-4)
        
        # Test with barn (microbarns, millibarns, etc.)
        assert_approx_equal(normalize_area("1 µbarn"), 1e-34)
        assert_approx_equal(self.area_io.normalize_area("1 µbarn"), 1e-34)
        assert_approx_equal(normalize_area("1 mbarn"), 1e-31)
        assert_approx_equal(self.area_io.normalize_area("1 mbarn"), 1e-31)

    def test_area_normalization_lists(self):
        # Test list input
        input_list = ["1 m²", "100 cm²", "1 ft²"]
        result = normalize_area(input_list)
        result_class = self.area_io.normalize_area(input_list)
        expected = [1.0, 0.01, scipy.constants.foot**2]
        for i, val in enumerate(expected):
            assert_approx_equal(result[i], val)
            assert_approx_equal(result_class[i], val)

    def test_area_normalization_numpy_arrays(self):
        # Test numpy array input
        input_array = np.array(["1 m²", "100 cm²", "1 ft²"])
        result = normalize_area(input_array)
        result_class = self.area_io.normalize_area(input_array)
        expected = np.array([1.0, 0.01, scipy.constants.foot**2])
        np.testing.assert_array_almost_equal(result, expected)
        np.testing.assert_array_almost_equal(result_class, expected)

    def test_convert_area_to_square_meters_basic(self):
        # Basic conversions
        assert_approx_equal(convert_area_to_square_meters(1.0, "m²"), 1.0)
        assert_approx_equal(self.area_io.convert_area_to_square_meters(1.0, "m²"), 1.0)
        assert_approx_equal(convert_area_to_square_meters(1.0, "square meter"), 1.0)
        assert_approx_equal(self.area_io.convert_area_to_square_meters(1.0, "square meter"), 1.0)
        assert_approx_equal(convert_area_to_square_meters(5.0, "m²"), 5.0)
        assert_approx_equal(self.area_io.convert_area_to_square_meters(5.0, "m²"), 5.0)
        assert_approx_equal(convert_area_to_square_meters(3, "m²"), 3)
        assert_approx_equal(self.area_io.convert_area_to_square_meters(3, "m²"), 3)

    def test_convert_area_to_square_meters_metric(self):
        # Metric units
        assert_approx_equal(convert_area_to_square_meters(1.0, "mm²"), 1e-6)
        assert_approx_equal(self.area_io.convert_area_to_square_meters(1.0, "mm²"), 1e-6)
        assert_approx_equal(convert_area_to_square_meters(1.0, "cm²"), 1e-4)
        assert_approx_equal(self.area_io.convert_area_to_square_meters(1.0, "cm²"), 1e-4)
        assert_approx_equal(convert_area_to_square_meters(1.0, "dm²"), 1e-2)
        assert_approx_equal(self.area_io.convert_area_to_square_meters(1.0, "dm²"), 1e-2)
        assert_approx_equal(convert_area_to_square_meters(1.0, "km²"), 1e6)
        assert_approx_equal(self.area_io.convert_area_to_square_meters(1.0, "km²"), 1e6)
        assert_approx_equal(convert_area_to_square_meters(1000000, "mm²"), 1.0)
        assert_approx_equal(self.area_io.convert_area_to_square_meters(1000000, "mm²"), 1.0)
        assert_approx_equal(convert_area_to_square_meters(10000, "cm²"), 1.0)
        assert_approx_equal(self.area_io.convert_area_to_square_meters(10000, "cm²"), 1.0)
        assert_approx_equal(convert_area_to_square_meters(100, "dm²"), 1.0)
        assert_approx_equal(self.area_io.convert_area_to_square_meters(100, "dm²"), 1.0)
        assert_approx_equal(convert_area_to_square_meters(0.000001, "km²"), 1.0)
        assert_approx_equal(self.area_io.convert_area_to_square_meters(0.000001, "km²"), 1.0)

    def test_convert_area_to_square_meters_imperial(self):
        # Imperial units
        expected_sq_inch = scipy.constants.inch**2
        expected_sq_foot = scipy.constants.foot**2
        expected_sq_yard = scipy.constants.yard**2
        
        assert_approx_equal(convert_area_to_square_meters(1.0, "in²"), expected_sq_inch)
        assert_approx_equal(self.area_io.convert_area_to_square_meters(1.0, "in²"), expected_sq_inch)
        assert_approx_equal(convert_area_to_square_meters(1.0, "square inch"), expected_sq_inch)
        assert_approx_equal(self.area_io.convert_area_to_square_meters(1.0, "square inch"), expected_sq_inch)
        assert_approx_equal(convert_area_to_square_meters(1.0, "ft²"), expected_sq_foot)
        assert_approx_equal(self.area_io.convert_area_to_square_meters(1.0, "ft²"), expected_sq_foot)
        assert_approx_equal(convert_area_to_square_meters(1.0, "square foot"), expected_sq_foot)
        assert_approx_equal(self.area_io.convert_area_to_square_meters(1.0, "square foot"), expected_sq_foot)
        assert_approx_equal(convert_area_to_square_meters(1.0, "yd²"), expected_sq_yard)
        assert_approx_equal(self.area_io.convert_area_to_square_meters(1.0, "yd²"), expected_sq_yard)
        assert_approx_equal(convert_area_to_square_meters(1.0, "square yard"), expected_sq_yard)
        assert_approx_equal(self.area_io.convert_area_to_square_meters(1.0, "square yard"), expected_sq_yard)

    def test_convert_area_to_square_meters_agricultural(self):
        # Agricultural units
        assert_approx_equal(convert_area_to_square_meters(1.0, "acre"), 4046.8564224)
        assert_approx_equal(self.area_io.convert_area_to_square_meters(1.0, "acre"), 4046.8564224)
        assert_approx_equal(convert_area_to_square_meters(1.0, "hectare"), 10000.0)
        assert_approx_equal(self.area_io.convert_area_to_square_meters(1.0, "hectare"), 10000.0)
        assert_approx_equal(convert_area_to_square_meters(1.0, "ha"), 10000.0)
        assert_approx_equal(self.area_io.convert_area_to_square_meters(1.0, "ha"), 10000.0)
        assert_approx_equal(convert_area_to_square_meters(1.0, "are"), 100.0)
        assert_approx_equal(self.area_io.convert_area_to_square_meters(1.0, "are"), 100.0)

    def test_convert_area_to_square_meters_scientific(self):
        # Scientific units
        assert_approx_equal(convert_area_to_square_meters(1.0, "barn"), 1e-28)
        assert_approx_equal(self.area_io.convert_area_to_square_meters(1.0, "barn"), 1e-28)
        assert_approx_equal(convert_area_to_square_meters(1.0, "b"), 1e-28)
        assert_approx_equal(self.area_io.convert_area_to_square_meters(1.0, "b"), 1e-28)

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

    def test_class_instance_vs_function_consistency(self):
        """Test that class methods and functions produce identical results"""
        test_cases = [
            "1.0 m²", "100 cm²", "1 ft²", "1 acre", "2.5 hectares", "1000 barn"
        ]
        
        for case in test_cases:
            with self.subTest(case=case):
                function_result = normalize_area(case)
                class_result = self.area_io.normalize_area(case)
                assert_approx_equal(function_result, class_result, 
                                  err_msg=f"Results differ for {case}")

    def test_class_convert_consistency(self):
        """Test that class convert method and function produce identical results"""
        test_cases = [
            (1.0, "m²"), (100.0, "cm²"), (1.0, "ft²"), (1.0, "acre")
        ]
        
        for value, unit in test_cases:
            with self.subTest(value=value, unit=unit):
                function_result = convert_area_to_square_meters(value, unit)
                class_result = self.area_io.convert_area_to_square_meters(value, unit)
                assert_approx_equal(function_result, class_result, 
                                  err_msg=f"Results differ for {value} {unit}")

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

class TestAreaUnits(unittest.TestCase):
    def setUp(self):
        # Create an EngineerIO instance with area units included
        self.io = EngineerAreaIO.instance()

    def test_area_unit_regex_end_matching(self):
        """Test that area unit regexes only match at the end of strings"""
        # Test units that should match at the end
        end_match_cases = [
            "100 m²",
            "50ft²", 
            "25 yd²",
            "10acre",
            "5 hectare",
            "2.5ha",
            "7barn"
        ]
        
        for case in end_match_cases:
            with self.subTest(case=case):
                # Should find a match when searching the whole string
                units_match = self.io.units_regex.search(case) if self.io.units_regex else None
                alias_match = self.io.unit_alias_regex.search(case) if self.io.unit_alias_regex else None
                
                # At least one should match
                self.assertTrue(units_match is not None or alias_match is not None,
                              f"No regex matched for case: {case}")
                
                # The match should be at the end of the string
                if units_match:
                    self.assertTrue(case.endswith(units_match.group(1)),
                                  f"Units regex match '{units_match.group(1)}' not at end of '{case}'")
                if alias_match:
                    self.assertTrue(case.endswith(alias_match.group(1)),
                                  f"Alias regex match '{alias_match.group(1)}' not at end of '{case}'")

    def test_area_unit_regex_no_middle_matching(self):
        """Test that area unit regexes do NOT match in the middle of strings"""
        # Test cases where unit appears in middle - should NOT match
        middle_no_match_cases = [
            "m²value",  # Unit in middle
            "ft²test",  # Unit in middle  
            "havalue",  # Unit in middle
            "acretest", # Unit in middle
            "barnvalue", # Unit in middle
            "square meter test", # Alias in middle
            "sq ft value", # Alias in middle
        ]
        
        for case in middle_no_match_cases:
            with self.subTest(case=case):
                # Should NOT find any matches when unit is in middle
                units_match = self.io.units_regex.search(case) if self.io.units_regex else None
                alias_match = self.io.unit_alias_regex.search(case) if self.io.unit_alias_regex else None
                
                self.assertIsNone(units_match, 
                                f"Units regex incorrectly matched in middle for case: {case}")
                self.assertIsNone(alias_match,
                                f"Alias regex incorrectly matched in middle for case: {case}")

    def test_unicode_area_units(self):
        """Test recognition of Unicode area units (²)"""
        # Note: we only normalize() here, so c in cm => 1/100
        # but normalize doesnt know about the squared-ness of the units
        # This is why e.g. 500 cm² is normalized to 5.0m², not 0.05m²
        
        # Test square meters with Unicode
        result = self.io.normalize("7000 m²")
        self.assertEqual(result.value, 7000.0)
        self.assertEqual(result.unit, 'm²')
        
        # Test other Unicode area units
        result = self.io.normalize("500 cm²")
        self.assertEqual(result.value, 5.0)
        self.assertEqual(result.unit, 'm²')
        
        result = self.io.normalize("1.5 km²")
        self.assertEqual(result.value, 1500.0)
        self.assertEqual(result.unit, 'm²')
        
        result = self.io.normalize("250 mm²")
        self.assertEqual(result.value, 0.250)
        self.assertEqual(result.unit, 'm²')

    def test_caret_area_units(self):
        """Test recognition of caret notation area units (^2)"""
        # Note: we only normalize() here, so c in cm => 1/100
        # but normalize doesnt know about the squared-ness of the units
        # This is why e.g. 500 cm² is normalized to 5.0m², not 0.05m²
        
        # Test square meters with caret notation
        result = self.io.normalize("7000 m^2")
        self.assertEqual(result.value, 7000.0)
        self.assertEqual(result.unit, 'm²')
        
        # Test other caret notation area units
        result = self.io.normalize("500 cm^2")
        self.assertEqual(result.value, 5.0)
        self.assertEqual(result.unit, 'm²')
        
        result = self.io.normalize("1.5 km^2")
        self.assertEqual(result.value, 1500)
        self.assertEqual(result.unit, 'm²')

    def test_imperial_area_units(self):
        """Test recognition of imperial area units"""
        result = self.io.normalize("100 in²")
        self.assertEqual(result.value, 100.0)
        self.assertEqual(result.unit, 'in²')
        
        result = self.io.normalize("50 ft²")
        self.assertEqual(result.value, 50.0)
        self.assertEqual(result.unit, 'ft²')
        
        result = self.io.normalize("25 yd²")
        self.assertEqual(result.value, 25.0)
        self.assertEqual(result.unit, 'yd²')

    def test_other_area_units(self):
        """Test recognition of other area units"""
        result = self.io.normalize("10 acre")
        self.assertEqual(result.value, 10.0)
        self.assertEqual(result.unit, 'acre')
        
        result = self.io.normalize("5 hectare")
        self.assertEqual(result.value, 5.0)
        self.assertEqual(result.unit, 'ha')
        
        result = self.io.normalize("0.7 hectares")
        self.assertEqual(result.value, 0.7)
        self.assertEqual(result.unit, 'ha')
        
        result = self.io.normalize("2.5 ha")
        self.assertEqual(result.value, 2.5)
        self.assertEqual(result.unit, 'ha')

    def test_area_units_with_prefixes(self):
        """Test area units with SI prefixes"""
        result = self.io.normalize("2.5k m²")
        self.assertEqual(result.value, 2500.0)
        self.assertEqual(result.unit, 'm²')

    def test_split_unit_area_units(self):
        """Test split_unit function with area units"""
        from UliEngineering.EngineerIO import UnitSplitResult
        
        # Unicode notation
        self.assertEqual(self.io.split_unit("7000 m²"), UnitSplitResult('7000', '', 'm²'))
        self.assertEqual(self.io.split_unit("500cm²"), UnitSplitResult('500c', '', 'm²'))
        # No space variant
        self.assertEqual(self.io.split_unit("7000m²"), UnitSplitResult('7000', '', 'm²'))
        
        # Caret notation
        self.assertEqual(self.io.split_unit("7000 m^2"), UnitSplitResult('7000', '', 'm²'))
        self.assertEqual(self.io.split_unit("500cm^2"), UnitSplitResult('500c', '', 'm²'))
        
        # Imperial units
        self.assertEqual(self.io.split_unit("100 in²"), UnitSplitResult('100', '', 'in²'))
        self.assertEqual(self.io.split_unit("50ft²"), UnitSplitResult('50', '', 'ft²'))
        
        # Agricultural units
        self.assertEqual(self.io.split_unit("0.7 hectares"), UnitSplitResult('0.7', '', 'ha'))

    def test_area_units_no_space(self):
        """Test area units without spaces between number and unit"""
        result = self.io.normalize("7000m²")
        self.assertEqual(result.value, 7000.0)
        self.assertEqual(result.unit, 'm²')
        
        result = self.io.normalize("500cm^2")
        self.assertEqual(result.value, 5.0)
        self.assertEqual(result.unit, 'm²')
        
        result = self.io.normalize("100in²")
        self.assertEqual(result.value, 100.0)
        self.assertEqual(result.unit, 'in²')


class TestUnitAliases(unittest.TestCase):
    def setUp(self):
        # Create an EngineerIO instance with area aliases for testing
        self.io = EngineerAreaIO.instance()

    def test_unit_alias_regex_compilation(self):
        """Test that the unit alias regex is compiled correctly"""
        self.assertIsNotNone(self.io.unit_alias_regex)
        
    def test_split_unit_with_aliases_no_space(self):
        """Test splitting units with aliases that have no spaces"""
        from UliEngineering.EngineerIO import UnitSplitResult
        
        # Test caret notation aliases
        self.assertEqual(self.io.split_unit("100m^2"), UnitSplitResult('100', '', 'm²'))
        self.assertEqual(self.io.split_unit("50cm^2"), UnitSplitResult('50c', '', 'm²'))
        self.assertEqual(self.io.split_unit("25km^2"), UnitSplitResult('25k', '', 'm²'))
        
        # Test abbreviated aliases
        self.assertEqual(self.io.split_unit("100sqm"), UnitSplitResult('100', '', 'm²'))
        self.assertEqual(self.io.split_unit("75acres"), UnitSplitResult('75', '', 'acre'))
        self.assertEqual(self.io.split_unit("30hectares"), UnitSplitResult('30', '', 'ha'))

    def test_split_unit_with_aliases_with_space(self):
        """Test splitting units with aliases that contain spaces"""
        from UliEngineering.EngineerIO import UnitSplitResult
        
        # Test full spelled out aliases with spaces
        self.assertEqual(self.io.split_unit("100 square meters"), UnitSplitResult('100', '', 'm²'))
        self.assertEqual(self.io.split_unit("50 square millimeters"), UnitSplitResult('50 m', '', 'm²'))
        self.assertEqual(self.io.split_unit("25 square kilometers"), UnitSplitResult('25 k', '', 'm²'))
        
        # Test abbreviated aliases with spaces  
        self.assertEqual(self.io.split_unit("100 sq m"), UnitSplitResult('100', '', 'm²'))
        self.assertEqual(self.io.split_unit("50 sq mm"), UnitSplitResult('50 m', '', 'm²'))
        self.assertEqual(self.io.split_unit("25 sq km"), UnitSplitResult('25 k', '', 'm²'))

    def test_split_unit_with_regex_special_characters(self):
        """Test aliases containing regex special characters"""
        from UliEngineering.EngineerIO import UnitSplitResult
        
        # Test caret (^) character - needs proper escaping in regex
        self.assertEqual(self.io.split_unit("100m^2"), UnitSplitResult('100', '', 'm²'))
        self.assertEqual(self.io.split_unit("50µm^2"), UnitSplitResult('50µ', '', 'm²'))
        self.assertEqual(self.io.split_unit("25nm^2"), UnitSplitResult('25n', '', 'm²'))
        
        # Test unicode characters (µ)
        self.assertEqual(self.io.split_unit("100 square µm"), UnitSplitResult('100 µ', '', 'm²'))
        self.assertEqual(self.io.split_unit("50 µm squared"), UnitSplitResult('50 µ', '', 'm²'))

    def test_split_unit_alias_precedence(self):
        """Test that longer aliases are matched before shorter ones"""
        from UliEngineering.EngineerIO import UnitSplitResult
        
        # "square millimeters" should match before "millimeters"
        self.assertEqual(self.io.split_unit("100 square millimeters"), UnitSplitResult('100 m', '', 'm²'))
        
        # "square meters" should match before "meters" 
        self.assertEqual(self.io.split_unit("100 square meters"), UnitSplitResult('100', '', 'm²'))

    def test_normalize_with_aliases(self):
        """Test full normalization with unit aliases"""
        # Test with spaces
        result = self.io.normalize("100 square meters")
        self.assertEqual(result.value, 100.0)
        self.assertEqual(result.unit, 'm²')
        
        # Test with caret notation
        result = self.io.normalize("50 cm^2")
        # NOTE: The direct EngineerIO instance we use here
        # (normalize() instead of normalize_area())
        # Does not handle area units properly, so 0.5 is correct here!
        self.assertEqual(result.value,0.5)
        self.assertEqual(result.unit, 'm²')
        
    def test_invalid_double_unit(self):
        with self.assertRaises(ValueError):
            # Will be aliased to "2.5kmm²"
            self.io.normalize("2.5k square millimeters")

    def test_split_unit_no_alias_fallback(self):
        """Test that non-aliased units still work correctly"""
        from UliEngineering.EngineerIO import UnitSplitResult
        
        # Test regular units that don't have aliases
        self.assertEqual(self.io.split_unit("100m²"), UnitSplitResult('100', '', 'm²'))

    def test_split_unit_no_unit(self):
        """Test that strings without units work correctly with alias regex"""
        from UliEngineering.EngineerIO import UnitSplitResult
        
        self.assertEqual(self.io.split_unit("100"), UnitSplitResult('100', '', ''))
        self.assertEqual(self.io.split_unit("50.5"), UnitSplitResult('50.5', '', ''))

if __name__ == '__main__':
    unittest.main()
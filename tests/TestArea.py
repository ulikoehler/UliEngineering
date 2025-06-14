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
        
        # Square millimeters
        assert_approx_equal(normalize_area("1.0 mm²"), 1e-6)
        assert_approx_equal(normalize_area("1.0 mm^2"), 1e-6)
        assert_approx_equal(normalize_area("1.0 square millimeter"), 1e-6)
        assert_approx_equal(normalize_area("1.0 square millimeters"), 1e-6)
        assert_approx_equal(normalize_area("1.0 sq mm"), 1e-6)
        assert_approx_equal(normalize_area("1000000 mm²"), 1.0)  # 1 m² = 1,000,000 mm²
        
        # Square centimeters
        assert_approx_equal(normalize_area("1.0 cm²"), 1e-4)
        assert_approx_equal(normalize_area("1.0 cm^2"), 1e-4)
        assert_approx_equal(normalize_area("1.0 square centimeter"), 1e-4)
        assert_approx_equal(normalize_area("1.0 square centimeters"), 1e-4)
        assert_approx_equal(normalize_area("1.0 sq cm"), 1e-4)
        assert_approx_equal(normalize_area("10000 cm²"), 1.0)  # 1 m² = 10,000 cm²
        
        # Square decimeters
        assert_approx_equal(normalize_area("1.0 dm²"), 1e-2)
        assert_approx_equal(normalize_area("1.0 dm^2"), 1e-2)
        assert_approx_equal(normalize_area("1.0 square decimeter"), 1e-2)
        assert_approx_equal(normalize_area("1.0 square decimeters"), 1e-2)
        assert_approx_equal(normalize_area("1.0 sq dm"), 1e-2)
        assert_approx_equal(normalize_area("100 dm²"), 1.0)  # 1 m² = 100 dm²
        
        # Square kilometers
        assert_approx_equal(normalize_area("1.0 km²"), 1e6)
        assert_approx_equal(normalize_area("1.0 km^2"), 1e6)
        assert_approx_equal(normalize_area("1.0 square kilometer"), 1e6)
        assert_approx_equal(normalize_area("1.0 square kilometers"), 1e6)
        assert_approx_equal(normalize_area("1.0 sq km"), 1e6)
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
        assert_approx_equal(normalize_area(f"{a4_area} cm²"), a4_area * 1e-4)

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

if __name__ == '__main__':
    unittest.main()
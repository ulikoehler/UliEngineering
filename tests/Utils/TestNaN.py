#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from UliEngineering.Utils.NaN import none_to_nan
from UliEngineering.EngineerIO.Types import NormalizeResult


class TestNoneToNaN(unittest.TestCase):
    
    def test_none_to_nan_simple_none(self):
        """Test that None is converted to NaN"""
        result = none_to_nan(None)
        self.assertTrue(np.isnan(result))
    
    def test_none_to_nan_simple_values(self):
        """Test that simple non-None values are returned unchanged"""
        test_values = [42, 3.14, -1, 0, True, False]
        for value in test_values:
            with self.subTest(value=value):
                result = none_to_nan(value)
                self.assertEqual(result, value)
    
    def test_none_to_nan_string_handling(self):
        """Test string handling - empty strings become NaN, others are stripped"""
        # Empty string should become NaN
        result = none_to_nan("")
        self.assertTrue(np.isnan(result))
        
        # Whitespace-only strings should become NaN
        result = none_to_nan("   ")
        self.assertTrue(np.isnan(result))
        
        result = none_to_nan("\t\n  ")
        self.assertTrue(np.isnan(result))
        
        # Non-empty strings should be stripped and returned
        result = none_to_nan("  hello  ")
        self.assertEqual(result, "hello")
        
        result = none_to_nan("test")
        self.assertEqual(result, "test")
        
        result = none_to_nan("123")
        self.assertEqual(result, "123")
    
    def test_none_to_nan_list_handling(self):
        """Test list handling with various elements"""
        # List with None values
        input_list = [1, None, 3, None, 5]
        result = none_to_nan(input_list)
        
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0], 1)
        self.assertTrue(np.isnan(result[1]))
        self.assertEqual(result[2], 3)
        self.assertTrue(np.isnan(result[3]))
        self.assertEqual(result[4], 5)
    
    def test_none_to_nan_mixed_list(self):
        """Test list with mixed types including strings and None"""
        input_list = [1, None, "test", "", "  ", 3.14]
        result = none_to_nan(input_list)
        
        self.assertEqual(len(result), 6)
        self.assertEqual(result[0], 1)
        self.assertTrue(np.isnan(result[1]))
        self.assertEqual(result[2], "test")
        self.assertTrue(np.isnan(result[3]))  # Empty string
        self.assertTrue(np.isnan(result[4]))  # Whitespace string
        self.assertEqual(result[5], 3.14)
    
    def test_none_to_nan_tuple_handling(self):
        """Test tuple handling (should be treated as iterable)"""
        input_tuple = (1, None, "hello", "")
        result = none_to_nan(input_tuple)
        
        # Result should be a list (since we return list for iterables)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], 1)
        self.assertTrue(np.isnan(result[1]))
        self.assertEqual(result[2], "hello")
        self.assertTrue(np.isnan(result[3]))
    
    def test_none_to_nan_nested_iterables(self):
        """Test nested iterables"""
        input_nested = [[1, None], [None, "test"], ["", 3]]
        result = none_to_nan(input_nested)
        
        self.assertEqual(len(result), 3)
        
        # First nested list
        self.assertEqual(result[0][0], 1)
        self.assertTrue(np.isnan(result[0][1]))
        
        # Second nested list
        self.assertTrue(np.isnan(result[1][0]))
        self.assertEqual(result[1][1], "test")
        
        # Third nested list
        self.assertTrue(np.isnan(result[2][0]))  # Empty string
        self.assertEqual(result[2][1], 3)
    
    def test_none_to_nan_empty_list(self):
        """Test empty list handling"""
        result = none_to_nan([])
        self.assertEqual(result, [])
    
    def test_none_to_nan_numpy_array(self):
        """Test numpy array handling"""
        input_array = np.array([1, 2, 3])
        result = none_to_nan(input_array)
        
        # NumPy arrays are iterable, so should be converted to list
        self.assertIsInstance(result, list)
        self.assertEqual(result, [1, 2, 3])
    
    def test_none_to_nan_string_edge_cases(self):
        """Test edge cases for string handling"""
        # Various whitespace characters
        test_cases = [
            ("\r\n", True),  # Should become NaN
            ("\t", True),    # Should become NaN
            ("  \n  ", True), # Should become NaN
            ("0", False),     # Should remain "0"
            ("  0  ", False), # Should become "0" (stripped)
            ("False", False), # Should remain "False"
        ]
        
        for input_str, should_be_nan in test_cases:
            with self.subTest(input_str=repr(input_str)):
                result = none_to_nan(input_str)
                if should_be_nan:
                    self.assertTrue(np.isnan(result))
                else:
                    self.assertEqual(result, input_str.strip())
    
    def test_none_to_nan_complex_types(self):
        """Test with complex number and other special types"""
        # Complex numbers
        complex_val = 3 + 4j
        result = none_to_nan(complex_val)
        self.assertEqual(result, complex_val)
        
        # Boolean values
        self.assertEqual(none_to_nan(True), True)
        self.assertEqual(none_to_nan(False), False)
        
        # Zero values
        self.assertEqual(none_to_nan(0), 0)
        self.assertEqual(none_to_nan(0.0), 0.0)
    
if __name__ == '__main__':
    unittest.main()

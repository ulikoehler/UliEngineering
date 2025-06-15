#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
from numpy.testing import assert_allclose, assert_approx_equal
from UliEngineering.EngineerIO import *
from UliEngineering.Exceptions import EngineerIOException
from UliEngineering.Units import *
from UliEngineering.EngineerIO import _format_with_suffix, SplitResult, UnitSplitResult, NormalizeResult, _default_units, _area_unit_aliases, _area_units
from parameterized import parameterized
import functools
import numpy as np
import unittest
import pytest

class TestEngineerIO(unittest.TestCase):
    def setUp(self):
        self.io = EngineerIO()

    def test_normalize_interpunctation(self):
        for suffix in ["", "k", " kV", "V/√Hz", "µV"]:
            self.assertEqual(normalize_interpunctation("1234" + suffix), '1234' + suffix)
            self.assertEqual(normalize_interpunctation("123.4" + suffix), '123.4' + suffix)
            self.assertEqual(normalize_interpunctation("123,4" + suffix), '123.4' + suffix)
            self.assertEqual(normalize_interpunctation("1,234.5" + suffix), '1234.5' + suffix)
            self.assertEqual(normalize_interpunctation("1.234,5" + suffix), '1234.5' + suffix)
            self.assertEqual(normalize_interpunctation("1.234,5" + suffix), '1234.5' + suffix)
        self.assertEqual(normalize_interpunctation(""), "")

    def test_split_input(self):
        self.assertEqual(self.io.split_input("1234"), SplitResult('', '1234', '', '', ''))
        self.assertEqual(self.io.split_input("1234k"), SplitResult('', '1234', 'k', '', ''))
        self.assertEqual(self.io.split_input("1234kΩ"), SplitResult('', '1234', 'k', '', 'Ω'))
        self.assertEqual(self.io.split_input("1.234kΩ"), SplitResult('', '1.234', 'k', '', 'Ω'))
        self.assertEqual(self.io.split_input("1,234kΩ"), SplitResult('', '1.234', 'k', '', 'Ω'))
        self.assertEqual(self.io.split_input("1,234.56kΩ"), SplitResult('', '1234.56', 'k', '', 'Ω'))
        self.assertEqual(self.io.split_input("1k234"), SplitResult('', '1.234', 'k', '', ''))
        self.assertEqual(self.io.split_input("1k234Ω"), SplitResult('', '1.234', 'k', '', 'Ω'))
        self.assertEqual(self.io.split_input("1,234.56Ω"), SplitResult('', '1234.56', '', '', 'Ω'))
        self.assertEqual(self.io.split_input("1A"), SplitResult('', '1', '', '', 'A'))
        self.assertEqual(self.io.split_input("0Ω"), SplitResult('', '0', '', '', 'Ω'))
        self.assertEqual(self.io.split_input("±5%"), SplitResult('±', '5', '', '', '%'))
        self.assertEqual(self.io.split_input("± 5%"), SplitResult('±', '5', '', '', '%'))
        self.assertEqual(self.io.split_input("± 5 %"), SplitResult('±', '5', '', '', '%'))
        self.assertEqual(self.io.split_input("0 Ω"), SplitResult('', '0', '', '', 'Ω'))
        self.assertEqual(self.io.split_input("1"), SplitResult('', '1', '', '', ''))
        self.assertEqual(self.io.split_input("1k234 Ω"), SplitResult('', '1.234', 'k', '', 'Ω'))
        self.assertEqual(self.io.split_input("-1,234.56kΩ"), SplitResult('', '-1234.56', 'k', '', 'Ω'))
        self.assertEqual(self.io.split_input("-1e3kΩ"), SplitResult('', '-1e3', 'k', '', 'Ω'))
        self.assertEqual(self.io.split_input("1e-3kΩ"), SplitResult('', '1e-3', 'k', '', 'Ω'))
        self.assertEqual(self.io.split_input("-4e6nA"), SplitResult('', '-4e6', 'n', '', 'A'))
        self.assertEqual(self.io.split_input("3.2 MHz"), SplitResult('', '3.2', 'M', '', 'Hz'))
        self.assertEqual(self.io.split_input("3.2 °C"), SplitResult('', '3.2', '', '°', 'C'))
        self.assertEqual(self.io.split_input("3k2 °C"), SplitResult('', '3.2', 'k', '°', 'C'))
        self.assertEqual(self.io.split_input("100 mV"), SplitResult('', '100', 'm', '', 'V'))
        self.assertEqual(self.io.split_input("Δ3.2 MHz"), SplitResult('Δ', '3.2', 'M', '', 'Hz'))
        self.assertEqual(self.io.split_input("3.20 €"), SplitResult('', '3.20', '', '', '€'))
        self.assertEqual(self.io.split_input("0.000014 €"), SplitResult('', '0.000014', '', '', '€'))

    @parameterized.expand([("1,234.56kfA",),
                    ("1.23k45A",),
                    ("1,234.56kfA",),
                    ("foobar",),
                    (None,),
                    (["1.2 J", "foobar"],),
                    ("1k2 MA",),
                    ("1.2kkA",),
                    ("1k2kA",),
                    ("1k2.4",),
                    ("k2",),
                    ("A",),
                    ("k",),
                    ("ky",),
                    ("kA",),
                    ("kfA",),
                    ("AA",),
                    ("kΔ",),
                    ("Δ",),
                    ("AΔ",),
                    ("ΔA",),
                    ("ΔAΔ",),
                    (" ",),
                    ("",)])
    def test_normalize_numeric_invalid(self, s):
        with self.assertRaises((EngineerIOException, ValueError)):
            print(self.io.normalize_numeric(s))

    def test_split_unit(self):
        self.assertEqual(self.io.split_unit("1234"), UnitSplitResult('1234', '', ''))
        self.assertEqual(self.io.split_unit("1234k"), UnitSplitResult('1234k', '', ''))
        self.assertEqual(self.io.split_unit("1234kΩ"), UnitSplitResult('1234k', '', 'Ω'))
        self.assertEqual(self.io.split_unit("1.234kΩ"), UnitSplitResult('1.234k', '', 'Ω'))
        self.assertEqual(self.io.split_unit("1,234kΩ"), UnitSplitResult('1,234k', '', 'Ω'))
        self.assertEqual(self.io.split_unit("1,234.56kΩ"), UnitSplitResult('1,234.56k', '', 'Ω'))
        self.assertEqual(self.io.split_unit("1k234"), UnitSplitResult('1k234', '', ''))
        self.assertEqual(self.io.split_unit("1k234Ω"), UnitSplitResult('1k234', '',  'Ω'))
        self.assertEqual(self.io.split_unit("1,234.56Ω"), UnitSplitResult('1,234.56', '', 'Ω'))
        self.assertEqual(self.io.split_unit("1A"), UnitSplitResult('1', '', 'A'))
        self.assertEqual(self.io.split_unit("1"), UnitSplitResult('1', '', ''))
        self.assertEqual(self.io.split_unit("1k234 Ω"), UnitSplitResult('1k234', '', 'Ω'))
        self.assertEqual(self.io.split_unit("-1,234.56kΩ"), UnitSplitResult('-1,234.56k', '', 'Ω'))
        self.assertEqual(self.io.split_unit("-1e3kΩ"), UnitSplitResult('-1e3k', '', 'Ω'))
        self.assertEqual(self.io.split_unit("1e-3kΩ"), UnitSplitResult('1e-3k', '', 'Ω'))
        self.assertEqual(self.io.split_unit("-4e6nA"), UnitSplitResult('-4e6n', '', 'A'))
        self.assertEqual(self.io.split_unit("3.2 MHz"), UnitSplitResult('3.2 M', '', 'Hz'))
        self.assertEqual(self.io.split_unit("3.2 °C"), UnitSplitResult('3.2', '°', 'C'))
        self.assertEqual(self.io.split_unit("3k2 °C"), UnitSplitResult('3k2', '°', 'C'))
        self.assertEqual(self.io.split_unit("100 mV"), UnitSplitResult('100 m', '', 'V'))
        self.assertEqual(self.io.split_unit("50 °C/W"), UnitSplitResult('50', '°', 'C/W'))
        self.assertEqual(self.io.split_unit(""), UnitSplitResult('', '', ''))

    def test_normalize(self):
        self.assertEqual(self.io.normalize("100 kΩ"), NormalizeResult(value=1e5, unit="Ω", prefix_multiplier=1e3, original_number=100))
        # Test with bytes input
        self.assertEqual(self.io.normalize("100 kΩ".encode("utf8")), NormalizeResult(value=1e5, unit="Ω", prefix_multiplier=1e3, original_number=100))

    def test_format_with_suffix(self):
        self.assertEqual(_format_with_suffix(1.01, "A"), '1.01 A')
        self.assertEqual(_format_with_suffix(1, "A"), '1.00 A')
        self.assertEqual(_format_with_suffix(101, "A"), '101 A')
        self.assertEqual(_format_with_suffix(99.9, "A"), '99.9 A')
        self.assertEqual(_format_with_suffix(1000.0, ""), '1000')
        # More significant digits
        self.assertEqual(_format_with_suffix(1.01, "A", significant_digits=4), '1.010 A')

    def test_format(self):
        self.assertEqual(self.io.format(1.0e-15, "V"), '1.00 fV')
        self.assertEqual(self.io.format(234.6789e-3, "V"), '235 mV')
        self.assertEqual(self.io.format(234.6789, "V"), '235 V')
        self.assertEqual(self.io.format(2345.6789, "V"), '2.35 kV')
        self.assertEqual(self.io.format(2345.6789e6, "V"), '2.35 GV')
        self.assertEqual(self.io.format(2345.6789e12, "V"), '2.35 EV')
        self.assertEqual(self.io.format(2.3456789e-6, "V"), '2.35 µV')
        self.assertEqual(self.io.format(2.3456789e-6, "°C"), '2.35 µ°C')
        self.assertEqual(self.io.format(-2.3456789e-6, "°C"), '-2.35 µ°C')
        self.assertEqual(self.io.format(np.nan, "V"), '- V')
        # More significant digits
        self.assertEqual(self.io.format(-2.3456789e-6, "°C", 4), '-2.346 µ°C')
        self.assertEqual(self.io.format(-2.3456789e-6, "°C", 5), '-2.3457 µ°C')
        self.assertEqual(self.io.format(-2.3456789e-6, "°C", 2), '-2.3 µ°C')
        
        
    def test_format_negative(self):
        self.assertEqual(self.io.format(-1.0e-15, "V"), '-1.00 fV')
        self.assertEqual(self.io.format(-234.6789e-3, "V"), '-235 mV')
        self.assertEqual(self.io.format(-234.6789, "V"), '-235 V')
        self.assertEqual(self.io.format(-2345.6789, "V"), '-2.35 kV')
        self.assertEqual(self.io.format(-2345.6789e6, "V"), '-2.35 GV')
        self.assertEqual(self.io.format(-2345.6789e12, "V"), '-2.35 EV')
        self.assertEqual(self.io.format(-2.3456789e-6, "V"), '-2.35 µV')
        self.assertEqual(self.io.format(-2.3456789e-6, "°C"), '-2.35 µ°C')
        self.assertEqual(self.io.format(np.nan, "V"), '- V')
        # More significant digits
        self.assertEqual(self.io.format(-2.3456789e-6, "°C", 4), '-2.346 µ°C')
        self.assertEqual(self.io.format(-2.3456789e3, "°C", 4), '-2.346 k°C')
        self.assertEqual(self.io.format(-2.3456789e-6, "°C", 5), '-2.3457 µ°C')
        self.assertEqual(self.io.format(-2.3456789e3, "°C", 5), '-2.3457 k°C')
        self.assertEqual(self.io.format(-2.3456789e-6, "°C", 2), '-2.3 µ°C')
        self.assertEqual(self.io.format(-100, "°C", 1), '-100 °C')

    def test_format_invalid(self):
        with self.assertRaises(ValueError):
            self.io.format(1.0e-25, "V")

    def test_format_no_unit(self):
        self.assertEqual(self.io.format(1.999999, ""), '2.00')
        self.assertEqual(self.io.format(1.999999, None), '2.00')

    def testRounding(self):
        self.assertEqual(self.io.format(1.999999, ""), '2.00')
        self.assertEqual(self.io.format(19.99999, ""), '20.0')
        self.assertEqual(self.io.format(199.9999, ""), '200')

    def testIsValidUnitPrefix(self):
        for c in "fpnuµmkMGT":
            self.assertIn(c, self.io.unit_prefix_exp_map)

    def test_exp_unit_prefix_map(self):
        self.assertEqual(self.io.unit_prefix_exp_map["f"], -15)
        self.assertEqual(self.io.unit_prefix_exp_map["k"], 3)
        self.assertEqual(self.io.unit_prefix_exp_map["u"], -6)
        self.assertEqual(self.io.unit_prefix_exp_map["µ"], -6)
        self.assertEqual(self.io.unit_prefix_exp_map["T"], 12)
        self.assertEqual(self.io.unit_prefix_exp_map[""], 0)
        # Check "in" operator
        self.assertIn("k", self.io.unit_prefix_exp_map)
        # Invalid unit_prefix_exp_map
        self.assertNotIn("B", self.io.unit_prefix_exp_map)

    def test_exp_unit_prefix_map2(self):
        self.assertEqual("", self.io.exp_unit_prefix_map[0])
        self.assertEqual("k", self.io.exp_unit_prefix_map[1])
        self.assertEqual("M", self.io.exp_unit_prefix_map[2])
        self.assertEqual("n", self.io.exp_unit_prefix_map[-3])

    def test_normalize_numeric_safe(self):
        self.assertEqual(self.io.normalize_numeric_safe(1.25), 1.25)
        self.assertEqual(self.io.normalize_numeric_safe("1.25"), 1.25)
        self.assertEqual(self.io.normalize_numeric_safe("1.25 V"), 1.25)
        self.assertEqual(self.io.normalize_numeric_safe("1k25 V"), 1250.0)
        self.assertEqual(self.io.normalize_numeric_safe(b"1k25 V"), 1250.0)
        assert_allclose(self.io.normalize_numeric_safe(["1k25 V", "4.25 A"]), [1250.0, 4.25])
        # Invalid inputs and partially invalid inputs
        self.assertIsNone(self.io.normalize_numeric_safe("foobar"))
        assert_allclose(self.io.normalize_numeric_safe(["foobar", "1.2 J"]), [np.nan, 1.2])

    def test_normalize_numeric(self):
        self.assertEqual(self.io.normalize_numeric(1.25), 1.25)
        self.assertEqual(self.io.normalize_numeric("1.25"), 1.25)
        self.assertEqual(self.io.normalize_numeric("1.25 V"), 1.25)
        self.assertEqual(self.io.normalize_numeric("1k25 V"), 1250.0)
        self.assertEqual(self.io.normalize_numeric(b"1k25 V"), 1250.0)
        assert_allclose(self.io.normalize_numeric(["1k25 V", "4.25 A"]), np.asarray([1250.0, 4.25]))

    def test_normalize_numeric_verify_unit(self):
        self.assertEqual(self.io.normalize_numeric_verify_unit(1.25, Unit("V")), 1.25)
        self.assertEqual(self.io.normalize_numeric_verify_unit("1.25", Unit("V")), 1.25)
        self.assertEqual(self.io.normalize_numeric_verify_unit("1.25 V", Unit("V")), 1.25)
        self.assertEqual(self.io.normalize_numeric_verify_unit("1k25 V", Unit("V")), 1250.0)
        self.assertEqual(self.io.normalize_numeric_verify_unit(b"1k25 V", Unit("V")), 1250.0)
        assert_allclose(self.io.normalize_numeric_verify_unit(["1k25 V", "4.25 V"], Unit("V")), np.asarray([1250.0, 4.25]))

    @parameterized.expand([
        ("1.25 A", Unit("V")),
        ("1.25 kA", Unit("V")),
        ("1.25 V", Unit("A")),
    ])
    def test_normalize_numeric_verify_unit_raises(self, s, reference):
        # Tests that should not verify any unite
        with self.assertRaises(InvalidUnitInContextException):
            self.io.normalize_numeric_verify_unit(s, reference)

    def test_safe_normalize(self):
        self.assertEqual(self.io.safe_normalize("1.25 kV"), second=NormalizeResult(value=1250, unit='V', original_number=1.25, prefix_multiplier=1000))
        self.assertIsNone(self.io.safe_normalize("1x25"))

    # Just basic tests for autoFormat. Specific tests in other modules that have annotated functions

    def testAutoFormatValid(self):
        @returns_unit("V")
        def testfn(n=1.0): return n
        self.assertEqual(self.io.auto_format(testfn), "1.00 V")
        # Test functools.partial() behaviour
        testfn2 = functools.partial(testfn, n=2.0)
        self.assertEqual(self.io.auto_format(testfn2), "2.00 V")
        # Test nested functools.partial() behaviour
        testfn3 = functools.partial(testfn2, n=3.0)
        self.assertEqual(self.io.auto_format(testfn3), "3.00 V")

    def testAutoFormatInvalid1(self):
        with self.assertRaises(UnannotatedReturnValueError):
            self.io.auto_format(lambda: 0) # Callable but not annotated

    def testAutoFormatInvalid2(self):
        with self.assertRaises(UnannotatedReturnValueError):
            self.io.auto_format(None) # Not even callable
            
    def testAutoFormatInvalid3(self):
        with self.assertRaises(UnannotatedReturnValueError):
            self.io.auto_format(7.5) # Not even callable

    @pytest.mark.filterwarnings("ignore: divide by zero encountered in log10")
    def test_auto_suffix_1d(self):
        arr = np.arange(-4., 5., .5)
        self.assertEqual(self.io.auto_suffix_1d(arr), (1., ""))
        arr = 1e-3 * np.arange(-4., 5., .5)
        self.assertEqual(self.io.auto_suffix_1d(arr), (1e3, "m"))
        arr = 1e9 * np.arange(-4., 5., .5)
        self.assertEqual(self.io.auto_suffix_1d(arr), (1e-9, "G"))
        arr = np.arange(1000., 2000., 5)
        self.assertEqual(self.io.auto_suffix_1d(arr), (1e-3, "k"))
        # Test out of limits
        arr = 1e-40 * np.arange(-4., 5., .5)
        self.assertEqual(self.io.auto_suffix_1d(arr), (1e24, "y"))
        arr = 1e40 * np.arange(-4., 5., .5)
        self.assertEqual(self.io.auto_suffix_1d(arr), (1e-21, "Y"))

    def test_special_units(self):
        """
        Test ppm, ppb and %
        """
        # %
        assert_approx_equal(self.io.normalize_numeric_safe("1%"), 0.01)
        assert_approx_equal(self.io.normalize_numeric_safe("5%"), 0.05)
        assert_approx_equal(self.io.normalize_numeric_safe("-5%"), -0.05)
        assert_approx_equal(self.io.normalize_numeric_safe("1.25"), 1.25)
        assert_approx_equal(self.io.normalize_numeric_safe("125 %"), 1.25)
        assert_approx_equal(self.io.normalize_numeric_safe("125.0 %"), 1.25)
        assert_approx_equal(self.io.normalize_numeric_safe("1.25 %"), 0.0125)
        # ppm
        assert_approx_equal(self.io.normalize_numeric_safe("1.25 ppm"), 1.25e-6)
        assert_approx_equal(self.io.normalize_numeric_safe("12.5 ppm"), 1.25e-5)
        assert_approx_equal(self.io.normalize_numeric_safe("12.5ppm"), 1.25e-5)
        # ppb
        assert_approx_equal(self.io.normalize_numeric_safe("1.25 ppb"), 1.25e-9)
        assert_approx_equal(self.io.normalize_numeric_safe("12.5 ppb"), 1.25e-8)
        assert_approx_equal(self.io.normalize_numeric_safe("12.5ppb"), 1.25e-8)

    def test_time_units(self):
        """
        Test time units using the normal numeric parser.
        This does not use the timerange parser.
        """
        assert_approx_equal(self.io.normalize_numeric_safe("1.25 s"), 1.25)
        assert_approx_equal(self.io.normalize_numeric_safe("1.25 min"), 1.25)
        assert_approx_equal(self.io.normalize_numeric_safe("1.25 h"), 1.25)


class TestNormalizeTimespan(unittest.TestCase):
    def setUp(self):
        self.io = EngineerIO()

    def test_float_int_input(self):
        assert_approx_equal(self.io.normalize_timespan(1), 1)
        assert_approx_equal(self.io.normalize_timespan(1.25), 1.25)
    
    def test_numpy_input(self):
        assert_approx_equal(self.io.normalize_timespan(np.float64(1)), 1)
        assert_approx_equal(self.io.normalize_timespan(np.float64(1.25)), 1.25)

    def test_numpy_2d_input(self):
        assert_allclose(self.io.normalize_timespan(np.asarray([[1, 2], [3, 4]])), [[1,2], [3,4]])
        assert_allclose(self.io.normalize_timespan(np.asarray([[1.25, 2.25], [3.25, 4.25]])), [[1.25,2.25], [3.25,4.25]])

    def test_numpy_array_input(self):
        print(self.io.normalize_timespan(np.asarray([1, 2, 3])))
        assert_allclose(self.io.normalize_timespan(np.asarray([1, 2, 3])), [1,2,3])
        assert_allclose(self.io.normalize_timespan(np.asarray([1.25, 2.25, 3.25])), [1.25,2.25,3.25])

    def test_list_input_1d(self):
        assert_allclose(self.io.normalize_timespan([1, 2, 3]), [1,2,3])
        assert_allclose(self.io.normalize_timespan([1.25, 2.25, 3.25]), [1.25,2.25,3.25])
    
    def test_list_input_2d(self):
        assert_allclose(self.io.normalize_timespan([[1, 2], [3, 4]]), [[1,2], [3,4]])
        assert_allclose(self.io.normalize_timespan([[1.25, 2.25], [3.25, 4.25]]), [[1.25,2.25], [3.25,4.25]])

    def test_semicomplete_decimal_input(self):
        assert_approx_equal(self.io.normalize_timespan("1."), 1.)
        assert_approx_equal(self.io.normalize_timespan("1.s"), 1.)
        assert_approx_equal(self.io.normalize_timespan(".0 s"), 0.0)
        assert_approx_equal(self.io.normalize_timespan("1.h"), 3600)

    def test_normalize_timespan(self):
        # Test behavior with no unit
        assert_approx_equal(self.io.normalize_timespan("1.25"), 1.25)
        assert_approx_equal(self.io.normalize_timespan("1.25 s"), 1.25)
        assert_approx_equal(self.io.normalize_timespan("1.25 min"), 1.25 * 60)
        assert_approx_equal(self.io.normalize_timespan("1.25 h"), 1.25 * 3600)
        assert_approx_equal(self.io.normalize_timespan("1.25 d"), 1.25 * 86400)
        assert_approx_equal(self.io.normalize_timespan("1.25 w"), 1.25 * 86400 * 7)
        assert_approx_equal(self.io.normalize_timespan("1.25 months"), 1.25*31556952/12)
        assert_approx_equal(self.io.normalize_timespan("1.25 y"), 1.25*31556952)
        # Test negative values
        assert_approx_equal(self.io.normalize_timespan("-1.25"), -1.25)
        assert_approx_equal(self.io.normalize_timespan("-1.25 s"), -1.25)
        assert_approx_equal(self.io.normalize_timespan("-1.25 min"), -1.25 * 60)
        assert_approx_equal(self.io.normalize_timespan("-1.25 h"), -1.25 * 3600)

    def test_normalize_timespan_subseconds(self):
        # Test behavior with no unit
        assert_approx_equal(self.io.normalize_timespan("1.25 ms"), 1.25e-3)
        assert_approx_equal(self.io.normalize_timespan("1.25 µs"), 1.25e-6)
        assert_approx_equal(self.io.normalize_timespan("1.25 ns"), 1.25e-9)
        assert_approx_equal(self.io.normalize_timespan("1.25 ps"), 1.25e-12)
        assert_approx_equal(self.io.normalize_timespan("1.25 fs"), 1.25e-15)
        assert_approx_equal(self.io.normalize_timespan("1.25 as"), 1.25e-18)
        # Test negative values
        assert_approx_equal(self.io.normalize_timespan("-1.25 ms"), -1.25e-3)
        assert_approx_equal(self.io.normalize_timespan("-1.25 µs"), -1.25e-6)
        assert_approx_equal(self.io.normalize_timespan("-1.25 ns"), -1.25e-9)
        assert_approx_equal(self.io.normalize_timespan("-1.25 ps"), -1.25e-12)
        assert_approx_equal(self.io.normalize_timespan("-1.25 fs"), -1.25e-15)
        assert_approx_equal(self.io.normalize_timespan("-1.25 as"), -1.25e-18)


class TestAllSuffixes(unittest.TestCase):
    def setUp(self):
        self.io = EngineerIO()

    def test_all_suffixes_basic(self):
        """Test basic functionality with simple strings"""
        self.assertEqual(self.io.all_suffixes("abc123"), ["3", "23", "123", "c123", "bc123", "abc123"])
        self.assertEqual(self.io.all_suffixes("test"), ["t", "st", "est", "test"])
        
    def test_all_suffixes_single_char(self):
        """Test with single character strings"""
        self.assertEqual(self.io.all_suffixes("a"), ["a"])
        self.assertEqual(self.io.all_suffixes("1"), ["1"])
        
    def test_all_suffixes_empty_string(self):
        """Test with empty string"""
        self.assertEqual(self.io.all_suffixes(""), [])
        
    def test_all_suffixes_numeric(self):
        """Test with numeric strings"""
        self.assertEqual(self.io.all_suffixes("1234"), ["4", "34", "234", "1234"])
        self.assertEqual(self.io.all_suffixes("42"), ["2", "42"])
        
    def test_all_suffixes_with_units(self):
        """Test with engineering notation strings"""
        self.assertEqual(self.io.all_suffixes("100kΩ"), ["Ω", "kΩ", "0kΩ", "00kΩ", "100kΩ"])
        self.assertEqual(self.io.all_suffixes("1.5MHz"), ["z", "Hz", "MHz", "5MHz", ".5MHz", "1.5MHz"])
        
    def test_all_suffixes_special_chars(self):
        """Test with special characters"""
        self.assertEqual(self.io.all_suffixes("a-b_c"), ["c", "_c", "b_c", "-b_c", "a-b_c"])
        self.assertEqual(self.io.all_suffixes("x.y"), ["y", ".y", "x.y"])
        
    def test_all_suffixes_unicode(self):
        """Test with unicode characters"""
        self.assertEqual(self.io.all_suffixes("αβγ"), ["γ", "βγ", "αβγ"])
        self.assertEqual(self.io.all_suffixes("1µV"), ["V", "µV", "1µV"])


class TestUnitAliases(unittest.TestCase):
    def setUp(self):
        # Create an EngineerIO instance with area aliases for testing
        self.io = EngineerIO(
            units=_area_units(),
            unit_aliases=_area_unit_aliases()
        )

    def test_unit_alias_regex_compilation(self):
        """Test that the unit alias regex is compiled correctly"""
        self.assertIsNotNone(self.io.unit_alias_regex)
        
    def test_split_unit_with_aliases_no_space(self):
        """Test splitting units with aliases that have no spaces"""
        # Test caret notation aliases
        self.assertEqual(self.io.split_unit("100m^2"), UnitSplitResult('100', '', 'm²'))
        self.assertEqual(self.io.split_unit("50cm^2"), UnitSplitResult('50c', '', 'm²'))
        self.assertEqual(self.io.split_unit("25km^2"), UnitSplitResult('25k', '', 'm²'))
        
        # Test abbreviated aliases
        self.assertEqual(self.io.split_unit("100sqm"), UnitSplitResult('100', '', 'm²'))
        self.assertEqual(self.io.split_unit("75acres"), UnitSplitResult('75', '', 'acre'))
        self.assertEqual(self.io.split_unit("30hectares"), UnitSplitResult('30', '', 'hectare'))

    def test_split_unit_with_aliases_with_space(self):
        """Test splitting units with aliases that contain spaces"""
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
        # Test caret (^) character - needs proper escaping in regex
        self.assertEqual(self.io.split_unit("100m^2"), UnitSplitResult('100', '', 'm²'))
        self.assertEqual(self.io.split_unit("50µm^2"), UnitSplitResult('50µ', '', 'm²'))
        self.assertEqual(self.io.split_unit("25nm^2"), UnitSplitResult('25n', '', 'm²'))
        
        # Test unicode characters (µ)
        self.assertEqual(self.io.split_unit("100 square µm"), UnitSplitResult('100 µ', '', 'm²'))
        self.assertEqual(self.io.split_unit("50 µm squared"), UnitSplitResult('50 µ', '', 'm²'))

    def test_split_unit_alias_precedence(self):
        """Test that longer aliases are matched before shorter ones"""
        # "square millimeters" should match before "millimeters"
        self.assertEqual(self.io.split_unit("100 square millimeters"), UnitSplitResult('100m', '', 'm²'))
        
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
        self.assertEqual(result.value, 50.0)
        self.assertEqual(result.unit, 'cm²')
        
        # Test with prefixes and aliases
        result = self.io.normalize("2.5k square millimeters")
        self.assertEqual(result.value, 2500.0)
        self.assertEqual(result.unit, 'mm²')

    def test_split_unit_no_alias_fallback(self):
        """Test that non-aliased units still work correctly"""
        # Test regular units that don't have aliases
        self.assertEqual(self.io.split_unit("100V"), UnitSplitResult('100', '', 'V'))
        self.assertEqual(self.io.split_unit("50 Hz"), UnitSplitResult('50', '', 'Hz'))
        self.assertEqual(self.io.split_unit("25Ω"), UnitSplitResult('25', '', 'Ω'))

    def test_split_unit_no_unit(self):
        """Test that strings without units work correctly with alias regex"""
        self.assertEqual(self.io.split_unit("100"), UnitSplitResult('100', '', ''))
        self.assertEqual(self.io.split_unit("50.5"), UnitSplitResult('50.5', '', ''))


class TestAreaUnits(unittest.TestCase):
    def setUp(self):
        # Create an EngineerIO instance with area units included
        self.io = EngineerIO.area_instance

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
        
    def test_generate_unit_alias_pattern_with_unicode(self):
        """Test unit alias pattern generation with unicode characters"""
        aliases = {
            'µm squared': 'µm²',
            'degrees celsius': '°C',
            'chinese yuan': '元',
            'ohm resistance': 'Ω'
        }
        io = EngineerIO(units=set(), unit_aliases=aliases)
        pattern = io._generate_unit_alias_pattern()
        # Should properly handle unicode characters
        self.assertIn(re.escape('µm squared'), pattern)
        self.assertIn(re.escape('degrees celsius'), pattern)
        self.assertIn(re.escape('chinese yuan'), pattern)
        self.assertIn(re.escape('ohm resistance'), pattern)
        
    def test_generate_unit_alias_pattern_with_regex_special_chars(self):
        """Test unit alias pattern generation with regex special characters"""
        aliases = {
            'volt+ampere': 'VA',
            'power*time': 'Pt',
            'resistance[ohm]': 'R',
            'frequency(hertz)': 'f',
            'voltage.rms': 'Vrms',
            'current^2': 'I²'
        }
        io = EngineerIO(units=set(), unit_aliases=aliases)
        pattern = io._generate_unit_alias_pattern()
        # Should escape regex special characters
        self.assertIn(r'volt\+ampere', pattern)
        self.assertIn(r'power\*time', pattern)
        self.assertIn(r'resistance\[ohm\]', pattern)
        self.assertIn(r'frequency\(hertz\)', pattern)
        self.assertIn(r'voltage\.rms', pattern)
        self.assertIn(r'current\^2', pattern)
        
    def test_generate_unit_alias_pattern_length_sorting(self):
        """Test that aliases are sorted by length (longest first)"""
        aliases = {
            'V': 'volt',
            'volt': 'V',
            'square millimeter': 'mm²'
        }
        io = EngineerIO(units=set(), unit_aliases=aliases)
        pattern = io._generate_unit_alias_pattern()
        # Should have longest first: square millimeter, then volt, then V
        parts = pattern[1:-2].split('|')  # Remove outer parentheses and $
        self.assertEqual(parts[0], re.escape('square millimeter'))
        self.assertEqual(parts[1], re.escape('volt'))
        self.assertEqual(parts[2], re.escape('V'))
        
    def test_generate_unit_alias_pattern_empty(self):
        """Test unit alias pattern generation with empty aliases dict"""
        io = EngineerIO(units=set(), unit_aliases={})
        pattern = io._generate_unit_alias_pattern()
        self.assertIsNone(pattern)
        
    def test_generate_unit_alias_pattern_with_spaces(self):
        """Test unit alias pattern generation with spaces in aliases"""
        aliases = {
            'square meter': 'm²',
            'cubic centimeter': 'cm³',
            'degrees per second': '°/s',
            'meters per second': 'm/s'
        }
        io = EngineerIO(units=set(), unit_aliases=aliases)
        pattern = io._generate_unit_alias_pattern()
        # Should properly handle spaces in aliases
        self.assertIn(re.escape('square meter'), pattern)
        self.assertIn(re.escape('cubic centimeter'), pattern)
        self.assertIn(re.escape('degrees per second'), pattern)
        self.assertIn(re.escape('meters per second'), pattern)
        
    def test_pattern_compilation_with_fake_units(self):
        """Test that generated patterns compile correctly with fake units"""
        fake_units = {'testunit', 'fakeΩ', 'µtest', 'unit[1]', 'test+volt'}
        fake_aliases = {
            'fake square meter': 'm²',
            'test µ unit': 'µU',
            'unit(special)': 'US'
        }
        
        io = EngineerIO(units=fake_units, unit_aliases=fake_aliases)
        
        # Both regexes should compile without errors
        self.assertIsNotNone(io.units_regex)
        self.assertIsNotNone(io.unit_alias_regex)
        
        # Test that they can match their respective patterns
        self.assertIsNotNone(io.units_regex.search('100testunit'))
        self.assertIsNotNone(io.units_regex.search('50fakeΩ'))
        self.assertIsNotNone(io.unit_alias_regex.search('100 fake square meter'))
        self.assertIsNotNone(io.unit_alias_regex.search('50 test µ unit'))

    def test_pattern_matching_precedence_with_fake_data(self):
        """Test that longer patterns are matched first with fake data"""
        fake_units = {'A', 'ABC', 'ABCDEF'}
        fake_aliases = {
            'test': 'T',
            'test unit': 'TU', 
            'test unit long': 'TUL'
        }
        
        io = EngineerIO(units=fake_units, unit_aliases=fake_aliases)
        
        # Longest unit should match first
        match = io.units_regex.search('100ABCDEF')
        self.assertEqual(match.group(1), 'ABCDEF')
        
        # Longest alias should match first
        match = io.unit_alias_regex.search('100 test unit long')
        self.assertEqual(match.group(1), 'test unit long')


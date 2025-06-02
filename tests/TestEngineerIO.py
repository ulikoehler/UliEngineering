#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_allclose, assert_approx_equal
from UliEngineering.EngineerIO import *
from UliEngineering.Units import *
from UliEngineering.EngineerIO import _format_with_suffix, SplitResult, UnitSplitResult, NormalizeResult
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
        self.assertTupleEqual(self.io.split_input("1234"), SplitResult('', '1234', '', '', ''))
        self.assertTupleEqual(self.io.split_input("1234k"), SplitResult('', '1234', 'k', '', ''))
        self.assertTupleEqual(self.io.split_input("1234kΩ"), SplitResult('', '1234', 'k', '', 'Ω'))
        self.assertTupleEqual(self.io.split_input("1.234kΩ"), SplitResult('', '1.234', 'k', '', 'Ω'))
        self.assertTupleEqual(self.io.split_input("1,234kΩ"), SplitResult('', '1.234', 'k', '', 'Ω'))
        self.assertTupleEqual(self.io.split_input("1,234.56kΩ"), SplitResult('', '1234.56', 'k', '', 'Ω'))
        self.assertTupleEqual(self.io.split_input("1k234"), SplitResult('', '1.234', 'k', '', ''))
        self.assertTupleEqual(self.io.split_input("1k234Ω"), SplitResult('', '1.234', 'k', '', 'Ω'))
        self.assertTupleEqual(self.io.split_input("1,234.56Ω"), SplitResult('', '1234.56', '', '', 'Ω'))
        self.assertTupleEqual(self.io.split_input("1A"), SplitResult('', '1', '', '', 'A'))
        self.assertTupleEqual(self.io.split_input("0Ω"), SplitResult('', '0', '', '', 'Ω'))
        self.assertTupleEqual(self.io.split_input("±5%"), SplitResult('±', '5', '', '', '%'))
        self.assertTupleEqual(self.io.split_input("± 5%"), SplitResult('±', '5', '', '', '%'))
        self.assertTupleEqual(self.io.split_input("± 5 %"), SplitResult('±', '5', '', '', '%'))
        self.assertTupleEqual(self.io.split_input("0 Ω"), SplitResult('', '0', '', '', 'Ω'))
        self.assertTupleEqual(self.io.split_input("1"), SplitResult('', '1', '', '', ''))
        self.assertTupleEqual(self.io.split_input("1k234 Ω"), SplitResult('', '1.234', 'k', '', 'Ω'))
        self.assertTupleEqual(self.io.split_input("-1,234.56kΩ"), SplitResult('', '-1234.56', 'k', '', 'Ω'))
        self.assertTupleEqual(self.io.split_input("-1e3kΩ"), SplitResult('', '-1e3', 'k', '', 'Ω'))
        self.assertTupleEqual(self.io.split_input("1e-3kΩ"), SplitResult('', '1e-3', 'k', '', 'Ω'))
        self.assertTupleEqual(self.io.split_input("-4e6nA"), SplitResult('', '-4e6', 'n', '', 'A'))
        self.assertTupleEqual(self.io.split_input("3.2 MHz"), SplitResult('', '3.2', 'M', '', 'Hz'))
        self.assertTupleEqual(self.io.split_input("3.2 °C"), SplitResult('', '3.2', '', '°', 'C'))
        self.assertTupleEqual(self.io.split_input("3k2 °C"), SplitResult('', '3.2', 'k', '°', 'C'))
        self.assertTupleEqual(self.io.split_input("100 mV"), SplitResult('', '100', 'm', '', 'V'))
        self.assertTupleEqual(self.io.split_input("Δ3.2 MHz"), SplitResult('Δ', '3.2', 'M', '', 'Hz'))
        self.assertTupleEqual(self.io.split_input("3.20 €"), SplitResult('', '3.20', '', '', '€'))
        self.assertTupleEqual(self.io.split_input("0.000014 €"), SplitResult('', '0.000014', '', '', '€'))

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
        with self.assertRaises(ValueError):
            print(self.io.normalize_numeric(s))

    def test_split_unit(self):
        self.assertTupleEqual(self.io.split_unit("1234"), UnitSplitResult('1234', '', ''))
        self.assertTupleEqual(self.io.split_unit("1234k"), UnitSplitResult('1234k', '', ''))
        self.assertTupleEqual(self.io.split_unit("1234kΩ"), UnitSplitResult('1234k', '', 'Ω'))
        self.assertTupleEqual(self.io.split_unit("1.234kΩ"), UnitSplitResult('1.234k', '', 'Ω'))
        self.assertTupleEqual(self.io.split_unit("1,234kΩ"), UnitSplitResult('1,234k', '', 'Ω'))
        self.assertTupleEqual(self.io.split_unit("1,234.56kΩ"), UnitSplitResult('1,234.56k', '', 'Ω'))
        self.assertTupleEqual(self.io.split_unit("1k234"), UnitSplitResult('1k234', '', ''))
        self.assertTupleEqual(self.io.split_unit("1k234Ω"), UnitSplitResult('1k234', '',  'Ω'))
        self.assertTupleEqual(self.io.split_unit("1,234.56Ω"), UnitSplitResult('1,234.56', '', 'Ω'))
        self.assertTupleEqual(self.io.split_unit("1A"), UnitSplitResult('1', '', 'A'))
        self.assertTupleEqual(self.io.split_unit("1"), UnitSplitResult('1', '', ''))
        self.assertTupleEqual(self.io.split_unit("1k234 Ω"), UnitSplitResult('1k234', '', 'Ω'))
        self.assertTupleEqual(self.io.split_unit("-1,234.56kΩ"), UnitSplitResult('-1,234.56k', '', 'Ω'))
        self.assertTupleEqual(self.io.split_unit("-1e3kΩ"), UnitSplitResult('-1e3k', '', 'Ω'))
        self.assertTupleEqual(self.io.split_unit("1e-3kΩ"), UnitSplitResult('1e-3k', '', 'Ω'))
        self.assertTupleEqual(self.io.split_unit("-4e6nA"), UnitSplitResult('-4e6n', '', 'A'))
        self.assertTupleEqual(self.io.split_unit("3.2 MHz"), UnitSplitResult('3.2 M', '', 'Hz'))
        self.assertTupleEqual(self.io.split_unit("3.2 °C"), UnitSplitResult('3.2', '°', 'C'))
        self.assertTupleEqual(self.io.split_unit("3k2 °C"), UnitSplitResult('3k2', '°', 'C'))
        self.assertTupleEqual(self.io.split_unit("100 mV"), UnitSplitResult('100 m', '', 'V'))
        self.assertTupleEqual(self.io.split_unit("50 °C/W"), UnitSplitResult('50', '°', 'C/W'))
        self.assertTupleEqual(self.io.split_unit(""), UnitSplitResult('', '', ''))


    def test_normalize(self):
        self.assertTupleEqual(self.io.normalize("100 kΩ"), NormalizeResult('', 1e5, '', "Ω"))
        self.assertTupleEqual(self.io.normalize("100 kΩ".encode("utf8")), NormalizeResult('', 1e5, '', "Ω"))

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

    def testIsValidSuffix(self):
        for c in "fpnuµmkMGT":
            self.assertIn(c, self.io.suffix_exp_map)

    def test_exp_suffix_map(self):
        self.assertEqual(self.io.suffix_exp_map["f"], -15)
        self.assertEqual(self.io.suffix_exp_map["k"], 3)
        self.assertEqual(self.io.suffix_exp_map["u"], -6)
        self.assertEqual(self.io.suffix_exp_map["µ"], -6)
        self.assertEqual(self.io.suffix_exp_map["T"], 12)
        self.assertEqual(self.io.suffix_exp_map[""], 0)
        # Check "in" operator
        self.assertIn("k", self.io.suffix_exp_map)
        # Invalid suffix_exp_map
        self.assertNotIn("B", self.io.suffix_exp_map)

    def test_exp_suffix_map2(self):
        self.assertEqual("", self.io.exp_suffix_map[0])
        self.assertEqual("k", self.io.exp_suffix_map[1])
        self.assertEqual("M", self.io.exp_suffix_map[2])
        self.assertEqual("n", self.io.exp_suffix_map[-3])

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
        self.assertTupleEqual(self.io.safe_normalize("1.25 kV"), NormalizeResult('', 1250., '', 'V'))
        self.assertIsNone(self.io.safe_normalize("1x25"))

    # Just basic tests for autoFormat. Specific tests in other modules that have annotated functions

    def testAutoFormatValid(self):
        def testfn(n=1.0) -> Unit("V"): return n
        self.assertEqual(self.io.auto_format(testfn), "1.00 V")
        # Test functools.partial() behaviour
        testfn2 = functools.partial(testfn, n=2.0)
        self.assertEqual(self.io.auto_format(testfn2), "2.00 V")
        # Test nested functools.partial() behaviour
        testfn3 = functools.partial(testfn2, n=3.0)
        self.assertEqual(self.io.auto_format(testfn3), "3.00 V")

    def testAutoFormatInvalid1(self):
        with self.assertRaises(UnannotatedReturnValueError):
            self.io.auto_format(self.io.format) # Callable but not annotated

    def testAutoFormatInvalid2(self):
        with self.assertRaises(ValueError):
            self.io.auto_format(None)

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


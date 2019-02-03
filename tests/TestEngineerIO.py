#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_allclose, assert_approx_equal
from nose.tools import assert_equal, assert_tuple_equal, assert_is_none, assert_true, assert_false, raises, assert_in, assert_not_in
from UliEngineering.EngineerIO import *
from UliEngineering.Units import *
from UliEngineering.EngineerIO import _formatWithSuffix
from nose_parameterized import parameterized
import functools
import numpy as np

class TestEngineerIO(object):
    def __init__(self):
        self.io = EngineerIO()

    def test_normalize_interpunctation(self):
        for suffix in ["", "k", " kV", "V/√Hz", "µV"]:
            assert_equal(normalize_interpunctation("1234" + suffix), '1234' + suffix)
            assert_equal(normalize_interpunctation("123.4" + suffix), '123.4' + suffix)
            assert_equal(normalize_interpunctation("123,4" + suffix), '123.4' + suffix)
            assert_equal(normalize_interpunctation("1,234.5" + suffix), '1234.5' + suffix)
            assert_equal(normalize_interpunctation("1.234,5" + suffix), '1234.5' + suffix)
            assert_equal(normalize_interpunctation("1.234,5" + suffix), '1234.5' + suffix)
        assert_equal(normalize_interpunctation(""), "")

    def test_split_input(self):
        assert_tuple_equal(self.io.split_input("1234"), ('1234', '', ''))
        assert_tuple_equal(self.io.split_input("1234k"), ('1234', 'k', ''))
        assert_tuple_equal(self.io.split_input("1234kΩ"), ('1234', 'k', 'Ω'))
        assert_tuple_equal(self.io.split_input("1.234kΩ"), ('1.234', 'k', 'Ω'))
        assert_tuple_equal(self.io.split_input("1,234kΩ"), ('1.234', 'k', 'Ω'))
        assert_tuple_equal(self.io.split_input("1,234.56kΩ"), ('1234.56', 'k', 'Ω'))
        assert_tuple_equal(self.io.split_input("1k234"), ('1.234', 'k', ''))
        assert_tuple_equal(self.io.split_input("1k234Ω"), ('1.234', 'k', 'Ω'))
        assert_tuple_equal(self.io.split_input("1,234.56Ω"), ('1234.56', '', 'Ω'))
        assert_tuple_equal(self.io.split_input("1A"), ('1', '', 'A'))
        assert_tuple_equal(self.io.split_input("1"), ('1', '', ''))
        assert_tuple_equal(self.io.split_input("1k234 Ω"), ('1.234', 'k', 'Ω'))
        assert_tuple_equal(self.io.split_input("-1,234.56kΩ"), ('-1234.56', 'k', 'Ω'))
        assert_tuple_equal(self.io.split_input("-1e3kΩ"), ('-1e3', 'k', 'Ω'))
        assert_tuple_equal(self.io.split_input("1e-3kΩ"), ('1e-3', 'k', 'Ω'))
        assert_tuple_equal(self.io.split_input("-4e6nA"), ('-4e6', 'n', 'A'))
        assert_tuple_equal(self.io.split_input("3.2 MHz"), ('3.2', 'M', 'Hz'))
        assert_tuple_equal(self.io.split_input("3.2 °C"), ('3.2', '', 'C'))
        assert_tuple_equal(self.io.split_input("3k2 °C"), ('3.2', 'k', 'C'))
        assert_tuple_equal(self.io.split_input("3.2 ΔMHz"), ('3.2', 'M', 'Hz'))
        assert_tuple_equal(self.io.split_input("100 mV"), ('100', 'm', 'V'))
        assert_tuple_equal(self.io.split_input("3.2 ΔHz"), ('3.2', '', 'Hz'))
        assert_tuple_equal(self.io.split_input("Δ3.2 MHz"), ('3.2', 'M', 'Hz'))

    @parameterized([("1,234.56kfA",),
                    ("1.23k45A",),
                    ("1,234.56kfA",),
                    ("foobar",),
                    (None,),
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
    @raises(ValueError)
    def test_normalize_numeric_invalid(self, s):
        print(self.io.normalize_numeric(s))

    def test_split_unit(self):
        assert_tuple_equal(self.io.split_unit("1234"), ('1234', ''))
        assert_tuple_equal(self.io.split_unit("1234k"), ('1234k', ''))
        assert_tuple_equal(self.io.split_unit("1234kΩ"), ('1234k', 'Ω'))
        assert_tuple_equal(self.io.split_unit("1.234kΩ"), ('1.234k', 'Ω'))
        assert_tuple_equal(self.io.split_unit("1,234kΩ"), ('1,234k', 'Ω'))
        assert_tuple_equal(self.io.split_unit("1,234.56kΩ"), ('1,234.56k', 'Ω'))
        assert_tuple_equal(self.io.split_unit("1k234"), ('1k234', ''))
        assert_tuple_equal(self.io.split_unit("1k234Ω"), ('1k234',  'Ω'))
        assert_tuple_equal(self.io.split_unit("1,234.56Ω"), ('1,234.56', 'Ω'))
        assert_tuple_equal(self.io.split_unit("1A"), ('1', 'A'))
        assert_tuple_equal(self.io.split_unit("1"), ('1', ''))
        assert_tuple_equal(self.io.split_unit("1k234 Ω"), ('1k234', 'Ω'))
        assert_tuple_equal(self.io.split_unit("-1,234.56kΩ"), ('-1,234.56k', 'Ω'))
        assert_tuple_equal(self.io.split_unit("-1e3kΩ"), ('-1e3k', 'Ω'))
        assert_tuple_equal(self.io.split_unit("1e-3kΩ"), ('1e-3k', 'Ω'))
        assert_tuple_equal(self.io.split_unit("-4e6nA"), ('-4e6n', 'A'))
        assert_tuple_equal(self.io.split_unit("3.2 MHz"), ('3.2 M', 'Hz'))
        assert_tuple_equal(self.io.split_unit("3.2 °C"), ('3.2', 'C'))
        assert_tuple_equal(self.io.split_unit("3k2 °C"), ('3k2', 'C'))
        assert_tuple_equal(self.io.split_unit("3.2 ΔMHz"), ('3.2 ΔM', 'Hz'))
        assert_tuple_equal(self.io.split_unit("100 mV"), ('100 m', 'V'))
        assert_tuple_equal(self.io.split_unit("3.2 ΔHz"), ('3.2', 'Hz'))
        assert_tuple_equal(self.io.split_unit(""), ('', ''))


    def test_normalize(self):
        assert_tuple_equal(self.io.normalize("100 kΩ"), (1e5, "Ω"))
        assert_tuple_equal(self.io.normalize("100 kΩ".encode("utf8")), (1e5, "Ω"))

    def test_formatWithSuffix(self):
        assert_equal(_formatWithSuffix(1.01, "A"), '1.01 A')
        assert_equal(_formatWithSuffix(1, "A"), '1.00 A')
        assert_equal(_formatWithSuffix(101, "A"), '101 A')
        assert_equal(_formatWithSuffix(99.9, "A"), '99.9 A')
        assert_equal(_formatWithSuffix(1000.0, ""), '1000')

    def test_format(self):
        assert_equal(self.io.format(1.0e-15, "V"), '1.00 fV')
        assert_equal(self.io.format(234.6789e-3, "V"), '235 mV')
        assert_equal(self.io.format(234.6789, "V"), '235 V')
        assert_equal(self.io.format(2345.6789, "V"), '2.35 kV')
        assert_equal(self.io.format(2345.6789e6, "V"), '2.35 GV')
        assert_equal(self.io.format(2345.6789e12, "V"), '2.35 EV')
        assert_equal(self.io.format(2.3456789e-6, "V"), '2.35 µV')
        assert_equal(self.io.format(2.3456789e-6, "°C"), '2.35 µ°C')
        assert_equal(self.io.format(-2.3456789e-6, "°C"), '-2.35 µ°C')

    @raises(ValueError)
    def test_format_invalid(self):
        self.io.format(1.0e-25, "V")

    def testRounding(self):
        assert_equal(self.io.format(1.999999, ""), '2.00')
        assert_equal(self.io.format(19.99999, ""), '20.0')
        assert_equal(self.io.format(199.9999, ""), '200')

    def testIsValidSuffix(self):
        for c in "fpnuµmkMGT":
            assert_in(c, self.io.suffix_exp_map)

    def test_exp_suffix_map(self):
        assert_equal(self.io.suffix_exp_map["f"], -15)
        assert_equal(self.io.suffix_exp_map["k"], 3)
        assert_equal(self.io.suffix_exp_map["u"], -6)
        assert_equal(self.io.suffix_exp_map["µ"], -6)
        assert_equal(self.io.suffix_exp_map["T"], 12)
        assert_equal(self.io.suffix_exp_map[""], 0)
        # Check "in" operator
        assert_in("k", self.io.suffix_exp_map)
        # Invalid suffix_exp_map
        assert_not_in("B", self.io.suffix_exp_map)

    def test_exp_suffix_map(self):
        assert_equal("", self.io.exp_suffix_map[0])
        assert_equal("k", self.io.exp_suffix_map[1])
        assert_equal("M", self.io.exp_suffix_map[2])
        assert_equal("n", self.io.exp_suffix_map[-3])

    def test_normalize_numeric_safe(self):
        assert_equal(self.io.normalize_numeric_safe(1.25), 1.25)
        assert_equal(self.io.normalize_numeric_safe("1.25"), 1.25)
        assert_equal(self.io.normalize_numeric_safe("1.25 V"), 1.25)
        assert_equal(self.io.normalize_numeric_safe("1k25 V"), 1250.0)
        assert_equal(self.io.normalize_numeric_safe(b"1k25 V"), 1250.0)
        assert_allclose(self.io.normalize_numeric_safe(["1k25 V", "4.25 A"]), [1250.0, 4.25])
        # Invalid inputs and partially invalid inputs
        assert_is_none(self.io.normalize_numeric_safe("foobar"))
        assert_allclose(self.io.normalize_numeric_safe(["foobar", "1.2 J"]), [np.nan, 1.2])

    def test_normalize_numeric(self):
        assert_equal(self.io.normalize_numeric(1.25), 1.25)
        assert_equal(self.io.normalize_numeric("1.25"), 1.25)
        assert_equal(self.io.normalize_numeric("1.25 V"), 1.25)
        assert_equal(self.io.normalize_numeric("1k25 V"), 1250.0)
        assert_equal(self.io.normalize_numeric(b"1k25 V"), 1250.0)
        assert_allclose(self.io.normalize_numeric(["1k25 V", "4.25 A"]), np.asarray([1250.0, 4.25]))

    @raises(ValueError)
    def test_normalize_numeric_invalid(self):
        self.io.normalize_numeric(["1.2 J", "foobar"])

    def test_safe_normalize(self):
        assert_tuple_equal(self.io.safe_normalize("1.25 kV"), (1250., "V"))
        assert_is_none(self.io.safe_normalize("1x25"))

    # Just basic tests for autoFormat. Specific tests in other modules that have annotated functions

    def testAutoFormatValid(self):
        def testfn(n=1.0) -> Unit("V"): return n
        assert_equal(self.io.auto_format(testfn), "1.00 V")
        # Test functools.partial() behaviour
        testfn2 = functools.partial(testfn, n=2.0)
        assert_equal(self.io.auto_format(testfn2), "2.00 V")
        # Test nested functools.partial() behaviour
        testfn3 = functools.partial(testfn2, n=3.0)
        assert_equal(self.io.auto_format(testfn3), "3.00 V")

    @raises(UnannotatedReturnValueError)
    def testAutoFormatInvalid1(self):
        self.io.auto_format(self.io.format) # Callable but not annotated

    @raises(ValueError)
    def testAutoFormatInvalid2(self):
        self.io.auto_format(None)

    def test_auto_suffix_1d(self):
        arr = np.arange(-4., 5., .5)
        assert_equal(self.io.auto_suffix_1d(arr), (1., ""))
        arr = 1e-3 * np.arange(-4., 5., .5)
        assert_equal(self.io.auto_suffix_1d(arr), (1e3, "m"))
        arr = 1e9 * np.arange(-4., 5., .5)
        assert_equal(self.io.auto_suffix_1d(arr), (1e-9, "G"))
        arr = np.arange(1000., 2000., 5)
        assert_equal(self.io.auto_suffix_1d(arr), (1e-3, "k"))
        # Test out of limits
        arr = 1e-40 * np.arange(-4., 5., .5)
        assert_equal(self.io.auto_suffix_1d(arr), (1e24, "y"))
        arr = 1e40 * np.arange(-4., 5., .5)
        assert_equal(self.io.auto_suffix_1d(arr), (1e-21, "Y"))

    def test_special_units(self):
        """
        Test ppm, ppb and %
        """
        # %
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
        Test ppm, ppb and %
        """
        assert_approx_equal(self.io.normalize_numeric_safe("1.25 s"), 1.25)
        assert_approx_equal(self.io.normalize_numeric_safe("1.25 min"), 1.25)
        assert_approx_equal(self.io.normalize_numeric_safe("1.25 h"), 1.25)


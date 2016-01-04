#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from nose.tools import assert_equal, assert_tuple_equal, assert_is_none, assert_true, assert_false, raises
from UliEngineering.EngineerIO import *
from UliEngineering.EngineerIO import _formatWithSuffix

class TestEngineerIO(object):
    def testNormalizeCommaToPoint(self):
        for suffix in ["", "k", " kV", "V/√Hz", "µV"]:
            assert_equal(normalizeCommaToPoint("1234" + suffix), '1234' + suffix)
            assert_equal(normalizeCommaToPoint("123.4" + suffix), '123.4' + suffix)
            assert_equal(normalizeCommaToPoint("123,4" + suffix), '123.4' + suffix)
            assert_equal(normalizeCommaToPoint("1,234.5" + suffix), '1234.5' + suffix)
            assert_equal(normalizeCommaToPoint("1.234,5" + suffix), '1234.5' + suffix)
            assert_equal(normalizeCommaToPoint("1.234,5" + suffix), '1234.5' + suffix)
    def testSplitSuffixSeparator(self):
        assert_tuple_equal(splitSuffixSeparator("1234"), ('1234', '', ''))
        assert_tuple_equal(splitSuffixSeparator("1234k"), ('1234', 'k', ''))
        assert_tuple_equal(splitSuffixSeparator("1234kΩ"), ('1234', 'k', 'Ω'))
        assert_tuple_equal(splitSuffixSeparator("1.234kΩ"), ('1.234', 'k', 'Ω'))
        assert_tuple_equal(splitSuffixSeparator("1,234kΩ"), ('1.234', 'k', 'Ω'))
        assert_tuple_equal(splitSuffixSeparator("1,234.56kΩ"), ('1234.56', 'k', 'Ω'))
        assert_tuple_equal(splitSuffixSeparator("1k234"), ('1.234', 'k', ''))
        assert_tuple_equal(splitSuffixSeparator("1k234Ω"), ('1.234', 'k', 'Ω'))
        assert_tuple_equal(splitSuffixSeparator("1,234.56Ω"), ('1234.56', '', 'Ω'))
        assert_tuple_equal(splitSuffixSeparator("1A"), ('1', '', 'A'))
        assert_tuple_equal(splitSuffixSeparator("1"), ('1', '', ''))
        assert_tuple_equal(splitSuffixSeparator("1k234 Ω"), ('1.234', 'k', 'Ω'))
        assert_tuple_equal(splitSuffixSeparator("-1,234.56kΩ"), ('-1234.56', 'k', 'Ω'))
        assert_tuple_equal(splitSuffixSeparator("-1e3kΩ"), ('-1e3', 'k', 'Ω'))
        assert_tuple_equal(splitSuffixSeparator("1e-3kΩ"), ('1e-3', 'k', 'Ω'))
        assert_tuple_equal(splitSuffixSeparator("-4e6nA"), ('-4e6', 'n', 'A'))
        assert_tuple_equal(splitSuffixSeparator("3.2 MHz"), ('3.2', 'M', 'Hz'))
        assert_tuple_equal(splitSuffixSeparator("3.2 °C"), ('3.2', '', 'C'))
        assert_tuple_equal(splitSuffixSeparator("3k2 °C"), ('3.2', 'k', 'C'))
        assert_tuple_equal(splitSuffixSeparator("3.2 ΔMHz"), ('3.2', 'M', 'Hz'))
        assert_tuple_equal(splitSuffixSeparator("100 mV"), ('100', 'm', 'V'))
        assert_tuple_equal(splitSuffixSeparator("3.2 ΔHz"), ('3.2', '', 'Hz'))

        assert_is_none(splitSuffixSeparator("Δ3.2 MHz"))
        assert_is_none(splitSuffixSeparator("1,234.56kfA"))
        assert_is_none(splitSuffixSeparator("1.23k45A"))
        assert_is_none(splitSuffixSeparator("1,234.56kfA"))
        assert_is_none(splitSuffixSeparator("foobar"))
        assert_is_none(splitSuffixSeparator(None))
        assert_is_none(splitSuffixSeparator("1k2 MA"))
        assert_is_none(splitSuffixSeparator("1.2kkA"))
        assert_is_none(splitSuffixSeparator("1k2kA"))
        assert_is_none(splitSuffixSeparator("A"))
        assert_is_none(splitSuffixSeparator("AA"))
        assert_is_none(splitSuffixSeparator(" "))

    def testNormalizeEngineerInput(self):
        assert_is_none(normalizeEngineerInput("3.2°G"))
        assert_tuple_equal(normalizeEngineerInput("100 kΩ"), (1e5, "Ω"))

    def test_formatWithSuffix(self):
        assert_equal(_formatWithSuffix(1.01, "A"), '1.01 A')
        assert_equal(_formatWithSuffix(1, "A"), '1.00 A')
        assert_equal(_formatWithSuffix(101, "A"), '101 A')
        assert_equal(_formatWithSuffix(99.9, "A"), '99.9 A')
        assert_equal(_formatWithSuffix(1000.0, ""), '1000')

    def testFormatValue(self):
        assert_equal(formatValue(1.0e-15, "V"), '1.00 fV')
        assert_equal(formatValue(1.0e-25, "V"), None)
        assert_equal(formatValue(234.6789e-3, "V"), '235 mV')
        assert_equal(formatValue(234.6789, "V"), '235 V')
        assert_equal(formatValue(2345.6789, "V"), '2.35 kV')
        assert_equal(formatValue(2345.6789e6, "V"), '2.35 GV')
        assert_equal(formatValue(2345.6789e12, "V"), '2.35 EV')
        assert_equal(formatValue(2.3456789e-6, "V"), '2.35 µV')
        assert_equal(formatValue(2.3456789e-6, "°C"), '2.35 µ°C')
        assert_equal(formatValue(-2.3456789e-6, "°C"), '-2.35 µ°C')

    def testRounding(self):
        assert_equal(formatValue(1.999999, ""), '2.00')
        assert_equal(formatValue(19.99999, ""), '20.0')
        assert_equal(formatValue(199.9999, ""), '200')

    def testIsValidSuffix(self):
        for c in "fpnuµmkMGT":
            assert_true(isValidSuffix(c))
        assert_true(isValidSuffix(""))
        assert_true(isValidSuffix(None))

    def testGetSuffixMultiplier(self):
        assert_equal(getSuffixMultiplier("f"), -15)
        assert_equal(getSuffixMultiplier("k"), 3)
        assert_equal(getSuffixMultiplier("u"), -6)
        assert_equal(getSuffixMultiplier("µ"), -6)
        assert_equal(getSuffixMultiplier("T"), 12)
        assert_equal(getSuffixMultiplier(""), 0)

    def testAutoNormalizeEngineerInput(self):
        assert_equal(autoNormalizeEngineerInput(1.25), (1.25, ''))
        assert_equal(autoNormalizeEngineerInput("1.25"), (1.25, ''))
        assert_equal(autoNormalizeEngineerInput("1.25 V"), (1.25, "V"))
        assert_equal(autoNormalizeEngineerInput("1k25 V"), (1250.0, "V"))
        assert_equal(autoNormalizeEngineerInput(b"1.25 V"), (1.25, "V"))

    def testAutoNormalizeEngineerInputIgnoreUnit(self):
        assert_equal(autoNormalizeEngineerInputNoUnit(1.25), 1.25)
        assert_equal(autoNormalizeEngineerInputNoUnit("1.25"), 1.25)
        assert_equal(autoNormalizeEngineerInputNoUnit("1.25 V"), 1.25)
        assert_equal(autoNormalizeEngineerInputNoUnit("1k25 V"), 1250.0)
        assert_equal(autoNormalizeEngineerInputNoUnit(b"1.25 V"), 1.25)

    # Just basic tests for autoFormat. Specific tests in other modules that have annotated functions

    def testAutoFormatValid(self):
        def testfn(n=1.0) -> Quantity("V"): return n
        assert_equal(autoFormat(testfn), "1.00 V")
        # Test functools.partial() behaviour
        testfn2 = functools.partial(testfn, n=2.0)
        assert_equal(autoFormat(testfn2), "2.00 V")
        # Test nested functools.partial() behaviour
        testfn3 = functools.partial(testfn2, n=3.0)
        assert_equal(autoFormat(testfn3), "3.00 V")

    @raises(UnannotatedReturnValueError)
    def testAutoFormatInvalid1(self):
        autoFormat(autoFormat)

    @raises(ValueError)
    def testAutoFormatInvalid2(self):
        autoFormat(None)

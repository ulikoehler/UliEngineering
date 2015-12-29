#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from nose.tools import assert_equal, assert_tuple_equal, assert_is_none
from UliEngineering.EngineerIO import *

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
        assert_is_none(splitSuffixSeparator(""))

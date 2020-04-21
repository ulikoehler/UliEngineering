#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nose.tools import assert_equal, assert_true, assert_false, raises
from UliEngineering.Utils.String import *
from parameterized import parameterized
import unittest

class TestSplitNth(unittest.TestCase):
    def testSimple(self):
        assert_equal("a", split_nth("a,b,c,d,e,f"))
        assert_equal("a", split_nth("a,b,c,d,e,f", nth=1))
        assert_equal("b", split_nth("a,b,c,d,e,f", nth=2))
        assert_equal("c", split_nth("a,b,c,d,e,f", nth=3))
        assert_equal("d", split_nth("a,b,c,d,e,f", nth=4))
        assert_equal("e", split_nth("a,b,c,d,e,f", nth=5))
        assert_equal("f", split_nth("a,b,c,d,e,f", nth=6))

    def testMultiChar(self):
        assert_equal("ab", split_nth("ab,cd,ef,gh,ij,kl"))
        assert_equal("ab", split_nth("ab,cd,ef,gh,ij,kl", nth=1))
        assert_equal("cd", split_nth("ab,cd,ef,gh,ij,kl", nth=2))
        assert_equal("ef", split_nth("ab,cd,ef,gh,ij,kl", nth=3))
        assert_equal("gh", split_nth("ab,cd,ef,gh,ij,kl", nth=4))
        assert_equal("ij", split_nth("ab,cd,ef,gh,ij,kl", nth=5))
        assert_equal("kl", split_nth("ab,cd,ef,gh,ij,kl", nth=6))

    def testOtherDelimiter(self):
        assert_equal("a", split_nth("a;b;c;d;e;f", delimiter=';'))
        assert_equal("a", split_nth("a;b;c;d;e;f", delimiter=';', nth=1))
        assert_equal("b", split_nth("a;b;c;d;e;f", delimiter=';', nth=2))
        assert_equal("c", split_nth("a;b;c;d;e;f", delimiter=';', nth=3))
        assert_equal("d", split_nth("a;b;c;d;e;f", delimiter=';', nth=4))
        assert_equal("e", split_nth("a;b;c;d;e;f", delimiter=';', nth=5))
        assert_equal("f", split_nth("a;b;c;d;e;f", delimiter=';', nth=6))
        assert_equal("", split_nth("aa;bb;;dd;ee;ff", delimiter=';', nth=3))
        assert_equal("", split_nth("aa;bb;;", delimiter=';', nth=3))
        assert_equal("", split_nth("aa;bb;;", delimiter=';', nth=4))

    def testEmpty(self):
        assert_equal("", split_nth("aa,bb,,dd,ee,ff", nth=3))
        assert_equal("", split_nth("aa,bb,,", nth=3))
        assert_equal("", split_nth("aa,bb,,", nth=4))
        assert_equal("", split_nth("", nth=1))

    @parameterized.expand(["a", "abc", "abcdef", "abc,def"])
    @raises(ValueError)
    def testInvalidFirst(self, param):
        split_nth(param, nth=3)

    @raises(ValueError)
    def testInvalidOther(self):
        split_nth("abc,def", nth=3)


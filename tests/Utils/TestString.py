#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nose.tools import self.assertEqual, assert_true, assert_false, raises
from UliEngineering.Utils.String import *
from parameterized import parameterized
import unittest

class TestSplitNth(unittest.TestCase):
    def testSimple(self):
        self.assertEqual("a", split_nth("a,b,c,d,e,f"))
        self.assertEqual("a", split_nth("a,b,c,d,e,f", nth=1))
        self.assertEqual("b", split_nth("a,b,c,d,e,f", nth=2))
        self.assertEqual("c", split_nth("a,b,c,d,e,f", nth=3))
        self.assertEqual("d", split_nth("a,b,c,d,e,f", nth=4))
        self.assertEqual("e", split_nth("a,b,c,d,e,f", nth=5))
        self.assertEqual("f", split_nth("a,b,c,d,e,f", nth=6))

    def testMultiChar(self):
        self.assertEqual("ab", split_nth("ab,cd,ef,gh,ij,kl"))
        self.assertEqual("ab", split_nth("ab,cd,ef,gh,ij,kl", nth=1))
        self.assertEqual("cd", split_nth("ab,cd,ef,gh,ij,kl", nth=2))
        self.assertEqual("ef", split_nth("ab,cd,ef,gh,ij,kl", nth=3))
        self.assertEqual("gh", split_nth("ab,cd,ef,gh,ij,kl", nth=4))
        self.assertEqual("ij", split_nth("ab,cd,ef,gh,ij,kl", nth=5))
        self.assertEqual("kl", split_nth("ab,cd,ef,gh,ij,kl", nth=6))

    def testOtherDelimiter(self):
        self.assertEqual("a", split_nth("a;b;c;d;e;f", delimiter=';'))
        self.assertEqual("a", split_nth("a;b;c;d;e;f", delimiter=';', nth=1))
        self.assertEqual("b", split_nth("a;b;c;d;e;f", delimiter=';', nth=2))
        self.assertEqual("c", split_nth("a;b;c;d;e;f", delimiter=';', nth=3))
        self.assertEqual("d", split_nth("a;b;c;d;e;f", delimiter=';', nth=4))
        self.assertEqual("e", split_nth("a;b;c;d;e;f", delimiter=';', nth=5))
        self.assertEqual("f", split_nth("a;b;c;d;e;f", delimiter=';', nth=6))
        self.assertEqual("", split_nth("aa;bb;;dd;ee;ff", delimiter=';', nth=3))
        self.assertEqual("", split_nth("aa;bb;;", delimiter=';', nth=3))
        self.assertEqual("", split_nth("aa;bb;;", delimiter=';', nth=4))

    def testEmpty(self):
        self.assertEqual("", split_nth("aa,bb,,dd,ee,ff", nth=3))
        self.assertEqual("", split_nth("aa,bb,,", nth=3))
        self.assertEqual("", split_nth("aa,bb,,", nth=4))
        self.assertEqual("", split_nth("", nth=1))

    @parameterized.expand(["a", "abc", "abcdef", "abc,def"])
    def testInvalidFirst(self, param):
        with self.assertRaises(ValueError):
            split_nth(param, nth=3)

    def testInvalidOther(self):
        with self.assertRaises(ValueError):
            split_nth("abc,def", nth=3)

